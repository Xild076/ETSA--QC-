"""
Coreference resolution following OLD architecture pattern.
This module provides proper entity clustering with ID-based coreference,
as opposed to category-based entity extraction.
"""
import os
import warnings
import logging
import spacy
import re
from typing import Dict, Any, List, Tuple
from functools import lru_cache

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")

from transformers.utils import logging as tlog
tlog.set_verbosity_error()

logger = logging.getLogger(__name__)

def _cpu(d): 
    return "cpu" if d in (None, "-1", "CPU", "cpu") else d

@lru_cache(maxsize=None)
def _mav(device: str = "cpu"):
    """Lazily import and initialize Maverick coref model for the given device."""
    try:
        from maverick import Maverick
    except Exception as exc:
        logger.warning("Maverick not available, using fallback coreference resolution")
        return None
    logger.info(f"Initializing Maverick with device: {device}")
    return Maverick(device=_cpu(device))

@lru_cache(maxsize=None)
def _spacy():
    """Load a spaCy English model, preferring larger ones if available."""
    logger.info("Loading spaCy model...")
    for m in ("en_core_web_trf", "en_core_web_lg", "en_core_web_sm"):
        try:
            return spacy.load(m)
        except Exception:
            continue
    raise RuntimeError("No spaCy English model found. Please install one (e.g., 'en_core_web_sm').")

def _extract_entity_mentions(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    """Extract entity mentions from text using spaCy NER and noun chunks."""
    nlp = _spacy()
    doc = nlp(text)
    mentions = []
    
    # Get named entities and noun chunks
    entities_seen = set()
    
    # Add named entities
    for ent in doc.ents:
        if ent.text.strip() and ent.text not in entities_seen:
            mentions.append((ent.text, (ent.start_char, ent.end_char)))
            entities_seen.add(ent.text)
    
    # Add noun chunks (but avoid duplicates)
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        if chunk_text and chunk_text not in entities_seen:
            # Filter out very short or stop words
            if len(chunk_text) > 2 and chunk.root.pos_ in ('NOUN', 'PROPN'):
                mentions.append((chunk_text, (chunk.start_char, chunk.end_char)))
                entities_seen.add(chunk_text)
    
    # Add pronouns that might be important
    for token in doc:
        if token.pos_ == 'PRON' and token.text.lower() in ['it', 'they', 'them', 'this', 'that']:
            if token.text not in entities_seen:
                mentions.append((token.text, (token.idx, token.idx + len(token.text))))
                entities_seen.add(token.text)
    
    return mentions

def _simple_coreference_clustering(mentions: List[Tuple[str, Tuple[int, int]]]) -> Dict[int, Dict[str, Any]]:
    """Simple coreference clustering based on text similarity and heuristics."""
    clusters = {}
    cluster_id = 0
    
    # Group mentions by similarity
    unassigned_mentions = mentions.copy()
    
    while unassigned_mentions:
        # Start a new cluster with the first unassigned mention
        current_mention = unassigned_mentions.pop(0)
        current_text, current_span = current_mention
        
        cluster_mentions = [current_mention]
        cluster_id += 1
        
        # Find similar mentions to group with this one
        remaining_mentions = []
        for mention_text, mention_span in unassigned_mentions:
            should_cluster = False
            
            # Exact match
            if mention_text.lower() == current_text.lower():
                should_cluster = True
            
            # Pronoun referring to entity
            elif current_text.lower() in ['it', 'they', 'them', 'this', 'that'] and mention_text.lower() not in ['it', 'they', 'them', 'this', 'that']:
                should_cluster = True
            elif mention_text.lower() in ['it', 'they', 'them', 'this', 'that'] and current_text.lower() not in ['it', 'they', 'them', 'this', 'that']:
                should_cluster = True
            
            # Definite reference (e.g., "dessert" and "the dessert")
            elif (current_text.lower().startswith('the ') and mention_text.lower() == current_text.lower()[4:]) or \
                 (mention_text.lower().startswith('the ') and current_text.lower() == mention_text.lower()[4:]):
                should_cluster = True
            
            # Head word matching for compound nouns
            elif len(current_text.split()) > 1 and len(mention_text.split()) > 1:
                current_head = current_text.split()[-1].lower()
                mention_head = mention_text.split()[-1].lower()
                if current_head == mention_head:
                    should_cluster = True
            
            if should_cluster:
                cluster_mentions.append((mention_text, mention_span))
            else:
                remaining_mentions.append((mention_text, mention_span))
        
        unassigned_mentions = remaining_mentions
        
        # Create cluster
        clusters[cluster_id] = {
            "entity_references": cluster_mentions,
            "canonical_name": current_text  # Use first mention as canonical
        }
    
    return clusters

def _split_into_clauses(text: str) -> List[str]:
    """Split text into clauses/sentences."""
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def _map_entities_to_clauses(clusters: Dict[int, Dict[str, Any]], clauses: List[str], text: str) -> Dict[str, Dict[str, List]]:
    """Map entities to clauses based on their positions."""
    sent_map = {}
    
    # Calculate clause boundaries
    clause_boundaries = []
    current_pos = 0
    for clause in clauses:
        start_pos = text.find(clause, current_pos)
        if start_pos != -1:
            end_pos = start_pos + len(clause)
            clause_boundaries.append((start_pos, end_pos))
            current_pos = end_pos
        else:
            # Fallback: just use length-based boundaries
            clause_boundaries.append((current_pos, current_pos + len(clause)))
            current_pos += len(clause)
    
    # Map entities to clauses
    for i, (clause_start, clause_end) in enumerate(clause_boundaries):
        clause_key = f"clause_{i}"
        clause_entities = []
        
        for cluster_id, cluster_data in clusters.items():
            entity_refs = cluster_data.get("entity_references", [])
            for mention_text, (mention_start, mention_end) in entity_refs:
                # Check if mention overlaps with clause
                if mention_start >= clause_start and mention_end <= clause_end:
                    clause_entities.append((mention_text, (mention_start, mention_end)))
        
        sent_map[clause_key] = {"entities": clause_entities}
    
    return sent_map

def resolve_entities_and_clauses(text: str) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Dict[str, List]]]:
    """
    Main function that performs coreference resolution following OLD architecture.
    
    Returns:
        - clusters: Dict mapping cluster_id -> {entity_references: [...], canonical_name: "..."}
        - sent_map: Dict mapping clause_key -> {entities: [...]}
    """
    logger.info("Starting coreference resolution with OLD architecture approach")
    
    try:
        # Step 1: Extract entity mentions
        mentions = _extract_entity_mentions(text)
        logger.debug(f"Extracted {len(mentions)} entity mentions: {mentions}")
        
        # Step 2: Perform coreference clustering
        clusters = _simple_coreference_clustering(mentions)
        logger.debug(f"Created {len(clusters)} coreference clusters")
        
        # Step 3: Split into clauses
        clauses = _split_into_clauses(text)
        logger.debug(f"Split into {len(clauses)} clauses: {clauses}")
        
        # Step 4: Map entities to clauses
        sent_map = _map_entities_to_clauses(clusters, clauses, text)
        logger.debug(f"Mapped entities to {len(sent_map)} clauses")
        
        return clusters, sent_map
        
    except Exception as e:
        logger.error(f"Coreference resolution failed: {e}")
        return {}, {}

def resolve(text: str) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Dict[str, List]]]:
    """
    Compatibility function matching OLD architecture interface.
    """
    return resolve_entities_and_clauses(text)
