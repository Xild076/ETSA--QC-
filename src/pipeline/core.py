from __future__ import annotations
import re
from typing import Any, Dict, List

from ..utils import logger
from . import config


def process(text: str, pipeline_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    The core processing function for the pipeline, which segments the input text
    into clauses and performs initial filtering based on the configuration.
    """
    # Clause segmentation
    if pipeline_context.get('clauses'):
        clauses = pipeline_context['clauses']
    else:
        clauses = clause_segment(text)
        # Aggressively split clauses based on conjunctions
        clauses = aggressive_clause_split(clauses)
        
    pipeline_context['clauses'] = clauses
    logger.info(f"Segmented into {len(clauses)} clauses: {clauses}")

    # ...existing code...
    return pipeline_context


def aggressive_clause_split(clauses: list[str]) -> list[str]:
    """
    Further splits clauses based on a list of conjunctions to handle
    contrastive and complex sentences more effectively. This implementation
    uses a simpler, more robust partitioning logic.
    """
    conjunctions = [
        "but", "however", "while", "although", "whereas", "though", 
        "on the other hand", "in contrast", "conversely", "yet"
    ]
    
    final_clauses = []
    for clause in clauses:
        temp_clauses = [clause]
        for conj in conjunctions:
            new_temp_clauses = []
            for temp_clause in temp_clauses:
                # Use partition to split the clause by the conjunction
                before, separator, after = temp_clause.partition(f' {conj} ')
                if separator:
                    # If the conjunction was found, add the part before it
                    if before.strip():
                        new_temp_clauses.append(before.strip())
                    # And add the conjunction combined with the part after it
                    if after.strip():
                        new_temp_clauses.append(f'{separator.strip()} {after.strip()}')
                else:
                    # If no split, just keep the clause
                    new_temp_clauses.append(temp_clause)
            temp_clauses = new_temp_clauses
        final_clauses.extend(temp_clauses)

    # Final cleanup to remove any empty strings
    final_clauses = [c.strip(" ,.") for c in final_clauses if c.strip()]
    
    if len(final_clauses) > len(clauses):
        logger.info(f"Aggressively split clauses into: {final_clauses}")
        
    return final_clauses


def clause_segment(text: str) -> list[str]:
    """
    Segments the input text into clauses based on punctuation and conjunctions.
    """
    # First, split by major clause-ending punctuation
    clauses = re.split(r'[.!?]', text)
    
    # Further split by commas and semicolons, which often separate clauses
    clauses = [re.split(r'[,;]', clause) for clause in clauses]
    
    # Flatten the list of lists into a single list of strings
    clauses = [item.strip() for sublist in clauses for item in sublist if item.strip()]
    
    return clauses