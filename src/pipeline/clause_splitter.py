import spacy
import benepar
import nltk
from nltk.tree import ParentedTree
import re
from typing import List

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

class SpacyClauseSplitter:
    def __init__(self, model_name="benepar_en3"):
        self.nlp = spacy.load("en_core_web_sm")
        try:
            if spacy.__version__.startswith('2'):
                self.nlp.add_pipe(benepar.BeneparComponent(model_name))
            else:
                self.nlp.add_pipe("benepar", config={"model": model_name})
            self.has_benepar = True
        except Exception:
            self.has_benepar = False

    def _get_clause_quality_score(self, clause: str) -> float:
        score = 0.0
        if clause and len(clause.strip()) > 0:
            score += 0.3
        if any(char in clause for char in '.,!?'):
            score += 0.2
        if len(clause.split()) >= 3:
            score += 0.3
        if clause.strip()[0].isupper():
            score += 0.2
        return min(score, 1.0)

    def _enhanced_dependency_split(self, text: str) -> List[str]:
        doc = self.nlp(text)
        clauses = []
        current_clause = []
        
        for token in doc:
            current_clause.append(token.text)
            
            if token.text in ['.', '!', '?']:
                if current_clause:
                    clause = ' '.join(current_clause).strip()
                    if clause and self._get_clause_quality_score(clause) > 0.5:
                        clauses.append(clause)
                current_clause = []
            elif token.dep_ == "mark" and token.text.lower() in ["when", "while", "although", "because", "since"]:
                if len(current_clause) > 1:
                    clause = ' '.join(current_clause[:-1]).strip()
                    if clause and self._get_clause_quality_score(clause) > 0.5:
                        clauses.append(clause)
                    current_clause = [token.text]
            elif token.text in [','] and token.head.dep_ in ["ROOT", "ccomp"]:
                if len(current_clause) > 3:
                    clause = ' '.join(current_clause).strip()
                    if self._get_clause_quality_score(clause) > 0.6:
                        clauses.append(clause)
                        current_clause = []
        
        if current_clause:
            clause = ' '.join(current_clause).strip()
            if clause and self._get_clause_quality_score(clause) > 0.5:
                clauses.append(clause)
        
        return clauses

    def _extract_clauses_from_parse_tree(self, tree) -> List[str]:
        """Extract clause boundaries using benepar constituency parsing."""
        clauses = []
        
        def extract_clauses(subtree):
            # Extract S, SBAR, VP clauses
            if subtree.label() in ['S', 'SBAR', 'SINV']:
                clause_text = ' '.join(subtree.leaves())
                if len(clause_text.split()) >= 3:  # Minimum clause length
                    clauses.append(clause_text.strip())
            
            # Extract coordinated clauses (CC patterns)
            elif subtree.label() in ['VP', 'NP'] and len(list(subtree.subtrees())) > 3:
                for child in subtree:
                    if hasattr(child, 'label') and child.label() == 'CC':
                        # Found coordination, split here
                        left_part = []
                        right_part = []
                        cc_found = False
                        
                        for item in subtree:
                            if hasattr(item, 'label') and item.label() == 'CC':
                                cc_found = True
                            elif not cc_found:
                                left_part.extend(item.leaves() if hasattr(item, 'leaves') else [str(item)])
                            else:
                                right_part.extend(item.leaves() if hasattr(item, 'leaves') else [str(item)])
                        
                        if left_part and right_part:
                            left_text = ' '.join(left_part).strip()
                            right_text = ' '.join(right_part).strip()
                            if len(left_text.split()) >= 2:
                                clauses.append(left_text)
                            if len(right_text.split()) >= 2:
                                clauses.append(right_text)
            
            # Recursively process children
            for child in subtree:
                if hasattr(child, 'label'):
                    extract_clauses(child)
        
        extract_clauses(tree)
        return clauses

    def _benepar_enhanced_split(self, text: str) -> List[str]:
        """Enhanced clause splitting using both benepar and spaCy."""
        doc = self.nlp(text)
        all_clauses = []
        
        for sent in doc.sents:
            try:
                # Get constituency parse tree
                parse_tree = list(sent._.parse_string)
                if parse_tree:
                    tree = ParentedTree.fromstring(parse_tree[0])
                    benepar_clauses = self._extract_clauses_from_parse_tree(tree)
                    
                    # If benepar found good clauses, use them
                    if benepar_clauses and len(benepar_clauses) > 1:
                        for clause in benepar_clauses:
                            if self._get_clause_quality_score(clause) > 0.5:
                                all_clauses.append(clause)
                    else:
                        # Fall back to dependency parsing for this sentence
                        dep_clauses = self._enhanced_dependency_split(sent.text)
                        all_clauses.extend(dep_clauses)
                else:
                    # No parse tree available, use dependency parsing
                    dep_clauses = self._enhanced_dependency_split(sent.text)
                    all_clauses.extend(dep_clauses)
                    
            except Exception:
                # Fall back to dependency parsing on error
                dep_clauses = self._enhanced_dependency_split(sent.text)
                all_clauses.extend(dep_clauses)
        
        return all_clauses

    def split(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        
        if self.has_benepar:
            try:
                return self._benepar_enhanced_split(text)
            except Exception:
                pass
        
        return self._enhanced_dependency_split(text)