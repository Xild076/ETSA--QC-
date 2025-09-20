import json
import os
import numpy as np
import logging
from typing import Dict, List, Callable, Any

logger = logging.getLogger(__name__)

class SurveyFormulaLoader:
    def __init__(self, formula_path: str = None):
        if formula_path is None:
            current_dir = os.path.dirname(__file__)
            formula_path = os.path.join(current_dir, 'optimal_formulas', 'all_optimal_parameters.json')
        
        self.formula_path = formula_path
        self.formulas = self._load_formulas()
    
    def _load_formulas(self) -> Dict[str, Any]:
        try:
            with open(self.formula_path, 'r') as f:
                data = json.load(f)
            
            formulas = {}
            for entry in data['entries']:
                category = entry['category']
                if category not in formulas:
                    formulas[category] = entry
            
            return formulas
        except Exception as e:
            logger.error(f"Failed to load survey formulas: {e}")
            return {}
    
    def get_actor_function(self) -> Callable:
        if 'actor' not in self.formulas:
            return lambda inputs: np.mean(inputs) if inputs else 0.0
        
        params = self.formulas['actor']['params_by_key']
        
        def actor_formula(inputs: List[float]) -> float:
            if len(inputs) < 3:
                return np.mean(inputs) if inputs else 0.0
            
            s_init_actor, s_action, s_init_target = inputs[0], inputs[1], inputs[2]
            
            if s_action >= 0 and s_init_target >= 0:
                a, b, c = params['pos_pos_params']
            elif s_action >= 0 and s_init_target < 0:
                a, b, c = params['pos_neg_params']
            elif s_action < 0 and s_init_target >= 0:
                a, b, c = params['neg_pos_params']
            else:
                a, b, c = params['neg_neg_params']
            
            lambda_actor = a
            weighted_interaction = b * s_action * s_init_target
            bias = c
            
            result = lambda_actor * s_init_actor + (1 - lambda_actor) * weighted_interaction + bias
            return np.tanh(result)
        
        return actor_formula
    
    def get_target_function(self) -> Callable:
        if 'target' not in self.formulas:
            return lambda inputs: np.mean(inputs) if inputs else 0.0
        
        params = self.formulas['target']['params_by_key']['params']
        lambda_target, w_action, b_target = params
        
        def target_formula(inputs: List[float]) -> float:
            if len(inputs) < 2:
                return np.mean(inputs) if inputs else 0.0
            
            s_init_target, s_action = inputs[0], inputs[1]
            
            w_signed = w_action if s_action >= 0 else -w_action
            b_signed = b_target if s_action >= 0 else -b_target
            
            result = lambda_target * s_init_target + (1 - lambda_target) * w_signed * s_action + b_signed
            return np.tanh(result)
        
        return target_formula
    
    def get_association_function(self) -> Callable:
        if 'association' not in self.formulas:
            return lambda inputs: np.mean(inputs) if inputs else 0.0
        
        params = self.formulas['association']['params_by_key']['params']
        lambda_entity, w_entity, b_entity = params
        
        def association_formula(inputs: List[float]) -> float:
            if len(inputs) < 2:
                return np.mean(inputs) if inputs else 0.0
            
            s_init_entity, s_other = inputs[0], inputs[1]
            
            w_signed = w_entity if s_other >= 0 else -w_entity
            b_signed = b_entity if s_other >= 0 else -b_entity
            
            result = lambda_entity * s_init_entity + (1 - lambda_entity) * w_signed * s_other + b_signed
            return np.tanh(result)
        
        return association_formula
    
    def get_parent_function(self) -> Callable:
        if 'parent' not in self.formulas:
            return lambda inputs: np.mean(inputs) if inputs else 0.0
        
        params = self.formulas['parent']['params_by_key']['params']
        lambda_parent, w_child, b_parent = params
        
        def parent_formula(inputs: List[float]) -> float:
            if len(inputs) < 2:
                return np.mean(inputs) if inputs else 0.0
            
            s_init_parent, s_child = inputs[0], inputs[1]
            
            result = lambda_parent * s_init_parent + (1 - lambda_parent) * w_child * s_child + b_parent
            return np.tanh(result)
        
        return parent_formula
    
    def get_child_function(self) -> Callable:
        if 'child' not in self.formulas:
            return lambda inputs: np.mean(inputs) if inputs else 0.0
        
        params = self.formulas['child']['params_by_key']
        
        def child_formula(inputs: List[float]) -> float:
            if len(inputs) < 2:
                return np.mean(inputs) if inputs else 0.0
            
            s_init_child, s_parent = inputs[0], inputs[1]
            
            if s_parent >= 0 and s_init_child >= 0:
                a, b, c = params['pos_pos_params']
            elif s_parent >= 0 and s_init_child < 0:
                a, b, c = params['pos_neg_params']
            elif s_parent < 0 and s_init_child >= 0:
                a, b, c = params['neg_pos_params']
            else:
                a, b, c = params['neg_neg_params']
            
            lambda_child = a
            weighted_interaction = b * s_parent
            bias = c
            
            result = lambda_child * s_init_child + (1 - lambda_child) * weighted_interaction + bias
            return np.tanh(result)
        
        return child_formula
    
    def get_aggregate_function(self) -> Callable:
        if 'aggregate' not in self.formulas:
            return lambda scores: np.mean(scores) if scores else 0.0
        
        params = self.formulas['aggregate']['params_by_key']['params']
        max_alpha, m_alpha, min_alpha, max_beta, m_beta, b_beta, max_n, min_weight = params
        
        def aggregate_formula(scores: List[float]) -> float:
            if not scores:
                return 0.0
            if len(scores) == 1:
                return scores[0]
            
            n = len(scores)
            n_clamped = min(n, max_n)
            
            alpha = max(min_alpha, min(max_alpha, m_alpha * n_clamped + min_alpha))
            beta = max(min_alpha, min(max_beta, m_beta * n_clamped + b_beta))
            
            weights = []
            for i in range(1, n + 1):
                weight = (i ** (alpha - 1)) * ((n - i + 1) ** (beta - 1))
                weights.append(weight)
            
            weight_sum = sum(weights)
            if weight_sum == 0:
                return np.mean(scores)
            
            normalized_weights = [max(min_weight, w / weight_sum) for w in weights]
            weight_sum_norm = sum(normalized_weights)
            if weight_sum_norm > 0:
                normalized_weights = [w / weight_sum_norm for w in normalized_weights]
            
            weighted_sum = sum(w * s for w, s in zip(normalized_weights, scores))
            return weighted_sum
        
        return aggregate_formula

def get_actor_function() -> Callable:
    loader = SurveyFormulaLoader()
    return loader.get_actor_function()

def get_target_function() -> Callable:
    loader = SurveyFormulaLoader()
    return loader.get_target_function()

def get_association_function() -> Callable:
    loader = SurveyFormulaLoader()
    return loader.get_association_function()

def get_parent_function() -> Callable:
    loader = SurveyFormulaLoader()
    return loader.get_parent_function()

def get_child_function() -> Callable:
    loader = SurveyFormulaLoader()
    return loader.get_child_function()

def get_aggregate_function() -> Callable:
    loader = SurveyFormulaLoader()
    return loader.get_aggregate_function()
