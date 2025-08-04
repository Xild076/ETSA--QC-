import pickle
import json
import numpy as np
import os


class SentimentFormula:
    def __init__(self, name, model_type, function, params=None):
        self.name = name
        self.model_type = model_type
        self.function = function
        self.params = params
    
    def __call__(self, *args):
        if self.params is None:
            return self.function(*args)
        return self.function(*args, *self.params)
    
    def set_params(self, params):
        self.params = params
    
    def get_info(self):
        return {
            'name': self.name,
            'model_type': self.model_type,
            'params': self.params,
            'function_name': getattr(self.function, '__name__', 'anonymous')
        }
    
    def print(self):
        print(f"SentimentFormula(name={self.name}, model_type={self.model_type}, params={self.params})")
    
    def copy(self):
        return SentimentFormula(
            name=self.name,
            model_type=self.model_type,
            function=self.function,
            params=self.params.copy() if isinstance(self.params, (list, dict)) else self.params
        )
    
    def save(self, file_path):
        try:
            params_to_save = self.params
            if isinstance(self.params, np.ndarray):
                params_to_save = self.params.tolist()
            elif isinstance(self.params, list):
                params_to_save = [p.tolist() if isinstance(p, np.ndarray) else p for p in self.params]
            
            with open(file_path, 'w') as f:
                json.dump({
                    'name': self.name,
                    'model_type': self.model_type,
                    'function': pickle.dumps(self.function).decode('latin1'),
                    'params': params_to_save
                }, f, indent=2)
            print(f"Formula '{self.name}' saved successfully to {file_path}")
        except Exception as e:
            print(f"Error saving formula: {e}")
            raise
    
    def load(self, file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.name = data['name']
                self.model_type = data['model_type']
                self.function = pickle.loads(data['function'].encode('latin1'))
                
                params = data['params']
                if isinstance(params, list) and len(params) > 0 and all(isinstance(p, (int, float)) for p in params):
                    self.params = np.array(params)
                else:
                    self.params = params
                    
            print(f"Formula '{self.name}' loaded successfully from {file_path}")
        except Exception as e:
            print(f"Error loading formula: {e}")
            raise
    
    @classmethod
    def load_from_file(cls, file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                function = pickle.loads(data['function'].encode('latin1'))
                
                params = data['params']
                if isinstance(params, list) and len(params) > 0 and all(isinstance(p, (int, float)) for p in params):
                    params = np.array(params)
                
                return cls(
                    name=data['name'],
                    model_type=data['model_type'],
                    function=function,
                    params=params
                )
        except Exception as e:
            print(f"Error loading formula from file: {e}")
            raise

def save_multiple_formulas(formulas, directory):
    os.makedirs(directory, exist_ok=True)
    for formula in formulas:
        filename = f"{formula.name.replace(' ', '_')}.json"
        filepath = os.path.join(directory, filename)
        formula.save(filepath)
    print(f"Saved {len(formulas)} formulas to {directory}")

def load_multiple_formulas(directory):
    formulas = []
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return formulas
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            try:
                formula = SentimentFormula.load_from_file(filepath)
                formulas.append(formula)
            except Exception as e:
                print(f"Failed to load formula from {filename}: {e}")
    
    print(f"Loaded {len(formulas)} formulas from {directory}")
    return formulas


def actor_formula_v1(X, lambda_actor, w, b):
    s_init_actor, driver = X
    s_new = lambda_actor * s_init_actor + (1 - lambda_actor) * w * driver + b
    return np.tanh(s_new)

def actor_formula_v2(X, w_actor, w_driver, b):
    s_init_actor, driver = X
    s_new = w_actor * s_init_actor + w_driver * driver + b
    return np.tanh(s_new)

def target_formula_v1(X, lambda_target, w, b):
    s_init_target, s_action = X
    s_new = lambda_target * s_init_target + (1 - lambda_target) * w * s_action + b
    return np.tanh(s_new)

def target_formula_v2(X, w_target, w_action, b):
    s_init_target, s_action = X
    s_new = w_target * s_init_target + w_action * s_action + b
    return np.tanh(s_new)

def assoc_formula_v1(X, lambda_val, w, b):
    s_init, s_other = X
    s_new = lambda_val * s_init + (1 - lambda_val) * w * s_other + b
    return np.tanh(s_new)

def assoc_formula_v2(X, w_entity, w_other, b):
    s_init, s_other = X
    s_new = w_entity * s_init + w_other * s_other + b
    return np.tanh(s_new)

def belong_formula_v1(X, lambda_parent, w, b):
    s_entity, s_other = X
    s_new = lambda_parent * s_entity + (1 - lambda_parent) * w * s_other + b
    return np.tanh(s_new)

def belong_formula_v2(X, w_parent, w_child, b):
    s_entity, s_child = X
    s_new = w_parent * s_entity + w_child * s_child + b
    return np.tanh(s_new)


actor_formula_lambda = SentimentFormula(
    name="actor_formula_lambda",
    model_type="actor",
    function=actor_formula_v1
)

actor_formula_weighted = SentimentFormula(
    name="actor_formula_weights",
    model_type="actor",
    function=actor_formula_v2
)

target_formula_lambda = SentimentFormula(
    name="target_formula_lambda",
    model_type="target",
    function=target_formula_v1
)

target_formula_weighted = SentimentFormula(
    name="target_formula_weights",
    model_type="target",
    function=target_formula_v2
)

assoc_formula_lambda = SentimentFormula(
    name="assoc_formula_lambda",
    model_type="assoc",
    function=assoc_formula_v1
)

assoc_formula_weighted = SentimentFormula(
    name="assoc_formula_weights",
    model_type="assoc",
    function=assoc_formula_v2
)

belong_formula_lambda = SentimentFormula(
    name="belong_formula_lambda",
    model_type="belong",
    function=belong_formula_v1
)

belong_formula_weighted = SentimentFormula(
    name="belong_formula_weights",
    model_type="belong",
    function=belong_formula_v2
)

