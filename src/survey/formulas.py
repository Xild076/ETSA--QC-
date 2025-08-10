import numpy as np

def null_identity(X):
    return X[0]

def null_avg(X):
    return np.mean(X)

def null_linear(X, w1, w2, b):
    return w1 * X[0] + w2 * X[1] + b

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