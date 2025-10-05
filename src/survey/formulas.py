"""Parametric scoring formulas used in survey calibration."""

from __future__ import annotations

from typing import Sequence

import numpy as np

def null_identity(values: Sequence[float]) -> float:
    """Return the first entry unchanged."""
    return float(values[0])

def null_avg(values: Sequence[float]) -> float:
    """Average all provided entries."""
    return float(np.mean(values))

def null_linear(values: Sequence[float], w1: float, w2: float, b: float) -> float:
    """Apply a simple linear combination to two scores."""
    return w1 * values[0] + w2 * values[1] + b

def actor_formula_v1(values: Sequence[float], lambda_actor: float, w: float, b: float) -> float:
    """Blend actor prior sentiment with a driver signal using inertia ``lambda_actor``."""
    s_init_actor, driver = values
    s_new = lambda_actor * s_init_actor + (1 - lambda_actor) * w * driver + b
    return np.tanh(s_new)

def actor_formula_v2(values: Sequence[float], w_actor: float, w_driver: float, b: float) -> float:
    """Combine actor prior sentiment and driver directly with tunable weights."""
    s_init_actor, driver = values
    s_new = w_actor * s_init_actor + w_driver * driver + b
    return np.tanh(s_new)

def target_formula_v1(values: Sequence[float], lambda_target: float, w: float, b: float) -> float:
    """Diffuse target sentiment with the action sentiment under inertia ``lambda_target``."""
    s_init_target, s_action = values
    s_new = lambda_target * s_init_target + (1 - lambda_target) * w * s_action + b
    return np.tanh(s_new)

def target_formula_v2(values: Sequence[float], w_target: float, w_action: float, b: float) -> float:
    """Linearly mix target prior sentiment and action sentiment."""
    s_init_target, s_action = values
    s_new = w_target * s_init_target + w_action * s_action + b
    return np.tanh(s_new)

def assoc_formula_v1(values: Sequence[float], lambda_val: float, w: float, b: float) -> float:
    """Blend entity sentiment with a related entity under inertia ``lambda_val``."""
    s_init, s_other = values
    s_new = lambda_val * s_init + (1 - lambda_val) * w * s_other + b
    return np.tanh(s_new)

def assoc_formula_v2(values: Sequence[float], w_entity: float, w_other: float, b: float) -> float:
    """Linearly combine sentiments of two associated entities."""
    s_init, s_other = values
    s_new = w_entity * s_init + w_other * s_other + b
    return np.tanh(s_new)

def belong_formula_v1(values: Sequence[float], lambda_parent: float, w: float, b: float) -> float:
    """Model parent-child sentiment diffusion with inertia ``lambda_parent``."""
    s_entity, s_other = values
    s_new = lambda_parent * s_entity + (1 - lambda_parent) * w * s_other + b
    return np.tanh(s_new)

def belong_formula_v2(values: Sequence[float], w_parent: float, w_child: float, b: float) -> float:
    """Linearly mix parent and child sentiment contributions."""
    s_entity, s_child = values
    s_new = w_parent * s_entity + w_child * s_child + b
    return np.tanh(s_new)