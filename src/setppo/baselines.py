"""Executable heuristics and an information-restricted external-policy interface.

CPO, RCPO, and sTRPO training is not fabricated from aggregate result rows.
"""
from dataclasses import replace
import numpy as np


VARIANTS = {
    "full": {},
    "no-adversarial": {"adversarial": False},
    "no-residual": {"selector": "posterior"},
    "no-belief": {"use_belief": False},
    "no-hierarchy": {"hierarchical": False},
    "no-set": {"set_encoder": False},
    "no-kfb": {"allow_feedback": False},
    "posterior": {"selector": "posterior"},
    "shuffled": {"selector": "shuffled"},
    "random-selector": {"selector": "random"},
    "oracle": {"selector": "oracle"},
}


def variant_config(config, name):
    return replace(config, **VARIANTS[name])


def heuristic(observation, name, rng):
    if name == "random":
        return int(rng.integers(len(observation.actions)))
    if name == "kfb-first":
        repairs = np.flatnonzero(observation.kinds == 1)
        return int(rng.choice(repairs)) if len(repairs) else int(rng.integers(len(observation.actions)))
    if name == "knowledge-greedy":
        # Public posterior mastery/expected gap only, without privileged reader/Eve states.
        scores = observation.items[:, -4] + (observation.kinds == 1) * observation.items[:, -2]
        return int(np.argmax(scores))
    raise ValueError(f"Unknown built-in heuristic {name}")


def external_policy(factory_path, config):
    """Load user-supplied CPO/RCPO/sTRPO policy: module:factory -> select_action(obs).

    A policy receives ONLY the same legitimate Observation as Set-PPO.
    Exact original baseline checkpoints and A/B/C configurations were not supplied.
    """
    import importlib
    module, factory = factory_path.split(":", 1)
    policy = getattr(importlib.import_module(module), factory)(config)
    if not hasattr(policy, "select_action"):
        raise TypeError("External policy must implement select_action(observation)")
    return policy

