"""Gymnasium bindings for the OpenFront.io simulation."""

from .bridge import BridgeError, NodeBridge
from .checkpoint import load_run_config, resolve_config, save_run_config
from .env import MASK_KEYS, OpenFrontEnv, sample_masked_action
from .vec import OpenFrontVecEnv

__all__ = [
    "OpenFrontEnv",
    "OpenFrontVecEnv",
    "NodeBridge",
    "BridgeError",
    "sample_masked_action",
    "MASK_KEYS",
    "load_run_config",
    "save_run_config",
    "resolve_config",
]

try:  # optional: lets `gym.make("OpenFront-v0")` work
    from gymnasium.envs.registration import register

    register(id="OpenFront-v0", entry_point="openfront_gym.env:OpenFrontEnv")
except Exception:  # pragma: no cover - registration is a convenience only
    pass
