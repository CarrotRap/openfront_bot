"""`OpenFrontVecEnv` must be observationally identical to `DummyVecEnv`.

The pipelined vec env re-implements the VecEnv contract by hand — auto-reset,
`terminal_observation`, mask plumbing — so it is worth pinning against the
reference implementation rather than trusting it.

    python tests/test_vec_equivalence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openfront_gym import OpenFrontEnv, OpenFrontVecEnv  # noqa: E402

CONFIG = {"map": "big_plains", "bots": 6, "grid": 16, "maxSteps": 40, "maxPlayers": 8}
N_ENVS = 3
N_STEPS = 120


def _env_fns():
    return [
        (lambda i=i: OpenFrontEnv(config=CONFIG, seed_prefix=f"w{i}"))
        for i in range(N_ENVS)
    ]


def _run(venv):
    obs = venv.reset()
    rng = np.random.default_rng(123)
    rewards, dones, masks, terminals = [], [], [], []
    for _ in range(N_STEPS):
        actions = rng.integers([4, 9, 6, 7, 256], size=(N_ENVS, 5))
        masks.append(np.stack(venv.env_method("action_masks")))
        obs, reward, done, infos = venv.step(actions)
        rewards.append(reward.copy())
        dones.append(done.copy())
        terminals.append(sum("terminal_observation" in i for i in infos))
    venv.close()
    return {
        "rewards": np.array(rewards),
        "dones": np.array(dones),
        "masks": np.array(masks),
        "terminals": np.array(terminals),
        "final_obs": obs["spatial"].copy(),
    }


def test_pipelined_matches_dummy() -> None:
    reference = _run(DummyVecEnv(_env_fns()))
    pipelined = _run(OpenFrontVecEnv(_env_fns()))

    for key in reference:
        assert np.array_equal(reference[key], pipelined[key]), f"{key} differs"

    # A run with no episode boundary would not exercise auto-reset at all.
    assert reference["terminals"].sum() > 0, "no episode ended; test is vacuous"


if __name__ == "__main__":
    test_pipelined_matches_dummy()
    print("OK — pipelined vec env matches DummyVecEnv exactly")
