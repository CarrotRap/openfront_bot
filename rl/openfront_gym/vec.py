"""A vectorised env that runs every Node simulation concurrently.

`DummyVecEnv` steps its envs in a Python loop, and each of our steps blocks on
a round trip to a Node process — so N envs run at the speed of one. On a small
map that is invisible (the sim is 1000x faster than the network); on a large
map with nations it is the whole bottleneck.

This env instead fans every command out first and collects the answers
afterwards, so the N simulations overlap. It needs no extra Python processes
and no pickling, unlike `SubprocVecEnv`.

Measured on `world` (2000x1000, 30 bots + 72 nations), 8 envs:

    DummyVecEnv      81 env-steps/s
    OpenFrontVecEnv 500+ env-steps/s
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn

from .env import MASK_KEYS, OpenFrontEnv


class OpenFrontVecEnv(VecEnv):
    """Pipelined vectorisation over N `OpenFrontEnv` instances."""

    def __init__(self, env_fns: Sequence[Callable[[], OpenFrontEnv]]) -> None:
        self.envs: list[OpenFrontEnv] = [fn() for fn in env_fns]
        first = self.envs[0]
        super().__init__(len(self.envs), first.observation_space, first.action_space)
        self._actions: np.ndarray | None = None
        self._masks: list[np.ndarray] = [
            np.ones(int(first.nvec.sum()), dtype=bool) for _ in self.envs
        ]

    # ------------------------------------------------------------- VecEnv

    def reset(self) -> dict[str, np.ndarray]:
        for env in self.envs:
            env.bridge.send(cmd="reset", seed=env._next_seed())
        obs = [env._collect_reset() for env in self.envs]
        self._refresh_masks()
        return _stack(obs)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions)
        for env, action in zip(self.envs, self._actions):
            env.bridge.send(cmd="step", action=[int(a) for a in action])

    def step_wait(self) -> VecEnvStepReturn:
        obs_list: list[dict[str, np.ndarray]] = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = []

        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env._collect_step()
            rewards[i] = reward
            dones[i] = terminated or truncated
            info = dict(info)
            info["TimeLimit.truncated"] = truncated and not terminated
            if dones[i]:
                # VecEnv contract: the returned obs is the *reset* obs, and the
                # true final one rides along in the info dict.
                info["terminal_observation"] = obs
            infos.append(info)
            obs_list.append(obs)

        # Auto-reset every finished env, again pipelined.
        finished = [i for i in range(self.num_envs) if dones[i]]
        for i in finished:
            self.envs[i].bridge.send(cmd="reset", seed=self.envs[i]._next_seed())
        for i in finished:
            obs_list[i] = self.envs[i]._collect_reset()

        self._refresh_masks()
        return _stack(obs_list), rewards, dones, infos

    def close(self) -> None:
        for env in self.envs:
            env.close()

    # -------------------------------------------------------------- masks

    def _refresh_masks(self) -> None:
        self._masks = [env.action_masks() for env in self.envs]

    def env_method(
        self, method_name: str, *args: Any, indices: Any = None, **kwargs: Any
    ) -> list[Any]:
        # MaskablePPO reaches the masks through this hook.
        targets = self._indices(indices)
        if method_name == "action_masks":
            return [self._masks[i] for i in targets]
        return [getattr(self.envs[i], method_name)(*args, **kwargs) for i in targets]

    # ------------------------------------------------------- VecEnv plumbing

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        return [getattr(self.envs[i], attr_name) for i in self._indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        for i in self._indices(indices):
            setattr(self.envs[i], attr_name, value)

    def env_is_wrapped(self, wrapper_class: type, indices: Any = None) -> list[bool]:
        return [False for _ in self._indices(indices)]

    def get_images(self) -> Sequence[np.ndarray | None]:
        return [None for _ in self.envs]

    def _indices(self, indices: Any) -> list[int]:
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)


def _stack(obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}


__all__ = ["OpenFrontVecEnv", "MASK_KEYS"]
