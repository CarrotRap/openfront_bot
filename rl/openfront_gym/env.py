"""Gymnasium environment for OpenFront.io.

One instance owns one headless Node process running the real game simulation
(``src/core``), against the game's own scripted opponents (tribes/nations).

Observation (Dict):
    spatial  (C, G, G) float32  ego-centric map view, C channels (static +
                                 K per-slot channels appended)
    minimap  (MC, M, M) float32 fixed-resolution global view (sees past the
                                 zoom window)
    scalar   (S,)      float32  global state of the agent
    players  (K, PF)   float32  the K most relevant opponents
    mask_*             uint8    per-head action masks

Action: MultiDiscrete([n_action_types, K + 1, n_ratios, n_units, G * G])
    [action_type, target_slot, ratio, unit_type, cell]

`action_masks()` returns the flat concatenated mask expected by
sb3-contrib's MaskablePPO.
"""

from __future__ import annotations

import itertools
import os
import subprocess
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .bridge import BridgeError, NodeBridge

_INSTANCE_COUNTER = itertools.count()

MASK_KEYS = (
    "mask_action_type",
    "mask_target",
    "mask_ratio",
    "mask_unit",
    "mask_cell",
)
DENSE_KEYS = ("spatial", "minimap", "scalar", "players")
OBS_KEYS = DENSE_KEYS + MASK_KEYS


class OpenFrontEnv(gym.Env):
    """A single-agent view of an OpenFront FFA game."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        game_root: str | None = None,
        seed_prefix: str | None = None,
        include_masks_in_obs: bool = True,
        quiet: bool = True,
    ) -> None:
        super().__init__()
        self.bridge = NodeBridge(
            game_root=game_root,
            stderr_to=subprocess.DEVNULL if quiet else None,
        )
        header, _ = self.bridge.request(cmd="init", config=config or {})
        self.spec_info: dict[str, Any] = header["spec"]
        self.config: dict[str, Any] = self.spec_info["config"]

        self.channels: list[str] = self.spec_info["observation"]["spatial"]["channels"]
        self.scalar_names: list[str] = self.spec_info["observation"]["scalar"]["names"]
        self.player_features: list[str] = self.spec_info["observation"]["players"][
            "features"
        ]
        self.action_types: list[str] = self.spec_info["action"]["action_types"]
        self.ratios: list[float] = self.spec_info["action"]["ratios"]
        self.build_units: list[str] = self.spec_info["action"]["build_units"]
        self.grid: int = self.spec_info["action"]["grid"]

        self.nvec = np.array(self.spec_info["action"]["nvec"], dtype=np.int64)
        self.action_space = spaces.MultiDiscrete(self.nvec)

        obs_spec = self.spec_info["observation"]
        obs_dict: dict[str, spaces.Space] = {
            "spatial": spaces.Box(
                0.0, 1.0, tuple(obs_spec["spatial"]["shape"]), np.float32
            ),
            "minimap": spaces.Box(
                0.0, 1.0, tuple(obs_spec["minimap"]["shape"]), np.float32
            ),
            "scalar": spaces.Box(
                0.0, 1.0, tuple(obs_spec["scalar"]["shape"]), np.float32
            ),
            "players": spaces.Box(
                0.0, 1.0, tuple(obs_spec["players"]["shape"]), np.float32
            ),
        }
        self.minimap_channels: list[str] = obs_spec["minimap"]["channels"]
        self._include_masks = include_masks_in_obs
        if include_masks_in_obs:
            for key in MASK_KEYS:
                obs_dict[key] = spaces.MultiBinary(int(obs_spec[key]["shape"][0]))
        self.observation_space = spaces.Dict(obs_dict)

        # Distinct by default: two envs sharing a prefix would play byte-for-byte
        # identical games, which silently wastes every worker after the first.
        self._seed_prefix = seed_prefix or f"p{os.getpid()}i{next(_INSTANCE_COUNTER)}"
        self._episode = 0
        self._masks: dict[str, np.ndarray] = {}
        self._extra: dict[str, np.ndarray] = {}
        self.last_info: dict[str, Any] = {}
        # Populated only when the env was created with `emitRender: True`.
        self.last_render: dict[str, Any] | None = None

    # ------------------------------------------------------------- gym API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.bridge.send(cmd="reset", seed=self._next_seed(seed))
        return self._collect_reset(), self.last_info

    def step(
        self, action: np.ndarray | list[int]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action_list = [int(a) for a in np.asarray(action).reshape(-1)]
        self.bridge.send(cmd="step", action=action_list)
        return self._collect_step()

    # The send/collect halves are separate so `OpenFrontVecEnv` can fan
    # commands out to every Node process before reading any answer back.

    def _next_seed(self, seed: int | None = None) -> str:
        """Game seed for the next episode.

        An explicit `seed` must name the same game in every process, so it
        deliberately skips the per-instance prefix — that prefix exists only to
        keep *unseeded* workers from replaying identical games, and baking a
        pid into an explicit seed makes `reset(seed=0)` irreproducible across
        runs.
        """
        if seed is not None:
            return f"seed-{seed}"
        tag = self._episode
        self._episode += 1
        return f"{self._seed_prefix}-{tag}"

    def _collect_reset(self) -> dict[str, np.ndarray]:
        header, arrays = self.bridge.recv()
        obs = self._unpack(arrays)
        self.last_info = dict(header.get("info", {}))
        self._capture_render(header, arrays)
        return obs

    def _collect_step(
        self,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        header, arrays = self.bridge.recv()
        obs = self._unpack(arrays)
        self.last_info = dict(header.get("info", {}))
        self._capture_render(header, arrays)
        return (
            obs,
            float(header["reward"]),
            bool(header["terminated"]),
            bool(header["truncated"]),
            self.last_info,
        )

    def close(self) -> None:
        self.bridge.close()

    def render(self) -> str:
        i = self.last_info
        return (
            f"tick={i.get('tick')} tiles={i.get('tiles')} "
            f"troops={i.get('troops')} gold={i.get('gold')} "
            f"alive={i.get('players_alive')} last={i.get('action_label')}"
        )

    # ------------------------------------------------------------- masking

    def action_masks(self) -> np.ndarray:
        """Flat mask over every head, in nvec order (MaskablePPO convention)."""
        return np.concatenate(
            [self._masks[k].astype(bool) for k in MASK_KEYS], dtype=bool
        )

    def masks_by_head(self) -> dict[str, np.ndarray]:
        """Per-head masks, plus the conditional build/boat cell masks."""
        out = {k: self._masks[k].copy() for k in MASK_KEYS}
        out.update({k: v.copy() for k, v in self._extra.items()})
        return out

    # ------------------------------------------------------------ internals

    def _unpack(self, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self._masks = {k: arrays[k] for k in MASK_KEYS}
        self._extra = {
            k: arrays[k] for k in ("mask_cell_build", "mask_cell_boat") if k in arrays
        }
        obs: dict[str, np.ndarray] = {}
        for key in DENSE_KEYS:
            obs[key] = np.clip(arrays[key], 0.0, 1.0).astype(np.float32, copy=True)
        if self._include_masks:
            for key in MASK_KEYS:
                obs[key] = arrays[key].astype(np.int8, copy=True)
        return obs

    def _capture_render(
        self, header: dict[str, Any], arrays: dict[str, np.ndarray]
    ) -> None:
        render = header.get("render")
        if render is None or "render_pixels" not in arrays:
            self.last_render = None
            return
        self.last_render = {**render, "pixels": arrays["render_pixels"]}

    # ------------------------------------------------------------- helpers

    def describe_action(self, action: np.ndarray | list[int]) -> str:
        a = [int(x) for x in np.asarray(action).reshape(-1)]
        kind = self.action_types[a[0]]
        if kind == "NOOP":
            return "NOOP"
        if kind == "ATTACK":
            slot = "TerraNullius" if a[1] == 0 else f"player_slot_{a[1] - 1}"
            return f"ATTACK {slot} with {self.ratios[a[2]]:.0%} of troops"
        cell = f"cell({a[4] % self.grid},{a[4] // self.grid})"
        if kind == "BOAT_ATTACK":
            return f"BOAT_ATTACK -> {cell} with {self.ratios[a[2]]:.0%} of troops"
        return f"BUILD {self.build_units[a[3]]} -> {cell}"


def sample_masked_action(
    env: OpenFrontEnv, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Uniform random action that respects the masks — a sane baseline."""
    rng = rng or np.random.default_rng()
    masks = env.masks_by_head()
    out = []
    for key, n in zip(MASK_KEYS, env.nvec):
        valid = np.flatnonzero(masks[key][:n])
        out.append(int(rng.choice(valid)) if valid.size else 0)
    return np.array(out, dtype=np.int64)


__all__ = ["OpenFrontEnv", "BridgeError", "sample_masked_action", "MASK_KEYS"]
