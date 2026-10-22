"""Recover the env config a saved policy expects.

A MaskablePPO checkpoint pins the observation shapes it was trained on, but
nothing about the game behind them. Loading one into a differently-shaped env
fails deep inside SB3 with a message about Box shapes that says nothing about
which flag to change — so resolve it here instead.

Two sources, in order:

1. ``<model>.config.json``, written next to the checkpoint by `train.py`. It
   carries the whole env config: map, bots, nations, window settings.
2. The checkpoint's own `observation_space`, which still pins `grid` and
   `maxPlayers` even when the sidecar is missing (models trained before it
   existed, or checkpoints moved around).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SIDECAR_SUFFIX = ".config.json"


def sidecar_path(model_path: str | Path) -> Path:
    """`runs/ppo.zip` -> `runs/ppo.config.json`."""
    p = Path(model_path)
    return p.with_name(p.stem + SIDECAR_SUFFIX)


def save_run_config(model_path: str | Path, config: dict[str, Any]) -> Path:
    path = sidecar_path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    return path


def load_run_config(model_path: str | Path) -> dict[str, Any] | None:
    path = sidecar_path(model_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def shapes_from_model(model: Any) -> dict[str, int]:
    """`grid` and `maxPlayers` as pinned by the checkpoint's observation space."""
    space = model.observation_space
    out: dict[str, int] = {}
    try:
        out["grid"] = int(space["spatial"].shape[-1])
        out["maxPlayers"] = int(space["players"].shape[0])
    except (KeyError, TypeError, IndexError):
        pass
    return out


def resolve_config(
    model_path: str | Path | None,
    model: Any | None,
    overrides: dict[str, Any],
    defaults: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Merge defaults < checkpoint < explicit CLI flags.

    `overrides` holds only the flags the user actually typed (None elsewhere).
    Returns the config plus human-readable notes about what was inferred, so
    the caller can print why it is not using the defaults.
    """
    config = dict(defaults)
    notes: list[str] = []

    if model_path is not None:
        sidecar = load_run_config(model_path)
        if sidecar:
            config.update(sidecar)
            notes.append(f"config lue depuis {sidecar_path(model_path).name}")
        if model is not None:
            for key, value in shapes_from_model(model).items():
                if config.get(key) != value:
                    notes.append(f"{key}={value} (imposé par le modèle)")
                config[key] = value

    explicit = {k: v for k, v in overrides.items() if v is not None}
    for key, value in explicit.items():
        if key in ("grid", "maxPlayers") and config.get(key) != value:
            raise ValueError(
                f"--{_flag(key)} = {value} contredit le modèle, qui attend "
                f"{config.get(key)}. Retirez le drapeau ou rechargez le bon "
                f"modèle : les shapes d'observation ne sont pas négociables."
            )
    config.update(explicit)
    return config, notes


def _flag(key: str) -> str:
    """`maxPlayers` -> `max-players`, so the message names the actual flag."""
    return "".join("-" + c.lower() if c.isupper() else c for c in key)


__all__ = [
    "load_run_config",
    "save_run_config",
    "shapes_from_model",
    "resolve_config",
    "sidecar_path",
]
