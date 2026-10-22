#!/usr/bin/env python
"""Compare a trained policy against the masked-random baseline.

    python eval.py --model runs/ppo_world.zip --episodes 10

The map, opponents and observation shapes come from the checkpoint (its
`.config.json` sidecar, plus its own observation space); flags only override.

Both policies are run on the *same* explicit seeds, so the comparison is not a
coin flip between two different sets of games — with `reset(seed=n)` the map,
the spawn and every opponent are identical for both.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from openfront_gym import OpenFrontEnv, sample_masked_action  # noqa: E402
from openfront_gym.checkpoint import resolve_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None, help="a saved MaskablePPO .zip")
    p.add_argument("--episodes", type=int, default=10)
    # None = "not given", so --model can supply it from its sidecar.
    p.add_argument("--map", default=None)
    p.add_argument("--map-source", default=None, choices=["testdata", "resources"])
    p.add_argument("--bots", type=int, default=None)
    p.add_argument("--nations", default=None)
    p.add_argument("--difficulty", default=None)
    p.add_argument("--grid", type=int, default=None)
    p.add_argument("--max-players", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--window-tiles", default=None)
    p.add_argument("--deterministic", action="store_true")
    return p.parse_args()


def rollout(env: OpenFrontEnv, act, seed: int, max_steps: int) -> dict[str, float]:
    obs, _ = env.reset(seed=seed)
    total, steps, invalid = 0.0, 0, 0
    info: dict = {}
    for _ in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(act(obs, seed))
        total += reward
        steps += 1
        invalid += bool(info.get("invalid"))
        if terminated or truncated:
            break
    return {
        "return": total,
        "steps": steps,
        "tiles": info.get("tiles", 0),
        "alive": float(bool(info.get("alive"))),
        "won": float(bool(info.get("won"))),
        "invalid": invalid,
    }


def summarize(name: str, rows: list[dict[str, float]]) -> None:
    def col(k: str) -> np.ndarray:
        return np.array([r[k] for r in rows], dtype=float)

    print(
        f"  {name:22s} retour {col('return').mean():8.2f} +/- {col('return').std():5.2f}"
        f" | survie {col('alive').mean():5.0%}"
        f" | victoires {col('won').mean():5.0%}"
        f" | tuiles {col('tiles').mean():9.0f}"
        f" | durée {col('steps').mean():5.0f}"
        f" | rejets {col('invalid').mean():5.0f}"
    )


def main() -> None:
    args = parse_args()
    model = None
    if args.model:
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(args.model, device="cpu")

    window = args.window_tiles
    overrides = {
        "map": args.map,
        "mapSource": args.map_source,
        "bots": args.bots,
        "nations": args.nations,
        "difficulty": args.difficulty,
        "grid": args.grid,
        "maxPlayers": args.max_players,
        "maxSteps": args.max_steps,
        "windowTiles": (
            None if window is None else
            window if window in ("adaptive", "auto") else int(window)
        ),
    }
    defaults = {
        "map": "big_plains", "mapSource": "testdata", "bots": 6,
        "nations": "disabled", "difficulty": "Medium", "grid": 16,
        "maxPlayers": 8, "maxSteps": 600, "windowTiles": "adaptive",
    }
    try:
        config, notes = resolve_config(args.model, model, overrides, defaults)
    except ValueError as exc:
        print(f"erreur : {exc}")
        raise SystemExit(2)

    env = OpenFrontEnv(config=config)
    seeds = list(range(args.episodes))

    for note in notes:
        print(f"  {note}")
    print(
        f"\n{config['map']} · {config['bots']} bots + {config['nations']} nations"
        f" · {config['difficulty']}"
    )
    print(f"{args.episodes} épisodes, mêmes graines pour les deux politiques\n")

    rng = np.random.default_rng(0)
    max_steps = config["maxSteps"]
    baseline = [
        rollout(env, lambda o, s: sample_masked_action(env, rng), s, max_steps)
        for s in seeds
    ]
    summarize("aléatoire (masqué)", baseline)

    if model is not None:

        def act(obs, _seed):
            action, _ = model.predict(
                obs,
                action_masks=env.action_masks(),
                deterministic=args.deterministic,
            )
            return np.asarray(action).reshape(-1)

        trained = [rollout(env, act, s, max_steps) for s in seeds]
        summarize(Path(args.model).stem, trained)

        gap = np.mean([r["return"] for r in trained]) - np.mean(
            [r["return"] for r in baseline]
        )
        print(f"\n  écart : {gap:+.2f}")
    env.close()


if __name__ == "__main__":
    main()
