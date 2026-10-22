#!/usr/bin/env python
"""Train a MaskablePPO agent on OpenFront.

    python train.py --timesteps 200000 --envs 8 --map big_plains --bots 6

Each parallel env is its own Node process running the real game simulation, so
`--envs` is bounded by cores and RAM rather than by anything in the bridge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from openfront_gym import OpenFrontEnv, OpenFrontVecEnv  # noqa: E402
from openfront_gym.checkpoint import save_run_config  # noqa: E402
from openfront_gym.policy import DEFAULT_POLICY_KWARGS, OpenFrontPolicy  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--map", default="big_plains")
    p.add_argument("--map-source", default="testdata", choices=["testdata", "resources"])
    p.add_argument("--bots", type=int, default=6)
    p.add_argument("--nations", default="disabled")
    p.add_argument("--difficulty", default="Medium")
    p.add_argument("--grid", type=int, default=64, help="side of the action grid")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--ticks-per-step", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--vec",
        default="pipelined",
        choices=["pipelined", "dummy", "subproc"],
        help="pipelined runs the Node sims concurrently; dummy serialises them "
        "(fine only on small maps)",
    )
    p.add_argument("--max-players", type=int, default=8, help="K opponent slots")
    p.add_argument("--window-tiles", default="adaptive")
    p.add_argument("--window-min-tiles", type=int, default=64)
    p.add_argument("--window-zoom", type=float, default=2.5)
    p.add_argument(
        "--init-from",
        default=None,
        help="continue from a saved model instead of starting fresh — le "
        "curriculum se fait comme ça. grid et maxPlayers doivent correspondre.",
    )
    p.add_argument("--out", default="runs/ppo_openfront")
    p.add_argument("--tensorboard", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "map": args.map,
        "mapSource": args.map_source,
        "bots": args.bots,
        "nations": args.nations,
        "difficulty": args.difficulty,
        "grid": args.grid,
        "maxSteps": args.max_steps,
        "ticksPerStep": args.ticks_per_step,
        "maxPlayers": args.max_players,
        "windowTiles": (
            args.window_tiles
            if args.window_tiles in ("adaptive", "auto")
            else int(args.window_tiles)
        ),
        "windowMinTiles": args.window_min_tiles,
        "windowZoom": args.window_zoom,
    }

    def make_env(rank: int):
        def _init():
            return OpenFrontEnv(config=config, seed_prefix=f"w{rank}")

        return _init

    fns = [make_env(i) for i in range(args.envs)]
    vec_cls = {
        "pipelined": OpenFrontVecEnv,
        "dummy": DummyVecEnv,
        "subproc": SubprocVecEnv,
    }[args.vec]
    # VecMonitor rather than a per-env Monitor: the pipelined vec env owns its
    # envs directly, and episode stats belong at the vector level either way.
    venv = VecMonitor(vec_cls(fns))

    if args.init_from:
        # Reuse the weights, take the new env and hyperparameters. The shapes
        # must match, and SB3 says so clearly enough if they do not.
        print(f"reprise depuis {args.init_from}")
        model = MaskablePPO.load(
            args.init_from,
            env=venv,
            device=args.device,
            tensorboard_log=args.tensorboard,
            custom_objects={
                "learning_rate": args.lr,
                "ent_coef": args.ent_coef,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
            },
        )
    else:
        # Custom policy class: provides a spatial pointer head for `cell` and
        # a transformer+pointer stack for `target_slot`. See
        # openfront_gym/policy.py and rl/game_bot/nn_schema_v2.pdf.
        model = MaskablePPO(
            OpenFrontPolicy,
            venv,
            policy_kwargs=DEFAULT_POLICY_KWARGS,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
            learning_rate=args.lr,
            device=args.device,
            tensorboard_log=args.tensorboard,
            verbose=1,
        )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"policy parameters: {n_params:,}")

    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=True)
    finally:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out))
        # Sidecar: the checkpoint pins the observation shapes but knows nothing
        # about the game behind them. Without this, replaying a model means
        # retyping every flag correctly or hitting an opaque shape error.
        sidecar = save_run_config(f"{out}.zip", config)
        print(f"saved -> {out}.zip  (+ {sidecar.name})")
        venv.close()


if __name__ == "__main__":
    main()
