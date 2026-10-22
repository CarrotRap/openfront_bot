"""Drive the Node bot client with a trained MaskablePPO policy.

Reads one observation per line on stdin (JSON), prints one action per line
on stdout (JSON). Same stdio pattern as rl/bridge.

See src/python_bridge.ts for the exact JSON protocol — flat arrays with a
`<name>_shape` sibling for every tensor that isn't 1D.

Launched by bot_client.ts as a child process:

    const py = new PyBridge("../.venv/bin/python",
      ["drive.py", "runs/foo.zip", "--deterministic", "--mask-noop"]);

Flags:
  --deterministic   argmax instead of sampling; use with a well-trained model.
                    On a lightly trained policy, argmax collapses to NOOP (the
                    single most-probable action across heads) — pair with
                    --mask-noop then.
  --mask-noop       force the policy to pick ATTACK/BOAT/BUILD on action_type,
                    unless they're all masked out. NOOP still has mass over
                    whatever action_type ends up chosen so the model can
                    "stand still" implicitly via a no-op target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# SB3 checkpoints pickle the custom policy / extractor by qualified name
# (`openfront_gym.policy.OpenFrontPolicy` etc.). The module must be
# importable before `MaskablePPO.load` runs, otherwise cloudpickle fails
# with `No module named 'openfront_gym'`. The package lives at
# rl/openfront_gym, one directory up from this script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import openfront_gym  # noqa: F401 — register custom classes with cloudpickle

import numpy as np
from sb3_contrib import MaskablePPO

# A_NOOP = 0 in rl/bridge/src/spec.ts — the first slot of the action_type
# head. Dropping it in the mask forces the policy to pick a real action.
A_NOOP_INDEX = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("model", help="Path to the MaskablePPO .zip checkpoint")
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="argmax instead of sampling; needs --mask-noop with a short-trained policy",
    )
    p.add_argument(
        "--mask-noop",
        action="store_true",
        help="mask NOOP in action_type so the bot always does something concrete",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = MaskablePPO.load(args.model, device="cpu")
    print(
        f"[drive] loaded {args.model} deterministic={args.deterministic} mask_noop={args.mask_noop}",
        file=sys.stderr,
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            masks = payload["masks"]
            obs = {
                "spatial": np.asarray(payload["spatial"], dtype=np.float32).reshape(
                    payload["spatial_shape"]
                ),
                "minimap": np.asarray(payload["minimap"], dtype=np.float32).reshape(
                    payload["minimap_shape"]
                ),
                "scalar": np.asarray(payload["scalar"], dtype=np.float32),
                "players": np.asarray(payload["players"], dtype=np.float32).reshape(
                    payload["players_shape"]
                ),
                # Training-time observation_space has mask_* entries (see
                # env.py MASK_KEYS). The policy's extractor ignores them,
                # but SB3's obs_to_tensor iterates over the space keys and
                # trips on anything missing. Pass them through unchanged.
                "mask_action_type": np.asarray(masks["action_type"], dtype=np.int8),
                "mask_target": np.asarray(masks["target_slot"], dtype=np.int8),
                "mask_ratio": np.asarray(masks["ratio"], dtype=np.int8),
                "mask_unit": np.asarray(masks["unit_type"], dtype=np.int8),
                "mask_cell": np.asarray(masks["cell"], dtype=np.int8),
            }
            at_mask = np.asarray(masks["action_type"], dtype=bool).copy()
            # Only drop NOOP if at least one real action is still available —
            # otherwise we'd hand MaskablePPO an all-false distribution for
            # action_type and the policy would raise (and even if it didn't,
            # the sampled action would be undefined).
            if args.mask_noop and at_mask[A_NOOP_INDEX] and at_mask[1:].any():
                at_mask[A_NOOP_INDEX] = False
            flat_mask = np.concatenate(
                [
                    at_mask,
                    np.asarray(masks["target_slot"], dtype=bool),
                    np.asarray(masks["ratio"], dtype=bool),
                    np.asarray(masks["unit_type"], dtype=bool),
                    np.asarray(masks["cell"], dtype=bool),
                ]
            )
            obs_batched = {k: v[None, ...] for k, v in obs.items()}
            action, _ = model.predict(
                obs_batched,
                deterministic=args.deterministic,
                action_masks=flat_mask,
            )
            action = np.asarray(action).flatten().tolist()
            out = {
                "action_type": int(action[0]),
                "target_slot": int(action[1]),
                "ratio": int(action[2]),
                "unit_type": int(action[3]),
                "cell": int(action[4]),
            }
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()
        except Exception as e:
            print(f"[drive] error on line: {e}", file=sys.stderr)
            sys.stdout.write('{"error":true}\n')
            sys.stdout.flush()


if __name__ == "__main__":
    main()
