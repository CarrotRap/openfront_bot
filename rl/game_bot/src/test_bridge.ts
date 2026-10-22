/**
 * Smoke test for python_bridge — étape 2a.
 *
 * Spawns `drive.py` with a trained model, feeds it a hand-crafted zero
 * observation with the right shapes, prints the action we get back.
 *
 * Usage:
 *   tsx src/test_bridge.ts <model.zip> [--grid 16] [--max-players 12]
 *
 * The shapes MUST match the model's checkpoint.
 */

import { PyBridge, type Observation } from "./python_bridge";
import {
  MC,
  MINIMAP_SIDE,
  S,
  PF,
  totalChannels,
} from "../../bridge/src/spec";

type Args = { model: string; grid: number; maxPlayers: number };

function parseArgs(argv: string[]): Args {
  if (argv.length === 0)
    throw new Error("usage: test_bridge.ts <model.zip> [--grid 16] [--max-players 12]");
  const out: Args = { model: argv[0], grid: 16, maxPlayers: 12 };
  for (let i = 1; i < argv.length; i++) {
    const key = argv[i];
    const val = () => {
      const v = argv[++i];
      if (v === undefined) throw new Error(`missing value for ${key}`);
      return v;
    };
    switch (key) {
      case "--grid":
        out.grid = parseInt(val(), 10);
        break;
      case "--max-players":
        out.maxPlayers = parseInt(val(), 10);
        break;
      default:
        throw new Error(`unknown arg: ${key}`);
    }
  }
  return out;
}

function ones(n: number): number[] {
  return Array(n).fill(1);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const G = args.grid;
  const K = args.maxPlayers;
  const C = totalChannels(K);
  const spatialLen = C * G * G;
  const minimapLen = MC * MINIMAP_SIDE * MINIMAP_SIDE;
  const playersLen = K * PF;

  process.stderr.write(
    `[test] shapes: spatial=(${C},${G},${G}) minimap=(${MC},${MINIMAP_SIDE},${MINIMAP_SIDE}) scalar=${S} players=(${K},${PF})\n`,
  );

  const obs: Observation = {
    spatial: Array(spatialLen).fill(0),
    spatial_shape: [C, G, G],
    minimap: Array(minimapLen).fill(0),
    minimap_shape: [MC, MINIMAP_SIDE, MINIMAP_SIDE],
    scalar: Array(S).fill(0),
    players: Array(playersLen).fill(0),
    players_shape: [K, PF],
    masks: {
      action_type: ones(4),
      target_slot: ones(K + 1),
      ratio: ones(6),
      unit_type: ones(7),
      cell: ones(G * G),
    },
  };

  const py = new PyBridge("../.venv/bin/python", ["drive.py", args.model]);
  process.stderr.write("[test] sending zero observation...\n");
  const action = await py.predict(obs);
  process.stderr.write(`[test] action: ${JSON.stringify(action)}\n`);
  py.close();
}

main().catch((e) => {
  process.stderr.write(`[test] fatal: ${e}\n`);
  process.exit(1);
});
