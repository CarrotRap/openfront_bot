/**
 * WebSocket bot client — étape 2b.
 *
 * Joins a running OpenFront game, maintains a local simulation synchronized
 * with the server's Turn stream, and plays via a trained MaskablePPO policy
 * loaded in a Python child process.
 *
 * Flow:
 *   open     →  send `join`
 *   lobby_info  →  remember our server-assigned clientID
 *   start    →  init LocalSim with the authoritative GameStartInfo,
 *               build Encoder + ActionCodec against that sim
 *   turn     →  applyTurn into the local sim; every N ticks after we have
 *               spawned, encode an observation, call Python, encode an
 *               intent, send it to the server
 */
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { WebSocket } from "ws";
import {
  ClientJoinMessage,
  ServerMessage,
} from "src/core/Schemas";
import {
  createGameWireContext,
  encodeClientMessage,
  decodeServerMessage,
} from "src/core/ZbinWire";
import type { ZbContext } from "../../../OpenFrontIO/zbin";
import { LocalSim } from "./local_sim";
import { PyBridge, flatten, type Observation } from "./python_bridge";
import { CAN_BUILD_OFFSET, Encoder } from "../../bridge/src/observation";
import { ActionCodec } from "../../bridge/src/actions";
import { MC, MINIMAP_SIDE, S, PF, totalChannels } from "../../bridge/src/spec";
import { pickBestSpawn, shouldRepick } from "./spawn_picker";

type Args = {
  gameID: string;
  host: string;
  username: string;
  spectator: boolean;
  model: string | null;
  grid: number;
  maxPlayers: number;
  ticksPerStep: number;
  /** Episode length in RL steps the policy was trained with. Normalises the
   *  time-progression scalars; a mismatch here makes the policy think the game
   *  is further along than it is and triggers endgame behaviours (far boats,
   *  reckless ratios, no finishing) early. */
  maxSteps: number;
  windowMinTiles: number;
  windowZoom: number;
  /** argmax instead of sampling — passes --deterministic to drive.py. */
  deterministic: boolean;
  /** Force non-NOOP on action_type — passes --mask-noop to drive.py. */
  maskNoop: boolean;
};

/**
 * Values written next to a trained checkpoint by rl/openfront_gym/checkpoint.py
 * (sidecar `<model>.config.json`). Loading these at runtime keeps the policy's
 * observation preprocessing identical to training no matter which model is
 * passed — in particular `maxSteps` and the window hyperparameters, which
 * directly shape the scalar vector.
 */
interface TrainingSidecar {
  grid?: number;
  maxPlayers?: number;
  ticksPerStep?: number;
  maxSteps?: number;
  windowMinTiles?: number;
  windowZoom?: number;
}

function loadSidecar(modelPath: string): TrainingSidecar | null {
  // Checkpoint path is "runs/foo.zip" or "runs/foo"; sidecar is "runs/foo.config.json".
  const base = modelPath.endsWith(".zip") ? modelPath.slice(0, -4) : modelPath;
  const sidecarPath = `${base}.config.json`;
  try {
    const txt = fs.readFileSync(sidecarPath, "utf8");
    return JSON.parse(txt) as TrainingSidecar;
  } catch (e) {
    process.stderr.write(
      `[bot] no sidecar at ${sidecarPath} — falling back to CLI/defaults\n`,
    );
    return null;
  }
}

function parseArgs(argv: string[]): { args: Args; explicit: Set<string> } {
  if (argv.length === 0) {
    throw new Error(
      "usage: bot_client.ts <gameID> [--host ws://localhost:3001] [--model path.zip] [--grid 64] [--max-players 8] [--ticks-per-step 10]",
    );
  }
  const out: Args = {
    gameID: argv[0],
    host: "ws://localhost:3001",
    username: "BotBoy",
    spectator: false,
    model: null,
    grid: 64,
    maxPlayers: 8,
    ticksPerStep: 10,
    // Will be overridden by the sidecar when --model is passed. Keeping the
    // hardcoded fallback at 3000 (not 1200) because 3000 is the current
    // training default, so an eyes-off demo matches recent runs even if the
    // sidecar is missing. The sidecar is the source of truth when present.
    maxSteps: 3000,
    windowMinTiles: 64,
    windowZoom: 2.5,
    deterministic: false,
    maskNoop: false,
  };
  const explicit = new Set<string>();
  for (let i = 1; i < argv.length; i++) {
    const key = argv[i];
    const val = () => {
      const v = argv[++i];
      if (v === undefined) throw new Error(`missing value for ${key}`);
      return v;
    };
    switch (key) {
      case "--host":
        out.host = val();
        explicit.add(key);
        break;
      case "--username":
        out.username = val();
        explicit.add(key);
        break;
      case "--spectator":
        out.spectator = true;
        explicit.add(key);
        break;
      case "--model":
        out.model = val();
        explicit.add(key);
        break;
      case "--grid":
        out.grid = parseInt(val(), 10);
        explicit.add(key);
        break;
      case "--max-players":
        out.maxPlayers = parseInt(val(), 10);
        explicit.add(key);
        break;
      case "--ticks-per-step":
        out.ticksPerStep = parseInt(val(), 10);
        explicit.add(key);
        break;
      case "--max-steps":
        out.maxSteps = parseInt(val(), 10);
        explicit.add(key);
        break;
      case "--window-min-tiles":
        out.windowMinTiles = parseInt(val(), 10);
        explicit.add(key);
        break;
      case "--window-zoom":
        out.windowZoom = parseFloat(val());
        explicit.add(key);
        break;
      case "--deterministic":
        out.deterministic = true;
        explicit.add(key);
        break;
      case "--mask-noop":
        out.maskNoop = true;
        explicit.add(key);
        break;
      default:
        throw new Error(`unknown arg: ${key}`);
    }
  }
  return { args: out, explicit };
}

/**
 * Overlay the sidecar on top of the CLI args: a field the user set explicitly
 * on the command line wins; everything else is taken from the sidecar if
 * present. Shape-defining fields (grid, maxPlayers) that differ between
 * sidecar and CLI are an error — not matching them crashes the policy at
 * forward time with a shape mismatch, better fail loudly here.
 */
function applySidecar(
  args: Args,
  explicit: Set<string>,
  sidecar: TrainingSidecar,
): Args {
  const out = { ...args };
  const shapeKeys: { cli: string; sidecar: keyof TrainingSidecar; arg: keyof Args }[] = [
    { cli: "--grid", sidecar: "grid", arg: "grid" },
    { cli: "--max-players", sidecar: "maxPlayers", arg: "maxPlayers" },
  ];
  for (const { cli, sidecar: sk, arg } of shapeKeys) {
    const sv = sidecar[sk];
    if (sv === undefined) continue;
    if (explicit.has(cli) && (out[arg] as number) !== sv) {
      throw new Error(
        `${cli}=${out[arg]} contradicts the model's trained ${sk}=${sv}; the policy will not run with the wrong shape. Remove the CLI flag or use a matching model.`,
      );
    }
    (out[arg] as number) = sv;
  }
  const overrideKeys: { cli: string; sidecar: keyof TrainingSidecar; arg: keyof Args }[] = [
    { cli: "--ticks-per-step", sidecar: "ticksPerStep", arg: "ticksPerStep" },
    { cli: "--max-steps", sidecar: "maxSteps", arg: "maxSteps" },
    { cli: "--window-min-tiles", sidecar: "windowMinTiles", arg: "windowMinTiles" },
    { cli: "--window-zoom", sidecar: "windowZoom", arg: "windowZoom" },
  ];
  for (const { cli, sidecar: sk, arg } of overrideKeys) {
    const sv = sidecar[sk];
    if (sv === undefined || explicit.has(cli)) continue;
    (out[arg] as number) = sv;
  }
  return out;
}

async function main() {
  let { args, explicit } = parseArgs(process.argv.slice(2));
  if (args.model) {
    const sidecar = loadSidecar(args.model);
    if (sidecar) {
      args = applySidecar(args, explicit, sidecar);
      process.stderr.write(
        `[bot] sidecar applied: grid=${args.grid} maxPlayers=${args.maxPlayers} ticksPerStep=${args.ticksPerStep} maxSteps=${args.maxSteps} windowMinTiles=${args.windowMinTiles} windowZoom=${args.windowZoom}\n`,
      );
    }
  }
  const token = crypto.randomUUID();
  const gitCommit = "DEV";

  // Python inference stays null when --model is not passed; the bot then
  // never sends an intent and just watches, useful for debugging the sim.
  const driveArgs = ["drive.py", args.model ?? ""];
  if (args.deterministic) driveArgs.push("--deterministic");
  if (args.maskNoop) driveArgs.push("--mask-noop");
  const py = args.model
    ? new PyBridge("../.venv/bin/python", driveArgs)
    : null;

  const sim = { current: null as LocalSim | null };
  const assets = {
    encoder: null as Encoder | null,
    codec: null as ActionCodec | null,
    step: 0,
    tickCounter: 0,
    pending: false,
    // Tracks the previous value of me.goldEarned() so we can hand the encoder
    // the per-step income DELTA, matching env.ts:388's computation. Passing
    // the cumulative goldEarned here (as we did before) made the "income
    // rate" scalar saturate at 1.0 after a few minutes — training only ever
    // saw values up to ~0.3.
    prevGoldEarned: 0,
  };
  // Spawn-phase state: we re-pick every ~500ms while in spawn phase so we
  // react when a human moves their own spawn. lastSpawnTile lets us avoid
  // flooding the server with intents that would resolve to the same centre.
  const spawnState = {
    lastPickTick: null as number | null,
    lastSpawnTile: null as number | null,
  };
  let myClientID: string | null = null;
  let zbinCtx: ZbContext | null = null;

  const url = `${args.host}/w0/${args.gameID}`;
  process.stderr.write(`[bot] connecting to ${url}\n`);
  const ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";

  let pingTimer: NodeJS.Timeout | null = null;

  ws.on("open", () => {
    process.stderr.write(`[bot] socket open — sending join\n`);
    const join: ClientJoinMessage = {
      type: "join",
      token,
      gameID: args.gameID,
      username: args.username,
      clanTag: null,
      cosmetics: undefined,
      turnstileToken: null,
      spectator: args.spectator,
      gitCommit,
    };
    ws.send(encodeClientMessage(join, undefined));
    // Match the browser client's cadence (5s). The server closes with
    // `no_heartbeat` after a few missed pings.
    pingTimer = setInterval(() => {
      if (ws.readyState === ws.OPEN) {
        ws.send(encodeClientMessage({ type: "ping" }, zbinCtx ?? undefined));
      }
    }, 5000);
  });

  ws.on("message", async (data: ArrayBuffer | Buffer) => {
    const buf =
      data instanceof ArrayBuffer
        ? new Uint8Array(data)
        : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    let msg: ServerMessage;
    try {
      msg = decodeServerMessage(buf, zbinCtx ?? undefined);
    } catch (e) {
      process.stderr.write(`[bot] decode error on ${buf.byteLength} bytes: ${e}\n`);
      return;
    }

    if (msg.type === "lobby_info") {
      myClientID = msg.myClientID;
      process.stderr.write(`[bot] lobby_info — my clientID=${myClientID}\n`);
      return;
    }

    if (msg.type === "start") {
      // Seed the zbin dictionary for later turn frames.
      zbinCtx = createGameWireContext(msg.gameStartInfo.players);

      if (myClientID === null) {
        // Fall back to the roster — our username should be unique in a dev
        // lobby. In a real game the lobby_info always precedes start.
        myClientID =
          msg.gameStartInfo.players.find((p) => p.username === args.username)
            ?.clientID ?? null;
      }
      if (myClientID === null) {
        process.stderr.write(`[bot] no clientID at start, aborting\n`);
        ws.close();
        return;
      }

      process.stderr.write(
        `[bot] start — players=${msg.gameStartInfo.players.length} map=${msg.gameStartInfo.config.gameMap}\n`,
      );
      sim.current = new LocalSim(myClientID);
      await sim.current.init(msg.gameStartInfo);
      return;
    }

    if (msg.type === "turn") {
      const current = sim.current;
      if (current === null) return; // start not received yet
      current.applyTurn(msg.turn);
      assets.tickCounter++;

      // Spawn-phase picker: active for the whole manual-spawn window, even
      // after we've already placed one pick — humans may move their own
      // spawns during the ~20s window and we want to react. We re-score
      // every 5 ticks (≈500ms) and send a new SpawnIntent only if the best
      // tile moved. Each re-pick is deterministic game-side: SpawnExecution
      // relinquishes the old tiles and reconquers around the new centre
      // (OpenFrontIO/src/core/execution/SpawnExecution.ts:79-97).
      if (
        current.game !== null &&
        current.game.inSpawnPhase() &&
        !current.game.config().isRandomSpawn() &&
        myClientID !== null
      ) {
        const tick = current.game.ticks();
        if (shouldRepick(tick, spawnState.lastPickTick)) {
          const pick = pickBestSpawn(current.game, myClientID);
          if (pick !== null && pick.tile !== spawnState.lastSpawnTile) {
            ws.send(
              encodeClientMessage(
                { type: "intent", intent: { type: "spawn", tile: pick.tile } },
                zbinCtx ?? undefined,
              ),
            );
            process.stderr.write(
              `[bot] spawn-pick tick=${tick} tile=${pick.tile} minDist=${pick.minDist} (sampled=${pick.sampled} rejected=${pick.rejected})\n`,
            );
            spawnState.lastSpawnTile = pick.tile;
          }
          spawnState.lastPickTick = tick;
        }
      }

      // Build encoder/codec once we're spawned — the Encoder needs the Game
      // and a Player, and the Player only appears after the spawn phase.
      if (current.spawned && current.game !== null && current.me !== null) {
        if (assets.encoder === null) {
          const map = current.game;
          const extent = Math.max(map.width(), map.height());
          // Every field here must mirror what env.ts (training) uses so the
          // scalar observation matches. maxSteps had been hardcoded at 1200
          // while recent training used 3000, which made the policy read a
          // time-progression ≈2.5x further along than reality — triggering
          // endgame behaviours (far-boat all-ins, high-ratio attacks, no
          // finishing).
          assets.encoder = new Encoder(current.game, {
            grid: args.grid,
            windowMode: "adaptive",
            windowTiles: extent,
            windowMinTiles: args.windowMinTiles,
            windowMaxTiles: 0,
            windowZoom: args.windowZoom,
            maxPlayers: args.maxPlayers,
            maxSteps: args.maxSteps,
            ticksPerStep: args.ticksPerStep,
          });
          assets.codec = new ActionCodec(
            current.game,
            assets.encoder,
            myClientID!,
            args.grid,
            args.maxPlayers,
          );
          process.stderr.write(`[bot] spawned, encoder+codec ready\n`);
        }

        if (
          assets.tickCounter >= args.ticksPerStep &&
          !assets.pending &&
          py !== null
        ) {
          assets.tickCounter = 0;
          assets.pending = true;
          await sendDecision(ws, current, assets, py, args).catch((e) =>
            process.stderr.write(`[bot] decision error: ${e}\n`),
          );
          assets.pending = false;
        }
      }
      return;
    }

    process.stderr.write(`[bot] ${msg.type}\n`);
  });

  ws.on("close", (code: number, reason: Buffer) => {
    process.stderr.write(
      `[bot] closed code=${code} reason=${reason.toString() || "(none)"}\n`,
    );
    if (pingTimer) clearInterval(pingTimer);
    if (py) py.close();
    process.exit(code === 1000 ? 0 : 1);
  });

  ws.on("error", (err: Error) => {
    process.stderr.write(`[bot] error: ${err.message}\n`);
  });

  process.on("SIGINT", () => {
    process.stderr.write(`[bot] SIGINT — closing\n`);
    ws.close(1000, "sigint");
  });
}

async function sendDecision(
  ws: WebSocket,
  sim: LocalSim,
  assets: {
    encoder: Encoder | null;
    codec: ActionCodec | null;
    step: number;
    prevGoldEarned: number;
  },
  py: PyBridge,
  args: Args,
): Promise<void> {
  const me = sim.me!;
  const encoder = assets.encoder!;
  const codec = assets.codec!;
  // Mirror env.ts:388 — the encoder expects per-step income (Δ goldEarned),
  // not cumulative goldEarned, otherwise the "income rate" scalar saturates
  // and the policy sees a feature it never saw during training.
  const goldEarnedNow = Number(me.goldEarned());
  const goldIncome = Math.max(0, goldEarnedNow - assets.prevGoldEarned);
  assets.prevGoldEarned = goldEarnedNow;
  const enc = encoder.encode(me, assets.step, goldIncome);
  // Critical: hand codec.masks the TAIL of obs.scalar (the can_build_*
  // slots), exactly as env.ts:461-464 does during training. The codec
  // writes 0/1 per unit type INTO that view, so the next policy forward
  // sees the real "can I afford/place each unit" flags on the scalar
  // input. Passing a dummy buffer (previous behaviour) left those scalars
  // at zero — the policy thought it could never build anything and acted
  // accordingly.
  const masks = codec.masks(me, enc, enc.scalar.subarray(CAN_BUILD_OFFSET));

  const G = args.grid;
  const K = args.maxPlayers;
  const obs: Observation = {
    spatial: flatten(enc.spatial),
    spatial_shape: [totalChannels(K), G, G],
    minimap: flatten(enc.minimap),
    minimap_shape: [MC, MINIMAP_SIDE, MINIMAP_SIDE],
    scalar: flatten(enc.scalar),
    players: flatten(enc.players),
    players_shape: [K, PF],
    masks: {
      action_type: flatten(masks.actionType),
      target_slot: flatten(masks.target),
      ratio: flatten(masks.ratio),
      unit_type: flatten(masks.unit),
      cell: flatten(masks.cell),
    },
  };
  // Light sanity check on shapes before we send.
  if (obs.spatial.length !== totalChannels(K) * G * G) {
    throw new Error(
      `spatial length mismatch: got ${obs.spatial.length} vs expected ${totalChannels(K) * G * G}`,
    );
  }

  const action = await py.predict(obs);
  if (action === null) return;

  const decoded = codec.decode(me, enc, [
    action.action_type,
    action.target_slot,
    action.ratio,
    action.unit_type,
    action.cell,
  ]);
  assets.step++;

  if (decoded.intent === null) {
    process.stderr.write(`[bot] step=${assets.step} ${decoded.label} (noop)\n`);
    return;
  }

  // The server stamps clientID on turns it relays; for client→server we send
  // the bare Intent (without the clientID bridge adds for its own executor).
  const { clientID: _stampId, ...intent } = decoded.intent;
  ws.send(
    encodeClientMessage({ type: "intent", intent } as any, undefined),
  );
  process.stderr.write(`[bot] step=${assets.step} sent ${decoded.label}\n`);
}

main().catch((e) => {
  process.stderr.write(`[bot] fatal: ${e}\n`);
  process.exit(1);
});
