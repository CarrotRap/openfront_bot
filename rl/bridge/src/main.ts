/**
 * Bridge entry point.
 *
 * Owns one `OpenFrontEnv` and serves `init` / `spec` / `reset` / `step` /
 * `close` commands over the stdio protocol. One process = one environment,
 * so Python can run several in parallel with a vectorised wrapper.
 */
import path from "path";
import { fileURLToPath } from "url";
import { DEFAULT_CONFIG, EnvConfig, OpenFrontEnv, StepResult } from "./env";
import {
  Frame,
  packArrays,
  readCommands,
  redirectConsoleToStderr,
  sendFrame,
} from "./protocol";
import {
  ACTION_TYPES,
  BUILD_UNITS,
  MC,
  MINIMAP_CHANNELS,
  MINIMAP_SIDE,
  PF,
  PLAYER_FEATURES,
  RATIOS,
  S,
  SCALAR_NAMES,
  channelsFor,
  totalChannels,
} from "./spec";

redirectConsoleToStderr();

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_REPO_ROOT = path.resolve(HERE, "../../../OpenFrontIO");

let env: OpenFrontEnv | null = null;
let cfg: EnvConfig = DEFAULT_CONFIG;
let repoRoot = DEFAULT_REPO_ROOT;

function specFrame(): Frame {
  const G = cfg.grid;
  const C = totalChannels(cfg.maxPlayers);
  const channels = channelsFor(cfg.maxPlayers);
  return {
    ok: true,
    spec: {
      observation: {
        spatial: { shape: [C, G, G], dtype: "float32", channels },
        minimap: {
          shape: [MC, MINIMAP_SIDE, MINIMAP_SIDE],
          dtype: "float32",
          channels: MINIMAP_CHANNELS,
        },
        scalar: { shape: [S], dtype: "float32", names: SCALAR_NAMES },
        players: {
          shape: [cfg.maxPlayers, PF],
          dtype: "float32",
          features: PLAYER_FEATURES,
        },
        mask_action_type: { shape: [ACTION_TYPES.length], dtype: "uint8" },
        mask_target: { shape: [cfg.maxPlayers + 1], dtype: "uint8" },
        mask_ratio: { shape: [RATIOS.length], dtype: "uint8" },
        mask_unit: { shape: [BUILD_UNITS.length], dtype: "uint8" },
        mask_cell: { shape: [G * G], dtype: "uint8" },
      },
      action: {
        nvec: [
          ACTION_TYPES.length,
          cfg.maxPlayers + 1,
          RATIOS.length,
          BUILD_UNITS.length,
          G * G,
        ],
        heads: ["action_type", "target_slot", "ratio", "unit_type", "cell"],
        action_types: ACTION_TYPES,
        ratios: RATIOS,
        build_units: BUILD_UNITS,
        grid: G,
      },
      config: cfg,
    },
  };
}

function sendStep(result: StepResult): void {
  const { obs, masks, render } = result;
  const G = cfg.grid;
  const C = totalChannels(cfg.maxPlayers);
  const arrays = [
    { name: "spatial", array: obs.spatial, shape: [C, G, G] },
    {
      name: "minimap",
      array: obs.minimap,
      shape: [MC, MINIMAP_SIDE, MINIMAP_SIDE],
    },
    { name: "scalar", array: obs.scalar, shape: [S] },
    { name: "players", array: obs.players, shape: [cfg.maxPlayers, PF] },
    {
      name: "mask_action_type",
      array: masks.actionType,
      shape: [ACTION_TYPES.length],
    },
    { name: "mask_target", array: masks.target, shape: [cfg.maxPlayers + 1] },
    { name: "mask_ratio", array: masks.ratio, shape: [RATIOS.length] },
    { name: "mask_unit", array: masks.unit, shape: [BUILD_UNITS.length] },
    { name: "mask_cell", array: masks.cell, shape: [G * G] },
    { name: "mask_cell_build", array: masks.cellBuild, shape: [G * G] },
    { name: "mask_cell_boat", array: masks.cellBoat, shape: [G * G] },
  ];
  if (render !== null) {
    arrays.push({
      name: "render_pixels",
      array: render.pixels,
      shape: [render.height, render.width],
    });
  }
  const { descriptors, payload } = packArrays(arrays);
  sendFrame(
    {
      ok: true,
      reward: result.reward,
      terminated: result.terminated,
      truncated: result.truncated,
      info: result.info,
      // The pixels ride in the binary payload; everything else is small.
      render:
        render === null
          ? null
          : {
              width: render.width,
              height: render.height,
              step: render.step,
              palette: render.palette,
              window: render.window,
              structures: render.structures,
              players: render.players,
            },
      arrays: descriptors,
    },
    payload,
  );
}

function sendError(e: unknown): void {
  const err = e as Error;
  sendFrame(
    { ok: false, error: err?.message ?? String(e), stack: err?.stack },
    Buffer.alloc(0),
  );
}

/** Deep-merges the caller's partial config onto the defaults. */
function mergeConfig(partial: Partial<EnvConfig> | undefined): EnvConfig {
  const merged = { ...DEFAULT_CONFIG, ...(partial ?? {}) };
  merged.rewardWeights = {
    ...DEFAULT_CONFIG.rewardWeights,
    ...(partial?.rewardWeights ?? {}),
  };
  return merged;
}

readCommands(async (cmd) => {
  try {
    switch (cmd.cmd) {
      case "init": {
        cfg = mergeConfig(cmd.config as Partial<EnvConfig> | undefined);
        if (typeof cmd.repoRoot === "string") repoRoot = cmd.repoRoot;
        env = new OpenFrontEnv(cfg, repoRoot);
        sendFrame(specFrame(), Buffer.alloc(0));
        return;
      }
      case "spec":
        sendFrame(specFrame(), Buffer.alloc(0));
        return;
      case "reset": {
        if (env === null) throw new Error("init must be called before reset");
        const seed = String(cmd.seed ?? "seed-0");
        sendStep(await env.reset(seed));
        return;
      }
      case "step": {
        if (env === null) throw new Error("init must be called before step");
        const action = (cmd.action as number[]) ?? [];
        sendStep(env.step(action));
        return;
      }
      case "close":
        sendFrame({ ok: true, closed: true }, Buffer.alloc(0));
        process.exit(0);
        return;
      default:
        throw new Error(`unknown command: ${String(cmd.cmd)}`);
    }
  } catch (e) {
    sendError(e);
  }
});

// Keep the process alive while stdin is open.
process.stdin.resume();
