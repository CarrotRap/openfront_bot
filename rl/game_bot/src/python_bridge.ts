/**
 * Spawn `drive.py` and pipe observations in / actions out over stdio.
 *
 * Protocol: one JSON object per newline-terminated line in each direction.
 *
 *   obs  → {"spatial":[...], "spatial_shape":[C,G,G],
 *           "minimap":[...], "minimap_shape":[Mc,M,M],
 *           "scalar":[...],
 *           "players":[...], "players_shape":[K,PF],
 *           "masks":{"action_type":[...], "target_slot":[...], ...}}
 *   act  ← {"action_type":n, "target_slot":n, "ratio":n, "unit_type":n, "cell":n}
 *
 * All arrays are flat; shapes travel alongside so the Python side can
 * reshape without baked-in constants. This mirrors the memory layout of the
 * Float32Arrays the bridge Encoder produces, so passing them through is
 * a .toArray() away.
 */
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";

export interface Observation {
  spatial: number[];
  spatial_shape: [number, number, number]; // (C, G, G)
  minimap: number[];
  minimap_shape: [number, number, number]; // (Mc, M, M)
  scalar: number[]; // (S,)
  players: number[];
  players_shape: [number, number]; // (K, PF)
  masks: {
    action_type: number[];
    target_slot: number[];
    ratio: number[];
    unit_type: number[];
    cell: number[];
  };
}

export interface Action {
  action_type: number;
  target_slot: number;
  ratio: number;
  unit_type: number;
  cell: number;
}

export class PyBridge {
  private readonly proc: ChildProcessWithoutNullStreams;
  private buffer = "";
  private pending: Array<(line: string) => void> = [];
  private closed = false;

  constructor(cmd: string, args: string[]) {
    this.proc = spawn(cmd, args, { stdio: ["pipe", "pipe", "pipe"] });
    this.proc.stdout.setEncoding("utf8");
    this.proc.stdout.on("data", (chunk: string) => this.onStdout(chunk));
    this.proc.stderr.setEncoding("utf8");
    this.proc.stderr.on("data", (chunk: string) =>
      process.stderr.write(`[py] ${chunk}`),
    );
    this.proc.on("close", (code) => {
      this.closed = true;
      process.stderr.write(`[py] child exited with code ${code}\n`);
      for (const resolve of this.pending) resolve("");
      this.pending = [];
    });
  }

  private onStdout(chunk: string) {
    this.buffer += chunk;
    for (;;) {
      const idx = this.buffer.indexOf("\n");
      if (idx === -1) break;
      const line = this.buffer.slice(0, idx);
      this.buffer = this.buffer.slice(idx + 1);
      const waiter = this.pending.shift();
      if (waiter) waiter(line);
    }
  }

  async predict(obs: Observation): Promise<Action | null> {
    if (this.closed) return null;
    this.proc.stdin.write(JSON.stringify(obs) + "\n");
    const line = await new Promise<string>((resolve) =>
      this.pending.push(resolve),
    );
    if (!line) return null;
    const parsed = JSON.parse(line);
    if (parsed.error) return null;
    return parsed as Action;
  }

  close() {
    if (!this.closed) {
      this.proc.stdin.end();
      this.proc.kill();
    }
  }
}

/** Helper: turn a Float32Array of shape dims[] into a flat `number[]`. */
export function flatten(arr: Float32Array | Uint8Array): number[] {
  return Array.from(arr);
}
