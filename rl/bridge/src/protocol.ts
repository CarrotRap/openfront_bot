/**
 * Wire format between the Node simulation and the Python environment.
 *
 * Commands travel up as newline-delimited JSON on stdin. Responses travel
 * down on stdout as binary frames, so observation tensors never pay for JSON
 * encoding:
 *
 *   [uint32 jsonLen LE][uint32 binLen LE][json utf8][binary payload]
 *
 * The JSON header describes each array's dtype, shape and byte offset into the
 * payload, so the Python side rebuilds them with a zero-copy `np.frombuffer`.
 */
import fs from "fs";

export interface ArrayDescriptor {
  name: string;
  dtype: "float32" | "uint8" | "int32";
  shape: number[];
  offset: number;
  nbytes: number;
}

export type Frame = Record<string, unknown> & { arrays?: ArrayDescriptor[] };

/**
 * The game logs to console on its own; anything it writes must not land in
 * the binary stream. Call this before touching stdout.
 */
export function redirectConsoleToStderr(): void {
  const toStderr =
    (prefix: string) =>
    (...args: unknown[]): void => {
      try {
        process.stderr.write(`${prefix} ${args.map(String).join(" ")}\n`);
      } catch {
        /* stderr closed; nothing useful left to do */
      }
    };
  console.log = toStderr("[node]");
  console.info = toStderr("[node]");
  console.debug = () => {};
  console.warn = toStderr("[warn]");
  console.error = toStderr("[err]");
}

/** Writes one frame to stdout, looping until the OS has taken every byte. */
export function sendFrame(header: Frame, payload: Buffer): void {
  const json = Buffer.from(JSON.stringify(header), "utf8");
  const head = Buffer.allocUnsafe(8);
  head.writeUInt32LE(json.length, 0);
  head.writeUInt32LE(payload.length, 4);
  writeAll(Buffer.concat([head, json, payload]));
}

function writeAll(buf: Buffer): void {
  let written = 0;
  while (written < buf.length) {
    try {
      written += fs.writeSync(1, buf, written, buf.length - written);
    } catch (e) {
      const err = e as NodeJS.ErrnoException;
      if (err.code === "EAGAIN") continue; // pipe full, retry
      throw e;
    }
  }
}

/** Packs typed arrays into one payload and returns their descriptors. */
export function packArrays(
  entries: { name: string; array: Float32Array | Uint8Array; shape: number[] }[],
): { descriptors: ArrayDescriptor[]; payload: Buffer } {
  const descriptors: ArrayDescriptor[] = [];
  const chunks: Buffer[] = [];
  let offset = 0;
  for (const { name, array, shape } of entries) {
    const buf = Buffer.from(array.buffer, array.byteOffset, array.byteLength);
    descriptors.push({
      name,
      dtype: array instanceof Float32Array ? "float32" : "uint8",
      shape,
      offset,
      nbytes: buf.length,
    });
    chunks.push(buf);
    offset += buf.length;
  }
  return { descriptors, payload: Buffer.concat(chunks) };
}

/** Reads newline-delimited JSON commands from stdin, one callback per line. */
export function readCommands(
  onCommand: (cmd: Record<string, unknown>) => Promise<void>,
): void {
  let buffer = "";
  let queue: Promise<void> = Promise.resolve();
  process.stdin.setEncoding("utf8");
  process.stdin.on("data", (chunk: string) => {
    buffer += chunk;
    let nl: number;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (line.length === 0) continue;
      // Serialise: a step must finish before the next command is handled.
      queue = queue.then(() => onCommand(JSON.parse(line)));
    }
  });
  process.stdin.on("end", () => {
    queue.then(() => process.exit(0));
  });
}
