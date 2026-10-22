/**
 * Heuristic spawn picker for the live bot.
 *
 * Strategy: maximise the empty area around us, which in practice means
 * "put ourselves as far as possible from every other spawned player (humans,
 * nations and bots)". Rough algorithm per call:
 *
 *   1. Collect spawnTile() of every player that has already spawned.
 *   2. Sample N random land tiles (ignoring our own current spawn — we're
 *      about to overwrite it).
 *   3. Score each by the minimum Manhattan distance to a competitor spawn.
 *   4. Return the sampled tile with the highest score.
 *
 * Called every half-second during the spawn phase so we react when a human
 * moves their own pick: SpawnExecution relinquishes the previous tiles and
 * re-conquers around the new centre, so repeated picks are cheap and legal
 * (OpenFrontIO/src/core/execution/SpawnExecution.ts:79-97).
 */
import { Game, Player } from "src/core/game/Game";
import type { TileRef } from "src/core/game/GameMap";

export interface SpawnPickResult {
  tile: TileRef;
  minDist: number;
  sampled: number;
  rejected: number;
}

/**
 * @param selfID — our own client/player id. We exclude our current spawn from
 *                 the "avoid" set so each re-pick is measured against *others*,
 *                 not our previous choice.
 */
export function pickBestSpawn(
  game: Game,
  selfID: string,
  options: { numSamples?: number; seed?: number } = {},
): SpawnPickResult | null {
  const numSamples = options.numSamples ?? 500;
  // Mix the seed with the current tick so successive picks explore different
  // candidates — otherwise the LCG below gives the same sequence every call.
  const seedBase =
    (options.seed ?? Math.floor(Math.random() * 0xffffffff)) ^ game.ticks();

  const avoid: TileRef[] = [];
  for (const p of game.allPlayers()) {
    if (p.clientID() === selfID) continue;
    const st = p.spawnTile();
    if (st !== undefined) avoid.push(st);
  }

  let s = seedBase >>> 0;
  const rnd = () => {
    // LCG — Numerical Recipes. Not for crypto, fine for sampling.
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 0x100000000;
  };

  const W = game.width();
  const H = game.height();
  let bestTile: TileRef | null = null;
  let bestScore = -Infinity;
  let rejected = 0;

  for (let i = 0; i < numSamples; i++) {
    const x = Math.floor(rnd() * W);
    const y = Math.floor(rnd() * H);
    const tile = game.ref(x, y);

    // getSpawn() in SpawnExecution rejects water, owned and border tiles —
    // mirror those filters so we don't bother the server with a tile it will
    // discard (relinquish still happens, so a bad pick wastes a round-trip).
    if (!game.isLand(tile) || game.hasOwner(tile) || game.isBorder(tile)) {
      rejected++;
      continue;
    }

    let minDist = Infinity;
    if (avoid.length === 0) {
      // Nothing to avoid yet — fall back to picking any valid tile. All
      // candidates tie at +Infinity, so the first one wins (fine).
      minDist = Number.MAX_SAFE_INTEGER;
    } else {
      for (const other of avoid) {
        const d = game.manhattanDist(tile, other);
        if (d < minDist) minDist = d;
      }
    }

    if (minDist > bestScore) {
      bestScore = minDist;
      bestTile = tile;
    }
  }

  if (bestTile === null) return null;
  return {
    tile: bestTile,
    minDist: bestScore === Number.MAX_SAFE_INTEGER ? -1 : bestScore,
    sampled: numSamples,
    rejected,
  };
}

/**
 * Lightweight debounce: should we re-pick given the last pick's tick?
 * ticksPerRepick=5 ≈ 500ms at the default 100ms turn interval.
 */
export function shouldRepick(
  currentTick: number,
  lastPickTick: number | null,
  ticksPerRepick: number = 5,
): boolean {
  if (lastPickTick === null) return true;
  return currentTick - lastPickTick >= ticksPerRepick;
}
