/**
 * Local game simulation, synchronized with the server by replaying Turn messages.
 *
 * OpenFront is deterministic: every client runs its own copy of `src/core`, and
 * the server only relays `Turn` messages that bundle each tick's intents. We do
 * the same here — we build a `GameRunner` from the `GameStartInfo` the server
 * hands us, then feed every incoming `Turn` into it. The sim never diverges from
 * the one running in every browser connected to the same game.
 *
 * Reuses `createGameRunner` from `src/core/GameRunner`, which handles terrain
 * loading, nation creation and executor wiring. We only have to pass a
 * `GameMapLoader` that reads from disk — the test harness's `NodeGameMapLoader`
 * does exactly that (and we import it rather than duplicating).
 */

import path from "node:path";
import { fileURLToPath } from "node:url";
import { createGameRunner, GameRunner } from "src/core/GameRunner";
import { Game, Player } from "src/core/game/Game";
import { ClientID, GameStartInfo, Turn } from "src/core/Schemas";
import { NodeGameMapLoader } from "../../../OpenFrontIO/tests/perf/fullgame/NodeGameMapLoader";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// rl/game_bot/src/local_sim.ts → OpenFrontIO/resources/maps.
const DEFAULT_MAPS_DIR = path.resolve(
  __dirname,
  "../../../OpenFrontIO/resources/maps",
);

export class LocalSim {
  private runner: GameRunner | null = null;
  private _game: Game | null = null;
  private _me: Player | null = null;
  private readonly mapsDir: string;

  constructor(
    private readonly clientID: ClientID,
    mapsDir: string = DEFAULT_MAPS_DIR,
  ) {
    this.mapsDir = mapsDir;
  }

  async init(gameStart: GameStartInfo): Promise<void> {
    const loader = new NodeGameMapLoader(this.mapsDir);
    this.runner = await createGameRunner(
      gameStart,
      this.clientID,
      loader,
      (gu) => {
        if ("errMsg" in gu) {
          process.stderr.write(`[sim] game error: ${gu.errMsg}\n`);
          if (gu.stack) {
            // First few frames of the stack are enough to attribute the crash.
            const frames = gu.stack.split("\n").slice(0, 6).join("\n");
            process.stderr.write(`[sim]   stack:\n${frames}\n`);
          }
        }
      },
    );
    this._game = this.runner.game;
    process.stderr.write(
      `[sim] initialized game=${gameStart.gameID} map=${gameStart.config.gameMap} humans=${gameStart.players.length}\n`,
    );
  }

  /**
   * Apply one Turn (from the server) by appending it to the runner's queue
   * and consuming every pending tick. The server sends exactly one Turn per
   * sim tick, so this draws at most one tick, but we loop anyway in case
   * two Turns arrived in the same WebSocket read.
   */
  applyTurn(turn: Turn): void {
    if (this.runner === null) {
      throw new Error("applyTurn before init");
    }
    this.runner.addTurn(turn);
    while (this.runner.executeNextTick()) {
      // one Turn per tick in practice, but drain anyway
    }
    // Resolve `me` lazily — the player only appears in game.players() once
    // the spawn phase has placed them on the map.
    if (this._me === null && this._game !== null) {
      this._me =
        this._game.players().find((p) => p.clientID() === this.clientID) ??
        null;
    }
  }

  get game(): Game | null {
    return this._game;
  }

  get me(): Player | null {
    return this._me;
  }

  /** Convenience: have we received enough turns to be past the spawn phase? */
  get spawned(): boolean {
    return this._me !== null && this._me.isAlive();
  }
}
