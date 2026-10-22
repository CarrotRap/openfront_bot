/**
 * Headless OpenFront environment: one agent, N scripted opponents.
 *
 * Wraps the real `GameRunner` + `Executor` pipeline, so the agent plays
 * against exactly the simulation the game ships — nations, tribes, economy,
 * combat and all. The agent is a `PlayerType.Human` whose intents are stamped
 * with a fixed clientID and injected into the turn queue.
 */
import fs from "fs";
import path from "path";
import { Config } from "../../../OpenFrontIO/src/core/configuration/Config";
import { Executor } from "../../../OpenFrontIO/src/core/execution/ExecutionManager";
import {
  Difficulty,
  Game,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  Player,
  PlayerInfo,
  PlayerType,
} from "../../../OpenFrontIO/src/core/game/Game";
import { createGame } from "../../../OpenFrontIO/src/core/game/GameImpl";
import { createNationsForGame } from "../../../OpenFrontIO/src/core/game/NationCreation";
import {
  genTerrainFromBin,
  MapManifest,
} from "../../../OpenFrontIO/src/core/game/TerrainMapLoader";
import { GameRunner } from "../../../OpenFrontIO/src/core/GameRunner";
import { PseudoRandom } from "../../../OpenFrontIO/src/core/PseudoRandom";
import {
  GameConfig,
  GameStartInfo,
  StampedIntent,
} from "../../../OpenFrontIO/src/core/Schemas";
import { simpleHash } from "../../../OpenFrontIO/src/core/Util";
import { ActionCodec, Masks } from "./actions";
import { CAN_BUILD_OFFSET, EncodedObservation, Encoder } from "./observation";
import { Renderer, RenderFrame } from "./render";
import { ACTION_TYPES, BUILD_UNITS, RATIOS } from "./spec";

const AGENT_CLIENT_ID = "AGENTRL0";
const AGENT_PLAYER_ID = "agentrl0";
const MAX_SPAWN_TURNS = 400;

export interface RewardWeights {
  /** Multiplier on positive Δtiles / tilesRef — i.e. tiles GAINED. */
  tilesGain: number;
  /** Multiplier on negative Δtiles / tilesRef — i.e. tiles LOST. */
  tilesLoss: number;
  troops: number;
  /** Weight on log(1+income) per step — rewards *earning*, not hoarding. */
  gold: number;
  invalid: number;
  death: number;
  win: number;
  step: number;
  /** Reward per opponent eliminated during this step. */
  elim: number;
  /** v4 — progress toward the actual win condition (80% of land), capped at 1.
   *  Replaces v3's linear `hold`: `hold` created explosive variance (dominant
   *  episodes gave 20x more reward than mid-size ones, destabilising gradient
   *  updates). `progress` plateaus at the win threshold so the policy has
   *  nothing to gain from runaway domination — only from reaching the bar
   *  faster and holding it longer. Reward is also PER-STEP: being at 80% for
   *  more steps strictly beats peaking there briefly. */
  progress: number;
}

export interface EnvConfig {
  /** Map directory name, e.g. "big_plains" (test map) or "oceania". */
  map: string;
  /** Where to look for the map: bundled test fixtures or production maps. */
  mapSource: "testdata" | "resources";
  mapSize: "Normal" | "Compact";
  bots: number;
  nations: "default" | "disabled" | number;
  difficulty: "Easy" | "Medium" | "Hard" | "Impossible";
  grid: number;
  /**
   * Observation window, in game tiles.
   *   "adaptive" — zooms out in octaves as the agent's territory grows
   *   "auto"     — pinned to the whole map
   *   a number   — pinned to that many tiles
   */
  windowTiles: number | "auto" | "adaptive";
  /** Adaptive mode: smallest window, i.e. the closest zoom. */
  windowMinTiles: number;
  /** Adaptive mode: largest window; 0 clamps to the map's longest side. */
  windowMaxTiles: number;
  /** Adaptive mode: window side as a multiple of the agent's linear extent. */
  windowZoom: number;
  maxPlayers: number;
  ticksPerStep: number;
  maxSteps: number;
  instantBuild: boolean;
  infiniteGold: boolean;
  /**
   * How the territory reward is normalised.
   *   "fair_share" — by (land / players), so "owning your fair share" is worth
   *                  the same on every map and at every player count
   *   "map_share"  — by total land, which makes the signal ~16x weaker on a
   *                  large map than on a small one
   */
  tilesRewardMode: "fair_share" | "map_share";
  /** Attach a full-map picture to every step, for the live web viewer. */
  emitRender: boolean;
  /** Longest side of that picture, in pixels. */
  renderMaxDim: number;
  rewardWeights: RewardWeights;
}

export const DEFAULT_CONFIG: EnvConfig = {
  map: "big_plains",
  mapSource: "testdata",
  mapSize: "Normal",
  bots: 6,
  nations: "disabled",
  difficulty: "Medium",
  grid: 64,
  windowTiles: "adaptive",
  windowMinTiles: 64,
  windowMaxTiles: 0,
  windowZoom: 2.5,
  maxPlayers: 8,
  ticksPerStep: 10,
  maxSteps: 1200,
  instantBuild: false,
  infiniteGold: false,
  tilesRewardMode: "fair_share",
  emitRender: false,
  renderMaxDim: 256,
  rewardWeights: {
    // v4-neutral: dense signals kept small so they don't dictate STRATEGY
    // (how to grow), only enough to keep the gradient informative. The
    // objective-aligned terms (progress, elim, win, death) carry the weight.
    tilesGain: 0.5,
    tilesLoss: 0.5,
    troops: 0.05,
    gold: 0.02,
    invalid: -0.01,
    // Terminal signals bumped so they dominate the dense signals integrated
    // over an episode. death=-15 vs the policy's typical per-episode reward
    // (~200-500 pre-death) is now significant; win=+100 pays off clearly.
    death: -15.0,
    win: 100.0,
    // Zeroed: `progress` already encodes "time passing without growth = bad".
    step: 0,
    // Big elim bonus: v4-neutral stays silent about HOW to win, but a KO is
    // objectively a win-condition acceleration (one fewer competitor for the
    // 80% threshold), so it's rewarded directly. 15 per KO = 300 for wiping
    // 20 opponents, same order of magnitude as the win bonus itself.
    elim: 15.0,
    // Progress toward the actual win condition. Each step, +progress * r
    // where r = min(1, tiles / tilesToWin). tilesToWin = 0.80 * land (see
    // src/core/configuration/Config.ts: PERCENT_TILES_OWNED_TO_WIN = 80).
    // Replaces the v3 `hold` term. Caps naturally, so there's no runaway on
    // large-empire episodes — the policy's incentive is to reach the bar
    // and hold it, not to balloon beyond.
    progress: 2.0,
  },
};

export interface StepResult {
  obs: EncodedObservation;
  masks: Masks;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  info: Record<string, unknown>;
  render: RenderFrame | null;
}

interface Terrain {
  gameMap: Awaited<ReturnType<typeof genTerrainFromBin>>;
  miniGameMap: Awaited<ReturnType<typeof genTerrainFromBin>>;
  manifest: MapManifest;
}

/** Raw map bytes, cached across episodes; the GameMap itself is not. */
interface MapBins {
  manifest: MapManifest;
  mapBin: Uint8Array;
  miniBin: Uint8Array;
}

export class OpenFrontEnv {
  private game!: Game;
  private runner!: GameRunner;
  private encoder!: Encoder;
  private codec!: ActionCodec;
  private me!: Player;
  /** Territory that one step of the reward is measured against. */
  private tilesReference = 1;
  /** Tiles the agent must own to win (0.80 × land on World). Cached per
   *  episode because `land` is fixed once the terrain is loaded. */
  private tilesToWin = 1;
  private renderer: Renderer | null = null;
  private bins: MapBins | null = null;

  private turnNumber = 0;
  private step_ = 0;
  private prevTiles = 0;
  private prevTroops = 0;
  private prevGold = 0;
  private prevGoldEarned = 0;
  /** Alive opponents at the end of the previous step (excludes self). */
  private prevAliveOpponents = 0;
  private done = false;
  private fatal: string | null = null;

  constructor(
    private readonly cfg: EnvConfig,
    private readonly repoRoot: string,
  ) {}

  /** Number of grid cells; the size of the positional action head. */
  get cells(): number {
    return this.cfg.grid * this.cfg.grid;
  }

  get actionNvec(): number[] {
    return [
      ACTION_TYPES.length,
      this.cfg.maxPlayers + 1,
      RATIOS.length,
      BUILD_UNITS.length,
      this.cells,
    ];
  }

  async reset(seed: string): Promise<StepResult> {
    const terrain = await this.loadTerrain();
    const gameConfig: GameConfig = {
      gameMap: GameMapType.World, // only a label here; terrain comes from disk
      gameMapSize:
        this.cfg.mapSize === "Compact" ? GameMapSize.Compact : GameMapSize.Normal,
      gameMode: GameMode.FFA,
      gameType: GameType.Singleplayer,
      difficulty: Difficulty[this.cfg.difficulty],
      nations: this.cfg.nations,
      donateGold: false,
      donateTroops: false,
      bots: this.cfg.bots,
      infiniteGold: this.cfg.infiniteGold,
      infiniteTroops: false,
      instantBuild: this.cfg.instantBuild,
      randomSpawn: true,
    };
    const config = new Config(gameConfig, null, false);

    const gameStart: GameStartInfo = {
      gameID: seed,
      lobbyCreatedAt: 0,
      config: gameConfig,
      players: [],
    };
    const agent = new PlayerInfo(
      "AGENT",
      PlayerType.Human,
      AGENT_CLIENT_ID,
      AGENT_PLAYER_ID,
    );
    const nations = config.spawnNations()
      ? createNationsForGame(
          gameStart,
          terrain.manifest.nations ?? [],
          terrain.manifest.additionalNations ?? [],
          1,
          new PseudoRandom(simpleHash(seed)),
        )
      : [];

    this.game = createGame(
      [agent],
      nations,
      terrain.gameMap,
      terrain.miniGameMap,
      config,
      terrain.manifest.teamGameSpawnAreas,
    );
    this.fatal = null;
    this.runner = new GameRunner(
      this.game,
      new Executor(this.game, seed, AGENT_CLIENT_ID),
      (gu) => {
        if ("errMsg" in gu) this.fatal = gu.errMsg;
      },
    );
    this.runner.init();

    // Run the spawn phase. randomSpawn places the agent automatically, so the
    // policy only ever sees the post-spawn game.
    this.turnNumber = 0;
    while (this.game.inSpawnPhase() && this.turnNumber < MAX_SPAWN_TURNS) {
      this.tickOnce([]);
    }
    if (this.game.inSpawnPhase()) this.game.endSpawnPhase();
    // Let the world settle for one step so every tribe is on the map.
    for (let i = 0; i < this.cfg.ticksPerStep; i++) this.tickOnce([]);

    const me = this.game.players().find((p) => p.clientID() === AGENT_CLIENT_ID);
    if (me === undefined) throw new Error("agent player was never spawned");
    this.me = me;

    const mapExtent = Math.max(this.game.width(), this.game.height());
    const adaptive = this.cfg.windowTiles === "adaptive";
    this.encoder = new Encoder(this.game, {
      grid: this.cfg.grid,
      windowMode: adaptive ? "adaptive" : "fixed",
      windowTiles:
        this.cfg.windowTiles === "auto"
          ? mapExtent
          : adaptive
            ? mapExtent
            : (this.cfg.windowTiles as number),
      windowMinTiles: this.cfg.windowMinTiles,
      windowMaxTiles: this.cfg.windowMaxTiles,
      windowZoom: this.cfg.windowZoom,
      maxPlayers: this.cfg.maxPlayers,
      maxSteps: this.cfg.maxSteps,
      ticksPerStep: this.cfg.ticksPerStep,
    });
    this.codec = new ActionCodec(
      this.game,
      this.encoder,
      AGENT_CLIENT_ID,
      this.cfg.grid,
      this.cfg.maxPlayers,
    );
    this.renderer = this.cfg.emitRender
      ? new Renderer(this.game, this.cfg.renderMaxDim)
      : null;

    // Fixed for the episode: players die off, and a reference that drifted
    // would stop the reward telescoping.
    const land = Math.max(1, this.game.numLandTiles());
    this.tilesReference =
      this.cfg.tilesRewardMode === "fair_share"
        ? Math.max(1, land / Math.max(1, this.game.players().length)) / 10
        : land / 100;
    // 80% of land is the win threshold (PERCENT_TILES_OWNED_TO_WIN=80 in
    // OpenFront's Config.ts). Overtime can lower it, but the progress
    // reward caps at 1 so the sim-side threshold drop doesn't change the
    // signal shape — just makes the win easier to clinch at the terminal.
    this.tilesToWin = Math.max(1, Math.floor(land * 0.8));

    this.step_ = 0;
    this.done = false;
    this.prevTiles = me.numTilesOwned();
    this.prevTroops = me.troops();
    this.prevGold = Number(me.gold());
    this.prevGoldEarned = Number(me.goldEarned());
    // game.players() returns alive-only; subtract self so we only count
    // opponents and never credit an elim for our own death (that's w.death).
    this.prevAliveOpponents = Math.max(0, this.game.players().length - 1);

    return this.observe(0, false, false, { reset: true });
  }

  step(action: number[]): StepResult {
    if (this.done) throw new Error("step() after episode end; call reset()");
    const obs = this.currentObs();
    const decoded = this.codec.decode(this.me, obs, action);
    const intents: StampedIntent[] = decoded.intent ? [decoded.intent] : [];

    for (let i = 0; i < this.cfg.ticksPerStep; i++) {
      this.tickOnce(i === 0 ? intents : []);
      if (this.fatal !== null) break;
      if (!this.me.isAlive()) break;
    }
    this.step_++;

    const w = this.cfg.rewardWeights;
    const alive = this.me.isAlive() && this.fatal === null;
    const winner = this.game.getWinner();
    const won = winner !== null && winner === this.me;

    const tiles = alive ? this.me.numTilesOwned() : 0;
    const troops = alive ? this.me.troops() : 0;
    const gold = alive ? Number(this.me.gold()) : 0;
    const goldEarned = alive ? Number(this.me.goldEarned()) : this.prevGoldEarned;

    // v2: income (gold earned this step), not Δbalance — spending no longer
    // costs reward. Clamped at 0 in the pathological case where goldEarned
    // somehow decreased (shouldn't, it's cumulative).
    const income = Math.max(0, goldEarned - this.prevGoldEarned);

    // v2: opponents eliminated this step. game.players() is alive-only, and
    // we exclude self so our own death stays attributed to w.death.
    const aliveOpponentsNow = alive
      ? Math.max(0, this.game.players().length - 1)
      : this.prevAliveOpponents;
    const eliminated = Math.max(0, this.prevAliveOpponents - aliveOpponentsNow);

    // v4-neutral: symmetric gain/loss, small weight — just a dense gradient
    // signal, not a loss-aversion mechanism (that was v2's design and led
    // to passivity). The strategic shape is carried by `progress` + `elim`.
    const dtiles = (tiles - this.prevTiles) / this.tilesReference;
    const tileReward =
      dtiles >= 0 ? w.tilesGain * dtiles : w.tilesLoss * dtiles;
    // v4 progress reward: capped at the win threshold so there's no runaway
    // and no incentive to balloon past 80% land. The ratio is clamped at 1
    // because "being 90% isn't 12% better than 80% — you've already won".
    const progressReward = alive
      ? w.progress * Math.min(1, tiles / this.tilesToWin)
      : 0;

    let reward =
      tileReward +
      progressReward +
      w.troops * (Math.log1p(troops) - Math.log1p(this.prevTroops)) +
      w.gold * Math.log1p(income) +
      w.elim * eliminated +
      w.step;
    if (decoded.invalid) reward += w.invalid;

    const terminated = !alive || winner !== null;
    if (!alive) reward += w.death;
    if (won) reward += w.win;
    const truncated = !terminated && this.step_ >= this.cfg.maxSteps;

    this.prevTiles = tiles;
    this.prevTroops = troops;
    this.prevGold = gold;
    this.prevAliveOpponents = aliveOpponentsNow;
    this.prevGoldEarned = goldEarned;
    this.done = terminated || truncated;

    return this.observe(reward, terminated, truncated, {
      action_label: decoded.label,
      invalid: decoded.invalid,
      tiles,
      troops: Math.round(troops),
      gold,
      income,
      won,
      alive,
      tick: this.game.ticks(),
      players_alive: this.game.players().filter((p) => p.isAlive()).length,
      error: this.fatal,
    });
  }

  private cachedObs: EncodedObservation | null = null;

  private currentObs(): EncodedObservation {
    if (this.cachedObs === null) throw new Error("no observation encoded yet");
    return this.cachedObs;
  }

  private observe(
    reward: number,
    terminated: boolean,
    truncated: boolean,
    info: Record<string, unknown>,
  ): StepResult {
    const income = Number(info.income ?? 0);
    const obs = this.encoder.encode(this.me, this.step_, income);
    const masks = this.codec.masks(
      this.me,
      obs,
      obs.scalar.subarray(CAN_BUILD_OFFSET),
    );
    this.cachedObs = obs;
    const render = this.renderer?.frame(this.me, obs) ?? null;
    return { obs, masks, reward, terminated, truncated, info, render };
  }

  private tickOnce(intents: StampedIntent[]): void {
    this.runner.addTurn({ turnNumber: this.turnNumber++, intents });
    this.runner.executeNextTick();
  }

  /**
   * Builds a fresh GameMap for the episode. The bytes are cached but the
   * GameMap is not: the simulation writes tile ownership (and nuked water)
   * straight into it, so reusing one would leak the previous episode.
   */
  private async loadTerrain(): Promise<Terrain> {
    const bins = this.loadBins();
    const compact = this.cfg.mapSize === "Compact";
    const gameMap = await genTerrainFromBin(
      compact ? bins.manifest.map4x : bins.manifest.map,
      bins.mapBin,
    );
    const miniGameMap = await genTerrainFromBin(
      compact ? bins.manifest.map16x : bins.manifest.map4x,
      bins.miniBin,
    );
    return { gameMap, miniGameMap, manifest: bins.manifest };
  }

  private loadBins(): MapBins {
    if (this.bins !== null) return this.bins;
    const base =
      this.cfg.mapSource === "testdata"
        ? path.join(this.repoRoot, "tests/testdata/maps")
        : path.join(this.repoRoot, "resources/maps");
    // Resolve case-insensitively: macOS filesystems match any casing but
    // Linux does not, so a config written on one OS can break on another.
    const want = this.cfg.map.toLowerCase();
    const entries = fs.readdirSync(base);
    const match = entries.find((e) => e.toLowerCase() === want);
    if (!match) {
      throw new Error(
        `map "${this.cfg.map}" not found in ${base} (available: ` +
          `${entries.slice(0, 20).join(", ")}...)`,
      );
    }
    const dir = path.join(base, match);
    const manifest = JSON.parse(
      fs.readFileSync(path.join(dir, "manifest.json"), "utf8"),
    ) as MapManifest;
    const read = (f: string) =>
      new Uint8Array(fs.readFileSync(path.join(dir, f)));
    const compact = this.cfg.mapSize === "Compact";
    this.bins = {
      manifest,
      mapBin: read(compact ? "map4x.bin" : "map.bin"),
      miniBin: read(compact ? "map16x.bin" : "map4x.bin"),
    };
    return this.bins;
  }
}
