/**
 * Observation encoder.
 *
 * Produces four tensors per step, plus the action masks:
 *
 *   spatial  float32 (C, G, G)   ego-centric view of the map (C includes
 *                                 K per-slot channels at the end)
 *   minimap  float32 (MC, M, M)  low-res global view at a fixed resolution,
 *                                 so the policy can see what is happening
 *                                 outside its zoom window
 *   scalar   float32 (S,)        global state of the agent
 *   players  float32 (K, PF)     the K most relevant opponents
 *
 * The spatial view is a square window of `windowTiles` game tiles centred on
 * the agent's territory, nearest-neighbour sampled down to G x G cells. Units
 * are painted on top of the sampled terrain so a single structure is never
 * lost to the downsampling.
 */
import {
  Attack,
  Game,
  Player,
  PlayerType,
  Relation,
  UnitType,
} from "../../../OpenFrontIO/src/core/game/Game";
import { TileRef } from "../../../OpenFrontIO/src/core/game/GameMap";
import {
  BUILD_UNITS,
  CHANNELS,
  C_STATIC,
  MC,
  MINIMAP_CHANNELS,
  MINIMAP_SIDE,
  PF,
  S,
  STRUCTURES_ANTIAIR,
  STRUCTURES_ECON,
  STRUCTURES_MIL,
  STRUCTURE_TYPES,
  totalChannels,
} from "./spec";

const CH: Record<(typeof CHANNELS)[number], number> = Object.fromEntries(
  CHANNELS.map((n, i) => [n, i]),
) as Record<(typeof CHANNELS)[number], number>;

const MCH: Record<(typeof MINIMAP_CHANNELS)[number], number> =
  Object.fromEntries(MINIMAP_CHANNELS.map((n, i) => [n, i])) as Record<
    (typeof MINIMAP_CHANNELS)[number],
    number
  >;

const ECON_SET = new Set<UnitType>(STRUCTURES_ECON);
const MIL_SET = new Set<UnitType>(STRUCTURES_MIL);
const ANTIAIR_SET = new Set<UnitType>(STRUCTURES_ANTIAIR);
const STRUCTURE_SET = new Set<UnitType>(STRUCTURE_TYPES);

/** Where the per-slot `slot_i` channel starts in the spatial tensor. */
const SLOT_CHANNEL_BASE = C_STATIC;

export interface ViewGeometry {
  /** Top-left corner of the sampled window, in game-tile coordinates. */
  originX: number;
  originY: number;
  /** Size of one grid cell, in game tiles. */
  cellSize: number;
  grid: number;
  /** Rung of the zoom ladder this window sits on, and how many rungs exist. */
  zoomLevel: number;
  zoomLevels: number;
}

export interface EncoderOptions {
  grid: number;
  /** "fixed" pins the window at windowTiles; "adaptive" zooms with the agent. */
  windowMode: "fixed" | "adaptive";
  windowTiles: number;
  windowMinTiles: number;
  /** 0 = clamp to the map's longest side. */
  windowMaxTiles: number;
  /** Window side as a multiple of the agent's own linear extent. */
  windowZoom: number;
  maxPlayers: number;
  maxSteps: number;
  ticksPerStep: number;
}

/**
 * Hysteresis for the zoom ladder: step out as soon as the agent outgrows the
 * current level, but step back in only once it fits comfortably (85 %) inside
 * the next smaller one. Without the gap the window would flap between two
 * levels on the boundary, remapping every cell each step.
 */
const SHRINK_MARGIN = 0.85;

export interface EncodedObservation {
  spatial: Float32Array;
  minimap: Float32Array;
  scalar: Float32Array;
  players: Float32Array;
  /** Opponents occupying slots 1..K of the target head (slot 0 = TerraNullius). */
  slotPlayers: (Player | null)[];
  geometry: ViewGeometry;
  /** Owner ids of everything the agent's border touches, for the attack mask. */
  landNeighbors: Set<number>;
  /** Count of my border tiles whose neighbour belongs to each opponent id. */
  borderAdjCount: Map<number, number>;
}

export class Encoder {
  /** Discrete window sizes, each twice the previous. */
  private readonly ladder: number[];
  private level = 0;
  /** Full spatial channel count: static channels + maxPlayers slot channels. */
  private readonly C: number;

  constructor(
    private readonly game: Game,
    private readonly opts: EncoderOptions,
  ) {
    this.ladder = buildLadder(game, opts);
    this.C = totalChannels(opts.maxPlayers);
  }

  /** Current window side, in map tiles. */
  windowTiles(): number {
    return this.ladder[this.level];
  }

  /** Full spatial channel count the encoder writes. */
  channelCount(): number {
    return this.C;
  }

  /**
   * Picks the zoom level for this step. The window tracks the agent's linear
   * extent — the larger of its bounding box and the side of a square of equal
   * area, so a long thin empire is covered as well as a compact one.
   *
   * Levels are discrete octaves rather than a continuous scale: a window that
   * drifted by a few tiles every step would map the same ground to a different
   * grid cell each time, and no cell could ever acquire a stable meaning for
   * the policy or for the positional action head.
   */
  private updateZoom(me: Player): void {
    if (this.opts.windowMode === "fixed") return;
    const bb = me.largestClusterBoundingBox;
    const bbExtent = bb
      ? Math.max(bb.max.x - bb.min.x, bb.max.y - bb.min.y)
      : 0;
    const areaExtent = Math.sqrt(Math.max(1, me.numTilesOwned()));
    const wanted = Math.max(bbExtent, areaExtent) * this.opts.windowZoom;

    while (this.level < this.ladder.length - 1 && wanted > this.ladder[this.level]) {
      this.level++;
    }
    while (this.level > 0 && wanted < this.ladder[this.level - 1] * SHRINK_MARGIN) {
      this.level--;
    }
  }

  encode(me: Player, step: number, goldIncome: number): EncodedObservation {
    const g = this.game;
    const G = this.opts.grid;
    this.updateZoom(me);
    const geometry = this.geometry(me);
    const spatial = new Float32Array(this.C * G * G);

    // Classify every live player once, so the per-cell loop is integer work.
    // 1 = self, 2 = ally (alliance or team), 3 = enemy.
    const category = new Map<number, number>();
    const strength = new Map<number, number>();
    for (const p of g.players()) {
      if (!p.isAlive()) continue;
      const cat = p === me ? 1 : me.isFriendly(p) ? 2 : 3;
      category.set(p.smallID(), cat);
      const tiles = Math.max(1, p.numTilesOwned());
      strength.set(p.smallID(), clamp01(Math.log1p(p.troops() / tiles) / 8));
    }

    // Slot id -> opponent smallID, so painting a tile straight into its
    // slot channel is O(1).
    const slotPlayers = this.pickOpponents(me);
    const slotOfSmallID = new Map<number, number>();
    for (let i = 0; i < slotPlayers.length; i++) {
      const p = slotPlayers[i];
      if (p !== null) slotOfSmallID.set(p.smallID(), i);
    }

    const idx = (c: number, gy: number, gx: number) =>
      (c * G + gy) * G + gx;
    const landNeighbors = new Set<number>();
    const borderAdjCount = new Map<number, number>();

    for (let gy = 0; gy < G; gy++) {
      for (let gx = 0; gx < G; gx++) {
        const x = Math.round(geometry.originX + (gx + 0.5) * geometry.cellSize);
        const y = Math.round(geometry.originY + (gy + 0.5) * geometry.cellSize);
        if (!g.isValidCoord(x, y)) {
          spatial[idx(CH.off_map, gy, gx)] = 1;
          continue;
        }
        const t = g.ref(x, y);
        const isLand = g.isLand(t);
        if (isLand) spatial[idx(CH.land, gy, gx)] = 1;
        else if (g.isOcean(t)) spatial[idx(CH.ocean, gy, gx)] = 1;
        if (g.isShoreline(t)) spatial[idx(CH.shoreline, gy, gx)] = 1;
        if (g.hasFallout(t)) spatial[idx(CH.fallout, gy, gx)] = 1;

        const ownerId = g.ownerID(t);
        const cat = category.get(ownerId) ?? 0;
        const st = strength.get(ownerId) ?? 0;
        if (cat === 1) {
          spatial[idx(CH.owner_self, gy, gx)] = 1;
          spatial[idx(CH.buildable, gy, gx)] = 1;
          spatial[idx(CH.strength_self, gy, gx)] = st;
          if (g.isBorder(t)) spatial[idx(CH.self_border, gy, gx)] = 1;
        } else if (cat === 2) {
          spatial[idx(CH.owner_ally, gy, gx)] = 1;
          spatial[idx(CH.strength_ally, gy, gx)] = st;
        } else if (cat === 3) {
          spatial[idx(CH.owner_enemy, gy, gx)] = 1;
          spatial[idx(CH.strength_enemy, gy, gx)] = st;
          const slot = slotOfSmallID.get(ownerId);
          if (slot !== undefined) {
            spatial[idx(SLOT_CHANNEL_BASE + slot, gy, gx)] = 1;
          }
        } else if (isLand) {
          spatial[idx(CH.owner_neutral_land, gy, gx)] = 1;
        }
        // A landing site: enemy or neutral land reachable from the sea.
        if (isLand && cat !== 1 && cat !== 2 && g.isShore(t)) {
          spatial[idx(CH.boat_target, gy, gx)] = 1;
        }
      }
    }

    // Units — structures split by role, warships painted separately because
    // they move and the structure channels do not cover them.
    for (const unit of g.units()) {
      if (!unit.isActive()) continue;
      const utype = unit.type();
      const isStructure = STRUCTURE_SET.has(utype);
      const isWarship = utype === UnitType.Warship;
      const isTrade = utype === UnitType.TradeShip;
      if (!isStructure && !isWarship && !isTrade) continue;
      const cell = this.tileToCell(unit.tile(), geometry);
      if (cell === null) continue;
      const owner = unit.owner();
      const mine = owner === me;
      if (isStructure) {
        if (ECON_SET.has(utype)) {
          spatial[idx(mine ? CH.econ_self : CH.econ_enemy, cell.gy, cell.gx)] = 1;
        } else if (MIL_SET.has(utype)) {
          spatial[idx(mine ? CH.mil_self : CH.mil_enemy, cell.gy, cell.gx)] = 1;
        } else if (ANTIAIR_SET.has(utype)) {
          spatial[
            idx(mine ? CH.antiair_self : CH.antiair_enemy, cell.gy, cell.gx)
          ] = 1;
        }
      } else if (isWarship) {
        spatial[idx(mine ? CH.warship_self : CH.warship_enemy, cell.gy, cell.gx)] = 1;
      } else if (isTrade) {
        spatial[idx(CH.trade_route, cell.gy, cell.gx)] = 1;
      }
    }

    // Agent's own border: paints a self-cell + records which owners touch it.
    // The counts feed `border_threat_log` per opponent.
    let borderLen = 0;
    const neighborBuf: TileRef[] = [0, 0, 0, 0];
    for (const t of me.borderTiles()) {
      borderLen++;
      const cell = this.tileToCell(t, geometry);
      if (cell !== null) {
        spatial[idx(CH.owner_self, cell.gy, cell.gx)] = 1;
        spatial[idx(CH.self_border, cell.gy, cell.gx)] = 1;
        spatial[idx(CH.buildable, cell.gy, cell.gx)] = 1;
      }
      const n = g.neighbors4(t, neighborBuf);
      for (let i = 0; i < n; i++) {
        const nt = neighborBuf[i];
        if (!g.isLand(nt)) continue;
        const owner = g.ownerID(nt);
        if (owner === me.smallID()) continue;
        landNeighbors.add(owner);
        borderAdjCount.set(owner, (borderAdjCount.get(owner) ?? 0) + 1);
      }
    }

    // Attack flow intensity painted on the frontier cells. For each attack we
    // normalise its troop count against the attacker's capacity so a 1k attack
    // from a 100k empire reads differently from a 1k attack from a 2k empire.
    for (const a of me.incomingAttacks()) {
      this.paintAttackFlow(spatial, idx, geometry, a, CH.attack_in_intensity);
    }
    for (const a of me.outgoingAttacks()) {
      this.paintAttackFlow(spatial, idx, geometry, a, CH.attack_out_intensity);
    }

    const minimap = this.encodeMinimap(me, category, strength);
    const players = this.encodePlayers(me, slotPlayers, borderAdjCount);
    const scalar = this.encodeScalars(
      me,
      step,
      goldIncome,
      borderLen,
      landNeighbors.size,
    );

    return {
      spatial,
      minimap,
      scalar,
      players,
      slotPlayers,
      geometry,
      landNeighbors,
      borderAdjCount,
    };
  }

  /**
   * Low-resolution picture of the whole map, at a fixed MINIMAP_SIDE x
   * MINIMAP_SIDE regardless of the zoom window. Lets the policy notice an
   * empire forming on the other side of the world — the ego-centric window
   * cannot.
   */
  private encodeMinimap(
    me: Player,
    category: Map<number, number>,
    strength: Map<number, number>,
  ): Float32Array {
    const g = this.game;
    const M = MINIMAP_SIDE;
    const out = new Float32Array(MC * M * M);
    const stepX = g.width() / M;
    const stepY = g.height() / M;
    const idx = (c: number, my: number, mx: number) =>
      (c * M + my) * M + mx;
    for (let my = 0; my < M; my++) {
      const y = Math.min(g.height() - 1, Math.floor((my + 0.5) * stepY));
      for (let mx = 0; mx < M; mx++) {
        const x = Math.min(g.width() - 1, Math.floor((mx + 0.5) * stepX));
        if (!g.isValidCoord(x, y)) continue;
        const t = g.ref(x, y);
        if (!g.isLand(t)) continue;
        out[idx(MCH.land, my, mx)] = 1;
        const ownerId = g.ownerID(t);
        const cat = category.get(ownerId) ?? 0;
        if (cat === 1) out[idx(MCH.owner_self, my, mx)] = 1;
        else if (cat === 2) out[idx(MCH.owner_ally, my, mx)] = 1;
        else if (cat === 3) out[idx(MCH.owner_enemy, my, mx)] = 1;
        if (cat !== 0) out[idx(MCH.strength, my, mx)] = strength.get(ownerId) ?? 0;
      }
    }
    return out;
  }

  /**
   * Paint one attack's intensity onto frontier cells. The attack carries a
   * `borderSize` and a sample of clustered positions along its front; we use
   * the positions to find grid cells and set an intensity scaled by the
   * attack's troop count.
   */
  private paintAttackFlow(
    spatial: Float32Array,
    idx: (c: number, gy: number, gx: number) => number,
    geometry: ViewGeometry,
    attack: Attack,
    channel: number,
  ): void {
    const intensity = clamp01(Math.log1p(attack.troops()) / 16);
    if (intensity === 0) return;
    const positions = attack.clusteredPositions();
    for (const t of positions) {
      const cell = this.tileToCell(t, geometry);
      if (cell === null) continue;
      // Max-pool so multiple attacks on the same cell stay bounded in [0,1].
      const i = idx(channel, cell.gy, cell.gx);
      if (spatial[i] < intensity) spatial[i] = intensity;
    }
  }

  /**
   * Window centred on the agent's largest territory cluster, then slid back
   * inside the map so no view wastes cells on emptiness it could have spent
   * on terrain. A window wider than the map stays centred on the map.
   */
  geometry(me: Player): ViewGeometry {
    const g = this.game;
    const G = this.opts.grid;
    const W = this.windowTiles();
    const { x: cx, y: cy } = this.centroid(me);
    return {
      originX: slide(cx - W / 2, W, g.width()),
      originY: slide(cy - W / 2, W, g.height()),
      cellSize: W / G,
      grid: G,
      zoomLevel: this.level,
      zoomLevels: this.ladder.length,
    };
  }

  /**
   * Centre of a player's largest territory cluster, falling back to its spawn
   * tile: `largestClusterBoundingBox` stays undefined until PlayerExecution
   * has run its first border pass.
   */
  centroid(p: Player): { x: number; y: number } {
    const g = this.game;
    const bb = p.largestClusterBoundingBox;
    if (bb) return { x: (bb.min.x + bb.max.x) / 2, y: (bb.min.y + bb.max.y) / 2 };
    const spawn = p.spawnTile();
    if (spawn !== undefined) return { x: g.x(spawn), y: g.y(spawn) };
    return { x: g.width() / 2, y: g.height() / 2 };
  }

  tileToCell(
    tile: TileRef,
    geo: ViewGeometry,
  ): { gx: number; gy: number } | null {
    const gx = Math.floor((this.game.x(tile) - geo.originX) / geo.cellSize);
    const gy = Math.floor((this.game.y(tile) - geo.originY) / geo.cellSize);
    if (gx < 0 || gy < 0 || gx >= geo.grid || gy >= geo.grid) return null;
    return { gx, gy };
  }

  /** Centre tile of a grid cell, clamped onto the map. */
  cellToTile(cellIndex: number, geo: ViewGeometry): TileRef | null {
    const gx = cellIndex % geo.grid;
    const gy = Math.floor(cellIndex / geo.grid);
    const x = Math.round(geo.originX + (gx + 0.5) * geo.cellSize);
    const y = Math.round(geo.originY + (gy + 0.5) * geo.cellSize);
    if (!this.game.isValidCoord(x, y)) return null;
    return this.game.ref(x, y);
  }

  /**
   * Opponents that fill slots 1..K: bordering players first, then the nearest,
   * so the slot the policy learns to attack stays semantically stable.
   */
  private pickOpponents(me: Player): (Player | null)[] {
    const g = this.game;
    const { x: mx, y: my } = this.centroid(me);

    const scored = g
      .players()
      .filter((p) => p !== me && p.isAlive() && p.numTilesOwned() > 0)
      .map((p) => {
        const { x: px, y: py } = this.centroid(p);
        const dist = Math.hypot(px - mx, py - my);
        return { p, dist, borders: me.sharesBorderWith(p) ? 0 : 1 };
      })
      .sort((a, b) => a.borders - b.borders || a.dist - b.dist);

    const slots: (Player | null)[] = new Array(this.opts.maxPlayers).fill(null);
    for (let i = 0; i < Math.min(scored.length, slots.length); i++) {
      slots[i] = scored[i].p;
    }
    return slots;
  }

  private encodePlayers(
    me: Player,
    slots: (Player | null)[],
    borderAdjCount: Map<number, number>,
  ): Float32Array {
    const g = this.game;
    const K = this.opts.maxPlayers;
    const out = new Float32Array(K * PF);
    const land = Math.max(1, g.numLandTiles());
    const diag = Math.hypot(g.width(), g.height());
    const myTargets = new Set(me.targets().map((p) => p.smallID()));

    // Build quick lookups for the per-slot attack strength (not just a flag).
    const incomingBy = new Map<number, number>();
    for (const a of me.incomingAttacks()) {
      const id = a.attacker().smallID();
      incomingBy.set(id, (incomingBy.get(id) ?? 0) + a.troops());
    }
    const outgoingBy = new Map<number, number>();
    for (const a of me.outgoingAttacks()) {
      const tgt = a.target();
      if (!tgt.isPlayer()) continue;
      const id = (tgt as Player).smallID();
      outgoingBy.set(id, (outgoingBy.get(id) ?? 0) + a.troops());
    }

    // Compact relation-graph features: for each slot, how many *other* slots
    // are their allies, and how many *other* slots are attacking them. The
    // policy can use this to see coalition structure without the K×K matrix.
    const nAlliesAmongSlots: number[] = new Array(K).fill(0);
    const nAttackersOnThem: number[] = new Array(K).fill(0);
    for (let i = 0; i < K; i++) {
      const pi = slots[i];
      if (pi === null) continue;
      for (let j = 0; j < K; j++) {
        if (i === j) continue;
        const pj = slots[j];
        if (pj === null) continue;
        if (pi.isAlliedWith(pj) || pi.isOnSameTeam(pj)) {
          nAlliesAmongSlots[i]++;
        }
        if (pj.outgoingAttacks().some((a) => a.target() === pi)) {
          nAttackersOnThem[i]++;
        }
      }
    }

    const currentTick = g.ticks();
    const { x: mx, y: my } = this.centroid(me);
    const buildUnitsPrefix = PF - BUILD_UNITS.length;

    for (let i = 0; i < slots.length; i++) {
      const p = slots[i];
      if (p === null) continue;
      const o = i * PF;
      const { x: px, y: py } = this.centroid(p);
      let f = 0;
      out[o + f++] = 1; // valid
      // Economy & demography
      out[o + f++] = p.numTilesOwned() / land;
      out[o + f++] = Math.log1p(p.numTilesOwned()) / Math.log1p(land);
      out[o + f++] = clamp01(Math.log1p(p.troops()) / 16);
      out[o + f++] = clamp01(Math.log1p(Number(p.gold())) / 25);
      out[o + f++] = clamp01(Math.log1p(Number(p.goldEarned())) / 25);
      out[o + f++] = clamp01(Math.log1p(Number(p.tradeGold())) / 25);
      out[o + f++] = p.troops() / Math.max(1, p.troops() + me.troops());
      out[o + f++] =
        p.numTilesOwned() /
        Math.max(1, p.numTilesOwned() + me.numTilesOwned());
      out[o + f++] = me.sharesBorderWith(p) ? 1 : 0;
      out[o + f++] = clamp01(Math.hypot(px - mx, py - my) / diag);
      // Diplomacy
      out[o + f++] = relationValue(me, p);
      out[o + f++] = me.isAlliedWith(p) ? 1 : 0;
      out[o + f++] = me.isOnSameTeam(p) ? 1 : 0;
      out[o + f++] = myTargets.has(p.smallID()) ? 1 : 0;
      out[o + f++] = p.targets().some((t) => t === me) ? 1 : 0;
      out[o + f++] = incomingBy.has(p.smallID()) ? 1 : 0;
      out[o + f++] = outgoingBy.has(p.smallID()) ? 1 : 0;
      out[o + f++] = clamp01(
        Math.log1p(incomingBy.get(p.smallID()) ?? 0) / 16,
      );
      out[o + f++] = clamp01(
        Math.log1p(outgoingBy.get(p.smallID()) ?? 0) / 16,
      );
      out[o + f++] = p.isTraitor() ? 1 : 0;
      out[o + f++] = clamp01(Math.log1p(p.betrayals()) / 4);
      // Alliance expiry — the alliance's remaining lifetime, 0 if no alliance.
      const al = me.allianceWith(p);
      out[o + f++] = al
        ? clamp01((al.expiresAt() - currentTick) / 1000)
        : 0;
      out[o + f++] = clamp01(p.tradingPartners().length / 8);
      out[o + f++] = p.hasEmbargoAgainst(me) ? 1 : 0;
      out[o + f++] = me.hasEmbargoAgainst(p) ? 1 : 0;
      out[o + f++] = p.type() === PlayerType.Nation ? 1 : 0;
      out[o + f++] = p.isDisconnected() ? 1 : 0;
      out[o + f++] = p.isImmune() ? 1 : 0;
      out[o + f++] = p.inDoomsdayClock() ? 1 : 0;
      out[o + f++] = p.isDecaying() ? 1 : 0;
      // Compact relation-graph
      out[o + f++] = clamp01(nAlliesAmongSlots[i] / Math.max(1, K - 1));
      out[o + f++] = clamp01(nAttackersOnThem[i] / Math.max(1, K - 1));
      // Border threat: their troops per tile times tiles adjacent to my border.
      const adj = borderAdjCount.get(p.smallID()) ?? 0;
      const perTile = p.troops() / Math.max(1, p.numTilesOwned());
      out[o + f++] = clamp01(Math.log1p(perTile * adj) / 16);
      // Per-unit counts — these end the row, so BUILD_UNITS.length features.
      // `buildUnitsPrefix` keeps us aligned if someone adds features above.
      f = buildUnitsPrefix;
      for (const u of BUILD_UNITS) {
        out[o + f++] = clamp01(Math.log1p(p.unitCount(u)) / 4);
      }
    }
    return out;
  }

  private encodeScalars(
    me: Player,
    step: number,
    goldIncome: number,
    borderLen: number,
    numNeighbors: number,
  ): Float32Array {
    const g = this.game;
    const out = new Float32Array(S);
    const land = Math.max(1, g.numLandTiles());
    const alive = g.players().filter((p) => p.isAlive());
    const maxTroops = Math.max(1, g.config().maxTroops(me));
    const gold = Number(me.gold());
    const rank = alive.filter(
      (p) => p.numTilesOwned() > me.numTilesOwned(),
    ).length;
    const totalPlayers = Math.max(1, g.players().length);
    const claimedTiles = alive.reduce((s, p) => s + p.numTilesOwned(), 0);

    // Average relation to opponents (0..1). Diplomatic "temperature".
    const others = alive.filter((p) => p !== me);
    let relSum = 0;
    for (const p of others) relSum += relationValue(me, p);
    const avgRel = others.length > 0 ? relSum / others.length : 0.5;

    const inc = me.incomingAttacks();
    const outg = me.outgoingAttacks();
    let i = 0;
    out[i++] = clamp01(g.ticks() / (this.opts.maxSteps * this.opts.ticksPerStep));
    out[i++] = clamp01(step / this.opts.maxSteps);
    out[i++] = me.numTilesOwned() / land;
    out[i++] = Math.log1p(me.numTilesOwned()) / Math.log1p(land);
    out[i++] = clamp01(Math.log1p(me.troops()) / 16);
    out[i++] = clamp01(me.troops() / maxTroops);
    out[i++] = clamp01(Math.log1p(gold) / 25);
    out[i++] = clamp01(Math.log1p(Math.max(0, goldIncome)) / 20);
    out[i++] = clamp01(Math.log1p(Number(me.tradeGold())) / 25);
    out[i++] = alive.length > 1 ? rank / (alive.length - 1) : 0;
    out[i++] = alive.length / totalPlayers;
    out[i++] = clamp01(alive.length / 32);
    out[i++] = clamp01(claimedTiles / land);
    out[i++] = clamp01(inc.length / 5);
    out[i++] = clamp01(
      Math.log1p(inc.reduce((s, a) => s + a.troops(), 0)) / 16,
    );
    out[i++] = clamp01(outg.length / 5);
    out[i++] = clamp01(
      Math.log1p(outg.reduce((s, a) => s + a.troops(), 0)) / 16,
    );
    out[i++] = clamp01(me.allies().length / 8);
    out[i++] = clamp01(me.incomingAllianceRequests().length / 8);
    out[i++] = clamp01(me.tradingPartners().length / 8);
    out[i++] = clamp01(Math.log1p(me.betrayals()) / 4);
    out[i++] = avgRel;
    out[i++] = me.isTraitor() ? 1 : 0;
    out[i++] = me.inDoomsdayClock() ? 1 : 0;
    out[i++] = clamp01(me.doomsdayClockTicks() / 300);
    out[i++] = me.isDecaying() ? 1 : 0;
    out[i++] = me.isImmune() ? 1 : 0;
    out[i++] = g.inSpawnPhase() ? 1 : 0;
    out[i++] = clamp01(Math.log1p(borderLen) / 10);
    out[i++] = clamp01(numNeighbors / 8);

    // Which zoom level this observation was taken at. Without it the same
    // picture would be ambiguous between a small empire seen close up and a
    // large one seen from far away.
    const window = this.windowTiles();
    const first = this.ladder[0];
    const last = this.ladder[this.ladder.length - 1];
    out[i++] =
      last > first
        ? clamp01(Math.log(window / first) / Math.log(last / first))
        : 0;
    out[i++] = clamp01(window / Math.max(g.width(), g.height()));

    for (const u of BUILD_UNITS) {
      out[i++] = clamp01(Math.log1p(me.unitCount(u)) / 4);
    }
    for (const u of BUILD_UNITS) {
      const cost = Number(g.unitInfo(u).cost(g, me));
      out[i++] = cost <= 0 ? 1 : clamp01(gold / cost / 2);
    }
    // Filled in by the action masker, which already computes buildability.
    return out;
  }
}

/** Offset of the `can_build_*` block, written by the action masker. */
export const CAN_BUILD_OFFSET = S - BUILD_UNITS.length;

function relationValue(me: Player, other: Player): number {
  const r = me.relation(other);
  switch (r) {
    case Relation.Hostile:
      return 0;
    case Relation.Distrustful:
      return 1 / 3;
    case Relation.Neutral:
      return 2 / 3;
    case Relation.Friendly:
      return 1;
    default:
      return 2 / 3;
  }
}

/**
 * Window sizes the encoder may use, smallest first, each twice the previous
 * and the last clamped to the map. In fixed mode the ladder has a single rung.
 */
function buildLadder(game: Game, opts: EncoderOptions): number[] {
  const mapExtent = Math.max(game.width(), game.height());
  if (opts.windowMode === "fixed") {
    return [Math.min(Math.max(8, opts.windowTiles), mapExtent)];
  }
  const max = Math.min(
    opts.windowMaxTiles > 0 ? opts.windowMaxTiles : mapExtent,
    mapExtent,
  );
  const min = Math.min(Math.max(8, opts.windowMinTiles), max);
  const ladder: number[] = [];
  for (let w = min; w < max; w *= 2) ladder.push(w);
  ladder.push(max);
  return ladder;
}

/** Keeps a window of width `w` inside `[0, extent]`, centring it if it cannot fit. */
function slide(origin: number, w: number, extent: number): number {
  if (w >= extent) return (extent - w) / 2;
  return Math.min(Math.max(origin, 0), extent - w);
}

function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
