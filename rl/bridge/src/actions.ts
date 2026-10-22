/**
 * Action masking and decoding.
 *
 * The starter action space is a flat MultiDiscrete of five independent heads:
 *
 *   [ action_type, target_slot, ratio, unit_type, cell ]
 *
 * Heads irrelevant to the sampled `action_type` are simply ignored when the
 * intent is built. Masks are computed per head, unconditionally — an
 * approximation that keeps the policy a plain product distribution. Combos
 * that slip through (e.g. BUILD a Port on an inland cell) are rejected by the
 * engine and reported back as `invalid`, which carries a small penalty.
 */
import {
  Game,
  Player,
  UnitType,
} from "../../../OpenFrontIO/src/core/game/Game";
import { TileRef } from "../../../OpenFrontIO/src/core/game/GameMap";
import {
  ClientID,
  Intent,
  StampedIntent,
} from "../../../OpenFrontIO/src/core/Schemas";
import { EncodedObservation, Encoder } from "./observation";
import {
  A_ATTACK,
  A_BOAT,
  A_BUILD,
  A_NOOP,
  ACTION_TYPES,
  BUILD_UNITS,
  CHANNELS,
  RATIOS,
} from "./spec";

const CHANNEL_INDEX: Record<string, number> = Object.fromEntries(
  CHANNELS.map((n, i) => [n, i]),
);

const PROBE_COUNT = 12;

export interface Masks {
  actionType: Uint8Array;
  target: Uint8Array;
  ratio: Uint8Array;
  unit: Uint8Array;
  cell: Uint8Array;
  cellBuild: Uint8Array;
  cellBoat: Uint8Array;
}

export interface DecodedAction {
  intent: StampedIntent | null;
  invalid: boolean;
  label: string;
}

export class ActionCodec {
  constructor(
    private readonly game: Game,
    private readonly encoder: Encoder,
    private readonly clientID: ClientID,
    private readonly grid: number,
    private readonly maxPlayers: number,
  ) {}

  /**
   * Builds every head's mask and back-fills the `can_build_*` scalars, which
   * come from the same buildability probes.
   */
  masks(me: Player, obs: EncodedObservation, canBuildOut: Float32Array): Masks {
    const g = this.game;
    const cells = this.grid * this.grid;
    const gold = me.gold();
    const probes = this.probeTiles(me);

    const target = new Uint8Array(this.maxPlayers + 1);
    if (obs.landNeighbors.has(0)) target[0] = 1; // TerraNullius
    for (let i = 0; i < this.maxPlayers; i++) {
      const p = obs.slotPlayers[i];
      if (p === null) continue;
      if (obs.landNeighbors.has(p.smallID()) && me.canAttackPlayer(p)) {
        target[i + 1] = 1;
      }
    }

    const unit = new Uint8Array(BUILD_UNITS.length);
    for (let i = 0; i < BUILD_UNITS.length; i++) {
      const type = BUILD_UNITS[i];
      if (g.config().isUnitDisabled(type)) continue;
      if (gold < g.unitInfo(type).cost(g, me)) continue;
      const ok =
        type === UnitType.Warship
          ? me.unitCount(UnitType.Port) > 0
          : probes.some((t) => me.canBuild(type, t) !== false);
      if (ok) unit[i] = 1;
      canBuildOut[i] = ok ? 1 : 0;
    }

    // Cell masks read straight off the spatial channels already computed.
    const cellBuild = this.channelMask(obs, "buildable");
    const cellBoat = this.channelMask(obs, "boat_target");
    const cell = new Uint8Array(cells);
    for (let i = 0; i < cells; i++) {
      cell[i] = cellBuild[i] | cellBoat[i];
    }
    ensureNonEmpty(cell);

    const actionType = new Uint8Array(ACTION_TYPES.length);
    actionType[A_NOOP] = 1;
    actionType[A_ATTACK] = target.some((v) => v === 1) && me.troops() > 1 ? 1 : 0;
    actionType[A_BOAT] =
      cellBoat.some((v) => v === 1) && me.troops() > 1 ? 1 : 0;
    actionType[A_BUILD] = unit.some((v) => v === 1) ? 1 : 0;

    const ratio = new Uint8Array(RATIOS.length).fill(1);
    ensureNonEmpty(target);
    ensureNonEmpty(unit);

    return { actionType, target, ratio, unit, cell, cellBuild, cellBoat };
  }

  decode(me: Player, obs: EncodedObservation, action: number[]): DecodedAction {
    const [aType, aTarget, aRatio, aUnit, aCell] = action;
    const ratio = RATIOS[clampIdx(aRatio, RATIOS.length)];

    switch (aType) {
      case A_NOOP:
        return { intent: null, invalid: false, label: "NOOP" };

      case A_ATTACK: {
        const slot = clampIdx(aTarget, this.maxPlayers + 1);
        let targetID: string | null = null;
        if (slot > 0) {
          const p = obs.slotPlayers[slot - 1];
          if (p === null || !me.canAttackPlayer(p)) {
            return { intent: null, invalid: true, label: "ATTACK/no-target" };
          }
          targetID = p.id();
        } else if (!obs.landNeighbors.has(0)) {
          return { intent: null, invalid: true, label: "ATTACK/no-neutral" };
        }
        const troops = Math.floor(me.troops() * ratio);
        if (troops < 1) {
          return { intent: null, invalid: true, label: "ATTACK/no-troops" };
        }
        return {
          intent: this.stamp({ type: "attack", targetID, troops }),
          invalid: false,
          label: `ATTACK[${slot}] x${ratio}`,
        };
      }

      case A_BOAT: {
        const dst = this.encoder.cellToTile(
          clampIdx(aCell, this.grid * this.grid),
          obs.geometry,
        );
        if (dst === null) {
          return { intent: null, invalid: true, label: "BOAT/off-map" };
        }
        const troops = Math.floor(me.troops() * ratio);
        if (troops < 1) {
          return { intent: null, invalid: true, label: "BOAT/no-troops" };
        }
        // mask_cell_boat only knows "enemy or neutral land on a shore"; it
        // cannot know whether a sea route exists, because testing that for
        // every cell would cost a spatial search per cell per step. Run the
        // engine's own check once, here, on the cell actually chosen —
        // otherwise an unreachable landing is a silent no-op that costs the
        // agent a turn and teaches it nothing.
        if (me.canBuild(UnitType.TransportShip, dst) === false) {
          return { intent: null, invalid: true, label: "BOAT/unreachable" };
        }
        return {
          intent: this.stamp({ type: "boat", troops, dst }),
          invalid: false,
          label: `BOAT->${dst} x${ratio}`,
        };
      }

      case A_BUILD: {
        const type = BUILD_UNITS[clampIdx(aUnit, BUILD_UNITS.length)];
        const target = this.encoder.cellToTile(
          clampIdx(aCell, this.grid * this.grid),
          obs.geometry,
        );
        if (target === null) {
          return { intent: null, invalid: true, label: "BUILD/off-map" };
        }
        if (this.game.unitInfo(type).cost(this.game, me) > me.gold()) {
          return { intent: null, invalid: true, label: `BUILD/${type}/broke` };
        }
        // canBuild snaps to a legal nearby tile (ports search the shoreline),
        // so send the snapped ref rather than the raw cell centre.
        const snapped = me.canBuild(type, target);
        if (snapped === false) {
          return { intent: null, invalid: true, label: `BUILD/${type}/illegal` };
        }
        return {
          intent: this.stamp({ type: "build_unit", unit: type, tile: snapped }),
          invalid: false,
          label: `BUILD ${type}@${snapped}`,
        };
      }

      default:
        return { intent: null, invalid: true, label: "UNKNOWN" };
    }
  }

  private stamp(intent: Intent): StampedIntent {
    return { ...intent, clientID: this.clientID } as StampedIntent;
  }

  private channelMask(
    obs: EncodedObservation,
    channel: "buildable" | "boat_target",
  ): Uint8Array {
    const cells = this.grid * this.grid;
    const out = new Uint8Array(cells);
    const base = CHANNEL_INDEX[channel] * cells;
    for (let i = 0; i < cells; i++) {
      out[i] = obs.spatial[base + i] > 0 ? 1 : 0;
    }
    return out;
  }

  /** A spread of owned tiles used to test whether a unit type is placeable. */
  private probeTiles(me: Player): TileRef[] {
    const border: TileRef[] = [];
    for (const t of me.borderTiles()) border.push(t);
    if (border.length === 0) {
      const spawn = me.spawnTile();
      return spawn === undefined ? [] : [spawn];
    }
    const stride = Math.max(1, Math.floor(border.length / PROBE_COUNT));
    const out: TileRef[] = [];
    for (let i = 0; i < border.length && out.length < PROBE_COUNT; i += stride) {
      out.push(border[i]);
    }
    return out;
  }
}

function clampIdx(v: number, n: number): number {
  const i = Math.floor(Number(v));
  if (!Number.isFinite(i) || i < 0) return 0;
  return i >= n ? n - 1 : i;
}

/** A fully-masked head would break categorical sampling; fall back to open. */
function ensureNonEmpty(mask: Uint8Array): void {
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] === 1) return;
  }
  mask.fill(1);
}
