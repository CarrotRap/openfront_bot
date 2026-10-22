/**
 * Single source of truth for the observation / action layout.
 *
 * Both sides of the bridge read their shapes from here: the TypeScript
 * encoder writes the arrays, and the Python Gymnasium env builds its spaces
 * from the `spec` frame this module produces. Changing a layout here changes
 * it on both sides at once — there is nothing to keep in sync by hand.
 */
import { UnitType } from "../../../OpenFrontIO/src/core/game/Game";

/**
 * Static spatial channels. The full channel list used at runtime is
 * `channelsFor(maxPlayers)` — it appends one per-opponent-slot channel at the
 * end so the CNN can tie each tile to a specific slot in the players branch.
 */
export const CHANNELS = [
  "land",
  "ocean",
  "shoreline",
  "off_map",
  "buildable",
  "boat_target",
  "fallout",
  "owner_self",
  "owner_ally",
  "owner_enemy",
  "owner_neutral_land",
  "self_border",
  // Per-category strength (replaces the old single `owner_strength`).
  "strength_self",
  "strength_ally",
  "strength_enemy",
  // Structures grouped by role.
  "econ_self",
  "econ_enemy",
  "mil_self",
  "mil_enemy",
  "antiair_self",
  "antiair_enemy",
  // Dynamic naval units painted every frame (they move).
  "warship_self",
  "warship_enemy",
  // Active trade routes between ports, a proxy for economic dependencies.
  "trade_route",
  // Attack flow intensity painted on the frontier cells.
  "attack_in_intensity",
  "attack_out_intensity",
] as const;

export const C_STATIC = CHANNELS.length;

/** Full list of spatial channels for a given opponent-slot count. */
export function channelsFor(maxPlayers: number): string[] {
  const slots = Array.from({ length: maxPlayers }, (_, i) => `slot_${i}`);
  return [...CHANNELS, ...slots];
}

/** Total spatial channel count: static channels + K per-slot channels. */
export function totalChannels(maxPlayers: number): number {
  return C_STATIC + maxPlayers;
}

/** Action head 0: what kind of action to take. */
export const ACTION_TYPES = ["NOOP", "ATTACK", "BOAT_ATTACK", "BUILD"] as const;
export type ActionTypeName = (typeof ACTION_TYPES)[number];

export const A_NOOP = 0;
export const A_ATTACK = 1;
export const A_BOAT = 2;
export const A_BUILD = 3;

/** Action head 2: fraction of troops / gold committed. */
export const RATIOS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.8] as const;

/** Action head 3: buildable unit types exposed to the starter agent. */
export const BUILD_UNITS: UnitType[] = [
  UnitType.City,
  UnitType.Port,
  UnitType.Factory,
  UnitType.DefensePost,
  UnitType.MissileSilo,
  UnitType.SAMLauncher,
  UnitType.Warship,
];

/** Economic structures — grouped into the `econ_*` spatial channels. */
export const STRUCTURES_ECON: UnitType[] = [
  UnitType.City,
  UnitType.Port,
  UnitType.Factory,
];
/** Offensive / defensive structures — grouped into the `mil_*` channels. */
export const STRUCTURES_MIL: UnitType[] = [
  UnitType.DefensePost,
  UnitType.MissileSilo,
];
/** Anti-air structures — grouped into the `antiair_*` channels. */
export const STRUCTURES_ANTIAIR: UnitType[] = [UnitType.SAMLauncher];

/** Union of static structure types (buildings, not mobile units). */
export const STRUCTURE_TYPES: UnitType[] = [
  ...STRUCTURES_ECON,
  ...STRUCTURES_MIL,
  ...STRUCTURES_ANTIAIR,
];

/**
 * Per-opponent features. The policy's pointer head over target slots reads the
 * K embeddings produced from this vector, so every feature here must be a
 * stable, human-visible fact about that opponent.
 */
export const PLAYER_FEATURES = [
  // Validity flag — slot 0 of every row. Must stay index 0 so the pooling code
  // in policy.py can mask empty slots with `players[..., :1]`.
  "valid",
  // Economy & demography
  "tiles_share",
  "tiles_log",
  "troops_log",
  "gold_log",
  "gold_earned_log",
  "trade_gold_log",
  "troops_share_vs_self",
  "tiles_share_vs_self",
  "shares_border",
  "distance",
  // Diplomacy
  "relation",
  "is_allied",
  "is_same_team",
  "self_targets_them",
  "they_target_self",
  "attacking_self",
  "self_attacking_them",
  "attack_in_strength_log",
  "attack_out_strength_log",
  "is_traitor",
  "betrayals_log",
  "alliance_remaining",
  "trading_partners_norm",
  "has_embargo_me",
  "me_has_embargo",
  "is_nation",
  "is_disconnected",
  "is_immune",
  "in_doomsday",
  "is_decaying",
  // Compact relation-graph features
  "n_allies_among_slots",
  "n_attackers_on_them",
  "border_threat_log",
  // Per-unit counts (log1p scaled)
  ...BUILD_UNITS.map((u) => `count_${slug(u)}`),
] as const;

export const PF = PLAYER_FEATURES.length;

/** Names of the global scalar vector, in order. Built once at module load. */
export const SCALAR_NAMES: string[] = [
  "tick_norm",
  "step_norm",
  "tiles_share",
  "tiles_log",
  "troops_log",
  "troops_ratio",
  "gold_log",
  "gold_income_log",
  "trade_gold_log",
  "rank_norm",
  "alive_share",
  "num_alive_norm",
  "map_claimed",
  "n_incoming_attacks",
  "incoming_attack_troops_log",
  "n_outgoing_attacks",
  "outgoing_attack_troops_log",
  "n_allies",
  "n_alliance_requests",
  "n_trading_partners",
  "n_betrayals_log",
  "avg_relation_neighbors",
  "is_traitor",
  "in_doomsday_clock",
  "doomsday_ticks_norm",
  "is_decaying",
  "is_immune",
  "in_spawn_phase",
  "border_len_log",
  "n_neighbors",
  // The observation window rescales with the agent, so the policy must be
  // told which zoom level it is looking at — the same 8 green cells mean very
  // different things at 64 tiles/window and at 1024.
  "window_scale",
  "window_frac",
  ...BUILD_UNITS.map((u) => `count_${slug(u)}`),
  ...BUILD_UNITS.map((u) => `afford_${slug(u)}`),
  ...BUILD_UNITS.map((u) => `can_build_${slug(u)}`),
];

export const S = SCALAR_NAMES.length;

/** Side of the fixed-resolution global minimap, in cells. */
export const MINIMAP_SIDE = 32;

/** Spatial channels painted on the global minimap, in order. */
export const MINIMAP_CHANNELS = [
  "land",
  "owner_self",
  "owner_ally",
  "owner_enemy",
  "strength",
] as const;

export const MC = MINIMAP_CHANNELS.length;

function slug(u: UnitType): string {
  return String(u).toLowerCase().replace(/\s+/g, "_");
}
