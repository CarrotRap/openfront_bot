/**
 * Renderer for the live web viewer.
 *
 * Produces a downsampled, palettised picture of the whole map plus the
 * geometry of the agent's observation window, so the browser can draw the
 * game and outline exactly what the policy is looking at. This runs only when
 * the env is created with `emitRender: true` — training never pays for it.
 */
import {
  Game,
  Player,
  PlayerType,
  UnitType,
} from "../../../OpenFrontIO/src/core/game/Game";
import { EncodedObservation } from "./observation";
import { STRUCTURE_TYPES } from "./spec";

/** Fixed palette slots that come before the per-player colours. */
const OCEAN = 0;
const WATER = 1;
const NEUTRAL_LAND = 2;
const FALLOUT = 3;
const FIRST_PLAYER = 4;

const BASE_PALETTE: [number, number, number][] = [
  [17, 34, 64], // ocean
  [38, 70, 110], // lake / coastal water
  [72, 78, 62], // unclaimed land
  [150, 60, 140], // fallout
];

export interface RenderFrame {
  width: number;
  height: number;
  /** Map tiles per rendered pixel. */
  step: number;
  palette: [number, number, number][];
  /** Agent's observation window, in rendered-pixel coordinates. */
  window: {
    x: number;
    y: number;
    w: number;
    h: number;
    grid: number;
    /** Window side and grid-cell side in map tiles, plus the zoom rung. */
    tiles: number;
    cellTiles: number;
    zoomLevel: number;
    zoomLevels: number;
  };
  structures: { x: number; y: number; type: string; self: boolean }[];
  players: {
    slot: number | null;
    name: string;
    color: string;
    tiles: number;
    troops: number;
    isSelf: boolean;
    isAlly: boolean;
    type: string;
  }[];
  pixels: Uint8Array;
}

const STRUCTURE_SET = new Set<UnitType>(STRUCTURE_TYPES);

export class Renderer {
  private readonly step: number;
  private readonly width: number;
  private readonly height: number;
  private readonly pixels: Uint8Array;

  constructor(
    private readonly game: Game,
    maxDim = 256,
  ) {
    const longest = Math.max(game.width(), game.height());
    this.step = Math.max(1, Math.ceil(longest / maxDim));
    this.width = Math.ceil(game.width() / this.step);
    this.height = Math.ceil(game.height() / this.step);
    this.pixels = new Uint8Array(this.width * this.height);
  }

  frame(me: Player, obs: EncodedObservation): RenderFrame {
    const g = this.game;
    const palette: [number, number, number][] = [...BASE_PALETTE];
    const slotOf = new Map<number, number>();

    // One palette entry per living player, self and allies pinned to colours
    // the eye can pick out instantly.
    const players: RenderFrame["players"] = [];
    const slotIndex = new Map<number, number>();
    obs.slotPlayers.forEach((p, i) => {
      if (p !== null) slotIndex.set(p.smallID(), i);
    });

    for (const p of g.players()) {
      if (!p.isAlive()) continue;
      const isSelf = p === me;
      const isAlly = !isSelf && me.isFriendly(p);
      const rgb = isSelf
        ? ([74, 222, 128] as [number, number, number])
        : isAlly
          ? ([56, 189, 248] as [number, number, number])
          : enemyColor(p.smallID());
      const index = FIRST_PLAYER + palette.length - BASE_PALETTE.length;
      if (index > 255) break; // palette is 8-bit; ignore the long tail
      palette.push(rgb);
      slotOf.set(p.smallID(), index);
      players.push({
        slot: slotIndex.get(p.smallID()) ?? null,
        name: p.name(),
        color: `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`,
        tiles: p.numTilesOwned(),
        troops: Math.round(p.troops()),
        isSelf,
        isAlly,
        type:
          p.type() === PlayerType.Nation
            ? "nation"
            : p.type() === PlayerType.Human
              ? "agent"
              : "bot",
      });
    }
    players.sort((a, b) => b.tiles - a.tiles);

    for (let py = 0; py < this.height; py++) {
      const y = Math.min(g.height() - 1, py * this.step);
      for (let px = 0; px < this.width; px++) {
        const x = Math.min(g.width() - 1, px * this.step);
        const t = g.ref(x, y);
        let value: number;
        if (!g.isLand(t)) {
          value = g.isOcean(t) ? OCEAN : WATER;
        } else if (g.hasFallout(t)) {
          value = FALLOUT;
        } else {
          value = slotOf.get(g.ownerID(t)) ?? NEUTRAL_LAND;
        }
        this.pixels[py * this.width + px] = value;
      }
    }

    const structures: RenderFrame["structures"] = [];
    for (const unit of g.units()) {
      if (!unit.isActive() || !STRUCTURE_SET.has(unit.type())) continue;
      structures.push({
        x: g.x(unit.tile()) / this.step,
        y: g.y(unit.tile()) / this.step,
        type: String(unit.type()),
        self: unit.owner() === me,
      });
    }

    const geo = obs.geometry;
    return {
      width: this.width,
      height: this.height,
      step: this.step,
      palette,
      window: {
        x: geo.originX / this.step,
        y: geo.originY / this.step,
        w: (geo.cellSize * geo.grid) / this.step,
        h: (geo.cellSize * geo.grid) / this.step,
        grid: geo.grid,
        tiles: Math.round(geo.cellSize * geo.grid),
        cellTiles: Math.round(geo.cellSize),
        zoomLevel: geo.zoomLevel,
        zoomLevels: geo.zoomLevels,
      },
      structures,
      players,
      pixels: this.pixels,
    };
  }
}

/**
 * Stable, well-separated hue per player, deterministic across episodes. The
 * whole green band (70-170) is reserved so no opponent reads as the agent.
 */
function enemyColor(smallID: number): [number, number, number] {
  const raw = (smallID * 137.508) % 260;
  const hue = raw > 70 ? raw + 100 : raw;
  return hslToRgb(hue, 0.62, 0.55);
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = h / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  const [r, g, b] =
    hp < 1
      ? [c, x, 0]
      : hp < 2
        ? [x, c, 0]
        : hp < 3
          ? [0, c, x]
          : hp < 4
            ? [0, x, c]
            : hp < 5
              ? [x, 0, c]
              : [c, 0, x];
  const m = l - c / 2;
  return [
    Math.round((r + m) * 255),
    Math.round((g + m) * 255),
    Math.round((b + m) * 255),
  ];
}
