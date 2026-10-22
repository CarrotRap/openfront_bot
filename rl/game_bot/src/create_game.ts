/**
 * Create a private game on the local dev server via the admin bot HTTP API.
 *
 * The admin-bot HTTP API ships with the server (see
 * `OpenFrontIO/src/server/AdminBotRoutes.ts`). In dev the key is baked into
 * the start:server-dev script as the string below. 404 means the admin bot
 * routes are disabled (missing key); 401 means the key doesn't match.
 *
 * Usage:
 *   tsx src/create_game.ts [--map big_plains] [--bots 6] [--nations default]
 *                          [--difficulty Medium] [--host http://localhost:3000]
 *
 * Prints the gameID to stdout and the worker URL (used by the WebSocket
 * client) to stderr for logs.
 */
import { Difficulty, GameMapType, GameType } from "src/core/game/Game";

const DEV_ADMIN_BOT_KEY =
  "WARNING_DEV_ADMIN_BOT_KEY_DO_NOT_USE_IN_PRODUCTION";
const DEV_ADMIN_BOT_HEADER = "x-admin-bot-key";

type Args = {
  host: string;
  map: string;
  bots: number;
  // GameConfigSchema.shape.nations accepts a uint 1-400 for a custom count,
  // or "default" (all manifest nations) / "disabled" (no NPCs). Our earlier
  // implementation sent a bogus `disableNPCs` field that the server strips
  // during zod validation, so every "custom count" demo silently fell back
  // to "default" (72 nations on World). The number form is passed through
  // as-is now.
  nations: number | "default" | "disabled";
  difficulty: "Easy" | "Medium" | "Hard" | "Impossible";
  /** false → the bot picks its own spawn tile (via spawn_picker). */
  randomSpawn: boolean;
};

// Admin bot routes are registered on each worker (see Worker.ts:368), not on
// the master, so we hit worker 0 directly on its port rather than going
// through the vite /api proxy (which routes to the master).
function parseArgs(argv: string[]): Args {
  const out: Args = {
    host: "http://localhost:3001",
    // Must be one of the names in GameMapType (see the error from the server
    // for the full list). The dash-case names used in rl/README.md (big_plains,
    // world) are the headless-sim naming only — the real server uses the
    // display names with spaces: "World", "Europe", "Pangaea", etc.
    map: "World",
    bots: 6,
    nations: "default",
    difficulty: "Medium",
    // Default off → bot picks its spawn. demo.sh keeps this default; pass
    // --random-spawn to let the server auto-place everyone (training-like).
    randomSpawn: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    const val = () => {
      const v = argv[++i];
      if (v === undefined) throw new Error(`missing value for ${key}`);
      return v;
    };
    switch (key) {
      case "--host":
        out.host = val();
        break;
      case "--map":
        out.map = val();
        break;
      case "--bots":
        out.bots = parseInt(val(), 10);
        break;
      case "--nations": {
        const v = val();
        if (v === "default" || v === "disabled") {
          out.nations = v;
        } else {
          const n = parseInt(v, 10);
          if (!Number.isFinite(n) || n < 1 || n > 400) {
            throw new Error(
              `--nations must be an integer in [1,400], "default" or "disabled" (got "${v}")`,
            );
          }
          out.nations = n;
        }
        break;
      }
      case "--difficulty":
        out.difficulty = val() as Args["difficulty"];
        break;
      case "--random-spawn":
        // Accept both `--random-spawn` (bare → true) and
        // `--random-spawn false` for a bit of CLI ergonomics.
        if (argv[i + 1] === "false" || argv[i + 1] === "true") {
          out.randomSpawn = argv[++i] === "true";
        } else {
          out.randomSpawn = true;
        }
        break;
      default:
        throw new Error(`unknown arg: ${key}`);
    }
  }
  return out;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  const config = {
    gameType: GameType.Private,
    gameMap: args.map as unknown as GameMapType,
    bots: args.bots,
    difficulty: args.difficulty as unknown as Difficulty,
    // Real field name on GameConfigSchema. "disabled" turns off NPCs, a
    // number sets an exact count (createNationsForGame in
    // src/core/game/NationCreation.ts), "default" uses the manifest count.
    nations: args.nations,
    infiniteGold: false,
    infiniteTroops: false,
    instantBuild: false,
    // Default: false → the bot (and any human in the lobby) picks their own
    // spawn tile via SpawnIntent. bot_client runs spawn_picker during the
    // ~20s manual spawn phase. --random-spawn restores the training-style
    // auto-placement where the NN never sees a SPAWN action.
    randomSpawn: args.randomSpawn,
  };

  const res = await fetch(`${args.host}/api/adminbot/create_game`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      [DEV_ADMIN_BOT_HEADER]: DEV_ADMIN_BOT_KEY,
    },
    body: JSON.stringify(config),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(
      `create_game failed: ${res.status} ${res.statusText} — ${text}`,
    );
  }

  const data = (await res.json()) as {
    gameID: string;
    workerIndex: number;
    workerPath: string;
  };

  // gameID → stdout (easy to capture in a shell var).
  // Context → stderr.
  process.stdout.write(`${data.gameID}\n`);
  process.stderr.write(
    `[create_game] created gameID=${data.gameID} worker=${data.workerIndex} path=${data.workerPath}\n`,
  );
  // Browser URL goes through the vite dev server (port 9000, which proxies
  // the WebSocket to the worker). The path shape is /game/<id> — the
  // client's Main.ts matches /^\/(?:w\d+\/)?game\/([^/]+)/.
  process.stderr.write(
    `[create_game] open in Chrome: http://localhost:9000/game/${data.gameID}\n`,
  );
}

main().catch((e) => {
  process.stderr.write(`[create_game] fatal: ${e}\n`);
  process.exit(1);
});
