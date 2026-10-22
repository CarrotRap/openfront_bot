/**
 * Force-start an admin-bot private lobby.
 *
 * Admin-bot lobbies are created with no creator (see AdminBotRoutes.ts:94),
 * so no human sees the Start button in the UI. The patched `force_start`
 * endpoint sets `startsAt` to now, matching what Start does for a host.
 *
 * Usage:
 *   tsx src/start_game.ts <gameID> [--host http://localhost:3001] [--delay 5]
 */

const DEV_ADMIN_BOT_KEY =
  "WARNING_DEV_ADMIN_BOT_KEY_DO_NOT_USE_IN_PRODUCTION";
const DEV_ADMIN_BOT_HEADER = "x-admin-bot-key";

type Args = { gameID: string; host: string; delaySec: number };

function parseArgs(argv: string[]): Args {
  if (argv.length === 0)
    throw new Error("usage: start_game.ts <gameID> [--host http://localhost:3001] [--delay 5]");
  const out: Args = {
    gameID: argv[0],
    host: "http://localhost:3001",
    delaySec: 5,
  };
  for (let i = 1; i < argv.length; i++) {
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
      case "--delay":
        out.delaySec = parseInt(val(), 10);
        break;
      default:
        throw new Error(`unknown arg: ${key}`);
    }
  }
  return out;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const res = await fetch(
    `${args.host}/api/adminbot/game/${args.gameID}/force_start`,
    {
      method: "POST",
      headers: {
        "content-type": "application/json",
        [DEV_ADMIN_BOT_HEADER]: DEV_ADMIN_BOT_KEY,
      },
      body: JSON.stringify({ delaySec: args.delaySec }),
    },
  );
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`force_start failed: ${res.status} — ${text}`);
  }
  process.stderr.write(
    `[start_game] game ${args.gameID} starting in ${args.delaySec}s\n`,
  );
}

main().catch((e) => {
  process.stderr.write(`[start_game] fatal: ${e}\n`);
  process.exit(1);
});
