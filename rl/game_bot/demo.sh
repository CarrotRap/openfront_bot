#!/usr/bin/env bash
# Launch the full bot-in-the-real-game demo in one command.
#
#   1. Creates a private game via the admin HTTP API
#   2. Starts the bot WebSocket client with the given model (background)
#   3. Opens Chrome on the game URL
#   4. Waits for ENTER so you can get into the browser lobby
#   5. Force-starts the game
#   6. Keeps the bot running until Ctrl-C, then cleans up
#
# Usage:
#   ./demo.sh --model ../runs/v5.zip
#   ./demo.sh --model ../runs/v5.zip --map Pangaea --bots 6 --nations default
#
# Prereqs: `cd ../OpenFrontIO && npm run dev` already running in another terminal.

set -euo pipefail

# ---- defaults -------------------------------------------------------------
MODEL=""
MAP="World"
BOTS=6
# NATIONS: a number sets an exact nation count, "default" uses all map-manifest
# nations (72 on World), "disabled" turns NPCs off entirely. Default here
# matches the latest trained curriculum step (v10/v11: nations=50). Previously
# "16" was passed but silently ignored by create_game and the server fell
# back to 72 — the real demo was wildly off from training.
NATIONS="50"
DIFFICULTY="Easy"
GRID=64
MAX_PLAYERS=8
TICKS=10
START_DELAY=3
# Sampling (not argmax). Mirrors how the policy acted during training —
# NOOP falls proportionally (≈its marginal probability), the rest is spread
# over real actions. Argmax collapses to NOOP on the huge joint action space
# unless NOOP is masked, so sampling is the honest default here.
DETERMINISTIC=0
HOST_HTTP="http://localhost:3001"
HOST_WS="ws://localhost:3001"
BROWSER_URL_BASE="http://localhost:9000"

# ---- argparse -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)        MODEL="$2"; shift 2 ;;
    --map)          MAP="$2"; shift 2 ;;
    --bots)         BOTS="$2"; shift 2 ;;
    --nations)      NATIONS="$2"; shift 2 ;;
    --difficulty)   DIFFICULTY="$2"; shift 2 ;;
    --grid)         GRID="$2"; shift 2 ;;
    --max-players)  MAX_PLAYERS="$2"; shift 2 ;;
    --ticks-per-step) TICKS="$2"; shift 2 ;;
    --start-delay)  START_DELAY="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,16p' "$0" | sed 's/^# //; s/^#//'
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "error: --model is required (e.g. --model ../runs/v5.zip)" >&2
  exit 2
fi
if [[ ! -f "$MODEL" ]]; then
  echo "error: model not found: $MODEL" >&2
  exit 2
fi

# ---- paths ----------------------------------------------------------------
HERE="$(cd "$(dirname "$0")" && pwd)"
TSX="$HERE/../../OpenFrontIO/node_modules/.bin/tsx"
if [[ ! -x "$TSX" ]]; then
  echo "error: tsx not found at $TSX — did you run \`npm run inst\` in OpenFrontIO?" >&2
  exit 2
fi
cd "$HERE"

# ---- cleanup on exit ------------------------------------------------------
BOT_PID=""
cleanup() {
  if [[ -n "$BOT_PID" ]] && kill -0 "$BOT_PID" 2>/dev/null; then
    echo
    echo "[demo] shutting down bot (pid=$BOT_PID)"
    kill "$BOT_PID" 2>/dev/null || true
    wait "$BOT_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# ---- 1. create game -------------------------------------------------------
echo "[demo] creating game — map=$MAP bots=$BOTS nations=$NATIONS difficulty=$DIFFICULTY"
GAME_ID=$("$TSX" src/create_game.ts \
  --host "$HOST_HTTP" \
  --map "$MAP" \
  --bots "$BOTS" \
  --nations "$NATIONS" \
  --difficulty "$DIFFICULTY" \
  --random-spawn true
  )

if [[ -z "$GAME_ID" ]]; then
  echo "[demo] create_game returned empty gameID; aborting" >&2
  exit 1
fi
echo "[demo] gameID=$GAME_ID"

# ---- 2. start bot in background -------------------------------------------
echo "[demo] starting bot client with model=$MODEL deterministic=$DETERMINISTIC"
BOT_FLAGS=()
[[ "$DETERMINISTIC" == "1" ]] && BOT_FLAGS+=(--deterministic)
# ${BOT_FLAGS[@]-} fallback avoids `set -u` crashing on an empty array (bash <4.4).
"$TSX" src/bot_client.ts "$GAME_ID" \
  --host "$HOST_WS" \
  --model "$MODEL" \
  --grid "$GRID" \
  --max-players "$MAX_PLAYERS" \
  --ticks-per-step "$TICKS" \
  ${BOT_FLAGS[@]+"${BOT_FLAGS[@]}"} &
BOT_PID=$!

# Small grace period so the join message lands before we open the browser.
sleep 1

# ---- 3. open browser ------------------------------------------------------
BROWSER_URL="$BROWSER_URL_BASE/game/$GAME_ID"
echo
echo "============================================================"
echo "   open in Chrome: $BROWSER_URL"
echo "============================================================"
if command -v open >/dev/null 2>&1; then
  open "$BROWSER_URL" || true
fi

# ---- 4. wait for user -----------------------------------------------------
echo
read -r -p "[demo] press ENTER once you are in the lobby > " _

# ---- 5. force start -------------------------------------------------------
echo "[demo] force-starting in $START_DELAY seconds..."
"$TSX" src/start_game.ts "$GAME_ID" \
  --host "$HOST_HTTP" \
  --delay "$START_DELAY"

# ---- 6. keep bot alive until user quits -----------------------------------
echo
echo "[demo] game is running. Ctrl-C to stop the bot and exit."
wait "$BOT_PID"
