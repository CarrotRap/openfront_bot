#!/usr/bin/env python
"""Live web viewer: watch the agent play, and see exactly what it sees.

    python viewer/server.py                       # masked-random policy
    python viewer/server.py --model runs/ppo.zip  # a trained MaskablePPO

Opens http://127.0.0.1:8420. A background thread drives one env and pushes a
frame per step over Server-Sent Events; the browser draws the map, outlines the
agent's observation window and renders the observation channels themselves.

Standard library only — no web framework to install.
"""

from __future__ import annotations

import argparse
import base64
import json
import queue
import sys
import threading
import time
import webbrowser
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openfront_gym import OpenFrontEnv, sample_masked_action  # noqa: E402
from openfront_gym.checkpoint import resolve_config  # noqa: E402

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------- PNG


def encode_png(indices: np.ndarray, palette: list[list[int]]) -> str:
    """8-bit palettised PNG as a data URL.

    The map is a few solid colours, so an indexed PNG is a couple of KB —
    far cheaper than shipping raw pixels, and the browser decodes it natively.
    """
    height, width = indices.shape
    plte = b"".join(bytes(c[:3]) for c in palette)

    raw = bytearray()
    for row in indices:
        raw.append(0)  # filter type 0 (None)
        raw.extend(row.tobytes())

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            len(data).to_bytes(4, "big")
            + tag
            + data
            + zlib.crc32(tag + data).to_bytes(4, "big")
        )

    ihdr = width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes([8, 3, 0, 0, 0])
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"PLTE", plte)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


# --------------------------------------------------------------------- loop


class Session:
    """Runs the environment on its own thread and fans frames out to browsers."""

    def __init__(
        self, config: dict[str, Any], model: Any | None, policy_name: str
    ) -> None:
        self.config = {**config, "emitRender": True}
        self.env = OpenFrontEnv(config=self.config)
        self.model = model
        self.policy_name = policy_name

        self.subscribers: list[queue.Queue] = []
        self.lock = threading.Lock()
        self.paused = False
        self.step_delay = 0.12
        self.reset_requested = True
        self.episode = 0
        self.latest: dict[str, Any] | None = None
        self._rng = np.random.default_rng()

        threading.Thread(target=self._run, daemon=True).start()

    # ---------------------------------------------------------- subscribers

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=4)
        with self.lock:
            self.subscribers.append(q)
        if self.latest is not None:
            q.put(self.latest)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def _publish(self, payload: dict[str, Any]) -> None:
        self.latest = payload
        with self.lock:
            targets = list(self.subscribers)
        for q in targets:
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass  # a slow browser drops frames rather than stalling the sim

    # ----------------------------------------------------------- simulation

    def _act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        if self.model is None:
            return sample_masked_action(self.env, self._rng)
        action, _ = self.model.predict(
            obs, action_masks=self.env.action_masks(), deterministic=False
        )
        return np.asarray(action).reshape(-1)

    def _run(self) -> None:
        obs: dict[str, np.ndarray] | None = None
        step = 0
        while True:
            if self.reset_requested or obs is None:
                self.reset_requested = False
                self.episode += 1
                step = 0
                obs, info = self.env.reset()
                self._emit(obs, None, 0.0, info, step, "reset")
                continue
            if self.paused:
                time.sleep(0.05)
                continue

            action = self._act(obs)
            label = self.env.describe_action(action)
            prev_masks = self.env.masks_by_head()
            obs, reward, terminated, truncated, info = self.env.step(action)
            step += 1
            self._emit(
                obs,
                action,
                reward,
                info,
                step,
                "running",
                label=label,
                masks=prev_masks,
            )
            if terminated or truncated:
                self._emit(
                    obs, action, reward, info, step,
                    "won" if info.get("won") else
                    ("dead" if not info.get("alive") else "truncated"),
                    label=label,
                )
                time.sleep(2.0)
                self.reset_requested = True
                continue
            time.sleep(self.step_delay)

    def _emit(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray | None,
        reward: float,
        info: dict[str, Any],
        step: int,
        status: str,
        label: str = "",
        masks: dict[str, np.ndarray] | None = None,
    ) -> None:
        render = self.env.last_render
        if render is None:
            return
        payload: dict[str, Any] = {
            "episode": self.episode,
            "step": step,
            "status": status,
            "policy": self.policy_name,
            "reward": round(float(reward), 4),
            "info": {k: v for k, v in info.items() if k != "error"},
            "action": None if action is None else [int(a) for a in action],
            "action_label": label,
            "map": encode_png(render["pixels"], render["palette"]),
            "mapW": render["width"],
            "mapH": render["height"],
            # Map tiles per rendered pixel (distinct from "step", the RL step).
            "mapStep": render["step"],
            "window": render["window"],
            "players": render["players"],
            "structures": render["structures"][:400],
            "channels": self.env.channels,
            # Quantised so the whole observation fits in a few KB of base64.
            "spatial": base64.b64encode(
                (np.clip(obs["spatial"], 0, 1) * 255).astype(np.uint8).tobytes()
            ).decode("ascii"),
            "spatialShape": list(obs["spatial"].shape),
            "minimapChannels": self.env.minimap_channels,
            "minimap": base64.b64encode(
                (np.clip(obs["minimap"], 0, 1) * 255).astype(np.uint8).tobytes()
            ).decode("ascii"),
            "minimapShape": list(obs["minimap"].shape),
            "scalarNames": self.env.scalar_names,
            "scalar": [round(float(v), 4) for v in obs["scalar"]],
            # Per-opponent features: raw (K, PF) tensor + the feature names so
            # the browser can label each column without duplicating the schema.
            "playerFeatures": self.env.player_features,
            "playersObs": [
                [round(float(v), 4) for v in row] for row in obs["players"]
            ],
            "maskCounts": (
                {k: int(v.sum()) for k, v in masks.items()} if masks else {}
            ),
            "config": self.config,
        }
        self._publish(payload)


# --------------------------------------------------------------------- HTTP


class QuietHTTPServer(ThreadingHTTPServer):
    """A browser closing an SSE stream is normal, not a server error."""

    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def make_handler(session: Session):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # quiet console
            pass

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = (HERE / "index.html").read_bytes()
                self._send(200, "text/html; charset=utf-8", body)
            elif self.path == "/events":
                self._stream()
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self) -> None:
            if self.path != "/control":
                self._send(404, "text/plain", b"not found")
                return
            length = int(self.headers.get("Content-Length", 0))
            cmd = json.loads(self.rfile.read(length) or b"{}")
            action = cmd.get("action")
            if action == "pause":
                session.paused = True
            elif action == "play":
                session.paused = False
            elif action == "step":
                session.paused = True
                session.reset_requested = False
            elif action == "reset":
                session.reset_requested = True
                session.paused = False
            elif action == "speed":
                session.step_delay = max(0.0, float(cmd.get("value", 0.12)))
            self._send(
                200,
                "application/json",
                json.dumps(
                    {"paused": session.paused, "delay": session.step_delay}
                ).encode(),
            )

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = session.subscribe()
            try:
                while True:
                    try:
                        payload = q.get(timeout=15)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    self.wfile.write(
                        b"data: " + json.dumps(payload).encode() + b"\n\n"
                    )
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                session.unsubscribe(q)

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None, help="path to a saved MaskablePPO .zip")
    # Everything below defaults to None: with --model, the checkpoint's sidecar
    # and observation space supply the values, and a flag only overrides them.
    p.add_argument("--map", default=None)
    p.add_argument("--map-source", default=None, choices=["testdata", "resources"])
    p.add_argument("--bots", type=int, default=None)
    p.add_argument("--nations", default=None)
    p.add_argument("--difficulty", default=None)
    p.add_argument("--grid", type=int, default=None)
    p.add_argument("--max-players", type=int, default=None, help="K opponent slots")
    p.add_argument("--window-min-tiles", type=int, default=None)
    p.add_argument("--window-zoom", type=float, default=None)
    p.add_argument(
        "--window-tiles",
        default=None,
        help='"adaptive" (zooms out as the agent grows), "auto" (whole map), '
        "or a tile count for a fixed window. Match whatever the model was "
        "trained with, or you shift its input distribution.",
    )
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--ticks-per-step", type=int, default=None)
    p.add_argument("--render-max-dim", type=int, default=384)
    p.add_argument("--port", type=int, default=8420)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    model, policy_name = None, "aléatoire (masqué)"
    if args.model:
        from sb3_contrib import MaskablePPO

        print(f"chargement du modèle {args.model} ...")
        model = MaskablePPO.load(args.model, device="cpu")
        policy_name = Path(args.model).name

    window = args.window_tiles
    overrides = {
        "map": args.map,
        "mapSource": args.map_source,
        "bots": args.bots,
        "nations": args.nations,
        "difficulty": args.difficulty,
        "grid": args.grid,
        "maxPlayers": args.max_players,
        "windowTiles": (
            None if window is None else
            window if window in ("auto", "adaptive") else int(window)
        ),
        "windowMinTiles": args.window_min_tiles,
        "windowZoom": args.window_zoom,
        "maxSteps": args.max_steps,
        "ticksPerStep": args.ticks_per_step,
    }
    defaults = {
        "map": "big_plains",
        "mapSource": "testdata",
        "bots": 6,
        "nations": "disabled",
        "difficulty": "Medium",
        "grid": 64,
        "maxPlayers": 8,
        "windowTiles": "adaptive",
        "windowMinTiles": 64,
        "windowZoom": 2.5,
        "maxSteps": 1200,
        "ticksPerStep": 10,
    }
    try:
        config, notes = resolve_config(args.model, model, overrides, defaults)
    except ValueError as exc:
        print(f"erreur : {exc}")
        raise SystemExit(2)
    for note in notes:
        print(f"  {note}")
    config["renderMaxDim"] = args.render_max_dim

    session = Session(config, model, policy_name)
    server = QuietHTTPServer(("127.0.0.1", args.port), make_handler(session))
    url = f"http://127.0.0.1:{args.port}"
    print(f"viewer -> {url}   (policy: {session.policy_name})")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        session.env.close()


if __name__ == "__main__":
    main()
