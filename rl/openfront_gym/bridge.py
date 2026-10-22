"""Low-level transport to the Node simulation process.

Speaks the framing documented in ``rl/bridge/src/protocol.ts``: newline JSON
up, ``[uint32 jsonLen][uint32 binLen][json][payload]`` frames down. Arrays are
rebuilt with ``np.frombuffer`` so observation tensors never round-trip
through JSON.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

_DTYPES = {"float32": np.float32, "uint8": np.uint8, "int32": np.int32}

# rl/openfront_gym/bridge.py -> rl/ -> openfront_bot/
_RL_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _RL_ROOT.parent

DEFAULT_GAME_ROOT = _PROJECT_ROOT / "OpenFrontIO"
DEFAULT_ENTRY = _RL_ROOT / "bridge" / "src" / "main.ts"


class BridgeError(RuntimeError):
    """The Node process reported a failure, or died."""


def _resolve_tsx(game_root: Path) -> str:
    """tsx ships with the game's devDependencies; fall back to PATH."""
    local = game_root / "node_modules" / ".bin" / "tsx"
    if local.exists():
        return str(local)
    found = shutil.which("tsx")
    if found:
        return found
    raise BridgeError(
        f"tsx not found. Run `npm run inst` inside {game_root}, "
        "or install tsx globally."
    )


class NodeBridge:
    """One Node process running one headless OpenFront game."""

    def __init__(
        self,
        game_root: str | os.PathLike[str] | None = None,
        entry: str | os.PathLike[str] | None = None,
        node_memory_mb: int = 4096,
        stderr_to: int | None = None,
    ) -> None:
        self.game_root = Path(game_root or DEFAULT_GAME_ROOT).resolve()
        self.entry = Path(entry or DEFAULT_ENTRY).resolve()
        if not self.entry.exists():
            raise BridgeError(f"bridge entry point not found: {self.entry}")

        env = dict(os.environ)
        env["NODE_OPTIONS"] = f"--max-old-space-size={node_memory_mb}"
        self.proc = subprocess.Popen(
            [_resolve_tsx(self.game_root), str(self.entry)],
            cwd=str(self.game_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_to if stderr_to is not None else sys.stderr,
            env=env,
            bufsize=0,
        )

    # ---------------------------------------------------------------- I/O

    def _read_exactly(self, n: int) -> bytes:
        assert self.proc.stdout is not None
        chunks: list[bytes] = []
        remaining = n
        while remaining > 0:
            chunk = self.proc.stdout.read(remaining)
            if not chunk:
                code = self.proc.poll()
                raise BridgeError(
                    f"node process closed the stream (exit code {code}); "
                    "check its stderr output above"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def send(self, **command: Any) -> None:
        """Queues a command without waiting for its answer.

        Split from :meth:`recv` so a vectorised env can fan commands out to
        every Node process first and collect the answers afterwards — the
        simulations then run concurrently instead of one at a time.
        """
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write((json.dumps(command) + "\n").encode("utf8"))
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            raise BridgeError("node process is gone") from exc

    def recv(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Reads the answer to the command sent earlier."""
        json_len, bin_len = struct.unpack("<II", self._read_exactly(8))
        header = json.loads(self._read_exactly(json_len).decode("utf8"))
        payload = self._read_exactly(bin_len) if bin_len else b""

        if not header.get("ok", False):
            raise BridgeError(header.get("error", "unknown bridge error"))

        arrays: dict[str, np.ndarray] = {}
        for desc in header.get("arrays", []):
            dtype = _DTYPES[desc["dtype"]]
            start = desc["offset"]
            raw = payload[start : start + desc["nbytes"]]
            arrays[desc["name"]] = np.frombuffer(raw, dtype=dtype).reshape(
                desc["shape"]
            )
        return header, arrays

    def request(self, **command: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Sends one command and waits for its answer."""
        self.send(**command)
        return self.recv()

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(b'{"cmd":"close"}\n')
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
            self.proc.wait(timeout=5)

    def __del__(self) -> None:  # best effort; Gym calls close() explicitly
        try:
            self.close()
        except Exception:
            pass
