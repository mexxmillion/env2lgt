# Copyright 2024-2026 Maung Maung Hla Win <mexxmillion@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""DA-2 depth backend — persistent inference daemon.

DA-2 inference itself is ~150ms on a 3090, but starting a fresh Python +
loading the model costs ~13 seconds. To avoid that on every bake, we spawn
*one* worker process per backend instance and reuse it via line-delimited
JSON over stdin/stdout.

`Da2Backend` implements the `DepthBackend` protocol (`env2lgt.depth.base`).
DA-2 produces **scale-invariant** depth, so `is_metric = False` — `bake.py`
keeps `scene_scale` as the primary meters-per-unit knob.

The daemon is started lazily on the first `estimate_depth` call. If it
crashes, the next call respawns it. Get an instance via
`env2lgt.depth.get_backend("da2")`.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

import OpenImageIO as oiio  # type: ignore[import-not-found]


_DEFAULT_DA2_ENV_CANDIDATES = (
    r"E:\conda\envs\env2lgt-da2",
    r"C:\conda\envs\env2lgt-da2",
)
_DEFAULT_DA2_REPO_CANDIDATES = (
    r"E:\models\DA-2",
    r"C:\models\DA-2",
)


# ---------- env path discovery ----------

def _da2_python() -> Path:
    env = os.environ.get("ENV2LGT_DA2_ENV")
    if env:
        py = Path(env) / "python.exe"
        if py.exists():
            return py
        raise RuntimeError(f"ENV2LGT_DA2_ENV is set to {env} but no python.exe there.")
    for cand in _DEFAULT_DA2_ENV_CANDIDATES:
        py = Path(cand) / "python.exe"
        if py.exists():
            return py
    raise RuntimeError(
        "Could not locate the DA-2 conda env. Set ENV2LGT_DA2_ENV to the env directory, "
        f"or create it at one of: {', '.join(_DEFAULT_DA2_ENV_CANDIDATES)}"
    )


def _da2_repo() -> Path:
    repo = os.environ.get("ENV2LGT_DA2_REPO")
    if repo and Path(repo).is_dir():
        return Path(repo)
    for cand in _DEFAULT_DA2_REPO_CANDIDATES:
        if Path(cand).is_dir():
            return Path(cand)
    raise RuntimeError(
        "Could not locate the DA-2 repo (configs/infer.json). Set ENV2LGT_DA2_REPO or "
        f"clone the repo to one of: {', '.join(_DEFAULT_DA2_REPO_CANDIDATES)}"
    )


# ---------- shared helpers ----------

def _hash_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()[:16]


def _read_distance_exr(path: Path) -> np.ndarray:
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise IOError(f"Could not read DA-2 output {path}: {oiio.geterror()}")
    try:
        spec = inp.spec()
        pixels = inp.read_image(format="float")
    finally:
        inp.close()
    arr = np.asarray(pixels, dtype=np.float32).reshape(spec.height, spec.width, spec.nchannels)
    return arr[..., 0] if arr.shape[-1] > 1 else arr.squeeze(-1)


# ---------- backend ----------

class Da2Backend:
    """DepthBackend backed by a persistent DA-2 inference daemon."""

    name = "da2"
    is_metric = False

    def __init__(self) -> None:
        self._daemon: subprocess.Popen | None = None
        self._daemon_lock = threading.Lock()
        self._stderr_drain_thread: threading.Thread | None = None

    # ----- daemon lifecycle -----

    @staticmethod
    def _drain_stderr(proc: subprocess.Popen) -> None:
        """Drain stderr in a background thread so the daemon doesn't block on a
        full pipe. Lines are echoed to our own stderr with a [da2-daemon] tag."""
        try:
            for line in proc.stderr:  # type: ignore[union-attr]
                if not line:
                    break
                sys.stderr.write(f"[da2-daemon] {line.rstrip()}\n")
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass

    def _spawn_daemon(self) -> subprocess.Popen:
        py = _da2_python()
        repo = _da2_repo()
        config_path = repo / "configs" / "infer.json"
        helper = Path(__file__).resolve().parents[2] / "scripts" / "da2_infer.py"
        if not helper.exists():
            raise RuntimeError(f"DA-2 helper script not found at {helper}")

        env = os.environ.copy()
        env["ENV2LGT_DA2_REPO"] = str(repo)
        env.setdefault("PYTHONPATH", str(repo / "src"))
        env.setdefault("HF_HOME", os.environ.get("HF_HOME", r"E:\models\huggingface"))
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            [str(py), str(helper), "--serve", "--config", str(config_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        # Drain stderr so the daemon never blocks on a full pipe.
        self._stderr_drain_thread = threading.Thread(
            target=self._drain_stderr, args=(proc,), daemon=True
        )
        self._stderr_drain_thread.start()

        # Wait for the {"ready": true} line.
        ready_line = proc.stdout.readline()  # type: ignore[union-attr]
        if not ready_line:
            rc = proc.poll()
            raise RuntimeError(f"DA-2 daemon exited before signalling ready (rc={rc})")
        try:
            msg = json.loads(ready_line)
        except json.JSONDecodeError:
            raise RuntimeError(f"DA-2 daemon sent non-JSON line: {ready_line!r}")
        if not msg.get("ready"):
            raise RuntimeError(f"DA-2 daemon error: {msg}")
        return proc

    def _get_daemon(self) -> subprocess.Popen:
        with self._daemon_lock:
            if self._daemon is not None and self._daemon.poll() is None:
                return self._daemon
            self._daemon = self._spawn_daemon()
            return self._daemon

    def shutdown(self) -> None:
        """Politely shut the daemon down. Idempotent."""
        with self._daemon_lock:
            if self._daemon is None or self._daemon.poll() is not None:
                self._daemon = None
                return
            try:
                self._daemon.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")  # type: ignore[union-attr]
                self._daemon.stdin.flush()  # type: ignore[union-attr]
                self._daemon.wait(timeout=5)
            except Exception:  # noqa: BLE001
                try:
                    self._daemon.kill()
                except Exception:  # noqa: BLE001
                    pass
            self._daemon = None

    def is_running(self) -> bool:
        return self._daemon is not None and self._daemon.poll() is None

    # ----- inference -----

    def estimate_depth(
        self, exr_path: str | Path, cache_dir: str | Path | None = None
    ) -> np.ndarray:
        """Run DA-2 on `exr_path`, return distance (H, W) float32.

        Caches the result by file hash next to the EXR (or in `cache_dir`).
        Persists the daemon across calls — first call pays ~14s cold start,
        later calls are ~200ms total.
        """
        exr_path = Path(exr_path).resolve()
        if not exr_path.is_file():
            raise FileNotFoundError(exr_path)
        file_hash = _hash_file(exr_path)
        cache_dir = Path(cache_dir) if cache_dir else exr_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{exr_path.stem}.{file_hash}.distance.exr"

        if cache_path.exists():
            return _read_distance_exr(cache_path)

        daemon = self._get_daemon()
        req = json.dumps({"cmd": "infer", "input": str(exr_path), "output": str(cache_path)}) + "\n"
        daemon.stdin.write(req)  # type: ignore[union-attr]
        daemon.stdin.flush()  # type: ignore[union-attr]
        resp_line = daemon.stdout.readline()  # type: ignore[union-attr]
        if not resp_line:
            rc = daemon.poll()
            raise RuntimeError(f"DA-2 daemon exited mid-request (rc={rc})")
        resp = json.loads(resp_line)
        if not resp.get("ok"):
            raise RuntimeError(
                f"DA-2 inference failed: {resp.get('error')}\n{resp.get('trace', '')}"
            )
        return _read_distance_exr(cache_path)
