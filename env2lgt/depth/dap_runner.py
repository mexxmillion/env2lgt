"""DAP depth backend — persistent inference daemon.

DAP ([Insta360-Research-Team/DAP](https://github.com/Insta360-Research-Team/DAP),
CVPR 2026) is a studio-approvable alternative to DA-2: MIT-licensed, DINOv3
backbone. It produces **metric** depth, so `is_metric = True` — `bake.py`
treats `scene_scale` as a fine-tune multiplier rather than the primary knob.

`DapBackend` mirrors `Da2Backend`'s daemon manager: one worker process per
instance, line-delimited JSON over stdin/stdout, lazy spawn, respawn on
crash. The daemon runs `scripts/dap_infer.py` inside the `env2lgt-dap`
conda env. Get an instance via `env2lgt.depth.get_backend("dap")`.

NOTE: this is integration scaffolding. The daemon/IPC/cache plumbing here is
complete and backend-agnostic, but the actual DAP model load + inference
lives in `scripts/dap_infer.py`, which has TODO markers for the wiring that
must be done against a real DAP checkout + weights.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from env2lgt.depth.da2_runner import _hash_file, _read_distance_exr

_DEFAULT_DAP_ENV_CANDIDATES = (
    r"E:\conda\envs\env2lgt-dap",
    r"C:\conda\envs\env2lgt-dap",
)
_DEFAULT_DAP_REPO_CANDIDATES = (
    r"E:\models\DAP",
    r"C:\models\DAP",
)
_DEFAULT_DAP_WEIGHTS_CANDIDATES = (
    r"E:\models\DAP-weights",
    r"C:\models\DAP-weights",
)


# ---------- env path discovery ----------

def _dap_python() -> Path:
    env = os.environ.get("ENV2LGT_DAP_ENV")
    if env:
        py = Path(env) / "python.exe"
        if py.exists():
            return py
        raise RuntimeError(f"ENV2LGT_DAP_ENV is set to {env} but no python.exe there.")
    for cand in _DEFAULT_DAP_ENV_CANDIDATES:
        py = Path(cand) / "python.exe"
        if py.exists():
            return py
    raise RuntimeError(
        "Could not locate the DAP conda env. Create it from environment-dap.yml, "
        "then set ENV2LGT_DAP_ENV to the env directory, or use one of: "
        f"{', '.join(_DEFAULT_DAP_ENV_CANDIDATES)}"
    )


def _dap_repo() -> Path:
    repo = os.environ.get("ENV2LGT_DAP_REPO")
    if repo and Path(repo).is_dir():
        return Path(repo)
    for cand in _DEFAULT_DAP_REPO_CANDIDATES:
        if Path(cand).is_dir():
            return Path(cand)
    raise RuntimeError(
        "Could not locate the DAP repo. Clone https://github.com/Insta360-Research-Team/DAP "
        "and set ENV2LGT_DAP_REPO, or clone to one of: "
        f"{', '.join(_DEFAULT_DAP_REPO_CANDIDATES)}"
    )


def _dap_weights() -> Path:
    """Locate the directory holding DAP's model.pth."""
    w = os.environ.get("ENV2LGT_DAP_WEIGHTS")
    if w and (Path(w) / "model.pth").is_file():
        return Path(w)
    for cand in _DEFAULT_DAP_WEIGHTS_CANDIDATES:
        if (Path(cand) / "model.pth").is_file():
            return Path(cand)
    raise RuntimeError(
        "Could not locate DAP weights (model.pth). Download from "
        "https://huggingface.co/Insta360-Research/DAP-weights and set "
        "ENV2LGT_DAP_WEIGHTS to the directory containing it, or place it at "
        f"one of: {', '.join(_DEFAULT_DAP_WEIGHTS_CANDIDATES)}"
    )


# ---------- backend ----------

class DapBackend:
    """DepthBackend backed by a persistent DAP inference daemon."""

    name = "dap"
    is_metric = True

    def __init__(self) -> None:
        self._daemon: subprocess.Popen | None = None
        self._daemon_lock = threading.Lock()
        self._stderr_drain_thread: threading.Thread | None = None

    # ----- daemon lifecycle -----

    @staticmethod
    def _drain_stderr(proc: subprocess.Popen) -> None:
        """Drain stderr in a background thread so the daemon doesn't block on a
        full pipe. Lines are echoed to our own stderr with a [dap-daemon] tag."""
        try:
            for line in proc.stderr:  # type: ignore[union-attr]
                if not line:
                    break
                sys.stderr.write(f"[dap-daemon] {line.rstrip()}\n")
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass

    def _spawn_daemon(self) -> subprocess.Popen:
        py = _dap_python()
        repo = _dap_repo()
        helper = Path(__file__).resolve().parents[2] / "scripts" / "dap_infer.py"
        if not helper.exists():
            raise RuntimeError(f"DAP helper script not found at {helper}")

        env = os.environ.copy()
        env["ENV2LGT_DAP_REPO"] = str(repo)
        env["ENV2LGT_DAP_WEIGHTS"] = str(_dap_weights())
        env.setdefault("PYTHONPATH", str(repo))
        env.setdefault("HF_HOME", os.environ.get("HF_HOME", r"E:\models\huggingface"))
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            [str(py), str(helper), "--serve", "--repo", str(repo)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        self._stderr_drain_thread = threading.Thread(
            target=self._drain_stderr, args=(proc,), daemon=True
        )
        self._stderr_drain_thread.start()

        ready_line = proc.stdout.readline()  # type: ignore[union-attr]
        if not ready_line:
            rc = proc.poll()
            raise RuntimeError(f"DAP daemon exited before signalling ready (rc={rc})")
        try:
            msg = json.loads(ready_line)
        except json.JSONDecodeError:
            raise RuntimeError(f"DAP daemon sent non-JSON line: {ready_line!r}")
        if not msg.get("ready"):
            raise RuntimeError(f"DAP daemon error: {msg}")
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
        """Run DAP on `exr_path`, return *metric* depth (H, W) float32.

        Cached by file hash next to the EXR (or in `cache_dir`). The cache
        key carries a `.dap.` tag so DA-2 and DAP results never collide.
        """
        exr_path = Path(exr_path).resolve()
        if not exr_path.is_file():
            raise FileNotFoundError(exr_path)
        file_hash = _hash_file(exr_path)
        cache_dir = Path(cache_dir) if cache_dir else exr_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{exr_path.stem}.{file_hash}.dap.distance.exr"

        if cache_path.exists():
            return _read_distance_exr(cache_path)

        daemon = self._get_daemon()
        req = json.dumps({"cmd": "infer", "input": str(exr_path), "output": str(cache_path)}) + "\n"
        daemon.stdin.write(req)  # type: ignore[union-attr]
        daemon.stdin.flush()  # type: ignore[union-attr]
        resp_line = daemon.stdout.readline()  # type: ignore[union-attr]
        if not resp_line:
            rc = daemon.poll()
            raise RuntimeError(f"DAP daemon exited mid-request (rc={rc})")
        resp = json.loads(resp_line)
        if not resp.get("ok"):
            raise RuntimeError(
                f"DAP inference failed: {resp.get('error')}\n{resp.get('trace', '')}"
            )
        return _read_distance_exr(cache_path)
