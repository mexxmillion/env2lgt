"""Depth-backend registry.

Pick a backend by name (``da2`` | ``dap``) or fall back to the
``ENV2LGT_DEPTH_BACKEND`` env var (default ``da2`` — don't break existing
installs). Backends are instantiated lazily and cached one-per-name, so the
DA² / DAP inference daemon is spawned at most once per process.
"""

from __future__ import annotations

import atexit
import os

from env2lgt.depth.base import DepthBackend

DEFAULT_BACKEND = "da2"
AVAILABLE_BACKENDS = ("da2", "dap")

_instances: dict[str, DepthBackend] = {}


def _resolve_name(name: str | None) -> str:
    if name is None:
        name = os.environ.get("ENV2LGT_DEPTH_BACKEND", DEFAULT_BACKEND)
    name = (name or DEFAULT_BACKEND).strip().lower()
    if name not in AVAILABLE_BACKENDS:
        raise ValueError(
            f"Unknown depth backend {name!r}. Expected one of: "
            f"{', '.join(AVAILABLE_BACKENDS)}"
        )
    return name


def get_backend(name: str | None = None) -> DepthBackend:
    """Return the (cached) depth backend for `name`.

    `name=None` consults `ENV2LGT_DEPTH_BACKEND`, then falls back to `da2`.
    """
    name = _resolve_name(name)
    inst = _instances.get(name)
    if inst is not None:
        return inst
    if name == "da2":
        from env2lgt.depth.da2_runner import Da2Backend

        inst = Da2Backend()
    elif name == "dap":
        from env2lgt.depth.dap_runner import DapBackend

        inst = DapBackend()
    else:  # pragma: no cover — _resolve_name already guards this
        raise ValueError(name)
    _instances[name] = inst
    return inst


def shutdown_all() -> None:
    """Shut down every instantiated backend. Idempotent; registered atexit."""
    for inst in list(_instances.values()):
        try:
            inst.shutdown()
        except Exception:  # noqa: BLE001
            pass


atexit.register(shutdown_all)
