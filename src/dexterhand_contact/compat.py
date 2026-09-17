"""Compatibility shims required by legacy MANO/chumpy assets on Python 3.11+."""

from __future__ import annotations

import inspect
import numpy as np


def enable_legacy_mano_compatibility() -> None:
    """Restore aliases used by old ``chumpy`` releases before importing smplx."""
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    for name, value in {
        "bool": bool, "int": int, "float": float, "complex": complex,
        "object": object, "unicode": str, "str": str,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, value)
