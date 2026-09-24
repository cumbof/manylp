"""Array backend: NumPy on the CPU, CuPy on a CUDA device.

The certification kernel is written once against the NumPy API and runs on
either.  Only regular, data-parallel work (GEMM, elementwise bound checks,
reductions) ever touches the device; irregular pivoting stays on the CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Backend:
    name: str          # "cpu" or "cuda"
    xp: Any            # numpy or cupy
    device_id: int = 0

    @property
    def is_gpu(self) -> bool:
        return self.name == "cuda"

    def asarray(self, a, dtype=None):
        if self.is_gpu:
            with self.xp.cuda.Device(self.device_id):
                return self.xp.asarray(a, dtype=dtype)
        return np.asarray(a, dtype=dtype)

    def to_host(self, a) -> np.ndarray:
        if self.is_gpu and not isinstance(a, np.ndarray):
            return self.xp.asnumpy(a)
        return np.asarray(a)

    def synchronize(self) -> None:
        if self.is_gpu:
            self.xp.cuda.Device(self.device_id).synchronize()

    def device_ctx(self):
        if self.is_gpu:
            return self.xp.cuda.Device(self.device_id)
        return _NullCtx()


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def get_backend(device: str = "cpu") -> Backend:
    """``"cpu"``, ``"cuda"`` / ``"gpu"`` (device 0) or ``"cuda:<id>"``."""
    device = device.lower()
    if device == "cpu":
        return Backend("cpu", np)
    if device in ("gpu", "cuda") or device.startswith("cuda:"):
        import cupy as cp

        dev_id = int(device.split(":", 1)[1]) if ":" in device else 0
        return Backend("cuda", cp, dev_id)
    raise ValueError(f"unknown device {device!r}")


def gpu_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False
