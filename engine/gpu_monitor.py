from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import pynvml


@dataclass
class GPUMonitorSnapshot:
    samples: int
    gpu_util_mean: float
    gpu_util_max: float
    mem_util_mean: float
    mem_util_max: float
    mem_used_gb_mean: float
    mem_used_gb_max: float


class GPUMonitor:
    def __init__(self, *, device_index: int = 0, interval_seconds: float = 0.2) -> None:
        self.device_index = device_index
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpu_utils: list[float] = []
        self._mem_utils: list[float] = []
        self._mem_used_gb: list[float] = []
        self._handle = None
        self._enabled = False
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._enabled = True
        except Exception:
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        if not self._enabled:
            return
        self._stop_event.clear()
        self._gpu_utils.clear()
        self._mem_utils.clear()
        self._mem_used_gb.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> GPUMonitorSnapshot:
        if not self._enabled:
            return GPUMonitorSnapshot(
                samples=0,
                gpu_util_mean=0.0,
                gpu_util_max=0.0,
                mem_util_mean=0.0,
                mem_util_max=0.0,
                mem_used_gb_mean=0.0,
                mem_used_gb_max=0.0,
            )
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4))
            self._thread = None
        return self.snapshot()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self._gpu_utils.append(float(util.gpu))
                self._mem_utils.append(float(util.memory))
                self._mem_used_gb.append(float(mem.used) / (1024**3))
            except Exception:
                pass
            time.sleep(self.interval_seconds)

    def snapshot(self) -> GPUMonitorSnapshot:
        if not self._gpu_utils:
            return GPUMonitorSnapshot(
                samples=0,
                gpu_util_mean=0.0,
                gpu_util_max=0.0,
                mem_util_mean=0.0,
                mem_util_max=0.0,
                mem_used_gb_mean=0.0,
                mem_used_gb_max=0.0,
            )
        return GPUMonitorSnapshot(
            samples=len(self._gpu_utils),
            gpu_util_mean=sum(self._gpu_utils) / len(self._gpu_utils),
            gpu_util_max=max(self._gpu_utils),
            mem_util_mean=sum(self._mem_utils) / len(self._mem_utils),
            mem_util_max=max(self._mem_utils),
            mem_used_gb_mean=sum(self._mem_used_gb) / len(self._mem_used_gb),
            mem_used_gb_max=max(self._mem_used_gb),
        )
