"""Execution controls and opt-in profiling; independent of numerical fit policy."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
import os
import time
import math


def available_cpu_ids():
    try:
        return tuple(sorted(os.sched_getaffinity(0)))
    except AttributeError:
        if os.name == "nt":
            import ctypes
            mask, system = ctypes.c_size_t(), ctypes.c_size_t()
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = ctypes.c_void_p
            kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
            if kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(mask), ctypes.byref(system)):
                return tuple(index for index in range(mask.value.bit_length()) if mask.value & (1 << index))
        return tuple(range(os.cpu_count() or 1))


def available_cpus():
    return len(available_cpu_ids())


@dataclass(frozen=True)
class ExecutionOptions:
    profile_fit: bool = False
    fit_workers: int = 1
    generator_workers: int = 1
    query_threads: int = 0
    fit_query_backend: str = "open3d"
    collision_sampling: str = "legacy"
    collision_max_edge_m: float = .05
    fit_objective: str = "collision-first"
    cpu_budget: int = 0
    auto_fit_workers: bool = False
    gpu_workers: int = 1
    penetration_weight: float = 1.0

    def __post_init__(self):
        available = available_cpus()
        for name in ("fit_workers", "generator_workers", "gpu_workers"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        if type(self.query_threads) is not int or self.query_threads < 0:
            raise ValueError("query_threads must be nonnegative")
        if type(self.cpu_budget) is not int or not 0 <= self.cpu_budget <= available:
            raise ValueError("cpu_budget exceeds available CPUs or is negative")
        if self.fit_query_backend not in ("open3d", "warp", "hybrid"):
            raise ValueError("Unknown fitting query backend")
        if self.collision_sampling not in ("legacy", "longest-edge"):
            raise ValueError("Unknown collision sampler")
        if not math.isfinite(self.collision_max_edge_m) or self.collision_max_edge_m <= 0:
            raise ValueError("collision_max_edge_m must be finite and positive")
        if self.fit_objective not in ("balanced", "collision-first"):
            raise ValueError("Unknown fitting objective")
        if not math.isfinite(self.penetration_weight) or self.penetration_weight <= 0:
            raise ValueError("penetration_weight must be finite and positive")
        if self.fit_query_backend == "warp" and self.fit_workers != 1:
            raise ValueError("The experimental Warp backend requires --fit-workers 1")
        if self.cpu_budget and self.fit_query_backend == "hybrid" and self.fit_workers > self.gpu_workers:
            raise ValueError("candidate_workers for hybrid cannot exceed execution.gpu_workers")
        budget = self.cpu_budget or available
        if self.fit_workers > budget:
            raise ValueError("fit_workers exceeds available CPU budget")
        if self.query_threads * self.fit_workers > budget:
            raise ValueError("fit_workers * query_threads exceeds available CPU budget")

    @classmethod
    def from_args(cls, args):
        return cls(**{name: getattr(args, name, field.default) for name, field in cls.__dataclass_fields__.items()})

    def resolved_query_threads(self):
        if self.query_threads:
            return self.query_threads
        if self.cpu_budget:
            return self.cpu_budget // self.fit_workers
        return 0 if self.fit_workers == 1 else max(1, available_cpus() // self.fit_workers)

    def for_candidates(self, pending_count):
        """Resolve the worker allocation once cache hits and invalid candidates are known."""
        if not self.cpu_budget or not pending_count:
            return self
        workers = self.fit_workers
        if self.auto_fit_workers:
            limit = {"open3d": 4, "hybrid": self.gpu_workers, "warp": 1}[self.fit_query_backend]
            workers = min(limit, max(1, self.cpu_budget // (self.query_threads or 2)))
        workers = min(workers, pending_count)
        return replace(self, fit_workers=workers, auto_fit_workers=False,
                       query_threads=self.query_threads or self.cpu_budget // workers)

    def record(self):
        fields = asdict(self)
        if not self.cpu_budget:
            for key in ("cpu_budget", "auto_fit_workers", "gpu_workers"):
                fields.pop(key)
        return dict(fields, actual_query_threads=self.resolved_query_threads(),
                    candidate_blas_threads=1 if self.fit_workers > 1 or self.cpu_budget else "automatic",
                    available_cpus=available_cpus())

    def argv(self):
        result = ["--" + name.replace("_", "-") for name in ("profile_fit",) if getattr(self, name)]
        for name in ("fit_workers", "generator_workers", "query_threads", "fit_query_backend",
                     "collision_sampling", "collision_max_edge_m", "fit_objective", "penetration_weight"):
            result.extend(["--" + name.replace("_", "-"), str(getattr(self, name))])
        if self.cpu_budget:
            result.extend(["--cpu-budget", str(self.cpu_budget), "--gpu-workers", str(self.gpu_workers)])
            if self.auto_fit_workers:
                result.append("--auto-fit-workers")
        return result


def add_execution_arguments(parser):
    parser.add_argument("--profile-fit", action="store_true", help="Write optional fitting performance records")
    parser.add_argument("--fit-workers", type=int, default=1)
    parser.add_argument("--generator-workers", type=int, default=1)
    parser.add_argument("--query-threads", type=int, default=0)
    parser.add_argument("--fit-query-backend", choices=["open3d", "warp", "hybrid"], default="open3d")
    parser.add_argument("--collision-sampling", choices=["legacy", "longest-edge"], default="legacy")
    parser.add_argument("--collision-max-edge-m", type=float, default=.05)
    parser.add_argument("--fit-objective", choices=["balanced", "collision-first"], default="collision-first")
    parser.add_argument("--penetration-weight", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--cpu-budget", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--gpu-workers", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--auto-fit-workers", action="store_true", help=argparse.SUPPRESS)


def peak_memory_bytes():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Memory(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in
                ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                 "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Memory(); counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32"); kernel.GetCurrentProcess.restype = wintypes.HANDLE
        api = ctypes.WinDLL("psapi").GetProcessMemoryInfo
        api.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
        return int(counters.PeakWorkingSetSize) if api(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb) else None
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if __import__('sys').platform == 'darwin' else 1024)
    except ImportError:
        return None


class Profile:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.counters = defaultdict(int)
        self.seconds = defaultdict(float)
        self.started = time.perf_counter()

    def count(self, name, value=1):
        if self.enabled:
            self.counters[name] += value

    @contextmanager
    def time(self, name):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] += time.perf_counter() - start

    def record(self):
        return dict(enabled=self.enabled, wall_seconds=time.perf_counter()-self.started,
                    seconds=dict(self.seconds), counters=dict(self.counters), peak_memory_bytes=peak_memory_bytes())


_options = ContextVar("chair_execution", default=ExecutionOptions())
_profile = ContextVar("chair_profile", default=None)

def current_options():
    return _options.get()

def current_profile():
    return _profile.get()

@contextmanager
def execution_scope(options, profile=None):
    a = _options.set(options); b = _profile.set(profile)
    try:
        yield
    finally:
        _options.reset(a); _profile.reset(b)

def count(name, value=1):
    profile = current_profile()
    if profile is not None:
        profile.count(name, value)
