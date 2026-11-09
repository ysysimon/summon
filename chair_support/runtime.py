"""Execution controls and opt-in profiling; independent of numerical fit policy."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import os
import time


def available_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        if os.name == "nt":
            import ctypes
            mask, system = ctypes.c_size_t(), ctypes.c_size_t()
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = ctypes.c_void_p
            kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
            if kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(mask), ctypes.byref(system)):
                return bin(mask.value).count("1")
        return os.cpu_count() or 1


@dataclass(frozen=True)
class ExecutionOptions:
    profile_fit: bool = False
    fit_workers: int = 1
    generator_workers: int = 1
    query_threads: int = 0
    fit_query_backend: str = "open3d"

    def __post_init__(self):
        for name in ("fit_workers", "generator_workers"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        if type(self.query_threads) is not int or self.query_threads < 0:
            raise ValueError("query_threads must be nonnegative")
        if self.fit_query_backend not in ("open3d", "warp"):
            raise ValueError("Unknown fitting query backend")
        if self.fit_query_backend == "warp" and self.fit_workers != 1:
            raise ValueError("The experimental Warp backend requires --fit-workers 1")
        if self.fit_workers > available_cpus():
            raise ValueError("fit_workers exceeds available CPU budget")
        if self.query_threads and self.query_threads * self.fit_workers > available_cpus():
            raise ValueError("fit_workers * query_threads exceeds available CPU budget")

    @classmethod
    def from_args(cls, args):
        return cls(**{name: getattr(args, name, field.default) for name, field in cls.__dataclass_fields__.items()})

    def resolved_query_threads(self):
        if self.query_threads:
            return self.query_threads
        return 0 if self.fit_workers == 1 else max(1, available_cpus() // self.fit_workers)

    def record(self):
        return dict(asdict(self), actual_query_threads=self.resolved_query_threads(),
                    candidate_blas_threads=1 if self.fit_workers > 1 else "automatic", available_cpus=available_cpus())

    def argv(self):
        result = ["--" + name.replace("_", "-") for name in ("profile_fit",) if getattr(self, name)]
        for name in ("fit_workers", "generator_workers", "query_threads", "fit_query_backend"):
            result.extend(["--" + name.replace("_", "-"), str(getattr(self, name))])
        return result


def add_execution_arguments(parser):
    parser.add_argument("--profile-fit", action="store_true", help="Write optional fitting performance records")
    parser.add_argument("--fit-workers", type=int, default=1)
    parser.add_argument("--generator-workers", type=int, default=1)
    parser.add_argument("--query-threads", type=int, default=0)
    parser.add_argument("--fit-query-backend", choices=["open3d", "warp"], default="open3d")


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
