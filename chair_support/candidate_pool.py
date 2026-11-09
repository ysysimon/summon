"""Spawn-only candidate workers with bounded lifetime and mmap input reopening."""
from dataclasses import dataclass
from pathlib import Path
import multiprocessing as mp
import numpy as np


@dataclass(frozen=True)
class MappedArray:
    filename: str
    dtype: str
    shape: tuple
    offset: int
    order: str


def pack_arrays(value):
    if isinstance(value, np.memmap) and value.filename and value.flags.c_contiguous:
        # Views can have an offset beyond their original memmap's data pointer.
        base = value
        while isinstance(base.base, np.memmap):
            base = base.base
        offset = int(base.offset) + value.ctypes.data - base.ctypes.data
        return MappedArray(str(Path(value.filename).resolve()), value.dtype.str, value.shape, offset, "C")
    if isinstance(value, dict):
        return {k: pack_arrays(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(pack_arrays(v) for v in value)
    if isinstance(value, list):
        return [pack_arrays(v) for v in value]
    return value


def unpack_arrays(value):
    if isinstance(value, MappedArray):
        return np.memmap(value.filename, dtype=value.dtype, shape=value.shape,
                         offset=value.offset, order=value.order, mode="r")
    if isinstance(value, dict):
        return {k: unpack_arrays(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(unpack_arrays(v) for v in value)
    if isinstance(value, list):
        return [unpack_arrays(v) for v in value]
    return value


def ordered_spawn_map(function, tasks, workers, initializer=None, initargs=(), timeout=7200):
    """Yield original order; any failure/consumer cancellation terminates our pool only."""
    pool = mp.get_context("spawn").Pool(workers, initializer=initializer, initargs=initargs)
    complete = False
    try:
        pending = [pool.apply_async(function, (task,)) for task in tasks]
        pool.close()
        for task in pending:
            yield task.get(timeout=timeout)
        complete = True
    finally:
        if not complete:
            pool.terminate()
        pool.join()
