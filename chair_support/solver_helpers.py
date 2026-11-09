"""Bounded exact-pose caches; optimization epochs never invalidate certificates."""
from collections import OrderedDict
from copy import deepcopy
import ntpath
import os
from pathlib import PurePosixPath, PureWindowsPath
import numpy as np


def pose_key(pose):
    value = np.ascontiguousarray(pose, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("Non-finite optimizer pose")
    return value.shape, value.tobytes()


def candidate_directory_key(name, windows=None):
    """Validate one directory component and detect Windows filesystem aliases."""
    windows = os.name == "nt" if windows is None else windows
    path_type = PureWindowsPath if windows else PurePosixPath
    if not name or name in (".", "..") or path_type(name).name != name:
        raise ValueError("Invalid candidate asset identifier")
    if windows:
        stem = name.split(".", 1)[0].upper().rstrip(" ")
        reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
        reserved.update(prefix + digit for prefix in ("COM", "LPT") for digit in "123456789¹²³")
        if (name.endswith((".", " ")) or stem in reserved
                or any(c in '<>:"/\\|?*' or ord(c) < 32 for c in name)):
            raise ValueError("Invalid Windows candidate asset identifier")
        return ntpath.normcase(name)
    return name


class PoseCache:
    def __init__(self, capacity=128):
        self.capacity = capacity
        self.epoch = 0
        self.values = OrderedDict()
        self.hits = self.misses = 0

    def invalidate(self):
        self.epoch += 1
        self.values.clear()

    def get(self, pose, namespace, compute, copy=False):
        key = self.epoch, namespace, pose_key(pose)
        if key in self.values:
            self.hits += 1
            value = self.values.pop(key)
        else:
            self.misses += 1
            value = compute()
        self.values[key] = value
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return deepcopy(value) if copy else value


def choose_current_result(choices, cost, violation):
    """Compare every historical solution under the *current* objective/constraints."""
    scored = []
    for result in choices:
        result.fun = float(cost(result.x))
        scored.append((result, float(violation(result.x))))
    feasible = [r for r, v in scored if v <= 1e-6]
    return min(feasible, key=lambda r: r.fun) if feasible else min(scored, key=lambda rv: (rv[1], rv[0].fun))[0]


def batched_support_metrics(samples, scene, rotation, translation, scale, query, metrics):
    """One ray batch for a region, restoring its original frame and sample order."""
    if not samples:
        return []
    points = np.concatenate([sample[2] for sample in samples])
    gaps, valid = query(points, scene, rotation, translation, scale)
    result, start = [], 0
    for frame, _, points, weights in samples:
        end = start + len(points)
        result.append((frame, metrics(gaps[start:end], valid[start:end], weights)))
        start = end
    return result
