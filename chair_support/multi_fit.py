"""Per-instance observation masks and hash-bound single-candidate checkpoints."""
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .selection import select_support


def valid_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", name):
        raise ValueError("Unsafe instance or asset identifier: " + repr(name))
    return name


def visit_masks(visits, nframes):
    """Map original frame indices to local visit rows; -1 means unassigned."""
    interaction = np.full(nframes, -1, dtype=int)
    support = np.full(nframes, -1, dtype=int)
    if not visits:
        raise ValueError("A fitted instance requires at least one visit")
    for i, visit in enumerate(visits):
        a, b = visit["start"], visit["end"]
        if not (type(a) is int and type(b) is int and 0 <= a < b <= nframes) or (interaction[a:b] >= 0).any():
            raise ValueError("Invalid or overlapping source-frame visits")
        interaction[a:b] = i
        for low, high in visit["support"]:
            if (not (type(low) is int and type(high) is int and a <= low < high <= b)
                    or (support[low:high] >= 0).any()):
                raise ValueError("Invalid source-frame support interval")
            support[low:high] = i
    return interaction, support


def instance_contacts(world, labels, indices, interaction_ids, identity, locations):
    """Assign each observed seat-class vertex only within its visit and its location's footprint.

    A hand can reach the idle neighboring chair. At a Voronoi boundary its
    owner is unresolved and neither chair may use it as a fitting target.
    """
    contact = np.isin(labels, [3, 4]) & (interaction_ids[indices, None] >= 0)
    if locations is None:
        return contact
    own = np.asarray(locations[identity], float)
    others = [np.asarray(xy, float) for key, xy in locations.items() if key != identity]
    if any(a.shape != (2,) or not np.isfinite(a).all() for a in [own] + others):
        raise ValueError("Invalid inferred seat anchor")
    if not others:
        return contact
    for row, frame in enumerate(indices):
        vertices = np.flatnonzero(contact[row])
        if not len(vertices):
            continue
        xy = world[frame, vertices, :2]
        distance = np.linalg.norm(xy - own, axis=1)
        nearest_other = np.min([np.linalg.norm(xy - other, axis=1) for other in others], axis=0)
        # 2 cm matches the support sampling grid. Do not resolve sub-cell ties.
        contact[row, vertices] = distance + .02 < nearest_other
    return contact


def retain_return_visits(visits, selected, valid, support_ids, anchors, weights, pelvis, forward, limits, penetration):
    """Estimation can use a common plane, but may not silently eliminate a return visit."""
    chosen = set(map(int, selected))
    evidence = []
    for index, visit in enumerate(visits):
        frames = valid[support_ids[valid] == index]
        active = [int(f) for f in frames if f in chosen]
        fallback = bool(len(frames) and not active)
        if fallback:
            samples = {key: [s for s in value if support_ids[s[0]] == index] for key, value in anchors.items()}
            local, _ = select_support(frames, weights, samples, pelvis, forward, limits, penetration)
            active = list(map(int, local))
            chosen.update(active)
        evidence.append(dict(start=visit["start"], end=visit["end"], support=visit["support"],
                             selected_source_frames=active, visit_local_proxy=fallback,
                             status="support_evidence" if active else "support_unverified"))
    return np.asarray(sorted(chosen), int), evidence


def evaluate_visits(visits, measurements, policy=None):
    from .policy import FitPolicy
    policy = policy or FitPolicy()
    if not (policy.support_validation or policy.penetration_validation):
        return [dict(start=v["start"],end=v["end"],status="skipped_disabled",checked_source_frames=0,verified=False,rejection_reasons=[]) for v in visits]
    from .engine import support_checks

    by_region = {region: {row["frame"]: row for row in rows} for region, rows in measurements.items()}
    result = []
    for visit in visits:
        frames = visit["selected_source_frames"]
        reasons = set() if frames else {"missing_source_frames"}
        for frame in frames:
            if any(frame not in rows for rows in by_region.values()):
                reasons.add("missing_source_frames")
                continue
            reasons.update(support_checks({region: rows[frame] for region, rows in by_region.items()}, policy))
        result.append(dict(start=visit["start"], end=visit["end"], checked_source_frames=len(frames),
                           verified=not reasons, rejection_reasons=sorted(reasons)))
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class CandidateCache:
    """Publish only completed immutable directories. Bind a receipt to every byte restored."""
    def __init__(self, root, input_key, solver_key, visits, phase):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.key = digest([input_key, solver_key, visits, phase])

    def lookup(self, asset, output):
        from .engine import resolve, sha256

        meta = json.loads(asset.read_text(encoding="utf-8"))
        identity = valid_name(meta.get("id", meta.get("asset_id", asset.parent.name)))
        if meta.get("annotation_error"):
            return None, None, identity
        dependencies = [sha256(asset)]
        for key, default in (("mesh", "mesh.obj"), ("parts", "parts.npz")):
            dependencies.append(sha256(resolve(asset.parent, meta.get(key, default))))
        cache = self.root / digest([self.key, identity, dependencies])
        receipt_path = cache / "receipt.json"
        if not receipt_path.is_file():
            return None, cache, identity
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        bundle = cache / "candidate"
        if receipt["key"] != cache.name or receipt["files"] != self.files(bundle):
            raise ValueError("Corrupted multi-chair candidate cache: " + str(cache))
        destination = output / "candidates" / identity
        shutil.copytree(bundle, destination)
        record_path = destination / "result.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["id"] != identity:
            raise ValueError("Cached asset identifier mismatch")
        record["source_asset"] = str(asset.resolve())
        from .engine import write_json
        write_json(record_path, record)
        return record, cache, identity

    @staticmethod
    def files(folder):
        from .engine import sha256
        return {p.relative_to(folder).as_posix(): sha256(p) for p in sorted(folder.rglob("*")) if p.is_file()}

    def publish(self, cache, directory):
        if cache is None or cache.is_dir():
            return
        with tempfile.TemporaryDirectory(dir=self.root) as tmp:
            stage = Path(tmp) / "complete"
            bundle = stage / "candidate"
            shutil.copytree(directory, bundle)
            files = self.files(bundle)
            if "result.json" not in files:
                raise ValueError("Incomplete multi-chair candidate")
            (stage / "receipt.json").write_text(json.dumps(dict(key=cache.name, files=files)), encoding="utf-8")
            try:
                stage.rename(cache)
            except OSError:
                # Another process can finish this exact input concurrently.
                if not cache.is_dir():
                    raise
