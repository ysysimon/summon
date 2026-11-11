"""Recognize visits to fixed seating locations using motion and contact predictions only.

All intervals are half-open on the original source-frame grid. Instance IDs describe
the independent-chair hypothesis; a shared support surface may explain nearby visits.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree

from .engine import (
    DEFAULTS,
    REGIONS,
    body_regions,
    collect_support_samples,
    contiguous_runs,
    resolve,
    sha256,
    unit,
    write_json,
)
from .selection import weighted_quantile

RADIUS = 0.20
MAX_HEIGHT_DIFFERENCE = 0.10
MIN_ALIGNMENT = np.cos(np.radians(60))
CONTACT_HZ = 15.0


def _cloud(points):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return cloud


def original_clusters(world, labels, indices):
    """The two voxel/DBSCAN passes in SUMMON fit_best_obj, before furniture fitting.

    SUMMON pools the points across time. To measure a temporal baseline we project
    its retained voxel clusters back to real prediction samples by nearest voxel.
    """
    import config as original
    from utils import pred_subset_to_mpcat40

    mapped = pred_subset_to_mpcat40[labels]
    samples = world[indices]
    pieces, classes = [], []
    for category in original.classes_eps:
        pts = samples[mapped == category]
        if len(pts):
            voxels = np.asarray(_cloud(pts).voxel_down_sample(original.voxel_size).points)
            pieces.append(voxels)
            classes.append(np.full(len(voxels), category, int))
    objects = []
    members = []
    if pieces:
        pts, cls = np.concatenate(pieces), np.concatenate(classes)
        clusters = np.asarray(_cloud(pts).cluster_dbscan(original.voting_eps, original.cluster_min_points))
        votes = np.full(len(pts), -1)
        for group in np.unique(clusters[clusters >= 0]):
            mask = clusters == group
            votes[mask] = np.bincount(cls[mask]).argmax()
        for category in (3, 10):
            if np.any(votes == category):
                cloud = _cloud(pts[votes == category]).voxel_down_sample(original.voxel_size)
                voxels = np.asarray(cloud.points)
                clusters = np.asarray(
                    cloud.cluster_dbscan(original.classes_eps[category], original.cluster_min_points)
                )
                for group in np.unique(clusters[clusters >= 0]):
                    part = voxels[clusters == group]
                    objects.append(
                        dict(
                            instance_id=f"cluster-{len(objects) + 1}",
                            mpcat40=category,
                            center_m=np.mean(part, axis=0).tolist(),
                            voxel_count=len(part),
                        )
                    )
                    members.append(part)
    assignments = np.full(len(indices), -1, int)
    if objects:
        tree = cKDTree(np.concatenate(members))
        ids = np.repeat(np.arange(len(members)), [len(p) for p in members])
        for row, verts in enumerate(samples):
            pts = verts[np.isin(labels[row], (3, 4))]
            if len(pts):
                dist, near = tree.query(pts)
                hits = ids[near[dist <= original.voxel_size * np.sqrt(3)]]
                if len(hits):
                    assignments[row] = np.bincount(hits).argmax()
    # Nearest real sample on the integer source grid. Retain empty/no-contact samples.
    midpoints = (indices[:-1] + indices[1:]) / 2
    grid = assignments[np.searchsorted(midpoints, np.arange(len(world)))]
    edges = np.r_[0, np.flatnonzero(np.diff(grid)) + 1, len(grid)].astype(int)
    events = []
    for lo, hi in zip(edges[:-1].tolist(), edges[1:].tolist()):
        if grid[lo] >= 0:
            events.append(
                dict(instance_id=objects[grid[lo]]["instance_id"], start=lo, end=hi, support=[[lo, hi]])
            )
    return dict(
        method="SUMMON two-pass 3D voxel DBSCAN; nearest retained voxel temporal attribution",
        clusters=objects,
        spatial_cluster_count=len(objects),
        source_sample_instance_ids=[objects[i]["instance_id"] if i >= 0 else None for i in assignments],
        events=events,
    )


def _kinematics(joints, names, fps):
    j = {
        name: joints[:, names.index(name)]
        for name in ("Hips", "LeftLeg", "RightLeg", "LeftShin", "RightShin", "LeftFoot", "RightFoot")
    }
    pelvis = j["Hips"]
    right = unit((j["RightLeg"] - j["LeftLeg"]) * [1, 1, 0])
    forward = np.cross([0.0, 0.0, 1.0], right)
    knees_ahead = (j["LeftShin"] + j["RightShin"] - j["RightLeg"] - j["LeftLeg"]) * 0.5
    forward *= np.where((forward * knees_ahead).sum(axis=1) >= 0, 1, -1)[:, None]
    thigh_angles, knee_angles = [], []
    for side in ("Left", "Right"):
        thigh = unit(j[side + "Shin"] - j[side + "Leg"])
        shin = unit(j[side + "Foot"] - j[side + "Shin"])
        knee_angles.append(np.degrees(np.arccos(np.clip(np.sum(-thigh * shin, axis=1), -1, 1))))
        thigh_angles.append(np.degrees(np.arcsin(np.clip(np.abs(thigh[:, 2]), 0, 1))))
    smoothed = uniform_filter1d(pelvis, max(1, round(fps * 0.10)), axis=0, mode="nearest")
    speed = np.gradient(smoothed, axis=0) * fps if len(joints) > 1 else np.zeros_like(smoothed)
    return pelvis, forward[:, :2], np.asarray(thigh_angles), np.asarray(knee_angles), speed


def _visits(dense, faces, joints, names, labels, indices, fps, seat_labels=(3, 4)):
    pelvis, forward, thigh, knee, speed = _kinematics(joints, names, fps)
    times = np.arange(len(pelvis))
    seat_counts = np.isin(labels, seat_labels).sum(axis=1)
    contacts = np.interp(times, indices, seat_counts) >= DEFAULTS["min_chair_vertices"]
    thigh_max = thigh.max(axis=0)
    bent = (knee.min(axis=0) >= DEFAULTS["min_knee_degrees"]) & (
        knee.max(axis=0) <= DEFAULTS["max_knee_degrees"]
    )
    sitting = (thigh_max <= DEFAULTS["thigh_angle_degrees"]) & bent
    stable = (np.abs(speed[:, 2]) <= 0.22) & (np.linalg.norm(speed[:, :2], axis=1) <= 0.32)
    plausible = [
        [int(a), int(b)]
        for a, b in contiguous_runs(sitting & stable)
        if b - a >= max(1, round(0.12 * fps)) or a == 0 or b == len(pelvis)
    ]
    core = sitting & stable & contacts
    # Only bridge brief evidence dropouts while the same seated posture persists.
    for a, b in contiguous_runs(~core):
        if 0 < a and b < len(core) and b - a <= round(0.15 * fps) and np.all(sitting[a:b] & stable[a:b]):
            core[a:b] = True
    runs = [(int(a), int(b)) for a, b in contiguous_runs(core) if b - a >= max(1, round(fps * 0.12))]
    if not runs:
        return [], plausible
    # A person can stay low while sliding between positions. Keep stationary
    # plateaus on either side; do not periodically cut a sliding path into seats.
    bounded = []
    for a, b in runs:
        if np.linalg.norm(np.ptp(pelvis[a:b, :2], axis=0)) <= RADIUS:
            bounded.append((a, b))
            continue
        for lo, hi in contiguous_runs(np.linalg.norm(speed[a:b, :2], axis=1) <= 0.08):
            if (
                hi - lo >= max(1, round(0.12 * fps))
                and np.linalg.norm(np.ptp(pelvis[a + lo : a + hi, :2], axis=0)) <= RADIUS
            ):
                bounded.append((a + int(lo), a + int(hi)))
    ctx = dict(dense=dense, faces=faces)
    regions = body_regions(ctx)
    step = max(1, round(fps / CONTACT_HZ))
    sampled = [np.unique(np.r_[np.arange(a, b, step), b - 1]) for a, b in bounded]
    if not sampled:
        return [], plausible
    anchors, _ = collect_support_samples(ctx, regions, np.unique(np.concatenate(sampled)))
    observations = {}
    for row in range(len(anchors[REGIONS[0]])):
        points = []
        for region in REGIONS:
            frame, _, verts, weights = anchors[region][row]
            points.append(
                [
                    weighted_quantile(verts[:, 0], weights, 0.5),
                    weighted_quantile(verts[:, 1], weights, 0.5),
                    weighted_quantile(verts[:, 2], weights, 0.2),
                ]
            )
        observations[int(frame)] = points
    visits = []
    for (a, b), sample in zip(bounded, sampled):
        good = [int(f) for f in sample if int(f) in observations]
        if not good:
            continue
        geometry = np.asarray([observations[f] for f in good])
        # Even a slow descent can meet the instantaneous speed criterion. A fixed
        # seat also requires the undersides of both buttocks to settle at a
        # consistent height. Use the existing support gap budget independently
        # for each side rather than tuning an absolute pelvis/seat height.
        glute_z = geometry[:, :2, 2]
        limit = np.quantile(glute_z, 0.20, axis=0) + DEFAULTS["gluteus_gap"]
        timeline = np.arange(a, b)
        envelope = np.column_stack([np.interp(timeline, good, side) for side in glute_z.T])
        # Preserve a gap when downward-facing anatomy samples were unavailable.
        measured_neighbor = np.interp(timeline, sample, np.isin(sample, good).astype(float)) > 0.5
        settled = np.all(envelope <= limit, axis=1) & measured_neighbor
        support = [[int(lo + a), int(hi + a)] for lo, hi in contiguous_runs(settled)]
        measured = np.all(glute_z <= limit, axis=1)
        if not support or not np.any(measured):
            continue
        anatomy = np.median(geometry[measured], axis=0)
        center = np.median(anatomy, axis=0)
        visits.append(
            dict(
                start=a,
                end=b,
                support=support,
                anchor_xy_m=center[:2].tolist(),
                height_m=float(center[2]),
                forward_xy=unit(np.median(forward[a:b], axis=0)).tolist(),
                footprint_xy_m=anatomy[:, :2].tolist(),
                anchor_samples=len(good),
            )
        )
    if not visits:
        return [], plausible
    # Two stable intervals interrupted by a small adjustment at the same seat
    # remain one visit unless the knees extend, the body rises or travels away.
    merged = []
    for v in visits:
        earlier = merged[-1] if merged else None
        if earlier:
            a, b = earlier["end"], v["start"]
            nearby = _compatible(v, earlier)
            upright = np.any(thigh_max[a:b] > 60) or np.any(knee.max(axis=0)[a:b] > 145)
            excursion = np.any(np.linalg.norm(pelvis[a:b, :2] - pelvis[a - 1, :2], axis=1) > RADIUS)
            rise = np.any(pelvis[a:b, 2] > min(pelvis[a - 1, 2], pelvis[b, 2]) + 0.10)
            if nearby and not upright and not excursion and not rise:
                earlier["end"] = v["end"]
                earlier["support"].extend(v["support"])
                earlier["anchor_samples"] += v["anchor_samples"]
                continue
        merged.append(v)
    # Expand only the contiguous descent and rise next to each stable visit.
    # The middle of a gap bounds both searches so events cannot overlap.
    support_bounds = [(v["start"], v["end"]) for v in merged]
    for i, v in enumerate(merged):
        a, b = support_bounds[i]
        left = 0 if not i else (support_bounds[i - 1][1] + a) // 2
        right = len(pelvis) if i + 1 == len(merged) else (b + support_bounds[i + 1][0]) // 2
        near = np.linalg.norm(pelvis[:, :2] - np.median(pelvis[a:b, :2], axis=0), axis=1) <= 0.40
        descent = near & ((thigh_max <= 70) | ((speed[:, 2] < -0.055) & (knee.max(axis=0) < 172)))
        rise = near & ((thigh_max <= 70) | ((speed[:, 2] > 0.055) & (knee.max(axis=0) < 172)))
        while a > left and descent[a - 1]:
            a -= 1
        while b < right and rise[b]:
            b += 1
        v["start"], v["end"] = a, b
    # A low transfer can keep sitting/stability gates active throughout. Evidence
    # for the first position cannot verify another stationary position in the
    # same temporal run, even when its contact labels are missing completely.
    supports = [pair for visit in merged for pair in visit["support"]]
    unverified = []
    for a, b in plausible:
        observed = [(max(a, lo), min(b, hi)) for lo, hi in supports if lo < b and hi > a]
        if not observed:
            unverified.append([a, b])
            continue
        covered = np.zeros(b - a, bool)
        for lo, hi in observed:
            center = np.median(pelvis[lo:hi], axis=0)
            heading = unit(np.median(forward[lo:hi], axis=0))
            offset = pelvis[a:b] - center
            covered |= (
                (np.linalg.norm(offset[:, :2], axis=1) <= RADIUS)
                & (np.abs(offset[:, 2]) <= MAX_HEIGHT_DIFFERENCE)
                & ((forward[a:b] @ heading) >= MIN_ALIGNMENT)
            )
            covered[lo - a : hi - a] = True
        resting = np.linalg.norm(speed[a:b, :2], axis=1) <= 0.08
        for lo, hi in contiguous_runs(resting & ~covered):
            if hi - lo >= max(1, round(0.12 * fps)) or a + lo == 0 or a + hi == len(pelvis):
                unverified.append([int(a + lo), int(a + hi)])
    return merged, unverified


def _compatible(a, b):
    offset = np.subtract(a["anchor_xy_m"], b["anchor_xy_m"])
    aligned = np.dot(a["forward_xy"], b["forward_xy"]) >= MIN_ALIGNMENT
    footprint = np.linalg.norm(np.subtract(a["footprint_xy_m"], b["footprint_xy_m"]), axis=1)
    return (
        np.linalg.norm(offset) <= RADIUS
        and abs(a["height_m"] - b["height_m"]) <= MAX_HEIGHT_DIFFERENCE
        and aligned
        and np.max(footprint) <= 1.5 * RADIUS
    )


def assign_locations(visits):
    if not visits:
        return [], []
    coords = np.asarray([v["anchor_xy_m"] + [0.0] for v in visits])
    # Each visit gets one median observation. DBSCAN's single-link chains are
    # refined with complete pair compatibility so an intermediate visit cannot
    # merge distinct stationary seats. Single visits must remain eligible.
    proposals = np.asarray(_cloud(coords).cluster_dbscan(RADIUS, 1))
    groups = []
    for i, v in enumerate(visits):
        eligible = [
            group
            for group in groups
            if proposals[group[0]] == proposals[i] and all(_compatible(v, visits[j]) for j in group)
        ]
        if eligible:
            best = min(
                eligible, key=lambda group: np.mean([np.linalg.norm(coords[j] - coords[i]) for j in group])
            )
            best.append(i)
        else:
            groups.append([i])
    instances = []
    for group in groups:
        name = f"seat-{len(instances) + 1:02d}"
        for i in group:
            visits[i]["instance_id"] = name
        instances.append(
            dict(
                instance_id=name,
                visit_indices=group,
                anchor_xy_m=np.median([visits[i]["anchor_xy_m"] for i in group], axis=0).tolist(),
                height_m=float(np.median([visits[i]["height_m"] for i in group])),
                forward_xy=unit(np.median([visits[i]["forward_xy"] for i in group], axis=0)).tolist(),
            )
        )
    ambiguities = []
    for i, a in enumerate(instances):
        for b in instances[i + 1 :]:
            offset = np.subtract(a["anchor_xy_m"], b["anchor_xy_m"])
            if (
                np.linalg.norm(offset) <= 1.0
                and abs(offset @ a["forward_xy"]) <= 0.28
                and abs(offset @ b["forward_xy"]) <= 0.28
                and abs(a["height_m"] - b["height_m"]) <= MAX_HEIGHT_DIFFERENCE
                and np.dot(a["forward_xy"], b["forward_xy"]) >= MIN_ALIGNMENT
            ):
                ambiguities.append(
                    dict(
                        kind="possible_shared_seating_surface",
                        instance_ids=[a["instance_id"], b["instance_id"]],
                    )
                )
    return instances, ambiguities


def recognize(manifest_path, vertices_path, contacts_path, indices_path, *, case_id=None, baseline=False, seating=False):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    fit = manifest["fitting_data"]
    if (fit["units"], fit["up_axis"]) != ("m", "Z"):
        raise ValueError("Recognition requires Z-up world geometry in metres")
    for field in ("faces", "world", "joints", "joint_names"):
        if sha256(resolve(path.parent, fit[field])) != fit["sha256"][field]:
            raise ValueError(f"Recognition geometry hash mismatch: {field}")
    world = np.load(vertices_path, mmap_mode="r", allow_pickle=False)
    dense = np.load(resolve(path.parent, fit["world"]), mmap_mode="r", allow_pickle=False)
    joints = np.load(resolve(path.parent, fit["joints"]), mmap_mode="r", allow_pickle=False)
    faces = np.load(resolve(path.parent, fit["faces"]), allow_pickle=False)
    names = json.loads(resolve(path.parent, fit["joint_names"]).read_text(encoding="utf-8"))
    if isinstance(names, dict):
        names = names["joint_names"]
    labels = np.load(contacts_path, allow_pickle=False)
    indices = np.load(indices_path, allow_pickle=False)
    n = len(world)
    fps = float(manifest["source_fps"])
    stride = manifest.get("prediction_stride", 8)
    if (
        not n
        or world.shape != (n, 655, 3)
        or dense.shape != (n, 10475, 3)
        or joints.shape != (n, len(names), 3)
        or len(set(names)) != len(names)
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or faces.dtype.kind not in "iu"
        or faces.min() < 0
        or faces.max() >= dense.shape[1]
        or not np.isfinite(joints).all()
        or not np.isfinite(fps)
        or fps <= 0
        or type(stride) is not int
        or stride <= 0
    ):
        raise ValueError("Inconsistent motion geometry, sample rate, or stride")
    for start in range(0, n, 32):
        if (
            not np.isfinite(dense[start : start + 32]).all()
            or not np.isfinite(world[start : start + 32]).all()
        ):
            raise ValueError("Nonfinite recognition geometry")
    vertex_hash = sha256(vertices_path)
    binding = fit.get("single_export_sha256", {})
    producer = resolve(path.parent, manifest["world"])
    producer_hash = sha256(producer)
    if binding and (binding["world"] != producer_hash or binding["dense"] != fit["sha256"]["world"]):
        raise ValueError("Recognition geometry and producer motion have different hashes")
    if vertex_hash != producer_hash:
        other = np.load(producer, mmap_mode="r", allow_pickle=False)
        if other.shape != world.shape:
            raise ValueError("Recognition input differs from producer motion")
        for start in range(0, n, 128):
            error = np.linalg.norm(other[start : start + 128] - world[start : start + 128], axis=-1)
            if not np.isfinite(error).all() or error.max() > 0.0001:
                raise ValueError("Recognition input differs from producer motion by more than 0.1 mm")
    if (
        indices.dtype.kind not in "iu"
        or not np.array_equal(indices, np.arange(0, n, stride))
        or labels.shape not in ((len(indices), 655), (len(indices), 655, 1))
        or labels.dtype.kind not in "iu"
        or labels.max() > 7
        or labels.min() < 0
    ):
        raise ValueError("Invalid ContactFormer labels or source-frame grid")
    labels = labels.reshape(-1, 655)
    conversion = manifest["conversion"]
    visits, unverified = _visits(dense, faces, joints, names, labels, indices, fps,
                                 seat_labels=(3, 4, 7) if seating else (3, 4))
    instances, ambiguities = assign_locations(visits)
    if seating:
        from .seating import classify_locations
        instances, visits = classify_locations(instances, visits, labels, indices)
    report = dict(
        version=1,
        method="time-bounded stable visits + world-space DBSCAN",
        case_id=case_id if case_id is not None else manifest["sequence"],
        frame_count=n,
        source_fps=fps,
        prediction_stride=stride,
        source_frame_indices=indices.tolist(),
        bvh_sha256=conversion["bvh_sha256"],
        shape_sha256=conversion["shape_sha256"],
        prediction_sha256=sha256(contacts_path),
        source_frame_indices_sha256=sha256(indices_path),
        world_sha256=vertex_hash,
        geometry_sha256=fit["sha256"],
        location_count=len(instances),
        chair_instance_count=len(instances),
        hypothesis="independent stationary chairs",
        instances=instances,
        events=visits,
        ambiguities=ambiguities,
        unverified_intervals=unverified,
        warnings=["seated_posture_without_confirmed_support"] if unverified else [],
        clustering_radius_m=RADIUS,
    )
    if baseline:
        report["original_dbscan"] = original_clusters(world, labels, indices)
    if seating:
        from copy import deepcopy
        report.update(version=2, locations=deepcopy(instances), furniture_instance_count=len(instances),
                      hypothesis="independent category-aware seating", category_policy="support-sample votes; stable raw-label tie break")
        report.pop("chair_instance_count", None)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--vertices-path", required=True, type=Path)
    parser.add_argument("--contacts", required=True, type=Path)
    parser.add_argument("--frame-indices", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--seating", action="store_true")
    args = parser.parse_args(argv)
    report = recognize(
        args.geometry,
        args.vertices_path,
        args.contacts,
        args.frame_indices,
        case_id=args.case_id,
        baseline=args.baseline,
        seating=args.seating,
    )
    args.output.parent.mkdir(exist_ok=True, parents=True)
    write_json(args.output, report)


if __name__ == "__main__":
    main()
