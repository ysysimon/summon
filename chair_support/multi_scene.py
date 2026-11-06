"""Rank fixed-seat combinations and certify geometry in shared world/source coordinates."""

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import heapq
from itertools import combinations, product
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import open3d as o3d
import scipy
from scipy.spatial import ConvexHull, QhullError
import trimesh

from .collision import ClosedSolid, MAXIMUM_PENETRATION, collision_check
from .engine import read_asset, sha256
from .geometry_audit import GeometryAudit, NUMERIC_MARGIN, children, prepare_geometry


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
    with NamedTemporaryFile(dir=path.parent, prefix=".partial-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
        except BaseException:
            stream.close()
            temporary.unlink()
            raise
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def content_key(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def array_key(values):
    array = np.asarray(values)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    for start in range(0, len(array), 32):
        digest.update(np.ascontiguousarray(array[start : start + 32]).view(np.uint8))
    return digest.hexdigest()


def scene_inputs(ctx, input_key):
    return dict(
        source=input_key,
        dense=array_key(ctx["dense"]),
        faces=array_key(ctx["faces"]),
        frame_count=len(ctx["dense"]),
    )


def source_files(record, directory):
    source = (directory / record["source_asset"]).resolve(strict=True)
    meta = read(source)
    return [
        source,
        source.parent / meta.get("mesh", "mesh.obj"),
        source.parent / meta.get("parts", "parts.npz"),
    ]


def binding(fit_dir, inputs, instance_id):
    path = fit_dir / "fit_result.json"
    result = read(path)
    records = {}
    for c in result["candidates"]:
        if not c.get("qualified"):
            continue
        if c["id"] in records or not math.isfinite(c["loss"]):
            raise ValueError("Invalid or duplicate qualified candidate")
        mesh = fit_dir / c["mesh"]
        files = [mesh, fit_dir / c["parts"], mesh.with_name("collision.json"), *source_files(c, fit_dir)]
        records[c["id"]] = [sha256(p) for p in files]
    return dict(
        version=1, instance_id=instance_id, inputs=inputs, fit_result_sha256=sha256(path), candidates=records
    )


def seal_instance(ctx, fit_dir, input_key, instance_id):
    """Call immediately after a fresh fit, before its candidate checkpoint is reusable."""
    root = Path(fit_dir)
    seal = binding(root, scene_inputs(ctx, input_key), instance_id)
    write_atomic(root / "scene_binding.json", seal)
    return seal


def inside(root, path):
    path = (root / path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Candidate artifact outside its fitting directory")
    return path


@dataclass
class Candidate:
    instance: str
    record: dict
    directory: Path
    mesh: object
    labels: np.ndarray
    geometry: GeometryAudit
    outer: list
    frames: list

    @property
    def key(self):
        return self.instance, self.record["id"]


def validated_frames(path, frame_count):
    certificate = read(path)
    rows = certificate["frames"]
    if (
        certificate["status"] != "certified_below_limit"
        or certificate["uncertain_frames"]
        or certificate["checked_source_frames"] != frame_count
        or certificate["required_source_frames"] != frame_count
        or len(rows) != frame_count
    ):
        raise ValueError("Candidate lacks a complete single-chair certificate")
    for number, row in enumerate(rows):
        if row["frame"] != number or row["uncertain"]:
            raise ValueError("Candidate certificate frame mapping is incomplete")
        bounds = []
        for direction in ("chair_to_body", "body_to_chair"):
            value = row["directions"][direction]
            if (
                value["uncertain"]
                or value["upper_bound"] is None
                or not 0 <= value["lower_bound"] <= value["upper_bound"] <= MAXIMUM_PENETRATION
            ):
                raise ValueError("Invalid directional full-motion certificate")
            bounds.append(value["upper_bound"])
        if not max(bounds) <= row["penetration_upper_bound"] <= MAXIMUM_PENETRATION:
            raise ValueError("Invalid per-frame penetration upper bound")
    return rows


def load_candidate(instance, record, fit_dir, frame_count):
    mesh_file = inside(fit_dir, record["mesh"])
    frames = validated_frames(mesh_file.with_name("collision.json"), frame_count)
    src, src_mesh, _ = source_files(record, fit_dir)
    meta, local, parts = read_asset(src)
    if sha256(src_mesh) != record["source_geometry_sha256"]:
        raise ValueError("Fitted candidate and raw geometry differ")
    pose = record["transform"]
    rotation, translation, scale = (
        np.asarray(pose["rotation"]),
        np.asarray(pose["translation"]),
        float(pose["scale"]),
    )
    if (
        rotation.shape != (3, 3)
        or translation.shape != (3,)
        or not np.isfinite(translation).all()
        or not np.isclose(np.linalg.det(rotation), 1.0)
        or not np.isfinite(scale)
    ):
        raise ValueError("Invalid candidate pose")
    procedural = "infinigen" in str(meta.get("source", "")).lower() or "generator" in meta
    prepared = prepare_geometry(local, parts, max_scale=1.0 if procedural else 1.10)
    solids = prepared.posed(rotation, translation, scale)
    v = local.vertices @ rotation.T * scale + translation
    posed = trimesh.Trimesh(v, local.faces, process=False)
    displayed = trimesh.load(mesh_file, force="mesh", process=False)
    if (
        not np.array_equal(posed.faces, displayed.faces)
        or displayed.vertices.shape != posed.vertices.shape
        or not np.allclose(displayed.vertices, posed.vertices, rtol=0, atol=5.1e-8)
    ):
        raise ValueError("Exported mesh and certified world pose disagree")
    # Reuse exact source faces and transform as seen by the single-chair verifier.
    # Exported OBJ decimal rounding cannot replace them in a near-threshold test.
    by_id = {s["id"]: s for s in solids}
    records, outer = [], []
    for original in prepared.records:
        r = dict(original)
        piece = by_id.get(r["id"])
        if r["mode"] == "bounds_only":
            b = np.asarray(r["bounds"])
            # Preserve oriented extent for pair tests. A world AABB is conservative for collision_check.
            box = trimesh.creation.box(extents=b[1] - b[0])
            verts = (box.vertices + b.mean(0)) @ rotation.T * scale + translation
            r["bounds"] = [verts.min(0).tolist(), verts.max(0).tolist()]
            piece = dict(
                vertices=verts,
                faces=box.faces,
                mode="bounds_only",
                id=r["id"],
                instance=r["instance"],
                parts=r["parts"],
            )
        if piece is not None:
            outer.append(dict(piece, bounds=np.array([piece["vertices"].min(0), piece["vertices"].max(0)])))
        records.append(r)
    world = GeometryAudit(records=records, solids=solids, face_components=prepared.face_components)
    labels = np.asarray(parts.get("face_part_ids", np.full(len(posed.faces), -1, dtype=np.int32)))
    return Candidate(instance, record, fit_dir, posed, labels, world, outer, frames)


def aabb_gap(a, b):
    return float(np.max(np.maximum(a[0] - b[1], b[0] - a[1])))


def outer_box(pieces):
    bounds = np.stack([p["bounds"] for p in pieces])
    return np.stack((bounds[:, 0].min(0), bounds[:, 1].max(0)))


def plane_relation(a, b):
    """Test directional separation on original world vertices, including exact oblique contact."""
    for src, other in ((a, b), (b, a)):
        vertices = src["vertices"]
        if "_hull" not in src:
            try:
                src["_hull"] = ConvexHull(vertices)
            except QhullError:
                src["_hull"] = None
        hull = src["_hull"]
        if hull is None:
            continue
        other_v = other["vertices"]
        center = other_v.mean(0)
        distances = hull.equations[:, :3] @ center + hull.equations[:, 3]
        # Facets facing away from this interior point cannot separate the hulls.
        # A bounded shortlist is a speed optimisation; unresolved pairs proceed
        # to the continuous surface check below.
        indices = np.flatnonzero(distances >= -NUMERIC_MARGIN)
        nearest = indices[np.argsort(-distances[indices], kind="stable")[:64]]
        for i in nearest:
            axis = hull.equations[i, :3]
            left = vertices @ axis
            right = other_v @ axis
            gap = right.min() - left.max()
            if gap > 2 * NUMERIC_MARGIN:
                return "separated"
            rounding = max(1.0, np.abs(vertices).max(), np.abs(other_v).max()) * 32 * np.finfo(float).eps
            if gap > rounding:
                return "touching_or_close"
            if gap < -rounding:
                continue
            # On a candidate common plane use exact projections of the limiting
            # vertices. Do not round micron-scale physical overlaps into contact.
            exact_axis = [Fraction(float(x)) for x in axis]

            def exact_dot(point):
                return sum(Fraction(float(y)) * z for y, z in zip(point, exact_axis))

            upper = max(map(exact_dot, vertices[left >= left.max() - 2 * rounding]))
            lower = min(map(exact_dot, other_v[right <= right.min() + 2 * rounding]))
            if upper <= lower:
                return "touching_or_close"
    return None


def piece_solid(piece):
    if "_solid" not in piece:
        piece["_solid"] = ClosedSolid(
            piece["vertices"], piece["faces"], [dict(piece, id=0)], "furniture", prepared=True
        )
    return piece["_solid"]


def surface_clearance(source, target, both_exact, query_budget=75000):
    """Certify signed distance over every source triangle, not only sampled vertices."""
    triangles = source["vertices"][source["faces"]]
    solid = piece_solid(target)
    count = 0
    while len(triangles):
        if count + len(triangles) > query_budget:
            return "undetermined", None
        centers = triangles.mean(1)
        radius = np.linalg.norm(triangles - centers[:, None, :], axis=2).max(1)
        lo, hi, _ = solid.signed_bounds(centers)
        count += len(triangles)
        if np.any(hi < -NUMERIC_MARGIN):
            i = int(hi.argmin())
            if both_exact:
                return "overlap", dict(point=centers[i].tolist(), interior_depth_lower_m=float(-hi[i]))
            return "undetermined", None
        pending = lo - radius <= 2 * NUMERIC_MARGIN
        if not np.any(pending):
            return "separated", None
        if np.any(radius[pending] < NUMERIC_MARGIN):
            return "undetermined", None
        triangles = children(triangles[pending])
    return "separated", None


def piece_pair(a, b):
    gap = aabb_gap(a["bounds"], b["bounds"])
    if gap > 2 * NUMERIC_MARGIN:
        return "separated", None
    if gap >= 0:
        # The two interiors are disjoint even at an exactly shared supporting plane.
        # Leave the union depth calculation to the verifier at tangencies.
        return "touching_or_close", None
    boundary = plane_relation(a, b)
    if boundary:
        return boundary, None
    both_exact = all(p["mode"] in ("original", "stitched") for p in (a, b))
    if both_exact:
        # Coincident shells and containment can have no transverse edge crossing.
        # A verified strictly interior point of BOTH original solids proves overlap.
        samples = np.array(
            [
                a["vertices"].mean(0),
                b["vertices"].mean(0),
                (np.maximum(a["bounds"][0], b["bounds"][0]) + np.minimum(a["bounds"][1], b["bounds"][1]))
                * 0.5,
            ]
        )
        upper_a = piece_solid(a).signed_bounds(samples)[1]
        upper_b = piece_solid(b).signed_bounds(samples)[1]
        interior = np.maximum(upper_a, upper_b)
        if interior.min() < -NUMERIC_MARGIN:
            i = int(interior.argmin())
            return "overlap", dict(point=samples[i].tolist(), interior_depth_lower_m=float(-interior[i]))
    statuses = []
    for source, target in ((a, b), (b, a)):
        status, witness = surface_clearance(source, target, both_exact)
        if status == "overlap":
            return status, witness
        statuses.append(status)
    return ("separated" if all(s == "separated" for s in statuses) else "undetermined"), None


def static_pair(left, right):
    item = dict(instances=[left.instance, right.instance], candidates=[left.record["id"], right.record["id"]])
    unknown = any(r["mode"] == "unknown" for c in (left, right) for r in c.geometry.records)
    if not unknown and aabb_gap(outer_box(left.outer), outer_box(right.outer)) > 2 * NUMERIC_MARGIN:
        return dict(item, status="separated", method="disjoint_conservative_outer_aabbs")
    status = "undetermined" if unknown else "separated"
    ambiguous = []
    for a, b in product(left.outer, right.outer):
        kind, witness = piece_pair(a, b)
        which = dict(component_ids=[a["id"], b["id"]], physical_part_ids=[a["instance"], b["instance"]])
        if kind == "overlap":
            return dict(
                item, status=kind, method="verified_original_solid_interior", witness=dict(witness, **which)
            )
        if kind == "undetermined":
            status = "undetermined"
            ambiguous.append(which)
        elif kind == "touching_or_close" and status == "separated":
            status = kind
    return dict(
        item,
        status=status,
        method="outer_component_pair_tests",
        unknown_extent=unknown,
        unresolved_pairs=ambiguous,
    )


def union_collision(ctx, candidates):
    """Retain each original / stitched solid, hull, authored extent and raw face's attribution."""
    vertices, faces, labels = [], [], []
    solids, records, face_map, owners = [], [], [], []
    face_offset = vertex_offset = 0
    for candidate in candidates:
        num_records = len(records)
        for local in candidate.geometry.records:
            global_id = num_records + local["id"]
            record = dict(
                local, id=global_id, source_faces=(np.asarray(local["source_faces"]) + face_offset).tolist()
            )
            owners.append((candidate.instance, local["id"], face_offset))
            records.append(record)
        for piece in candidate.geometry.solids:
            ids = piece["face_ids"]
            solids.append(
                dict(piece, id=num_records + piece["id"], face_ids=None if ids is None else ids + face_offset)
            )
        vertices.append(candidate.mesh.vertices)
        faces.append(candidate.mesh.faces + vertex_offset)
        labels.append(candidate.labels)
        face_map.append(candidate.geometry.face_components + num_records)
        face_offset += len(candidate.mesh.faces)
        vertex_offset += len(candidate.mesh.vertices)
    raw = trimesh.Trimesh(np.concatenate(vertices), np.concatenate(faces), process=False)
    audit = GeometryAudit(records=records, solids=solids, face_components=np.concatenate(face_map))
    check = collision_check(ctx, raw, {"face_part_ids": np.concatenate(labels)}, prepared=audit)
    if check.get("penetration_witness"):
        w = check["penetration_witness"]
        seat, local_component, offset = owners[w["furniture_component"]]
        w.update(
            seat_instance_id=seat,
            candidate_component=local_component,
            candidate_face=w["furniture_face"] - offset,
        )
    return check


def reused_certificate(candidates, count):
    frames = []
    worst = None
    for frame in range(count):
        dirs = {}
        for name in ("chair_to_body", "body_to_chair"):
            rows = [c.frames[frame]["directions"][name] for c in candidates]
            low = max((r["lower_bound"] for r in rows), default=0.0)
            high = max((r["upper_bound"] for r in rows), default=0.0)
            dirs[name] = dict(lower_bound=low, upper_bound=high, uncertain=False)
            if low > 0 and (worst is None or low > worst["lower_bound"]):
                i = next(i for i, r in enumerate(rows) if r["lower_bound"] == low)
                worst = dict(
                    rows[i].get("worst") or {},
                    frame=frame,
                    direction=name,
                    seat_instance_id=candidates[i].instance,
                    candidate_id=candidates[i].record["id"],
                    lower_bound=low,
                    sample_lower_bound=low,
                )
        frames.append(
            dict(
                frame=frame,
                maximum_penetration=max(r["lower_bound"] for r in dirs.values()),
                penetration_upper_bound=max(r["upper_bound"] for r in dirs.values()),
                directions=dirs,
                uncertain=False,
            )
        )
    lower = max((r["maximum_penetration"] for r in frames), default=0.0)
    upper = max((r["penetration_upper_bound"] for r in frames), default=0.0)
    return dict(
        status="certified_below_limit",
        method="disjoint_instance_certificates" if candidates else "empty_scene",
        required_source_frames=count,
        checked_source_frames=count,
        frames=frames,
        maximum_penetration=lower,
        maximum_penetration_lower_bound=lower,
        maximum_penetration_upper_bound=upper,
        uncertain_frames=[],
        penetration_witness=worst,
        worst_collision_frame=worst["frame"] if worst else None,
    )


def ranked_choices(ordered):
    """Visit the Cartesian product lazily in ascending summed loss, with deterministic ties."""
    start = (0,) * len(ordered)

    def cost(indices):
        return math.fsum(rows[i]["loss"] for rows, i in zip(ordered, indices))

    queue, seen = [(cost(start), start)], {start}
    while queue:
        score, indices = heapq.heappop(queue)
        yield score, indices
        for axis, rows in enumerate(ordered):
            if indices[axis] + 1 >= len(rows):
                continue
            update = list(indices)
            update[axis] += 1
            key = tuple(update)
            if key not in seen:
                seen.add(key)
                heapq.heappush(queue, (cost(key), key))


def cached_outcome(path, key):
    if path.is_file():
        try:
            saved = read(path)
            if saved["key"] == key and saved["sha256"] == content_key(saved["outcome"]):
                return saved["outcome"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return None


def assemble_scene(ctx, fit_dirs, output, max_combinations=64, input_key=None, cache_dir=None):
    """Combine already world-posed candidates. All human checks use the full source grid."""
    if type(max_combinations) is not int or max_combinations < 1:
        raise ValueError("Scene combination budget must be a positive integer")
    output = Path(output).resolve()
    count = len(ctx["dense"])
    if count < 1:
        raise ValueError("Empty human motion")
    inputs = scene_inputs(ctx, input_key)
    ordered, seals, dirs = [], {}, {}
    report = dict(
        status="no_valid_scene",
        selected=[],
        attempts=[],
        source_frames=count,
        collision=None,
        combination_budget=max_combinations,
        rejection_reasons=[],
    )
    missing = {}
    unverified = False
    for instance, dir_path in sorted(fit_dirs.items()):
        root = Path(dir_path).resolve()
        seal = binding(root, inputs, instance)
        if read(root / "scene_binding.json") != seal:
            raise ValueError(
                "Instance candidate checkpoint does not match the scene inputs: " + str(instance)
            )
        # An overall fit_result SHA includes absolute asset paths in legacy
        # outputs; use it to validate this run, not as a shared-cache identity.
        dirs[instance], seals[instance] = root, seal["candidates"]
        fit = read(root / "fit_result.json")
        candidates = fit["candidates"]
        pool = sorted((c for c in candidates if c.get("qualified")), key=lambda c: (c["loss"], c["id"]))
        if not pool:
            failures = {c["id"]: c.get("rejection_reasons", []) for c in candidates}
            missing[instance] = dict(
                fitting_status=fit["status"], reasons=fit.get("rejection_reasons", []), candidates=failures
            )
            reasons = fit.get("rejection_reasons", []) + [r for row in failures.values() for r in row]
            unverified |= any(
                "undetermined" in r or "unverified" in r or r == "no_support_evidence" for r in reasons
            )
        ordered.append(pool)
    if missing:
        report.update(rejection_reasons=["no_qualified_candidate_for_instance"], rejected_instances=missing)
        if unverified:
            report["status"] = "undetermined"
        write_atomic(output / "scene_result.json", report)
        return report
    impl = {
        name: sha256(Path(__file__).parent / name)
        for name in ("multi_scene.py", "collision.py", "geometry_audit.py", "engine.py")
    }
    impl.update(
        numpy=np.__version__, scipy=scipy.__version__, trimesh=trimesh.__version__, open3d=o3d.__version__
    )
    loaded, pairs = {}, {}
    keys = sorted(dirs)
    report["total_combinations"] = math.prod(len(rows) for rows in ordered)
    for score, indices in ranked_choices(ordered):
        if len(report["attempts"]) >= max_combinations:
            report.update(status="search_budget_exhausted", rejection_reasons=["search_budget_exhausted"])
            break
        chosen = [rows[i] for rows, i in zip(ordered, indices)]
        ids = [c["id"] for c in chosen]
        bindings = [
            dict(
                instance=k,
                candidate=c["id"],
                transform=c["transform"],
                source_and_fitted_sha256=seals[k][c["id"]],
            )
            for k, c in zip(keys, chosen)
        ]
        cache_key = content_key(dict(inputs=inputs, candidates=bindings, implementation=impl))
        cache = output / "checkpoints" / (cache_key + ".json")
        shared = Path(cache_dir).resolve() / (cache_key + ".json") if cache_dir else cache
        attempt = dict(
            candidates=dict(zip(keys, ids)), score=score, checkpoint=cache.relative_to(output).as_posix()
        )
        outcome = cached_outcome(cache, cache_key)
        if outcome is None and shared != cache:
            outcome = cached_outcome(shared, cache_key)
        attempt["checkpoint_reused"] = outcome is not None
        if outcome is None:
            choice = []
            for instance, rows, i in zip(keys, ordered, indices):
                k = (instance, rows[i]["id"])
                if k not in loaded:
                    loaded[k] = load_candidate(instance, rows[i], dirs[instance], count)
                choice.append(loaded[k])
            furniture = []
            for a, b in combinations(choice, 2):
                pair_key = (a.key, b.key)
                if pair_key not in pairs:
                    pairs[pair_key] = static_pair(a, b)
                furniture.append(pairs[pair_key])
            statuses = {p["status"] for p in furniture}
            collision = None
            if "overlap" in statuses:
                state = "furniture_overlap"
            else:
                collision = (
                    reused_certificate(choice, count)
                    if statuses <= {"separated"}
                    else union_collision(ctx, choice)
                )
                state = {
                    "certified_below_limit": "success",
                    "rejected": "measured_penetration",
                    "undetermined": "collision_depth_undetermined",
                }[collision["status"]]
                if "undetermined" in statuses and state != "measured_penetration":
                    state = "furniture_geometry_undetermined"
            outcome = dict(status=state, furniture_pairs=furniture, collision=collision)
            if shared != cache:
                write_atomic(shared, dict(key=cache_key, sha256=content_key(outcome), outcome=outcome))
        # Every run includes its evidence with no dependency on the shared cache.
        write_atomic(cache, dict(key=cache_key, sha256=content_key(outcome), outcome=outcome))
        attempt.update(status=outcome["status"], furniture_pairs=outcome["furniture_pairs"])
        if outcome["collision"] and outcome["collision"].get("penetration_witness"):
            attempt["penetration_witness"] = outcome["collision"]["penetration_witness"]
        report["attempts"].append(attempt)
        if outcome["status"] == "success":
            selected = []
            for k, c in zip(keys, chosen):
                selected.append(
                    dict(
                        instance_id=k,
                        candidate_id=c["id"],
                        loss=c["loss"],
                        transform=c["transform"],
                        mesh=Path(os.path.relpath(inside(dirs[k], c["mesh"]), output)).as_posix(),
                        parts=Path(os.path.relpath(inside(dirs[k], c["parts"]), output)).as_posix(),
                    )
                )
            report.update(status="success", selected=selected, collision=outcome["collision"])
            break
    if report["status"] == "no_valid_scene":
        reasons = sorted({a["status"] for a in report["attempts"]})
        report["rejection_reasons"] = reasons
        if any("undetermined" in s for s in reasons):
            report["status"] = "undetermined"
    write_atomic(output / "scene_result.json", report)
    return report
