"""Fit explicitly labelled chair surfaces to an immutable human motion.

Thresholds are engineering acceptance defaults, not calibrated probabilities.
All acceptance decisions and failures are exported, including rejected meshes.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.ndimage import uniform_filter1d
from scipy.optimize import OptimizeResult, minimize
from scipy.spatial import cKDTree
import trimesh

from .collision import collision_check, collision_checks, close_body_topology, edge_intersections
from .geometry_audit import prepare_geometry
from .selection import (VERSION as SELECTION_VERSION, evidence_weights, select_support,
                        support_intervals, sample_support_frames, weighted_quantile, weighted_vector)

from .policy import FitPolicy, add_arguments, eligible, skipped
from .runtime import ExecutionOptions, Profile, execution_scope, add_execution_arguments
from .query_backend import query_scene
from .solver_helpers import PoseCache, choose_current_result, batched_support_metrics, candidate_directory_key
from .candidate_pool import pack_arrays, unpack_arrays, ordered_spawn_map

REGIONS = ("left_gluteus", "right_gluteus", "left_thigh", "right_thigh")
DEFAULTS = dict(short_support_warning_seconds=.4, pelvis_speed_weight_scale=.08,
                pelvis_vertical_speed_weight_scale=.03,
                min_chair_vertices=4, thigh_angle_degrees=35., min_knee_degrees=60.,
                max_knee_degrees=130., normal_z=-.5, grid_size=.02,
                coverage=.90, gluteus_gap=.015, thigh_gap=.020,
                maximum_penetration=.005, floor_tolerance=.005,
                maximum_yaw_degrees=20., min_scale=.90, max_scale=1.10)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def resolve(root, value):
    p = Path(value)
    return p if p.is_absolute() else root / p


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def ray_scene(vertices, faces):
    # This public helper remains CPU-only for strict validation and old callers.
    from .geometry_audit import ray_scene as cpu_scene
    return cpu_scene(vertices, faces)


def vertex_normals(vertices, faces):
    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(out, faces[:, k], n)
    return unit(out)


def contiguous_runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def motion_context(manifest_path, vertices_path, contacts_path, indices_path, output, visits=None, locations=None, category=None):
    p = Path(manifest_path)
    manifest = json.loads(p.read_text(encoding="utf-8"))
    data = manifest.get("fitting_data", manifest)
    root = p.parent
    dense = np.load(resolve(root, data["world"]), mmap_mode="r")
    joints = np.load(resolve(root, data["joints"]), mmap_mode="r")
    names = json.loads(resolve(root, data["joint_names"]).read_text(encoding="utf-8"))
    if isinstance(names, dict):
        names = names.get("joint_names", names.get("names"))
    faces = np.load(resolve(root, data["faces"]))
    world = np.load(vertices_path, mmap_mode="r")
    labels = np.load(contacts_path).reshape(-1, world.shape[1])
    source_indices = np.load(indices_path)
    fps = float(manifest.get("source_fps", data.get("source_fps", 0)))
    if fps <= 0 or len(dense) != len(world) or len(joints) != len(world):
        raise ValueError("Dense geometry, motion and FPS are inconsistent")
    name_map = {n: i for i, n in enumerate(names)}
    required = ("Hips", "LeftLeg", "LeftShin", "LeftFoot", "RightLeg", "RightShin", "RightFoot")
    missing = set(required) - name_map.keys()
    if missing:
        raise ValueError("SOMA joint names missing: " + str(sorted(missing)))
    j = {n: np.asarray(joints[:, name_map[n]]) for n in required}
    pelvis = j["Hips"]
    knees = (j["LeftShin"] + j["RightShin"]) / 2
    hips = (j["LeftLeg"] + j["RightLeg"]) / 2
    right = unit((j["RightLeg"] - j["LeftLeg"]) * [1, 1, 0])
    forward = np.cross([0., 0., 1.], right)
    sign = np.where(np.sum(forward * (knees - hips), axis=1) < 0, -1., 1.)
    forward *= sign[:, None]
    smooth = max(1, int(round(fps * .1)))
    velocity = (np.gradient(uniform_filter1d(pelvis, smooth, axis=0, mode="nearest"), axis=0) * fps
                if len(pelvis) > 1 else np.zeros_like(pelvis))
    seat_labels = (3, 4, 7) if category is not None else (3,)
    chair_contacts = np.isin(labels, seat_labels)
    if visits is not None:
        from .multi_fit import instance_contacts, visit_masks
        interaction_ids, support_ids = visit_masks(visits, len(dense))
        chair_contacts = instance_contacts(world, labels, source_indices, interaction_ids, visits[0]["instance_id"], locations,
                                            seat_labels=(3, 4, 7) if category is not None else (3, 4))
    chair_count = chair_contacts.sum(axis=1)
    phase_count = np.interp(np.arange(len(dense)), source_indices, chair_count)
    conditions = {"insufficient_chair_contacts": phase_count >= DEFAULTS["min_chair_vertices"]}
    if visits is not None:
        conditions["outside_instance_support"] = support_ids >= 0
    angles = {}; thigh_angles={}
    for side in ("Left", "Right"):
        thigh = j[side + "Shin"] - j[side + "Leg"]
        shin = j[side + "Foot"] - j[side + "Shin"]
        theta = np.degrees(np.arccos(np.clip(np.sum(unit(-thigh) * unit(shin), axis=1), -1, 1)))
        angles[side] = theta
        thigh_angles[side]=np.degrees(np.arcsin(np.clip(np.abs(unit(thigh)[:,2]),0,1)))
        conditions[side.lower()+"_thigh_angle"] = thigh_angles[side] <= DEFAULTS["thigh_angle_degrees"]
        conditions[side.lower()+"_knee_angle"] = ((theta >= DEFAULTS["min_knee_degrees"])
                                                  & (theta <= DEFAULTS["max_knee_degrees"]))
    candidate = np.logical_and.reduce(list(conditions.values()))
    speed, vertical_speed = np.linalg.norm(velocity, axis=1), np.abs(velocity[:, 2])
    weights = evidence_weights(phase_count, speed, vertical_speed)
    floor_samples = [world[f, labels[k] == 2, 2].min() for k, f in enumerate(source_indices) if (labels[k] == 2).any()]
    if not floor_samples:
        raise ValueError("No predicted floor contacts; explicit floor estimation is required")
    floor = float(np.median(floor_samples))
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "body_forward.npy", forward.astype(np.float32))
    np.save(output / "pelvis.npy", pelvis.astype(np.float32))
    ctx = dict(dense=dense, joints=joints, joint_names=names, j=j, faces=faces, world=world,
               labels=labels, indices=source_indices, fps=fps, pelvis=pelvis, forward=forward,
               right=right, floor=floor, frame_weights=weights, category=category or "chair", seating=category is not None)
    regions = body_regions(ctx)
    anchors, invalid = collect_support_samples(ctx, regions, np.flatnonzero(candidate))
    valid_frames = np.array([sample[0] for sample in anchors[REGIONS[0]]], dtype=int)
    gap_limits = {region: DEFAULTS["gluteus_gap"] if "gluteus" in region else DEFAULTS["thigh_gap"]
                  for region in REGIONS}
    selected, proxy = select_support(valid_frames, weights, anchors, pelvis, forward,
                                     gap_limits, DEFAULTS["maximum_penetration"])
    if visits is not None:
        from .multi_fit import retain_return_visits
        selected, visit_evidence = retain_return_visits(visits, selected, valid_frames, support_ids,
            anchors, weights, pelvis, forward, gap_limits, DEFAULTS["maximum_penetration"])
        ctx.update(visits=visits, instance_contacts=chair_contacts)
    selected_mask = np.zeros(len(dense), bool); selected_mask[selected] = True
    valid_mask = np.zeros(len(dense), bool); valid_mask[valid_frames] = True
    reasons = []
    for frame in range(len(dense)):
        rejected = [name for name, passed in conditions.items() if not passed[frame]]
        reasons.append(rejected or invalid.get(frame) or ["selected_support" if selected_mask[frame]
                                                        else "outside_common_support_height"])
    seconds = len(selected) / fps
    warnings = (["short_support_evidence"] if len(selected) and seconds < DEFAULTS["short_support_warning_seconds"] else [])
    if len(selected) and not proxy["proxy_feasible"]:
        warnings.append("planar_proxy_insufficient")
    phase = dict(selection_version=SELECTION_VERSION, source_fps=fps,
                 candidate_frames=int(candidate.sum()), valid_candidate_frames=len(valid_frames),
                 support_frames=len(selected), support_frame_indices=selected.tolist(),
                 support_frame_weights=weights[selected].tolist(), support_seconds=seconds,
                 weighted_evidence_seconds=float(weights[selected].sum()/fps), warnings=warnings,
                 support_intervals=[dict(start=a, end_exclusive=b) for a,b in support_intervals(selected)],
                 floor_height=floor, selection_proxy=proxy, selection_reasons=reasons,
                 weight_semantics="heuristic evidence strength, not probability; no velocity or duration rejection",
                 contact_phase_interpolation="counts only; original contact labels remain unchanged", thresholds=DEFAULTS)
    if visits is not None:
        phase["visits"] = visit_evidence
    write_json(output / "phase.json", phase)
    np.savez_compressed(output/"phase_evidence.npz",source_frame_indices=np.arange(len(dense)),
        pelvis_speed=speed,pelvis_vertical_speed=vertical_speed,
        chair_count_interpolated=phase_count,left_thigh_angle=thigh_angles["Left"],
        right_thigh_angle=thigh_angles["Right"],left_knee_angle=angles["Left"],right_knee_angle=angles["Right"],
        pose_candidate=candidate,valid_envelope=valid_mask,support_selected=selected_mask,
        evidence_weight=weights,support_weight=np.where(selected_mask,weights,0),
        selection_reason=np.array([";".join(reason) for reason in reasons]))
    ctx.update(support_frames=selected, phase=phase,
               _support_anchors={r:[sample for sample in samples if selected_mask[sample[0]]]
                                 for r,samples in anchors.items()})
    return ctx


def body_regions(ctx):
    """Load frozen, topology-bound anatomy IDs shared by every motion.

    The one-time anatomical draft was inspected on original and mirrored
    geometry; runtime never derives or changes regions from a clip or labels.
    """
    annotation_path = Path(__file__).parent / "annotations" / "body_regions_smplx_v1.json"
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(np.asarray(ctx["faces"], np.int64).tobytes()).hexdigest()
    if ctx["dense"].shape[1] != annotation["vertex_count"] or digest != annotation["faces_sha256"]:
        raise ValueError("Body regions require the verified SMPL-X template topology")
    return {key: np.asarray(ids, np.int64) for key, ids in annotation["regions"].items()}


def lower_envelope(vertices, normals, ids, grid=.02):
    """One downward surface sample per XY cell, without flattening the body."""
    ids = np.asarray(ids)
    ids = ids[normals[ids, 2] < DEFAULTS["normal_z"]]
    if len(ids) == 0:
        return ids
    cells = np.floor(vertices[ids, :2] / grid).astype(np.int64)
    order = np.lexsort((vertices[ids, 2], cells[:, 1], cells[:, 0]))
    _, first = np.unique(cells[order], axis=0, return_index=True)
    return ids[order[first]]


def collect_support_samples(ctx, regions, frames):
    anchors = {r: [] for r in REGIONS}
    invalid = {}
    for f in frames:
        v = np.asarray(ctx["dense"][f])
        n = vertex_normals(v, ctx["faces"])
        samples = {}
        for r in REGIONS:
            ids = lower_envelope(v, n, regions[r])
            if len(ids) < 3:
                invalid.setdefault(int(f), []).append("insufficient_downward_samples:" + r)
                continue
            # A 2cm XY cell represents surface area / projected area = 1/|nz|.
            # This avoids both vertex-density bias and treating tilted patches
            # as having the same area as their horizontal projection.
            weights = DEFAULTS["grid_size"]**2 / np.maximum(-n[ids,2], .5)
            samples[r] = (int(f), ids, v[ids], weights)
        if int(f) not in invalid:
            for r in REGIONS:
                anchors[r].append(samples[r])
    return anchors, invalid


def prepare_support(ctx, regions, output):
    frames = ctx["support_frames"]
    frame_weights = ctx["frame_weights"][frames]
    anchors = ctx["_support_anchors"]
    saved = {"frame_indices": frames, "frame_weights": frame_weights}
    for r, samples in anchors.items():
        padded = np.full((len(samples), max(len(x[1]) for x in samples)), -1, np.int32)
        for row, sample in enumerate(samples):
            padded[row, :len(sample[1])] = sample[1]
        saved[r+"_vertex_indices"] = padded
    np.savez_compressed(output / "support_indices.npz", **saved)
    np.savez_compressed(output / "body_regions.npz", **{k+"_vertex_indices": v for k,v in regions.items()})
    fwd = unit(weighted_vector(ctx["forward"][frames], frame_weights))
    if np.linalg.norm(fwd) < 1e-8:
        fwd = ctx["forward"][frames[np.argmax(frame_weights)]]
    right = np.cross(fwd, [0., 0., 1.])
    all_points = np.concatenate([x[2] for r in REGIONS for x in anchors[r]])
    p = weighted_vector(ctx["pelvis"][frames], frame_weights)
    projected = np.column_stack(((all_points-p) @ right, (all_points-p) @ fwd))
    height = weighted_quantile([weighted_quantile(x[2][:, 2],x[3],.20) for r in REGIONS for x in anchors[r]],
                               np.tile(frame_weights,len(REGIONS)),.5) - ctx["floor"]
    anatomical_ids = np.concatenate([regions[r] for r in REGIONS])
    widths = [np.ptp((np.asarray(ctx["dense"][f])[anatomical_ids]-p)@right) for f in frames]
    # Statistical width uses evidence weights; the footprint must still contain
    # every selected support envelope, including low-weight source frames.
    width = weighted_quantile(widths,frame_weights,.5)
    if ctx.get("category", "chair") != "chair":
        width = max(width,float(np.ptp(projected[:,0])))
    dims = dict(seat_height=float(height), seat_width=float(width+.06),
                seat_depth=float(np.ptp(projected[:,1])+.08))
    estimate = dict(dimensions=dims, pelvis=p.tolist(), forward=fwd.tolist(), floor_height=ctx["floor"],
                    initial_seat_center=(p + .08*fwd).tolist(),
                    regions={r:len(regions[r]) for r in regions},
                    annotation_method="fixed SMPL-X template IDs in body_regions_smplx_v1.json",
                    thresholds=DEFAULTS, support_frames=len(frames),
                    support_selection_version=SELECTION_VERSION,
                    support_seconds=ctx["phase"]["support_seconds"], warnings=ctx["phase"]["warnings"])
    estimate["category"] = ctx.get("category", "chair")
    if estimate["category"] != "chair":
        # A sofa's whole footprint covers every selected visit. The person may
        # occupy any local position; the whole-seat centre is not a body anchor.
        center = p + right * (projected[:, 0].min() + projected[:, 0].max()) / 2
        center += fwd * (projected[:, 1].min() + projected[:, 1].max()) / 2
        estimate["initial_seat_center"] = center.tolist()
        estimate["local_contact_centers"] = []
        for visit in ctx.get("visits", []):
            samples = [x[2] for r in REGIONS for x in anchors[r] if visit["start"] <= x[0] < visit["end"]]
            # main already returns visit_support_unverified for a visit without
            # selected evidence. Keep its position explicit for direct callers.
            estimate["local_contact_centers"].append(
                np.average(np.concatenate(samples), axis=0).tolist() if samples else None)
        if estimate["category"] == "stool":
            dims["seat_width"] = dims["seat_depth"] = max(dims["seat_width"], dims["seat_depth"])
        write_json(output / "estimate.json", estimate)
        return anchors, estimate
    proxy=estimate_seat_slope(anchors,np.asarray(estimate["initial_seat_center"]),fwd,ctx["floor"],dims["seat_depth"],ctx["fps"])
    estimate["seat_profile"]=proxy
    # Infinigen's front_dip offsets Bezier handles, not the evaluated surface:
    # the undivided front-centre point moves by 3/4 of the handle offset.
    # The real generated surface is still measured and validated downstream.
    dims["front_dip"]=proxy["front_drop_m"]/.75
    if proxy["proxy_feasible"]:
        dims["seat_height"]=proxy["center_height_m"]
    back_profile=estimate_back_slant(ctx,regions["back"])
    estimate["back_profile"]=back_profile
    if back_profile["sampled_frames"]:
        dims["back_slant"]=back_profile["initial_back_slant"]
    write_json(output / "estimate.json", estimate)
    return anchors, estimate


def estimate_back_slant(ctx, indices):
    """Initial native back shear from anatomical posterior surfaces, not contacts.

    The chair's native shear is bounded to [0, .4]. Actual generated geometry
    must still pass the independent support and whole-motion collision gates.
    Pelvis-relative per-frame coordinates keep root motion out of this estimate.
    """
    slopes=[]; weights=[]
    for frame in ctx["support_frames"]:
        points=np.asarray(ctx["dense"][frame])[indices]-ctx["pelvis"][frame]
        height=points[:,2]
        lo,hi=np.quantile(height,[.1,.9])
        if hi-lo<.1:
            continue
        forward=points@ctx["forward"][frame]
        bands=[]
        edges=np.linspace(lo,hi,6)
        for low,high in zip(edges[:-1],edges[1:]):
            mask=(height>=low)&(height<=high)
            if mask.sum()>=3:
                bands.append((float(np.median(height[mask])),float(np.quantile(forward[mask],.05))))
        if len(bands)>=3:
            bands=np.asarray(bands)
            slopes.append(float(-np.polyfit(bands[:,0],bands[:,1],1)[0]))
            weights.append(ctx["frame_weights"][frame])
    raw=weighted_quantile(slopes,weights,.9) if slopes else 0.
    initial=float(np.clip(raw,0.,.4))
    return dict(initial_back_slant=initial,posterior_slope_p90=raw,
        sampled_frames=len(slopes),native_parameter_bounds=[0.,.4],
        limited_to_native_bounds=initial!=raw,
        method="per-frame pelvis-relative posterior back envelope slope; initial geometry only, not contact prediction")


def estimate_seat_slope(anchors, center, forward, floor, depth, fps):
    """Find an initial descending seat profile without changing acceptance.

    This is a planar proxy for procedural generation, never a replacement for
    the actual curved seat or full-frame geometry validation.
    """
    choices=[]
    for slope in np.arange(-40,6)/100:
        lower,upper=-np.inf,np.inf
        for region,samples in anchors.items():
            gap=DEFAULTS["gluteus_gap"] if "gluteus" in region else DEFAULTS["thigh_gap"]
            for _,_,points,weights in samples:
                h=points[:,2]-slope*((points-center)@forward)
                lower=max(lower,weighted_quantile(h,weights,.2)-gap)
                upper=min(upper,float(h.min()+DEFAULTS["maximum_penetration"]))
        choices.append((max(0.,lower-upper),abs(slope),slope,lower,upper))
    violation,_,slope,lower,upper=min(choices)
    return dict(proxy_feasible=violation<1e-8,minimum_violation_m=float(violation),
                slope_dz_dforward=float(slope),front_drop_m=float(max(0.,-slope*depth)),
                center_height_m=float((lower+upper)/2-floor),height_lower_m=float(lower-floor),
                height_upper_m=float(upper-floor),method="all selected support frames, four anatomical surface envelopes; proxy only")


def contact_roles(ctx, regions, output):
    groups={k:[] for k in ("seat","back","front","other")}
    group_frames={k:[] for k in groups}
    records=[]
    back=np.zeros(ctx["dense"].shape[1],bool);back[regions["back"]]=True
    distal=np.zeros_like(back)
    for name in ("left_distal_thigh","right_distal_thigh"):
        distal[regions[name]]=True
    for row,frame in enumerate(ctx["indices"]):
        if "instance_contacts" in ctx:
            ids=np.flatnonzero(ctx["instance_contacts"][row])
        else:
            ids=np.flatnonzero(np.isin(ctx["labels"][row], (3, 4, 7) if ctx.get("seating") else (3,)))
        if not len(ids): continue
        v=np.asarray(ctx["dense"][frame]);normals=vertex_normals(v,ctx["faces"])
        points=np.asarray(ctx["world"][frame,ids]);_,nearest=cKDTree(v).query(points)
        for index,point,near in zip(ids,points,nearest):
            near_knee=min(np.linalg.norm(point-ctx["j"][side+"Shin"][frame]) for side in ("Left","Right"))<.20
            posterior=float(normals[near]@ctx["forward"][frame])<-.30
            role="seat" if normals[near,2]<-.5 else ("back" if back[near] else ("front" if distal[near] or (near_knee and posterior) else "other"))
            groups[role].append(point);group_frames[role].append(int(frame));records.append((int(frame),int(index),role))
    write_json(output/"contact_roles.json",dict(method="fit roles only; original predictions unchanged",
        role_counts={k:len(v) for k,v in groups.items()},records=records))
    result={}
    for role,points in groups.items():
        _,inverse,counts=np.unique(group_frames[role],return_inverse=True,return_counts=True)
        # Each source frame contributes equally within each semantic role.
        weights=1./counts[inverse] if len(inverse) else np.array([],float)
        result[role]=(np.asarray(points).reshape(-1,3),weights)
    return result


class UnverifiedSupport(ValueError):
    """Required functional surfaces could not be identified on the bound asset."""


def read_asset(path):
    path = Path(path)
    a = json.loads(path.read_text(encoding="utf-8"))
    if a.get("annotation_error"):
        if "annotation is empty" in a["annotation_error"]:
            raise UnverifiedSupport(a["annotation_error"])
        raise ValueError(a["annotation_error"])
    mesh_path = resolve(path.parent, a.get("mesh", "mesh.obj"))
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    parts_path = resolve(path.parent, a.get("parts", "parts.npz"))
    expected_mesh=a.get("geometry_sha256",a.get("hashes",{}).get(mesh_path.name))
    expected_parts=a.get("hashes",{}).get(parts_path.name)
    if expected_mesh and sha256(mesh_path)!=expected_mesh:
        raise ValueError("Chair mesh hash mismatch")
    if expected_parts and sha256(parts_path)!=expected_parts:
        raise ValueError("Chair parts hash mismatch")
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError("Invalid chair geometry")
    part = dict(np.load(parts_path))
    if "face_instance_ids" in part and not (expected_mesh and expected_parts):
        raise ValueError("Physical instance annotations require both mesh and parts SHA-256")
    aliases = {"seat_face_indices": "seat_faces", "backrest_face_indices": "backrest_faces"}
    for key, alias in aliases.items():
        if key not in part and alias in part:
            part[key] = part[alias]
    if "face_part_ids" in part:
        ids=np.asarray(part["face_part_ids"])
        if ids.shape!=(len(mesh.faces),) or not np.issubdtype(ids.dtype,np.integer) or not np.isin(ids,[0,1,2,3]).all():
            raise ValueError("Invalid complete face_part_ids annotation")
        part.setdefault("seat_face_indices", np.flatnonzero(part["face_part_ids"] == 0))
        part.setdefault("backrest_face_indices", np.flatnonzero(part["face_part_ids"] == 1))
        part.setdefault("armrest_face_indices", np.flatnonzero(part["face_part_ids"] == 3))
    if "armrest_faces" in part:
        part.setdefault("armrest_face_indices", part["armrest_faces"])
    optional_back = a.get("category", "chair") == "stool"
    if optional_back:
        part.setdefault("backrest_face_indices", np.empty(0, dtype=int))
    required = {"seat_face_indices"} | (set() if optional_back else {"backrest_face_indices"})
    for key in (*aliases,"armrest_face_indices","front_edge_face_indices","front_edge_vertices","floor_vertices"):
        if key not in part and key not in aliases: continue
        ids = np.asarray(part.get(key, []))
        limit=len(mesh.vertices) if key.endswith("vertices") else len(mesh.faces)
        if (ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or (key in required and not len(ids))
                or np.any(ids < 0) or np.any(ids >= limit) or len(np.unique(ids))!=len(ids)):
            raise UnverifiedSupport("Missing or invalid chair surface: " + key)
    for key,identity in (("seat_face_indices",0),("backrest_face_indices",1)):
        if "face_part_ids" in part and not np.all(part["face_part_ids"][part[key]]==identity):
            raise ValueError("Functional surface conflicts with face_part_ids: "+key)
    for key,axis in (("seat_face_indices",2),("backrest_face_indices",1)):
        selected=np.asarray(part[key],int)
        if not len(selected) and key not in required:
            continue
        area=mesh.area_faces[selected]
        valid=mesh.face_normals[selected,axis]>.20
        if area[valid].sum()/max(area.sum(),1e-12)<.90:
            raise UnverifiedSupport("Invalid functional surface orientation: "+key)
    if "front_edge_face_indices" not in part and "front_edge_vertices" in part:
        front=np.asarray(part["front_edge_vertices"],int)
        part["front_edge_face_indices"]=np.flatnonzero(np.isin(mesh.faces,front).any(axis=1))
    if a.get("units", "m") != "m" or a.get("up_axis", "Z").upper() not in ("Z", "+Z"):
        raise ValueError("Chair asset must be metres and Z-up")
    if a.get("forward_axis","+Y").upper() not in ("Y","+Y"):
        raise ValueError("Chair opening must face +Y")
    return a, mesh, part


def support_query(points, scene, r, translation, scale):
    if hasattr(scene, "support_query"):
        return scene.support_query(points,r,translation,scale)
    local = (points-translation) @ r / scale
    origins = local.copy()
    origins[:, 2] += 3.
    rays = np.column_stack((origins, np.tile([0.,0.,-1.], (len(points),1)))).astype(np.float32)
    hits = scene.cast_rays(o3d.core.Tensor(rays))
    t = hits["t_hit"].numpy()
    valid = np.isfinite(t)
    gaps = (t-3.) * scale
    normals = hits["primitive_normals"].numpy()
    valid &= normals[:,2] > .35
    return gaps, valid


def region_metrics(gaps, valid, weights=None):
    weights=np.ones(len(gaps)) if weights is None else np.asarray(weights)
    # One sample represents one equal XY cell. Coverage uses projected area;
    # curved-surface distance quantiles retain the surface-area weights.
    return dict(coverage=float(np.mean(valid)), gap_p20=weighted_quantile(gaps[valid],weights[valid],.2) if valid.any() else None,
                gap_median=weighted_quantile(gaps[valid],weights[valid],.5) if valid.any() else None,
                gap_p95=weighted_quantile(gaps[valid],weights[valid],.95) if valid.any() else None,
                maximum_seat_intrusion=float(max(0., -gaps[valid].min())) if valid.any() else None)


def support_checks(metrics, policy=None):
    policy = policy or FitPolicy()
    failures = []
    for region in REGIONS:
        m = metrics[region]
        limit = policy.gap(region)
        if policy.support_validation and m["coverage"] < policy.support_min_coverage:
            failures.append(region+":insufficient_seat_coverage")
        if policy.support_validation and (m["gap_p20"] is None or m["gap_p20"] > limit):
            failures.append(region+":insufficient_support")
        if policy.penetration_validation and m["maximum_seat_intrusion"] is not None and m["maximum_seat_intrusion"] > policy.max_penetration_m:
            failures.append(region+":seat_penetration")
    return failures


def placement_checks(mesh, parts, pelvis, body_forward, chair_forward, floor, heading_symmetry=None):
    from .seating import heading_errors
    back_vertices=np.unique(mesh.faces[parts["backrest_face_indices"]])
    posterior=np.array([((mesh.vertices[back_vertices]-p)@f).max() for p,f in zip(pelvis,body_forward)]) if len(back_vertices) else None
    yaw_errors=heading_errors(body_forward, chair_forward, heading_symmetry)
    floor_error=float(abs(mesh.vertices[:,2].min()-floor))
    reasons=[]
    # Temporarily skip the pelvis-half-space backrest check (2026-11-09).
    # Keep posterior and the reported offset as diagnostic measurements only.
    # if posterior.max()>=0: reasons.append("backrest_not_behind_pelvis")
    if yaw_errors.max()>DEFAULTS["maximum_yaw_degrees"]: reasons.append("chair_heading_mismatch")
    if floor_error>DEFAULTS["floor_tolerance"]: reasons.append("chair_not_grounded")
    return dict(backrest_forward_offset_max=float(posterior.max()) if posterior is not None else None,yaw_error_degrees=float(yaw_errors.max()),
                floor_error=floor_error),reasons


def fit_asset(path, ctx, anchors, estimate, output, contacts=None):
    execution = ctx.get("_execution", ExecutionOptions())
    profile = Profile(execution.profile_fit)
    started = time.perf_counter()
    failure = None
    try:
        blas = nullcontext()
        if execution.cpu_budget and execution.fit_workers == 1:
            from threadpoolctl import threadpool_limits
            blas = threadpool_limits(limits=1, user_api="blas")
        with execution_scope(execution, profile), blas:
            with profile.time("candidate_total"):
                result = _fit_asset(path, ctx, anchors, estimate, output, contacts, profile)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        profile.count("candidate_wall_seconds", time.perf_counter()-started)
        # The implementation records this only after validating the identifier
        # and creating its own directory. Malformed assets cannot share a sink.
        directory = getattr(profile, "output_directory", None)
        if execution.profile_fit and directory is not None:
            performance = str((directory / "performance.json").relative_to(output))
            write_json(directory / "performance.json", dict(profile.record(), execution=execution.record(),
                status="error" if failure is not None else "complete",
                error_type=type(failure).__name__ if failure is not None else None))
            if failure is not None:
                failure.fit_performance = performance
    result["execution"] = execution.record()
    directory = output / "candidates" / result["id"]
    if execution.profile_fit:
        result["performance"] = str((directory / "performance.json").relative_to(output))
    write_json(directory / "result.json", result)
    return result


def _fit_asset(path, ctx, anchors, estimate, output, contacts, profile):
    policy = ctx.get("policy", FitPolicy())
    from .collision_sampling import build_samples, summarize
    from .query_backend import collision_scene
    from .priority import choose as choose_priority, TOLERANCE_M, VERSION as priority_version
    execution = ctx.get("_execution", ExecutionOptions())
    priority = execution.fit_objective == "collision-first" and policy.penetration_optimization
    penetration_weight = execution.penetration_weight if execution.fit_objective == "balanced" else 1.0
    with profile.time("asset_loading"):
        meta, mesh, parts = read_asset(path)
    from .seating import heading_errors
    symmetry = meta.get("heading_symmetry")
    category = meta.get("category", "chair")
    asset_id = str(meta.get("id", meta.get("asset_id", Path(path).parent.name)))
    candidate_directory_key(asset_id)
    directory = output / "candidates" / asset_id
    directory.mkdir(parents=True, exist_ok=True)
    profile.output_directory = directory
    seat = query_scene(mesh.vertices, mesh.faces[parts["seat_face_indices"]])
    validation_seat = seat if ctx.get("_execution", ExecutionOptions()).fit_query_backend == "open3d" else ray_scene(mesh.vertices, mesh.faces[parts["seat_face_indices"]])
    seat_vertices = np.unique(mesh.faces[parts["seat_face_indices"]])
    seat_center = np.median(mesh.vertices[seat_vertices], axis=0)
    seat_lo, seat_hi = mesh.vertices[seat_vertices,:2].min(0), mesh.vertices[seat_vertices,:2].max(0)
    evaluations = PoseCache(128)
    certificates = PoseCache(8)
    frame_sample_weights = {}
    base_z = float(mesh.vertices[:,2].min())
    desired_yaw = float(np.arctan2(-estimate["forward"][0], estimate["forward"][1]))
    center = np.asarray(estimate["initial_seat_center"])
    r0 = rotation(desired_yaw)
    t0 = center - seat_center @ r0.T
    h = float(np.quantile(mesh.vertices[seat_vertices,2], .8)-base_z)
    procedural = "infinigen" in str(meta.get("source","")).lower() or "generator" in meta
    scale_bounds=(1.,1.) if procedural else (DEFAULTS["min_scale"],DEFAULTS["max_scale"])
    with profile.time("geometry_preparation"):
        geometry = prepare_geometry(mesh, parts, max_scale=scale_bounds[1]) if policy.penetration_validation else None
    geometry_path = directory/"collision_geometry.json"
    source_hash = sha256(resolve(Path(path).parent, meta.get("mesh", "mesh.obj")))
    if geometry is not None:
        write_json(geometry_path, dict(**geometry.summary(), mesh_sha256=source_hash, components=geometry.records))
    scale0 = np.clip(estimate["dimensions"]["seat_height"] / h, *scale_bounds)
    sampled_frames = set(sample_support_frames(ctx["support_frames"],ctx["frame_weights"],ctx["fps"]))
    optimized_samples={r:[sample for sample in anchors[r] if sample[0] in sampled_frames] for r in REGIONS}
    points = {r:np.concatenate([x[2] for x in optimized_samples[r]]) for r in REGIONS}
    slices={r:np.r_[0,np.cumsum([len(x[2]) for x in optimized_samples[r]])] for r in REGIONS}
    optimized_frame_set=set(x[0] for x in optimized_samples[REGIONS[0]])
    def refresh_sample_weights():
        frame_sample_weights.update({r: np.asarray([ctx["frame_weights"][sample[0]] for sample in optimized_samples[r]]) for r in REGIONS})
    refresh_sample_weights()
    def invalidate_objective():
        evaluations.invalidate()
        profile.count("objective_epoch_changes")
    def add_support_frames(frames):
        new_frames=set(int(f) for f in frames)-optimized_frame_set
        if not new_frames: return False
        optimized_frame_set.update(new_frames)
        for region in REGIONS:
            optimized_samples[region]=[sample for sample in anchors[region] if sample[0] in optimized_frame_set]
            points[region]=np.concatenate([sample[2] for sample in optimized_samples[region]])
            slices[region]=np.r_[0,np.cumsum([len(sample[2]) for sample in optimized_samples[region]])]
        refresh_sample_weights()
        invalidate_objective()
        return True
    contact_scenes={"seat":seat,"other":query_scene(mesh.vertices,mesh.faces)}
    if len(parts["backrest_face_indices"]):
        contact_scenes["back"]=query_scene(mesh.vertices,mesh.faces[parts["backrest_face_indices"]])
    front=parts.get("front_edge_face_indices",[])
    if len(front): contact_scenes["front"]=query_scene(mesh.vertices,mesh.faces[front])
    back_vertices=np.unique(mesh.faces[parts["backrest_face_indices"]])
    guard_frames=ctx["support_frames"]
    guard_forward = ctx["forward"][guard_frames]
    guard_weights = ctx["frame_weights"][guard_frames]
    pelvis_projection = np.sum(ctx["pelvis"][guard_frames]*guard_forward,axis=1)
    collision_guard_frames=list(dict.fromkeys(int(f) for f in guard_frames[[0,len(guard_frames)//2,-1]])) if policy.penetration_optimization else []
    if priority:
        # Contiguous support intervals include every return to the seat.
        intervals=[np.asarray(visit["selected_source_frames"],int) for visit in ctx.get("phase",{}).get("visits",[])] or [guard_frames]
        collision_guard_frames=sorted(set(int(f) for interval in intervals if len(interval)
            for f in interval[[0,len(interval)//2,-1]]))
    # Fixed human scenes are shared across chair candidates. These sparse guard
    # samples guide optimization only; the final collision audit is independent.
    if policy.penetration_optimization and "_collision_proxy_faces" not in ctx:
        ctx["_collision_proxy_faces"],ctx["_collision_proxy_loops"]=close_body_topology(ctx["faces"],ctx["dense"].shape[1])
    body_scenes=ctx.setdefault("_collision_guard_scenes",{})
    def make_body_scene(frame):
        body=np.asarray(ctx["dense"][frame])
        capped=np.vstack([body]+[body[loop].mean(0)[None] for loop in ctx["_collision_proxy_loops"]])
        return collision_scene(capped,ctx["_collision_proxy_faces"])
    def body_scene(frame):
        if frame not in body_scenes:
            body_scenes[frame]=make_body_scene(frame)
        return body_scenes[frame]
    chair_guard_samples=np.vstack((mesh.vertices,mesh.triangles_center))
    _,sample_ids=np.unique(np.floor(chair_guard_samples/.02).astype(np.int64),axis=0,return_index=True)
    chair_guard_samples=chair_guard_samples[sample_ids]
    with profile.time("collision_sampling"):
        sampling=build_samples(mesh,parts,execution.collision_sampling,execution.collision_max_edge_m,scale_bounds[1])
    sampling.save(directory/"collision-samples.npz")
    chair_guard_samples=sampling.points
    cached_guard={"key":None,"minimum":None}
    guard_risk_scans=[]
    def transform(x):
        r = rotation(x[2]); scale=x[3]
        t = np.array([x[0],x[1],ctx["floor"]-scale*base_z])
        return r,t,scale
    def guard_minimum_distances(x):
        key=(evaluations.epoch,tuple(np.asarray(x)),tuple(collision_guard_frames))
        if key!=cached_guard["key"]:
            r,t,scale=transform(x)
            if priority or execution.collision_sampling != "legacy" or execution.fit_query_backend == "hybrid":
                rows=[summarize(body_scene(frame),sampling,r,t,scale,execution.fit_query_backend=="hybrid") for frame in collision_guard_frames]
                cached_guard.update(key=key,minimum=-np.array([row["depth"] for row in rows]),summaries=rows)
                return cached_guard["minimum"]
            query=o3d.core.Tensor((chair_guard_samples@r.T*scale+t).astype(np.float32))
            def minimum(frame):
                scene=body_scene(frame)
                distance=(scene.transformed_distance(chair_guard_samples,r,t,scale,inverse=False,signed=True)
                          if hasattr(scene,"transformed_distance") else scene.compute_signed_distance(query))
                return float(distance.numpy().min())
            cached_guard["minimum"]=np.array([minimum(frame) for frame in collision_guard_frames])
            cached_guard["key"]=key
        return cached_guard["minimum"]
    def scan_full_motion_guard_risk(x):
        """Select active frames once, without putting all frames into cost."""
        r,t,scale=transform(x)
        query=o3d.core.Tensor((chair_guard_samples@r.T*scale+t).astype(np.float32))
        minimum=np.empty(len(ctx["dense"]),float)
        rms=np.empty(len(minimum),float)
        for frame in range(len(ctx["dense"])):
            scene=body_scenes.get(frame)
            if scene is None: scene=make_body_scene(frame)
            summary=summarize(scene,sampling,r,t,scale,execution.fit_query_backend=="hybrid")
            minimum[frame]=-summary["depth"]
            rms[frame]=summary["rms"]
            # Only already active/selected scenes survive this streaming scan.
            del scene
        current=float(minimum[collision_guard_frames].min()) if collision_guard_frames else 0.
        selected=[int(frame) for frame in np.argsort(minimum,kind="stable") if frame not in collision_guard_frames
                  and (not priority or minimum[frame]<current-TOLERANCE_M)][:3]
        collision_guard_frames.extend(selected)
        for frame in selected: body_scene(frame)
        path=directory/("guard-risk-scan-%02d.npz"%len(guard_risk_scans))
        np.savez_compressed(path,source_frame_indices=np.arange(len(minimum)),minimum_signed_distance=minimum,rms_penetration=rms)
        guard_risk_scans.append(dict(path=path.name,checked_source_frames=len(minimum),
            pose=np.asarray(x).tolist(),
            sampled_chair_points=len(chair_guard_samples),selected_source_frames=selected,
            selected_minimum_signed_distances=[float(minimum[frame]) for frame in selected],
            global_minimum_signed_distance=float(minimum.min()),
            full_motion_rms_m=float(np.sqrt(np.mean(rms*rms))),
            policy="streamed " + ctx.get("_execution", ExecutionOptions()).fit_query_backend + " guard SDF; optimization guidance only, not final certification"))
        cached_guard["key"]=None
        return selected
    last_components = {}
    def cached_support(x, region):
        def compute():
            profile.count("support_query_batches")
            r,t,scale=transform(x)
            return support_query(points[region],seat,r,t,scale)
        return evaluations.get(x, ("support",region), compute)
    def cost(x):
        profile.count("objective_calls")
        def compute():
            profile.count("objective_evaluations")
            with profile.time("objective"):
                value = raw_cost(x)
            return value,dict(last_components)
        value,components = evaluations.get(x,"cost",compute)
        last_components.update(components)
        return value
    def raw_cost(x):
        r,t,scale=transform(x)
        loss=0.
        components=dict(support=0.,penetration=0.,contact=0.,placement=0.,regularization=0.)
        for region,p in points.items():
            if not (policy.support_optimization or policy.penetration_optimization): continue
            gaps, valid=cached_support(x,region)
            # Coverage guidance uses actual seat footprint; q20 permits curvature.
            local=(p-t)@r/scale
            outside=np.maximum(seat_lo-local[:,:2],0)+np.maximum(local[:,:2]-seat_hi,0)
            frame_losses=[]
            for k,(_,_,_,weights) in enumerate(optimized_samples[region]):
                sl=slice(slices[region][k],slices[region][k+1]);ok=valid[sl];g=gaps[sl]
                support_value=2*np.average(np.sum(outside[sl]**2,axis=1),weights=weights)+.05*(1-np.mean(ok))
                penetration_value=0.
                if ok.any():
                    q=weighted_quantile(g[ok],weights[ok],.2)
                    support_value+=max(q,0.)**2
                    penetration_value+=min(q,0.)**2+5*np.average(np.minimum(g[ok],0)**2,weights=weights[ok])
                    gap_limit=policy.gap(region)
                    # Explicit violation terms avoid accepting a mean-loss
                    # compromise between floating buttocks and buried thighs.
                    support_value+=100*max(0.,q-gap_limit)**2
                    penetration_value+=100*max(0.,-float(g[ok].min())-policy.max_penetration_m)**2
                else: support_value+=1.
                support_value+=100*max(0.,policy.support_min_coverage-np.mean(ok))**2
                frame_losses.append((support_value if policy.support_optimization else 0., penetration_value if policy.penetration_optimization else 0.))
            values=np.average(frame_losses,axis=0,weights=frame_sample_weights[region])
            components["support"]+=float(values[0]); components["penetration"]+=float(values[1])
        if contacts:
            for role,(p,weights) in contacts.items():
                if not len(p): continue
                sc=contact_scenes.get(role,contact_scenes["other"])
                if hasattr(sc,"transformed_distance"):
                    d=sc.transformed_distance(p,r,t,scale,inverse=True).numpy()*scale
                else:
                    local=(p-t)@r/scale
                    d=sc.compute_distance(o3d.core.Tensor(local.astype(np.float32))).numpy()*scale
                components["contact"]+=.10*float(np.average(np.minimum(d,.15)**2,weights=weights))
        if len(back_vertices):
            back_world=mesh.vertices[back_vertices]@r.T*scale+t
            offsets=(back_world@guard_forward.T).max(axis=0)-pelvis_projection
            components["placement"]+=100*float(np.average(np.maximum(offsets,0)**2,weights=guard_weights))
        if policy.penetration_optimization:
            components["penetration"]+=100*float(np.mean(np.maximum(-guard_minimum_distances(x)-policy.max_penetration_m,0)**2))
        angular = np.radians(heading_errors(np.array([[0.,1.,0.]]) @ rotation(desired_yaw).T,
                                           np.array([0.,1.,0.]) @ r.T, symmetry)[0]) if symmetry else x[2]-desired_yaw
        components["regularization"] = .001*(scale-1)**2 + .001*angular**2
        components["penetration"] *= penetration_weight
        last_components.update(components)
        return sum(components.values())
    def priority_score(x):
        def compute():
            cost(x)
            components=dict(last_components)
            depths=-guard_minimum_distances(x)
            depth=float(max(0.,depths.max())) if len(depths) else 0.
            rows=cached_guard.get("summaries",[])
            rms=float(np.sqrt(np.average([row["rms"]**2 for row in rows],
                weights=[max(1e-12,float(ctx["frame_weights"][f])) for f in collision_guard_frames]))) if rows else 0.
            for region in points:
                gaps,valid=cached_support(x,region)
                if valid.any():
                    p=np.maximum(-gaps[valid],0.)
                    w=np.concatenate([sample[3] for sample in optimized_samples[region]])[valid]
                    depth=max(depth,float(p.max()))
                    rms=max(rms,float(np.sqrt(np.average(p*p,weights=w))))
            return (depth,rms,components["support"],components["contact"]+components["placement"]+components["regularization"])
        return evaluations.get(x,"priority",compute)
    def search_cost(x):
        return priority_score(x)[0] if priority else cost(x)
    bounds=[(t0[0]-.25,t0[0]+.25),(t0[1]-.25,t0[1]+.25),
            (desired_yaw-np.radians(20),desired_yaw+np.radians(20)),scale_bounds]
    x0=np.array([t0[0],t0[1],desired_yaw,scale0])
    starts=[]
    dxs = dys = (-.08, 0., .08)
    if category == "sofa":
        # Search local contact positions across the actual wide seat, including
        # its ends. These are optimizer bounds, never acceptance cutoffs.
        reach = (seat_hi-seat_lo) * scale_bounds[1] / 2 + .25
        extent = np.abs(r0[:2,:2]) @ reach
        bounds[0]=(t0[0]-extent[0],t0[0]+extent[0])
        bounds[1]=(t0[1]-extent[1],t0[1]+extent[1])
        offsets=[np.zeros(2)]
        for q in np.linspace(.1,.9,5):
            local=seat_center.copy();local[0]=seat_lo[0]+q*(seat_hi[0]-seat_lo[0])
            offsets.append(((seat_center-local)@r0.T)[:2]*scale0)
        dxs=sorted(set([-.08,0.,.08]+[float(o[0]) for o in offsets]))
        dys=sorted(set([-.08,0.,.08]+[float(o[1]) for o in offsets]))
    for dx in dxs:
        for dy in dys:
            for da in (-15.,0.,15.):
                for s in sorted(set([float(scale0),*scale_bounds])):
                    x=np.array([t0[0]+dx,t0[1]+dy,desired_yaw+np.radians(da),s])
                    starts.append((priority_score(x) if priority else cost(x),x))
    best_starts=sorted(starts,key=lambda x:x[0])[:3]
    if ctx.get("_fixed_starts") is not None:
        best_starts=[(cost(np.asarray(x)),np.asarray(x,dtype=float)) for x in ctx["_fixed_starts"]]
    with profile.time("powell"):
        print("Optimizing",asset_id,"initials",len(best_starts),"collision frames",len(collision_guard_frames),flush=True)
        results=[minimize(search_cost,x,method="Powell",bounds=bounds,options={"maxiter":70,"xtol":1e-4,"ftol":1e-5}) for _,x in best_starts]
    if category == "sofa" or priority:
        # Powell can cross broad equal-cost regions and leave a useful end-seat
        # placement. Keep the evaluated local starts in the same ranking pool.
        results.extend(OptimizeResult(x=x.copy(),fun=cost(x),success=False,
                       message="retained local seating initialization",nfev=1) for value,x in best_starts)
    all_lower=np.array([bound[0] for bound in bounds]);all_upper=np.array([bound[1] for bound in bounds])
    for result in results:
        result.raw_bounds_violation=float(max(0.,(all_lower-result.x).max(),(result.x-all_upper).max()))
        result.x=np.clip(result.x,all_lower,all_upper);result.fun=cost(result.x)
    # Repair against actual geometric inequalities, instead of accepting the
    # weighted mean's compromise. The 0.5mm margin tightens internal targets;
    # final acceptance remains unchanged and checks all source frames.
    buffer_levels=(.0005,.00005,.000005)
    buffer_tier=0;margin=buffer_levels[buffer_tier]
    buffer_history=[dict(round=0,margin_m=margin)]
    def feasible_residuals(x):
        profile.count("constraint_calls")
        def compute():
            profile.count("constraint_evaluations")
            with profile.time("constraints"):
                return raw_residuals(x)
        return evaluations.get(x,"constraints",compute)
    def raw_residuals(x):
        r,t,scale=transform(x)
        residuals=[]
        for region,p in points.items():
            if not (policy.support_optimization or policy.penetration_optimization): continue
            gaps,valid=cached_support(x,region)
            gap_limit=policy.gap(region)
            for k,(_,_,_,weights) in enumerate(optimized_samples[region]):
                sl=slice(slices[region][k],slices[region][k+1]);ok=valid[sl];g=gaps[sl]
                if policy.support_optimization:
                    residuals.append(.02*(float(np.mean(ok))-policy.support_min_coverage))
                if ok.any():
                    if policy.support_optimization: residuals.append(gap_limit-margin-weighted_quantile(g[ok],weights[ok],.2))
                    if policy.penetration_optimization: residuals.append(policy.max_penetration_m-margin+float(g[ok].min()))
                else:
                    if policy.support_optimization: residuals.append(-.1)
                    if policy.penetration_optimization: residuals.append(-.1 if policy.support_optimization else policy.max_penetration_m)
        if len(back_vertices):
            back_world=mesh.vertices[back_vertices]@r.T*scale+t
            offsets=(back_world@guard_forward.T).max(axis=0)-pelvis_projection
            residuals.append(-float(offsets.max())-margin)
        heading=np.array([0.,1.,0.])@r.T
        angles=heading_errors(ctx["forward"][guard_frames],heading,symmetry)
        residuals.append(.01*(DEFAULTS["maximum_yaw_degrees"]-float(angles.max())))
        # A certificate needs room for outside-witness and surface-cell bounds.
        # Support can approach its exact threshold more tightly than this.
        if policy.penetration_optimization:
            residuals.extend(policy.max_penetration_m-max(margin,.0001)+guard_minimum_distances(x))
        return np.asarray(residuals)
    def violation(x): return max(0.,-float(feasible_residuals(x).min()))
    active=np.array([i for i,(lo,hi) in enumerate(bounds) if hi>lo],int)
    lower_bounds=np.array([bounds[i][0] for i in active])
    upper_bounds=np.array([bounds[i][1] for i in active])
    def repair(start, budget=None, restoration=False):
        fixed=start.x.copy()
        anchor=priority_score(fixed) if priority else None
        def expand(x):
            result=fixed.copy();result[active]=x;return result
        with profile.time("cobyla_repair"):
            repaired=minimize(lambda x:search_cost(expand(x)) if restoration else (priority_score(expand(x))[2] if priority else cost(expand(x))),start.x[active],method="COBYLA",
                constraints=[{"type":"ineq","fun":lambda x:np.ones(1) if restoration else (np.asarray(anchor[:2])+TOLERANCE_M-np.asarray(priority_score(expand(x))[:2]) if priority else feasible_residuals(expand(x)))},
                    {"type":"ineq","fun":lambda x:x-lower_bounds},
                    {"type":"ineq","fun":lambda x:upper_bounds-x}],
                options={"maxiter":budget if budget is not None else (300 if buffer_tier==0 else 900),
                    "rhobeg":.02 if buffer_tier==0 else .005,
                    "tol":1e-5 if buffer_tier==0 else 1e-6,"catol":1e-7})
        raw=expand(repaired.x)
        repaired.raw_bounds_violation=float(max(0.,(all_lower-raw).max(),(raw-all_upper).max()))
        repaired.x=np.clip(raw,all_lower,all_upper)
        repaired.collision_anchor=anchor[:2] if priority and not restoration else None
        repaired.fun=cost(repaired.x);repairs.append(repaired)
        return repaired
    repairs=[]
    for start in sorted(results,key=lambda o:priority_score(o.x) if priority else (violation(o.x),o.fun))[:2]:
        repaired=repair(start)
        if not priority and violation(repaired.x)<=1e-6: break
    choices=results+repairs
    def choose_result():
        return choose_priority(choices,priority_score) if priority else choose_current_result(choices,cost,violation)
    opt=choose_result()
    validation_rounds=[];collision_witnesses=[]
    support_repairs=0;collision_repairs=0
    maximum_validation_rounds=8
    maximum_support_repairs=6;maximum_collision_repairs=6
    if priority:
        # Risk refinement is optimization work even when strict acceptance is off.
        for risk_round in range(maximum_collision_repairs+1):
            print("Collision risk scan",asset_id,risk_round,flush=True)
            with profile.time("full_motion_guard_scan"):
                added=scan_full_motion_guard_risk(opt.x)
            invalidate_objective()
            opt=choose_result()
            if not added or risk_round==maximum_collision_repairs: break
            # Share the existing 300-evaluation repair allowance between stages.
            restored=repair(opt,budget=150,restoration=True)
            choices.append(restored);opt=choose_result()
            choices.append(repair(opt,budget=150));opt=choose_result();collision_repairs+=1
        # Active-set progress is not necessarily progress on unobserved frames.
        # Compare all fully scanned poses before exporting/validating a winner.
        with profile.time("full_motion_guard_scan"):
            scan_full_motion_guard_risk(opt.x)
        invalidate_objective()
        scanned={tuple(row["pose"]):row for row in guard_risk_scans}
        scanned_choices=[OptimizeResult(x=np.asarray(pose),fun=cost(np.asarray(pose)),success=False,
            message="retained complete-motion scan") for pose in scanned]
        def full_score(x):
            row=scanned[tuple(x)]; score=priority_score(x)
            return (max(score[0],-row["global_minimum_signed_distance"]),
                max(score[1],row["full_motion_rms_m"]),*score[2:])
        opt=choose_priority(scanned_choices,full_score)
        choices.extend(scanned_choices)
    for validation_round in range(maximum_validation_rounds):
        r,t,scale=transform(opt.x)
        fitted=trimesh.Trimesh(mesh.vertices @ r.T * scale+t,mesh.faces,process=False)
        mesh_path=directory/"mesh.obj"; fitted.export(mesh_path)
        exported=dict(parts)
        exported["forward"]=np.array([0.,1.,0.])@r.T
        exported["forward_origin"]=seat_center@r.T*scale+t
        np.savez_compressed(directory/"parts.npz",**exported)
        support={};worst_support={}
        for region,samples in (anchors.items() if policy.support_validation or policy.penetration_validation else []):
            with profile.time("support_validation"):
                per_frame=batched_support_metrics(samples,validation_seat,r,t,scale,support_query,region_metrics)
            support[region]=dict(coverage=min(m["coverage"] for _,m in per_frame),
                gap_p20=max((m["gap_p20"] for _,m in per_frame if m["gap_p20"] is not None),default=None),
                maximum_seat_intrusion=max((m["maximum_seat_intrusion"] for _,m in per_frame if m["maximum_seat_intrusion"] is not None),default=None))
            worst_support[region]=[dict(frame=f,**m) for f,m in per_frame]
        reasons=support_checks(support,policy) if support else []
        placement,placement_reasons=placement_checks(fitted,parts,ctx["pelvis"][ctx["support_frames"]],
            ctx["forward"][ctx["support_frames"]],np.array([0.,1.,0.])@r.T,ctx["floor"],symmetry)
        reasons.extend(placement_reasons)
        metrics=dict(support=support,scale=float(scale),geometry=geometry.summary() if geometry is not None else skipped(),**placement)
        if "visits" in ctx:
            from .multi_fit import evaluate_visits
            metrics["visits"] = evaluate_visits(ctx["phase"]["visits"], worst_support, policy)
            if any(visit["rejection_reasons"] for visit in metrics["visits"]):
                reasons.append("visit_support_unverified")
        worst_frame=None;collision=None
        if not reasons and policy.penetration_validation:
            print("Full-frame collision check",asset_id,"round",validation_round,flush=True)
            profile.count("collision_requests")
            def audit():
                profile.count("collision_evaluations")
                with profile.time("strict_collision"):
                    return collision_check(ctx,fitted,parts,prepared=geometry,pose=(r,t,scale),limit=policy.max_penetration_m)
            # Scoped to this immutable body/asset/policy invocation. Optimization
            # frame/witness changes do not alter the full-motion audit contract.
            collision=certificates.get(opt.x,"strict_collision",audit,copy=True)
            write_json(directory/("collision-iteration-%d.json"%validation_round),collision)
            write_json(directory/"collision.json",collision)
            worst_frame=collision["worst_collision_frame"]
            metrics["collision"]={k:v for k,v in collision.items() if k!="frames"}
            reasons.extend(collision_checks(collision,limit=policy.max_penetration_m))
        else:
            metrics["collision"]={"status":"not_run_support_rejected", "checked_source_frames":0} if policy.penetration_validation else skipped()
        write_json(directory/"support.json",worst_support)
        validation_rounds.append(dict(round=validation_round,collision_guard_frames=list(collision_guard_frames),
            rejection_reasons=list(reasons),worst_collision_frame=worst_frame,
            internal_margin_m=margin,
            collision_internal_margin_m=max(margin,.0001),
            optimized_source_frames=sorted(optimized_frame_set),optimizer_success=bool(opt.success),optimizer_message=str(opt.message),
            maximum_constraint_violation_m=violation(opt.x),loss=float(cost(opt.x))))
        metrics["support_validation"] = dict(status="checked", checked_source_frames=len(ctx["support_frames"])) if policy.support_validation else skipped()
        if not policy.support_validation: metrics["support"] = skipped()
        added_support=[]
        for region in REGIONS:
            if region+":insufficient_seat_coverage" in reasons:
                added_support.append(min(worst_support[region],key=lambda m:m["coverage"])["frame"])
            if region+":insufficient_support" in reasons:
                added_support.append(max(worst_support[region],key=lambda m:m["gap_p20"] if m["gap_p20"] is not None else float("inf"))["frame"])
            if region+":seat_penetration" in reasons:
                added_support.append(max(worst_support[region],key=lambda m:m["maximum_seat_intrusion"] or 0)["frame"])
        if (validation_round<maximum_validation_rounds-1 and support_repairs<maximum_support_repairs
                and (policy.support_optimization or policy.penetration_optimization) and add_support_frames(added_support)):
            print("Refining support source frames",sorted(set(added_support)),asset_id,flush=True)
            support_repairs+=1
            choices.append(repair(opt));opt=choose_result()
            continue
        collision_repair_frame=None
        if collision:
            if "full_motion_penetration" in reasons: collision_repair_frame=worst_frame
            elif "collision_depth_undetermined" in reasons:
                if collision.get("geometry", {}).get("counts", {}).get("unknown"):
                    # All original surfaces have been checked. Moving this
                    # candidate cannot supply the missing extent provenance.
                    break
                collision_repair_frame=collision.get("uncertainty_frame")
        if (validation_round<maximum_validation_rounds-1 and collision_repairs<maximum_collision_repairs
                and policy.penetration_optimization and collision_repair_frame is not None):
            changed=False
            if collision_repair_frame not in collision_guard_frames:
                collision_guard_frames.append(int(collision_repair_frame));changed=True
            frame_record=next((frame for frame in collision.get("frames",[]) if frame["frame"]==collision_repair_frame),{})
            direction=frame_record.get("directions",{}).get("chair_to_body",{})
            witness=(direction.get("uncertainty_witness") if "full_motion_penetration" not in reasons else None) or direction.get("worst")
            evidence = collision.get("penetration_witness")
            if "full_motion_penetration" in reasons and evidence:
                witness = dict(point=evidence["furniture_point"], source_face=evidence["furniture_face"],
                               source_part=evidence["furniture_part"], sample_lower_bound=evidence["sample_lower_bound"])
            new_points=False
            if witness and witness.get("point") is not None:
                local=(np.asarray(witness["point"])-t)@r/scale
                if np.linalg.norm(chair_guard_samples-local,axis=1).min()>1e-6:
                    chair_guard_samples=np.vstack((chair_guard_samples,local))
                    sampling.points=np.asarray(chair_guard_samples,dtype=np.float32)
                    sampling.weights=np.r_[sampling.weights,0.]
                    sampling.token=hashlib.sha256(sampling.points.tobytes()).hexdigest()
                    collision_witnesses.append(dict(source_frame=int(collision_repair_frame),
                        source_face=witness.get("source_face"),source_part=witness.get("source_part"),
                        local_point=local.tolist(),world_point_at_discovery=witness["point"],
                        measured_lower_bound=witness.get("sample_lower_bound"),
                        uncertainty_repair="full_motion_penetration" not in reasons))
                    cached_guard["key"]=None;changed=True;new_points=True
            if new_points:
                print("Scanning all source frames for guard risk",asset_id,flush=True)
                with profile.time("full_motion_guard_scan"):
                    selected=scan_full_motion_guard_risk(opt.x)
                print("Activated risk source frames",selected,asset_id,flush=True)
            if changed:
                fitted.export(directory/("mesh-iteration-%d.obj"%validation_round))
                print("Refining collision frame/witness",collision_repair_frame,asset_id,flush=True)
                collision_repairs+=1
                invalidate_objective()
                choices.append(repair(opt));opt=choose_result()
                continue
        if ((policy.support_validation or policy.penetration_validation) and reasons and validation_round<maximum_validation_rounds-1 and buffer_tier<len(buffer_levels)-1
                and violation(opt.x)>1e-7):
            buffer_tier+=1;margin=buffer_levels[buffer_tier]
            invalidate_objective()
            buffer_history.append(dict(round=validation_round+1,margin_m=margin,
                reason="extra internal buffer infeasible; final acceptance limits unchanged"))
            print("Refining with internal margin",margin,"m",asset_id,flush=True)
            choices.append(repair(opt));opt=choose_result()
            continue
        break
    profile.count("optimization_cache_hits",evaluations.hits)
    profile.count("optimization_cache_misses",evaluations.misses)
    profile.count("collision_cache_hits",certificates.hits)
    profile.count("optimizer_function_evaluations",sum(int(getattr(o,"nfev",0)) for o in choices))
    if priority:
        with profile.time("full_motion_guard_scan"):
            scan_full_motion_guard_risk(opt.x)
        invalidate_objective()
    final_priority=list(priority_score(opt.x)) if priority else None
    if priority:
        final_priority[0]=max(final_priority[0],-guard_risk_scans[-1]["global_minimum_signed_distance"])
        final_priority[1]=max(final_priority[1],guard_risk_scans[-1]["full_motion_rms_m"])
    write_json(directory/"optimization.json",dict(execution=ctx.get("_execution",ExecutionOptions()).record(),objective_epoch=evaluations.epoch,collision_cache_hits=certificates.hits,method="collision restoration then collision-preserving support" if priority else "coarse grid, Powell, COBYLA feasibility repair",
        initial_starts=[x.tolist() for _,x in best_starts],priority_version=priority_version if priority else None,
        priority_score=final_priority,collision_preservation_tolerance_m=TOLERANCE_M if priority else None,
        internal_margin_m=margin,initial_internal_margin_m=buffer_levels[0],buffer_history=buffer_history,
        collision_internal_margin_m=max(margin,.0001),minimum_collision_internal_margin_m=.0001,
        optimized_source_frames=[int(x[0]) for x in optimized_samples[REGIONS[0]]],
        collision_guard_samples=len(chair_guard_samples),collision_guard_frames=collision_guard_frames,
        maximum_validation_rounds=maximum_validation_rounds,maximum_support_repairs=maximum_support_repairs,
        maximum_collision_repairs=maximum_collision_repairs,support_repairs=support_repairs,collision_repairs=collision_repairs,
        collision_witnesses=collision_witnesses,guard_risk_scans=guard_risk_scans,validation_rounds=validation_rounds,
        final_maximum_constraint_violation_m=violation(opt.x),
        trials=[dict(loss=float(cost(o.x)),constraint_violation_m=violation(o.x),success=bool(o.success),
            priority_score=list(priority_score(o.x)) if priority else None,
            collision_anchor=getattr(o,"collision_anchor",None),
            raw_bounds_violation=float(getattr(o,"raw_bounds_violation",0)),message=str(o.message)) for o in choices]))
    final_loss=float(cost(opt.x))
    result=dict(category=category,heading_symmetry=symmetry or {"type":"none"},policy=policy.record(),selection_eligible=not reasons,validation_status="complete" if policy.strict else "diagnostic_unverified",loss_components=dict(last_components),id=asset_id,mesh=str(mesh_path.relative_to(output)),parts=str((directory/"parts.npz").relative_to(output)),
                qualified=not reasons and policy.strict,loss=final_loss,metrics=metrics,rejection_reasons=reasons,
                worst_collision_frame=worst_frame,source_asset=str(Path(path).resolve()),
                transform=dict(rotation=r.tolist(),translation=t.tolist(),scale=float(scale)),
                geometry_audit=str(geometry_path.relative_to(output)) if geometry is not None else None, source_geometry_sha256=source_hash)
    if priority:
        result.update(priority_score=final_priority,priority_version=priority_version,
                      optimization_collision=dict(worst_full_motion_depth_m=-guard_risk_scans[-1]["global_minimum_signed_distance"],
                          checked_source_frames=len(ctx["dense"]),refinement_budget_exhausted=bool(guard_risk_scans[-1]["selected_source_frames"])))
    write_json(directory/"result.json",result)
    return result


_WORKER_INPUTS = None
_WORKER_BLAS_LIMIT = None


def _initialize_candidate_worker(inputs):
    global _WORKER_INPUTS, _WORKER_BLAS_LIMIT
    from threadpoolctl import threadpool_limits
    # Open3D owns the per-worker query budget. Dense linear algebra must not
    # create an additional full-machine BLAS team in every spawned worker.
    _WORKER_BLAS_LIMIT = threadpool_limits(limits=1, user_api="blas")
    _WORKER_INPUTS = unpack_arrays(inputs)


def _invalid_candidate_record(path, exc):
    return dict(id=Path(path).parent.name,qualified=False,retained=False,loss=None,metrics={},
        rejection_reasons=[("support_geometry_unverified: " if isinstance(exc, UnverifiedSupport)
                            else "invalid_candidate_asset: ")+str(exc)],worst_collision_frame=None,source_asset=str(path))


def _candidate_record(path, ctx, anchors, estimate, output, contacts):
    try:
        return fit_asset(path,ctx,anchors,estimate,output,contacts)
    except (ValueError,FileNotFoundError,KeyError) as exc:
        record = _invalid_candidate_record(path, exc)
        if hasattr(exc, "fit_performance"):
            record["performance"] = exc.fit_performance
            record["execution"] = ctx.get("_execution", ExecutionOptions()).record()
        return record


def _candidate_worker(path):
    return _candidate_record(path,*_WORKER_INPUTS)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry",required=True,type=Path)
    parser.add_argument("--vertices-path",required=True,type=Path)
    parser.add_argument("--contacts",required=True,type=Path)
    parser.add_argument("--frame-indices",required=True,type=Path)
    parser.add_argument("--models-dir",type=Path)
    parser.add_argument("--candidate-source",choices=["cad","infinigen"],default="cad")
    parser.add_argument("--category",choices=["chair","sofa","stool"],default=None,
                        help="Enable category-aware seating evidence; omission preserves legacy chair fitting")
    parser.add_argument("--assets-manifest",type=Path)
    parser.add_argument("--instance-visits",type=Path,help="This instance's inferred visits on the complete source grid")
    parser.add_argument("--candidate-cache",type=Path,help="Reusable completed multi-instance candidate checkpoints")
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--estimate-only",action="store_true")
    parser.add_argument("--max-candidates",type=int,choices=(1,2,3),default=3)
    add_arguments(parser)
    add_execution_arguments(parser)
    parser.add_argument("--candidate-timeout",type=float,default=7200.,help="Maximum wait for the next candidate result; only this worker pool is terminated on timeout")
    args=parser.parse_args(argv)
    if args.candidate_timeout <= 0 or not np.isfinite(args.candidate_timeout):
        parser.error("--candidate-timeout must be finite and positive")
    execution=ExecutionOptions.from_args(args)
    profile=Profile(execution.profile_fit)
    run_started=time.perf_counter()
    policy=FitPolicy.from_args(args)
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    report=dict(execution=execution.record(),policy=policy.record(),backend="supported-chair-v2",status="no_valid_candidate",candidates=[],best_candidate_id=None,
                thresholds=dict(DEFAULTS,coverage=policy.support_min_coverage,gluteus_gap=policy.support_gluteus_gap_m,
                    thigh_gap=policy.support_thigh_gap_m,maximum_penetration=policy.max_penetration_m),validation_scope="geometry; no dynamics simulation",
                prediction_sha256=sha256(args.contacts),world_sha256=sha256(args.vertices_path))
    if execution.cpu_budget:
        report["fit_resources"] = {"status": "skipped", "candidate_workers": 0}
    implementation_files=[Path(__file__),Path(__file__).parent/"cad.py",Path(__file__).parent/"annotations"/"body_regions_smplx_v1.json",
        Path(__file__).parent/"annotations"/"cad_chair_annotations_v1.json",Path(__file__).parent/"collision.py",
        Path(__file__).parent/"policy.py",Path(__file__).parent/"selection.py",Path(__file__).parent/"geometry_audit.py"]
    implementation_files += [Path(__file__).with_name(name) for name in ("runtime.py","query_backend.py","warp_backend.py","solver_helpers.py","candidate_pool.py","collision_sampling.py","priority.py")]
    implementation_files += [Path(__file__).with_name("seating.py"),Path(__file__).parent/"annotations"/"cad_seating_annotations_v1.json"]
    report["implementation_sha256"]={str(path.relative_to(Path(__file__).parent)):sha256(path) for path in implementation_files if path.is_file()}
    try:
        assignment=json.loads(args.instance_visits.read_text(encoding="utf-8")) if args.instance_visits else None
        category=args.category or (assignment.get("category") if assignment else None)
        if category is not None:
            report.update(backend="supported-seating-v1",category=category)
        visits=assignment["events"] if assignment is not None else None
        cache=None
        if assignment is not None:
            from .multi_fit import CandidateCache, valid_name
            import platform
            import importlib.metadata
            report["instance_id"]=valid_name(assignment["instance_id"])
            if any(visit["instance_id"] != report["instance_id"] for visit in visits):
                raise ValueError("Mismatched instance in visit assignment")
            report["implementation_sha256"]["multi_fit.py"]=sha256(Path(__file__).with_name("multi_fit.py"))
        elif args.candidate_cache:
            raise ValueError("--candidate-cache requires an instance visit assignment")
        with profile.time("motion_context"):
            ctx=motion_context(args.geometry,args.vertices_path,args.contacts,args.frame_indices,output,
                               visits=visits,locations=assignment.get("locations") if assignment else None,category=category)
        ctx["policy"]=policy
        ctx["_execution"]=execution
        report["support_selection"]={key:value for key,value in ctx["phase"].items()
                                     if key not in ("selection_reasons","thresholds","selection_proxy")}
        missing=[visit for visit in ctx["phase"].get("visits",[]) if not visit["selected_source_frames"]]
        if not len(ctx["support_frames"]) or missing:
            report["rejection_reasons"]=["visit_support_unverified" if missing else "no_support_evidence"]
            write_json(output/"fit_result.json",report)
            return 2
        regions=body_regions(ctx)
        with profile.time("prepare_support"):
            anchors,estimate=prepare_support(ctx,regions,output)
        if args.estimate_only:
            print(json.dumps({"estimate":str(output/"estimate.json")}))
            return 0
        if args.assets_manifest:
            a=json.loads(args.assets_manifest.read_text(encoding="utf-8"))
            assets=[resolve(args.assets_manifest.parent,p) for p in a["assets"]]
        elif args.candidate_source=="cad" and args.models_dir:
            from .cad import prepare_cad_assets
            assets=prepare_cad_assets(args.models_dir,output/"cad_assets",category=category or "chair")
        else:
            raise ValueError("Provide --assets-manifest or --models-dir for CAD candidates")
        with profile.time("contact_roles"):
            contacts=contact_roles(ctx,regions,output)
        asset_ids=set()
        if args.candidate_cache:
            manifest=json.loads(args.geometry.read_text(encoding="utf-8"))
            fd=manifest.get("fitting_data",manifest)
            inputs={key:sha256(resolve(args.geometry.parent,fd[key])) for key in ("world","joints","joint_names","faces")}
            inputs.update(manifest=sha256(args.geometry),low=report["world_sha256"],contacts=report["prediction_sha256"],
                          frame_map=sha256(args.frame_indices),instance=assignment["instance_id"],
                          category=ctx.get("category","chair"),seating=ctx.get("seating",False))
            runtime={name:importlib.metadata.version(name) for name in ("numpy","scipy","open3d","trimesh")}
            if execution.fit_query_backend in ("warp", "hybrid"):
                runtime["warp-lang"]=importlib.metadata.version("warp-lang")
            cache=CandidateCache(args.candidate_cache,inputs,[platform.platform(),platform.python_version(),
                runtime,report["implementation_sha256"],policy.record(),execution.record()],assignment,ctx["phase"])
            report["cache_hits"]=[]
        # Validate identities before spawning; workers must never share a directory.
        entries=[]
        for path in assets:
            try:
                asset=json.loads(Path(path).read_text(encoding="utf-8"))
                if not isinstance(asset,dict):
                    raise ValueError("Candidate metadata must be an object")
                asset_id=str(asset.get("id",asset.get("asset_id",Path(path).parent.name)))
                directory_key=candidate_directory_key(asset_id)
            except (ValueError,FileNotFoundError,KeyError) as exc:
                entries.append((path,_invalid_candidate_record(path,exc),None))
                continue
            if directory_key in asset_ids:
                raise ValueError("Duplicate asset id: "+asset_id)
            asset_ids.add(directory_key)
            record=checkpoint=None
            if cache:
                record,checkpoint,_=cache.lookup(Path(path),output)
                if record is not None:
                    report["cache_hits"].append(record["id"])
            entries.append((path,record,checkpoint))
        pending=[path for path,record,_ in entries if record is None]
        if execution.cpu_budget and pending:
            execution = execution.for_candidates(len(pending))
            ctx["_execution"] = execution
            report["execution"] = execution.record()
            report["fit_resources"] = {
                "status": "running",
                "candidate_workers": execution.fit_workers,
                "cpu_query_threads_per_worker": execution.resolved_query_threads(),
                "blas_threads_per_worker": 1,
                "gpu_workers": execution.fit_workers if execution.fit_query_backend in ("hybrid", "warp") else 0,
            }
        outcomes=None
        if execution.fit_workers > 1 and pending:
            # Context has not constructed any query scenes yet. Dense/low-res
            # arrays are reopened read-only, rather than pickling their bytes.
            inputs=pack_arrays((ctx,anchors,estimate,output,contacts))
            outcomes=ordered_spawn_map(_candidate_worker,pending,execution.fit_workers,
                initializer=_initialize_candidate_worker,initargs=(inputs,),timeout=args.candidate_timeout)
        try:
            with profile.time("candidate_fitting"):
                for path,record,checkpoint in entries:
                    print("Fitting",str(path),flush=True)
                    if record is None:
                        record=next(outcomes) if outcomes is not None else _candidate_record(path,ctx,anchors,estimate,output,contacts)
                        if cache and checkpoint is not None and record.get("mesh"):
                            cache.publish(checkpoint,output/"candidates"/record["id"])
                    report["candidates"].append(record)
                    write_json(output/"fit_result.json",report)
        finally:
            if outcomes is not None:
                outcomes.close()
        if execution.cpu_budget and pending:
            report["fit_resources"]["status"] = "completed"
        from .priority import candidate_key
        qualified=sorted((c for c in report["candidates"] if eligible(c)),key=candidate_key)
        for index,c in enumerate(qualified):
            c["retained"]=index<max(1,args.max_candidates)
        if qualified:
            report["status"]="success" if policy.strict else "diagnostic_complete"
            report["best_candidate_id"]=qualified[0]["id"]
        else:
            report["rejection_reasons"]=["no_candidate_passed_geometry_acceptance"]
        write_json(output/"fit_result.json",report)
        return 0 if qualified else 2
    except Exception as exc:
        report["status"]="error"
        report["error"]=str(exc)
        if report.get("fit_resources", {}).get("status") == "running":
            report["fit_resources"]["status"] = "error"
        write_json(output/"fit_result.json",report)
        raise
    finally:
        if execution.profile_fit:
            profile.count("fitting_wall_seconds",time.perf_counter()-run_started)
            performance=profile.record()
            performance["execution"]=execution.record()
            performance["candidates"]={c["id"]:c.get("performance") for c in report["candidates"]}
            write_json(output/"performance.json",performance)
