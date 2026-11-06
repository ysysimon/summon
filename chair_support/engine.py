"""Fit explicitly labelled chair surfaces to an immutable human motion.

Thresholds are engineering acceptance defaults, not calibrated probabilities.
All acceptance decisions and failures are exported, including rejected meshes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import trimesh

from .collision import collision_check, collision_checks, close_body_topology, edge_intersections
from .geometry_audit import prepare_geometry
from .selection import (VERSION as SELECTION_VERSION, evidence_weights, select_support,
                        support_intervals, sample_support_frames, weighted_quantile, weighted_vector)

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
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, np.float32)),
                        o3d.core.Tensor(np.asarray(faces, np.uint32)))
    return scene


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


def motion_context(manifest_path, vertices_path, contacts_path, indices_path, output):
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
    chair_count = (labels == 3).sum(axis=1)
    phase_count = np.interp(np.arange(len(dense)), source_indices, chair_count)
    conditions = {"insufficient_chair_contacts": phase_count >= DEFAULTS["min_chair_vertices"]}
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
               right=right, floor=floor, frame_weights=weights)
    regions = body_regions(ctx)
    anchors, invalid = collect_support_samples(ctx, regions, np.flatnonzero(candidate))
    valid_frames = np.array([sample[0] for sample in anchors[REGIONS[0]]], dtype=int)
    selected, proxy = select_support(valid_frames, weights, anchors, pelvis, forward,
        {region: DEFAULTS["gluteus_gap"] if "gluteus" in region else DEFAULTS["thigh_gap"]
         for region in REGIONS}, DEFAULTS["maximum_penetration"])
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
    width = max(weighted_quantile(widths,frame_weights,.5),float(np.ptp(projected[:,0])))
    dims = dict(seat_height=float(height), seat_width=float(width+.06),
                seat_depth=float(np.ptp(projected[:,1])+.08))
    estimate = dict(dimensions=dims, pelvis=p.tolist(), forward=fwd.tolist(), floor_height=ctx["floor"],
                    initial_seat_center=(p + .08*fwd).tolist(),
                    regions={r:len(regions[r]) for r in regions},
                    annotation_method="fixed SMPL-X template IDs in body_regions_smplx_v1.json",
                    thresholds=DEFAULTS, support_frames=len(frames),
                    support_selection_version=SELECTION_VERSION,
                    support_seconds=ctx["phase"]["support_seconds"], warnings=ctx["phase"]["warnings"])
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
        ids=np.flatnonzero(ctx["labels"][row]==3)
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
        if ids.shape!=(len(mesh.faces),) or not np.issubdtype(ids.dtype,np.integer) or not np.isin(ids,[0,1,2]).all():
            raise ValueError("Invalid complete face_part_ids annotation")
        part.setdefault("seat_face_indices", np.flatnonzero(part["face_part_ids"] == 0))
        part.setdefault("backrest_face_indices", np.flatnonzero(part["face_part_ids"] == 1))
    for key in (*aliases,"front_edge_face_indices","front_edge_vertices","floor_vertices"):
        if key not in part and key not in aliases: continue
        ids = np.asarray(part.get(key, []))
        limit=len(mesh.vertices) if key.endswith("vertices") else len(mesh.faces)
        if (ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or (key in aliases and not len(ids))
                or np.any(ids < 0) or np.any(ids >= limit) or len(np.unique(ids))!=len(ids)):
            raise UnverifiedSupport("Missing or invalid chair surface: " + key)
    for key,identity in (("seat_face_indices",0),("backrest_face_indices",1)):
        if "face_part_ids" in part and not np.all(part["face_part_ids"][part[key]]==identity):
            raise ValueError("Functional surface conflicts with face_part_ids: "+key)
    for key,axis in (("seat_face_indices",2),("backrest_face_indices",1)):
        selected=np.asarray(part[key],int)
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


def support_checks(metrics):
    failures = []
    for region in REGIONS:
        m = metrics[region]
        limit = DEFAULTS["gluteus_gap"] if "gluteus" in region else DEFAULTS["thigh_gap"]
        if m["coverage"] < DEFAULTS["coverage"]:
            failures.append(region+":insufficient_seat_coverage")
        if m["gap_p20"] is None or m["gap_p20"] > limit:
            failures.append(region+":insufficient_support")
        if m["maximum_seat_intrusion"] is not None and m["maximum_seat_intrusion"] > DEFAULTS["maximum_penetration"]:
            failures.append(region+":seat_penetration")
    return failures


def placement_checks(mesh, parts, pelvis, body_forward, chair_forward, floor):
    back_vertices=np.unique(mesh.faces[parts["backrest_face_indices"]])
    posterior=np.array([((mesh.vertices[back_vertices]-p)@f).max() for p,f in zip(pelvis,body_forward)])
    yaw_errors=np.degrees(np.arccos(np.clip(body_forward@chair_forward,-1.,1.)))
    floor_error=float(abs(mesh.vertices[:,2].min()-floor))
    reasons=[]
    if posterior.max()>=0: reasons.append("backrest_not_behind_pelvis")
    if yaw_errors.max()>DEFAULTS["maximum_yaw_degrees"]: reasons.append("chair_heading_mismatch")
    if floor_error>DEFAULTS["floor_tolerance"]: reasons.append("chair_not_grounded")
    return dict(backrest_forward_offset_max=float(posterior.max()),yaw_error_degrees=float(yaw_errors.max()),
                floor_error=floor_error),reasons


def fit_asset(path, ctx, anchors, estimate, output, contacts=None):
    meta, mesh, parts = read_asset(path)
    asset_id = str(meta.get("id", meta.get("asset_id", Path(path).parent.name)))
    directory = output / "candidates" / asset_id
    directory.mkdir(parents=True, exist_ok=True)
    seat = ray_scene(mesh.vertices, mesh.faces[parts["seat_face_indices"]])
    seat_vertices = np.unique(mesh.faces[parts["seat_face_indices"]])
    seat_center = np.median(mesh.vertices[seat_vertices], axis=0)
    base_z = float(mesh.vertices[:,2].min())
    desired_yaw = float(np.arctan2(-estimate["forward"][0], estimate["forward"][1]))
    center = np.asarray(estimate["initial_seat_center"])
    r0 = rotation(desired_yaw)
    t0 = center - seat_center @ r0.T
    h = float(np.quantile(mesh.vertices[seat_vertices,2], .8)-base_z)
    procedural = "infinigen" in str(meta.get("source","")).lower() or "generator" in meta
    scale_bounds=(1.,1.) if procedural else (DEFAULTS["min_scale"],DEFAULTS["max_scale"])
    geometry = prepare_geometry(mesh, parts, max_scale=scale_bounds[1])
    geometry_path = directory/"collision_geometry.json"
    source_hash = sha256(resolve(Path(path).parent, meta.get("mesh", "mesh.obj")))
    write_json(geometry_path, dict(**geometry.summary(), mesh_sha256=source_hash, components=geometry.records))
    scale0 = np.clip(estimate["dimensions"]["seat_height"] / h, *scale_bounds)
    sampled_frames = set(sample_support_frames(ctx["support_frames"],ctx["frame_weights"],ctx["fps"]))
    optimized_samples={r:[sample for sample in anchors[r] if sample[0] in sampled_frames] for r in REGIONS}
    points = {r:np.concatenate([x[2] for x in optimized_samples[r]]) for r in REGIONS}
    slices={r:np.r_[0,np.cumsum([len(x[2]) for x in optimized_samples[r]])] for r in REGIONS}
    optimized_frame_set=set(x[0] for x in optimized_samples[REGIONS[0]])
    def add_support_frames(frames):
        new_frames=set(int(f) for f in frames)-optimized_frame_set
        if not new_frames: return False
        optimized_frame_set.update(new_frames)
        for region in REGIONS:
            optimized_samples[region]=[sample for sample in anchors[region] if sample[0] in optimized_frame_set]
            points[region]=np.concatenate([sample[2] for sample in optimized_samples[region]])
            slices[region]=np.r_[0,np.cumsum([len(sample[2]) for sample in optimized_samples[region]])]
        return True
    contact_scenes={"seat":seat,"back":ray_scene(mesh.vertices,mesh.faces[parts["backrest_face_indices"]]),
                    "other":ray_scene(mesh.vertices,mesh.faces)}
    front=parts.get("front_edge_face_indices",[])
    if len(front): contact_scenes["front"]=ray_scene(mesh.vertices,mesh.faces[front])
    back_vertices=np.unique(mesh.faces[parts["backrest_face_indices"]])
    guard_frames=ctx["support_frames"]
    collision_guard_frames=list(dict.fromkeys(int(f) for f in guard_frames[[0,len(guard_frames)//2,-1]]))
    # Fixed human scenes are shared across chair candidates. These sparse guard
    # samples guide optimization only; the final collision audit is independent.
    if "_collision_proxy_faces" not in ctx:
        ctx["_collision_proxy_faces"],ctx["_collision_proxy_loops"]=close_body_topology(ctx["faces"],ctx["dense"].shape[1])
    body_scenes=ctx.setdefault("_collision_guard_scenes",{})
    def make_body_scene(frame):
        body=np.asarray(ctx["dense"][frame])
        capped=np.vstack([body]+[body[loop].mean(0)[None] for loop in ctx["_collision_proxy_loops"]])
        return ray_scene(capped,ctx["_collision_proxy_faces"])
    def body_scene(frame):
        if frame not in body_scenes:
            body_scenes[frame]=make_body_scene(frame)
        return body_scenes[frame]
    chair_guard_samples=np.vstack((mesh.vertices,mesh.triangles_center))
    _,sample_ids=np.unique(np.floor(chair_guard_samples/.02).astype(np.int64),axis=0,return_index=True)
    chair_guard_samples=chair_guard_samples[sample_ids]
    cached_guard={"key":None,"minimum":None}
    guard_risk_scans=[]
    def transform(x):
        r = rotation(x[2]); scale=x[3]
        t = np.array([x[0],x[1],ctx["floor"]-scale*base_z])
        return r,t,scale
    def guard_minimum_distances(x):
        key=(tuple(np.asarray(x)),tuple(collision_guard_frames))
        if key!=cached_guard["key"]:
            r,t,scale=transform(x)
            query=o3d.core.Tensor((chair_guard_samples@r.T*scale+t).astype(np.float32))
            cached_guard["minimum"]=np.array([float(body_scene(frame).compute_signed_distance(query).numpy().min()) for frame in collision_guard_frames])
            cached_guard["key"]=key
        return cached_guard["minimum"]
    def scan_full_motion_guard_risk(x):
        """Select active frames once, without putting all frames into cost."""
        r,t,scale=transform(x)
        query=o3d.core.Tensor((chair_guard_samples@r.T*scale+t).astype(np.float32))
        minimum=np.empty(len(ctx["dense"]),float)
        for frame in range(len(ctx["dense"])):
            scene=body_scenes.get(frame)
            if scene is None: scene=make_body_scene(frame)
            minimum[frame]=float(scene.compute_signed_distance(query).numpy().min())
            # Only already active/selected scenes survive this streaming scan.
            del scene
        selected=[int(frame) for frame in np.argsort(minimum) if frame not in collision_guard_frames][:3]
        collision_guard_frames.extend(selected)
        for frame in selected: body_scene(frame)
        path=directory/("guard-risk-scan-%02d.npz"%len(guard_risk_scans))
        np.savez_compressed(path,source_frame_indices=np.arange(len(minimum)),minimum_signed_distance=minimum)
        guard_risk_scans.append(dict(path=path.name,checked_source_frames=len(minimum),
            sampled_chair_points=len(chair_guard_samples),selected_source_frames=selected,
            selected_minimum_signed_distances=[float(minimum[frame]) for frame in selected],
            global_minimum_signed_distance=float(minimum.min()),
            policy="streamed Open3D guard SDF; optimization guidance only, not final certification"))
        cached_guard["key"]=None
        return selected
    def cost(x):
        r,t,scale=transform(x)
        loss=0.
        for region,p in points.items():
            gaps, valid=support_query(p,seat,r,t,scale)
            # Coverage guidance uses actual seat footprint; q20 permits curvature.
            local=(p-t)@r/scale
            lo,hi=mesh.vertices[seat_vertices,:2].min(0),mesh.vertices[seat_vertices,:2].max(0)
            outside=np.maximum(lo-local[:,:2],0)+np.maximum(local[:,:2]-hi,0)
            frame_losses=[]
            for k,(_,_,_,weights) in enumerate(optimized_samples[region]):
                sl=slice(slices[region][k],slices[region][k+1]);ok=valid[sl];g=gaps[sl]
                value=2*np.average(np.sum(outside[sl]**2,axis=1),weights=weights)+.05*(1-np.mean(ok))
                if ok.any():
                    q=weighted_quantile(g[ok],weights[ok],.2)
                    value+=q*q+5*np.average(np.minimum(g[ok],0)**2,weights=weights[ok])
                    gap_limit=DEFAULTS["gluteus_gap"] if "gluteus" in region else DEFAULTS["thigh_gap"]
                    # Explicit violation terms avoid accepting a mean-loss
                    # compromise between floating buttocks and buried thighs.
                    value+=100*max(0.,q-gap_limit)**2
                    value+=100*max(0.,-float(g[ok].min())-DEFAULTS["maximum_penetration"])**2
                else: value+=1.
                value+=100*max(0.,DEFAULTS["coverage"]-np.mean(ok))**2
                frame_losses.append(value)
            loss+=float(np.average(frame_losses,weights=[ctx["frame_weights"][sample[0]]
                                                         for sample in optimized_samples[region]]))
        if contacts:
            for role,(p,weights) in contacts.items():
                if not len(p): continue
                sc=contact_scenes.get(role,contact_scenes["other"])
                local=(p-t)@r/scale
                d=sc.compute_distance(o3d.core.Tensor(local.astype(np.float32))).numpy()*scale
                loss+=.10*float(np.average(np.minimum(d,.15)**2,weights=weights))
        back_world=mesh.vertices[back_vertices]@r.T*scale+t
        offsets=(back_world@ctx["forward"][guard_frames].T).max(axis=0)-np.sum(ctx["pelvis"][guard_frames]*ctx["forward"][guard_frames],axis=1)
        loss+=100*float(np.average(np.maximum(offsets,0)**2,weights=ctx["frame_weights"][guard_frames]))
        loss+=100*float(np.mean(np.maximum(-guard_minimum_distances(x)-DEFAULTS["maximum_penetration"],0)**2))
        loss += .001*(scale-1)**2 + .001*(x[2]-desired_yaw)**2
        return loss
    bounds=[(t0[0]-.25,t0[0]+.25),(t0[1]-.25,t0[1]+.25),
            (desired_yaw-np.radians(20),desired_yaw+np.radians(20)),scale_bounds]
    x0=np.array([t0[0],t0[1],desired_yaw,scale0])
    starts=[]
    for dx in (-.08,0.,.08):
        for dy in (-.08,0.,.08):
            for da in (-15.,0.,15.):
                for s in sorted(set([float(scale0),*scale_bounds])):
                    x=np.array([t0[0]+dx,t0[1]+dy,desired_yaw+np.radians(da),s])
                    starts.append((cost(x),x))
    best_starts=sorted(starts,key=lambda x:x[0])[:3]
    results=[minimize(cost,x,method="Powell",bounds=bounds,options={"maxiter":70,"xtol":1e-4,"ftol":1e-5}) for _,x in best_starts]
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
        r,t,scale=transform(x)
        residuals=[]
        for region,p in points.items():
            gaps,valid=support_query(p,seat,r,t,scale)
            gap_limit=DEFAULTS["gluteus_gap"] if "gluteus" in region else DEFAULTS["thigh_gap"]
            for k,(_,_,_,weights) in enumerate(optimized_samples[region]):
                sl=slice(slices[region][k],slices[region][k+1]);ok=valid[sl];g=gaps[sl]
                residuals.append(.02*(float(np.mean(ok))-DEFAULTS["coverage"]))
                if ok.any():
                    residuals.extend((gap_limit-margin-weighted_quantile(g[ok],weights[ok],.2),
                        DEFAULTS["maximum_penetration"]-margin+float(g[ok].min())))
                else: residuals.extend((-.1,-.1))
        back_world=mesh.vertices[back_vertices]@r.T*scale+t
        offsets=(back_world@ctx["forward"][guard_frames].T).max(axis=0)-np.sum(ctx["pelvis"][guard_frames]*ctx["forward"][guard_frames],axis=1)
        residuals.append(-float(offsets.max())-margin)
        heading=np.array([0.,1.,0.])@r.T
        angles=np.degrees(np.arccos(np.clip(ctx["forward"][guard_frames]@heading,-1.,1.)))
        residuals.append(.01*(DEFAULTS["maximum_yaw_degrees"]-float(angles.max())))
        # A certificate needs room for outside-witness and surface-cell bounds.
        # Support can approach its exact threshold more tightly than this.
        residuals.extend(DEFAULTS["maximum_penetration"]-max(margin,.0001)+guard_minimum_distances(x))
        return np.asarray(residuals)
    def violation(x): return max(0.,-float(feasible_residuals(x).min()))
    active=np.array([i for i,(lo,hi) in enumerate(bounds) if hi>lo],int)
    lower_bounds=np.array([bounds[i][0] for i in active])
    upper_bounds=np.array([bounds[i][1] for i in active])
    def repair(start):
        fixed=start.x.copy()
        def expand(x):
            result=fixed.copy();result[active]=x;return result
        repaired=minimize(lambda x:cost(expand(x)),start.x[active],method="COBYLA",
            constraints=[{"type":"ineq","fun":lambda x:feasible_residuals(expand(x))},
                {"type":"ineq","fun":lambda x:x-lower_bounds},
                {"type":"ineq","fun":lambda x:upper_bounds-x}],
            options={"maxiter":300 if buffer_tier==0 else 900,
                "rhobeg":.02 if buffer_tier==0 else .005,
                "tol":1e-5 if buffer_tier==0 else 1e-6,"catol":1e-7})
        raw=expand(repaired.x)
        repaired.raw_bounds_violation=float(max(0.,(all_lower-raw).max(),(raw-all_upper).max()))
        repaired.x=np.clip(raw,all_lower,all_upper)
        repaired.fun=cost(repaired.x);repairs.append(repaired)
        return repaired
    repairs=[]
    for start in sorted(results,key=lambda o:(violation(o.x),o.fun))[:2]:
        repaired=repair(start)
        if violation(repaired.x)<=1e-6: break
    choices=results+repairs
    def choose_result():
        feasible=[result for result in choices if violation(result.x)<=1e-6]
        return min(feasible,key=lambda o:o.fun) if feasible else min(choices,key=lambda o:(violation(o.x),o.fun))
    opt=choose_result()
    validation_rounds=[];collision_witnesses=[]
    support_repairs=0;collision_repairs=0
    maximum_validation_rounds=8
    maximum_support_repairs=6;maximum_collision_repairs=6
    for validation_round in range(maximum_validation_rounds):
        r,t,scale=transform(opt.x)
        fitted=trimesh.Trimesh(mesh.vertices @ r.T * scale+t,mesh.faces,process=False)
        mesh_path=directory/"mesh.obj"; fitted.export(mesh_path)
        exported=dict(parts)
        exported["forward"]=np.array([0.,1.,0.])@r.T
        exported["forward_origin"]=seat_center@r.T*scale+t
        np.savez_compressed(directory/"parts.npz",**exported)
        support={};worst_support={}
        for region,samples in anchors.items():
            per_frame=[(frame,region_metrics(*support_query(p,seat,r,t,scale),weights)) for frame,ids,p,weights in samples]
            support[region]=dict(coverage=min(m["coverage"] for _,m in per_frame),
                gap_p20=max((m["gap_p20"] for _,m in per_frame if m["gap_p20"] is not None),default=None),
                maximum_seat_intrusion=max((m["maximum_seat_intrusion"] for _,m in per_frame if m["maximum_seat_intrusion"] is not None),default=None))
            worst_support[region]=[dict(frame=f,**m) for f,m in per_frame]
        reasons=support_checks(support)
        placement,placement_reasons=placement_checks(fitted,parts,ctx["pelvis"][ctx["support_frames"]],
            ctx["forward"][ctx["support_frames"]],np.array([0.,1.,0.])@r.T,ctx["floor"])
        reasons.extend(placement_reasons)
        metrics=dict(support=support,scale=float(scale),geometry=geometry.summary(),**placement)
        worst_frame=None;collision=None
        if not reasons:
            print("Full-frame collision check",asset_id,"round",validation_round,flush=True)
            collision=collision_check(ctx,fitted,parts,prepared=geometry,pose=(r,t,scale))
            write_json(directory/("collision-iteration-%d.json"%validation_round),collision)
            write_json(directory/"collision.json",collision)
            worst_frame=collision["worst_collision_frame"]
            metrics["collision"]={k:v for k,v in collision.items() if k!="frames"}
            reasons.extend(collision_checks(collision))
        else:
            metrics["collision"]={"status":"not_run_support_rejected"}
        write_json(directory/"support.json",worst_support)
        validation_rounds.append(dict(round=validation_round,collision_guard_frames=list(collision_guard_frames),
            rejection_reasons=list(reasons),worst_collision_frame=worst_frame,
            internal_margin_m=margin,
            collision_internal_margin_m=max(margin,.0001),
            optimized_source_frames=sorted(optimized_frame_set),optimizer_success=bool(opt.success),optimizer_message=str(opt.message),
            maximum_constraint_violation_m=violation(opt.x),loss=float(cost(opt.x))))
        added_support=[]
        for region in REGIONS:
            if region+":insufficient_seat_coverage" in reasons:
                added_support.append(min(worst_support[region],key=lambda m:m["coverage"])["frame"])
            if region+":insufficient_support" in reasons:
                added_support.append(max(worst_support[region],key=lambda m:m["gap_p20"] if m["gap_p20"] is not None else float("inf"))["frame"])
            if region+":seat_penetration" in reasons:
                added_support.append(max(worst_support[region],key=lambda m:m["maximum_seat_intrusion"] or 0)["frame"])
        if (validation_round<maximum_validation_rounds-1 and support_repairs<maximum_support_repairs
                and add_support_frames(added_support)):
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
                and collision_repair_frame is not None):
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
                    collision_witnesses.append(dict(source_frame=int(collision_repair_frame),
                        source_face=witness.get("source_face"),source_part=witness.get("source_part"),
                        local_point=local.tolist(),world_point_at_discovery=witness["point"],
                        measured_lower_bound=witness.get("sample_lower_bound"),
                        uncertainty_repair="full_motion_penetration" not in reasons))
                    cached_guard["key"]=None;changed=True;new_points=True
            if new_points:
                print("Scanning all source frames for guard risk",asset_id,flush=True)
                selected=scan_full_motion_guard_risk(opt.x)
                print("Activated risk source frames",selected,asset_id,flush=True)
            if changed:
                fitted.export(directory/("mesh-iteration-%d.obj"%validation_round))
                print("Refining collision frame/witness",collision_repair_frame,asset_id,flush=True)
                collision_repairs+=1
                choices.append(repair(opt));opt=choose_result()
                continue
        if (reasons and validation_round<maximum_validation_rounds-1 and buffer_tier<len(buffer_levels)-1
                and violation(opt.x)>1e-7):
            buffer_tier+=1;margin=buffer_levels[buffer_tier]
            buffer_history.append(dict(round=validation_round+1,margin_m=margin,
                reason="extra internal buffer infeasible; final acceptance limits unchanged"))
            print("Refining with internal margin",margin,"m",asset_id,flush=True)
            choices.append(repair(opt));opt=choose_result()
            continue
        break
    write_json(directory/"optimization.json",dict(method="coarse grid, Powell, COBYLA feasibility repair",
        internal_margin_m=margin,initial_internal_margin_m=buffer_levels[0],buffer_history=buffer_history,
        collision_internal_margin_m=max(margin,.0001),minimum_collision_internal_margin_m=.0001,
        optimized_source_frames=[int(x[0]) for x in optimized_samples[REGIONS[0]]],
        collision_guard_samples=len(chair_guard_samples),collision_guard_frames=collision_guard_frames,
        maximum_validation_rounds=maximum_validation_rounds,maximum_support_repairs=maximum_support_repairs,
        maximum_collision_repairs=maximum_collision_repairs,support_repairs=support_repairs,collision_repairs=collision_repairs,
        collision_witnesses=collision_witnesses,guard_risk_scans=guard_risk_scans,validation_rounds=validation_rounds,
        final_maximum_constraint_violation_m=violation(opt.x),
        trials=[dict(loss=float(o.fun),constraint_violation_m=violation(o.x),success=bool(o.success),
            raw_bounds_violation=float(getattr(o,"raw_bounds_violation",0)),message=str(o.message)) for o in choices]))
    result=dict(id=asset_id,mesh=str(mesh_path.relative_to(output)),parts=str((directory/"parts.npz").relative_to(output)),
                qualified=not reasons,loss=float(cost(opt.x)),metrics=metrics,rejection_reasons=reasons,
                worst_collision_frame=worst_frame,source_asset=str(Path(path).resolve()),
                transform=dict(rotation=r.tolist(),translation=t.tolist(),scale=float(scale)),
                geometry_audit=str(geometry_path.relative_to(output)), source_geometry_sha256=source_hash)
    write_json(directory/"result.json",result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry",required=True,type=Path)
    parser.add_argument("--vertices-path",required=True,type=Path)
    parser.add_argument("--contacts",required=True,type=Path)
    parser.add_argument("--frame-indices",required=True,type=Path)
    parser.add_argument("--models-dir",type=Path)
    parser.add_argument("--candidate-source",choices=["cad","infinigen"],default="cad")
    parser.add_argument("--assets-manifest",type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--estimate-only",action="store_true")
    parser.add_argument("--max-candidates",type=int,choices=(1,2,3),default=3)
    args=parser.parse_args(argv)
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    report=dict(backend="supported-chair-v2",status="no_valid_candidate",candidates=[],best_candidate_id=None,
                thresholds=DEFAULTS,validation_scope="geometry; no dynamics simulation",
                prediction_sha256=sha256(args.contacts),world_sha256=sha256(args.vertices_path))
    implementation_files=[Path(__file__),Path(__file__).parent/"cad.py",Path(__file__).parent/"annotations"/"body_regions_smplx_v1.json",
        Path(__file__).parent/"annotations"/"cad_chair_annotations_v1.json",Path(__file__).parent/"collision.py",
        Path(__file__).parent/"selection.py",Path(__file__).parent/"geometry_audit.py"]
    report["implementation_sha256"]={str(path.relative_to(Path(__file__).parent)):sha256(path) for path in implementation_files if path.is_file()}
    try:
        ctx=motion_context(args.geometry,args.vertices_path,args.contacts,args.frame_indices,output)
        report["support_selection"]={key:value for key,value in ctx["phase"].items()
                                     if key not in ("selection_reasons","thresholds","selection_proxy")}
        if not len(ctx["support_frames"]):
            report["rejection_reasons"]=["no_support_evidence"]
            write_json(output/"fit_result.json",report)
            return 2
        regions=body_regions(ctx)
        anchors,estimate=prepare_support(ctx,regions,output)
        if args.estimate_only:
            print(json.dumps({"estimate":str(output/"estimate.json")}))
            return 0
        if args.assets_manifest:
            a=json.loads(args.assets_manifest.read_text(encoding="utf-8"))
            assets=[resolve(args.assets_manifest.parent,p) for p in a["assets"]]
        elif args.candidate_source=="cad" and args.models_dir:
            from .cad import prepare_cad_assets
            assets=prepare_cad_assets(args.models_dir,output/"cad_assets")
        else:
            raise ValueError("Provide --assets-manifest or --models-dir for CAD candidates")
        contacts=contact_roles(ctx,regions,output)
        for path in assets:
            print("Fitting",str(path),flush=True)
            try:
                record=fit_asset(path,ctx,anchors,estimate,output,contacts)
            except (ValueError,FileNotFoundError,KeyError) as exc:
                record=dict(id=Path(path).parent.name,qualified=False,retained=False,loss=None,metrics={},
                    rejection_reasons=[("support_geometry_unverified: " if isinstance(exc, UnverifiedSupport)
                                        else "invalid_candidate_asset: ")+str(exc)],worst_collision_frame=None,source_asset=str(path))
            report["candidates"].append(record)
            write_json(output/"fit_result.json",report)
        qualified=sorted((c for c in report["candidates"] if c["qualified"]),key=lambda c:c["loss"])
        for index,c in enumerate(qualified):
            c["retained"]=index<max(1,args.max_candidates)
        if qualified:
            report["status"]="success"
            report["best_candidate_id"]=qualified[0]["id"]
        else:
            report["rejection_reasons"]=["no_candidate_passed_geometry_acceptance"]
        write_json(output/"fit_result.json",report)
        return 0 if qualified else 2
    except Exception as exc:
        report["status"]="error"
        report["error"]=str(exc)
        write_json(output/"fit_result.json",report)
        raise
