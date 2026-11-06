"""Prepare attributable furniture solids in asset coordinates before fitting."""
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import QhullError
import trimesh


NUMERIC_MARGIN = 1e-5
PROXY_ERROR = .002
PROXY_QUERY_BUDGET = 250_000
PART_NAMES = {0: "seat", 1: "backrest", 2: "base"}


def ray_scene(vertices, faces):
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, np.float32)),
                        o3d.core.Tensor(np.asarray(faces, np.uint32)))
    return scene


def children(triangles):
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, bc, ca = (a+b)/2, (b+c)/2, (c+a)/2
    return np.stack((np.stack((a, ab, ca), 1), np.stack((ab, b, bc), 1),
                     np.stack((ca, bc, c), 1), np.stack((ab, bc, ca), 1)), 1).reshape(-1, 3, 3)


def edge_data(faces):
    directed = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                       return_inverse=True, return_counts=True)
    return directed, edges, inverse, counts


def face_groups(faces):
    # Include edges shared by more than two faces so a non-manifold junction
    # is diagnosed together, rather than becoming several apparently open pieces.
    _, _, inverse, _ = edge_data(faces)
    order = np.argsort(inverse, kind="stable")
    adjacent = inverse[order[:-1]] == inverse[order[1:]]
    left, right = order[:-1][adjacent] // 3, order[1:][adjacent] // 3
    graph = coo_matrix((np.ones(len(left), bool), (left, right)), shape=(len(faces), len(faces)))
    _, labels = connected_components(graph, directed=False)
    order = np.argsort(labels, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(labels[order]))+1)


def inspect_shell(vertices, faces, vertex_ids):
    directed, edges, inverse, counts = edge_data(faces)
    balance = np.bincount(inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1., -1.))
    triangles = vertices[faces] - vertices.mean(axis=0)
    volume = float(np.einsum("ij,ij->", triangles[:, 0],
                             np.cross(triangles[:, 1], triangles[:, 2]))/6)
    return dict(open_edges=vertex_ids[edges[counts == 1]].tolist(),
                nonmanifold_edges=vertex_ids[edges[counts > 2]].tolist(),
                winding_consistent=bool(np.all(balance[counts == 2] == 0)),
                signed_volume_m3=volume, degenerate_volume=abs(volume) < 1e-12)


def closed(inspection):
    return (not inspection["open_edges"] and not inspection["nonmanifold_edges"]
            and inspection["winding_consistent"] and not inspection["degenerate_volume"])


def stitch_opposite_boundaries(vertices, faces):
    """Join only unique, coincident, oppositely directed border pairs of one instance."""
    directed, _, inverse, counts = edge_data(faces)
    boundary = directed[counts[inverse] == 1]
    if not len(boundary):
        return None
    _, coords = np.unique(vertices, axis=0, return_inverse=True)
    keyed = coords[boundary]
    _, groups, multiplicity = np.unique(np.sort(keyed, axis=1), axis=0,
                                         return_inverse=True, return_counts=True)
    order = np.argsort(groups, kind="stable")
    paired = order[multiplicity[groups[order]] == 2].reshape(-1, 2)
    parents = np.arange(len(vertices))

    def root(v):
        while parents[v] != v:
            parents[v] = parents[parents[v]]
            v = parents[v]
        return v

    for a, b in paired:
        if keyed[a, 0] != keyed[a, 1] and np.array_equal(keyed[a], keyed[b, ::-1]):
            for u, v in zip(boundary[a], boundary[b, ::-1]):
                parents[root(v)] = root(u)
    if np.all(parents == np.arange(len(vertices))):
        return None
    return np.array([root(i) for i in range(len(vertices))])[faces]


def certified_hull(vertices, faces, max_scale):
    centered = vertices - vertices.mean(axis=0)
    if len(vertices) < 4 or np.ptp(centered @ np.linalg.svd(centered, full_matrices=False)[2][-1]) <= 2*NUMERIC_MARGIN:
        return None, "insufficient_thickness", None
    try:
        hull = trimesh.convex.convex_hull(vertices)
    except (QhullError, ValueError, RuntimeError, IndexError):
        return None, "hull_construction_failed", None
    if not hull.is_volume or not np.isfinite(hull.vertices).all():
        return None, "invalid_hull", None
    normals = hull.face_normals.copy()
    offsets = np.einsum("ij,ij->i", normals, hull.triangles[:, 0])
    center = hull.vertices.mean(axis=0)
    radii = offsets - normals @ center
    if radii.min() <= NUMERIC_MARGIN:
        return None, "insufficient_thickness", None
    # Inflate the hull to pay for containment tolerance rather than accepting
    # a vertex slightly outside the mesh that is actually used by the solver.
    def plane_excess(plane_offsets):
        return max((np.einsum("ik,jk->ij", vertices[i:i+128], normals)-plane_offsets).max()
                   for i in range(0, len(vertices), 128))
    excess = plane_excess(offsets)
    factor = 1. + (max(0., excess) + NUMERIC_MARGIN)/radii.min()
    hull.vertices = center + (hull.vertices-center)*factor
    if not hull.is_volume or plane_excess(np.einsum("ij,ij->i", normals, hull.triangles[:, 0])) > 0:
        return None, "invalid_hull", None
    scene = ray_scene(vertices, faces)
    pending = hull.triangles
    queries, bound = 0, 0.
    while len(pending):
        queries += len(pending)
        if queries > PROXY_QUERY_BUDGET:
            return None, "proxy_query_budget_exhausted", None
        centers = pending.mean(axis=1)
        radius = np.linalg.norm(pending-centers[:, None], axis=2).max(axis=1)
        distance = scene.compute_distance(o3d.core.Tensor(centers.astype(np.float32))).numpy()
        if not np.isfinite(distance).all() or np.any((distance-NUMERIC_MARGIN)*max_scale > PROXY_ERROR):
            return None, "proxy_shape_error", None
        upper = (distance + radius + NUMERIC_MARGIN)*max_scale
        fits = upper <= PROXY_ERROR
        if fits.any():
            bound = max(bound, float(upper[fits].max()))
        pending = children(pending[~fits])
    return hull, None, bound


@dataclass
class GeometryAudit:
    max_scale: float = 1.
    records: list = field(default_factory=list)
    solids: list = field(default_factory=list)
    face_components: object = None

    def summary(self):
        return dict(version=1, proxy_error_budget_m=PROXY_ERROR,
                    counts=dict(Counter(r["mode"] for r in self.records)),
                    component_modes=[r["mode"] for r in self.records])

    def posed(self, rotation, translation, scale):
        if not 0 < scale <= self.max_scale + 1e-12 or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0):
            raise ValueError("Pose exceeds the verified rigid transform and scale budget")
        return [dict(piece, vertices=piece["vertices"] @ rotation.T * scale + translation)
                for piece in self.solids]


def prepare_geometry(mesh, parts=None, max_scale=1.):
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all() or not np.isfinite(max_scale) or max_scale <= 0:
        raise ValueError("Invalid furniture geometry or scale budget")
    parts = {} if parts is None else parts
    labels = np.asarray(parts.get("face_part_ids", np.full(len(mesh.faces), -1)))
    instances = np.asarray(parts.get("face_instance_ids", np.full(len(mesh.faces), -1)))
    for key, values in (("face_part_ids", labels), ("face_instance_ids", instances)):
        if values.shape != (len(mesh.faces),) or values.dtype.kind not in "iu" or np.any(values < -1):
            raise ValueError("Invalid " + key)
    present = np.unique(instances[instances >= 0])
    def declared_ids(key):
        ids = np.asarray(parts.get(key, np.empty(0, int)))
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids)
                or not np.isin(ids, present).all()):
            raise ValueError("Invalid " + key)
        return ids
    envelope_ids = set(declared_ids("convex_envelope_ids").tolist())
    box_ids = declared_ids("instance_bound_ids")
    boxes = np.asarray(parts.get("instance_bounds", np.empty((0, 2, 3))), float)
    if (("instance_bounds" in parts) != ("instance_bound_ids" in parts)
            or boxes.shape != (len(box_ids), 2, 3) or not np.isfinite(boxes).all()
            or np.any(boxes[:, 0] > boxes[:, 1])):
        raise ValueError("Invalid instance_bounds")
    declared_boxes = dict(zip(box_ids.tolist(), boxes))
    audit = GeometryAudit(max_scale=max_scale, face_components=np.empty(len(mesh.faces), np.int32))

    def record(face_ids, instance, inspections, mode, vertices, faces, **extra):
        i = len(audit.records)
        names = sorted({PART_NAMES.get(int(n), "furniture") for n in labels[face_ids]})
        item = dict(id=i, instance=instance, parts=names, mode=mode,
                    source_faces=face_ids.tolist(), shells=inspections, **extra)
        audit.records.append(item)
        audit.face_components[face_ids] = i
        if mode in ("original", "stitched", "hull"):
            audit.solids.append(dict(id=i, instance=instance, parts=names, mode=mode, vertices=vertices,
                                     faces=faces, face_ids=face_ids if mode != "hull" else None))

    for instance in np.unique(instances):
        group_ids = np.flatnonzero(instances == instance)
        box = declared_boxes.get(instance)
        if box is not None:
            points = mesh.vertices[np.unique(mesh.faces[group_ids])]
            if np.any(points.min(0) < box[0]) or np.any(points.max(0) > box[1]):
                raise ValueError("Authored instance_bounds do not contain their source faces")
        open_ids, issues = [], []
        for group in face_groups(mesh.faces[group_ids]):
            ids = group_ids[group]
            vertex_ids, local = np.unique(mesh.faces[ids], return_inverse=True)
            vertices, faces = mesh.vertices[vertex_ids], local.reshape(-1, 3)
            issue = inspect_shell(vertices, faces, vertex_ids)
            if closed(issue):
                record(ids, int(instance), [issue], "original", vertices, faces)
            elif instance < 0:
                record(ids, -1, [issue], "unknown", None, None, reason="unattributed_open_or_invalid_shell")
            else:
                open_ids.append(ids)
                issue["source_faces"] = ids.tolist()
                issues.append(issue)
        if not open_ids:
            continue
        ids = np.concatenate(open_ids)
        vertex_ids, local = np.unique(mesh.faces[ids], return_inverse=True)
        vertices, faces = mesh.vertices[vertex_ids], local.reshape(-1, 3)
        stitched = stitch_opposite_boundaries(vertices, faces)
        if stitched is not None:
            remaining = []
            shell_at = np.repeat(np.arange(len(open_ids)), [len(s) for s in open_ids])
            for p in face_groups(stitched):
                used, remap = np.unique(stitched[p], return_inverse=True)
                joined = remap.reshape(-1, 3)
                check = inspect_shell(vertices[used], joined, vertex_ids[used])
                if closed(check):
                    record(ids[p], int(instance), [issues[k] for k in np.unique(shell_at[p])], "stitched",
                           vertices[used], joined)
                else:
                    remaining.append(p)
            if not remaining:
                continue
            remaining = np.concatenate(remaining)
            ids = ids[remaining]
            issues = [issues[k] for k in np.unique(shell_at[remaining])]
        # Even with an instance ID, open surfaces alone cannot tell us how
        # far a missing wall extends. Require independent source provenance.
        if instance not in envelope_ids:
            if box is None:
                record(ids, int(instance), issues, "unknown", None, None, reason="no_trusted_extent")
            else:
                record(ids, int(instance), issues, "bounds_only", None, None,
                       reason="authored_box_only", bounds=box.tolist())
            continue
        all_vertices, all_local = np.unique(mesh.faces[group_ids], return_inverse=True)
        entire = mesh.vertices[all_vertices]
        hull, reason, error = None, "unreliable_topology", None
        if any(s["open_edges"] for s in issues) and not any(s["nonmanifold_edges"] for s in issues):
            hull, reason, error = certified_hull(entire, all_local.reshape(-1, 3), max_scale)
        if hull is not None:
            record(ids, int(instance), issues, "hull", hull.vertices, hull.faces,
                   proxy_surface_error_upper_m=error)
        else:
            record(ids, int(instance), issues, "bounds_only", None, None, reason=reason,
                   bounds=[entire.min(axis=0).tolist(), entire.max(axis=0).tolist()])
    return audit
