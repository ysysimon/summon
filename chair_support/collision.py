"""Conservative, bidirectional surface-to-solid penetration bounds.

The audit accepts only when every potentially penetrating source triangle in
every frame has an upper bound at or below the limit. Samples are lower bounds,
not maxima. Invalid solids and exhausted subdivision budgets remain undetermined.
"""
import math
from copy import copy
import numpy as np
import open3d as o3d
import trimesh
from .geometry_audit import NUMERIC_MARGIN, PART_NAMES, children as _children, prepare_geometry, ray_scene

MAXIMUM_PENETRATION = .005
MIN_CELL_RADIUS = .000015625
QUERY_BUDGET = 250000
WINDING_INTEGER_TOLERANCE = 1e-6
RAY_DIRECTIONS = np.array([[1., .371, .529], [-.291, 1., .193], [.127, -.413, 1.]])
RAY_DIRECTIONS /= np.linalg.norm(RAY_DIRECTIONS, axis=1)[:, None]


def close_body_topology(faces, count):
    """Cap original boundary loops; do not alter visible body vertices/faces."""
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                   return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Non-manifold body edges cannot be capped reliably")
    boundary = directed[counts[inverse] == 1]
    follow = {int(a): int(b) for a, b in boundary}
    if len(follow) != len(boundary):
        raise ValueError("Non-manifold body boundary cannot be capped reliably")
    loops, additions = [], []
    while follow:
        first = next(iter(follow)); loop, node = [], first
        while node not in loop:
            loop.append(node)
            if node not in follow:
                raise ValueError("Open body boundary is not a closed loop")
            node = follow.pop(node)
        if node != first or len(loop) > 128:
            raise ValueError("Unexpected body boundary loop")
        center = count + len(loops)
        additions.extend((loop[(i+1) % len(loop)], loop[i], center) for i in range(len(loop)))
        loops.append(np.array(loop))
    return np.concatenate((faces, np.asarray(additions, int).reshape(-1, 3))), loops


def edge_intersections(scene, vertices, edges, bounds, details=False):
    """All segment hits, including exits and subsequent disconnected solids.

    This is diagnostic only. The acceptance bound does not depend on hits.
    """
    a, b = vertices[edges[:, 0]], vertices[edges[:, 1]]
    overlap = np.all(np.maximum(a, b) >= bounds[0]-NUMERIC_MARGIN, axis=1)
    overlap &= np.all(np.minimum(a, b) <= bounds[1]+NUMERIC_MARGIN, axis=1)
    ids = np.flatnonzero(overlap); a, delta = a[ids], b[ids]-a[ids]
    lengths = np.linalg.norm(delta, axis=1); good = lengths > 1e-7
    ids, a, delta, lengths = ids[good], a[good], delta[good], lengths[good]
    empty = (np.empty((0, 3)), np.empty(0, int), np.empty(0, int))
    if not len(a):
        return empty if details else empty[0]
    direction = delta / lengths[:, None]
    answer = scene.list_intersections(o3d.core.Tensor(np.column_stack((a, direction)).astype(np.float32)))
    ray_ids = answer["ray_ids"].numpy().astype(int)
    distances = answer["t_hit"].numpy()
    good = (distances > 1e-4) & (distances < lengths[ray_ids]-1e-4)
    ray_ids, distances = ray_ids[good], distances[good]
    points = a[ray_ids] + direction[ray_ids] * distances[:, None]
    result = (points, ids[ray_ids], answer["primitive_ids"].numpy()[good].astype(int))
    return result if details else result[0]


class InvalidSolid(ValueError):
    def __init__(self, issue):
        self.issue = issue
        super().__init__(issue["reason"])


def component_topology(vertices, faces, labels=None):
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    groups = trimesh.graph.connected_components(mesh.face_adjacency,
                                                nodes=np.arange(len(faces)), min_len=1)
    result = []
    for number, face_ids in enumerate(groups):
        vertex_ids, remapped = np.unique(faces[face_ids], return_inverse=True)
        names = sorted({PART_NAMES.get(int(v), "part-"+str(int(v))) for v in labels[face_ids]}) if labels is not None else ["body"]
        result.append(dict(id=number, vertex_ids=vertex_ids, face_ids=np.asarray(face_ids),
                           faces=remapped.reshape(-1, 3), parts=names))
    return result


def solid_angle_winding(triangles, point):
    """Full float64 oriented solid angle; no ray selection or random jitter.

    Closed oriented surfaces have integer winding away from their boundary,
    including nonzero integer multiplicities in self-overlapping regions.
    Summing all faces with fsum keeps cancellation near zero deterministic.
    Boundary proximity is checked separately before using this classifier.
    """
    a, b, c = (np.asarray(triangles, np.float64)-point).transpose(1, 0, 2)
    la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
    numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
    denominator = (la*lb*lc + np.einsum("ij,ij->i", a, b)*lc
                   + np.einsum("ij,ij->i", b, c)*la
                   + np.einsum("ij,ij->i", c, a)*lb)
    angles = 2*np.arctan2(numerator, denominator)
    return math.fsum(angles)/(4*np.pi) if np.isfinite(angles).all() else float("nan")


class ClosedSolid:
    """Union of validated closed components, never an odd/even combined mesh.

    In overlaps, a component's internal boundary is not the union boundary.
    An outside witness supplies the upper distance bound; if the closest witness
    is hidden inside another component, the enclosing-box witness stays safe.
    """
    def __init__(self, vertices, faces, topology, kind, prepared=False):
        self.components = []
        self.kind = kind
        self.vertices = np.asarray(vertices)
        self.faces = np.asarray(faces)
        for info in topology:
            v = info["vertices"] if prepared else self.vertices[info["vertex_ids"]]
            f = info["faces"]
            mesh = trimesh.Trimesh(v, f, process=False)
            issue = dict(kind=kind, component=int(info["id"]), parts=info["parts"])
            if not prepared and (not np.isfinite(v).all() or not mesh.is_watertight
                                 or not mesh.is_winding_consistent or abs(mesh.volume) < 1e-12):
                raise InvalidSolid(dict(issue, reason="non_closed_or_degenerate_collision_component"))
            self.components.append(dict(info, scene=ray_scene(v, f),
                                        normals=np.asarray(mesh.face_normals),
                                        triangles=np.asarray(v[f], np.float64),
                                        winding_fallback_queries=0,
                                        winding_fallback_resolved=0,
                                        bounds=np.stack((v.min(0), v.max(0)))))
        if not self.components and not prepared:
            raise InvalidSolid(dict(kind=kind, reason="empty_collision_solid"))
        self.bounds = np.stack((np.min([self.vertices.min(0)]+[c["bounds"][0] for c in self.components], axis=0),
                                np.max([self.vertices.max(0)]+[c["bounds"][1] for c in self.components], axis=0)))
        self.scene = ray_scene(vertices, faces)

    @staticmethod
    def occupancy(component, points):
        """Oriented crossings define nonzero-winding occupancy, not parity.

        Multiple overlapping, consistently oriented layers remain occupied.
        Three fixed non-axis-aligned rays first agree on occupied/unoccupied.
        Disagreements and unreliable rays use a full float64 solid-angle sum;
        non-integer winding and points within the boundary margin stay uncertain.
        """
        n = len(points)
        origins = np.repeat(points, 3, axis=0)
        rays = np.column_stack((origins, np.tile(RAY_DIRECTIONS, (n, 1))))
        hits = component["scene"].list_intersections(o3d.core.Tensor(rays.astype(np.float32)))
        ray_ids = hits["ray_ids"].numpy().astype(int)
        faces = hits["primitive_ids"].numpy().astype(int)
        distance = hits["t_hit"].numpy()
        dot = np.einsum("ij,ij->i", component["normals"][faces], RAY_DIRECTIONS[ray_ids % 3])
        winding = np.bincount(ray_ids, weights=np.sign(dot), minlength=n*3).reshape(n, 3)
        unreliable = np.bincount(ray_ids, weights=((distance < NUMERIC_MARGIN)
                                    | (np.abs(dot) < 1e-7)), minlength=n*3).reshape(n, 3) > 0
        inside_rays = np.abs(winding) > .5
        inside = inside_rays.all(1) & ~unreliable.any(1)
        outside = (~inside_rays).all(1) & ~unreliable.any(1)
        uncertain = ~(inside | outside)
        ids = np.flatnonzero(uncertain)
        if len(ids):
            distances = component["scene"].compute_distance(
                o3d.core.Tensor(np.asarray(points[ids], np.float32))).numpy()
            for index, boundary_distance in zip(ids, distances):
                # A nearly coincident point can have an integer angle sum after
                # rounding; it must not receive an inside/outside certificate.
                if not np.isfinite(boundary_distance) or boundary_distance <= NUMERIC_MARGIN:
                    continue
                component["winding_fallback_queries"] += 1
                value = solid_angle_winding(component["triangles"], points[index])
                if not np.isfinite(value):
                    continue
                integer = round(value)
                if abs(value-integer) <= WINDING_INTEGER_TOLERANCE:
                    inside[index] = integer != 0
                    uncertain[index] = False
                    component["winding_fallback_resolved"] += 1
        return inside, uncertain

    def overlaps(self, triangles):
        lo, hi = triangles.min(1), triangles.max(1)
        possible = np.zeros(len(triangles), bool)
        for component in self.components:
            a, b = component["bounds"]
            possible |= np.all(hi >= a-NUMERIC_MARGIN, axis=1) & np.all(lo <= b+NUMERIC_MARGIN, axis=1)
        return possible

    def outside_with_clearance(self, points):
        valid = np.ones(len(points), bool)
        for component in self.components:
            lo, hi = component["bounds"]
            ids = np.flatnonzero(valid & np.all(points >= lo-NUMERIC_MARGIN, axis=1)
                                & np.all(points <= hi+NUMERIC_MARGIN, axis=1))
            if len(ids):
                occupied, uncertain = self.occupancy(component, points[ids])
                distance = component["scene"].compute_distance(
                    o3d.core.Tensor(points[ids].astype(np.float32))).numpy()
                valid[ids] &= ~occupied & ~uncertain & (distance >= NUMERIC_MARGIN)
        return valid

    def signed_bounds(self, points):
        """Return lower/upper signed union distances and nearest part labels."""
        points = np.asarray(points, np.float64)
        n = len(points)
        min_outside = np.full(n, np.inf)
        depth_lower = np.zeros(n)
        deepest = np.full(n, -1, int)
        inside = np.zeros(n, bool)
        ambiguous = np.zeros(n, bool)
        part = np.full(n, -1, int)
        observations = []
        for number, component in enumerate(self.components):
            d = component["scene"].compute_distance(
                o3d.core.Tensor(points.astype(np.float32))).numpy().astype(float)
            if not np.isfinite(d).all():
                raise InvalidSolid(dict(kind=self.kind, reason="non_finite_signed_distance"))
            occupied, uncertain = self.occupancy(component, points)
            observations.append((d, occupied, uncertain))
            closer = d < min_outside
            part[closer] = number
            min_outside = np.minimum(min_outside, d)
            inside |= occupied
            ambiguous |= uncertain
            better = occupied & (d-NUMERIC_MARGIN > depth_lower)
            deepest[better] = number
            depth_lower[better] = d[better]-NUMERIC_MARGIN
        phi_lower = min_outside-NUMERIC_MARGIN
        phi_upper = min_outside+NUMERIC_MARGIN
        possible_inside = inside | ambiguous
        if not possible_inside.any():
            return phi_lower, phi_upper, part
        # A point just beyond any side of the global box is provably outside.
        exit_distance = np.minimum(points-self.bounds[0], self.bounds[1]-points).min(1)
        upper = np.maximum(0., exit_distance) + 2*NUMERIC_MARGIN
        # Deep samples already reject; expensive witnesses only help certify
        # samples close enough to the 5 mm boundary to possibly pass.
        eligible = possible_inside & (depth_lower <= MAXIMUM_PENETRATION)
        for number, (component, observation) in enumerate(zip(self.components, observations)):
            d, occupied, uncertain = observation
            ids = np.flatnonzero(eligible & (occupied | uncertain))
            if not len(ids):
                continue
            hit = component["scene"].compute_closest_points(o3d.core.Tensor(points[ids].astype(np.float32)))
            closest = hit["points"].numpy().astype(float)
            direction = closest-points[ids]
            lengths = np.linalg.norm(direction, axis=1)
            normals = component["normals"][hit["primitive_ids"].numpy().astype(int)]
            direction = np.where((lengths > NUMERIC_MARGIN)[:, None],
                                 direction / np.maximum(lengths[:, None], 1e-12), normals)
            # Both sides are inspected: the nearest triangle can be an internal
            # layer in self-overlap. Only a verified outside witness is usable.
            for sign in (1., -1.):
                q = closest + sign * direction * (4*NUMERIC_MARGIN)
                valid = self.outside_with_clearance(q)
                upper[ids[valid]] = np.minimum(upper[ids[valid]],
                                              np.linalg.norm(q[valid]-points[ids[valid]], axis=1)+NUMERIC_MARGIN)
        # Near-surface signed values have no useful outward direction. Their
        # uncertainty stays conservative rather than being rounded to outside.
        phi_lower[possible_inside] = -upper[possible_inside]
        phi_upper[possible_inside] = -depth_lower[possible_inside]
        part[deepest >= 0] = deepest[deepest >= 0]
        return phi_lower, phi_upper, part


def surface_bounds(vertices, faces, target, source_labels=None, budget=QUERY_BUDGET):
    """Bound the entire source surface, including wholly interior triangles."""
    fallback_before = [sum(c[key] for c in target.components) for key in
                       ("winding_fallback_queries", "winding_fallback_resolved")]
    triangles = np.asarray(vertices, float)[faces]
    ids = np.arange(len(faces))
    possible = target.overlaps(triangles)
    triangles, ids = triangles[possible], ids[possible]
    lower_max = upper_max = area_lower = area_upper = area_estimate = 0.
    queries = 0; uncertain = False; worst = None; unresolved = []; uncertainty_reasons = []
    uncertainty_witness = None
    def unresolved_witness(point, face, reason, lower=None, upper=None):
        return dict(point=np.asarray(point).tolist(),source_face=int(face),
            source_part=(PART_NAMES.get(int(source_labels[face]), "other")
                         if source_labels is not None else "body"),
            reason=reason,lower_bound=lower,upper_bound=upper)
    while len(triangles):
        if queries + len(triangles) > budget:
            uncertain = True; unresolved.extend(ids.tolist()); upper_max = None
            uncertainty_reasons.append("query_budget_exhausted")
            centers = triangles.mean(1)
            radius = np.linalg.norm(triangles-centers[:,None],axis=-1).max(1)
            index = int(radius.argmax())
            # This is an unqueried cell center, not a measured penetration.
            # Both bounds stay null so downstream repair cannot mistake the
            # diagnostic point for a proven collision witness.
            uncertainty_witness = unresolved_witness(centers[index],ids[index],
                "query_budget_exhausted_before_sample")
            break
        centers = triangles.mean(1)
        radius = np.linalg.norm(triangles-centers[:, None], axis=-1).max(1)
        signed_lo, signed_hi, target_parts = target.signed_bounds(centers)
        queries += len(centers)
        point_lower = np.maximum(0., -signed_hi)
        cell_upper = np.maximum(0., -signed_lo + radius)
        index = int(point_lower.argmax())
        if worst is None or point_lower[index] > lower_max:
            lower_max = max(lower_max, float(point_lower[index]))
            component = target.components[int(target_parts[index])]
            worst = dict(point=centers[index].tolist(), source_face=int(ids[index]),
                         source_part=(PART_NAMES.get(int(source_labels[ids[index]]), "other")
                                      if source_labels is not None else "body"),
                         target_component=int(component["id"]), target_parts=component["parts"],
                         sample_lower_bound=float(point_lower[index]),
                         sample_upper_bound=float(max(0., -signed_lo[index])))
        accepted = cell_upper <= MAXIMUM_PENETRATION
        failed = point_lower > MAXIMUM_PENETRATION
        too_small = radius <= MIN_CELL_RADIUS
        final = accepted | failed | too_small
        # One measured counterexample suffices to reject. Stop refining this
        # surface, retaining valid (possibly loose) bounds for every queued cell.
        proven_failure = lower_max > MAXIMUM_PENETRATION
        if proven_failure:
            final[:] = True
        unresolved_here = too_small & ~accepted & ~failed & (not proven_failure)
        if unresolved_here.any():
            uncertain = True; unresolved.extend(ids[unresolved_here].tolist())
            uncertainty_reasons.append("minimum_cell_radius_reached")
            index = int(np.where(unresolved_here,cell_upper,-np.inf).argmax())
            if uncertainty_witness is None or cell_upper[index] > uncertainty_witness["upper_bound"]:
                uncertainty_witness = unresolved_witness(centers[index],ids[index],
                    "minimum_cell_radius_reached",float(point_lower[index]),float(cell_upper[index]))
        if final.any():
            upper_max = max(upper_max, float(cell_upper[final].max()))
            area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                           triangles[:, 2]-triangles[:, 0]), axis=1)/2
            fully_inside = signed_hi + radius < 0
            maybe_inside = signed_lo - radius < 0
            area_lower += float(area[final & fully_inside].sum())
            area_upper += float(area[final & maybe_inside].sum())
            area_estimate += float(area[final & (signed_hi < 0)].sum())
        pending = ~final
        triangles, ids = _children(triangles[pending]), np.repeat(ids[pending], 4)
        if len(triangles):
            possible = target.overlaps(triangles)
            triangles, ids = triangles[possible], ids[possible]
    return dict(lower_bound=lower_max, upper_bound=upper_max, uncertain=uncertain,
                query_count=queries, unresolved_face_indices=sorted(set(unresolved)),
                uncertainty_reasons=sorted(set(uncertainty_reasons)),
                uncertainty_witness=uncertainty_witness,
                winding_fallback_queries=sum(c["winding_fallback_queries"] for c in target.components)-fallback_before[0],
                winding_fallback_resolved=sum(c["winding_fallback_resolved"] for c in target.components)-fallback_before[1],
                worst=worst, penetrating_area_lower_bound=area_lower,
                penetrating_area_upper_bound=(None if upper_max is None else area_upper),
                penetrating_area_estimate=area_estimate)


def collision_checks(collision):
    if collision["maximum_penetration"] > MAXIMUM_PENETRATION:
        return ["full_motion_penetration"]
    upper = collision.get("maximum_penetration_upper_bound")
    complete = collision.get("checked_source_frames", 0) == collision.get("required_source_frames", -1)
    if collision.get("uncertain_frames") or not complete or upper is None or upper > MAXIMUM_PENETRATION:
        return ["collision_depth_undetermined"]
    return []


def furniture_bounds(body, faces, exact, outer):
    """Bound depth into the unknown solid between its verified inner and outer unions."""
    inner = surface_bounds(body, faces, exact)
    if exact is outer:
        return inner
    envelope = surface_bounds(body, faces, outer)
    high = envelope["upper_bound"]
    uncertain = high is None or high > MAXIMUM_PENETRATION
    inner.update(upper_bound=high, uncertain=uncertain and inner["lower_bound"] <= MAXIMUM_PENETRATION,
                 penetrating_area_upper_bound=envelope["penetrating_area_upper_bound"],
                 envelope_query_count=envelope["query_count"])
    if inner["uncertain"]:
        inner["uncertainty_reasons"] = sorted(set(envelope["uncertainty_reasons"] + ["proxy_contact_unverified"]))
    else:
        inner["uncertainty_reasons"] = []
    # The closest exit from the inner union may lie inside an outer envelope.
    if inner["worst"]:
        inner["worst"]["sample_upper_bound"] = high
    return inner


def collision_check(ctx, mesh, parts=None, prepared=None, pose=None):
    """Original surfaces prove lower bounds. Certified outer geometry bounds possible depth."""
    total = len(ctx["dense"])
    report = dict(maximum_penetration=0., maximum_penetration_lower_bound=0.,
                  maximum_penetration_upper_bound=0., worst_collision_frame=None,
                  edge_triangle_crossings=0, checked_source_frames=0,
                  geometry_checked_source_frames=0, required_source_frames=total,
                  body_boundary_caps=[], frames=[], uncertain_frames=[],
                  uncertainty_frame=None, uncertainty_reasons=[],
                  maximum_penetrating_surface_area_estimate=0., mesh_issues=[],
                  method="bidirectional nonzero-winding solid union, verified outside witnesses and adaptive full-surface Lipschitz bounds",
                  numeric_margin_m=NUMERIC_MARGIN, minimum_cell_radius_m=MIN_CELL_RADIUS,
                  query_budget_per_direction_frame=QUERY_BUDGET,
                  occupancy="three fixed oriented all-hit rays; disagreements use full float64 solid-angle winding; near-boundary/non-integer fallback stays uncertain",
                  winding_integer_tolerance=WINDING_INTEGER_TOLERANCE,
                  scan_policy="stop after a proven violation; partial furniture geometry still permits original-surface evidence",
                  maximum_definition="raw furniture to capped body; original furniture solids bound depth below and validated envelopes bound it above")
    try:
        labels = None if parts is None else parts.get("face_part_ids")
        if labels is not None:
            labels = np.asarray(labels)
            if labels.shape != (len(mesh.faces),) or labels.dtype.kind not in "iu":
                raise InvalidSolid(dict(kind="furniture", reason="invalid_face_part_ids"))
        if labels is None:
            labels = np.full(len(mesh.faces), -1, np.int8)
        if prepared is None:
            prepared = prepare_geometry(mesh, parts)
        rotation, translation, scale = (np.eye(3), np.zeros(3), 1.) if pose is None else pose
        pieces = prepared.posed(rotation, translation, scale)
        chair = ClosedSolid(mesh.vertices, mesh.faces, pieces, "furniture", prepared=True)
        geometry = prepared.summary()
        report["geometry"] = geometry
        unknown = [r["id"] for r in prepared.records if r["mode"] == "unknown"]
        boxes = {}
        for r in prepared.records:
            if r["mode"] == "bounds_only":
                corners = trimesh.bounds.corners(r["bounds"]) @ rotation.T * scale + translation
                boxes[r["id"]] = (corners.min(0), corners.max(0))
        separated_counts = {}
        geometry["separated_frames_by_component"] = separated_counts
        original_components = {c["id"]:c for c in chair.components if c["mode"] != "hull"}
        exact = chair
        if len(original_components) < len(pieces):
            exact = copy(chair)
            exact.components = list(original_components.values())
        hull_boxes = {c["id"]:c["bounds"] for c in chair.components if c["mode"] == "hull"}
        partial = bool(unknown or boxes or len(original_components) != len(pieces))
        faces, loops = close_body_topology(np.asarray(ctx["faces"]), ctx["dense"].shape[1])
        report["body_boundary_caps"] = [len(loop) for loop in loops]
        first = np.asarray(ctx["dense"][0])
        capped = np.vstack([first]+[first[loop].mean(0)[None] for loop in loops])
        body_topology = component_topology(capped, faces)
        chair_edges = mesh.edges_unique
        body_edges = trimesh.Trimesh(first, ctx["faces"], process=False).edges_unique
        for frame in range(total):
            body = np.asarray(ctx["dense"][frame])
            capped = np.vstack([body]+[body[loop].mean(0)[None] for loop in loops])
            report["geometry_checked_source_frames"] = frame+1
            target = ClosedSolid(capped, faces, body_topology, "body")
            chair_to_body = surface_bounds(mesh.vertices, mesh.faces, target, labels)
            # Original visible body surface is the source; caps define only the
            # target solid. Caps are never reported as physical human surface.
            # A bound-only part may also occlude an exit witness for another
            # solid. Exclude it only outside the body's full 5 mm neighbourhood.
            body_min, body_max = capped.min(0), capped.max(0)
            padding = MAXIMUM_PENETRATION + 2*NUMERIC_MARGIN
            nearby, active_hulls = [], []
            triangle_bounds = None
            for i, (lo, hi) in {**boxes, **hull_boxes}.items():
                separated = np.any(hi < body_min-padding) or np.any(lo > body_max+padding)
                if not separated:
                    if triangle_bounds is None:
                        triangles = capped[faces]
                        triangle_bounds = (triangles.min(1), triangles.max(1))
                    intersects = (np.all(triangle_bounds[1] >= lo-padding, axis=1)
                                  & np.all(triangle_bounds[0] <= hi+padding, axis=1))
                    # A box with no nearby boundary triangles can still be
                    # wholly inside the body. Verify an exterior representative.
                    separated = not intersects.any() and target.outside_with_clearance(((lo+hi)/2)[None])[0]
                if separated:
                    separated_counts[i] = separated_counts.get(i, 0) + 1
                elif i in boxes:
                    nearby.append(i)
                else:
                    active_hulls.append(i)
            outer = exact
            if active_hulls:
                outer = copy(chair)
                outer.components = [c for c in chair.components if c["mode"] != "hull" or c["id"] in active_hulls]
            body_to_chair = furniture_bounds(body, ctx["faces"], exact, outer)
            # A distant envelope cannot affect an exit within 5 mm. A looser
            # bound must include every envelope, even one outside that radius.
            if boxes or hull_boxes:
                high = body_to_chair["upper_bound"]
                if high is not None and high > MAXIMUM_PENETRATION and (boxes or len(active_hulls) < len(hull_boxes)):
                    body_to_chair["upper_bound"] = None
            if unknown or nearby:
                body_to_chair["uncertain"] = True
                body_to_chair["upper_bound"] = None
                body_to_chair["uncertainty_reasons"].append("unverified_furniture_extent")
                body_to_chair["nearby_bounds_only"] = nearby
            if body_to_chair["upper_bound"] is None:
                body_to_chair["penetrating_area_upper_bound"] = None
                if body_to_chair.get("worst"):
                    body_to_chair["worst"]["sample_upper_bound"] = None
            directions = {"chair_to_body": chair_to_body, "body_to_chair": body_to_chair}
            lower = max(item["lower_bound"] for item in directions.values())
            upper = (None if any(item["upper_bound"] is None for item in directions.values())
                     else max(item["upper_bound"] for item in directions.values()))
            uncertain = any(item["uncertain"] for item in directions.values())
            if uncertain:
                report["uncertain_frames"].append(frame)
                if report["uncertainty_frame"] is None:
                    report["uncertainty_frame"] = frame
                report["uncertainty_reasons"].extend(
                    dict(frame=frame, direction=name, reasons=item["uncertainty_reasons"])
                    for name, item in directions.items() if item["uncertain"])
            if lower > report["maximum_penetration"]:
                report["maximum_penetration"] = report["maximum_penetration_lower_bound"] = lower
                report["worst_collision_frame"] = frame
                name = max(directions, key=lambda key: directions[key]["lower_bound"])
                witness = directions[name].get("worst")
                if witness:
                    evidence = dict(witness, frame=frame, direction=name)
                    if name == "chair_to_body":
                        i = int(prepared.face_components[witness["source_face"]])
                        evidence.update(furniture_component=i, furniture_instance=prepared.records[i]["instance"],
                                        furniture_face=witness["source_face"], furniture_point=witness["point"])
                    else:
                        comp = original_components[witness["target_component"]]
                        hit = comp["scene"].compute_closest_points(o3d.core.Tensor([witness["point"]], dtype=o3d.core.Dtype.Float32))
                        face = int(comp["face_ids"][hit["primitive_ids"].numpy()[0]])
                        evidence.update(furniture_component=comp["id"], furniture_instance=comp["instance"],
                                        furniture_face=face, furniture_point=hit["points"].numpy()[0].tolist())
                    evidence["furniture_part"] = PART_NAMES.get(int(labels[evidence["furniture_face"]]), "furniture")
                    report["penetration_witness"] = evidence
            if upper is None or report["maximum_penetration_upper_bound"] is None:
                report["maximum_penetration_upper_bound"] = None
            else:
                report["maximum_penetration_upper_bound"] = max(report["maximum_penetration_upper_bound"], upper)
            crossings = len(edge_intersections(target.scene, mesh.vertices, chair_edges, target.bounds))
            crossings += len(edge_intersections(chair.scene, body, body_edges, chair.bounds))
            report["edge_triangle_crossings"] += crossings
            area = sum(item["penetrating_area_estimate"] for item in directions.values())
            report["maximum_penetrating_surface_area_estimate"] = max(report["maximum_penetrating_surface_area_estimate"], area)
            report["frames"].append(dict(frame=frame, maximum_penetration=lower,
                penetration_upper_bound=upper, edge_triangle_crossings=crossings,
                penetrating_surface_area_estimate=area, directions=directions,
                uncertain=uncertain))
            report["checked_source_frames"] = frame+1
            if (lower > MAXIMUM_PENETRATION or (uncertain and not partial)) and not ctx.get("collision_scan_rejected_frames", False):
                break
    except (InvalidSolid, ValueError) as exc:
        issue = exc.issue if isinstance(exc, InvalidSolid) else dict(reason=str(exc))
        issue["source_frame"] = report["checked_source_frames"]
        report["mesh_issues"].append(issue)
        if report["uncertainty_frame"] is None:
            report["uncertainty_frame"] = report["checked_source_frames"]
        report["uncertainty_reasons"].append(dict(frame=report["checked_source_frames"],
            direction="geometry", reasons=[issue["reason"]]))
        report["uncertain_frames"] = sorted(set(report["uncertain_frames"]) | set(range(report["checked_source_frames"], total)))
        report["maximum_penetration_upper_bound"] = None
    report["checked_frames_penetration_upper_bound"] = report["maximum_penetration_upper_bound"]
    if report["checked_source_frames"] < total:
        # A bound on inspected frames is never a bound on the full motion.
        report["maximum_penetration_upper_bound"] = None
    report["status"] = ("rejected" if report["maximum_penetration"] > MAXIMUM_PENETRATION
                        else "undetermined" if collision_checks(report) else "certified_below_limit")
    report["worst_collision_frame_scope"] = ("all source frames" if report["checked_source_frames"] == total
                                             else "checked source frames only")
    report["worst_collision_frame_selection"] = "largest measured penetration lower bound; not a claim of the exact continuous maximizer"
    return report
