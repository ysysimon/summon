"""Experimental CUDA queries, never used as collision certificates.

Mesh/BVH storage persists for the scene lifetime. Signed distance uses fast
winding only as optimizer guidance; all final acceptance stays on Open3D CPU.
"""
from collections import OrderedDict
import hashlib
import numpy as np
import warp as wp
from .runtime import current_profile


@wp.kernel
def _distance(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), signed: int,
              distances: wp.array(dtype=float)):
    i = wp.tid()
    p = points[i]
    if signed != 0:
        q = wp.mesh_query_point_sign_winding_number(mesh, p, 1.0e30)
        if q.result:
            nearest = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
            distances[i] = wp.length(nearest - p) * q.sign
        else:
            distances[i] = wp.nan
    else:
        q = wp.mesh_query_point_no_sign(mesh, p, 1.0e30)
        if q.result:
            nearest = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
            distances[i] = wp.length(nearest - p)
        else:
            distances[i] = wp.nan


@wp.kernel
def _rays(mesh: wp.uint64, origins: wp.array(dtype=wp.vec3), directions: wp.array(dtype=wp.vec3),
          distances: wp.array(dtype=float), normals: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    q = wp.mesh_query_ray(mesh, origins[i], directions[i], 1.0e30)
    if q.result:
        distances[i] = q.t
        normals[i] = q.normal
    else:
        distances[i] = wp.inf
        normals[i] = wp.vec3(0.0)


@wp.kernel
def _pose(points: wp.array(dtype=wp.vec3), rotation: wp.mat33, translation: wp.vec3,
          scale: float, inverse: int, output: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    if inverse != 0:
        output[i] = wp.transpose(rotation) * (points[i] - translation) / scale
    else:
        output[i] = rotation * points[i] * scale + translation


@wp.kernel
def _seat(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), rotation: wp.mat33,
          translation: wp.vec3, scale: float, gaps: wp.array(dtype=float), valid: wp.array(dtype=int)):
    i = wp.tid()
    local = wp.transpose(rotation) * (points[i] - translation) / scale
    q = wp.mesh_query_ray(mesh, local + wp.vec3(0.0, 0.0, 3.0), wp.vec3(0.0, 0.0, -1.0), 1.0e30)
    if q.result:
        gaps[i] = (q.t - 3.0) * scale
        valid[i] = int(q.normal[2] > 0.35)
    else:
        gaps[i] = wp.inf
        valid[i] = 0


class ArrayResult:
    def __init__(self, value):
        self.value = value
    def numpy(self):
        return self.value


class WarpScene:
    def __init__(self, vertices, faces, profile=None):
        wp.init()
        if not wp.is_cuda_available():
            raise RuntimeError("Warp backend explicitly requested but CUDA is unavailable")
        self.device = wp.get_device("cuda:0")
        self.profile = profile
        self.vertices = wp.array(np.asarray(vertices, np.float32), dtype=wp.vec3, device=self.device)
        self.indices = wp.array(np.asarray(faces, np.int32).reshape(-1), dtype=wp.int32, device=self.device)
        self.mesh = wp.Mesh(points=self.vertices, indices=self.indices, support_winding_number=True)
        self._points = OrderedDict()

    def _resident(self, points):
        points = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
        key = (points.shape, hashlib.sha256(points.tobytes()).digest())
        if key not in self._points:
            self._points[key] = wp.array(points, dtype=wp.vec3, device=self.device)
            if len(self._points) > 8:
                self._points.popitem(last=False)
        else:
            self._points.move_to_end(key)
        return self._points[key]

    @staticmethod
    def _transform(rotation, translation, scale):
        return wp.mat33(*np.asarray(rotation, np.float32).reshape(-1)), wp.vec3(*np.asarray(translation, np.float32)), float(scale)

    def transformed_distance(self, points, rotation, translation, scale, inverse=True, signed=False):
        """Transform resident samples; inverse returns distance in asset units."""
        points = np.asarray(points)
        def run():
            if not len(points):
                return ArrayResult(np.empty(0, np.float32))
            source = self._resident(points)
            r, t, s = self._transform(rotation, translation, scale)
            posed = wp.empty(len(points), dtype=wp.vec3, device=self.device)
            distance = wp.empty(len(points), dtype=float, device=self.device)
            wp.launch(_pose, dim=len(points), inputs=[source, r, t, s, int(inverse)], outputs=[posed], device=self.device)
            wp.launch(_distance, dim=len(points), inputs=[self.mesh.id, posed, int(signed)], outputs=[distance], device=self.device)
            values = distance.numpy()
            if not np.isfinite(values).all():
                raise RuntimeError("Warp transformed distance query returned an invalid result")
            return ArrayResult(values)
        return self._run("transformed_signed_distance" if signed else "transformed_distance", points, run)

    def support_query(self, points, rotation, translation, scale):
        points = np.asarray(points)
        def run():
            if not len(points):
                return np.empty(0, np.float32), np.empty(0, bool)
            source = self._resident(points)
            r, t, s = self._transform(rotation, translation, scale)
            gaps = wp.empty(len(points), dtype=float, device=self.device)
            valid = wp.empty(len(points), dtype=int, device=self.device)
            wp.launch(_seat, dim=len(points), inputs=[self.mesh.id, source, r, t, s], outputs=[gaps, valid], device=self.device)
            return gaps.numpy(), valid.numpy().astype(bool)
        return self._run("seat", points, run)

    @staticmethod
    def _numpy(points):
        return np.asarray(points.numpy() if hasattr(points, "numpy") else points, np.float32)

    def _run(self, name, points, fn):
        profile = self.profile if self.profile is not None else current_profile()
        try:
            if profile is None or not profile.enabled:
                return fn()
            profile.count("query.warp." + name + ".calls")
            profile.count("query.warp." + name + ".points", len(points))
            with profile.time("query.warp." + name):
                return fn()
        except Exception as exc:
            raise RuntimeError("Explicit Warp backend query failed: " + name) from exc

    def _distance(self, points, signed):
        points = self._numpy(points)
        shape = points.shape[:-1]; points = points.reshape(-1, 3)
        def run():
            if not len(points):
                return ArrayResult(np.empty(shape, np.float32))
            p = wp.array(points, dtype=wp.vec3, device=self.device)
            result = wp.empty(len(points), dtype=float, device=self.device)
            wp.launch(_distance, dim=len(points), inputs=[self.mesh.id, p, int(signed)], outputs=[result], device=self.device)
            values = result.numpy().reshape(shape)
            if not np.isfinite(values).all():
                raise RuntimeError("Warp mesh distance query returned an invalid result")
            return ArrayResult(values)
        return self._run("signed_distance" if signed else "distance", points, run)

    def compute_distance(self, points):
        return self._distance(points, False)

    def compute_signed_distance(self, points):
        return self._distance(points, True)

    def cast_rays(self, rays):
        rays = self._numpy(rays)
        shape = rays.shape[:-1]; rays = rays.reshape(-1, 6)
        def run():
            if not len(rays):
                return dict(t_hit=ArrayResult(np.empty(shape, np.float32)), primitive_normals=ArrayResult(np.empty(shape + (3,), np.float32)))
            origins = wp.array(rays[:, :3].copy(), dtype=wp.vec3, device=self.device)
            directions = wp.array(rays[:, 3:].copy(), dtype=wp.vec3, device=self.device)
            distances = wp.empty(len(rays), dtype=float, device=self.device)
            normals = wp.empty(len(rays), dtype=wp.vec3, device=self.device)
            wp.launch(_rays, dim=len(rays), inputs=[self.mesh.id, origins, directions], outputs=[distances, normals], device=self.device)
            return dict(t_hit=ArrayResult(distances.numpy().reshape(shape)), primitive_normals=ArrayResult(normals.numpy().reshape(shape + (3,))))
        return self._run("cast_rays", rays, run)
