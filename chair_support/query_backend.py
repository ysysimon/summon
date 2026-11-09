"""Optimization queries; strict certification explicitly uses cpu_scene."""
import numpy as np
import open3d as o3d
from .runtime import current_options, current_profile


class CPUScene:
    def __init__(self, vertices, faces, threads=0, profile=None):
        self.threads, self.profile = threads, profile
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, np.float32)),
                                 o3d.core.Tensor(np.asarray(faces, np.uint32)))

    def _query(self, method, points, **kwargs):
        profile = self.profile if self.profile is not None else current_profile()
        if profile is None or not profile.enabled:
            return getattr(self.scene, method)(points, nthreads=self.threads, **kwargs)
        profile.count("query." + method + ".calls")
        profile.count("query." + method + ".points", int(np.prod(points.shape[:-1])))
        with profile.time("query." + method):
            return getattr(self.scene, method)(points, nthreads=self.threads, **kwargs)

    def compute_distance(self, points):
        return self._query("compute_distance", points)

    def compute_signed_distance(self, points):
        return self._query("compute_signed_distance", points)

    def compute_closest_points(self, points):
        return self._query("compute_closest_points", points)

    def list_intersections(self, points):
        return self._query("list_intersections", points)

    def cast_rays(self, rays):
        return self._query("cast_rays", rays)


def cpu_scene(vertices, faces, execution=None, profile=None):
    execution = execution or current_options()
    return CPUScene(vertices, faces, execution.resolved_query_threads(), profile)


def query_scene(vertices, faces, execution=None, profile=None):
    execution = execution or current_options()
    if execution.fit_query_backend == "open3d":
        return cpu_scene(vertices, faces, execution, profile)
    try:
        from .warp_backend import WarpScene
    except ImportError as exc:
        raise RuntimeError("Warp fitting requires the separate environments/chair-fit-gpu environment") from exc
    try:
        return WarpScene(vertices, faces, profile=profile)
    except Exception as exc:
        raise RuntimeError("Explicit Warp backend initialization failed") from exc
