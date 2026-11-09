"""Backend equivalence and conservative query pruning regressions."""
import argparse
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d
import trimesh
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chair_support.runtime import ExecutionOptions, Profile, add_execution_arguments, execution_scope
from chair_support.query_backend import query_scene, cpu_scene
from chair_support.collision import ClosedSolid, component_topology, NUMERIC_MARGIN


class RuntimeTests(unittest.TestCase):
    def test_arguments_roundtrip_and_thread_budget(self):
        with patch("chair_support.runtime.available_cpus", return_value=8):
            options = ExecutionOptions(profile_fit=True, fit_workers=2)
            parser = argparse.ArgumentParser(); add_execution_arguments(parser)
            self.assertEqual(ExecutionOptions.from_args(parser.parse_args(options.argv())), options)
            self.assertEqual(options.resolved_query_threads(), 4)
            with self.assertRaises(ValueError): ExecutionOptions(fit_workers=2, query_threads=8)
            with self.assertRaises(ValueError): ExecutionOptions(fit_workers=2, fit_query_backend="warp")

    def test_cached_cpu_scene_profiles_current_candidate(self):
        mesh = trimesh.creation.box()
        first, second = Profile(True), Profile(True)
        with execution_scope(ExecutionOptions(), first):
            scene = cpu_scene(mesh.vertices, mesh.faces)
        with execution_scope(ExecutionOptions(), second):
            scene.compute_distance(o3d.core.Tensor([[2.,0,0]], dtype=o3d.core.Dtype.Float32))
        self.assertNotIn("query.compute_distance.calls", first.counters)
        self.assertEqual(second.counters["query.compute_distance.calls"], 1)

    def test_strict_scene_does_not_import_warp(self):
        mesh = trimesh.creation.box()
        with execution_scope(ExecutionOptions(fit_query_backend="warp")):
            scene = cpu_scene(mesh.vertices, mesh.faces)
            d = scene.compute_distance(o3d.core.Tensor([[2.,0,0]], dtype=o3d.core.Dtype.Float32)).numpy()
        self.assertAlmostEqual(float(d[0]), 1.5)


class PruningTests(unittest.TestCase):
    def test_outside_points_skip_occupancy_but_keep_unsigned_distance(self):
        mesh = trimesh.creation.box(extents=[.1,.1,.1])
        solid = ClosedSolid(mesh.vertices, mesh.faces, component_topology(mesh.vertices, mesh.faces), "body")
        points = np.array([[.2,0,0],[0,0,0],[.05 + NUMERIC_MARGIN*.5,0,0]])
        original = solid.occupancy
        seen = []
        def recording(component, queries):
            seen.extend(queries.tolist()); return original(component, queries)
        with patch.object(solid, "occupancy", side_effect=recording):
            low, high, _ = solid.signed_bounds(points)
        self.assertFalse(any(np.allclose(p, points[0]) for p in seen))
        self.assertTrue(any(np.array_equal(p, points[2]) for p in seen))
        self.assertGreater(low[0], .149)
        self.assertLess(high[1], 0)

    def test_pruned_and_unpruned_bounds_match_on_box_union(self):
        a = trimesh.creation.box(extents=[.1,.1,.1]); b = a.copy(); b.apply_translation([.07,0,0])
        mesh = trimesh.util.concatenate([a,b])
        solid = ClosedSolid(mesh.vertices, mesh.faces, component_topology(mesh.vertices,mesh.faces), "body")
        points = np.random.RandomState(7).uniform(-.2,.2,(100,3))
        actual = solid.signed_bounds(points)
        bounds = [c['bounds'].copy() for c in solid.components]
        for c in solid.components: c['bounds'] = np.array([[-10.,-10.,-10.],[10.,10.,10.]])
        expected = solid.signed_bounds(points)
        for x,y in zip(actual,expected): np.testing.assert_allclose(x,y,atol=1e-8,rtol=0)


@unittest.skipUnless(os.environ.get("YSYNTHETIC_TEST_WARP") == "1", "GPU smoke requires explicit opt-in")
class WarpTests(unittest.TestCase):
    def test_positive_fit_keeps_cpu_strict_certification(self):
        import test_supported_chair as fixture
        from chair_support.engine import fit_asset
        baseline = fixture.SupportTests().fit_control()
        def gpu_fit(path, ctx, *args, **kwargs):
            ctx['_execution'] = ExecutionOptions(fit_query_backend='warp')
            return fit_asset(path,ctx,*args,**kwargs)
        with patch.object(fixture, 'fit_asset', side_effect=gpu_fit):
            actual = fixture.SupportTests().fit_control()
        for result in (baseline,actual):
            self.assertTrue(result['qualified'],result['rejection_reasons'])
            collision = result['metrics']['collision']
            self.assertEqual(collision['checked_source_frames'],3)
            self.assertLessEqual(collision['maximum_penetration_upper_bound'],.005)
            self.assertIn('nonzero-winding',collision['method'])

    def test_cuda_distance_signed_distance_and_seat_rays(self):
        mesh = trimesh.creation.box(extents=[1.,1.,1.])
        gpu = query_scene(mesh.vertices, mesh.faces, ExecutionOptions(fit_query_backend="warp"))
        cpu = cpu_scene(mesh.vertices, mesh.faces)
        points = o3d.core.Tensor([[0.,0.,0.],[.7,.2,0.],[-.8,0.,.1]], dtype=o3d.core.Dtype.Float32)
        for name in ("compute_distance","compute_signed_distance"):
            np.testing.assert_allclose(getattr(gpu,name)(points).numpy(),getattr(cpu,name)(points).numpy(),atol=1e-5,rtol=0)
        rays = o3d.core.Tensor([[.1,.2,3.,0.,0.,-1.],[2.,0.,3.,0.,0.,-1.]], dtype=o3d.core.Dtype.Float32)
        g,c = gpu.cast_rays(rays),cpu.cast_rays(rays)
        np.testing.assert_allclose(g['t_hit'].numpy(),c['t_hit'].numpy(),atol=1e-5,rtol=0)
        np.testing.assert_allclose(g['primitive_normals'].numpy()[0],c['primitive_normals'].numpy()[0],atol=1e-5,rtol=0)

    def test_resident_transforms_and_support_match_cpu_queries(self):
        mesh = trimesh.creation.box(extents=[1.,1.,1.])
        gpu = query_scene(mesh.vertices, mesh.faces, ExecutionOptions(fit_query_backend="warp"))
        cpu = cpu_scene(mesh.vertices, mesh.faces)
        r = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        t = np.array([.2,.3,0.]); scale = 1.1
        local = np.array([[.1,.2,.51],[.6,.2,.5]])
        points = local @ r.T * scale + t
        actual = gpu.transformed_distance(points,r,t,scale).numpy()
        expected = cpu.compute_distance(o3d.core.Tensor(local.astype(np.float32))).numpy()
        np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=0)
        actual = gpu.transformed_distance(local,r,t,scale,inverse=False,signed=True).numpy()
        expected = cpu.compute_signed_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()
        np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=0)
        gaps, valid = gpu.support_query(points,r,t,scale)
        self.assertEqual(valid.tolist(), [True,False])
        self.assertAlmostEqual(gaps[0], .011, places=5)
        before = len(gpu._points)
        gpu.support_query(points,r,t,scale)
        self.assertEqual(len(gpu._points),before)
        points[0,0] += .1
        gpu.support_query(points,r,t,scale)
        self.assertEqual(len(gpu._points),before+1)


if __name__ == '__main__': unittest.main()
