"""Geometry counterexamples for the 5 mm collision acceptance contract."""
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chair_support.collision import (ClosedSolid, component_topology, collision_check,
    collision_checks, edge_intersections, ray_scene, surface_bounds,
    close_body_topology, solid_angle_winding)


def context(body, frames=1):
    return {"faces":np.asarray(body.faces), "dense":np.repeat(np.asarray(body.vertices)[None], frames, axis=0)}


def triangle_prism():
    angles = np.deg2rad([0., 90., 210.])
    lower = np.column_stack((.0125*np.cos(angles), .0125*np.sin(angles), np.full(3, .0149)))
    vertices = np.vstack((lower, lower+[0, 0, .0001]))
    faces = [[2,1,0], [3,4,5], [0,1,4], [0,4,3], [1,2,5], [1,5,4], [2,0,3], [2,3,5]]
    return trimesh.Trimesh(vertices, faces, process=False)


class CollisionContractTests(unittest.TestCase):
    def test_full_solid_angle_preserves_integer_overlap_multiplicity(self):
        a = trimesh.creation.box(extents=[.04,.04,.04])
        b = a.copy(); b.apply_translation([.02,0,0])
        mesh = trimesh.util.concatenate((a,b))
        self.assertAlmostEqual(solid_angle_winding(mesh.triangles, [.01,0,0]), 2.)
        self.assertAlmostEqual(solid_angle_winding(mesh.triangles, [.1,0,0]), 0.)

    def test_float64_fallback_keeps_noninteger_and_boundary_uncertainty(self):
        box = trimesh.creation.box(extents=[.1,.1,.1])
        solid = ClosedSolid(box.vertices, box.faces,
                            component_topology(box.vertices,box.faces), "body")
        component = solid.components[0]
        real_scene = component["scene"]
        # Reproduce a backend ray disagreement without selecting different rays.
        class DisagreeingScene:
            def list_intersections(self, rays):
                return {"ray_ids":o3d.core.Tensor([0],o3d.core.Dtype.UInt32),
                        "primitive_ids":o3d.core.Tensor([0],o3d.core.Dtype.UInt32),
                        "t_hit":o3d.core.Tensor([.2],o3d.core.Dtype.Float32)}
            def compute_distance(self, points):
                return real_scene.compute_distance(points)
        component["scene"] = DisagreeingScene()
        with patch("chair_support.collision.solid_angle_winding", return_value=.25):
            inside, uncertain = solid.occupancy(component,np.array([[.2,0,0]]))
            self.assertFalse(inside[0]); self.assertTrue(uncertain[0])
        with patch("chair_support.collision.solid_angle_winding", return_value=0.) as classifier:
            inside, uncertain = solid.occupancy(component,np.array([[.05,0,0]]))
            self.assertFalse(inside[0]); self.assertTrue(uncertain[0])
            classifier.assert_not_called()

    def test_exact_five_mm_remains_undetermined(self):
        box = trimesh.creation.box(extents=[.1,.1,.1])
        solid = ClosedSolid(box.vertices,box.faces,
                            component_topology(box.vertices,box.faces),"body")
        # The entire tiny triangle is exactly 5 mm below the top. Numeric
        # margins must remain in force even when spatial refinement is tiny.
        vertices = np.array([[0,0,.045],[.00001,0,.045],[0,.00001,.045]])
        result = surface_bounds(vertices,np.array([[0,1,2]]),solid)
        self.assertTrue(result["uncertain"])
        self.assertLess(result["lower_bound"], .005)
        self.assertGreater(result["upper_bound"], .005)
        self.assertEqual(result["uncertainty_witness"]["source_face"],0)
        self.assertEqual(result["uncertainty_witness"]["reason"],"minimum_cell_radius_reached")
        self.assertGreater(result["uncertainty_witness"]["upper_bound"],.005)

    def test_unknown_frame_stops_scan_without_mislabeling_measured_worst(self):
        body = trimesh.creation.box(extents=[.15,.15,.5])
        furniture = trimesh.creation.box(extents=[.04,.04,.02])
        furniture.apply_translation([0,0,-.257])
        # Exercise scan orchestration independently from triangle refinement:
        # frame 0 has a larger measured lower bound, whereas frame 1 cannot
        # resolve its surface within budget. Frame 2 must never be inspected.
        def direction(lower, unknown=False):
            return dict(lower_bound=lower,upper_bound=None if unknown else .0045,
                uncertain=unknown,query_count=250000 if unknown else 10,
                unresolved_face_indices=[0] if unknown else [],
                uncertainty_reasons=["query_budget_exhausted"] if unknown else [],
                worst=None,penetrating_area_estimate=0.)
        with patch("chair_support.collision.surface_bounds", side_effect=[
                direction(.004),direction(.003),direction(.001,True),direction(.001)]) as bound:
            result=collision_check(context(body,3),furniture)
        self.assertEqual(bound.call_count,4)
        self.assertEqual(result["checked_source_frames"],2)
        self.assertEqual(result["required_source_frames"],3)
        self.assertEqual(result["status"],"undetermined")
        self.assertEqual(result["uncertainty_frame"],1)
        self.assertEqual(result["uncertain_frames"],[1])
        self.assertEqual(result["worst_collision_frame"],0)
        self.assertEqual(result["worst_collision_frame_scope"],"checked source frames only")
        self.assertIsNone(result["maximum_penetration_upper_bound"])
        self.assertEqual(result["uncertainty_reasons"],[dict(frame=1,
            direction="chair_to_body",reasons=["query_budget_exhausted"])])

    def test_unresolved_geometry_stops_before_later_frames(self):
        # Exactly 5 mm overlap needs a numerical interval straddling the limit;
        # a budget/minimum-cell unknown on frame 0 cannot be rescued by frame 1.
        body=trimesh.creation.box(extents=[.15,.15,.5])
        furniture=trimesh.creation.box(extents=[.00002,.00002,.02])
        furniture.apply_translation([0,0,-.255])
        result=collision_check(context(body,3),furniture)
        self.assertEqual(result["status"],"undetermined")
        self.assertEqual(result["checked_source_frames"],1)
        self.assertEqual(result["uncertainty_frame"],0)
        self.assertTrue(result["uncertainty_reasons"])
        self.assertIsNone(result["maximum_penetration_upper_bound"])

    def test_real_mirror_frame30_far_point_uses_float64_fallback(self):
        # Optional integration fixture: retain original dense geometry locally,
        # without redistributing licensed body assets in the test source tree.
        geometry = Path(__file__).resolve().parents[3]/"artifacts/summon/runs/supported-chair-A053-M-v6/geometry"
        if not (geometry/"smplx_verts.npy").is_file():
            self.skipTest("local A053 mirror dense-geometry integration fixture unavailable")
        dense = np.load(geometry/"smplx_verts.npy",mmap_mode="r")
        body = np.asarray(dense[30])
        faces, loops = close_body_topology(np.load(geometry/"smplx_faces.npy"),len(body))
        capped = np.vstack([body]+[body[loop].mean(0)[None] for loop in loops])
        solid = ClosedSolid(capped,faces,component_topology(capped,faces),"body")
        # Observed old all-hit rays [0,0,1], although this chair-base point is
        # 459.7 mm outside the body. Full winding is approximately 2e-18.
        point = np.array([[-.11311867333333332,.14678597333333335,.008477953333333335]])
        low, high, _ = solid.signed_bounds(point)
        self.assertGreater(low[0], .459)
        self.assertGreaterEqual(high[0], low[0])
        self.assertGreater(sum(c["winding_fallback_resolved"] for c in solid.components),0)

    def test_complete_body_containment_is_not_zero_penetration(self):
        body = trimesh.creation.box(extents=[.02, .02, .02])
        furniture = trimesh.creation.box(extents=[.1, .1, .1])
        result = collision_check(context(body), furniture)
        self.assertIn("full_motion_penetration", collision_checks(result))
        self.assertGreater(result["maximum_penetration"], .039)
        self.assertEqual(result["frames"][0]["directions"]["chair_to_body"]["lower_bound"], 0.)

    def test_fully_inside_triangle_requires_continuous_upper_bound(self):
        body = trimesh.creation.icosphere(subdivisions=3, radius=.02)
        furniture = triangle_prism()
        self.assertTrue(furniture.is_watertight)
        result = collision_check(context(body), furniture)
        self.assertTrue(collision_checks(result))
        self.assertGreater(result["maximum_penetration"], .005)
        self.assertEqual(result["edge_triangle_crossings"], 0)
        direction = result["frames"][0]["directions"]["chair_to_body"]
        self.assertGreater(direction["query_count"], len(furniture.faces))

    def test_open_furniture_does_not_receive_a_false_certificate(self):
        body = trimesh.creation.icosphere(subdivisions=2, radius=.02)
        furniture = triangle_prism()
        furniture = trimesh.Trimesh(furniture.vertices, furniture.faces[:1], process=False)
        furniture.apply_translation([1., 0., 0.])  # No raw surface witness and no authored extent.
        result = collision_check(context(body), furniture)
        self.assertEqual(result["status"], "undetermined")
        self.assertIn("collision_depth_undetermined", collision_checks(result))
        self.assertIsNone(result["maximum_penetration_upper_bound"])

    def test_segment_returns_all_entries_and_exits(self):
        first = trimesh.creation.box(extents=[.2, .4, .4])
        second = first.copy(); second.apply_translation([1,0,0])
        target = trimesh.util.concatenate((first, second))
        scene = ray_scene(target.vertices, target.faces)
        v = np.array([[-1., .031, .027], [2., .031, .027]])
        hits, edges, faces = edge_intersections(scene, v, np.array([[0,1]]), target.bounds, True)
        self.assertEqual(len(hits), 4)
        np.testing.assert_allclose(np.sort(hits[:,0]), [-.1,.1,.9,1.1], atol=1e-6)

    def test_three_mm_can_pass_twenty_mm_cannot(self):
        body = trimesh.creation.box(extents=[.15,.15,.5])
        for depth in (.003, .02):
            furniture = trimesh.creation.box(extents=[.04,.04,.02])
            furniture.apply_translation([0,0,-.26+depth])
            result = collision_check(context(body), furniture)
            if depth == .003:
                self.assertEqual(collision_checks(result), [])
                self.assertLessEqual(result["maximum_penetration_upper_bound"], .005)
                self.assertEqual(result["checked_source_frames"], 1)
            else:
                self.assertIn("full_motion_penetration", collision_checks(result))

    def test_all_frames_include_later_deep_penetration(self):
        body = trimesh.creation.box(extents=[.15,.15,.5])
        ctx = context(body, 3)
        ctx["dense"][2,:,2] -= .02
        furniture = trimesh.creation.box(extents=[.04,.04,.02])
        furniture.apply_translation([0,0,-.257])
        result = collision_check(ctx, furniture)
        self.assertEqual(result["checked_source_frames"], 3)
        self.assertEqual(result["worst_collision_frame"], 2)
        self.assertIn("full_motion_penetration", collision_checks(result))

    def test_overlapping_components_are_union_not_parity(self):
        a = trimesh.creation.box(extents=[.04,.04,.04])
        b = a.copy(); b.apply_translation([.02,0,0])
        mesh = trimesh.util.concatenate((a,b))
        solid = ClosedSolid(mesh.vertices,mesh.faces,component_topology(mesh.vertices,mesh.faces),"furniture")
        low, high, _ = solid.signed_bounds(np.array([[.01,0,0]]))
        self.assertLess(high[0], -.009)
        self.assertLessEqual(low[0], high[0])
        body = trimesh.creation.box(extents=[.002,.002,.002]);body.apply_translation([.01,0,0])
        result = collision_check(context(body), mesh)
        self.assertIn("full_motion_penetration", collision_checks(result))

    def test_subdivision_budget_exhaustion_is_undetermined(self):
        mesh = trimesh.creation.box(extents=[.1,.1,.1])
        solid = ClosedSolid(mesh.vertices,mesh.faces,component_topology(mesh.vertices,mesh.faces),"body")
        source = trimesh.creation.box(extents=[.1,.1,.1]);source.apply_translation([.097,0,0])
        result = surface_bounds(source.vertices,source.faces,solid,budget=1)
        self.assertTrue(result["uncertain"])
        self.assertIsNone(result["upper_bound"])
        self.assertEqual(result["uncertainty_witness"]["reason"],"query_budget_exhausted_before_sample")
        self.assertIsNone(result["uncertainty_witness"]["lower_bound"])
        self.assertIsNone(result["uncertainty_witness"]["upper_bound"])
        self.assertEqual(len(result["uncertainty_witness"]["point"]),3)

    def test_oriented_winding_keeps_overlapping_layers_occupied(self):
        a = trimesh.creation.box(extents=[.04,.04,.04])
        b = a.copy(); b.apply_translation([.02,0,0])
        mesh = trimesh.util.concatenate((a,b))
        # Exercise the raw oriented occupancy of multiple overlapping layers
        # together, as encountered in a connected self-overlapping human mesh.
        topology = [dict(id=0,vertex_ids=np.arange(len(mesh.vertices)),
                         face_ids=np.arange(len(mesh.faces)),faces=mesh.faces,parts=["body"])]
        solid = ClosedSolid(mesh.vertices,mesh.faces,topology,"body")
        inside,uncertain = solid.occupancy(solid.components[0], np.array([[.01,0,0],[.1,0,0]]))
        self.assertEqual(inside.tolist(), [True,False])
        self.assertFalse(uncertain.any())

    def test_internal_component_boundaries_are_not_union_exit_witnesses(self):
        a = trimesh.creation.box(extents=[.054,.1,.1]);a.apply_translation([.023,0,0])
        b = trimesh.creation.box(extents=[.1,.054,.1]);b.apply_translation([0,.023,0])
        furniture = trimesh.util.concatenate((a,b))
        body = trimesh.creation.box(extents=[.0002,.0002,.0002])
        result = collision_check(context(body), furniture)
        # Each component's closest surface is only 4 mm away, but exiting their
        # union requires passing the corner at (-4,-4) mm: >5.6 mm.
        self.assertTrue(collision_checks(result))
        self.assertNotEqual(result["status"], "certified_below_limit")

    def test_rejected_scan_does_not_claim_unchecked_frames(self):
        body=trimesh.creation.box(extents=[.02,.02,.02])
        furniture=trimesh.creation.box(extents=[.1,.1,.1])
        result=collision_check(context(body,680),furniture)
        self.assertEqual(result["checked_source_frames"],1)
        self.assertEqual(result["required_source_frames"],680)
        self.assertEqual(result["status"],"rejected")
        self.assertEqual(result["worst_collision_frame_scope"],"checked source frames only")
        self.assertIsNone(result["maximum_penetration_upper_bound"])
        self.assertIsNotNone(result["checked_frames_penetration_upper_bound"])


if __name__ == "__main__":
    unittest.main()
