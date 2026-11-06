import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chair_support.collision import collision_check, collision_checks
from chair_support.engine import read_asset
from chair_support.geometry_audit import prepare_geometry, PROXY_ERROR


def motion(body, offsets=((0, 0, 0),)):
    return dict(faces=body.faces, dense=np.stack([body.vertices+p for p in offsets]))


def physical(mesh, identity=0):
    return dict(face_instance_ids=np.full(len(mesh.faces), identity, int),
                convex_envelope_ids=np.array([identity]),
                face_part_ids=np.full(len(mesh.faces), 2, int))


def open_ball():
    ball = trimesh.creation.icosphere(subdivisions=3, radius=.025)
    return trimesh.Trimesh(ball.vertices, ball.faces[1:], process=False)


class FurnitureGeometryTests(unittest.TestCase):
    def test_functional_surface_labels_do_not_split_a_solid(self):
        box = trimesh.creation.box()
        audit = prepare_geometry(box, {"face_part_ids": np.arange(len(box.faces)) % 3})
        self.assertEqual(audit.summary()["counts"], {"original": 1})
        self.assertEqual(audit.records[0]["parts"], ["backrest", "base", "seat"])

    def test_bind_instances_to_both_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            box = trimesh.creation.box()
            box.export(path/"mesh.obj")
            np.savez(path/"parts.npz", **physical(box))
            (path/"asset.json").write_text(json.dumps({"id": "unbound"}))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                read_asset(path/"asset.json")
        with self.assertRaisesRegex(ValueError, "face_instance_ids"):
            prepare_geometry(box, {"face_instance_ids": np.zeros(len(box.faces)-1, int)})

    def test_unique_seams_only_within_one_authored_instance(self):
        box = trimesh.creation.box(extents=[.03, .03, .03])
        soup = trimesh.Trimesh(box.triangles.reshape(-1, 3), np.arange(36).reshape(-1, 3), process=False)
        audit = prepare_geometry(soup, physical(soup))
        self.assertEqual(audit.summary()["counts"], {"stitched": 1})
        self.assertEqual(sorted(audit.records[0]["source_faces"]), list(range(12)))
        self.assertTrue(all(s["open_edges"] for s in audit.records[0]["shells"]))
        self.assertEqual(prepare_geometry(soup).summary()["counts"], {"unknown": 12})

    def test_two_rounded_parts_get_two_certified_hulls(self):
        first = open_ball()
        second = first.copy()
        second.apply_translation([.12, 0, 0])
        mesh = trimesh.util.concatenate([first, second])
        parts = physical(mesh)
        parts["face_instance_ids"][len(first.faces):] = 8
        parts["convex_envelope_ids"] = np.array([0, 8])
        audit = prepare_geometry(mesh, parts, max_scale=1.1)
        self.assertEqual(audit.summary()["counts"], {"hull": 2})
        for r in audit.records:
            self.assertLessEqual(r["proxy_surface_error_upper_m"], PROXY_ERROR)
        body = trimesh.creation.box(extents=[.012]*3)
        report = collision_check(motion(body,[(.06,0,0)]*2), mesh, parts, prepared=audit)
        self.assertEqual(report["status"], "certified_below_limit")
        self.assertEqual(report["geometry"]["separated_frames_by_component"], {0:2, 1:2})

    def test_hull_cannot_supply_a_penetration_witness(self):
        shell = open_ball()
        body = trimesh.creation.box(extents=[.004]*3)
        result = collision_check(motion(body), shell, physical(shell))
        self.assertEqual(result["geometry"]["counts"], {"hull":1})
        self.assertEqual(collision_checks(result), ["collision_depth_undetermined"])
        self.assertEqual(result["maximum_penetration_lower_bound"], 0)
        self.assertIn("proxy_contact_unverified", result["frames"][0]["directions"]["body_to_chair"]["uncertainty_reasons"])

    def test_thin_sheet_separation_requires_a_trusted_extent(self):
        sheet = trimesh.Trimesh([[.1,0,0],[.1,.04,0],[.1,0,.04]], [[0,1,2]], process=False)
        body = trimesh.creation.box(extents=[.01]*3)
        result = collision_check(motion(body), sheet, physical(sheet))
        self.assertEqual(result["geometry"]["counts"], {"bounds_only":1})
        self.assertEqual(result["status"], "certified_below_limit")
        self.assertEqual(collision_check(motion(body), sheet)["status"], "undetermined")
        grouping = {"face_instance_ids":np.zeros(1, int)}
        self.assertEqual(collision_check(motion(body), sheet, grouping)["status"], "undetermined")
        grouping.update(instance_bound_ids=np.array([0]), instance_bounds=np.array([[[.09,0,0],[.11,.04,.04]]]))
        self.assertEqual(collision_check(motion(body), sheet, grouping)["status"], "certified_below_limit")
        grouping["instance_bounds"][0,0,0] = .105
        with self.assertRaisesRegex(ValueError,"contain"):
            prepare_geometry(sheet, grouping)
        nearby = collision_check(motion(body,[(.1,0,0)]), sheet, physical(sheet))
        self.assertEqual(nearby["status"], "undetermined")
        self.assertIsNone(nearby["maximum_penetration_upper_bound"])

    def test_single_hull_cannot_bridge_a_u_shaped_part(self):
        xy = np.array([[-3,0],[3,0],[3,5],[2,5],[2,1],[-2,1],[-2,5],[-3,5]])*.01
        low = np.column_stack((xy,np.zeros(8)))
        caps = [[0,1,4],[0,4,5],[1,2,3],[1,3,4],[0,5,6],[0,6,7]]
        faces = [f[::-1] for f in caps] + [[i+8 for i in f] for f in caps[1:]]
        for i in range(8):
            j=(i+1)%8
            faces.extend([[i,j,j+8],[i,j+8,i+8]])
        mesh = trimesh.Trimesh(np.vstack([low,low+[0,0,.015]]),faces,process=False)
        audit=prepare_geometry(mesh,physical(mesh))
        self.assertEqual(audit.records[0]["mode"], "bounds_only")
        self.assertEqual(audit.records[0]["reason"], "proxy_shape_error")
        body=trimesh.creation.icosphere(subdivisions=1,radius=.004)
        report=collision_check(motion(body,[(0,.035,.007)]),mesh,physical(mesh),prepared=audit)
        self.assertEqual(report["status"], "undetermined")
        self.assertEqual(report["maximum_penetration"], 0)

    def test_unknown_geometry_does_not_hide_later_original_penetration(self):
        seat = trimesh.creation.box(extents=[.07]*3)
        scrap = trimesh.Trimesh([[2,0,0],[2,.01,0],[2,0,.02]],[[0,1,2]],process=False)
        mesh = trimesh.util.concatenate([seat,scrap])
        parts = {"face_part_ids":np.r_[np.zeros(len(seat.faces),int),2]}
        result=collision_check(motion(trimesh.creation.box(extents=[.01]*3),[(1,0,0),(0,0,0),(0,0,0)]),mesh,parts)
        self.assertEqual(collision_checks(result), ["full_motion_penetration"])
        self.assertEqual(result["checked_source_frames"], 2)
        self.assertEqual(result["uncertainty_frame"], 0)
        self.assertIsNone(result["maximum_penetration_upper_bound"])
        witness=result["penetration_witness"]
        self.assertEqual((witness["frame"],witness["direction"],witness["furniture_part"]),(1,"body_to_chair","seat"))

    def test_box_inside_body_is_not_reported_as_spatially_excluded(self):
        sheet = trimesh.Trimesh([[0,0,0],[.001,0,0],[0,.001,0]], [[0,1,2]], process=False)
        body = trimesh.creation.box(extents=[.1]*3)
        result = collision_check(motion(body), sheet, physical(sheet))
        self.assertEqual(result["geometry"]["separated_frames_by_component"], {})
        self.assertEqual(collision_checks(result), ["full_motion_penetration"])
        self.assertEqual(result["penetration_witness"]["direction"], "chair_to_body")

    def test_nonmanifold_edges_winding_and_volume_are_auditable(self):
        mesh = trimesh.creation.box()
        mesh = trimesh.Trimesh(mesh.vertices, np.vstack([mesh.faces,mesh.faces[0]]),process=False)
        record = prepare_geometry(mesh,physical(mesh)).records[0]
        self.assertEqual(record["mode"], "bounds_only")
        self.assertEqual(len(record["shells"][0]["nonmanifold_edges"]),3)
        self.assertIn("degenerate_volume", record["shells"][0])
        box = trimesh.creation.box()
        faces = box.faces.copy()
        faces[0] = faces[0, ::-1]
        issue=prepare_geometry(trimesh.Trimesh(box.vertices,faces,process=False)).records[0]
        self.assertFalse(issue["shells"][0]["winding_consistent"])
