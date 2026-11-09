"""Source-frame, consistency and geometric checks for multi-seat assembly."""

from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chair_support.collision import collision_check, collision_checks
from chair_support import multi_scene as scene
from chair_support.runtime import ExecutionOptions, execution_scope
from chair_support.policy import FitPolicy


def box(center, size):
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation(center)
    return mesh


def person(positions=((0, 0, 0.35),), size=(0.03, 0.03, 0.05)):
    body = box([0, 0, 0], size)
    return dict(dense=np.asarray([body.vertices + np.array(p) for p in positions]), faces=body.faces,
                policy=FitPolicy(support_validation=True, penetration_validation=True))


def set_up_candidate(ctx, root, name, center, loss=1.0, size=(0.1, 0.1, 0.1), angle=0.0):
    """Actual watertight assets and verifier outputs, with face annotation in source coordinates."""
    local = box([0, 0, 0], size)
    src = root / "assets" / name
    out = root / "candidates" / name
    for path in (src, out):
        path.mkdir(parents=True)
    local.export(src / "mesh.obj")
    # Match the fitter's source OBJ precision before transforming and certifying.
    local = trimesh.load(src / "mesh.obj", process=False)
    rotation = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])[:3, :3]
    transform = dict(rotation=rotation.tolist(), translation=list(center), scale=1.0)
    pose = trimesh.Trimesh(local.vertices @ rotation.T + np.asarray(center), local.faces, process=False)
    pose.export(out / "mesh.obj")
    parts = dict(
        seat_face_indices=np.where(local.face_normals[:, 2] > 0.9)[0],
        backrest_face_indices=np.where(local.face_normals[:, 1] > 0.9)[0],
    )
    np.savez(src / "parts.npz", **parts)
    np.savez(out / "parts.npz", **parts)
    meta = dict(id=name, source="cad", mesh="mesh.obj", parts="parts.npz")
    scene.write_atomic(src / "asset.json", meta)
    verified = collision_check(ctx, pose, parts)
    scene.write_atomic(out / "collision.json", verified)
    return dict(
        id=name,
        mesh=(out / "mesh.obj").relative_to(root).as_posix(),
        parts=(out / "parts.npz").relative_to(root).as_posix(),
        loss=loss,
        transform=transform,
        qualified=verified["status"] == "certified_below_limit",
        source_asset=str(src / "asset.json"),
        source_geometry_sha256=scene.sha256(src / "mesh.obj"),
        rejection_reasons=collision_checks(verified),
    )


class MultiSceneTests(unittest.TestCase):
    def setUp(self):
        self.work = TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.output = self.root / "assembly"
        self.ctx = person(positions=[(0, 0, 0.35), (0.3, 0, 0.35), (0, 0, 0.35)])
        self.dirs = {}

    def add(self, identity, specs):
        root = self.root / identity
        pool = [set_up_candidate(self.ctx, root, *args) for args in specs]
        scene.write_atomic(root / "fit_result.json", dict(status="success", candidates=pool))
        self.dirs[identity] = root
        scene.seal_instance(self.ctx, root, "motion+events", identity)
        return pool

    def assemble(self, budget=64):
        return scene.assemble_scene(self.ctx, self.dirs, self.output, budget, "motion+events")

    def rejected_pool(self, instance, reasons_by_id):
        fit = self.root / instance
        candidates = [dict(id=k, qualified=False, rejection_reasons=v) for k, v in reasons_by_id.items()]
        scene.write_atomic(
            fit / "fit_result.json",
            dict(
                status="no_valid_candidate",
                candidates=candidates,
                rejection_reasons=["no_candidate_passed_geometry_acceptance"]
                if candidates
                else ["no_support_evidence"],
            ),
        )
        self.dirs[instance] = fit
        scene.seal_instance(self.ctx, fit, "motion+events", instance)

    def test_entire_required_pool_rejected_even_with_other_unknowns(self):
        reasons = {
            "wrong_height": ["left_gluteus:insufficient_support", "visit_support_unverified"],
            "intrusion": ["full_motion_penetration", "collision_depth_undetermined"],
        }
        self.rejected_pool("A", reasons)
        self.rejected_pool("B", {"bad_mesh": ["support_geometry_unverified: missing seat surface"]})
        result = self.assemble()
        self.assertEqual(result["status"], "no_valid_scene")
        a, b = (result["rejected_instances"][k] for k in ("A", "B"))
        self.assertEqual(a["candidates"], reasons)
        self.assertEqual(a["candidate_states"], dict.fromkeys(reasons, "measured_rejection"))
        self.assertEqual((a["status"], b["status"]), ("no_valid_candidate", "undetermined"))

    def test_unknown_alternative_keeps_possible_combination_open(self):
        reasons = {"wrong_heading": ["chair_heading_mismatch"], "timeout": ["collision_depth_undetermined"]}
        self.rejected_pool("A", reasons)
        self.rejected_pool("B", {})
        result = self.assemble()
        self.assertEqual(result["status"], "undetermined")
        self.assertEqual(
            result["rejected_instances"]["A"]["candidate_states"],
            {"wrong_heading": "measured_rejection", "timeout": "undetermined"},
        )
        self.assertEqual(result["rejected_instances"]["B"]["reasons"], ["no_support_evidence"])

    def test_empty_scene_and_two_disjoint_seats_with_revisit(self):
        empty = self.assemble()
        self.assertEqual(empty["selected"], [])
        self.assertEqual(empty["collision"]["method"], "empty_scene")
        self.add("A", [("one", (0.0, 0.0, 0.0))])
        self.add("B", [("two", (0.3, 0.0, 0.0))])
        found = self.assemble()
        self.assertEqual(found["status"], "success")
        self.assertEqual([r["instance_id"] for r in found["selected"]], ["A", "B"])
        self.assertEqual(found["collision"]["checked_source_frames"], 3)
        self.assertEqual(found["collision"]["method"], "disjoint_instance_certificates")
        self.assertEqual([r["frame"] for r in found["collision"]["frames"]], [0, 1, 2])
        with patch.object(scene, "load_candidate", side_effect=AssertionError("valid checkpoint unused")):
            resumed = self.assemble()
            self.assertEqual(resumed["collision"], found["collision"])
            self.assertTrue(resumed["attempts"][0]["checkpoint_reused"])
        self.ctx["dense"][1, :, 0] += 0.2
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.assemble()

    def test_best_pair_conflicts_then_next_solution_and_budget(self):
        self.add("A", [("a", (0.0, 0.0, 0.0))])
        self.add("B", [("conflict", (0.025, 0.0, 0.0), 1.0), ("alternate", (0.21, 0.0, 0.0), 1.1)])
        limited = self.assemble(1)
        self.assertEqual(limited["status"], "search_budget_exhausted")
        self.assertEqual(limited["attempts"][0]["status"], "furniture_overlap")
        self.assertGreater(
            limited["attempts"][0]["furniture_pairs"][0]["witness"]["interior_depth_lower_m"], 0
        )
        full = self.assemble()
        self.assertEqual(full["selected"][1]["candidate_id"], "alternate")
        self.assertEqual(len(full["attempts"]), 2)

    def test_touching_is_allowed_and_forces_union_audit(self):
        self.add("A", [("al", (-0.05, 0.0, 0.0))])
        self.add("B", [("br", (0.05, 0.0, 0.0))])
        result = self.assemble()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["attempts"][0]["furniture_pairs"][0]["status"], "touching_or_close")
        self.assertNotEqual(result["collision"]["method"], "disjoint_instance_certificates")

    def test_individual_maximum_cannot_certify_touching_union(self):
        # The small body lies next to the shared internal wall of the two boxes.
        # Relative to either box alone every point is <5 mm from its exterior;
        # the nearest exterior of the actual union is >40 mm away.
        self.ctx = person(positions=[(0.0025, 0.0, 0.0)], size=(0.002, 0.002, 0.002))
        for name, center in (("A", (-0.05, 0.0, 0.0)), ("B", (0.05, 0.0, 0.0))):
            candidates = self.add(name, [("seat", center)])
            self.assertTrue(candidates[0]["qualified"])
        result = self.assemble()
        self.assertNotEqual(result["status"], "success")
        self.assertIn(
            result["attempts"][0]["status"], ("collision_depth_undetermined", "measured_penetration")
        )

    def test_overlapping_aabbs_do_not_reject_diagonal_separation(self):
        half_angle = np.pi / 4
        self.add("A", [("seat", (0.0, 0.0, 0.0), 1.0, (0.30, 0.03, 0.04), half_angle)])
        self.add("B", [("seat", (-0.04, 0.04, 0.0), 1.0, (0.30, 0.03, 0.04), half_angle)])
        result = self.assemble()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["attempts"][0]["furniture_pairs"][0]["method"], "outer_component_pair_tests")

    def test_transition_collision_against_idle_seat_is_preserved(self):
        self.ctx = person(positions=[(0.0, 0.0, 0.35), (0.3, 0.0, 0.0), (0.3, 0.0, 0.35)])
        self.add("A", [("accepted", (0.0, 0.0, 0.0))])
        rejected = self.add("B", [("idle", (0.3, 0.0, 0.0))])[0]
        self.assertFalse(rejected["qualified"])
        collision = scene.read(self.dirs["B"] / "candidates/idle/collision.json")
        self.assertEqual(collision["penetration_witness"]["frame"], 1)
        self.assertEqual(self.assemble()["status"], "no_valid_scene")

    def test_tampered_single_chair_certificate_is_not_reused(self):
        self.add("A", [("a", (0.0, 0.0, 0.0))])
        self.assemble()
        path = self.dirs["A"] / "candidates/a/collision.json"
        cert = scene.read(path)
        cert["frames"].pop()
        scene.write_atomic(path, cert)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.assemble()

    def test_shared_proof_reused_across_new_run_with_relocated_input_assets(self):
        pool = self.add("A", [("a", (0.0, 0.0, 0.0))])
        shared = self.root / "shared"
        first = scene.assemble_scene(
            self.ctx, self.dirs, self.output, input_key="motion+events", cache_dir=shared
        )
        self.assertFalse(first["attempts"][0]["checkpoint_reused"])
        moved = self.root / "new_run" / "fitting"
        fit = moved / "one"
        shutil.copytree(self.dirs["A"], fit)
        # Different absolute paths match the way two fresh run-input/refit jobs write fit_result.json.
        pool[0]["source_asset"] = str(fit / "assets/a/asset.json")
        scene.write_atomic(fit / "fit_result.json", dict(status="success", candidates=pool))
        scene.seal_instance(self.ctx, fit, "motion+events", "A")
        with patch.object(scene, "load_candidate", side_effect=AssertionError("shared checkpoint unused")):
            second = scene.assemble_scene(
                self.ctx, {"A": fit}, moved, input_key="motion+events", cache_dir=shared
            )
        self.assertTrue(second["attempts"][0]["checkpoint_reused"])
        self.assertTrue((moved / second["attempts"][0]["checkpoint"]).is_file())
        self.assertEqual((moved / second["selected"][0]["mesh"]).resolve(), (fit / pool[0]["mesh"]).resolve())
        self.assertEqual(first["collision"], second["collision"])

    def test_scene_cache_binds_query_implementation_and_execution(self):
        self.add("A", [("a", (0.0, 0.0, 0.0))])
        first = self.assemble()
        self.assertEqual(first["strict_query_backend"], "open3d")
        self.assertTrue(self.assemble()["attempts"][0]["checkpoint_reused"])
        original_hash = scene.sha256
        for name in ("query_backend.py", "runtime.py"):
            with self.subTest(module=name):
                def changed_hash(path):
                    return "changed-" + original_hash(path) if Path(path).name == name else original_hash(path)
                with patch.object(scene, "sha256", side_effect=changed_hash):
                    changed = self.assemble()
                self.assertFalse(changed["attempts"][0]["checkpoint_reused"])
                self.assertNotEqual(changed["attempts"][0]["checkpoint"], first["attempts"][0]["checkpoint"])
        with execution_scope(ExecutionOptions(query_threads=1)):
            changed = self.assemble()
        self.assertFalse(changed["attempts"][0]["checkpoint_reused"])
        self.assertEqual(changed["execution"]["actual_query_threads"], 1)
        self.assertEqual(changed["collision"], first["collision"])

    def test_overlapping_proxy_is_uncertain_not_measured_furniture_intersection(self):
        outer = box([0, 0, 0], [0.5, 0.5, 0.5])
        block = box([0.1, 0.1, 0], [0.04, 0.04, 0.04])

        def piece(mesh, mode):
            return dict(
                vertices=mesh.vertices,
                faces=mesh.faces,
                bounds=mesh.bounds,
                mode=mode,
                id=0,
                parts=["seat"],
                instance=0,
            )

        status, witness = scene.piece_pair(piece(outer, "hull"), piece(block, "original"))
        self.assertEqual(status, "undetermined")
        self.assertIsNone(witness)


if __name__ == "__main__":
    unittest.main()
