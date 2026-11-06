import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from chair_support.engine import REGIONS, contact_roles
from chair_support.multi_fit import (
    CandidateCache,
    evaluate_visits,
    instance_contacts,
    retain_return_visits,
    visit_masks,
)


class InstanceFittingTests(unittest.TestCase):
    def test_noncontiguous_visits_keep_source_grid(self):
        events = [
            dict(start=1, end=6, support=[[3, 5]]),
            dict(start=10, end=16, support=[[11, 13], [14, 16]]),
        ]
        assignment, support = visit_masks(events, 19)
        self.assertEqual(assignment.tolist(), [-1] + [0] * 5 + [-1] * 4 + [1] * 6 + [-1] * 3)
        self.assertEqual(np.flatnonzero(support >= 0).tolist(), [3, 4, 11, 12, 14, 15])
        with self.assertRaises(ValueError):
            visit_masks([events[0], dict(start=5, end=7, support=[])], 19)
        with self.assertRaises(ValueError):
            visit_masks([dict(start=1, end=6, support=[[5, 7]])], 19)

    def test_common_proxy_cannot_drop_return(self):
        events = [dict(start=0, end=1, support=[[0, 1]]), dict(start=2, end=3, support=[[2, 3]])]
        _, ids = visit_masks(events, 3)
        frames = np.array([0, 2])
        points = np.array([[0.0, 0.0, 0.4], [0.0, 0.1, 0.4], [0.1, 0.0, 0.4]])
        anchors = {key: [(f, np.arange(3), points, np.ones(3)) for f in frames] for key in REGIONS}
        selected, visits = retain_return_visits(
            events,
            frames[:1],
            frames,
            ids,
            anchors,
            np.ones(3),
            np.zeros((3, 3)),
            np.tile(np.array([0.0, 1.0, 0.0]), (3, 1)),
            dict.fromkeys(REGIONS, 0.015),
            0.005,
        )
        self.assertEqual(selected.tolist(), [0, 2])
        self.assertEqual([v["visit_local_proxy"] for v in visits], [False, True])
        valid = dict(coverage=1.0, gap_p20=0.003, maximum_seat_intrusion=0.001)
        measurements = {key: [dict(valid, frame=0), dict(valid, frame=2)] for key in REGIONS}
        measurements["left_gluteus"][1]["gap_p20"] = 0.016
        states = evaluate_visits(visits, measurements)
        self.assertTrue(states[0]["verified"])
        self.assertEqual(states[1]["rejection_reasons"], ["left_gluteus:insufficient_support"])

    def test_missing_visit_is_explicit(self):
        events = [dict(start=0, end=1, support=[])]
        _, ids = visit_masks(events, 1)
        frames = np.empty(0, int)
        selected, visits = retain_return_visits(events, frames, frames, ids, {}, None, None, None, {}, 0.005)
        self.assertEqual(len(selected), 0)
        self.assertEqual(visits[0]["status"], "support_unverified")

    def test_other_instances_contacts_excluded(self):
        ctx = {
            "indices": np.array([0, 1, 2]),
            "labels": np.array([[3], [4], [3]]),
            "interaction_ids": np.array([-1, 0, -1]),
            "world": np.zeros((3, 1, 3)),
            "dense": np.tile(np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]), (3, 1, 1)),
            "faces": np.array([[0, 1, 2]]),
            "forward": np.tile([0.0, 1.0, 0.0], (3, 1)),
            "j": {side + "Shin": np.zeros((3, 3)) for side in ("Left", "Right")},
        }
        regions = dict(back=[0], right_distal_thigh=[0], left_distal_thigh=[0])
        ctx["instance_contacts"] = instance_contacts(
            ctx["world"], ctx["labels"], ctx["indices"], ctx.pop("interaction_ids"), "A", None
        )
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            contact_roles(ctx, regions, folder)
            self.assertEqual(
                json.loads((folder / "contact_roles.json").read_text())["records"], [[1, 0, "seat"]]
            )

    def test_reaching_hand_on_idle_neighbor_and_boundary_points_are_excluded(self):
        # Hips on A, hand at B, ambiguous elbow on the bisector, chair-class
        # return to A. A changing seat class alone cannot reassign a vertex.
        world = np.tile(
            np.array([[[0.01, 0, 0.45], [0.6, 0, 0.6], [0.3, 0, 0.55], [0.295, 0, 0.65]]]), (3, 1, 1)
        )
        labels = np.array([[3, 3, 3, 4], [3, 3, 3, 4], [4, 3, 4, 4]])
        selected = instance_contacts(
            world, labels, np.arange(3), np.array([0, -1, 1]), "A", {"A": [0.0, 0.0], "B": [0.6, 0.0]}
        )
        self.assertEqual(
            selected.tolist(), [[True, False, False, False], [False] * 4, [True, False, False, False]]
        )

    def test_cache_requires_same_input_asset_and_intact_result(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            meta = base / "asset.json"
            meta.write_text(json.dumps({"id": "item"}))
            (base / "mesh.obj").write_bytes(b"original")
            (base / "parts.npz").write_bytes(b"parts")
            cache = CandidateCache(base / "cache", "input", "solver", [{"start": 0}], {})
            missing, location, identity = cache.lookup(meta, base / "out")
            self.assertIsNone(missing)
            work = base / "worker"
            work.mkdir()
            (work / "result.json").write_text(json.dumps({"id": "item", "qualified": False}))
            cache.publish(location, work)
            record, _, _ = cache.lookup(meta, base / "another")
            self.assertFalse(record["qualified"])
            self.assertEqual(record["source_asset"], str(meta))
            other = CandidateCache(base / "cache", "different", "solver", [{"start": 0}], {})
            self.assertIsNone(other.lookup(meta, base / "unused")[0])
            (base / "mesh.obj").write_bytes(b"other")
            self.assertIsNone(cache.lookup(meta, base / "unused")[0])
            (base / "mesh.obj").write_bytes(b"original")
            (location / "candidate/result.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Corrupted"):
                cache.lookup(meta, base / "invalid")


if __name__ == "__main__":
    unittest.main()
