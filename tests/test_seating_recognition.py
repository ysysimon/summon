"""Regression tests for time-preserving localization and the independent-chair hypothesis."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from chair_support import recognition as rec


def visit(x, height=0.45, forward=(0.0, 1.0)):
    return dict(
        anchor_xy_m=[x, 0.0],
        height_m=height,
        forward_xy=list(forward),
        footprint_xy_m=[[x - 0.08, 0], [x + 0.08, 0], [x - 0.08, 0.12], [x + 0.08, 0.12]],
    )


def synthetic_posture(sit, x, contact=True, fps=30):
    """Synthetic kinematics; provide four measured support patches per requested source frame."""
    sit = np.asarray(sit, float)
    x = np.broadcast_to(np.asarray(x, float), sit.shape)
    n = len(sit)
    names = ["Hips", "LeftLeg", "RightLeg", "LeftShin", "RightShin", "LeftFoot", "RightFoot"]
    joints = np.zeros((n, len(names), 3))
    joints[:, 0, 0] = x
    joints[:, 0, 2] = 1.0 - 0.38 * sit
    for side, offset in ((1, -0.09), (2, 0.09)):
        hip = joints[:, 0] + [offset, 0, 0]
        joints[:, side] = hip
        angle = np.pi / 2 * sit
        joints[:, side + 2] = hip + np.column_stack(
            [np.zeros(n), 0.38 * np.sin(angle), -0.38 * np.cos(angle)]
        )
        joints[:, side + 4] = joints[:, side + 2] + [0, 0, -0.40]
    indices = np.arange(0, n, 2)
    labels = np.zeros((len(indices), 655), int)
    labelled = np.broadcast_to(contact, sit.shape).astype(bool)
    labels[(sit[indices] > 0.5) & labelled[indices], :12] = 4

    def samples(ctx, regions, frames):
        tracks = {r: [] for r in rec.REGIONS}
        for f in frames:
            for i, key in enumerate(rec.REGIONS):
                xy = np.array(
                    [x[f] + (0.08 if i % 2 else -0.08), 0.12 if i >= 2 else 0.0, joints[f, 0, 2] - 0.15]
                )
                pts = xy + np.array([[-0.01, 0, 0], [0, 0.01, 0], [0.01, 0, 0]])
                tracks[key].append((f, np.arange(3), pts, np.ones(3)))
        return tracks, {}

    with (
        patch.object(rec, "body_regions", return_value={}),
        patch.object(rec, "collect_support_samples", side_effect=samples),
    ):
        visits, lacking = rec._visits(None, None, joints, names, labels, indices, fps)
    instances, ambiguities = rec.assign_locations(visits)
    return visits, instances, ambiguities, lacking


class RecognitionTests(unittest.TestCase):
    def test_return_to_first_location_and_nonoverlapping_source_intervals(self):
        # Both sitting when the clip starts and sitting when the clip ends.
        sit = np.r_[
            np.ones(22),
            np.linspace(1, 0, 8),
            np.zeros(12),
            np.linspace(0, 1, 8),
            np.ones(22),
            np.linspace(1, 0, 8),
            np.zeros(12),
            np.linspace(0, 1, 8),
            np.ones(22),
        ]
        x = np.interp(np.arange(len(sit)), [0, 31, 41, 79, 91, 121], [0, 0, 1.36, 1.36, 0, 0])
        events, locations, ambiguities, lacking = synthetic_posture(sit, x)
        self.assertEqual(len(locations), 2)
        self.assertEqual([e["instance_id"] for e in events], ["seat-01", "seat-02", "seat-01"])
        self.assertEqual((events[0]["start"], events[-1]["end"]), (0, len(sit)))
        self.assertFalse(ambiguities or lacking)
        last = 0
        for e in events:
            self.assertLessEqual(last, e["start"])
            last = e["end"]
            for a, b in e["support"]:
                self.assertTrue(e["start"] <= a < b <= e["end"])

    def test_same_location_two_visits(self):
        sit = np.r_[np.ones(24), np.linspace(1, 0, 8), np.zeros(12), np.linspace(0, 1, 8), np.ones(24)]
        e, loc, _, _ = synthetic_posture(sit, 0)
        self.assertEqual(len(e), 2)
        self.assertEqual(len(loc), 1)
        self.assertEqual(loc[0]["visit_indices"], [0, 1])

    def test_contact_dropouts_and_an_unobserved_second_visit(self):
        sit = np.ones(90)
        for hole, requested_length in ((slice(24, 26), 1), (slice(32, 56), 2)):
            visible = np.ones(90, bool)
            visible[hole] = False
            events, locations, _, _ = synthetic_posture(sit, 0, visible)
            self.assertEqual(len(locations), 1)
            self.assertEqual(len(events[0]["support"]), requested_length)
        sit = np.r_[np.ones(24), np.linspace(1, 0, 8), np.zeros(12), np.linspace(0, 1, 8), np.ones(24)]
        visible = np.arange(len(sit)) < 36
        events, _, _, missing = synthetic_posture(sit, 0, visible)
        self.assertEqual(len(events), 1)
        self.assertEqual(missing[-1][1], len(sit))
        self.assertGreater(missing[-1][0], events[0]["end"])

    def test_input_grid_and_world_are_checked_for_equal_length_stale_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = {
                "world": "dense.npy",
                "joints": "joints.npy",
                "faces": "faces.npy",
                "joint_names": "names.json",
            }
            np.save(root / fields["world"], np.zeros((4, 10475, 3), np.float32))
            np.save(root / fields["joints"], np.zeros((4, 1, 3), np.float32))
            np.save(root / fields["faces"], np.array([[0, 1, 2]]))
            (root / fields["joint_names"]).write_text('["Hips"]')
            fit = {
                **fields,
                "sha256": {k: rec.sha256(root / p) for k, p in fields.items()},
                "units": "m",
                "up_axis": "Z",
            }
            data = np.zeros((4, 655, 3), np.float32)
            np.save(root / "producer.npy", data)
            np.save(root / "changed.npy", data + 0.001)
            np.save(root / "labels.npy", np.zeros((2, 655, 1), int))
            np.save(root / "indices.npy", np.array([0, 1]))
            m = {"fitting_data": fit, "world": "producer.npy", "source_fps": 30, "prediction_stride": 2}
            path = root / "manifest.json"
            path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, "0.1 mm"):
                rec.recognize(path, root / "changed.npy", root / "labels.npy", root / "indices.npy")
            with self.assertRaisesRegex(ValueError, "source-frame grid"):
                rec.recognize(path, root / "producer.npy", root / "labels.npy", root / "indices.npy")

    def test_low_horizontal_transfer_does_not_seed_seats_along_a_path(self):
        events, locations, _, lacking = synthetic_posture(np.ones(180), np.arange(180) * 0.2 / 30)
        self.assertFalse(events or locations)
        self.assertTrue(lacking)
        x = np.r_[np.zeros(24), np.linspace(0, 0.62, 94), np.full(24, 0.62)]
        events, locations, _, _ = synthetic_posture(np.ones(len(x)), x)
        self.assertEqual(len(events), 2)
        self.assertEqual(len(locations), 2)

    def test_first_rest_does_not_verify_unlabelled_rest_after_low_transfer(self):
        x = np.r_[np.zeros(30), np.linspace(0, 0.62, 94), np.full(30, 0.62)]
        observed = np.arange(len(x)) < 30
        events, locations, _, unknown = synthetic_posture(np.ones(len(x)), x, observed)
        self.assertEqual(len(events), 1)
        self.assertEqual(len(locations), 1)
        self.assertEqual(len(unknown), 1)
        lo, hi = unknown[0]
        self.assertGreaterEqual(x[lo], 0.60)
        self.assertEqual(hi, len(x))
        self.assertGreaterEqual(hi - lo, 26)
        self.assertFalse(synthetic_posture(np.ones(len(x)), x)[-1])

    def test_unobserved_seated_posture_distinct_from_empty_walk(self):
        self.assertEqual(synthetic_posture(np.zeros(48), np.linspace(0, 1, 48))[:2], ([], []))
        self.assertFalse(synthetic_posture(np.zeros(48), 0)[-1])
        self.assertTrue(synthetic_posture(np.ones(48), 0, contact=False)[-1])

    def test_chain_does_not_merge_distant_seats(self):
        events = [visit(v) for v in [0, 0.17, 0.34, 0.02]]
        locations, _ = rec.assign_locations(events)
        self.assertEqual(len(locations), 2)
        self.assertEqual(events[0]["instance_id"], events[3]["instance_id"])
        self.assertNotEqual(events[0]["instance_id"], events[2]["instance_id"])

    def test_shared_surface_is_a_conditional_flag_for_both_nearby_offsets(self):
        for offset in (0.60, 0.78):
            events = [visit(0), visit(offset)]
            instances, ambiguous = rec.assign_locations(events)
            self.assertEqual(len(instances), 2)
            self.assertEqual(ambiguous[0]["instance_ids"], ["seat-01", "seat-02"])
        for event in (visit(0, height=0.70), visit(0, forward=(0, -1))):
            instances, ambiguous = rec.assign_locations([visit(0), event])
            self.assertEqual(len(instances), 2)
            self.assertFalse(ambiguous)

    def test_original_dbscan_reports_separate_space_and_time(self):
        world = np.zeros((12, 655, 3))
        xx, yy = np.meshgrid(np.arange(8) * 0.041, np.arange(8) * 0.041)
        verts = np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size) + 0.45]
        indices = np.arange(0, 12, 2)
        for i in range(len(world)):
            world[i, :64] = verts + [1.2 if 4 <= i < 8 else 0, 0, 0]
        labels = np.zeros((6, 655), int)
        labels[:, :64] = 3
        r = rec.original_clusters(world, labels, indices)
        self.assertEqual(r["spatial_cluster_count"], 2)
        self.assertEqual(len(r["events"]), 3)
        self.assertEqual(r["events"][0]["instance_id"], r["events"][-1]["instance_id"])


if __name__ == "__main__":
    unittest.main()
