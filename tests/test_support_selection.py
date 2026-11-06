"""Behavioral controls for short clips, transitions and evidence weighting."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chair_support import engine
from chair_support.selection import (evidence_weights, sample_support_frames,
                                     select_support, support_intervals, _maximum_overlap)


class SelectionTests(unittest.TestCase):
    def envelopes(self, frames, heights):
        samples = []
        for f, z in zip(frames, heights):
            points = np.array([[-.1, -.1, z], [.1, -.1, z], [-.1, .1, z], [.1, .1, z]])
            samples.append((int(f), np.arange(4), points, np.ones(4)))
        return {r: samples for r in engine.REGIONS}

    def choose(self, heights, weights=None):
        frames = np.arange(len(heights))
        weights = np.ones(len(frames)) if weights is None else np.asarray(weights)
        return select_support(frames, weights, self.envelopes(frames, heights),
                              np.zeros((len(frames), 3)), np.tile([0., 1., 0.], (len(frames), 1)),
                              {r: .015 if "gluteus" in r else .020 for r in engine.REGIONS}, .005)

    def test_single_and_eleven_frames_need_no_minimum_duration(self):
        for count in (1, 11):
            frames, proxy = self.choose(np.full(count, .5))
            np.testing.assert_array_equal(frames, np.arange(count))
            self.assertTrue(proxy["proxy_feasible"])

    def test_velocity_reduces_weight_without_removing_compatible_frames(self):
        weights = evidence_weights(np.full(4, 8), [0., .1, .5, 2.], [0., .04, .1, .5])
        self.assertTrue(np.all(weights > 0))
        self.assertTrue(np.all(np.diff(weights) < 0))
        frames, _ = self.choose(np.full(4, .5), weights)
        np.testing.assert_array_equal(frames, np.arange(4))

    def test_downward_and_upward_transitions_do_not_become_support(self):
        heights = [.8, .7, .6, .5, .5, .5, .6, .7, .8]
        frames, proxy = self.choose(heights, [.01, .03, .1, .9, 1., .9, .1, .03, .01])
        np.testing.assert_array_equal(frames, [3, 4, 5])
        self.assertAlmostEqual(proxy["slope_dz_dforward"], 0.)

    def test_disjoint_support_intervals_are_kept(self):
        frames, _ = self.choose([.5, .5, .8, .5, .5])
        self.assertEqual(support_intervals(frames), [(0, 2), (3, 5)])

    def test_planar_failure_still_keeps_strongest_curved_candidate(self):
        frames = np.array([0, 1])
        anchors = self.envelopes(frames, [.5, .5])
        for region, samples in anchors.items():
            for _, _, points, _ in samples:
                points[0, 2] -= .2
        selected, proxy = select_support(frames, np.array([.1, .9]), anchors, np.zeros((2, 3)),
            np.tile([0., 1., 0.], (2, 1)), {r: .015 for r in anchors}, .005)
        np.testing.assert_array_equal(selected, [1])
        self.assertFalse(proxy["proxy_feasible"])

    def test_interval_sweep_matches_exhaustive_closed_interval_search(self):
        rng = np.random.RandomState(7)
        for _ in range(30):
            lo = rng.randint(0, 8, 20).astype(float)
            hi = lo + rng.randint(-1, 4, 20)
            weights = rng.uniform(.01, 1, 20)
            mask = _maximum_overlap(lo, hi, weights)
            best = max((round(float(weights[(lo <= h) & (hi >= h)].sum()), 12),
                        int(((lo <= h) & (hi >= h)).sum())) for h in np.unique(np.r_[lo, hi]))
            self.assertEqual((round(float(weights[mask].sum()), 12), int(mask.sum())), best)

    def test_time_sampling_preserves_short_run_endpoints_and_strongest_frame(self):
        frames = np.r_[np.arange(1, 12), [80], np.arange(100, 103)]
        weights = np.ones(103); weights[6] = 2
        sampled = sample_support_frames(frames, weights, 120)
        self.assertTrue({1, 6, 11, 80, 100, 102}.issubset(set(sampled)))
        self.assertTrue(set(sampled).issubset(set(frames)))


class MotionEvidenceTests(unittest.TestCase):
    def context(self, count, *, chair=True, sway=False, invalid=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            boxes = []
            for x, y in ((-.12, -.1), (.12, -.1), (-.12, .1), (.12, .1)):
                patch_vertices = np.array([[-.04,-.04,0],[.04,-.04,0],[.04,.04,0],[-.04,.04,0]])
                boxes.append(trimesh.Trimesh(patch_vertices+[x,y,.5],[[0,2,1],[0,3,2]],process=False))
            body = trimesh.util.concatenate(boxes)
            dense = np.repeat(body.vertices[None], count, axis=0).astype(np.float32)
            names = ["Hips", "LeftLeg", "LeftShin", "LeftFoot", "RightLeg", "RightShin", "RightFoot"]
            joint = np.array([[0, 0, .62], [-.12, 0, .62], [-.12, .35, .62], [-.12, .35, .22],
                              [.12, 0, .62], [.12, .35, .62], [.12, .35, .22]], np.float32)
            joints = np.repeat(joint[None], count, axis=0)
            if sway:
                offset = .02*np.sin(np.arange(count)*2*np.pi/24)
                joints[:, :, 0] += offset[:, None]; dense[:, :, 0] += offset[:, None]
            world = np.zeros((count, 655, 3), np.float32)
            labels = np.zeros((len(range(0, count, 8)), 655, 1), np.int64)
            labels[:, :8, 0] = 3 if chair else 0; labels[:, 8, 0] = 2
            for key, array in dict(dense=dense, joints=joints, faces=body.faces, world=world,
                                   labels=labels, indices=np.arange(0, count, 8)).items():
                np.save(root/(key+".npy"), array)
            (root/"names.json").write_text(json.dumps(names))
            (root/"manifest.json").write_text(json.dumps(dict(source_fps=120, fitting_data=dict(
                world="dense.npy", joints="joints.npy", faces="faces.npy", joint_names="names.json"))))
            regions = {r: np.arange(i*4, (i+1)*4) for i, r in enumerate(engine.REGIONS)}
            if invalid:
                regions[engine.REGIONS[0]] = np.array([], int)
            with patch.object(engine, "body_regions", return_value=regions):
                ctx = engine.motion_context(root/"manifest.json", root/"world.npy", root/"labels.npy",
                                            root/"indices.npy", root/"output")
            with np.load(root/"output/phase_evidence.npz") as saved:
                evidence = dict(saved)
            phase = ctx["phase"]
            for key in ("dense", "joints", "world"):
                ctx[key]._mmap.close()
            return phase, evidence

    def test_short_clip_succeeds_with_warning_and_no_hidden_gradient_minimum(self):
        for count in (1, 11):
            phase, evidence = self.context(count)
            self.assertEqual(phase["support_frames"], count)
            self.assertIn("short_support_evidence", phase["warnings"])
            self.assertTrue(evidence["support_selected"].all())

    def test_swaying_seated_motion_exceeds_old_speed_but_is_selected(self):
        phase, evidence = self.context(120, sway=True)
        self.assertGreater(evidence["pelvis_speed"].max(), .08)
        self.assertEqual(phase["support_frames"], 120)
        self.assertNotIn("short_support_evidence", phase["warnings"])

    def test_no_contact_and_invalid_envelope_keep_explicit_reasons(self):
        phase, evidence = self.context(11, chair=False)
        self.assertEqual(phase["support_frames"], 0)
        self.assertIn("insufficient_chair_contacts", phase["selection_reasons"][0])
        phase, evidence = self.context(11, invalid=True)
        self.assertEqual(phase["support_frames"], 0)
        self.assertIn("insufficient_downward_samples:left_gluteus", phase["selection_reasons"][0])


if __name__ == "__main__":
    unittest.main()
