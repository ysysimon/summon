import argparse
import os
from itertools import product
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chair_support.policy import FitPolicy, add_arguments
from chair_support.multi_fit import CandidateCache
from chair_support import engine, multi_scene
from chair_support.collision import collision_check, collision_checks
import test_supported_chair as support_fixture
import test_multi_scene as scene_fixture


class PolicyTests(unittest.TestCase):
    def test_default_fit_optimizes_without_final_audits_or_certificates(self):
        fixture=support_fixture.SupportTests()
        from chair_support import runtime
        profile=runtime.Profile(True)
        with runtime.execution_scope(runtime.ExecutionOptions(profile_fit=True),profile), \
             patch.object(engine,'collision_check',side_effect=AssertionError('final audit')), \
             patch.object(engine,'prepare_geometry',side_effect=AssertionError('audit geometry')), \
             patch.object(engine,'batched_support_metrics',side_effect=AssertionError('support audit')), \
             patch.object(engine,'support_query',wraps=engine.support_query) as support, \
             patch.object(engine,'close_body_topology',wraps=engine.close_body_topology) as body, \
             patch.object(engine,'write_json',wraps=engine.write_json) as write:
            result=fixture.fit_control(policy=None)
        self.assertTrue(result['selection_eligible'],result['rejection_reasons'])
        self.assertFalse(result['qualified'])
        self.assertTrue(result['policy']['support_optimization'])
        self.assertTrue(result['policy']['penetration_optimization'])
        self.assertFalse(result['policy']['support_validation'])
        self.assertFalse(result['policy']['penetration_validation'])
        self.assertGreater(support.call_count,0)
        self.assertGreater(body.call_count,0)
        self.assertFalse(any(Path(c.args[0]).name.startswith('collision') for c in write.call_args_list))
        self.assertEqual(result['metrics']['support_validation']['status'],'skipped_disabled')
        self.assertEqual(result['metrics']['collision']['status'],'skipped_disabled')

    def test_cli_round_trip_and_validation(self):
        p=argparse.ArgumentParser();add_arguments(p)
        expected=FitPolicy(False,True,False,True,.8,.01,.03,.006)
        self.assertEqual(FitPolicy.from_args(p.parse_args(expected.argv())),expected)
        self.assertEqual(FitPolicy.from_args(p.parse_args([])),FitPolicy())
        self.assertTrue(FitPolicy.from_record().strict)
        self.assertTrue(FitPolicy.from_record({"support_validation":False}).penetration_validation)
        self.assertEqual(FitPolicy.from_record(FitPolicy().record()),FitPolicy())
        for bad in (float('nan'),float('inf'),-.001):
            with self.assertRaises(ValueError): FitPolicy(max_penetration_m=bad)
        with self.assertRaises(ValueError): FitPolicy(support_min_coverage=1.1)

    def test_intrusion_belongs_to_penetration_gate(self):
        metrics={r:dict(coverage=.8,gap_p20=.03,maximum_seat_intrusion=.01) for r in engine.REGIONS}
        support=engine.support_checks(metrics,FitPolicy(support_validation=True))
        penetration=engine.support_checks(metrics,FitPolicy(penetration_validation=True))
        self.assertTrue(support);self.assertTrue(penetration)
        self.assertFalse(any('seat_penetration' in x for x in support))
        self.assertTrue(all('seat_penetration' in x for x in penetration))

    def test_threshold_boundaries_and_candidate_cache_isolation(self):
        from tempfile import TemporaryDirectory
        p=FitPolicy(support_validation=True,penetration_validation=True,support_min_coverage=.8,support_gluteus_gap_m=.012,support_thigh_gap_m=.025,max_penetration_m=.006)
        measurements={r:dict(coverage=.8,gap_p20=p.gap(r),maximum_seat_intrusion=.006) for r in engine.REGIONS}
        self.assertEqual(engine.support_checks(measurements,p),[])
        measurements['left_gluteus']['gap_p20']+=1e-6
        self.assertIn('left_gluteus:insufficient_support',engine.support_checks(measurements,p))
        with TemporaryDirectory() as tmp:
            keys={CandidateCache(tmp,'inputs',FitPolicy(so,po,sv,pv).record(),{},{}).key for so,po,sv,pv in product((False,True),repeat=4)}
            self.assertEqual(len(keys),16)
            self.assertNotIn(CandidateCache(tmp,'inputs',p.record(),{},{}).key,keys)

    @unittest.skipUnless(os.name=='nt','Windows extended paths')
    def test_extended_source_path_with_parent_traversal(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'fit').mkdir();(root/'assets').mkdir()
            multi_scene.write_atomic(root/'assets/asset.json',{})
            directory=Path('\\\\?\\'+str(root/'fit'))
            paths=multi_scene.source_files(dict(source_asset='../assets/asset.json'),directory)
            self.assertTrue(paths[0].is_file())

    def test_all_switch_combinations_preserve_remaining_constraints(self):
        fixture=support_fixture.SupportTests()
        for so,po,sv,pv in product((False,True),repeat=4):
            policy=FitPolicy(so,po,sv,pv)
            with self.subTest(policy=policy), patch.object(engine,'prepare_geometry', wraps=engine.prepare_geometry) as geometry, \
                 patch.object(engine,'close_body_topology',wraps=engine.close_body_topology) as cap, \
                 patch.object(engine,'collision_check',wraps=engine.collision_check) as collision:
                r=fixture.fit_control(policy=policy)
                self.assertEqual(r['selection_eligible'],not r['rejection_reasons'])
                self.assertEqual(r['qualified'],r['selection_eligible'] and sv and pv)
                if not so:self.assertEqual(r['loss_components']['support'],0)
                if not po:self.assertEqual(r['loss_components']['penetration'],0)
                if not pv:collision.assert_not_called();geometry.assert_not_called()
                if not po:cap.assert_not_called()
                if not sv:self.assertEqual(r['metrics']['support_validation']['checked_source_frames'],0)
                if not pv:self.assertEqual(r['metrics']['collision']['status'],'skipped_disabled')

    def test_real_collision_threshold_propagates_to_certificate(self):
        body=trimesh.creation.box(extents=[.15,.15,.5])
        seat=trimesh.creation.box(extents=[.04,.04,.02]);seat.apply_translation([0,0,-.256])
        ctx=dict(dense=body.vertices[None],faces=body.faces)
        low=collision_check(ctx,seat,limit=.003)
        high=collision_check(ctx,seat,limit=.006)
        self.assertIn('full_motion_penetration',collision_checks(low,limit=.003))
        self.assertEqual(collision_checks(high,limit=.006),[])
        self.assertEqual(high['penetration_limit_m'],.006)

    def test_diagnostic_scene_requires_no_collision_artifact(self):
        fixture=scene_fixture.MultiSceneTests();fixture.setUp()
        try:
            rows=fixture.add('A',[('chair',(0,0,0),1.)])
            root=fixture.dirs['A'];policy=FitPolicy(support_validation=False,penetration_validation=False)
            fixture.ctx['policy']=policy
            for c in rows:
                c.update(qualified=False,selection_eligible=True)
                (root/c['mesh']).with_name('collision.json').unlink()
            multi_scene.write_atomic(root/'fit_result.json',dict(policy=policy.record(),candidates=rows))
            multi_scene.seal_instance(fixture.ctx,root,'motion+events','A')
            with patch.object(multi_scene,'static_pair',side_effect=AssertionError),patch.object(multi_scene,'load_candidate',side_effect=AssertionError):
                result=fixture.assemble()
            self.assertEqual(result['status'],'diagnostic_complete')
            self.assertEqual(result['collision']['checked_source_frames'],0)
            self.assertEqual(len(result['selected']),1)
            strict=dict(fixture.ctx,policy=FitPolicy(support_validation=True,penetration_validation=True))
            self.assertNotEqual(multi_scene.scene_inputs(strict,'x'),multi_scene.scene_inputs(fixture.ctx,'x'))
        finally:fixture.doCleanups()
