"""Optimization cache invariants and real spawn-worker result equivalence."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import trimesh
from chair_support import engine
from chair_support.solver_helpers import PoseCache, choose_current_result, batched_support_metrics, candidate_directory_key
from chair_support.candidate_pool import pack_arrays, unpack_arrays, ordered_spawn_map, MappedArray
from chair_support.runtime import ExecutionOptions
from chair_support.policy import FitPolicy
from chair_support.engine import (REGIONS, _candidate_record, _candidate_worker,
                                  _initialize_candidate_worker, ray_scene, support_query, region_metrics)


def sleeper(seconds):
    time.sleep(seconds)
    return seconds


def failure(_):
    raise RuntimeError("worker failure")


def blas_threads(_):
    from threadpoolctl import threadpool_info
    return [row["num_threads"] for row in threadpool_info() if row["user_api"] == "blas"]


def fixture(root):
    def box(size, center):
        mesh=trimesh.creation.box(extents=size);mesh.apply_translation(center);return mesh
    centers=[(-.12,-.10,.55),(.12,-.10,.55),(-.12,.10,.55),(.12,.10,.55)]
    body=trimesh.util.concatenate([box([.08,.08,.1],p) for p in centers])
    seat=box([.6,.6,.05],[0.,0.,.475]);back=box([.6,.04,.4],[0.,-.28,.7])
    legs=[box([.04,.04,.45],[x,y,.225]) for x in (-.26,.26) for y in (-.26,.26)]
    chair=trimesh.util.concatenate([seat,back,*legs])
    labels=np.r_[np.zeros(len(seat.faces),np.int8),np.ones(len(back.faces),np.int8),np.full(sum(len(m.faces) for m in legs),2,np.int8)]
    np.save(root/'body.npy',np.repeat(body.vertices[None],3,axis=0))
    ctx=dict(policy=FitPolicy(support_validation=True,penetration_validation=True),
        dense=np.load(root/'body.npy',mmap_mode='r'),faces=body.faces,support_frames=np.arange(3),
        frame_weights=np.ones(3),fps=30.,pelvis=np.tile([0.,0.,.62],(3,1)),forward=np.tile([0.,1.,0.],(3,1)),floor=0.)
    anchors={}
    for i,r in enumerate(REGIONS):
        ids=np.arange(i*8,(i+1)*8);ids=ids[np.isclose(body.vertices[ids,2],.5)]
        anchors[r]=[(f,ids,ctx['dense'][f,ids],np.ones(len(ids))) for f in range(3)]
    paths=[]
    for name in ['first','second']:
        folder=root/name;folder.mkdir();chair.export(folder/'mesh.obj')
        np.savez(folder/'parts.npz',face_part_ids=labels,seat_face_indices=np.flatnonzero((labels==0)&(chair.face_normals[:,2]>.9)),backrest_face_indices=np.flatnonzero((labels==1)&(chair.face_normals[:,1]>.9)))
        path=folder/'asset.json';path.write_text(json.dumps(dict(id=name,source='infinigen-test',up_axis='+Z')));paths.append(path)
    estimate=dict(dimensions={'seat_height':.5},forward=[0.,1.,0.],initial_seat_center=[0.,0.,.62])
    return paths,ctx,anchors,estimate


class SolverCacheTests(unittest.TestCase):
    def test_windows_candidate_directory_aliases_are_detected(self):
        self.assertEqual(candidate_directory_key('Chair', windows=True),
                         candidate_directory_key('chair', windows=True))
        for name in ('chair.', 'chair ', 'NUL', 'con.obj', 'LPT1', 'a:b', 'a/b', 'a\\b'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                candidate_directory_key(name, windows=True)

    def test_exact_pose_and_epoch_and_mutation_isolation(self):
        cache=PoseCache(2);calls=[]
        def compute():calls.append(1);return {'frames':[{'depth':.003}]}
        p=np.array([0.,1.,2.,1.])
        first=cache.get(p,'audit',compute,copy=True);first['frames'][0]['depth']=9
        self.assertEqual(cache.get(p,'audit',compute,copy=True)['frames'][0]['depth'],.003)
        self.assertEqual(len(calls),1)
        changed=p.copy();changed[1]=np.nextafter(changed[1],np.inf)
        cache.get(changed,'audit',compute);self.assertEqual(len(calls),2)
        cache.invalidate();cache.get(p,'audit',compute);self.assertEqual(len(calls),3)
        cache.get(p,('audit',.006),compute);self.assertEqual(len(calls),4)

    def test_certificate_cache_is_independent_of_optimization_epochs(self):
        objective=PoseCache();certificate=PoseCache();calls=[]
        pose=np.zeros(4)
        certificate.get(pose,'audit',lambda:calls.append(1))
        objective.invalidate()
        certificate.get(pose,'audit',lambda:calls.append(1))
        self.assertEqual(calls,[1])

    def test_historical_scores_are_refreshed_before_selection(self):
        old=SimpleNamespace(x=np.array([0.]),fun=-100.)
        new=SimpleNamespace(x=np.array([1.]),fun=100.)
        chosen=choose_current_result([old,new],lambda x:2-x[0],lambda x:0.)
        self.assertIs(chosen,new);self.assertEqual(old.fun,2.);self.assertEqual(new.fun,1.)

    def test_batch_support_matches_per_frame_metrics(self):
        mesh=trimesh.creation.box(extents=[1,1,.1]);scene=ray_scene(mesh.vertices,mesh.faces)
        samples=[(i,np.arange(2),np.array([[0.,0.,.06+i*.001],[.2,.2,.07]]),np.array([1.,2.])) for i in range(3)]
        expected=[(f,region_metrics(*support_query(p,scene,np.eye(3),np.zeros(3),1.),w)) for f,_,p,w in samples]
        self.assertEqual(expected,batched_support_metrics(samples,scene,np.eye(3),np.zeros(3),1.,support_query,region_metrics))


class SpawnCandidateTests(unittest.TestCase):
    def test_fit_reuses_strict_audit_after_witness_epoch_at_identical_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths,ctx,anchors,estimate=fixture(root)
            ctx['_execution']=ExecutionOptions(profile_fit=True,query_threads=1)
            original_audit=engine.collision_check
            audit_calls=[]

            def repeated_pose(function, start, method, **kwargs):
                # Exercise the complete repair/validation loop while controlling
                # the solver's output, including COBYLA's reduced coordinates.
                pose=np.array([0.,0.,0.,1.]) if method == 'Powell' else np.zeros(3)
                return SimpleNamespace(x=pose,fun=float(function(pose)),success=True,
                                       message='controlled identical placement',nfev=1)

            def reject_with_new_witness(*args, **kwargs):
                audit_calls.append(1)
                certificate=original_audit(*args, **kwargs)
                self.assertEqual(certificate['status'],'certified_below_limit')
                # Inject one measured rejection to activate a new witness and
                # objective epoch. The CPU audit itself still runs normally.
                certificate.update(status='rejected',maximum_penetration=.006,
                    maximum_penetration_lower_bound=.006,maximum_penetration_upper_bound=.006,
                    worst_collision_frame=0,penetration_witness=dict(
                        furniture_point=[.11,.07,.5],furniture_face=0,
                        furniture_part='seat',sample_lower_bound=.006))
                return certificate

            with patch.object(engine,'minimize',side_effect=repeated_pose), \
                    patch.object(engine,'collision_check',side_effect=reject_with_new_witness) as audit:
                result=engine.fit_asset(paths[0],ctx,anchors,estimate,root/'fit')
            self.assertEqual(audit.call_count,1)
            self.assertEqual(audit_calls,[1])
            self.assertFalse(result['qualified'])
            self.assertIn('full_motion_penetration',result['rejection_reasons'])
            directory=root/'fit'/'candidates'/'first'
            optimization=json.loads((directory/'optimization.json').read_text())
            performance=json.loads((directory/'performance.json').read_text())
            self.assertGreaterEqual(optimization['objective_epoch'],1)
            self.assertEqual(optimization['collision_repairs'],1)
            self.assertEqual(len(optimization['collision_witnesses']),1)
            self.assertGreaterEqual(performance['counters']['collision_requests'],2)
            self.assertEqual(performance['counters']['collision_evaluations'],1)
            self.assertGreaterEqual(performance['counters']['collision_cache_hits'],1)
            ctx['dense']._mmap.close()

    @unittest.skipUnless(os.name == 'nt', 'Windows filesystem alias rules')
    def test_case_aliases_fail_before_starting_candidate_writers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths,ctx,anchors,estimate=fixture(root)
            ctx['phase']={}
            for path,name in zip(paths,('Chair','chair')):
                meta=json.loads(path.read_text());meta['id']=name;path.write_text(json.dumps(meta))
            manifest=root/'assets.json';manifest.write_text(json.dumps({'assets':[str(p) for p in paths]}))
            body=str(root/'body.npy')
            argv=['--geometry',body,'--vertices-path',body,'--contacts',body,'--frame-indices',body,
                  '--assets-manifest',str(manifest),'--output',str(root/'output'),'--fit-workers','2']
            with patch.object(engine,'motion_context',return_value=ctx), \
                    patch.object(engine,'body_regions',return_value={}), \
                    patch.object(engine,'prepare_support',return_value=(anchors,estimate)), \
                    patch.object(engine,'contact_roles',return_value=None), \
                    patch.object(engine,'ordered_spawn_map') as spawn, \
                    patch.object(engine,'_candidate_record') as fit:
                with self.assertRaisesRegex(ValueError,'Duplicate asset id'):
                    engine.main(argv)
                spawn.assert_not_called();fit.assert_not_called()
            self.assertFalse((root/'output'/'candidates').exists())
            ctx['dense']._mmap.close()

    def test_worker_limits_blas_while_query_budget_is_separate(self):
        results = list(ordered_spawn_map(blas_threads, [None], 1,
            _initialize_candidate_worker, (None,), timeout=15))
        self.assertTrue(results[0])
        self.assertEqual(set(results[0]), {1})

    def test_failed_candidate_keeps_profile_in_its_own_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths,ctx,anchors,estimate=fixture(root)
            ctx['_execution']=ExecutionOptions(profile_fit=True)
            with patch('chair_support.engine.query_scene', side_effect=ValueError('query construction failed')):
                record=_candidate_record(paths[0],ctx,anchors,estimate,root/'failed',None)
            self.assertFalse(record['qualified'])
            perf=json.loads((root/'failed'/record['performance']).read_text())
            self.assertEqual(perf['status'],'error')
            self.assertEqual(perf['error_type'],'ValueError')
            self.assertGreater(perf['seconds']['candidate_total'],0)
            self.assertGreater(perf['counters']['candidate_wall_seconds'],0)
            ctx['dense']._mmap.close()

    def test_memmap_and_slice_are_reopened_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'a.npy';np.save(path,np.arange(48).reshape(4,4,3))
            source=np.load(path,mmap_mode='r');packed=pack_arrays({'a':source[1:3]})
            self.assertIsInstance(packed['a'],MappedArray)
            restored=unpack_arrays(packed)['a'];np.testing.assert_array_equal(restored,source[1:3])
            self.assertFalse(restored.flags.writeable)
            restored._mmap.close();source._mmap.close()

    def test_real_candidates_match_serial_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths,ctx,anchors,estimate=fixture(root)
            serial=[_candidate_record(p,ctx,anchors,estimate,root/'serial',None) for p in paths]
            ctx={k:v for k,v in ctx.items() if not k.startswith('_collision')}
            ctx['_execution']=ExecutionOptions(fit_workers=2,query_threads=1,profile_fit=True)
            inputs=pack_arrays((ctx,anchors,estimate,root/'parallel',None))
            parallel=list(ordered_spawn_map(_candidate_worker,paths,2,_initialize_candidate_worker,(inputs,),timeout=30))
            self.assertEqual([r['id'] for r in parallel],['first','second'])
            for a,b in zip(serial,parallel):
                for key in ['qualified','loss','transform','rejection_reasons','metrics']:
                    self.assertEqual(a[key],b[key],key)
                self.assertTrue(b['qualified'])
                perf=json.loads((root/'parallel'/b['performance']).read_text())
                self.assertGreater(perf['counters']['objective_calls'],perf['counters']['objective_evaluations'])
            ctx['dense']._mmap.close()

    def test_timeout_failure_and_cancel_reap_only_owned_children(self):
        before={p.pid for p in mp.active_children()}
        with self.assertRaises(mp.TimeoutError):list(ordered_spawn_map(sleeper,[20],1,timeout=.05))
        with self.assertRaisesRegex(RuntimeError,'worker failure'):list(ordered_spawn_map(failure,[0],1,timeout=15))
        pending=ordered_spawn_map(sleeper,[0,20],2,timeout=15)
        self.assertEqual(next(pending),0);pending.close()
        self.assertEqual({p.pid for p in mp.active_children()},before)


if __name__=='__main__':unittest.main()
