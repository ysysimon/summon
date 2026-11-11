import itertools
import os
from pathlib import Path
import sys
import unittest
import tempfile
import json
from unittest.mock import patch
import numpy as np
import trimesh
from scipy.optimize import OptimizeResult
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chair_support.collision_sampling import build_samples,summarize
from chair_support.query_backend import cpu_scene,collision_scene,query_scene
from chair_support.runtime import ExecutionOptions
from chair_support.priority import choose,ranked_combinations,scene_score

class SamplingTests(unittest.TestCase):
    def test_long_thin_faces_and_area_provenance(self):
        mesh=trimesh.Trimesh([[0,0,0],[2,0,0],[0,.001,0]],[[0,1,2]],process=False)
        before=mesh.vertices.copy()
        s=build_samples(mesh,{},"longest-edge",.05,1.1)
        self.assertLessEqual(s.max_edge,.05)
        self.assertAlmostEqual(s.weights.sum(),mesh.area)
        selected=s.parent>=0
        expected=np.einsum('ni,nij->nj',s.barycentric[selected],mesh.triangles[s.parent[selected]])
        np.testing.assert_allclose(s.points[selected],expected,atol=1e-7)
        np.testing.assert_array_equal(mesh.vertices,before)

    def test_large_triangle_interior_collision(self):
        mesh=trimesh.Trimesh([[-2,-2,0],[2,-2,0],[0,2,0]],[[0,1,2]],process=False)
        body=trimesh.creation.box(extents=[.1,.1,.1]);body.apply_translation([.5,0,0])
        scene=cpu_scene(body.vertices,body.faces)
        old=summarize(scene,build_samples(mesh,{}),np.eye(3),np.zeros(3),1)
        new=summarize(scene,build_samples(mesh,{},"longest-edge",.02),np.eye(3),np.zeros(3),1)
        self.assertEqual(old['depth'],0)
        self.assertGreater(new['depth'],.03)

    def test_support_cannot_buy_deeper_collision(self):
        scores=[(.03,.01,0,0),(.001,.0001,100,0)]
        choices=[OptimizeResult(x=np.array([i])) for i in range(2)]
        self.assertIs(choose(choices,lambda x:scores[int(x[0])]),choices[1])

    def test_scene_heap_uses_admissible_bounds(self):
        pools=[[dict(priority_score=[1,10,0,0]),dict(priority_score=[2,1,0,0])],
               [dict(priority_score=[10,0,0,0]),dict(priority_score=[11,0,0,0])]]
        expected=sorted((scene_score([pools[j][i] for j,i in enumerate(ids)],[.5,.5]),ids)
                        for ids in itertools.product(range(2),repeat=2))
        self.assertEqual(list(ranked_combinations(pools,[.5,.5])),expected)

    def test_failed_support_solver_cannot_replace_lower_collision_pose(self):
        from chair_support import engine
        from chair_support.policy import FitPolicy
        from test_solver_performance import fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths,ctx,anchors,estimate=fixture(root)
            ctx['policy']=FitPolicy()
            ctx['_execution']=ExecutionOptions(query_threads=1,collision_sampling='longest-edge')
            violated=[]
            def solver(function,start,method,**kwargs):
                pose=np.array([0.,0.,0.,1.]) if method=='Powell' else np.array([0.,.2,0.])
                if method=='COBYLA':
                    violated.append(np.min(kwargs['constraints'][0]['fun'](pose)))
                return OptimizeResult(x=pose,fun=function(pose),success=False,
                                      message='controlled failed support step',nfev=1)
            with patch.object(engine,'minimize',side_effect=solver):
                result=engine.fit_asset(paths[0],ctx,anchors,estimate,root/'fit')
            self.assertTrue(violated)
            self.assertLess(min(violated),-.001)
            self.assertLessEqual(result['priority_score'][0],1e-5)
            self.assertEqual(result['optimization_collision']['checked_source_frames'],3)
            self.assertEqual(result['metrics']['collision']['status'],'skipped_disabled')
            self.assertFalse(result['qualified'])
            record=json.loads((root/'fit/candidates/first/optimization.json').read_text())
            self.assertGreater(len(record['guard_risk_scans']),0)
            ctx['dense']._mmap.close()

    def test_cpu_chunking_preserves_area_statistics(self):
        from chair_support import collision_sampling
        body=trimesh.creation.box()
        samples=build_samples(trimesh.creation.box(extents=[.8,.8,.8]),{},'longest-edge',.07)
        scene=cpu_scene(body.vertices,body.faces)
        expected=summarize(scene,samples,np.eye(3),np.zeros(3),1.)
        with patch.object(collision_sampling,'CHUNK',113):
            actual=summarize(scene,samples,np.eye(3),np.zeros(3),1.)
        for key in ('depth','rms','squared_sum','penetrating_area','total_area'):
            self.assertAlmostEqual(expected[key],actual[key],places=8)
        self.assertEqual(expected['worst_index'],actual['worst_index'])

    def test_balanced_legacy_keeps_single_ray_and_old_search(self):
        from chair_support import engine
        from chair_support.policy import FitPolicy
        from chair_support.query_backend import CPUScene
        from test_solver_performance import fixture
        signed=CPUScene.compute_signed_distance
        rays=[]
        def query(scene,points,nsamples=1):
            rays.append(nsamples)
            return signed(scene,points,nsamples=nsamples)
        def solver(function,start,method,**kwargs):
            pose=np.array([0.,0.,0.,1.]) if method=='Powell' else np.zeros(3)
            return OptimizeResult(x=pose,fun=function(pose),success=True,message='fixed pose',nfev=1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths,ctx,anchors,estimate=fixture(root)
            ctx['policy']=FitPolicy()
            ctx['_execution']=ExecutionOptions(query_threads=1,fit_objective='balanced')
            with patch.object(CPUScene,'compute_signed_distance',query),patch.object(engine,'minimize',side_effect=solver):
                result=engine.fit_asset(paths[0],ctx,anchors,estimate,root/'fit')
            self.assertTrue(rays)
            self.assertEqual(set(rays),{1})
            self.assertNotIn('priority_score',result)
            record=json.loads((root/'fit/candidates/first/optimization.json').read_text())
            self.assertEqual(record['guard_risk_scans'],[])
            ctx['dense']._mmap.close()

    @unittest.skipUnless(os.environ.get('YSYNTHETIC_TEST_WARP')=='1','Explicit CUDA test')
    def test_hybrid_summary_and_cpu_support(self):
        body=trimesh.creation.box();samples=build_samples(trimesh.creation.box(extents=[.6,.6,.6]),{},'longest-edge',.03)
        options=ExecutionOptions(fit_query_backend='hybrid',fit_workers=2,query_threads=1)
        gpu=collision_scene(body.vertices,body.faces,options)
        cpu=cpu_scene(body.vertices,body.faces,options)
        self.assertFalse(hasattr(query_scene(body.vertices,body.faces,options),'transformed_distance'))
        a=summarize(cpu,samples,np.eye(3),np.zeros(3),1)
        b=summarize(gpu,samples,np.eye(3),np.zeros(3),1,True)
        for k in ('depth','rms','squared_sum','penetrating_area'):
            self.assertAlmostEqual(a[k],b[k],places=5)
        self.assertEqual(b,summarize(gpu,samples,np.eye(3),np.zeros(3),1,True))

    @unittest.skipUnless(os.environ.get('YSYNTHETIC_TEST_WARP')=='1','Explicit CUDA test')
    def test_hybrid_spawn_matches_serial(self):
        from test_solver_performance import fixture
        from chair_support.engine import _candidate_record,_candidate_worker,_initialize_candidate_worker
        from chair_support.candidate_pool import pack_arrays,ordered_spawn_map
        from chair_support.policy import FitPolicy
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths,ctx,anchors,estimate=fixture(root)
            ctx['policy']=FitPolicy()
            ctx['_execution']=ExecutionOptions(fit_query_backend='hybrid',query_threads=1,collision_sampling='longest-edge')
            serial=[_candidate_record(p,ctx,anchors,estimate,root/'serial',None) for p in paths]
            ctx={k:v for k,v in ctx.items() if not k.startswith('_collision')}
            ctx['_execution']=ExecutionOptions(fit_query_backend='hybrid',fit_workers=2,query_threads=1,collision_sampling='longest-edge')
            packed=pack_arrays((ctx,anchors,estimate,root/'parallel',None))
            parallel=list(ordered_spawn_map(_candidate_worker,paths,2,_initialize_candidate_worker,(packed,),timeout=90))
            for a,b in zip(serial,parallel):
                self.assertNotIn('error',a);self.assertNotIn('error',b)
                self.assertEqual(a['transform'],b['transform'])
                self.assertEqual(a['priority_score'],b['priority_score'])
            ctx['dense']._mmap.close()

if __name__=='__main__':unittest.main()
