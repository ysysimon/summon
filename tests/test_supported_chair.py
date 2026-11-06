"""Acceptance invariants, not snapshots of optimizer internals."""
import sys
from pathlib import Path
import unittest
import tempfile
import json
import numpy as np
import trimesh

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chair_support.engine import (REGIONS, support_checks, region_metrics, ray_scene,
    support_query, edge_intersections, close_body_topology, contiguous_runs)
from chair_support.engine import placement_checks,collision_check,collision_checks,read_asset,rotation
from chair_support.engine import fit_asset,estimate_back_slant


class SupportTests(unittest.TestCase):
    def test_back_initialization_tracks_lean_and_ignores_root_translation(self):
        z=np.linspace(.1,.4,100)
        body=np.column_stack((np.zeros(100),-.3*z,z))
        root=np.array([[0.,0.,0.],[1.,2.,3.]])
        ctx=dict(dense=body[None]+root[:,None],pelvis=root,
                 forward=np.tile([0.,1.,0.],(2,1)),support_frames=np.array([0,1]),frame_weights=np.ones(2))
        profile=estimate_back_slant(ctx,np.arange(100))
        self.assertAlmostEqual(profile['initial_back_slant'],.3,places=6)
        ctx['dense'][:,:,1]-=.4*z
        profile=estimate_back_slant(ctx,np.arange(100))
        self.assertEqual(profile['initial_back_slant'],.4)
        self.assertTrue(profile['limited_to_native_bounds'])

    def test_curved_patch_does_not_require_every_point_to_touch(self):
        # A convex underside touches near its low point, naturally receding.
        gaps=np.array([0.,.002,.006,.020,.040,.060])
        metrics={r:region_metrics(gaps,np.ones(len(gaps),bool)) for r in REGIONS}
        self.assertEqual(support_checks(metrics),[])

    def test_floating_and_narrow_seats_fail(self):
        floating={r:region_metrics(np.full(10,.08),np.ones(10,bool)) for r in REGIONS}
        self.assertTrue(any("insufficient_support" in s for s in support_checks(floating)))
        narrow={r:region_metrics(np.zeros(10),np.arange(10)<5) for r in REGIONS}
        self.assertTrue(any("coverage" in s for s in support_checks(narrow)))

    def test_coverage_uses_projected_area_not_surface_slope(self):
        # Equal XY cells cover equal footprints even when one surface is steep.
        metrics=region_metrics(np.array([.002,.010]),np.array([True,False]),np.array([1.,20.]))
        self.assertEqual(metrics["coverage"],.5)
        self.assertAlmostEqual(metrics["gap_p20"],.002)

    def test_actual_seat_footprint_not_infinite_plane(self):
        v=np.array([[-.2,-.2,0.],[.2,-.2,0.],[.2,.2,0.],[-.2,.2,0.]])
        scene=ray_scene(v,np.array([[0,1,2],[0,2,3]]))
        gap,valid=support_query(np.array([[0.,0.,.01],[1.,0.,.01]]),scene,np.eye(3),np.zeros(3),1.)
        self.assertEqual(valid.tolist(),[True,False])
        self.assertAlmostEqual(gap[0],.01,places=5)

    def test_edge_crossing_detected_with_endpoints_outside(self):
        box=trimesh.creation.box(extents=[1,1,1]);scene=ray_scene(box.vertices,box.faces)
        v=np.array([[-2.,0.,0.],[2.,0.,0.]])
        hits=edge_intersections(scene,v,np.array([[0,1]]),box.bounds)
        self.assertEqual(len(hits),2)
        np.testing.assert_allclose(np.sort(hits[:,0]),[-.5,.5],atol=1e-5)

    def test_open_loop_cap_does_not_change_visible_vertices(self):
        box=trimesh.creation.box()
        faces=box.faces[:-2]
        closed,loops=close_body_topology(faces,len(box.vertices))
        vertices=np.vstack([box.vertices]+[box.vertices[x].mean(0)[None] for x in loops])
        self.assertTrue(trimesh.Trimesh(vertices,closed,process=False).is_watertight)

    def test_phase_intervals_remain_separate(self):
        self.assertEqual(contiguous_runs(np.array([False,True,True,False,True])),[(1,3),(4,5)])

    def test_reversed_sideways_and_floating_geometry_fail(self):
        # A labelled back panel behind the pelvis, whose base touches z=0.
        panel=trimesh.creation.box(extents=[.5,.05,1.])
        panel.apply_translation([0.,-.3,.5])
        parts={"backrest_face_indices":np.arange(len(panel.faces))}
        pelvis=np.array([[0.,0.,.6]]);forward=np.array([[0.,1.,0.]])
        _,ok=placement_checks(panel,parts,pelvis,forward,forward[0],0.)
        self.assertEqual(ok,[])
        for degrees in (90,180):
            r=rotation(np.radians(degrees));turned=trimesh.Trimesh(panel.vertices@r.T,panel.faces,process=False)
            _,reasons=placement_checks(turned,parts,pelvis,forward,forward[0]@r.T,0.)
            self.assertIn("chair_heading_mismatch",reasons)
        floating=panel.copy();floating.apply_translation([0.,0.,.03])
        _,reasons=placement_checks(floating,parts,pelvis,forward,forward[0],0.)
        self.assertIn("chair_not_grounded",reasons)

    def test_deep_leg_intersection_fails_but_shallow_contact_can_pass(self):
        body=trimesh.creation.box(extents=[.15,.15,.5])
        ctx={"dense":body.vertices[None],"faces":body.faces}
        for depth in (.003,.02):
            seat=trimesh.creation.box(extents=[.04,.04,.02])
            seat.apply_translation([0.,0.,-.26+depth])
            result=collision_check(ctx,seat)
            self.assertGreater(result["edge_triangle_crossings"],0)
            if depth<.005: self.assertEqual(collision_checks(result),[])
            else: self.assertIn("full_motion_penetration",collision_checks(result))

    def test_invalid_part_indices_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);trimesh.creation.box().export(root/"mesh.obj")
            np.savez(root/"parts.npz",seat_faces=[99999],backrest_faces=[0])
            (root/"asset.json").write_text(json.dumps({"up_axis":"+Z"}))
            with self.assertRaisesRegex(ValueError,"chair surface"):
                read_asset(root/"asset.json")

    def fit_control(self,mesh_scale=1.,source="infinigen-test", offsets=None, selected=None, weights=None):
        def box(extents,center):
            result=trimesh.creation.box(extents=extents)
            result.apply_translation(center)
            return result
        centers=[(-.12,-.10,.55),(.12,-.10,.55),(-.12,.10,.55),(.12,.10,.55)]
        body=trimesh.util.concatenate([box([.08,.08,.1],p) for p in centers])
        seat=box([.6,.6,.05],[0.,0.,.475])
        back=box([.6,.04,.4],[0.,-.28,.7])
        legs=[box([.04,.04,.45],[x,y,.225]) for x in (-.26,.26) for y in (-.26,.26)]
        chair=trimesh.util.concatenate([seat,back,*legs])
        chair.apply_scale(mesh_scale)
        part_ids=np.r_[np.zeros(len(seat.faces),np.int8),np.ones(len(back.faces),np.int8),
                       np.full(sum(len(m.faces) for m in legs),2,np.int8)]
        stable=np.arange(3) if selected is None else np.asarray(selected)
        ctx=dict(dense=np.repeat(body.vertices[None],3,axis=0),faces=body.faces,support_frames=stable,
            frame_weights=np.ones(3) if weights is None else np.asarray(weights),
            fps=30.,pelvis=np.tile([0.,0.,.62],(3,1)),forward=np.tile([0.,1.,0.],(3,1)),floor=0.)
        if offsets is not None:
            ctx["dense"][:,:,2]+=np.asarray(offsets)[:,None]
        anchors={}
        for index,region in enumerate(REGIONS):
            ids=np.arange(index*8,(index+1)*8)
            ids=ids[np.isclose(body.vertices[ids,2],.5)]
            anchors[region]=[(int(f),ids,ctx["dense"][f,ids],np.ones(len(ids))) for f in stable]
        estimate=dict(dimensions={"seat_height":.5},forward=[0.,1.,0.],initial_seat_center=[0.,0.,.62])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);chair.export(root/"mesh.obj")
            np.savez(root/"parts.npz",face_part_ids=part_ids,
                seat_face_indices=np.flatnonzero((part_ids==0)&(chair.face_normals[:,2]>.9)),
                backrest_face_indices=np.flatnonzero((part_ids==1)&(chair.face_normals[:,1]>.9)))
            (root/"asset.json").write_text(json.dumps(dict(id="geometry-control",source=source,up_axis="+Z")))
            result=fit_asset(root/"asset.json",ctx,anchors,estimate,root/"fit")
            return result

    def test_known_supported_geometry_survives_complete_fit_and_audit(self):
        # A positive geometric control prevents an implementation that merely
        # rejects every candidate from satisfying the negative-case suite.
        result=self.fit_control()
        self.assertTrue(result["qualified"],result["rejection_reasons"])
        self.assertEqual(result["metrics"]["collision"]["checked_source_frames"],3)

    def test_infeasible_tall_cad_cannot_escape_scale_bounds(self):
        result=self.fit_control(mesh_scale=2.,source="3D-FUTURE-test")
        self.assertFalse(result["qualified"])
        self.assertGreaterEqual(result["metrics"]["scale"],.9)
        self.assertLessEqual(result["metrics"]["scale"],1.1)
        self.assertTrue(any("seat_penetration" in reason for reason in result["rejection_reasons"]))

    def test_low_weight_support_frame_cannot_hide_floating_body(self):
        result=self.fit_control(offsets=[.08,0,0],weights=[1e-6,1,1])
        self.assertFalse(result["qualified"])
        self.assertTrue(any("insufficient_support" in reason for reason in result["rejection_reasons"]))

    def test_transition_frame_outside_support_is_still_collision_checked(self):
        result=self.fit_control(offsets=[0,-.03,0],selected=[0,2])
        self.assertFalse(result["qualified"])
        self.assertIn("full_motion_penetration",result["rejection_reasons"])
        self.assertEqual(result["worst_collision_frame"],1)


if __name__=="__main__": unittest.main()
