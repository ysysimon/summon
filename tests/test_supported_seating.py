"""Category routing, shared furniture, and true-mesh seating controls."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chair_support import engine
from chair_support.multi_fit import instance_contacts
from chair_support.multi_scene import ranked_choices
from chair_support.policy import FitPolicy
from chair_support.seating import classify_locations, hypotheses, rank_score, heading_errors


def recognition_fixture():
    locations = [dict(instance_id=f"seat-{i}", anchor_xy_m=[x, 0.], category=category,
                      category_counts={"chair":0,"sofa":8,"stool":0}, support_sample_count=2)
                 for i, (x, category) in enumerate([(0., "sofa"), (2., "sofa"), (4., "chair")])]
    events = [dict(instance_id=f"seat-{i}", location_id=f"seat-{i}", start=i*4, end=i*4+3,
                   support=[[i*4,i*4+3]], category=location["category"])
              for i, location in enumerate(locations)]
    return dict(instances=deepcopy(locations), locations=locations, events=events)


class RoutingTests(unittest.TestCase):
    def test_support_votes_ignore_transition_noise_and_preserve_arrays(self):
        labels=np.full((5,6),3,int)
        labels[0]=4;labels[1,0]=4;labels[2]=7;labels[3]=7
        original=labels.copy()
        locations=[dict(instance_id="a"),dict(instance_id="b")]
        visits=[dict(instance_id="a",support=[[1,2]]),dict(instance_id="b",support=[[2,4]])]
        classify_locations(locations,visits,labels,np.arange(5))
        self.assertEqual([r["category"] for r in locations],["chair","stool"])
        self.assertEqual(locations[0]["category_counts"]["sofa"],1)
        np.testing.assert_array_equal(labels,original)

    def test_revisits_accumulate_and_ties_use_raw_label_order(self):
        labels=np.array([[4,4,3],[3,3,4],[7,7,7]])
        locations=[dict(instance_id="a")]
        visits=[dict(instance_id="a",support=[[0,1]]),dict(instance_id="a",support=[[1,2]])]
        classify_locations(locations,visits,labels,np.arange(3))
        self.assertEqual(locations[0]["category"],"chair")
        self.assertEqual(locations[0]["support_sample_count"],2)
        self.assertEqual([v["location_id"] for v in visits],["a","a"])

    def test_shared_sofa_keeps_locations_and_all_visits_without_distance_gate(self):
        rec=recognition_fixture();before=deepcopy(rec)
        choices=hypotheses(rec)
        self.assertEqual([len(h["instances"]) for h in choices],[3,2])
        shared=choices[1]
        self.assertEqual(shared["events"][0]["instance_id"],shared["events"][1]["instance_id"])
        self.assertNotEqual(shared["events"][1]["instance_id"],shared["events"][2]["instance_id"])
        self.assertEqual([v["support"] for v in shared["events"]],[v["support"] for v in rec["events"]])
        self.assertEqual(rec,before)
        self.assertEqual(len(shared["anchor_map"][shared["events"][0]["instance_id"]]),2)

    def test_chairs_and_stools_are_never_merged_by_sofa_hypothesis(self):
        rec=recognition_fixture()
        rec["locations"][0]["category"]="chair"
        rec["locations"][1]["category"]="stool"
        self.assertEqual(len(hypotheses(rec)),1)
        self.assertEqual(len(hypotheses(rec)[0]["instances"]),3)

    def test_shared_surface_hints_order_but_do_not_restrict_partitions(self):
        rec=recognition_fixture()
        rec["locations"][2]["category"]="sofa"
        rec["ambiguities"]=[dict(kind="possible_shared_seating_surface",instance_ids=["seat-1","seat-2"])]
        choices=hypotheses(rec)
        self.assertEqual(len(choices),5)
        self.assertEqual([len(row["instances"]) for row in choices[:3]],[3,2,1])
        self.assertEqual(choices[1]["instances"][1]["location_ids"],["seat-1","seat-2"])

    def test_bounded_enumeration_and_sample_weighted_ranking(self):
        rec=recognition_fixture()
        self.assertEqual(len(hypotheses(rec,1)),1)
        h=hypotheses(rec)[0]
        h["instances"][0]["support_sample_count"]=6
        score=rank_score(h,{"seat-0":1.,"seat-1":2.,"seat-2":3.})
        self.assertAlmostEqual(score["weighted_fit_loss"],1.6)
        self.assertAlmostEqual(score["total"],1.603)
        self.assertAlmostEqual(rank_score(h,{r["instance_id"]:0. for r in h["instances"]},0)["total"],0)

    def test_anchor_sets_assign_second_sofa_position_to_same_instance(self):
        world=np.array([[[0.,0.,.5],[1.,0.,.5],[2.,0.,.5]]])
        labels=np.array([[4,4,7]])
        mask=instance_contacts(world,labels,np.array([0]),np.array([0]),"sofa",
                               {"sofa":[[0.,0.],[1.,0.]],"stool":[[2.,0.]]},seat_labels=(3,4,7))
        self.assertEqual(mask.tolist(),[[True,True,False]])

    def test_scene_combinations_follow_the_same_sample_weights_as_hypotheses(self):
        pools=[[dict(loss=0.),dict(loss=.2)],[dict(loss=0.),dict(loss=.3)]]
        self.assertEqual([indices for _,indices in ranked_choices(pools)][:2],[(0,0),(1,0)])
        weighted=list(ranked_choices(pools,[.9,.1]))
        self.assertEqual([indices for _,indices in weighted][:2],[(0,0),(0,1)])
        self.assertAlmostEqual(weighted[1][0],.03)

    def test_heading_uses_equivalent_orientations(self):
        body=np.array([[0.,1.,0.],[1.,0.,0.]])
        np.testing.assert_allclose(heading_errors(body,[0,-1,0],{"type":"continuous"}),[0,0])
        np.testing.assert_allclose(heading_errors(body,[0,-1,0],{"type":"discrete","order":2}),[0,90])


def box(size,center):
    mesh=trimesh.creation.box(extents=size)
    mesh.apply_translation(center)
    return mesh


def fit_control(root,category,positions,strict,source="infinigen-test",asset_path=None,seat_height=.5,
                obstacle=None,initial_center=0.,obstacle_size=(.08,.08,.1)):
    body=trimesh.util.concatenate([box([.08,.08,.1],[x,y,seat_height+.05]) for x,y in ((-.12,-.10),(.12,-.10),(-.12,.10),(.12,.10))])
    if obstacle is not None:
        body=trimesh.util.concatenate([body,box(obstacle_size,obstacle)])
    width=1.8 if category=="sofa" else .6
    meshes=[box([width,.6,.05],[0,0,.475])]
    ids=[np.zeros(len(meshes[0].faces),np.int8)]
    if category=="sofa":
        back=box([width,.04,.4],[0,-.28,.7]);meshes.append(back);ids.append(np.ones(len(back.faces),np.int8))
    for x in (-width/2+.04,width/2-.04):
        for y in (-.26,.26):
            leg=box([.04,.04,.45],[x,y,.225]);meshes.append(leg);ids.append(np.full(len(leg.faces),2,np.int8))
    furniture=trimesh.util.concatenate(meshes);part_ids=np.concatenate(ids)
    furniture.export(root/"mesh.obj")
    np.savez(root/"parts.npz",face_part_ids=part_ids,
             seat_faces=np.flatnonzero((part_ids==0)&(furniture.face_normals[:,2]>.9)),
             backrest_faces=np.flatnonzero((part_ids==1)&(furniture.face_normals[:,1]>.9)))
    (root/"asset.json").write_text(json.dumps(dict(id="control",category=category,source=source,
        heading_symmetry={"type":"continuous" if category=="stool" else "none"})))
    n=len(positions);dense=np.array([body.vertices+[x,0,0] for x in positions])
    ctx=dict(dense=dense,faces=body.faces,support_frames=np.arange(n),frame_weights=np.ones(n),fps=30.,
             pelvis=np.array([[x,0,seat_height+.12] for x in positions]),forward=np.tile([0.,1.,0.],(n,1)),floor=0.,
             policy=FitPolicy(support_validation=strict,penetration_validation=strict),category=category)
    if category=="sofa":
        visits=[dict(instance_id="sofa",location_id=f"location-{i}",start=i,end=i+1,support=[[i,i+1]]) for i in range(n)]
        ctx.update(visits=visits,phase=dict(visits=[dict(v,selected_source_frames=[i]) for i,v in enumerate(visits)]))
    anchors={}
    for i,region in enumerate(engine.REGIONS):
        selected=np.arange(i*8,(i+1)*8)
        selected=selected[np.isclose(body.vertices[selected,2],seat_height)]
        anchors[region]=[(f,selected,dense[f,selected],np.ones(len(selected))) for f in range(n)]
    estimate=dict(dimensions={"seat_height":seat_height},forward=[0,1,0],initial_seat_center=[initial_center,0,seat_height+.12])
    return engine.fit_asset(asset_path or root/"asset.json",ctx,anchors,estimate,root/"fit")


class SolverTests(unittest.TestCase):
    def test_shared_footprint_estimate_preserves_legacy_chair_width(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            def capture(asset,ctx,anchors,*args):
                return ctx,anchors
            with patch.object(engine,"fit_asset",capture):
                ctx,anchors=fit_control(root,"sofa",[0.,1.],False)
            ctx.update(_support_anchors=anchors,phase=dict(support_seconds=2/30,warnings=[]))
            regions={region:np.arange(i*8,(i+1)*8) for i,region in enumerate(engine.REGIONS)}
            regions["back"]=np.arange(32)
            widths={}
            for category in ("chair","sofa"):
                ctx["category"]=category
                _,estimate=engine.prepare_support(ctx,regions,root)
                widths[category]=estimate["dimensions"]["seat_width"]
            self.assertAlmostEqual(widths["chair"],.38)
            self.assertAlmostEqual(widths["sofa"],1.38)

    def test_sofa_end_seating_can_move_beyond_the_old_centered_search(self):
        with TemporaryDirectory() as tmp:
            # A raised body part would hit the backrest of a body-centred sofa.
            # Supporting the same fixed body near the seat end avoids it.
            result=fit_control(Path(tmp),"sofa",[.6],True,obstacle=[.6,0.,.7],initial_center=.6,
                               obstacle_size=(.08,2.,.5))
            self.assertTrue(result["qualified"],result["rejection_reasons"])
            self.assertLess(result["transform"]["translation"][0],.35)

    def test_backless_stool_and_shared_sofa_fit_with_existing_strict_policy(self):
        for category,positions in (("stool",[0.,0.]),("sofa",[-.5,.5,-.5])):
            with self.subTest(category=category),TemporaryDirectory() as tmp:
                result=fit_control(Path(tmp),category,positions,True)
                self.assertTrue(result["qualified"],result["rejection_reasons"])
                self.assertEqual(result["category"],category)
                self.assertEqual(result["metrics"]["collision"]["checked_source_frames"],len(positions))
                if category=="sofa":
                    self.assertEqual(len(result["metrics"]["visits"]),3)
                    self.assertTrue(all(v["verified"] for v in result["metrics"]["visits"]))

    def test_default_mode_is_diagnostic_for_new_categories(self):
        for category in ("stool","sofa"):
            with self.subTest(category=category),TemporaryDirectory() as tmp:
                result=fit_control(Path(tmp),category,[0.],False,source="3D-FUTURE-test")
                self.assertTrue(result["selection_eligible"],result["rejection_reasons"])
                self.assertFalse(result["qualified"])
                self.assertEqual(result["metrics"]["collision"]["status"],"skipped_disabled")

    def test_circular_stool_does_not_require_an_arbitrary_heading(self):
        mesh=box([.5,.5,.5],[0,0,.25])
        metrics,reasons=engine.placement_checks(mesh,{"backrest_face_indices":np.array([],int)},
            np.array([[0,0,.6]]),np.array([[0,1,0]]),np.array([1,0,0]),0.,{"type":"continuous"})
        self.assertEqual(reasons,[])
        self.assertIsNone(metrics["backrest_forward_offset_max"])


if __name__=="__main__":
    unittest.main()
