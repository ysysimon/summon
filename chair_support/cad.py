"""Versioned functional-surface annotations for the eight bundled CAD chairs.

The predicates below select faces on the original mesh (no remeshing). Heights
and front directions were inspected on the bundled 3D-FUTURE geometry; masks
and coloured inspection meshes are exported so the annotations are reviewable.
"""
import json
from pathlib import Path

import numpy as np
import trimesh

from .engine import rotation, sha256, write_json

# Source OBJ is Y-up. The initial +90deg X rotation makes it Z-up.
# yaw then makes the chair's opening point along +Y. Seat height is a narrow
# search window, NOT a substituted planar seat for fitting/collision.
ANNOTATIONS = {
    "01fe3767-6f21-49cf-b8da-e16b53267dc8": (180., .46, .035),
    "0a79b3eb-03c5-3df4-9a60-b4b99c443d87": (180., .44, .045),
    "0f92470c-b703-49fb-b317-fb3e570363aa": (180., .91, .055),
    "252c2950-6bad-4bf7-a557-76b1a08f7394": (180., .43, .035),
    "4c54b216-fa90-4f66-8a03-ea59ee2be6ae": (90., .51, .055),
    "5149c843-f441-4589-a67f-e6ac7caf021e": (180., .48, .065),
    "6f09d6a0-dcd6-472a-9793-bafc17a8486f": (180., .49, .060),
    "925bc907-2909-43e4-a988-261bf098e3be": (180., .495, .045),
}


def prepare_cad_assets(models_dir, destination):
    models_dir, destination = Path(models_dir), Path(destination)
    chair_root = models_dir / "chair" if (models_dir / "chair").is_dir() else models_dir
    annotation_data=json.loads((Path(__file__).parent/"annotations"/"cad_chair_annotations_v1.json").read_text(encoding="utf-8"))
    assets=[]
    def rejected_asset(identity,reason):
        path=destination/identity/"asset.json"
        write_json(path,dict(id=identity,source="3D-FUTURE",annotation_error=reason))
        assets.append(path)
    for identity,(yaw,seat_height,band) in ANNOTATIONS.items():
        source=chair_root/identity/"raw_model.obj"
        if not source.is_file():
            rejected_asset(identity,"Missing CAD source mesh: "+str(source));continue
        if sha256(source)!=annotation_data["assets"][identity]["source_sha256"]:
            rejected_asset(identity,"CAD annotation does not match source mesh hash: "+identity);continue
        m=trimesh.load(source,force="mesh",process=False)
        vertices=np.column_stack((m.vertices[:,0],-m.vertices[:,2],m.vertices[:,1])) @ rotation(np.radians(yaw)).T
        vertices[:,2]-=vertices[:,2].min()
        m=trimesh.Trimesh(vertices,m.faces,process=False)
        normals,centers=m.face_normals,m.triangles_center
        seat=np.flatnonzero((normals[:,2]>.45)&(np.abs(centers[:,2]-seat_height)<band))
        if len(seat)<3:
            rejected_asset(identity,"CAD seat annotation is empty: "+identity);continue
        seat_vertices=np.unique(m.faces[seat])
        seat_center=np.median(vertices[seat_vertices],axis=0)
        back=np.flatnonzero((normals[:,1]>.20)&(centers[:,2]>seat_height+.04)&(centers[:,1]<seat_center[1]))
        # A curved junction can face both upward and forward. Seat support has
        # precedence inside its asset-specific height band; labels stay disjoint.
        back=np.setdiff1d(back,seat,assume_unique=True)
        if len(back)<3:
            rejected_asset(identity,"CAD backrest annotation is empty: "+identity);continue
        front_y=float(vertices[seat_vertices,1].max())
        front_vertices=seat_vertices[vertices[seat_vertices,1]>front_y-.035]
        front_faces=np.flatnonzero(np.isin(m.faces,front_vertices).any(axis=1)&(np.abs(centers[:,2]-seat_height)<band+.02))
        floor_vertices=np.flatnonzero(vertices[:,2]<=.005)
        ids=np.full(len(m.faces),2,np.int8);ids[seat]=0;ids[back]=1
        out=destination/identity;out.mkdir(parents=True,exist_ok=True)
        m.export(out/"mesh.obj")
        np.savez_compressed(out/"parts.npz",face_part_ids=ids,seat_face_indices=seat,
            backrest_face_indices=back,front_edge_face_indices=front_faces,
            front_edge_vertices=front_vertices,floor_vertices=floor_vertices,
            forward=np.array([0.,1.,0.]),forward_origin=seat_center)
        colors=np.array([[80,190,80,255],[80,130,240,255],[160,160,160,255]],np.uint8)
        m.visual.face_colors=colors[ids];m.export(out/"annotated.ply")
        write_json(out/"asset.json",dict(schema_version=1,id=identity,source="3D-FUTURE",
            mesh="mesh.obj",parts="parts.npz",units="m",up_axis="Z",forward_axis="+Y",
            source_mesh=str(source.resolve()),source_mesh_sha256=sha256(source),
            geometry_sha256=sha256(out/"mesh.obj"),
            annotation=dict(version=1,method="asset-specific height/direction and face-normal predicates",
                yaw_degrees=yaw,seat_height=seat_height,seat_height_band=band,
                seat_faces=len(seat),backrest_faces=len(back),review_mesh="annotated.ply")))
        assets.append(out/"asset.json")
    return assets
