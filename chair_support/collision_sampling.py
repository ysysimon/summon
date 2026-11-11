"""Optimization-only surface quadrature. Never exported or used as a certificate."""
from dataclasses import dataclass
import hashlib
import numpy as np
import open3d as o3d

VERSION = "longest-edge-v1"
CHUNK = 65536

@dataclass
class Samples:
    points: np.ndarray
    weights: np.ndarray
    parent: np.ndarray
    barycentric: np.ndarray
    parts: np.ndarray
    max_edge: float
    token: str

    def save(self, path):
        np.savez_compressed(path, points=self.points, area=self.weights, source_face=self.parent,
                            barycentric=self.barycentric, source_part=self.parts,
                            max_world_edge=self.max_edge, version=VERSION)

def build_samples(mesh, parts, mode="legacy", max_edge=.05, max_scale=1.):
    if not np.isfinite(max_edge) or max_edge <= 0 or not np.isfinite(max_scale) or max_scale <= 0:
        raise ValueError("Invalid sampling resolution or scale")
    original = np.vstack((mesh.vertices, mesh.triangles_center))
    _, ids = np.unique(np.floor(original/.02).astype(np.int64), axis=0, return_index=True)
    original = original[ids]
    face_parts = np.full(len(mesh.faces), "other", dtype="U32")
    for name, faces in parts.items():
        if name.endswith("_face_indices"):
            face_parts[np.asarray(faces, int)] = name.removesuffix("_face_indices")
    if mode == "legacy":
        # Lump each original face's area onto its centroid's retained voxel.
        _, inverse = np.unique(np.floor(np.vstack((mesh.vertices, mesh.triangles_center))/.02).astype(np.int64), axis=0, return_inverse=True)
        weights = np.bincount(inverse[len(mesh.vertices):], weights=mesh.area_faces, minlength=len(ids))
        parent = np.full(len(ids), -1, int)
        bary = np.zeros((len(ids), 3))
        labels = np.full(len(ids), "legacy", dtype="U32")
        points = original
        measured = float(np.linalg.norm(mesh.triangles-np.roll(mesh.triangles,-1,axis=1),axis=2).max()*max_scale)
    else:
        triangles = np.asarray(mesh.triangles).copy()
        parent = np.arange(len(triangles))
        bary = np.broadcast_to(np.eye(3), triangles.shape).copy()
        done, parents, coords = [], [], []
        measured = 0.
        while len(triangles):
            edges = np.linalg.norm(triangles-np.roll(triangles,-1,axis=1),axis=2)
            split = edges.max(axis=1)*max_scale > max_edge
            if (~split).any():
                done.append(triangles[~split]); parents.append(parent[~split]); coords.append(bary[~split].mean(axis=1))
                measured=max(measured,float(edges[~split].max()*max_scale))
            if not split.any(): break
            a=edges[split].argmax(axis=1); rows=np.arange(len(a))
            def divide(values):
                v=values[split]; p=v[rows,a]; q=v[rows,(a+1)%3]; r=v[rows,(a+2)%3]; m=(p+q)/2
                return np.stack((np.stack((p,m,r),axis=1),np.stack((m,q,r),axis=1)),axis=1).reshape(-1,3,3)
            triangles, bary = divide(triangles), divide(bary)
            parent=np.repeat(parent[split],2)
        children=np.concatenate(done); parent=np.concatenate(parents); bary=np.concatenate(coords)
        weights=np.linalg.norm(np.cross(children[:,1]-children[:,0],children[:,2]-children[:,0]),axis=1)/2
        labels=np.r_[np.full(len(original),"legacy",dtype="U32"),face_parts[parent]]
        points=np.vstack((original,children.mean(axis=1)))
        weights=np.r_[np.zeros(len(original)),weights]
        parent=np.r_[np.full(len(original),-1),parent]
        bary=np.vstack((np.zeros((len(original),3)),bary))
    points=np.ascontiguousarray(points,dtype=np.float32)
    weights=np.ascontiguousarray(weights,dtype=np.float64)
    token=hashlib.sha256(points.tobytes()+weights.tobytes()+VERSION.encode()).hexdigest()
    return Samples(points,weights,parent,bary,labels,measured,token)

def summarize(scene, samples, rotation, translation, scale, hybrid=False):
    from .runtime import count
    count("collision.requested_points",len(samples.points))
    if hybrid:
        return scene.collision_summary(samples,rotation,translation,scale)
    deepest, worst, squared, area, invalid = 0., -1, 0., 0., 0
    for start in range(0,len(samples.points),CHUNK):
        p=samples.points[start:start+CHUNK]; w=samples.weights[start:start+CHUNK]
        if hasattr(scene,"transformed_distance"):
            d=scene.transformed_distance(p,rotation,translation,scale,inverse=False,signed=True).numpy()
        else:
            posed=p@rotation.T*scale+translation
            d=np.zeros(len(p),np.float32)
            # A point outside the closed body's AABB cannot have negative distance.
            mask=np.all((posed>=scene.bounds[0]-1e-6)&(posed<=scene.bounds[1]+1e-6),axis=1) if hasattr(scene,"bounds") else np.ones(len(p),bool)
            # Majority parity avoids single-ray edge/vertex-hit sign artifacts.
            # Legacy balanced guards retain the original single-ray API path.
            if mask.any(): d[mask]=scene.compute_signed_distance(o3d.core.Tensor(posed[mask].astype(np.float32)),nsamples=3).numpy()
        invalid+=int((~np.isfinite(d)).sum())
        penetration=np.maximum(-d.astype(np.float64),0.)
        i=int(np.argmax(penetration)) if len(penetration) else 0
        if len(penetration) and penetration[i]>deepest: deepest=float(penetration[i]); worst=start+i
        squared+=float(np.dot(w,penetration**2)); area+=float(w[penetration>0].sum())
    if invalid: raise RuntimeError("Non-finite collision query; optimization evidence unavailable")
    total=float(samples.weights.sum())
    return dict(depth=deepest,rms=float(np.sqrt(squared/total)) if total else 0.,squared_sum=squared*scale**2,
                penetrating_area=area*scale**2,total_area=total*scale**2,worst_index=worst,invalid_count=invalid)
