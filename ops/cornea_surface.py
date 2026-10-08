"""Preserve the fitted corneal curve when refining its manufacturing mesh."""
import json
import numpy as np
from mathutils.kdtree import KDTree

def surface(src):
    from .ocular_fit import height
    from .ocular_steps import cornea_profile
    up=np.array(src['ocular_up']);right=np.array(src['ocular_right']);forward=np.cross(up,right)
    center=np.array(src['ocular_iris_center']);cc=np.array([center@right,center@forward])
    radius=float(src['workflow_cornea_radius_mm'])
    sphere_center=np.array(src['workflow_sphere_center_local'])
    sphere_radius=src['workflow_sphere_radius_mm'];model=json.loads(src.get('workflow_surface_model','null'))
    def z(xy):
        return height(xy,sphere_center,sphere_radius,model)+cornea_profile(np.linalg.norm(xy-cc,axis=1)/radius,radius,src['workflow_cornea_height_mm'])
    return up,right,forward,cc,radius,z

def reproject(coords,front,original_count,src):
    """Re-evaluate added corneal vertices, retaining original and boundary vertices."""
    if not src.get('workflow_has_iris') or not src.get('workflow_cornea_radius_mm'):return coords,0.
    up,right,forward,cc,radius,z=surface(src)
    ids=np.array(sorted({i for face in front for i in face if i>=original_count}),dtype=int)
    if not len(ids):return coords,0.
    xy=np.c_[coords[ids]@right,coords[ids]@forward];rho=np.linalg.norm(xy-cc,axis=1)/radius
    active=rho<1.;ids=ids[active];xy=xy[active];rho=rho[active]
    if not len(ids):return coords,0.
    counts={}
    for face in front:
        for a,b in zip(face,face[1:]+face[:1]):
            edge=tuple(sorted((a,b)));counts[edge]=counts.get(edge,0)+1
    boundary={i for edge,count in counts.items() if count==1 for i in edge}
    # Keep the join to the original rounded rim fixed. Fade before reaching it.
    if boundary:
        tree=KDTree(len(boundary))
        for k,i in enumerate(boundary):tree.insert(coords[i],k)
        tree.balance();distance=np.array([tree.find(coords[i])[2] for i in ids])
        weight=np.clip((distance-.25)/.5,0,1)
    else:weight=np.ones(len(ids))
    weight*=np.clip((1-rho)/.08,0,1);weight=weight*weight*(3-2*weight)
    delta=(z(xy)-coords[ids]@up)*weight
    # A large difference means this is no longer the authored fitted surface.
    if np.max(abs(delta))>.05:raise RuntimeError('The corneal surface was edited; Update eye preview to rebuild its fitted curve')
    result=coords.copy();result[ids]+=delta[:,None]*up
    return result,float(np.max(abs(delta)))

def smooth_normals(obj,src):
    """Use the continuous surface gradient only on the exposed clear cornea."""
    if not src.get('workflow_has_iris') or not src.get('workflow_cornea_radius_mm'):return 0
    up,right,forward,cc,radius,z=surface(src)
    me=obj.data;coords=np.array([v.co[:] for v in me.vertices]);normals=np.array([v.normal[:] for v in me.vertices])
    xy=np.c_[coords@right,coords@forward];rho=np.linalg.norm(xy-cc,axis=1)/radius
    outer={i for face in me.polygons if face.material_index==0 and np.dot(face.normal,up)>.2 for i in face.vertices}
    ids=np.array([i for i in outer if rho[i]<.98],dtype=int)
    if not len(ids):return 0
    ids=ids[abs(coords[ids]@up-z(xy[ids]))<.005]
    if not len(ids):return 0
    pos=xy[ids];step=.001
    dx=(z(pos+[step,0])-z(pos-[step,0]))/(2*step)
    dy=(z(pos+[0,step])-z(pos-[0,step]))/(2*step)
    desired=up-dx[:,None]*right-dy[:,None]*forward
    desired/=np.linalg.norm(desired,axis=1)[:,None]
    weight=np.clip((.98-rho[ids])/.10,0,1);weight=weight*weight*(3-2*weight)
    blended=normals[ids]*(1-weight[:,None])+desired*weight[:,None]
    blended/=np.linalg.norm(blended,axis=1)[:,None];normals[ids]=blended
    # Retain existing split normals on the internal interface and rounded back.
    corners=np.array([n.vector[:] for n in me.corner_normals]);lookup={int(i):n for i,n in zip(ids,blended)}
    for face in me.polygons:
        if face.material_index!=0 or np.dot(face.normal,up)<=.2:continue
        for loop in face.loop_indices:
            i=me.loops[loop].vertex_index
            if i in lookup:corners[loop]=lookup[i]
    me.normals_split_custom_set(corners.tolist())
    return len(ids)
