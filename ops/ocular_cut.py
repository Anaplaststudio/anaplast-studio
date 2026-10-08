"""Local, smooth eyelid cut contours; never move the source scan vertices."""
import numpy as np
from mathutils.kdtree import KDTree
from . import ocular as oc, ocular_marking as mark


def fill_cut_islands(obj,selected,world):
    """Remove small unpainted remnants enclosed by a verified eye cutout."""
    selected=selected.copy();adj=[[] for _ in selected]
    for edge in obj.data.edges:
        a,b=edge.vertices
        adj[a].append(b);adj[b].append(a)
    unseen=set(np.flatnonzero(~selected));filled=0
    while unseen:
        start=unseen.pop();group=[start];stack=[start];touches=False
        while stack:
            for j in adj[stack.pop()]:
                if selected[j]:touches=True
                elif j in unseen:unseen.remove(j);stack.append(j);group.append(j)
        if touches and len(group)<=256 and np.linalg.norm(np.ptp(world[group],axis=0))<2.5:
            selected[group]=True;filled+=len(group)
    return selected,filled


def cut_field(obj, selected, world, right, up, smoothing=.25):
    """Signed distance near the painted border, negative on the removed side.

    Smooth in physical millimetres, then intersect the original triangles with
    that contour. A 3-D neighbourhood restricts the change to this surface, so
    other surfaces behind the eye cannot be cut by the projected outline.
    Multiple loops retain holes, including an iris not yet transferred.
    """
    selected=np.asarray(selected,dtype=bool)
    if not selected.any():return np.ones(len(world))
    loops=mark.contours(obj,selected,world)
    raw=np.concatenate(loops)
    tree=KDTree(len(raw))
    for i,p in enumerate(raw):tree.insert(p,i)
    tree.balance()
    band=max(1.,smoothing*4)
    field=np.where(selected,-band,band)
    candidates=np.flatnonzero(np.all((world>=raw.min(axis=0)-band)&(world<=raw.max(axis=0)+band),axis=1))
    distances=np.array([tree.find(world[i])[2] for i in candidates])
    near=distances<band;indices=candidates[near];distances=distances[near]
    xy=np.c_[world[indices]@right,world[indices]@up]
    inside=np.zeros(len(indices),bool);distance=np.full(len(indices),np.inf)
    for loop in loops:
        length=np.linalg.norm(np.roll(loop,-1,axis=0)-loop,axis=1).sum()
        # Tiny contours may be intentional; retain them instead of dropping
        # fragments or silently filling an untransferred iris.
        contour=mark.smooth_loop(loop,smoothing,max(64,int(np.ceil(length/.04)))) if length>=1 else loop
        polygon=np.c_[contour@right,contour@up]
        d=oc.signed_distance(xy,polygon)
        inside^=d<0;distance=np.minimum(distance,np.abs(d))
    smooth=np.where(inside,-distance,distance)
    original=np.where(selected[indices],-distances,distances)
    weight=np.clip((band-distances)/(band*.4),0,1)
    field[indices]=weight*smooth+(1-weight)*original
    # Avoid exactly-on-vertex degeneracies when triangulating the retained side.
    field=np.where(np.abs(field)<1e-8,np.where(selected,-1e-8,1e-8),field)
    return field
