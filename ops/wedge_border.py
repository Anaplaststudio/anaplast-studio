"""Fair the painted rear boundary on the unchanged ear surface."""
import heapq,hashlib
import numpy as np
_cache={}


def refined_border(obj,raw,detail):
    from .wedge_marking import surface_graph,geometry_signature
    selected=np.asarray(raw)>.5
    if not selected.any():raise RuntimeError('Mark the region under the helix first')
    count=int(selected.sum())
    if detail<=0:return selected.astype(np.float32),{'mode':'RAW','raw_vertices':count,'boundary_vertices':count,'boundary_band_mm':0.,'required_vertices_retained':count}
    key=(geometry_signature(obj),hashlib.sha256(selected.tobytes()).hexdigest(),round(float(detail),5))
    if key in _cache:return _cache[key]
    co,edges,lengths,neighbors=surface_graph(obj);a,b=edges.T;crossing=selected[a]!=selected[b]
    if not crossing.any():raise RuntimeError('The marking needs a border on the ear surface')
    median=float(np.median(lengths[lengths>1e-6]));band=max(detail*3.,median*3.)
    distance=np.full(len(co),band);heap=[]
    for i in np.flatnonzero(crossing):
        for v in edges[i]:
            d=min(band,lengths[i]*.5)
            if d<distance[v]:distance[v]=d;heapq.heappush(heap,(d,int(v)))
    while heap:
        d,v=heapq.heappop(heap)
        if d>distance[v]:continue
        for other,length in neighbors[v]:
            trial=d+length
            if trial<distance[other]:distance[other]=trial;heapq.heappush(heap,(trial,other))
    # Work only in the geodesic border band. Nearby opposite ear sheets cannot
    # acquire marks merely because they are close in 3D space.
    active=distance<band-1e-8;signed=distance*np.where(selected,1.,-1.);fair=signed.copy()
    weights=1./np.maximum(lengths,.02)**2;den=np.bincount(a,weights,minlength=len(co))+np.bincount(b,weights,minlength=len(co));strength=min(400.,(detail/max(median,.02))**2*.5)
    for _ in range(100):
        mean=(np.bincount(a,weights*fair[b],minlength=len(co))+np.bincount(b,weights*fair[a],minlength=len(co)))/np.maximum(den,1e-9)
        updated=(signed+strength*mean)/(1.+strength);change=float(np.max(np.abs(updated[active]-fair[active])))
        fair[active]=updated[active]
        if change<1e-5:break
    # Preserve the selected surface area while replacing local teeth and nicks
    # with a smoother level set. The anatomy and raw paint are never edited.
    obj.data.calc_loop_triangles();tri=np.asarray([t.vertices[:] for t in obj.data.loop_triangles]);area=np.linalg.norm(np.cross(co[tri[:,1]]-co[tri[:,0]],co[tri[:,2]]-co[tri[:,0]]),axis=1)/6.
    vertex_area=np.bincount(tri.ravel(),np.repeat(area,3),minlength=len(co))
    ids=np.flatnonzero(active);order=ids[np.argsort(fair[ids])[::-1]];target=float(vertex_area[active&selected].sum());cum=np.cumsum(vertex_area[order]);index=min(len(order)-1,int(np.searchsorted(cum,target)));level=float(fair[order[index]])
    values=np.clip(.5+(fair-level)/(2.*max(detail,.1)),0.,1.);values[~active]=selected[~active]
    use=values>.5
    stats={'mode':'FAIRED_CURVES','detail_mm':float(detail),'boundary_band_mm':band,'raw_vertices':count,'boundary_vertices':int(use.sum()),'removed_mark_vertices':int((selected&~use).sum()),'added_mark_vertices':int((~selected&use).sum()),'raw_area_mm2':float(vertex_area[selected].sum()),'refined_area_mm2':float(vertex_area[use].sum()),'required_vertices_retained':int((selected&use).sum()),'source_geometry_unchanged':True}
    if len(_cache)>2:_cache.clear()
    _cache[key]=(values.astype(np.float32),stats);return _cache[key]
