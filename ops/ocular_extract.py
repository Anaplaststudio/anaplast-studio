"""Extract the highlighted eye front without fitting or flattening its surface."""
import numpy as np
import bpy,bmesh
from mathutils import Vector
from . import ocular as oc,ocular_marking as mark
from ..utils import mesh as mu


def extract_front(obj,selected):
    obj.data.calc_loop_triangles();world=np.array([obj.matrix_world@v.co for v in obj.data.vertices])
    points=[];faces=[];cache={}
    def vertex(key,co):
        if key not in cache:cache[key]=len(points);points.append(co)
        return cache[key]
    for tri in obj.data.loop_triangles:
        ids=list(tri.vertices);poly=[]
        for a,b in zip(ids,ids[1:]+ids[:1]):
            if selected[a]:poly.append(vertex(('v',a),world[a]))
            if selected[a]!=selected[b]:poly.append(vertex(('e',min(a,b),max(a,b)),(world[a]+world[b])*.5))
        for i in range(1,len(poly)-1):faces.append((poly[0],poly[i],poly[i+1]))
    edges={}
    for face in faces:
        for a,b in zip(face,face[1:]+face[:1]):edges.setdefault(tuple(sorted((a,b))),[]).append((a,b))
    boundary=[v[0] for v in edges.values() if len(v)==1];graph={}
    for a,b in boundary:graph.setdefault(a,[]).append(b);graph.setdefault(b,[]).append(a)
    if not graph or any(len(v)!=2 for v in graph.values()):raise RuntimeError('The highlighted surface has an open or branched border; refine the marks')
    start=min(graph);cur=start;previous=None;loop=[]
    while cur not in loop:
        loop.append(cur);n=graph[cur];nxt=n[0] if n[0]!=previous else n[1];previous,cur=cur,nxt
    if cur!=start or len(loop)!=len(graph):raise RuntimeError('Highlight one filled eye opening without holes')
    return np.array(points),faces,loop


def extend_front(points,faces,loop,trace,extension,u,r,f):
    if extension<=0:return points,faces
    xyz=points[loop];inner=np.column_stack((xyz@r,xyz@f));height=xyz@u
    smooth=mark.smooth_loop(xyz,.3,256);poly=np.column_stack((smooth@r,smooth@f))
    center,outer=oc.radial_outline(poly,extension,512)
    from mathutils.geometry import delaunay_2d_cdt
    # Constrain both borders: joining matching rings can fold across the small
    # steps of an exact scan boundary, even if the outer border is smooth.
    step=.4;lo=outer.min(axis=0);hi=outer.max(axis=0)
    X,Y=np.meshgrid(np.arange(lo[0],hi[0],step),np.arange(lo[1],hi[1],step))
    grid=np.c_[X.ravel(),Y.ravel()]
    grid=grid[(oc.signed_distance(grid,inner)>.45*step)&(oc.signed_distance(grid,outer)<-.45*step)]
    inputs=np.vstack((inner,outer,grid));n=len(inner);m=len(outer)
    edges=[(i,(i+1)%n) for i in range(n)]+[(n+i,n+(i+1)%m) for i in range(m)]
    output=delaunay_2d_cdt([Vector(q) for q in inputs],edges,[list(range(n,n+m))],1,1e-7,True)
    xy=np.array(output[0]);ids=[];result=list(points)
    local=np.column_stack((points@r,points@f))-center;x,y=local.T
    A=np.c_[np.ones(len(points)),x,y,x*x,x*y,y*y];coef=np.linalg.lstsq(A,points@u,rcond=None)[0]
    new_indices=[];new_xy=[]
    for i,orig in enumerate(output[3]):
        boundary_ids=[j for j in orig if j<n]
        if boundary_ids:ids.append(loop[boundary_ids[0]])
        else:
            ids.append(len(result));new_indices.append(len(result));new_xy.append(xy[i]);result.append(None)
    q=np.array(new_xy)
    # Continue the height at the nearest exact boundary edge, blending to a
    # smooth fitted surface only in the added region beyond the visible eye.
    closest=np.full(len(q),np.inf);boundary_z=np.zeros(len(q));boundary_xy=np.zeros_like(q)
    for j,(a,b) in enumerate(zip(inner,np.roll(inner,-1,axis=0))):
        edge=b-a;t=np.clip((q-a)@edge/max(edge@edge,1e-15),0,1);near=a+t[:,None]*edge
        distance=np.linalg.norm(q-near,axis=1);take=distance<closest
        closest[take]=distance[take];boundary_xy[take]=near[take]
        boundary_z[take]=(height[j]*(1-t)+height[(j+1)%n]*t)[take]
    local=boundary_xy-center
    gradient=np.c_[coef[1]+2*coef[3]*local[:,0]+coef[4]*local[:,1],coef[2]+coef[4]*local[:,0]+2*coef[5]*local[:,1]]
    tangent=boundary_z+np.sum(gradient*(q-boundary_xy),axis=1)
    x,y=(q-center).T;fitted=np.c_[np.ones(len(q)),x,y,x*x,x*y,y*y]@coef
    t=np.clip(closest/max(extension,1e-6),0,1);blend=t*t*(3-2*t);z=tangent*(1-blend)+fitted*blend
    for i,co in zip(new_indices,q[:,0,None]*r+q[:,1,None]*f+z[:,None]*u):result[i]=co
    faces=list(faces)
    outside=oc.signed_distance(np.array([xy[list(face)].mean(axis=0) for face in output[2]]),inner)>0
    for face,take in zip(output[2],outside):
        if take:
            for j in range(1,len(face)-1):faces.append((ids[face[0]],ids[face[j]],ids[face[j+1]]))
    used=set(i for face in faces for i in face)
    # CDT may retain unused points inside the aperture. Keep original ordering.
    keep=[i for i in range(len(result)) if i in used];remap={v:i for i,v in enumerate(keep)}
    return np.array([result[i] for i in keep]),[tuple(remap[i] for i in face) for face in faces]



def build(context):
    p=context.scene.ocular_production;ap=context.scene.anaplast;src=oc.reference(ap);obj=p.mark_obj;trace=ap.ocular_trace
    if abs(context.scene.unit_settings.scale_length-.001)>1e-8:raise RuntimeError('Use millimetre case units first')
    if not obj or not trace:raise RuntimeError('Highlight the sclera and iris and fit the borders first')
    if not src or src.type!='MESH' or src.modifiers:raise RuntimeError('Assign the original marked scan with applied modifiers')
    signature=mark.wm.geometry_signature(src)
    if obj.get('ocular_source_signature')!=signature or mark.wm.geometry_signature(obj)!=signature:
        raise RuntimeError('The scan or marking copy changed; start a new marking copy')
    if trace.get('ocular_marks_signature')!=mark.mark_signature(obj):raise RuntimeError('The highlights changed; fit their borders again')
    selected=(mark.values_for(obj,'SCLERA')>.5)|(mark.values_for(obj,'IRIS')>.5)
    coords,faces,loop=extract_front(src,selected);original_count=len(coords)
    u=np.array(trace['ocular_up']);r=np.array(trace['ocular_right']);f=np.cross(u,r)
    coords,faces=extend_front(coords,faces,loop,trace,ap.ocular_extension,u,r,f)
    bm=bmesh.new();pair=bm.verts.layers.int.new('ocular_back_index');layer=bm.faces.layers.int.new('ocular_front')
    vs=[bm.verts.new(v) for v in coords]
    for face in faces:
        fa=bm.faces.new([vs[i] for i in face]);fa[layer]=1
    # Posterior geometry is an axial offset, avoiding normal-offset folds on noisy scans.
    bm.normal_update();rim=[tuple(e.verts) for e in bm.edges if e.is_boundary]
    back=[bm.verts.new(v.co-Vector(u)*p.copy_thickness) for v in vs];mapping={v:i for i,v in enumerate(vs)}
    for face in list(bm.faces):bm.faces.new([back[mapping[v]] for v in reversed(face.verts)])
    for a,b in rim:bm.faces.new((a,back[mapping[a]],back[mapping[b]],b))
    bm.verts.index_update()
    for a,b in zip(vs,back):a[pair]=b.index+1
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bmesh.ops.triangulate(bm,faces=list(bm.faces))
    out=mu.new_mesh_object('Ocular_Copy_Work',bm,mu.get_collection(context.scene,oc.COLLECTION))
    try:
        from .mold_design import has_self_intersections
        stats=mu.mesh_stats(out);intersects=has_self_intersections(out)
        if any(stats[1:]) or intersects:raise RuntimeError(f'The copied shell folds or intersects itself ({stats}, intersections={intersects}). Check the marked border or reduce extension/thickness')
        marker=bpy.data.objects.get('Ocular_Cornea_Center')
        if not marker:raise RuntimeError('Fit the highlighted iris first')
        cc=marker.matrix_world.translation
        out['ocular_up']=list(u);out['ocular_right']=list(r);out['ocular_iris_center']=list(r*cc.dot(Vector(r))+f*cc.dot(Vector(f)))
        out['iris_diameter_mm']=ap.ocular_iris_diameter;out['anaplast_part']='OCULAR';out['source_scan']=src.name
        out['copied_front_vertex_count']=original_count;out['front_method']='Exact highlighted source triangles; fitted hidden extension only'
        out['axial_offset_mm']=p.copy_thickness;out['aperture_extension_mm']=ap.ocular_extension
        old=bpy.data.objects.get('Ocular_Shell')
        if old:mu.delete_object(old)
        out.name='Ocular_Shell';ap.ocular_obj=out
        for face in out.data.polygons:face.use_smooth=True
        out.color=(.88,.88,.8,1)
        ap.ocular_report='Copied highlighted front; original visible curvature retained'
        return out
    except Exception:mu.delete_object(out);raise
