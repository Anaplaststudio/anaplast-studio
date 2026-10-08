"""Compact gravity feed and local air return, for cap-down casting.

The two open cups flare outside the base. Their independent passages share one
cartridge, never a common lumen. This vents only the region beside the pedestal;
it is not automatic detection or venting of remote cavity pockets.
"""
import math
import json
import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def clean_seams(obj, _depth=0):
    """Stitch only sub-micron Boolean seams on generated compact inserts.

    Never resample a surface. Accept a candidate only if it is connected,
    closed, intersection-free and within 0.0005 mm of the input surface.
    Endpoint stitches are limited to 0.00025 mm.
    Otherwise restore its original mesh and fail the enclosing transaction.
    """
    from . import mold_inserts as mi, mold_design as md
    from .mold_breakable import clean_backing_topology
    if not md.has_self_intersections(obj):return
    original=obj.data
    limit=.00025
    failures=[]
    def duplicates(bm):
        bm.normal_update();bm.verts.index_update();groups={}
        for face in bm.faces:groups.setdefault(tuple(sorted(v.index for v in face.verts)),[]).append(face)
        doomed=[]
        for faces in groups.values():
            if len(faces)==2 and faces[0].normal.dot(faces[1].normal)<0:doomed.extend(faces)
            elif len(faces)>1:doomed.extend(faces[1:])
        if doomed:bmesh.ops.delete(bm,geom=doomed,context='FACES_ONLY')
        wire=[e for e in bm.edges if not e.link_faces]
        if wire:bmesh.ops.delete(bm,geom=wire,context='EDGES')
        orphan=[v for v in bm.verts if not v.link_faces]
        if orphan:bmesh.ops.delete(bm,geom=orphan,context='VERTS')
    for method in ('LOCAL','COINCIDENT'):
        candidate=original.copy();obj.data=candidate
        bm=bmesh.new();bm.from_mesh(candidate)
        try:
            if method=='COINCIDENT':bmesh.ops.remove_doubles(bm,verts=bm.verts[:],dist=.00005)
            bmesh.ops.triangulate(bm,faces=bm.faces[:])
            if method=='LOCAL':
                for _ in range(64):
                    bm.faces.ensure_lookup_table();tree=BVHTree.FromBMesh(bm)
                    pairs=[(a,b) for a,b in tree.overlap(tree) if a<b and set(bm.faces[a].verts).isdisjoint(bm.faces[b].verts)]
                    if not pairs:break
                    a,b=pairs[0]
                    near=[((x.co-y.co).length,x,y) for x in bm.faces[a].verts for y in bm.faces[b].verts if (x.co-y.co).length<limit]
                    if near:
                        _,x,y=min(near,key=lambda value:value[0])
                        bmesh.ops.pointmerge(bm,verts=[x,y],merge_co=x.co.copy())
                    else:
                        junctions=[]
                        for one,two in ((bm.faces[a],bm.faces[b]),(bm.faces[b],bm.faces[a])):
                            for vertex in one.verts:
                                for edge in two.edges:
                                    x,y=edge.verts;axis=y.co-x.co
                                    if axis.length_squared<1e-16:continue
                                    fraction=(vertex.co-x.co).dot(axis)/axis.length_squared
                                    if not .000001<fraction<.999999:continue
                                    distance=(vertex.co-x.co-axis*fraction).length
                                    if distance<limit:junctions.append((distance,vertex,edge,x,fraction))
                        if not junctions:break
                        _,vertex,edge,x,fraction=min(junctions,key=lambda value:value[0])
                        # Preserve the T-junction in BOTH incident triangles.
                        # Generic quad triangulation may choose the old edge
                        # again, stranding the inserted point in a zero-area
                        # triangle and repeating the same seam indefinitely.
                        opposites=[next(v for v in face.verts if v not in edge.verts)
                                   for face in edge.link_faces if len(face.verts)==3]
                        _,new=bmesh.utils.edge_split(edge,x,fraction)
                        for opposite in opposites:
                            bmesh.ops.connect_verts(bm,verts=[new,opposite],check_degenerate=True)
                        bmesh.ops.pointmerge(bm,verts=[vertex,new],merge_co=vertex.co.copy())
                        bmesh.ops.triangulate(bm,faces=bm.faces[:])
                duplicates(bm)
                # A two-edge crack has two virtually coincident endpoints.
                boundary=set(e for e in bm.edges if e.is_boundary)
                while boundary:
                    queue=[boundary.pop()];edges=[]
                    while queue:
                        edge=queue.pop();edges.append(edge)
                        for vertex in edge.verts:
                            for other in vertex.link_edges:
                                if other in boundary:boundary.remove(other);queue.append(other)
                    vertices=list({v for e in edges for v in e.verts})
                    if len(edges)==2 and len(vertices)==3:
                        dist,x,y=min((((x.co-y.co).length,x,y) for i,x in enumerate(vertices) for y in vertices[i+1:]),key=lambda value:value[0])
                        if dist<.0001:bmesh.ops.pointmerge(bm,verts=[x,y],merge_co=x.co.copy())
                # A collapsed sliver can leave a multiply-used tiny edge.
                # Collapse only those non-manifold edges, never valid detail.
                for _ in range(32):
                    short=[edge for edge in bm.edges if not edge.is_manifold and edge.calc_length()<limit]
                    if not short:break
                    edge=min(short,key=lambda edge:edge.calc_length());x,y=edge.verts
                    bmesh.ops.pointmerge(bm,verts=[x,y],merge_co=x.co.copy())
                    duplicates(bm)
                # Edge splitting can leave collinear, zero-thickness sheets
                # on a multiply-used edge. Remove only these incident sheets;
                # valid narrow faces elsewhere are left alone.
                for _ in range(8):
                    suspect={face for edge in bm.edges if not edge.is_manifold for face in edge.link_faces}
                    flat=[]
                    for face in suspect:
                        if len(face.verts)!=3:continue
                        edge=max(face.edges,key=lambda e:e.calc_length());x,y=edge.verts
                        axis=y.co-x.co
                        if axis.length<1e-12:flat.append(face);continue
                        third=next(v for v in face.verts if v not in edge.verts)
                        altitude=axis.cross(third.co-x.co).length/axis.length
                        if altitude<.00001:flat.append(face)
                    if not flat:break
                    bmesh.ops.delete(bm,geom=flat,context='FACES_ONLY');duplicates(bm)
                # Zip the collinear chains exposed by removing a collapsed
                # sheet. Split long boundary edges at existing chain vertices.
                # This adds connectivity, not a patch across an actual hole.
                for _ in range(128):
                    edges=[edge for edge in bm.edges if edge.is_boundary]
                    vertices=list({v for edge in edges for v in edge.verts})
                    pairs=[((x.co-y.co).length,x,y) for j,x in enumerate(vertices) for y in vertices[j+1:] if (x.co-y.co).length<.00001]
                    if pairs:
                        _,x,y=min(pairs,key=lambda p:p[0])
                        bmesh.ops.pointmerge(bm,verts=[x,y],merge_co=x.co.copy());duplicates(bm)
                        continue
                    junctions=[]
                    for edge in edges:
                        x,y=edge.verts;axis=y.co-x.co
                        if axis.length_squared<1e-16:continue
                        for vertex in vertices:
                            if vertex in edge.verts:continue
                            fraction=(vertex.co-x.co).dot(axis)/axis.length_squared
                            if not .000001<fraction<.999999:continue
                            distance=(vertex.co-x.co-axis*fraction).length
                            if distance<.00001:junctions.append((distance,edge,x,vertex,fraction))
                    if not junctions:break
                    _,edge,x,vertex,fraction=min(junctions,key=lambda p:p[0])
                    _,new=bmesh.utils.edge_split(edge,x,fraction)
                    bmesh.ops.pointmerge(bm,verts=[vertex,new],merge_co=vertex.co.copy())
            bm.to_mesh(candidate);candidate.update()
        finally:bm.free()
        try:
            if method=='COINCIDENT':clean_backing_topology(obj)
            mi.closed(obj,connected=True)
            if md.has_self_intersections(obj):
                if _depth>=2:raise ValueError('remaining intersections')
                clean_seams(obj,_depth+1)
                candidate=obj.data
            def surface(mesh):
                sample=bmesh.new()
                try:
                    sample.from_mesh(mesh);sample.transform(obj.matrix_world)
                    bmesh.ops.triangulate(sample,faces=sample.faces[:])
                    sample.faces.ensure_lookup_table();tree=BVHTree.FromBMesh(sample)
                    triangles=np.array([[v.co[:] for v in face.verts] for face in sample.faces],dtype=np.float64)
                    points=[v.co.copy() for v in sample.verts]+[Vector(p) for p in triangles.mean(axis=1)]
                    return tree,points,triangles
                finally:sample.free()
            def triangle_distance(point,triangle):
                # BVH single-precision distances become inaccurate on the very
                # skinny triangles we are checking. Recompute in float64.
                a,b,c=triangle-np.array(point,dtype=np.float64)
                n=np.cross(b-a,c-a);nn=float(n@n)
                if nn>1e-28:
                    q=n*float(a@n)/nn
                    weights=[float(np.cross(b-q,c-q)@n)/nn,float(np.cross(c-q,a-q)@n)/nn,float(np.cross(a-q,b-q)@n)/nn]
                    if min(weights)>=-1e-10:return float(np.linalg.norm(q))
                best=math.inf
                for x,y in ((a,b),(b,c),(c,a)):
                    axis=y-x;den=float(axis@axis)
                    t=max(0.,min(1.,-float(x@axis)/den)) if den else 0.
                    best=min(best,float(np.linalg.norm(x+axis*t)))
                return best
            old_tree,old_points,old_triangles=surface(original);new_tree,new_points,new_triangles=surface(candidate)
            error=0.
            for points,tree,triangles in ((new_points,old_tree,old_triangles),(old_points,new_tree,new_triangles)):
                low=triangles.min(axis=1);high=triangles.max(axis=1)
                for point in points:
                    nearest=tree.find_nearest(point)
                    if nearest[0] is None:raise ValueError('missing seam surface')
                    distance=nearest[3]
                    if distance>.0001:
                        distance=triangle_distance(point,triangles[nearest[2]])
                        if distance>.0005:
                            # BVH range queries use the same float32 distance
                            # test. Use float64 bounds for the fallback instead.
                            q=np.array(point,dtype=np.float64)
                            indices=np.flatnonzero(np.all(low<=q+.0005,axis=1)&np.all(high>=q-.0005,axis=1))
                            for index in indices:
                                distance=min(distance,triangle_distance(point,triangles[index]))
                    error=max(error,distance)
            if error>.0005:raise ValueError(f'seam surface deviation {error}')
            obj['compact_seam_error_mm']=error
            obj['compact_seam_method']=method
            if original.users==0:bpy.data.meshes.remove(original)
            return
        except (ValueError,RuntimeError) as exc:
            failures.append(f'{method}: {exc}')
            obj.data=original
            if candidate.users==0:bpy.data.meshes.remove(candidate)
    raise RuntimeError('The compact insert has a Boolean seam that cannot be closed without altering its surface; change the mounting point or use Original straight feed. '+ '; '.join(failures))


def enabled(item):
    return item is not None and item.wax_mode == 'GRAVITY' and item.reservoir_layout == 'SIDE_CUPS'


def dimensions(item, ocular=None):
    feed = item.feed_diameter * .5
    vent = item.vent_diameter * .5
    wall = 1.5
    distance = feed + vent + wall
    # Symmetric root envelope, including walls; independent of cup capacity.
    radius = max(distance * .5 + max(feed, vent) + wall,
                 ocular.stem * .5 if ocular else 0.)
    return dict(feed=feed, vent=vent, wall=wall, distance=distance, radius=radius)


def ports(work, item, center, base, frame, ocular=None,substitute_access=True):
    from . import mold_inserts as mi
    from . import mold_ocular_substitute as substitute
    d = dimensions(item, ocular)
    u, r, f = frame
    tree = mi.aw.world_tree(base)
    lo, hi = mi.bounds(base, u)
    xy = center-u*center.dot(u)
    candidates=[]
    former=substitute.plan(work,ocular,base,center,frame) if substitute_access and substitute.enabled(ocular) else None
    for direction in (r,-r,f,-f):
        if former and ocular.wax_pillars and abs(direction.dot(former['pillar_axis']))>.65:continue
        air_exit=substitute.outlet_distance(former,direction,frame) if former else d['radius']+3
        feed_exit=substitute.outlet_distance(former,-direction,frame) if former else d['radius']+3
        offsets=np.linspace(d['distance']*.5,air_exit,7)
        feed_offsets=np.linspace(d['distance']*.5,feed_exit,7)
        try:
            air=[mi.column(tree,xy+direction*float(t),frame,lo,hi)[1] for t in offsets]
            feed=[mi.column(tree,xy-direction*float(t),frame,lo,hi)[1] for t in feed_offsets]
        except RuntimeError:continue
        # In cap-down orientation, smaller mold-axis height is physically
        # higher. Prefer the highest unobstructed local side for air collection.
        candidates.append((max(air),direction,air,feed,offsets,feed_exit))
    if not candidates:raise RuntimeError('Compact passages extend beyond the base; move the mounting point inward')
    _,r,air,feed,offsets,feed_exit=min(candidates,key=lambda item:item[0])
    d['feed_exit']=feed_exit
    f=u.cross(r);frame=(u,r,f);d['flow_frame']=frame
    d['feed_z']=max(feed)+d['feed']+.3
    # Place the vent at the fitting contour on the chosen high side. The small
    # overlap opens its roof at the insert surface without cutting the host.
    level=max(air)+d['vent']-.1
    d['vent_branch']=[xy+r*float(t)+u*level for t in (offsets[0],offsets[-1])]
    d['vent_z'] = d['vent_branch'][0].dot(u)
    d['boss_top'] = max(d['feed_z']+d['feed'],max(p.dot(u) for p in d['vent_branch'])+d['vent'])+d['wall']
    d['xy'] = xy
    return d


def tube(work, name, points, radius, frame):
    """Closed elbow cutter: cylindrical legs with a spherical inside junction."""
    from . import mold_inserts as mi
    result = None
    for a,b in zip(points,points[1:]):
        axis=(b-a).normalized()
        right=frame[2]-axis*frame[2].dot(axis)
        if right.length<.01:right=frame[1]-axis*frame[1].dot(axis)
        right.normalize()
        leg=mi.cylinder(work,name,a,b,radius,(axis,right,axis.cross(right)))
        result=leg if result is None else mi.merge(work,result,leg)
        if b != points[-1]:
            bm=bmesh.new()
            bmesh.ops.create_uvsphere(bm,u_segments=32,v_segments=16,radius=radius)
            for v in bm.verts:v.co+=b
            mesh=bpy.data.meshes.new(name+'_elbow');bm.to_mesh(mesh);bm.free()
            ball=bpy.data.objects.new(name+'_elbow',mesh);work.col.objects.link(ball);work.add(ball)
            result=mi.merge(work,result,ball)
    return result


def make(work,item,blank,center,dock,frame,fill_ml,base,ocular=None,has_pedestal=False,port_data=None):
    from . import mold_inserts as mi
    from . import mold_ocular_substitute as substitute
    connection=work.substitute_connections[ocular.uid] if has_pedestal and substitute.enabled(ocular) else None
    if connection is not None:connection.update(cups=[],bores=[])
    u,r,f=frame
    d=port_data if port_data is not None else ports(work,item,center,base,frame,ocular,substitute_access=has_pedestal)
    frame=d['flow_frame'];u,r,f=frame
    xy=d['xy'];rear=dock['rear']-2.;neck=3.;h=item.cup_height
    if has_pedestal:
        end=Vector(blank['ocular_key_center_world'])
        if d['boss_top']>end.dot(u)-1.:
            raise RuntimeError('Not enough space beneath the ocular for separate side feed and vent. Reduce their bore sizes or move the mounting point; the ocular was not changed')
    else:
        root_start=min(center.dot(u)-1,dock['rear']+.2) if item.kind=='AIRWAY' else center.dot(u)-1
        boss=mi.cylinder(work,'Compact_feed_root',xy+u*root_start,xy+u*d['boss_top'],d['radius'],frame)
        blank=mi.merge(work,blank,boss)
    reserve=fill_ml*item.allowance/100
    capacities=[reserve,item.overflow_ml]
    radii=[d['feed'],d['vent']]
    tops=[max(rad+1.,(-rad+math.sqrt(max(0.,12*vol*1000/(math.pi*h)-3*rad*rad)))*.5)
          for rad,vol in zip(radii,capacities)]
    separation=tops[0]+tops[1]+2*d['wall']+1.
    funnels=[];paths=[];actual=[]
    for j,(rad,top) in enumerate(zip(radii,tops)):
        sign=-1 if j==0 else 1
        root=xy+r*(sign*d['distance']*.5)
        mouth=xy+r*(sign*separation*.5)+u*(rear-neck-h)
        throat=root+u*(rear-neck)
        centers=[mouth,throat,root+u*(rear+.5)]
        body=mi.loft(work,'Pour_cup' if j==0 else 'Overflow_cup',mi.round_rings(centers,[top+d['wall'],rad+d['wall'],rad+d['wall']],frame))
        if connection is not None:connection['cups'].append(work.copy(body,'Pedestal_cup_reference'))
        blank=mi.merge(work,blank,body)
        funnel=mi.loft(work,'Compact_cup_bore',mi.round_rings([mouth-u, mouth, throat,root+u*(rear+.7)],[top,top,rad,rad],frame))
        level=d['feed_z'] if j==0 else d['vent_z']
        # Independent elbow outlets on opposite sides immediately beside base.
        path=([root+u*(rear-neck-.2)]+d['vent_branch'] if j==1 else
              [root+u*(rear-neck-.2),root+u*level,xy+r*(sign*d['feed_exit'])+u*level])
        bore=tube(work,'Compact_feed_bore' if j==0 else 'Compact_air_bore',path,rad,frame)
        funnel=mi.merge(work,funnel,bore)
        if connection is not None:connection['bores'].append(work.copy(funnel,'Pedestal_bore_reference'))
        funnels.append(funnel);paths.append(path)
        actual.append(math.pi*h*(rad*rad+rad*top+top*top)/3000)
    if mi.intersection_volume(work,*funnels)>.00001:
        raise RuntimeError('Wax and air passages intersect; increase the common root wall or reduce bore sizes')
    blank['cast_preview_solid']=blank.data.copy()
    for tool in funnels:
        mi.aw.cut(blank,tool,'DIFFERENCE');work.delete(tool)
    mi.discard_boolean_dust(blank)
    mi.closed(blank,connected=True)
    tree=mi.aw.world_tree(blank)
    for path in paths:
        for a,b in zip(path,path[1:]):
            direction=b-a
            if tree.ray_cast(a,direction.normalized(),direction.length)[0] is not None:
                raise RuntimeError('Compact feed or vent is obstructed; adjust the mount or bore sizes')
    info=dict(fill_volume_ml=fill_ml,allowance_percent=item.allowance,
              reserve_target_ml=reserve,reservoir_capacity_ml=actual[0],overflow_capacity_ml=actual[1],
              mode='GRAVITY',layout='SIDE_CUPS',root_diameter_mm=2*d['radius'],
              feed_bore_mm=item.feed_diameter,vent_bore_mm=item.vent_diameter,
              fitting_orientation='CAP_DOWN_BASE_UP',vent_scope='LOCAL_PEDESTAL_ONLY',
              feed_path_world=[list(p) for p in paths[0]],vent_path_world=[list(p) for p in paths[1]],
              mouth_diameters_mm=[2*t for t in tops],passages_separate=True)
    blank['compact_reservoir_layout']='SIDE_CUPS'
    blank['compact_passages']=json.dumps([[list(p) for p in path] for path in paths])
    return blank,info
