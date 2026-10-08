"""Sequential base -> masked rear insert -> complementary cap construction.

Selection defines a projected insert footprint. Its upper interface follows
the selected rear heights and extends to the outside land for access. Anatomy
is subtracted from both parts, independently of that interface. Release is
checked separately: a mask or a normal test alone cannot establish demolding.
"""
import math,json
import bpy,bmesh,numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree
from ..utils import mesh as mu
from .report import rep
from .mold_auto import prism,heightfield_slab,convex_hull_2d,offset_polygon,dist_to_polygon,tube_along,rounded_slot
from .mold_design import OutlineSampler,interpolate_grid,has_self_intersections


def cut(target,cutter,operation):
    """Preserve the cavity: retry Boolean failure without remeshing the parts."""
    from .mold_breakable import clean_backing_topology
    for obj in (target,cutter):
        stats=mu.mesh_stats(obj)
        if any(stats[1:]):
            clean_backing_topology(obj);stats=mu.mesh_stats(obj)
            if any(stats[1:]):raise RuntimeError(f'{obj.name}: invalid Boolean input {stats}')
    saved=target.data.copy()
    try:
        for solver in ('MANIFOLD','EXACT'):
            if solver=='EXACT':
                old=target.data;target.data=saved.copy()
                if old.users==0:bpy.data.meshes.remove(old)
            mu.apply_boolean(target,cutter,operation,solver=solver)
            stats=mu.mesh_stats(target)
            if not any(stats[1:]):return
            clean_backing_topology(target)
            if not any(mu.mesh_stats(target)[1:]):return
        raise RuntimeError(f'{target.name}: Boolean did not form a closed part {mu.mesh_stats(target)}; no surface resampling was applied')
    finally:
        if saved.users==0:bpy.data.meshes.remove(saved)


def shade_part(obj,up):
    """Keep planar vertical walls flat; smooth curved surfaces across valid seams."""
    normal_matrix=obj.matrix_world.to_3x3().inverted().transposed()
    outer_faces=None;base=bpy.data.objects.get('Mold_Base')
    if base is not None and all(k in base for k in ('mold_hull','mold_skin','mold_land','mold_up','mold_right')):
        _,right,forward=frame_for(base)
        outline=offset_polygon([list(v) for v in base['mold_hull']],float(base['mold_skin'])+float(base['mold_land']))
        centers=np.array([obj.matrix_world@face.center for face in obj.data.polygons])
        if len(centers):
            xy=np.c_[centers@np.array(right),centers@np.array(forward)]
            outer_faces=np.abs(dist_to_polygon(xy,outline))<.003
    for face in obj.data.polygons:
        normal=(normal_matrix@face.normal).normalized()
        # Skinny Boolean triangles on the perimeter can have unreliable normals.
        # Identify these walls by position as well, without moving any vertices.
        face.use_smooth=abs(normal.dot(up))>=.03 and not (outer_faces is not None and outer_faces[face.index])
    obj.data.set_sharp_from_angle(angle=math.radians(35.))


def reference(p):
    from .shell import source_surface
    return source_surface(p)


def frame_for(base):
    u=Vector(base['mold_up']).normalized();r=Vector(base['mold_right']).normalized()
    r=(r-u*r.dot(u)).normalized();return u,r,u.cross(r).normalized()


def world_tree(obj):
    bm=bmesh.new();bm.from_mesh(obj.data);bm.transform(obj.matrix_world)
    try:return BVHTree.FromBMesh(bm)
    finally:bm.free()


def separate_islands(obj,col):
    """Keep one physical insert; return disconnected material for the next half."""
    bm=bmesh.new();bm.from_mesh(obj.data);bm.faces.ensure_lookup_table();bm.faces.index_update()
    seen=set();groups=[]
    for face in bm.faces:
        if face in seen:continue
        seen.add(face);stack=[face];group=[]
        while stack:
            face=stack.pop();group.append(face)
            for edge in face.edges:
                for other in edge.link_faces:
                    if other not in seen:seen.add(other);stack.append(other)
        groups.append(group)
    if len(groups)<2:bm.free();return None
    groups.sort(key=lambda fs:sum(f.calc_area() for f in fs),reverse=True)
    ids={face.index for group in groups[1:] for face in group}
    other=bm.copy();other.faces.ensure_lookup_table()
    bmesh.ops.delete(other,geom=[f for f in other.faces if f.index not in ids],context='FACES')
    bmesh.ops.delete(bm,geom=[f for f in bm.faces if f.index in ids],context='FACES')
    for mesh in (bm,other):
        orphan=[v for v in mesh.verts if not v.link_faces]
        if orphan:bmesh.ops.delete(mesh,geom=orphan,context='VERTS')
    bm.to_mesh(obj.data);bm.free();obj.data.update()
    result=mu.new_mesh_object('Auricular_Reassigned_work',other,col);result.matrix_world=obj.matrix_world.copy()
    return result


def remove_surface_slivers(obj,neighbors=()):
    """Remove disconnected numerical films without moving retained surfaces.

    Films below 0.001 mm are degenerate at this model scale. Films up to
    0.005 mm are removed only within 0.01 mm of a retained boundary.
    Substantial disconnected material is reassigned or rejected.
    """
    bm=bmesh.new();bm.from_mesh(obj.data);seen=set();groups=[]
    for face in bm.faces:
        if face in seen:continue
        seen.add(face);stack=[face];group=[]
        while stack:
            face=stack.pop();group.append(face)
            for edge in face.edges:
                for other in edge.link_faces:
                    if other not in seen:seen.add(other);stack.append(other)
        groups.append(group)
    if len(groups)<2:bm.free();return {'faces':0,'max_distance_mm':0.,'max_thickness_mm':0.}
    groups.sort(key=lambda fs:sum(f.calc_area() for f in fs),reverse=True)
    vertices=list({v for face in groups[0] for v in face.verts});indices={v:i for i,v in enumerate(vertices)}
    tree=BVHTree.FromPolygons([obj.matrix_world@v.co for v in vertices],[[indices[v] for v in face.verts] for face in groups[0]])
    trees=[tree]+[world_tree(other) for other in neighbors]
    discard=[];max_distance=0.;max_thickness=0.
    for group in groups[1:]:
        co=np.array([obj.matrix_world@v.co for v in {v for face in group for v in face.verts}],float)
        centered=co-co.mean(axis=0);_,_,axes=np.linalg.svd(centered,full_matrices=True)
        thickness=float(np.ptp(centered@axes[-1]))
        if thickness>.005:continue
        distance=max(min(t.find_nearest(Vector(q))[3] for t in trees) for q in co)
        if thickness>.001 and distance>.01:continue
        discard.extend(group);max_distance=max(max_distance,distance);max_thickness=max(max_thickness,thickness)
    count=len(discard)
    if discard:
        bmesh.ops.delete(bm,geom=discard,context='FACES')
        orphan=[v for v in bm.verts if not v.link_faces]
        if orphan:bmesh.ops.delete(bm,geom=orphan,context='VERTS')
        bm.to_mesh(obj.data);obj.data.update()
    bm.free();return {'faces':count,'max_distance_mm':max_distance,'max_thickness_mm':max_thickness}


def selected_points(p,frame):
    from .wedge_marking import selection_data
    return selection_data(p,frame)[0]


def largest_surface_region(obj,selected,weights):
    """Choose a connected surface by area, not by scan vertex density."""
    neighbors=[[] for _ in obj.data.vertices]
    for edge in obj.data.edges:
        a,b=edge.vertices
        if selected[a] and selected[b]:neighbors[a].append(b);neighbors[b].append(a)
    seen=np.zeros(len(selected),bool);best=[];best_area=0.
    for seed in np.flatnonzero(selected):
        if seen[seed]:continue
        seen[seed]=True;stack=[int(seed)];group=[]
        while stack:
            index=stack.pop();group.append(index)
            for other in neighbors[index]:
                if not seen[other]:seen[other]=True;stack.append(other)
        area=float(weights[group].sum())
        if area>best_area:best=group;best_area=area
    result=np.zeros(len(selected),bool);result[best]=True
    return result


def rear_undercut_suggestion(obj,base,draft,automatic=True,access_angle=0.):
    """Suggest the externally accessible rear patch, excluding inner ear folds.

    First detect undercuts in the opening direction. Automatic rear direction
    comes from the largest outward-facing, externally visible undercut surface.
    Then require visibility from that one rear direction and keep its connected
    patch. The editable source mask is never used as a selection side effect.
    """
    u,r,f=frame_for(base);tree=world_tree(obj)
    vertices=[obj.matrix_world@v.co for v in obj.data.vertices]
    normal_matrix=obj.matrix_world.to_3x3().inverted().transposed()
    normals=[(normal_matrix@v.normal).normalized() for v in obj.data.vertices]
    co=np.array(vertices);xy=np.c_[co@np.array(r),co@np.array(f)]
    center=np.mean(base['mold_hull'],axis=0) if 'mold_hull' in base else (xy.min(axis=0)+xy.max(axis=0))*.5
    radial=xy-center;length=np.linalg.norm(radial,axis=1);radial/=np.maximum(length[:,None],1e-9)
    travel=max(200.,float(np.linalg.norm(np.ptp(co,axis=0)))*3.)
    candidate=np.array([n.dot(u)<math.sin(draft) or tree.ray_cast(q+u*.02,u,travel)[0] is not None for q,n in zip(vertices,normals)])
    obj.data.calc_loop_triangles();weights=np.zeros(len(vertices),float)
    for tri in obj.data.loop_triangles:
        a,b,c=[vertices[i] for i in tri.vertices];area=(b-a).cross(c-a).length/6.
        for i in tri.vertices:weights[i]+=area
    if automatic:
        outside=np.zeros(len(vertices),bool)
        for index in np.flatnonzero(candidate):
            direction=r*float(radial[index,0])+f*float(radial[index,1])
            if length[index]>.01 and normals[index].dot(direction)>.02:
                outside[index]=tree.ray_cast(vertices[index]+direction*.02,direction,travel)[0] is None
        seed=largest_surface_region(obj,outside,weights)
        if seed.sum()<6:raise RuntimeError('No connected outer undercut was found; set the rear access direction manually or paint the rear region')
        direction=np.average(xy[seed]-center,axis=0,weights=weights[seed])
        if np.linalg.norm(direction)<.01:raise RuntimeError('The rear direction is ambiguous; set the access direction manually')
        access_angle=math.atan2(direction[1],direction[0])
    exit=r*math.cos(access_angle)+f*math.sin(access_angle)
    visible=np.zeros(len(vertices),bool)
    for index in np.flatnonzero(candidate):
        if normals[index].dot(exit)>.02:
            visible[index]=tree.ray_cast(vertices[index]+exit*.02,exit,travel)[0] is None
    straight=visible.copy()
    side=u.cross(exit).normalized()
    rear_directions=[(exit-u).normalized(),(exit+side*.5-u*.5).normalized(),(exit-side*.5-u*.5).normalized()]
    for index in np.flatnonzero(candidate&~visible):
        radial_world=r*float(radial[index,0])+f*float(radial[index,1])
        # Keep the rear-facing envelope; the downward views recover the
        # underside of the helix without opening selection into the concha.
        if normals[index].dot(exit)<-.05 or normals[index].dot(radial_world)<-.05:continue
        if (xy[index]-center)@np.array([math.cos(access_angle),math.sin(access_angle)])<0:continue
        for direction in rear_directions:
            if normals[index].dot(direction)>.02 and tree.ray_cast(vertices[index]+direction*.02,direction,travel)[0] is None:
                visible[index]=True;break
    selected=largest_surface_region(obj,visible,weights)
    if selected.sum()<6:raise RuntimeError('No rear undercut is visible in that direction; adjust the access direction or paint the rear region')
    return selected.astype(np.float32),{'access_angle':float(access_angle),'all_undercut_vertices':int(candidate.sum()),'rear_vertices':int(selected.sum()),'hidden_rear_vertices':int((selected&~straight).sum()),'rear_area_mm2':float(weights[selected].sum())}


def suggest(context):
    p=context.scene.anaplast;o=reference(p);base=bpy.data.objects.get('Mold_Base')
    if not o or not base:raise RuntimeError('Assign the sculpt and build the base first')
    if context.object and context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    values,details=rear_undercut_suggestion(o,base,p.wedge_draft,p.wedge_auto_exit,p.wedge_exit_angle)
    old=p.wedge_suggestion
    copy=mu.duplicate_object(o,'Wedge_Undercut_Suggestion',mu.get_collection(context.scene))
    a=copy.data.attributes.get('.sculpt_mask') or copy.data.attributes.new('.sculpt_mask','FLOAT','POINT')
    a.data.foreach_set('value',values);copy.data.update();copy.color=(.8,.25,.12,1)
    p.wedge_suggestion=copy;p.wedge_selection='AUTO'
    if old:mu.delete_object(old)
    copy.name='Wedge_Undercut_Suggestion'
    copy['rear_suggestion_report']=json.dumps(details)
    for ob in context.selected_objects:ob.select_set(False)
    o.hide_set(True);copy.hide_set(False);copy.select_set(True);context.view_layer.objects.active=copy
    bpy.ops.object.mode_set(mode='SCULPT')
    p.wedge_report=f'{int(values.sum()):,} rear vertices suggested. Review/edit this copy in Sculpt Mode.'
    return copy


def closed_reference(source,col,voxel,up,floor):
    obj=mu.duplicate_object(source,'Auricular_Solid_tmp',col)
    bm=bmesh.new();bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    edges=[e for e in bm.edges if e.is_boundary]
    if edges:
        # Extend only the boundary, never duplicate the whole ear downward.
        # Capping the warped original rim directly crosses the rear anatomy.
        result=bmesh.ops.extrude_edge_only(bm,edges=edges)
        for v in result['geom']:
            if isinstance(v,bmesh.types.BMVert):v.co+=up*(floor-v.co.dot(up))
        result=bmesh.ops.holes_fill(bm,edges=[e for e in bm.edges if e.is_boundary],sides=0)
        bmesh.ops.triangulate(bm,faces=list(result['faces']))
    bmesh.ops.recalc_face_normals(bm,faces=bm.faces);bm.transform(obj.matrix_world.inverted());bm.to_mesh(obj.data);bm.free()
    repaired=0.
    if not mu.is_closed(obj) or has_self_intersections(obj):
        repaired=mu.voxel_remesh(obj,voxel)
    if not mu.is_closed(obj):
        mu.delete_object(obj);raise RuntimeError('The sculpt could not be closed for the three-piece cavity')
    return obj,repaired


def release_samples(part,obstacles,up,travel):
    """Conservative sampled straight-pull obstruction test; no deformation model."""
    trees=[world_tree(o) for o in obstacles];blocked=0;total=0
    vertices=part.data.vertices;stride=max(1,len(vertices)//1800)
    for index in range(0,len(vertices),stride):
        v=vertices[index]
        q=part.matrix_world@v.co+up*.05;total+=1
        if any(tree.ray_cast(q,up,travel)[0] is not None for tree in trees):blocked+=1
    return {'blocked_samples':blocked,'samples':total,'travel_mm':travel,'status':'BLOCKED' if blocked else 'NO_SAMPLED_OBSTRUCTION'}


def shortest_end_outline(points,outline,radial,tangent):
    """Each end under the helix connects to its nearest outer rim point."""
    xy=points[:,:2];longitudinal=xy@tangent
    ends=[xy[np.argmin(longitudinal)],xy[np.argmax(longitudinal)]]
    outer=np.array(outline,float);connections=[]
    inside=xy.mean(axis=0)+radial*8.
    result=outer.copy()
    for end in ends:
        edges=np.roll(outer,-1,axis=0)-outer
        t=np.clip(np.einsum('ij,ij->i',end-outer,edges)/np.maximum(np.einsum('ij,ij->i',edges,edges),1e-12),0,1)
        feet=outer+edges*t[:,None];nearest=feet[np.argmin(np.linalg.norm(feet-end,axis=1))]
        line=nearest-end
        side=lambda q:float(line[0]*(q[1]-end[1])-line[1]*(q[0]-end[0]))
        sign=1. if side(inside)>=0 else -1.
        clipped=[]
        for a,b in zip(result,np.roll(result,-1,axis=0)):
            da,db=side(a)*sign,side(b)*sign
            if da>=-1e-8:clipped.append(a)
            if (da>=0)!=(db>=0):clipped.append(a+(b-a)*da/(da-db))
        result=np.array(clipped)
        if len(result)<3:raise RuntimeError('The selected rear ends do not define a usable wedge region')
        connections.append({'inner_xy':end.tolist(),'outer_xy':nearest.tolist(),'distance_mm':float(np.linalg.norm(nearest-end))})
    return result.tolist(),connections


def profile_interp(values_at,stations,values):
    """C1, shape-preserving section interpolation without angular strip seams."""
    x=np.asarray(stations,float);y=np.asarray(values,float);h=np.diff(x);delta=np.diff(y)/h;d=np.zeros(len(y))
    a,b=delta[:-1],delta[1:];same=a*b>0
    w1=2*h[1:]+h[:-1];w2=h[1:]+2*h[:-1]
    denominator=w1*np.where(same,b,1.)+w2*np.where(same,a,1.)
    d[1:-1]=np.where(same,(w1+w2)*a*b/np.where(abs(denominator)>1e-15,denominator,1.),0.)
    q=np.clip(np.asarray(values_at,float),x[0],x[-1]);i=np.clip(np.searchsorted(x,q,side='right')-1,0,len(x)-2);t=(q-x[i])/h[i]
    return (2*t**3-3*t*t+1)*y[i]+(t**3-2*t*t+t)*h[i]*d[i]+(-2*t**3+3*t*t)*y[i+1]+(t**3-t*t)*h[i]*d[i+1]


def anatomical_end_boundary(connection,points,inside,frame,source_tree,base_tree,gx,gy,roof,outline,floor,ceiling,grid,col):
    """Continue the terminal under-ear angle from the roof toward the base.

    The roof boundary keeps its shortest connection to the rounded rim. The
    bottom extends outward below that line, rather than cutting the whole
    crescent inward from a hinge on the base. Only a dividing surface is fitted;
    the original sculpt remains the cavity cutter.
    """
    u,r,f=frame;end=np.asarray(connection['inner_xy'],float);outer=np.asarray(connection['outer_xy'],float)
    along=outer-end;along/=np.linalg.norm(along);inward=np.array([-along[1],along[0]])
    if inward@(np.asarray(inside)-end)<0:inward=-inward
    origin=points[np.argmin(np.linalg.norm(points[:,:2]-np.asarray(connection.get('anatomical_inner_xy',end)),axis=1))]
    local=points[np.linalg.norm(points-origin,axis=1)<=7.]
    base_point=r*float(end[0])+f*float(end[1])+u*(ceiling+10.)
    base_normal=base_tree.ray_cast(base_point,-u,ceiling-floor+30.)[1]
    if base_normal is None:base_normal=u
    angles=[];weights=[]
    for q in local:
        n=source_tree.find_nearest(r*float(q[0])+f*float(q[1])+u*float(q[2]))[1]
        if n is None:continue
        weights.append(abs(n.dot(u))**4)
        angles.append(math.acos(min(1.,abs(n.dot(base_normal.normalized())))))
    if not weights or sum(weights)<1e-6:raise RuntimeError('Include more of the under-helix or lobe surface at the end of the wedge selection')
    angle=float(np.average(angles,weights=weights))
    if angle<math.radians(5.):raise RuntimeError('The selected terminal surface is nearly parallel to the base; extend the selection around its end transition')
    cot=1./math.tan(angle)
    # A ruled surface along the shortest connector follows the already fair
    # roof. Its height origin is the roof, not the lower fitting surface.
    extent=np.asarray(outline)@along
    stations=np.arange(extent.min()-5.,extent.max()+5.+grid,grid)
    xy=stations[:,None]*along+float(end@inward)*inward
    anchor=interpolate_grid(gx,gy,roof,xy[:,0],xy[:,1],float(origin[2]))
    anchor=np.clip(anchor,floor-4.,ceiling+4.)
    S=np.tile(stations,(3,1));Z=np.vstack([np.full_like(anchor,floor-5.),anchor,np.full_like(anchor,ceiling+5.)])
    # A terminal extension must never cut back into the existing crescent.
    # Above the connector's roof height keep the original end boundary; the
    # inclined portion belongs below it, toward the fitting surface.
    D=float(end@inward)+cot*np.minimum(Z-anchor[None,:],0.)
    normal=r*float(inward[0])+f*float(inward[1]);right=r*float(along[0])+f*float(along[1])
    result=heightfield_slab('Auricular_AnatomicalEnd_work',S,Z,D,np.ones_like(D,dtype=bool),-1000.,(normal,right,u),col)
    connection['end_angle_degrees']=math.degrees(angle)
    connection['end_anchor']='ROOF_SHORTEST_CONNECTOR'
    connection['end_extension']='OUTWARD_TOWARD_BASE'
    return result


def continued_rear_roof(points,radial,tangent,X,Y):
    """Continue local under-helix sections as ruled planes to the outer rim.

    Equal radial bins avoid bias from uneven scan triangulation. Robust fits
    use the middle of the marked rear surface, excluding its folded rim.
    Only the divider is fitted; the original sculpt still supplies the cavity.
    """
    rho=points[:,:2]@radial;long=points[:,:2]@tangent
    stations=np.linspace(float(long.min()),float(long.max()),max(12,int(np.ptp(long)/1.5)+1))
    anchors=[];levels=[];slopes=[]
    for station in stations:
        choose=np.abs(long-station)<=3.
        if choose.sum()<12:
            order=np.argsort(np.abs(long-station));choose=np.zeros(len(long),bool);choose[order[:min(100,len(long))]]=True
        rr=rho[choose];zz=points[choose,2]
        low,high=np.percentile(rr,[25,85]);anchor=high
        bins=np.arange(low,high+.5,.5);xx=[];yy=[]
        for left in bins:
            inside=(rr>=left)&(rr<left+.5)
            if inside.sum()>=3:xx.append(float(np.median(rr[inside]))-anchor);yy.append(float(np.median(zz[inside])))
        if len(xx)<3 or np.ptp(xx)<1.:
            raise RuntimeError('The marked rear region is too narrow to measure its angle; include more of the surface under the helix')
        xx=np.array(xx);yy=np.array(yy);A=np.c_[xx,np.ones(len(xx))];weight=np.ones(len(xx))
        for _ in range(5):
            fit=np.linalg.lstsq(A*weight[:,None]**.5,yy*weight**.5,rcond=None)[0]
            error=np.abs(yy-A@fit);scale=max(.05,float(np.median(error))*1.4826);weight=np.minimum(1.,1.5*scale/np.maximum(error,1e-9))
        anchors.append(anchor);slopes.append(float(fit[0]));levels.append(float(fit[1]))
    # Smooth between adjacent sections, without flattening their radial angle.
    for values in (anchors,levels,slopes):
        for _ in range(3):
            old=values[:]
            for i in range(1,len(values)-1):values[i]=(old[i-1]+2*old[i]+old[i+1])/4
    target_t=X*tangent[0]+Y*tangent[1];target_r=X*radial[0]+Y*radial[1]
    anchor=profile_interp(target_t,stations,anchors);level=profile_interp(target_t,stations,levels);slope=profile_interp(target_t,stations,slopes)
    roof=level+slope*(target_r-anchor)
    report=[{'station_mm':float(t),'anchor_mm':float(a),'height_mm':float(h),'slope':float(m),'roof_angle_degrees':float(math.degrees(math.atan(m)))} for t,a,h,m in zip(stations,anchors,levels,slopes)]
    return roof,report


def fair_sections(stations,values,smoothing):
    """Broad local linear regression removes ripples while retaining trends."""
    if smoothing<=0:return np.asarray(values,float)
    x=np.asarray(stations,float);y=np.asarray(values,float);result=[]
    for station in x:
        d=x-station;weights=np.exp(-.5*(d/smoothing)**2);weights[np.abs(d)>3*smoothing]=0.
        A=np.c_[np.ones(len(x)),d];fit=np.linalg.lstsq(A*weights[:,None]**.5,y*weights**.5,rcond=None)[0];result.append(fit[0])
    return np.asarray(result)


def ease_rear_roof(straight,sections,radial,tangent,X,Y,gx,gy,ground,outline,transition,smoothing=0.):
    """Match the marked angle, then ease to a locally parallel roof.

    Integrating a smoothstep falloff gives matching slope and curvature at
    both ends. The final gap follows each section's measured starting gap;
    it is not forced to one global vertical height.
    """
    stations=np.array([s['station_mm'] for s in sections]);anchors=np.array([s['anchor_mm'] for s in sections])
    levels=np.array([s['height_mm'] for s in sections]);slopes=np.array([s['slope'] for s in sections])
    ax=anchors*radial[0]+stations*tangent[0];ay=anchors*radial[1]+stations*tangent[1]
    base=interpolate_grid(gx,gy,ground,ax,ay,float(np.median(ground)))
    before=interpolate_grid(gx,gy,ground,ax-radial[0]*3.,ay-radial[1]*3.,base)
    after=interpolate_grid(gx,gy,ground,ax+radial[0]*3.,ay+radial[1]*3.,base)
    base_slopes=(after-before)/6.
    relative=fair_sections(stations,slopes-base_slopes,smoothing);gap=fair_sections(stations,levels-base,smoothing)
    base=fair_sections(stations,base,smoothing);base_slopes=fair_sections(stations,base_slopes,smoothing)
    anchors=fair_sections(stations,anchors,smoothing)
    poly=np.array(outline);rs=poly@radial;ts=poly@tangent;lengths=[]
    for station,anchor in zip(stations,anchors):
        hits=[]
        for i in range(len(poly)):
            j=(i+1)%len(poly)
            if (ts[i]<=station<ts[j]) or (ts[j]<=station<ts[i]):hits.append(rs[i]+(rs[j]-rs[i])*(station-ts[i])/(ts[j]-ts[i]))
        available=max(hits)-anchor if hits else transition
        lengths.append(max(.5,min(transition,max(.5,available*.85))))
    target_t=X*tangent[0]+Y*tangent[1];target_r=X*radial[0]+Y*radial[1]
    anchor=profile_interp(target_t,stations,anchors);g0=profile_interp(target_t,stations,gap)
    slope=profile_interp(target_t,stations,relative);length=profile_interp(target_t,stations,lengths)
    distance=target_r-anchor;t=np.clip(distance/length,0.,1.)
    integral=t-t**3+.5*t**4
    curved=ground+g0+slope*length*integral
    for section,g,m,L in zip(sections,gap,relative,lengths):
        section['starting_gap_mm']=float(g);section['relative_slope']=float(m);section['transition_mm']=float(L);section['outer_gap_mm']=float(g+m*L*.5)
    inner=profile_interp(target_t,stations,base)+g0+(slope+profile_interp(target_t,stations,base_slopes))*distance
    # A short smooth blend avoids introducing a seam where the fitted inner
    # base reference meets the reconstructed land used by the outer roof.
    blend=np.clip(distance/3.,0.,1.);blend=blend**3*(10-15*blend+6*blend**2)
    reconstructed_base=profile_interp(target_t,stations,base)+profile_interp(target_t,stations,base_slopes)*distance
    curved+=(reconstructed_base-ground)*(1-blend)
    return np.where(distance<0,inner,curved)


def rear_sweep(points,radial,tangent,frame,margin,overlap,grid,col,floor,outline,ceiling=None,connections=(),required_points=None):
    """Ruled rear insert: extend the selected rear surface toward the rim.

    Its footprint is measured from the rear, not flattened across the ear's
    front. Subtraction of the unchanged ear supplies the curved inner face.
    """
    u,r,f=frame;exit=r*float(radial[0])+f*float(radial[1]);side=u.cross(exit)
    yz=np.c_[points[:,:2]@tangent,points[:,2]];depth=points[:,:2]@radial
    profile=offset_polygon(convex_hull_2d(yz),margin)
    bounds=np.array(profile)
    extent=np.array(outline)@tangent
    gy=np.arange(min(extent.min(),bounds[:,0].min())-grid,max(extent.max(),bounds[:,0].max())+2*grid,grid)
    gz=np.arange(min(floor,bounds[:,1].min()-grid),max(bounds[:,1].max(),ceiling or bounds[:,1].max())+2*grid,grid)
    Y,Z=np.meshgrid(gy,gz)
    # Extend the two end profiles until the shortest connector planes meet the
    # rim. They are not forced to travel along the common rearward direction.
    sample_y=np.clip(Y,yz[:,0].min()+grid*.25,yz[:,0].max()-grid*.25)
    mask=dist_to_polygon(np.c_[sample_y.ravel(),Z.ravel()],profile).reshape(Y.shape)<=grid
    kd=KDTree(len(points))
    for i,q in enumerate(yz):kd.insert((float(q[0]),float(q[1]),0),i)
    kd.balance();D=np.empty_like(Y)
    for i,(y,z) in enumerate(zip(sample_y.ravel(),Z.ravel())):
        near=kd.find_n((y,z,0),6);weights=[1/max(d,.2)**2 for _,_,d in near]
        D.flat[i]=np.average([depth[j] for _,j,_ in near],weights=weights)-overlap
    for _ in range(12):
        a=np.pad(D,1,mode='edge');D=(a[:-2,1:-1]+a[2:,1:-1]+a[1:-1,:-2]+a[1:-1,2:]+4*D)/8
    # Follow the scan's anterior slope into the base. Extending a constant
    # radial column here would create the unwanted 90-degree front wall.
    lower_slopes=[];upper_slopes=[];lower_rows=[];upper_rows=[]
    for j in range(D.shape[1]):
        rows=np.flatnonzero(mask[:,j])
        if len(rows)<3:raise RuntimeError('The rear selection is too thin to determine its anterior slope')
        lo,hi=int(rows[0]),int(rows[-1]);count=min(len(rows),max(4,int(4./grid)))
        low=rows[:count];high=rows[-count:]
        station=float(np.clip(gy[j],yz[:,0].min(),yz[:,0].max()))
        near=np.abs(yz[:,0]-station)<=3.
        zz=yz[near,1];dd=depth[near]
        zfirst,zlast=np.percentile(zz,[10,45]);fit=(zz>=zfirst)&(zz<=zlast)
        if fit.sum()>=6 and np.ptp(zz[fit])>.5:
            lower_slopes.append(float(np.polyfit(zz[fit]-np.mean(zz[fit]),dd[fit],1)[0]))
        else:lower_slopes.append(float(np.polyfit(gz[low]-gz[lo],D[low,j],1)[0]))
        upper_slopes.append(float(np.polyfit(gz[high]-gz[hi],D[high,j],1)[0]))
        lower_rows.append(lo);upper_rows.append(hi)
    lower_slopes=fair_sections(gy,lower_slopes,3.);upper_slopes=fair_sections(gy,upper_slopes,3.)
    for j,(lo,hi) in enumerate(zip(lower_rows,upper_rows)):
        # Follow the actual selected surface within its measured range. Only
        # the continuation below that range uses the measured anterior slope.
        D[:lo,j]=D[lo,j]+lower_slopes[j]*(gz[:lo]-gz[lo])
        D[hi+1:,j]=D[hi,j]+upper_slopes[j]*(gz[hi+1:]-gz[hi])
    # Conservative cell bounds cover selected points even where scan density
    # is uneven. These constraints act on the internal divider, not anatomy.
    constraint_points=points if required_points is None else required_points
    constraint_yz=np.c_[constraint_points[:,:2]@tangent,constraint_points[:,2]]
    constraint_depth=constraint_points[:,:2]@radial
    iy=np.clip(((constraint_yz[:,0]-gy[0])/grid).astype(int),0,len(gy)-2)
    iz=np.clip(((constraint_yz[:,1]-gz[0])/grid).astype(int),0,len(gz)-2)
    constraint=np.full_like(D,np.inf)
    for dy,dz in ((0,0),(1,0),(0,1),(1,1)):
        np.minimum.at(constraint,(iz+dz,iy+dy),constraint_depth-overlap)
    # Fair an outward-enclosing divider rather than locally denting grid
    # corners to reach marks. This acts only on the constructed partition;
    # the original sculpt is subtracted afterward to define the cavity.
    sigma=max(1.,grid*2)/grid;radius=int(math.ceil(sigma*3))
    kernel=np.exp(-.5*(np.arange(-radius,radius+1)/sigma)**2);kernel/=kernel.sum()
    D=np.minimum(D,constraint)
    for axis in (0,1):
        D=np.apply_along_axis(lambda row:np.convolve(np.pad(row,(radius,radius),mode='edge'),kernel,mode='valid'),axis,D)
    while True:
        deficit=D-constraint;index=np.unravel_index(np.argmax(deficit),D.shape);amount=float(deficit[index])
        if amount<=1e-5:break
        j,i=index;dy=(np.arange(len(gz))-j)/sigma;dx=(np.arange(len(gy))-i)/sigma
        D-=(amount+1e-5)*np.exp(-.5*dy*dy)[:,None]*np.exp(-.5*dx*dx)[None,:]
    # The anterior divider may follow the scan below the helix, but must
    # never lean outward through the free land and clip keys or the mold rim.
    for j,station in enumerate(gy):
        near=np.abs(yz[:,0]-np.clip(station,yz[:,0].min(),yz[:,0].max()))<=2.
        D[:,j]=np.minimum(D[:,j],float(depth[near].max()))
    # Beyond the marked ends, follow their nearest-rim connectors instead of
    # continuing a constant radial column that misses part of the outer edge.
    for connection in connections:
        inner=np.asarray(connection['inner_xy']);outer=np.asarray(connection['outer_xy'])
        ti=float(inner@tangent);to=float(outer@tangent)
        if abs(to-ti)<1e-6:continue
        progress=np.clip((gy-ti)/(to-ti),0.,1.)
        D+=np.minimum(0.,float((outer-inner)@radial))*progress[None,:]
    mask[:]=True
    result=heightfield_slab('Auricular_RearSweep_work',Y,Z,D,mask,-200.,(exit,side,u),col)
    result['anterior_angles_degrees']=[float(math.degrees(math.atan2(1.,slope))) for slope in lower_slopes]
    return result


def build(context):
    if context.scene.get('anaplast_insert_state'):
        raise RuntimeError('Restore the original mold in Interchangeable inserts before rebuilding mold parts')
    from . import mold_volume
    volume_ledger = mold_volume.Ledger()
    from .explode import ensure_collapsed
    p=context.scene.anaplast;ensure_collapsed(context.scene)
    for obj in list(context.scene.objects):
        if obj.get('wedge_coverage_preview'):obj.hide_set(True)
    original_base=bpy.data.objects.get('Mold_Base');original_cap=bpy.data.objects.get('Mold_Cap')
    if not original_base:raise RuntimeError('Build the base first')
    prepared=bpy.data.objects.get('Auricular_CapBlank_Reference')
    if not prepared:raise RuntimeError('Use step 1 to build the base with this version')
    input_base,input_cap=bpy.data.objects.get('Common_Base_Reference') or original_base,prepared
    if original_base.get('three_piece'):
        input_base=bpy.data.objects.get('Auricular_Base_Reference')
        if not input_base:raise RuntimeError('Rebuild the base before changing the wedge')
    if p.m_bolts:raise RuntimeError('Clamp-bolt holes are not supported by the shared three-piece layout yet; set Clamp bolts to 0')
    if not input_base.get('auricular_plain_layout'):raise RuntimeError('Rebuild step 1: Base once with this version to replace the old overlapping feature layout')
    source=reference(p);frame=frame_for(input_base);u,r,f=frame
    from .wedge_marking import selection_data,restore_view
    points,required_points,border_info=selection_data(p,frame)
    restore_view(context)
    construction_points=required_points
    hull=[list(v) for v in original_base['mold_hull']]
    skin=float(original_base['mold_skin']);land=float(original_base['mold_land']);outline=offset_polygon(hull,skin+land)
    if dist_to_polygon(points[:,:2],hull).min()>skin:raise RuntimeError('The wedge selection is outside this mold')
    center=np.mean(hull,axis=0);direction=points[:,:2].mean(axis=0)-center
    angle=math.atan2(direction[1],direction[0]) if p.wedge_auto_exit else p.wedge_exit_angle
    radial=np.array([math.cos(angle),math.sin(angle)]);tangent=np.array([-radial[1],radial[0]])
    sampler=OutlineSampler(hull);edge=np.array(sampler.point(angle,skin+land+3.))
    from . import wedge_coverage
    poly,end_connections=wedge_coverage.resolve(p,construction_points,outline,radial,tangent)
    allco=np.array([original_base.matrix_world@v.co for v in original_base.data.vertices]);capco=np.array([prepared.matrix_world@v.co for v in prepared.data.vertices])
    zlo=float(min((allco@np.array(u)).min(),(capco@np.array(u)).min()))
    zhi=float(max((allco@np.array(u)).max(),(capco@np.array(u)).max()))
    col=mu.get_collection(context.scene);temps=[]
    def keep(obj):temps.append(obj);return obj
    base=keep(mu.duplicate_object(input_base,'Auricular_Base_work',col))
    cap=keep(mu.duplicate_object(input_cap,'Auricular_Cap_work',col))
    cap_cutters=[]
    def cut(target,cutter,operation):
        if target==cap:
            if operation!='DIFFERENCE':raise RuntimeError('Unsupported deferred cap operation')
            copy=keep(mu.duplicate_object(cutter,'Auricular_DeferredCut_work',col));cap_cutters.append(copy)
            return
        return globals()['cut'](target,cutter,operation)
    try:
        mold_volume.clear(context.scene,'Build the cap to finish the three-piece volume calculation')
        from .mold_auricular_layout import end_roof,make_plan,key_relief,add_mating_relief,cover_marked_roof
        prepared=keep(mu.duplicate_object(prepared,'Auricular_Prepared_work',col))
        solid,repair=closed_reference(source,col,p.m_voxel,u,zlo-5.);keep(solid)
        solid['silicone_volume_role']='cavity'
        print(f'Auricular: cavity reference closed; repair spacing {repair:.3f} mm',flush=True)
        # Both parts start from one cached mold volume with its matched keys,
        # ring, spillways and pry cuts, before the two-part swept cavity.
        block=keep(prism('Auricular_Block_work',outline,zlo,zhi,frame,col))
        old_mesh=cap.data;cap.data=prepared.data.copy();cap.matrix_world=prepared.matrix_world.copy()
        if old_mesh.users==0:bpy.data.meshes.remove(old_mesh)
        cut(cap,solid,'DIFFERENCE')
        print('Auricular: full cavity restored, forming rear interface',flush=True)
        grid=min(p.m_grid,.3 if p.m_key_type=='WEDGE' else .6);op=np.array(outline)
        gx=np.arange(op[:,0].min()-grid,op[:,0].max()+2*grid,grid);gy=np.arange(op[:,1].min()-grid,op[:,1].max()+2*grid,grid)
        X,Y=np.meshgrid(gx,gy);xy=np.c_[X.ravel(),Y.ravel()]
        mask=np.ones_like(X,dtype=bool)
        base_tree=world_tree(base);Hbase=np.full_like(X,float(original_base['mold_plane_h']));Hvalid=np.zeros_like(X,dtype=bool)
        for index,q in enumerate(xy):
            loc=base_tree.ray_cast(r*float(q[0])+f*float(q[1])+u*(zhi+10),-u,zhi-zlo+30)[0]
            if loc is not None:Hbase.flat[index]=loc.dot(u);Hvalid.flat[index]=True
        # Extend the measured land beyond the rim for open-ended cutters.
        # Falling back to a different plane outside the mesh folds the pry
        # tool downward at the rim and can leave a detached cap lip.
        if not Hvalid.any():raise RuntimeError('Cannot sample the base mating surface')
        while not Hvalid.all():
            a=np.pad(np.where(Hvalid,Hbase,0.),1);v=np.pad(Hvalid.astype(float),1)
            total=a[:-2,1:-1]+a[2:,1:-1]+a[1:-1,:-2]+a[1:-1,2:]
            count=v[:-2,1:-1]+v[2:,1:-1]+v[1:-1,:-2]+v[1:-1,2:]
            fill=(~Hvalid)&(count>0);Hbase[fill]=total[fill]/count[fill];Hvalid[fill]=True
        height,roof_sections=continued_rear_roof(points,radial,tangent,X,Y)
        dh=dist_to_polygon(xy,hull).reshape(X.shape)
        from .mold_design import flat_land
        # Reconstruct the broad land without copying keys or grooves into the
        # wedge roof. This does not modify the actual fitting/base mesh.
        ground,_=flat_land(gx,gy,Hbase,dh,np.ones_like(mask),offset_polygon(hull,skin),skin,float(original_base.get('mold_land_transition',p.m_land_transition)),float(original_base.get('mold_land_follow',p.m_land_follow)))
        height=ease_rear_roof(height,roof_sections,radial,tangent,X,Y,gx,gy,ground,poly,p.wedge_transition,p.wedge_top_smoothing)
        # The unchanged sculpt cuts the contact face. Inside the anatomy only,
        # overlap keeps the dividing surface safely within the sculpt volume.
        height+=p.wedge_height_offset*np.clip(-dh/2.,0.,1.)
        roof_lift=0.
        source_tree=world_tree(source)
        height=end_roof(height,X,Y,end_connections,points,points[:,:2].mean(axis=0)+radial*8.,frame,source_tree,p.wedge_end_blend)
        height,roof_lift=cover_marked_roof(height,gx,gy,construction_points,p.wedge_height_offset,p.wedge_top_smoothing,radial)
        region=dist_to_polygon(xy,poly).reshape(X.shape)<=0
        required_top=float(np.max(height[region]))+max(3.,p.m_cap_height)
        if required_top>zhi:
            extended=keep(mu.duplicate_object(prepared,'Auricular_ExtendedBlank_work',col))
            # The blank already has a planar outer top. Extend its existing
            # walls rather than unioning a second block with coplanar sides,
            # which can leave overlapping films on the cap perimeter.
            world=np.array([extended.matrix_world@v.co for v in extended.data.vertices])
            levels=world@np.array(u);top=float(levels.max());top_vertices=np.flatnonzero(levels>=top-.0001)
            inv=extended.matrix_world.inverted()
            for index in top_vertices:
                q=Vector(world[index])+u*(required_top-float(levels[index]));extended.data.vertices[int(index)].co=inv@q
            extended.data.update()
            if not len(top_vertices) or any(mu.mesh_stats(extended)[1:]):raise RuntimeError('The cap blank top could not be extended as a closed solid')
            prepared=extended;zhi=required_top
        cap['extended_top_mm']=zhi
        print('Auricular: planning both interfaces',flush=True)
        height_plain=height.copy()
        footprint=keep(rear_sweep(points,radial,tangent,frame,p.wedge_margin,p.wedge_height_offset,grid,col,zlo-1.,poly,ceiling=zhi+1.,connections=end_connections,required_points=construction_points))
        access=keep(prism('Auricular_Access_work',offset_polygon(outline,.5),zlo-1,zhi+1,frame,col));cut(footprint,access,'INTERSECT')
        # Use one continuous divider through the anatomical junction and land.
        # A separate prism outside the anatomical hull makes a vertical sheet
        # at the hull intersection, visible as a thin fin at the lobe.
        plan=make_plan(p,hull,outline,frame,gx,gy,Hbase,height_plain,world_tree(footprint),angle,poly)
        keys=plan['keys'];upper_keys=[k for k in keys if k['on_wedge']]
        bullet=p.m_key_type=='WEDGE';key_sign=-1. if bullet else 1.
        plan['base_key_vertices']=add_mating_relief(base,frame,gx,gy,Hbase,keys,p.key_clearance if bullet else 0.,key_sign)
        plan['socket_vertices']=add_mating_relief(prepared,frame,gx,gy,Hbase,keys,0. if bullet else p.key_clearance,key_sign)
        print('Auricular: shared layout',json.dumps(plan),flush=True)
        def point_at(a,d,field):
            x,y=sampler.point(a,d)
            return r*x+f*y+u*float(interpolate_grid(gx,gy,field,x,y,zlo))
        def channel(parts,pts,closed=False,guard=None):
            tool=keep(tube_along('Auricular_Channel_work',pts,p.m_spillway_width*.5,col,normal=u,closed=closed))
            tool['silicone_volume_role']='channels'
            if guard is not None:cut(tool,guard,'INTERSECT')
            for part in parts:
                if part==cap:cut(part,tool,'DIFFERENCE')
                else:volume_ledger.cut(part,tool,'DIFFERENCE',cut,'channels')
        def pry(parts,item,field):
            from .mold_design import contoured_pry_cutter
            tool=keep(contoured_pry_cutter('Auricular_Pry_work',item['rim'],item['outward'],item['tangent'],
                item['depth_mm'],p.m_pry_width,p.m_pry_height,frame,gx,gy,field,zlo,col))
            for part in parts:cut(part,tool,'DIFFERENCE')
        if p.m_ring:
            channel((base,prepared),[point_at(a,plan['ring_distance'],Hbase) for a in np.linspace(0,2*math.pi,512,endpoint=False)],True)
        for item in plan['outlets']:
            channel((base,prepared),[point_at(item['angle'],d,Hbase) for d in np.arange(plan['ring_distance'],skin+land+3.,.5)])
        for item in plan['pries']:pry((base,prepared),item,Hbase)
        height=height_plain+key_sign*key_relief(X,Y,upper_keys,p.key_clearance if bullet else 0.)
        blank=keep(heightfield_slab('Auricular_WedgeBlank_work',X,Y,height,mask,zhi-zlo+50,frame,col))
        cut(blank,footprint,'INTERSECT')
        wedge=keep(mu.duplicate_object(prepared,'Auricular_Wedge_work',col))
        cut(wedge,blank,'INTERSECT');cut(wedge,solid,'DIFFERENCE')
        upper=keep(heightfield_slab('Auricular_UpperSocket_work',X,Y,height_plain+key_sign*key_relief(X,Y,upper_keys,0. if bullet else p.key_clearance),mask,zhi-zlo+50,frame,col))
        cut(upper,footprint,'INTERSECT');cut(cap,upper,'DIFFERENCE')
        print('Auricular: adding upper ring, outlets and pry access',flush=True)
        if p.m_ring:
            # The roof blends down through the original lower interface at
            # each end. The same closed channel tool serves both mating parts.
            field=np.maximum(Hbase,height_plain)
            channel((wedge,cap),[point_at(a,plan['ring_distance'],field) for a in np.linspace(0,2*math.pi,512,endpoint=False)],True,footprint)
        for item in plan['outlets']:
            if item['on_wedge']:channel((wedge,cap),[point_at(item['angle'],d,height_plain) for d in np.arange(plan['ring_distance'],skin+land+3.,.5)])
        for item in plan['pries']:
            if item['on_wedge']:pry((wedge,cap),item,height_plain)
        base_cleanup=remove_surface_slivers(base)
        base_fragment=separate_islands(base,col)
        if base_fragment is not None:
            keep(base_fragment);raise RuntimeError('The feature layout leaves a disconnected base fragment; reduce feature sizes or enlarge the land')
        wedge_cleanup={'faces':0,'max_distance_mm':0.}
        fragments=separate_islands(wedge,col)
        if fragments is not None:
            keep(fragments);fragments['cap_operation']='UNION';cap_cutters.append(fragments)
        for part in (base,wedge):
            stats=mu.mesh_stats(part)
            if not stats[0] or any(stats[1:]):raise RuntimeError(f'{part.name}: the partition is not a closed solid; adjust the region or interface height')
        print('Auricular: checking removal paths',flush=True)
        release=release_samples(wedge,(base,),u,zhi-zlo+10)
        cap_release={'status':'NOT_BUILT'}
        sampled=required_points
        wedge_tree=world_tree(wedge)
        errors=[wedge_tree.find_nearest(r*float(q[0])+f*float(q[1])+u*float(q[2]))[3] for q in sampled]
        coverage={'samples':len(errors),'within_01mm_percent':float(100*np.mean(np.asarray(errors)<=.1)),'p95_mm':float(np.percentile(errors,95)),'method':'Refined under-helix region vertices and continuous border samples to retained wedge; raw paint retained separately; not a release guarantee','uncovered_vertices':int(np.count_nonzero(np.asarray(errors)>.1))}
        soft_contact=release_samples(wedge,(solid,),u,zhi-zlo+10)
        info={'selection_coverage':coverage,'border_preparation':border_info,'rear_coverage_margin_mm':0.,'wedge_after_cap':release,'cap_first':cap_release,'silicone_contact':soft_contact,'cutter_repair_mm':repair,'selection':p.wedge_selection,'base_coverage_mode':'INDEPENDENT_ENDS','base_end_modes':[p.wedge_end1_mode,p.wedge_end2_mode],'base_end_distances_mm':[p.wedge_end1_extra,p.wedge_end2_extra],'base_coverage_polygon':poly,'opening_axis':list(u),'exit_angle':angle,'end_connections':end_connections,'base_numerical_cleanup':base_cleanup,'wedge_numerical_cleanup':wedge_cleanup,'roof_sections':roof_sections,'feature_plan':plan,'roof_lift_mm':roof_lift,'roof_mode':'EASE_TO_LOCAL_BASE','anterior_mode':'FOLLOW_SELECTED_REAR_SLOPE','anterior_angles_degrees':list(footprint.get('anterior_angles_degrees',[])),'extended_top_mm':zhi}
        # Publish only after construction checks pass. Keep source and old model
        # intact until this point; failures simply remove the working copies.
        if not original_base.get('three_piece'):
            for original,name in ((input_base,'Auricular_Base_Reference'),):
                old=bpy.data.objects.get(name)
                if old:mu.delete_object(old)
                saved=mu.duplicate_object(original,name,col);saved.hide_set(True);saved.hide_render=True
        old_prepared=bpy.data.objects.get('Auricular_WedgeCapBlank_Reference')
        if old_prepared:mu.delete_object(old_prepared)
        saved_volume=mu.duplicate_object(prepared,'Auricular_WedgeCapBlank_Reference',col);saved_volume.hide_set(True);saved_volume.hide_render=True
        for original in (original_base,original_cap):
            if original is not None:mu.delete_object(original)
        for old in list(context.scene.objects):
            if old.name.startswith('Auricular_CapCut_'):mu.delete_object(old)
        for index,obj in enumerate(cap_cutters):
            temps.remove(obj);obj.name=f'Auricular_CapCut_{index:03d}';obj.hide_set(True);obj.hide_render=True
        old=bpy.data.objects.get('Mold_Wedge')
        if old:mu.delete_object(old)
        for part,name,color in ((base,'Mold_Base',(.93,.70,.25,1)),(wedge,'Mold_Wedge',(.25,.75,.48,1))):
            temps.remove(part);part.name=name;part.color=color;part['anaplast_part']='MOLD';part['three_piece']=True
            part['mold_up']=list(u);part['mold_right']=list(r);part['three_piece_report']=json.dumps(info)
            shade_part(part,u)
            part.hide_set(False)
        wedge['removal_order']=2;base['removal_order']=3
        wedge['silicone_volume_ledger']=json.dumps(volume_ledger.data)
        p.wedge_report=f"Wedge built: {coverage['within_01mm_percent']:.1f}% sampled marked coverage. Inspect, then build Cap."
        if coverage['uncovered_vertices']:p.wedge_report+=f" {coverage['uncovered_vertices']:,} marks remain over 0.1 mm away; use Check marked coverage."
        bpy.ops.anaplast.explode_view()
        for obj in context.selected_objects:obj.select_set(False)
        for obj in (base,wedge):obj.select_set(True)
        context.view_layer.objects.active=wedge
        if not bpy.app.background:
            for area in context.screen.areas:
                if area.type=='VIEW_3D':
                    region=next((v for v in area.regions if v.type=='WINDOW'),None)
                    if region:
                        with context.temp_override(area=area,region=region):bpy.ops.view3d.view_selected(use_all_regions=False)
        from .scene_helpers import organize
        organize(context.scene)
        return info
    finally:
        for obj in temps:
            if obj.name in bpy.data.objects:mu.delete_object(obj)


def build_cap(context):
    if context.scene.get('anaplast_insert_state'):
        raise RuntimeError('Restore the original mold in Interchangeable inserts before rebuilding mold parts')
    from . import mold_volume
    from .explode import ensure_collapsed
    ensure_collapsed(context.scene);p=context.scene.anaplast
    base=bpy.data.objects.get('Mold_Base');wedge=bpy.data.objects.get('Mold_Wedge');prepared=bpy.data.objects.get('Auricular_WedgeCapBlank_Reference') or bpy.data.objects.get('Auricular_CapBlank_Reference')
    col=mu.get_collection(context.scene)
    cutters=sorted((o for o in context.scene.objects if o.name.startswith('Auricular_CapCut_')),key=lambda o:o.name)
    if base is None or wedge is None or prepared is None or not cutters:raise RuntimeError('Build the base, then the wedge first')
    # If only Cap is rebuilt after shelling, restore its construction inputs
    # from the exact solid references, without changing the displayed parts.
    from . import mold_auricular_shell as shells
    display_base,display_wedge=base,wedge
    volume_ledger=mold_volume.Ledger(json.loads(wedge['silicone_volume_ledger'])) if 'silicone_volume_ledger' in wedge else None
    base=shells.solid_source(base);wedge=shells.solid_source(wedge)
    cap=mu.duplicate_object(prepared,'Auricular_FinalCap_work',col);revised=None;shell_parts=[]
    try:
        mold_volume.clear(context.scene,'Rebuild the wedge and cap to record separate cavity and channel volumes')
        for cutter in cutters:
            role=cutter.get('silicone_volume_role')
            operation=cutter.get('cap_operation','DIFFERENCE')
            if volume_ledger is not None and role in ('cavity','channels') and operation=='DIFFERENCE':
                volume_ledger.cut(cap,cutter,operation,cut,role)
            else:cut(cap,cutter,operation)
        cap_cleanup=remove_surface_slivers(cap,(wedge,base,reference(p)))
        fragments=separate_islands(cap,col)
        print('Cap cleanup',cap_cleanup,'remaining fragments',mu.mesh_stats(fragments) if fragments else None,flush=True)
        if fragments is not None:
            revised=mu.duplicate_object(wedge,'Auricular_WedgeJoin_work',col)
            try:
                cut(revised,fragments,'UNION')
                join_cleanup=remove_surface_slivers(revised,(cap,base,reference(p)))
                extra=separate_islands(revised,col)
                if extra is not None:
                    mu.delete_object(extra);raise RuntimeError('This region leaves a disconnected mold fragment; adjust the rear selection before building the cap')
            finally:mu.delete_object(fragments)
        stats=mu.mesh_stats(cap)
        if not stats[0] or any(stats[1:]):raise RuntimeError(f'The cap did not form a closed solid: {stats}')
        u,r,f=frame_for(base);lo,hi=mu.world_bbox(base)
        info=json.loads(wedge['three_piece_report']);info['cap_first']=release_samples(cap,(base,revised or wedge),u,(hi-lo).length+10)
        info['wedge_after_cap']=release_samples(revised or wedge,(base,),u,(hi-lo).length+10)
        info['cap_numerical_cleanup']=cap_cleanup
        if revised:info['join_numerical_cleanup']=join_cleanup
        final_wedge=revised or wedge
        if shells.wants_shell(p):
            shell_parts,shell_report=shells.make_parts((base,final_wedge,cap),shells.walls(p),p.m_voxel)
            info['shells']=shell_report
        else:info.pop('shells',None)
        solid_parts=shells.cache_solids((base,final_wedge,cap),col)
        targets=(display_base,display_wedge,cap)
        for target,finished in zip(targets,shell_parts or solid_parts):
            if target!=finished:
                old_mesh=target.data;target.data=finished.data.copy();target.matrix_world=finished.matrix_world.copy()
                if old_mesh.users==0:bpy.data.meshes.remove(old_mesh)
            target['auricular_shell']=bool(finished.get('auricular_shell'))
        base,wedge=display_base,display_wedge
        old=bpy.data.objects.get('Mold_Cap')
        if old:mu.delete_object(old)
        cap.name='Mold_Cap';cap['anaplast_part']='MOLD';cap['three_piece']=True;cap['mold_up']=list(u);cap['mold_right']=list(r)
        cap['removal_order']=1;cap.color=(.35,.55,.85,1);cap.hide_set(False);cap.hide_render=False
        for face in cap.data.polygons:face.use_smooth=True
        for obj in (base,wedge,cap):
            obj['three_piece_report']=json.dumps(info);shade_part(obj,u)
        if volume_ledger is not None:mold_volume.publish(context.scene,volume_ledger,(base,wedge,cap))
        blocked=info['cap_first']['blocked_samples']+info['wedge_after_cap']['blocked_samples']
        p.wedge_report=f'Three-piece mold built. {blocked} sampled rigid contacts to review.' if blocked else 'Three-piece mold built. No sampled rigid obstruction.'
        coverage=info.get('selection_coverage',{})
        if coverage.get('uncovered_vertices',0):p.wedge_report+=f" {coverage['uncovered_vertices']:,} marks need review; use Check marked coverage."
        if shells.wants_shell(p):p.wedge_report+=' Shell bodies: base and cap backs open; wedge open at outside rim.'
        bpy.ops.anaplast.explode_view()
        for obj in context.selected_objects:obj.select_set(False)
        for obj in (base,wedge,cap):obj.select_set(True)
        context.view_layer.objects.active=cap
        from .scene_helpers import organize
        organize(context.scene)
        return info
    except Exception:
        if cap.name in bpy.data.objects:mu.delete_object(cap)
        raise
    finally:
        if revised is not None and revised.name in bpy.data.objects:mu.delete_object(revised)
        for obj in shell_parts:
            if obj.name in bpy.data.objects:mu.delete_object(obj)


class ANAPLAST_OT_auricular_base(bpy.types.Operator):
    bl_idname='anaplast.auricular_base';bl_label='1: Build Base';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        # Three-piece features are planned together after the rear region is
        # known. Baking two-piece keys now would duplicate them under the wedge.
        p=context.scene.anaplast
        plain={'m_base_type':'FULL','m_key_type':'NONE','m_spillways':0,'m_ring':False,'m_pry':0,'m_bolts':0}
        saved={name:getattr(p,name) for name in plain}
        try:
            for name,value in plain.items():setattr(p,name,value)
            result=bpy.ops.anaplast.mold_two_part(which='BASE')
        finally:
            for name,value in saved.items():setattr(p,name,value)
        if 'FINISHED' in result:
            from .explode import ensure_collapsed
            ensure_collapsed(context.scene)
            cap=bpy.data.objects.get('Mold_Cap')
            if cap:mu.delete_object(cap)
            bpy.data.objects['Mold_Base']['auricular_plain_layout']=True
            context.scene.anaplast.wedge_report='Base built. Select the rear region, then build the wedge.'
        return result


class ANAPLAST_OT_auricular_shell_update(bpy.types.Operator):
    bl_idname='anaplast.auricular_shell_update';bl_label='Update full / shell bodies';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        from .mold_auricular_shell import update_bodies
        try:update_bodies(context)
        except Exception as e:rep(self,{'ERROR'},str(e));return {'CANCELLED'}
        rep(self,{'INFO'},'Three-piece bodies updated; cavity and mating surfaces retained')
        return {'FINISHED'}


class ANAPLAST_OT_auricular_cap(bpy.types.Operator):
    bl_idname='anaplast.auricular_cap';bl_label='3: Build Cap';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:build_cap(context)
        except Exception as e:rep(self,{'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}


def save_trace(context,points,frame):
    p=context.scene.anaplast;old=p.wedge_trace
    cu=bpy.data.curves.new('Wedge_Boundary','CURVE');cu.dimensions='3D';cu.bevel_depth=.08
    spline=cu.splines.new('POLY');spline.points.add(len(points)-1)
    for v,co in zip(spline.points,points):v.co=(*co,1)
    spline.use_cyclic_u=True;obj=bpy.data.objects.new('Wedge_Boundary',cu);mu.get_collection(context.scene).objects.link(obj)
    obj.show_in_front=True;obj['trace_up']=list(frame[0]);obj['trace_right']=list(frame[1]);p.wedge_trace=obj;p.wedge_selection='TRACE'
    if old:mu.delete_object(old)
    return obj


class ANAPLAST_OT_wedge_suggest(bpy.types.Operator):
    """Suggest the accessible rear undercut on an editable copy, excluding inner ear recesses"""
    bl_idname='anaplast.wedge_suggest';bl_label='Suggest Undercuts';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:suggest(context)
        except Exception as e:rep(self,{'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}


class ANAPLAST_OT_wedge_build(bpy.types.Operator):
    bl_idname='anaplast.wedge_build';bl_label='2: Build Wedge';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:info=build(context)
        except Exception as e:
            import traceback;traceback.print_exc();rep(self,{'ERROR'},str(e));return {'CANCELLED'}
        rep(self,{'WARNING'} if info['selection_coverage'].get('uncovered_vertices',0) else {'INFO'},context.scene.anaplast.wedge_report)
        return {'FINISHED'}


class ANAPLAST_OT_wedge_trace(bpy.types.Operator):
    """Click around the visible rear wedge region; Enter closes it, Backspace removes a point, Esc cancels"""
    bl_idname='anaplast.wedge_trace';bl_label='Trace Rear Wedge Boundary';bl_options={'REGISTER','UNDO'}
    def invoke(self,context,event):
        from .mold_auto import view_axes
        scan=reference(context.scene.anaplast)
        if scan is None:rep(self,{'ERROR'},'Assign the ear sculpt');return {'CANCELLED'}
        self.region=next((r for r in context.area.regions if r.type=='WINDOW'),None)
        self.rv=context.space_data.region_3d
        q=self.rv.view_rotation;self.frame=(q@Vector((0,0,1)),q@Vector((1,0,0)))
        self.scan=scan;self.points=[];self.screen=[]
        self.handle=bpy.types.SpaceView3D.draw_handler_add(self.draw_trace,(), 'WINDOW','POST_PIXEL')
        context.area.header_text_set('Click wedge boundary | Enter finish | Backspace undo point | Esc cancel')
        context.window.cursor_modal_set('CROSSHAIR');context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    def draw_trace(self):
        if len(self.screen)<2:return
        import gpu
        from gpu_extras.batch import batch_for_shader
        shader=gpu.shader.from_builtin('UNIFORM_COLOR');shader.bind();shader.uniform_float('color',(1,.5,.1,1))
        batch_for_shader(shader,'LINE_STRIP',{'pos':self.screen+[self.screen[0]]}).draw(shader)
    def finish(self,context):
        bpy.types.SpaceView3D.draw_handler_remove(self.handle,'WINDOW');context.area.header_text_set(None)
        context.window.cursor_modal_restore();context.area.tag_redraw()
    def modal(self,context,event):
        from bpy_extras import view3d_utils
        if event.type in {'ESC','RIGHTMOUSE'}:
            self.finish(context);return {'CANCELLED'}
        if event.type=='BACK_SPACE' and event.value=='PRESS':
            if self.points:self.points.pop();self.screen.pop()
        if event.type in {'RET','NUMPAD_ENTER'} and event.value=='PRESS':
            if len(self.points)<6:rep(self,{'WARNING'},'Place at least six points around the opening');return {'RUNNING_MODAL'}
            projected=np.array([[v.dot(self.frame[1]),v.dot(self.frame[0].cross(self.frame[1]))] for v in self.points])
            if len(convex_hull_2d(projected))<3:rep(self,{'WARNING'},'Trace a region with area');return {'RUNNING_MODAL'}
            save_trace(context,self.points,self.frame);self.finish(context);return {'FINISHED'}
        if event.type=='LEFTMOUSE' and event.value=='PRESS':
            co=(event.mouse_x-self.region.x,event.mouse_y-self.region.y)
            if not(0<=co[0]<self.region.width and 0<=co[1]<self.region.height):return {'RUNNING_MODAL'}
            origin=view3d_utils.region_2d_to_origin_3d(self.region,self.rv,co);direction=view3d_utils.region_2d_to_vector_3d(self.region,self.rv,co)
            inv=self.scan.matrix_world.inverted();local=inv.to_3x3()@direction
            hit,point,normal,face=self.scan.ray_cast(inv@origin,local.normalized())
            if hit:self.points.append(self.scan.matrix_world@point);self.screen.append(co)
            else:rep(self,{'WARNING'},'That point did not hit the ear sculpt')
        context.area.tag_redraw();return {'RUNNING_MODAL'}


_classes=(ANAPLAST_OT_auricular_shell_update,ANAPLAST_OT_wedge_suggest,ANAPLAST_OT_wedge_build,ANAPLAST_OT_wedge_trace,ANAPLAST_OT_auricular_base,ANAPLAST_OT_auricular_cap)
def register():
    for c in _classes:bpy.utils.register_class(c)
def unregister():
    for c in reversed(_classes):bpy.utils.unregister_class(c)
