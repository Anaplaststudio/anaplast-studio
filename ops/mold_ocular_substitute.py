"""Removable wax try-in former: unchanged front, swept rear access."""
import math
import numpy as np
import bpy,bmesh
from mathutils import Vector,Matrix


def enabled(item):
    return bool(item and item.kind=='OCULAR' and item.ocular_use=='SUBSTITUTE')


def plan(work,item,base,start,frame):
    from . import mold_inserts as mi,ocular_view
    cache=getattr(work,'substitute_plans',None)
    if cache is None:cache=work.substitute_plans={}
    if item.uid in cache:return cache[item.uid]
    scene=work.scene;source=scene.anaplast.ocular_obj
    if not source or item.ocular not in (source,scene.ocular_production.color_obj):
        raise RuntimeError('Choose this case\'s generated ocular for the wax try-in substitute')
    if source.get('gaze_edge_pending'):raise RuntimeError('Apply gaze and refit the ocular edge before building the substitute')
    mi.closed(source,connected=True)
    attr=source.data.attributes.get('ocular_front')
    if not attr or attr.domain!='FACE':raise RuntimeError('Regenerate the ocular to identify its protected front surface')
    world=np.array([source.matrix_world@v.co for v in source.data.vertices],dtype=float)
    front=sorted({int(v) for p,a in zip(source.data.polygons,attr.data) if a.value for v in p.vertices})
    if not front:raise RuntimeError('The ocular has no identified front surface')
    u,r,f=frame;start=Vector(start)
    # Keep ALL original front faces, including the visible opening, ahead of
    # the extra clearance. This conservative plane may constrain extreme gaze.
    guard=float((world[front]@np.array(u)).min())-.25
    center,forward,right,up,_=ocular_view.eye_frame(bpy.context)
    bm=bmesh.new()
    try:
        for p in world:bm.verts.new(p)
        hull=bmesh.ops.convex_hull(bm,input=list(bm.verts),use_existing_faces=False)
        vertices={v for face in hull['geom'] if isinstance(face,bmesh.types.BMFace) for v in face.verts}
        samples=np.array([v.co[:] for v in vertices])
    finally:bm.free()
    pivot=scene.ocular_view.pivot
    if pivot and 'gaze_rest_matrix' in pivot and source.parent==pivot:
        inverse=pivot.matrix_world.inverted();x0=float(pivot.get('gaze_angle_x',0));z0=float(pivot.get('gaze_angle_z',0))
        def transform(x,z):return ocular_view.gaze_matrix(pivot,x0+x,z0+z)@inverse
    else:
        x_axis=Vector((1,0,0)) if scene.anaplast.oriented else right
        z_axis=Vector((0,0,1)) if scene.anaplast.oriented else up
        def transform(x,z):
            return Matrix.Translation(center)@Matrix.Rotation(z,4,z_axis)@Matrix.Rotation(x,4,x_axis)@Matrix.Translation(-center)
    limits=[math.radians(item.gaze_vertical),math.radians(item.gaze_horizontal)]
    grids=[np.linspace(-a,a,max(2,2*math.ceil(math.degrees(a)/2.5))+1) if a else [0.] for a in limits]
    angles=np.linspace(0,2*math.pi,128,endpoint=False)
    normals=np.column_stack((np.cos(angles),np.sin(angles)))
    support=np.full(len(angles),item.stem*.5,dtype=float)
    for x in grids[0]:
        for z in grids[1]:
            matrix=np.array(transform(float(x),float(z)))
            moved=samples@matrix[:3,:3].T+matrix[:3,3]
            xy=np.column_stack(((moved-start)@r,(moved-start)@f))
            support=np.maximum(support,(xy@normals.T).max(axis=0))
    radius=float(np.linalg.norm(world-np.array(center),axis=1).max())
    # Conservative interpolation allowance between angular samples, plus a
    # small manufacturing clearance only in the protected rear region.
    step=max((float(g[1]-g[0]) for g in grids if len(g)>1),default=0.)
    allowance=item.gaze_clearance+radius*step*step*.5
    # Intersect supporting lines: a conservative smooth polygon without
    # retaining millions of rotated points or undersizing the access opening.
    support+=allowance
    profile=[tuple(np.linalg.solve(np.array((normals[j],normals[(j+1)%len(normals)])),
                 np.array((support[j],support[(j+1)%len(normals)])))) for j in range(len(normals))]
    profile=mi.aw.convex_hull_2d(profile)
    end=mi.ocular_center_contact(source,frame)[0]
    if guard<=start.dot(u)+1:raise RuntimeError('Not enough space behind the protected ocular front for rear access')
    data=dict(source=source,profile=profile,start=start,guard=guard,center=center,end=end,
              samples=len(grids[0])*len(grids[1]),allowance=allowance,
              access_size=list(np.ptp(np.asarray(profile),axis=0)),
              front_points=world[front][::max(1,len(front)//1000)])
    pillar_axis=Vector((0,0,1)) if scene.anaplast.oriented else Vector(up)
    pillar_axis-=u*pillar_axis.dot(u)
    if pillar_axis.length<.1:pillar_axis=Vector(f)
    data['pillar_axis']=pillar_axis.normalized()
    cache[item.uid]=data
    return data


def outlet_distance(data,direction,frame):
    _,r,f=frame;v=np.array((direction.dot(r),direction.dot(f)))
    poly=np.asarray(data['profile']);edges=np.roll(poly,-1,axis=0)-poly
    normal=np.column_stack((edges[:,1],-edges[:,0]));height=np.sum(normal*poly,axis=1)
    den=normal@v;valid=den>1e-10
    return float((height[valid]/den[valid]).min())+.6


def keep_upper(work,obj,base,frame):
    """Discard only disconnected material entirely behind the original base."""
    from . import mold_inserts as mi
    mi.discard_boolean_dust(obj)
    bm=bmesh.new();bm.from_mesh(obj.data);remaining=set(bm.verts);groups=[]
    try:
        while remaining:
            group=[];stack=[remaining.pop()]
            while stack:
                v=stack.pop();group.append(v)
                for e in v.link_edges:
                    other=e.other_vert(v)
                    if other in remaining:remaining.remove(other);stack.append(other)
            groups.append(group)
        if len(groups)<2:return
        u=frame[0];main=max(groups,key=lambda g:max((obj.matrix_world@v.co).dot(u) for v in g))
        tree=mi.aw.world_tree(base);lo,hi=mi.bounds(base,u);remove=[]
        for group in groups:
            if group is main:continue
            for v in group:
                q=obj.matrix_world@v.co
                if q.dot(u)>mi.column(tree,q,frame,lo,hi)[1]+.005:
                    raise RuntimeError('Rear access is split by the mold; move its mounting point')
            remove.extend(group)
        bmesh.ops.delete(bm,geom=remove,context='VERTS');bm.to_mesh(obj.data);obj.data.update()
    finally:bm.free()


def smooth_former(work,data,base,frame):
    """Retain the ocular front and loft a monotone, tangent rear shoulder."""
    from . import mold_inserts as mi,mold_insert_reservoir as compact
    source=data['source'];u,r,f=frame
    world=[source.matrix_world@v.co for v in source.data.vertices]
    faces=[tuple(p.vertices) for p,a in zip(source.data.polygons,source.data.attributes['ocular_front'].data) if a.value]
    normal_matrix=source.matrix_world.to_3x3().inverted().transposed()
    if min((normal_matrix@p.normal).dot(u) for p,a in zip(source.data.polygons,source.data.attributes['ocular_front'].data) if a.value)<.01:
        raise RuntimeError('The ocular front has an undercut in the mold removal direction; adjust its gaze or the mold direction')
    edges={};normals={}
    for ids in faces:
        n=(world[ids[1]]-world[ids[0]]).cross(world[ids[2]]-world[ids[0]])
        for a,b in zip(ids,ids[1:]+ids[:1]):edges[tuple(sorted((a,b)))]=edges.get(tuple(sorted((a,b))),0)+1
        for i in ids:normals[i]=normals.get(i,Vector())+n
    adjacency={}
    for (a,b),count in edges.items():
        if count==1:adjacency.setdefault(a,[]).append(b);adjacency.setdefault(b,[]).append(a)
    if not adjacency or any(len(v)!=2 for v in adjacency.values()):raise RuntimeError('The ocular front needs one closed border')
    ring=[next(iter(adjacency))];previous=None
    while True:
        nxt=next(v for v in adjacency[ring[-1]] if v!=previous)
        if nxt==ring[0]:break
        previous=ring[-1];ring.append(nxt)
        if len(ring)>len(adjacency):raise RuntimeError('Cannot follow the ocular front border')
    if len(ring)!=len(adjacency):raise RuntimeError('The ocular front has more than one border')
    origin=sum((world[i] for i in ring),Vector())/len(ring)
    polygon=np.asarray(data['profile']);edge=np.roll(polygon,-1,axis=0)-polygon
    normal=np.column_stack((edge[:,1],-edge[:,0]));height=np.sum(normal*polygon,axis=1)
    start=np.array([(origin-data['start']).dot(r),(origin-data['start']).dot(f)])
    lo,hi=mi.bounds(base,u);bottom=lo-.5;tree=mi.aw.world_tree(base)
    probes=[data['start']]+[data['start']+r*float(x)+f*float(y) for x,y in polygon]
    join_plane=max(mi.column(tree,q,frame,lo,hi)[1] for q in probes)+.5
    if join_plane>min(world[i].dot(u) for i in ring)-.5:
        raise RuntimeError('The fitting surface leaves too little height for a smooth removable former; review the mounting position')
    data['shoulder_join_plane']=join_plane
    controls=[]
    for i in ring:
        p=world[i];direction=p-origin;direction-=u*direction.dot(u);direction.normalize()
        ray=np.array([direction.dot(r),direction.dot(f)]);den=normal@ray;valid=den>1e-10
        radius=float(((height-normal@start)[valid]/den[valid]).min())
        initial=(p-origin).dot(direction)
        radius=max(radius,initial+.6)
        q=origin+direction*radius;q+=u*(join_plane-q.dot(u))
        floor=mi.column(tree,q,frame,lo,hi)[1]
        n=normals[i].normalized();slope=-n.dot(direction)/n.dot(u)
        if slope>.05:raise RuntimeError('The ocular border turns inward in the removal direction; refit its edge first')
        slope=min(0.,slope);span=radius-initial
        reach=min(span*.65,3.,max(.05,(p.dot(u)-join_plane)*.35/max(.01,-slope)))
        p1=p+(direction+u*slope)*reach
        p2=q+u*(p1.dot(u)-q.dot(u))
        controls.append((p,p1,p2,q))
    used=sorted({i for face in faces for i in face});mapping={old:j for j,old in enumerate(used)}
    verts=[world[i] for i in used];outfaces=[tuple(mapping[i] for i in face) for face in faces]
    oldring=[mapping[i] for i in ring]
    for t in np.linspace(0,1,81)[1:]:
        newring=[]
        for p,p1,p2,q in controls:
            newring.append(len(verts));verts.append((1-t)**3*p+3*(1-t)**2*t*p1+3*(1-t)*t*t*p2+t**3*q)
        for j in range(len(ring)):
            k=(j+1)%len(ring);outfaces.extend(((oldring[j],oldring[k],newring[k]),(oldring[j],newring[k],newring[j])))
        oldring=newring
    newring=[]
    for _,_,_,q in controls:newring.append(len(verts));verts.append(q+u*(bottom-q.dot(u)))
    for j in range(len(ring)):
        k=(j+1)%len(ring);outfaces.extend(((oldring[j],oldring[k],newring[k]),(oldring[j],newring[k],newring[j])))
    outfaces.append(tuple(reversed(newring)))
    mesh=bpy.data.meshes.new('Ocular_former');mesh.from_pydata(verts,[],outfaces);mesh.update()
    core=bpy.data.objects.new('Ocular_try_in_former',mesh);work.col.objects.link(core);work.add(core)
    bm=bmesh.new()
    try:
        bm.from_mesh(mesh);bmesh.ops.recalc_face_normals(bm,faces=bm.faces[:]);bmesh.ops.triangulate(bm,faces=bm.faces[:]);bm.to_mesh(mesh)
    finally:bm.free()
    mi.closed(core,connected=True)
    trim=work.copy(base,'Former_base_clearance')
    trim.matrix_world.translation+=u*.02
    mi.aw.cut(core,trim,'DIFFERENCE');work.delete(trim)
    keep_upper(work,core,base,frame);compact.clean_seams(core);mi.closed(core,connected=True)
    # Wax-contact faces must release toward -u. Downward-facing triangles
    # are allowed only on the underside that seats against the fitting base.
    for p in core.data.polygons:
        if p.normal.dot(u)>=-.0001:continue
        q=p.center
        gap=q.dot(u)-mi.column(tree,q,frame,lo,hi)[1]
        distance=tree.find_nearest(q)[3]
        if gap>.05 and distance>.025:
            raise RuntimeError(f'The former contains an undercut toward the fitting surface (normal {p.normal.dot(u):.5f}, height {gap:.3f} mm, fitting gap {distance:.3f} mm); its original front or mounting direction needs adjustment')
    core['former_removal_direction']=list(-u)
    core['former_exterior_undercut_check']=True
    body_tree=mi.aw.world_tree(core);outside=0.
    for p in world:
        hit=body_tree.find_nearest(p)
        if (p-hit[0]).dot(hit[1])>0:outside=max(outside,hit[3])
    if outside>.02:raise RuntimeError(f'The smooth former is too tight for the real ocular ({outside:.3f} mm); increase its rear clearance')
    data['neutral_ocular_max_outside_mm']=outside
    return core


def detach(work,obj,dock,frame,item):
    """Keep the real ocular pedestal and cut its matching key into the former."""
    from . import mold_inserts as mi,mold_insert_reservoir as compact
    u=frame[0];seat_frame=(u,Vector(dock['right']),Vector(dock['forward']))
    connection=work.substitute_connections[item.uid]
    center=Vector(dock['center']);plane=dock['rear']+.8
    clearance=0.  # Holder seats directly on the pedestal; no wax-filled sleeve.
    # Start with the upper body before it was joined to the contour plug.
    # Subtracting an identical plug from that joined copy leaves coincident
    # fitting triangles. The separate body has already been trimmed to base.
    former=work.copy(connection['former_body'],'Detachable_ocular_former')
    # Reuse the real pedestal directly. Intersecting two copies of the same
    # contour plug produces coplanar Boolean slivers at its fitting surface.
    lower=work.copy(connection['pedestal'],'Same_ocular_pedestal')
    if obj.get('compact_passages'):
        for body in connection['cups']:lower=mi.merge(work,lower,work.copy(body,'Same_pedestal_cup'))
        # Retain the exterior before drilling for cast-body previews. The
        # passages stay open in the physical piece; their wax is a sprue.
        obj['cast_preview_solid']=lower.data.copy()
        for bore in connection['bores']:mi.aw.cut(lower,bore,'DIFFERENCE')
    old=obj.data;obj.data=lower.data.copy();obj.matrix_world=lower.matrix_world.copy()
    if old.users==0:bpy.data.meshes.remove(old)
    work.delete(lower)
    # The stem enters from the fitting side; the selected ocular key is at
    # its original position deeper inside. Do not pierce the ocular front.
    end=connection['end']
    # Subtract the completed matching part, including the exact stem, seat
    # and male key. Inflated access cutters left a wax sleeve and key-shaped
    # membrane inside the holder. Its existing feed/vent bores stay open.
    mating=work.copy(obj,'Exact_pedestal_mating_tool')
    if obj.get('compact_passages'):
        old_mating=mating.data;mating.data=obj['cast_preview_solid'].copy()
        if old_mating.users==0:bpy.data.meshes.remove(old_mating)
    uncut=former.data.copy();failures=[]
    try:
        for solver in ('MANIFOLD','EXACT'):
            if solver=='EXACT':
                failed=former.data;former.data=uncut.copy()
                if failed.users==0:bpy.data.meshes.remove(failed)
            try:
                mi.mu.apply_boolean(former,mating,'DIFFERENCE',solver=solver)
                mi.discard_boolean_dust(former);compact.clean_seams(former);mi.closed(former,connected=True)
                break
            except RuntimeError as exc:failures.append(str(exc))
        else:raise RuntimeError('The exact pedestal contact could not be joined cleanly: '+'; '.join(failures))
    finally:
        if uncut.users==0:bpy.data.meshes.remove(uncut)
        work.delete(mating)
    # Cut the shared passages once, after the mating exterior. Subtracting
    # two already-drilled solids would compare identical bore walls.
    if obj.get('compact_passages'):
        for bore in connection['bores']:mi.aw.cut(former,bore,'DIFFERENCE')
    for part in (obj,former):
        mi.discard_boolean_dust(part);compact.clean_seams(part);mi.closed(part,connected=True)
        part['ocular_former_detachable']=True;part['former_joint_depth_mm']=item.key_depth
        part['former_joint_clearance_mm']=clearance;part['former_joint_plane']=plane
        part['former_joint_key']=item.key;part['ocular_key_center_world']=list(end)
    former['ocular_former_upper']=True
    former['ocular_casting_substitute']=True
    obj['ocular_former_upper']=False
    return former


def wax_support_grooves(work,item,core,data,base,frame):
    """Open draw-direction grooves cast four integral corner wax pillars."""
    from . import mold_inserts as mi,mold_insert_reservoir as compact
    source=data['source'];u=frame[0];world=np.array([source.matrix_world@v.co for v in source.data.vertices])
    origin=Vector(world.mean(axis=0));lo,hi=mi.bounds(base,u);tree=mi.aw.world_tree(base);details=[]
    vertical=Vector(data['pillar_axis']);side=vertical.cross(u).normalized()
    # The set of four positions is symmetric. Use the facial X midplane to
    # name the two sides; no angular or left/right setting is needed.
    toward_midline=Vector((1 if origin.x<0 else -1,0,0))
    medial=side if side.dot(toward_midline)>0 else -side
    corners=((1,1,'Upper medial'),(-1,1,'Lower medial'),(1,-1,'Upper lateral'),(-1,-1,'Lower lateral'))
    for vertical_sign,side_sign,name in corners:
        direction=(vertical*vertical_sign+medial*side_sign).normalized();tangent=direction.cross(u).normalized()
        xy=np.column_stack(((world-origin)@tangent,(world-origin)@direction))
        hull=np.asarray(mi.aw.convex_hull_2d(xy.tolist()));contact=int(np.argmax(xy[:,1]));ct=float(xy[contact,0])
        a=max(float(hull[:,0].min())+.001,ct-item.wax_pillar_width*.5)
        b=min(float(hull[:,0].max())-.001,ct+item.wax_pillar_width*.5)
        inner=[]
        for t in np.linspace(a,b,max(7,math.ceil((b-a)/.5)+1)):
            hits=[]
            for p,q in zip(hull,np.roll(hull,-1,axis=0)):
                if min(p[0],q[0])-1e-8<=t<=max(p[0],q[0])+1e-8 and abs(q[0]-p[0])>1e-10:
                    hits.append(float(p[1]+(q[1]-p[1])*(t-p[0])/(q[0]-p[0])))
            if not hits:raise RuntimeError('Cannot locate the ocular border for a wax pillar')
            inner.append((float(t),max(hits)+.02))
        outer=float(xy[:,1].max())+max(data['access_size'])+5
        profile=inner+[(b,outer),(a,outer)]
        contact_point=Vector(world[contact]);floor=mi.column(tree,contact_point,frame,lo,hi)[1]
        if any((contact_point-Vector(d['contact_world'])).length<item.wax_pillar_width+.3 for d in details):
            raise RuntimeError('The corner wax supports overlap on this ocular; reduce their width')
        bottom=max(floor+.1,contact_point.dot(u)-1.)
        if bottom>=contact_point.dot(u):raise RuntimeError('There is no room behind this ocular border for its wax support pillar')
        # Round the groove floor explicitly. Beveling the many short contact
        # edges creates slivers; matching offset rings retain clean topology.
        poly=np.asarray(profile);edge=np.roll(poly,-1,axis=0)-poly
        inward=np.column_stack((-edge[:,1],edge[:,0]));inward/=np.linalg.norm(inward,axis=1)[:,None]
        previous=np.roll(inward,1,axis=0)
        inset=(inward+previous)/np.maximum(1e-8,1+np.sum(inward*previous,axis=1))[:,None]
        xy_origin=origin-u*origin.dot(u);rings=[]
        for angle in np.linspace(0,math.pi*.5,9):
            points=poly+inset*(.25*(1-math.sin(angle)));z=bottom+.25*(1-math.cos(angle))
            rings.append([xy_origin+tangent*float(x)+direction*float(y)+u*z for x,y in points])
        rings.append([xy_origin+tangent*float(x)+direction*float(y)+u*(float((world@u).max())+5) for x,y in poly])
        cutter=mi.loft(work,'Wax_'+name+'_pillar_groove',rings)
        bm=bmesh.new()
        try:
            bm.from_mesh(cutter.data);bmesh.ops.triangulate(bm,faces=bm.faces[:]);bm.to_mesh(cutter.data);cutter.data.update()
        finally:bm.free()
        original=core.data.copy();failures=[]
        try:
            for solver in ('MANIFOLD','EXACT'):
                try:
                    mi.mu.apply_boolean(core,cutter,'DIFFERENCE',solver=solver)
                    mi.discard_boolean_dust(core);compact.clean_seams(core);mi.closed(core,connected=True)
                    break
                except RuntimeError as exc:
                    failures.append(str(exc));failed=core.data;core.data=original.copy()
                    if failed.users==0:bpy.data.meshes.remove(failed)
            else:raise RuntimeError('The wax pillar groove could not be formed cleanly: '+'; '.join(failures))
        finally:
            if original.users==0:bpy.data.meshes.remove(original)
            work.delete(cutter)
        details.append(dict(position=name,width_mm=b-a,ocular_clearance_mm=.02,contact_world=list(contact_point)))
    core['wax_support_pillars']=True
    return details


def build(work,item,blank,base,molds,start,frame,dock,ocular,root_radius=None):
    from . import mold_inserts as mi,mold_insert_reservoir as compact
    data=plan(work,item,base,start,frame);u=frame[0]
    if not hasattr(work,'substitute_connections'):work.substitute_connections={}
    if item.uid not in work.substitute_connections:
        # Use precisely the same generator and female cutters as the real
        # ocular. Only the temporary eye is cut; the patient's eye is intact.
        temporary_eye=work.copy(ocular,'Former_connection_reference_eye')
        reference=work.copy(blank,'Former_connection_reference_plug')
        mount_top=mi.bounds(reference,u)[1]
        cutters=[]
        reference=mi.pedestal(work,item,reference,temporary_eye,frame,start=start,root_radius=root_radius,recess_tools=cutters)
        work.delete(temporary_eye)
        work.substitute_connections[item.uid]=dict(pedestal=reference,cutters=cutters,start=Vector(start),
            end=Vector(reference['ocular_key_center_world']),root_radius=root_radius or item.stem*.5,mount_top=mount_top)
    seat_frame=(u,Vector(dock['right']),Vector(dock['forward']))
    # Keep the keyed neck wholly inside the broad rear body, including a
    # little room for the smooth shoulder above the fitting surface.
    sr,sf=seat_frame[1:];seat_center=Vector(dock['center'])
    extra=[seat_center+sr*x+sf*y for x,y in mi.d_profile(dock['seat_width_mm']*.5+.6,dock['key_flat'])]
    data['profile']=mi.aw.convex_hull_2d(data['profile']+[((p-data['start']).dot(frame[1]),(p-data['start']).dot(frame[2])) for p in extra])
    data['access_size']=list(np.ptp(np.asarray(data['profile']),axis=0))
    core=smooth_former(work,data,base,frame)
    pillars=wax_support_grooves(work,item,core,data,base,frame) if item.wax_pillars else []
    work.substitute_connections[item.uid]['former_body']=work.copy(core,'Unjoined_ocular_holder_reference')
    neck=mi.profile_solid(work,'Substitute_mounting_neck',
                          mi.d_profile(dock['seat_width_mm']*.5,dock['key_flat']),
                          Vector(dock['center']),dock['rear']+.1,data['shoulder_join_plane']-.1,seat_frame)
    # Fill the complete existing keyed plug cross-section. A narrower neck
    # leaves coincident, oppositely facing fitting triangles around it when
    # the rear access meets the plug. This positive overlap removes those
    # internal faces without widening the mounting seat or closing its bores.
    core=mi.merge(work,core,neck)
    mi.discard_boolean_dust(core);compact.clean_seams(core)
    blank=mi.merge(work,blank,core)
    blank['ocular_key_center_world']=list(data['end'])
    blank['ocular_casting_substitute']=True
    blank['gaze_horizontal_degrees']=item.gaze_horizontal;blank['gaze_vertical_degrees']=item.gaze_vertical
    blank['substitute_guard_plane']=data['guard'];blank['substitute_globe_center']=list(data['center'])
    mi.discard_boolean_dust(blank);compact.clean_seams(blank);mi.closed(blank,connected=True)
    tree=mi.aw.world_tree(blank)
    front_error=max(tree.find_nearest(Vector(p))[3] for p in data['front_points'])
    if front_error>.005:raise RuntimeError(f'The casting substitute changed the protected ocular front ({front_error:.3f} mm); the mold was not changed')
    return blank,dict(ocular_use='SUBSTITUTE',rear_access_mm=data['access_size'],
                     requested_gaze_horizontal=item.gaze_horizontal,requested_gaze_vertical=item.gaze_vertical,
                     angular_samples=data['samples'],rear_allowance_mm=data['allowance'],
                     wax_pillars=pillars,
                     neutral_ocular_max_outside_mm=data['neutral_ocular_max_outside_mm'],
                     protected_front=True,front_sample_error_mm=front_error,globe_center=list(data['center']),
                     note='Rear clearance envelope only. The protected eyelid opening may limit the full requested gaze range.')
