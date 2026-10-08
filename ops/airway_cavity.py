"""Experimental cavity-first nasal inserts. All construction is on work copies.

The case is upright (world Z up, X across the face). A sampled nearest-surface
distance is reported, never treated as a certified minimum wall or release test.
"""
import json
import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from ..utils import mesh as mu
from . import mold_inserts as mi, mold_auricular as aw, mold_auto as ma
from . import airway_marking as marks, substructure


def smooth(obj, count):
    mu.apply_modifier(obj, 'SMOOTH', lambda m: (setattr(m, 'factor', .5), setattr(m, 'iterations', count)))


def remove_numerical_specks(obj):
    mi.discard_boolean_dust(obj)
    bm=bmesh.new();bm.from_mesh(obj.data);unseen=set(bm.verts);groups=[]
    try:
        while unseen:
            group=[];stack=[unseen.pop()]
            while stack:
                v=stack.pop();group.append(v)
                for e in v.link_edges:
                    other=e.other_vert(v)
                    if other in unseen:unseen.remove(other);stack.append(other)
            groups.append(group)
        groups.sort(key=len,reverse=True)
        tiny=[]
        for group in groups[1:]:
            span=np.linalg.norm(np.ptp(np.array([v.co[:] for v in group]),axis=0))
            if len(group)<=100 and span<.01:tiny.extend(group)
        if tiny:bmesh.ops.delete(bm,geom=tiny,context='VERTS');bm.to_mesh(obj.data)
    finally:bm.free()
    mi.closed(obj,connected=True)


def blend_nostril_junctions(work,core,junctions,tips):
    """Gradually relax the join, leaving each visible aperture protected."""
    openings=[BVHTree.FromPolygons(t.tolist(),[tuple(range(len(t)))]) for t in tips]
    group=core.vertex_groups.new(name='Temporary airway junction blend')
    rounded=work.copy(core,'Airway_junction_fillet_work')
    substructure.offset_solid(rounded,1.2,.16,surface_level=0.)
    substructure.offset_solid(rounded,-1.2,.16,surface_level=0.)
    rounded_tree=aw.world_tree(rounded)
    count=0
    try:
        for v in core.data.vertices:
            distance=min(max(a.find_nearest(v.co)[3],b.find_nearest(v.co)[3]) for a,b in junctions)
            aperture=min(t.find_nearest(v.co)[3] for t in openings)
            fade=max(0.,min(1.,(3.5-distance)/3.5))
            protect=max(0.,min(1.,(aperture-1.5)/2.))
            weight=(fade*fade*(3-2*fade))*(protect*protect*(3-2*protect))
            if weight>.001:
                q=rounded_tree.find_nearest(v.co)[0]
                if q is not None:v.co=v.co.lerp(q,weight)
                group.add([v.index],weight,'REPLACE');count+=1
        mu.apply_modifier(core,'SMOOTH',lambda m:(setattr(m,'factor',.5),setattr(m,'iterations',180),setattr(m,'vertex_group',group.name)))
    finally:
        remaining=core.vertex_groups.get('Temporary airway junction blend')
        if remaining is not None:core.vertex_groups.remove(remaining)
        work.delete(rounded)
    core['airway_junction_blended_vertices']=count


def outer_patch(work, sculpt):
    bm=bmesh.new();bm.from_mesh(sculpt.data);bm.transform(sculpt.matrix_world);bm.verts.ensure_lookup_table()
    attr=sculpt.data.attributes.get('anaplast_outer')
    if attr is not None and attr.domain=='POINT':
        mask=np.zeros(len(sculpt.data.vertices));attr.data.foreach_get('value',mask)
        if not np.any(mask>.5):bm.free();raise RuntimeError('The Sculpt has no marked outer surface')
        bmesh.ops.delete(bm,geom=[v for v in bm.verts if mask[v.index]<.5],context='VERTS')
    elif not any(e.is_boundary for e in bm.edges):
        bm.free();raise RuntimeError('Choose the original open nasal Sculpt or a Sculpt retaining its outer-surface attribute')
    patch=work.add(mu.new_mesh_object('Airway_outer_surface_work',bm,work.col))
    if not len(patch.data.polygons):raise RuntimeError('The nasal outer surface is empty')
    return patch


def limit_patient_outline(work, obj, root):
    """Keep the main cavity inside the patient's actual frontal opening.

    Upright cases use X across the face and Z vertically. A bounding rectangle
    admits material above skin at the curved corners and is not an opening.
    Nostril branches are constructed separately below this main-cavity limit.
    """
    profile=np.asarray(root)[:,(0,2)]
    from .mold_design import has_self_intersections
    frame=(Vector((0,-1,0)),Vector((1,0,0)),Vector((0,0,1)))
    lo,hi=mu.world_bbox(obj);depth=max(abs(lo.y),abs(hi.y),float(np.abs(root[:,1]).max()))+20.
    boundary=work.add(ma.prism('Airway_patient_outline_limit_work',profile,-depth,depth,frame,work.col))
    mi.closed(boundary,connected=True)
    if has_self_intersections(boundary):raise RuntimeError('The patient-side opening folds over itself in front view; revise its outline')
    aw.cut(obj,boundary,'INTERSECT')
    if not len(obj.data.polygons):raise RuntimeError('The marked patient opening leaves no main cavity at this wall thickness')
    return boundary


def route(work, body, surface, tip, number):
    tree=aw.world_tree(body);co=np.array([body.matrix_world@v.co for v in body.data.vertices]);low,high=co.min(0),co.max(0);span=high-low
    start=tip.mean(0);offsets=tip-start;candidates=[]
    # Search the interior of the cavity, not a straight line to the base mark.
    for x in np.linspace(low[0]+span[0]*.25,high[0]-span[0]*.25,9):
        for y in np.linspace(low[1]+span[1]*.35,low[1]+span[1]*.75,18):
            for z in np.linspace(low[2]+span[2]*.12,low[2]+span[2]*.65,18):
                p=Vector((x,y,z));near,normal,_,depth=tree.find_nearest(p)
                if near is None or (p-near).dot(normal)>=0 or depth<2:continue
                if surface.find_nearest(p)[3]<7.5:continue
                candidates.append((float(np.linalg.norm(np.array(p)-start)),np.array(p)))
    if not candidates:raise RuntimeError('No roomy cavity connection was found inside the marked limits; inspect the Sculpt and patient opening')
    candidates.sort(key=lambda v:v[0]);ends=[p for _,p in candidates[::max(1,len(candidates)//8)][:8]]
    aperture=BVHTree.FromPolygons(tip.tolist(),[tuple(range(len(tip)))])
    def ring(t,control,end,stride=1):
        center=(1-t)**2*start+2*t*(1-t)*control+t*t*end
        first=Vector(control-start).normalized();tangent=Vector((1-t)*(control-start)+t*(end-control)).normalized()
        rotation=np.array(first.rotation_difference(tangent).to_matrix())
        return center+offsets[::stride]@rotation.T
    best=None
    for end in ends:
        for dx in (-3.,0.,3.):
            for dy in (-3.,0.,3.):
                for dz in (5.,8.,11.,14.):
                    control=start+np.array((dx,dy,dz));values=[];curvature=[]
                    for t in np.linspace(.12,.95,12):
                        for p in ring(t,control,end,16):
                            p=Vector(p)
                            if aperture.find_nearest(p)[3]<4.1:continue
                            q,n,_,d=surface.find_nearest(p);values.append(d if (p-q).dot(n)<0 else -d)
                    for t in np.linspace(0,1,20):
                        vel=2*((1-t)*(control-start)+t*(end-control));acc=2*(end-2*control+start)
                        curvature.append(np.linalg.norm(vel)**3/max(1e-6,np.linalg.norm(np.cross(vel,acc))))
                    score=(min(values) if values else -100.)-max(0.,8.-min(curvature))*2.
                    if best is None or score>best[0]:best=(score,control,end)
    score,control,end=best
    rings=[ring(t,control,end) for t in np.linspace(0,1,65)]
    rings.insert(0,tip-(control-start)/np.linalg.norm(control-start)*.4)
    obj=mi.loft(work,f'Curved_nostril_{number}_work',rings)
    if mi.intersection_volume(work,body,obj)<.05:raise RuntimeError('A curved nostril did not reach the main cavity')
    return obj,dict(nostril=number,start=start.tolist(),control=control.tolist(),end=end.tolist(),section_scale=1.)


def cores(work,item,frame):
    sculpt=aw.reference(work.scene.anaplast);base=bpy.data.objects.get('Mold_Base');ref=bpy.data.objects.get('Common_Base_Reference')
    if sculpt is None or base is None or ref is None:raise RuntimeError('Choose the nasal Sculpt and build the Mold Base first')
    patch=outer_patch(work,sculpt);surface=aw.world_tree(patch)
    target=float(item.airway_wall);inset=target+.2
    domain=work.add(ma.sculpt_solid_toward('Airway_inner_domain_work',patch,frame[0],70,mu.world_bbox(sculpt),work.col,max_faces=1500000))
    substructure.offset_solid(domain,-inset,.2,surface_level=0.)
    if not len(domain.data.polygons):raise RuntimeError('No nasal cavity remains at this silicone thickness')
    u=np.array(frame[0]);refco=np.array([ref.matrix_world@v.co for v in ref.data.vertices])
    crop=work.add(ma.prism('Airway_base_crop_work',np.array(base['mold_hull']),float((refco@u).min())+.1,float((refco@u).max())+80,frame,work.col))
    body=work.copy(domain,'Airway_cavity_work');aw.cut(body,crop,'INTERSECT');aw.cut(body,ref,'DIFFERENCE')
    tips=[marks.outline(item,'TIP1'),marks.outline(item,'TIP2')]
    roots=[marks.outline(item,'BASE1')]
    if item.chambers=='SEPARATE':roots.append(marks.outline(item,'BASE2'))
    coords=np.array([v.co[:] for v in body.data.vertices])
    if not len(coords):raise RuntimeError('No nasal cavity remains above the fitting surface')
    low,high=coords.min(0)-3,coords.max(0)+3
    X,Y=np.meshgrid(np.arange(low[0],high[0]+.5,.5),np.arange(low[1],high[1]+.5,.5))
    H=np.minimum.reduce([tip[:,2].min()+1.+.18*np.sqrt((X-tip[:,0].mean())**2+(Y-tip[:,1].mean())**2) for tip in tips])
    floor=work.add(ma.heightfield_slab('Airway_sloped_floor_work',X,Y,H,np.ones_like(X,bool),100,(Vector((0,0,1)),Vector((1,0,0)),Vector((0,1,0))),work.col))
    aw.cut(body,floor,'DIFFERENCE');mu.voxel_remesh(body,.2);smooth(body,180)
    for iteration in range(3):
        for v in body.data.vertices:
            q,n,_,d=surface.find_nearest(v.co)
            if 1e-6<d<target+.05:v.co=q+(v.co-q).normalized()*(target+.05)
        body.data.update()
        if iteration<2:smooth(body,10)
    result=[];reports=[]
    for index,root in enumerate(roots):
        core=work.copy(body,'Airway_curved_core_work');box=limit_patient_outline(work,core,root)
        mu.voxel_remesh(core,.15);smooth(core,80)
        selected=list(enumerate(tips,1)) if len(roots)==1 else [(index+1,tips[index])]
        routes=[];junctions=[];allowed=work.copy(box,'Airway_allowed_footprint_work')
        for number,tip in selected:
            port,report=route(work,core,surface,tip,number);routes.append(report)
            junctions.append((aw.world_tree(core),aw.world_tree(port)))
            allowed=mi.merge(work,allowed,work.copy(port,'Airway_allowed_nostril_work'))
            core=mi.merge(work,core,port)
        portal=mi.loft(work,'Airway_patient_connection_work',[root+u*t for t in np.linspace(-.6,8.,40)])
        aw.cut(portal,domain,'INTERSECT');aw.cut(portal,box,'INTERSECT');core=mi.merge(work,core,portal)
        mu.voxel_remesh(core,.16);smooth(core,100)
        blend_nostril_junctions(work,core,junctions,[t for _,t in selected])
        remove_numerical_specks(core)
        from .mold_design import has_self_intersections
        if has_self_intersections(core):
            mu.voxel_remesh(core,.12);smooth(core,12)
            core['airway_mount_cleanup_pending']=True
            fill_mount_pockets(core);del core['airway_mount_cleanup_pending']
            remove_numerical_specks(core)
        # Blending may never spread the main core back over the marked skin.
        # Only the explicitly built nostril passages may leave this silhouette.
        aw.cut(core,allowed,'INTERSECT');remove_numerical_specks(core)
        if has_self_intersections(core):raise RuntimeError('The curved airway contains folded faces; revise its opening outlines')
        # Measure vertices and triangle centres, explicitly excluding only the
        # local opening transition. This is a report, not an accuracy promise.
        aps=[BVHTree.FromPolygons(t.tolist(),[tuple(range(len(t)))]) for _,t in selected]
        core.data.calc_loop_triangles();minimum=float('inf')
        for point in [v.co for v in core.data.vertices]+[t.center for t in core.data.loop_triangles]:
            if min(a.find_nearest(point)[3] for a in aps)<4.1:continue
            minimum=min(minimum,surface.find_nearest(point)[3])
        report=dict(algorithm='CURVED_CAVITY_TEST',target_wall_mm=target,sampled_wall_mm=minimum,opening_transition_excluded_mm=4.1,removal_validated=False,nostril_routes=routes,junction_blend_band_mm=3.5,junction_blended_vertices=core.get('airway_junction_blended_vertices',0),patient_boundary='ACTUAL_OUTLINE',patient_boundary_enforced_after_blending=True)
        core['airway_cavity_report']=json.dumps(report);core['airway_path']='CURVED'
        for face in core.data.polygons:face.use_smooth=True
        result.append(core);reports.append(report)
    if len(result)>1 and mi.intersection_volume(work,*result)>.001:raise RuntimeError('Separate curved chambers overlap; adjust the patient outlines or select a shared chamber')
    center=Vector(np.mean(np.vstack(roots),axis=0))
    return center,item.dock_diameter,result,[Vector(t.mean(0)) for t in tips]


def attach(work,item,insert,core,base,host,cap,center,frame,apply):
    """Compact blank seat, with a support reaching the nasal core from behind."""
    u,r,f=frame;hit,normal,_,_=aw.world_tree(core).find_nearest(center)
    # The marked point may lie above a recessed / hollow mold. Anchor into
    # the actual removable plug, rather than assuming material 2 mm below it.
    tree=aw.world_tree(insert);low,high=mi.bounds(insert,u)
    xy=center-u*center.dot(u)
    top=tree.ray_cast(xy+u*(high+1),-u,high-low+2)[0]
    if top is None:
        point,n,_,_=tree.find_nearest(center)
        start=point-n*.25
    else:
        bottom=tree.ray_cast(top-u*.0001,-u,high-low+2)[0]
        depth=min(1.,(top-bottom).length*.5) if bottom is not None else .25
        start=top-u*depth
    end=hit-normal*2.;axis=(end-start).normalized();right=r-axis*r.dot(axis)
    if right.length<.01:right=f-axis*f.dot(axis)
    right.normalize();support=mi.cylinder(work,'Airway_mount_support_work',start,end,min(4.5,item.dock_diameter*.35),(axis,right,axis.cross(right)))
    insert=mount_union(work,insert,support)
    insert=mount_union(work,insert,work.copy(core,'Airway_attachment_work'))
    if apply:
        aw.cut(host,insert,'DIFFERENCE')
        if cap is not None:
            contact=work.copy(insert,'Airway_cap_contact_work');aw.cut(contact,cap,'INTERSECT')
            if len(contact.data.vertices):
                # A non-planar marked nostril is not a flat polygon. Test its
                # short swept collar, retaining the marked perimeter, rather
                # than rejecting normal cap contact by distance to that plane.
                trees=[];collars=[]
                for path in json.loads(core['airway_cavity_report'])['nostril_routes']:
                    tip=marks.outline(item,'TIP'+str(path['nostril']));start=np.array(path['start']);control=np.array(path['control']);end=np.array(path['end']);offsets=tip-start
                    first=Vector(control-start).normalized();rings=[tip-np.array(first)*.6]
                    for t in np.linspace(0,1,129):
                        centerline=(1-t)**2*start+2*t*(1-t)*control+t*t*end
                        tangent=Vector((1-t)*(control-start)+t*(end-control)).normalized()
                        rotation=np.array(first.rotation_difference(tangent).to_matrix())
                        rings.append(centerline+offsets@rotation.T)
                        if np.linalg.norm(centerline-start)>=4.1:break
                    collar=mi.loft(work,'Airway_nostril_contact_collar_work',rings);collars.append(collar);trees.append(aw.world_tree(collar))
                def permitted(point):
                    for tree in trees:
                        q,n,_,distance=tree.find_nearest(point)
                        if distance<=.25 or (point-q).dot(n)<0:return True
                    return False
                allowed=all(permitted(contact.matrix_world@v.co) for v in contact.data.vertices)
                for collar in collars:work.delete(collar)
                if not allowed:
                    raise RuntimeError('The curved airway crosses the cap away from its nostrils; inspect the preview before rebuilding')
            work.delete(contact);aw.cut(cap,insert,'DIFFERENCE')
    remove_numerical_specks(insert)
    return insert


def mount_union(work,left,right):
    from . import mold_design, mold_insert_reservoir
    original=left.data.copy();errors=[];fallback=None
    try:
        for solver in ('MANIFOLD','EXACT'):
            previous=left.data;left.data=original.copy()
            if previous.users==0:bpy.data.meshes.remove(previous)
            try:
                mu.apply_boolean(left,right,'UNION',solver=solver)
                fill_mount_pockets(left);remove_numerical_specks(left)
                if fallback is None:fallback=left.data.copy()
                if mold_design.has_self_intersections(left):mold_insert_reservoir.clean_seams(left)
                if mold_design.has_self_intersections(left):raise RuntimeError('crossing faces')
                work.delete(right);return left
            except RuntimeError as exc:errors.append(solver+': '+str(exc))
        # Only this experimental airway mount may be resampled. Flow bores
        # are cut afterward. Bound displacement and reject folding/disconnection.
        if fallback is not None:
            previous=left.data;left.data=fallback.copy()
            if previous.users==0:bpy.data.meshes.remove(previous)
            reference=aw.world_tree(left)
            mu.voxel_remesh(left,.08)
            left['airway_mount_cleanup_pending']=True
            fill_mount_pockets(left)
            del left['airway_mount_cleanup_pending']
            remove_numerical_specks(left)
            left.data.calc_loop_triangles()
            triangles=len(left.data.loop_triangles)
            if triangles>250000:
                mu.apply_modifier(left,'DECIMATE',lambda m:setattr(m,'ratio',250000/triangles))
                remove_numerical_specks(left)
            deviation=max(reference.find_nearest(left.matrix_world@v.co)[3] for v in left.data.vertices)
            if deviation<=.16 and not mold_design.has_self_intersections(left):
                left['airway_mount_resampled_mm']=.08
                left['airway_mount_sampled_deviation_mm']=deviation
                work.delete(right);return left
        raise RuntimeError('Airway mounting union failed: '+'; '.join(errors))
    finally:
        if original.users==0:bpy.data.meshes.remove(original)
        if fallback is not None and fallback.users==0:bpy.data.meshes.remove(fallback)


def fill_mount_pockets(obj):
    """Fill only tiny enclosed union pockets before any flow channels are cut.

    Preserve every exterior vertex. Disconnected exterior material still fails.
    """
    bm=bmesh.new();bm.from_mesh(obj.data);unseen=set(bm.verts);groups=[]
    try:
        while unseen:
            group=[];stack=[unseen.pop()]
            while stack:
                v=stack.pop();group.append(v)
                for e in v.link_edges:
                    other=e.other_vert(v)
                    if other in unseen:unseen.remove(other);stack.append(other)
            groups.append(group)
        groups.sort(key=len,reverse=True)
        if len(groups)<2:return
        outer=groups[0];ids={v:i for i,v in enumerate(outer)}
        faces={f for v in outer for f in v.link_faces}
        tree=BVHTree.FromPolygons([v.co for v in outer],[[ids[v] for v in f.verts] for f in faces])
        def inside(p):
            for direction in (Vector((1,.371,.193)).normalized(),Vector((.219,1,.417)).normalized(),Vector((.327,.131,1)).normalized()):
                origin=p.copy();count=0
                for _ in range(256):
                    hit=tree.ray_cast(origin,direction)
                    if hit[0] is None:break
                    count+=1;origin=hit[0]+direction*.00001
                else:return False
                if count%2!=1:return False
            return True
        removed=[]
        for group in groups[1:]:
            co=np.array([v.co[:] for v in group]);center=co.mean(0)
            if np.linalg.norm(np.ptp(co,axis=0))>2.5:continue
            faces={f for v in group for f in v.link_faces};volume=0.
            for face in faces:
                p=[np.array(v.co[:])-center for v in face.verts]
                for i in range(1,len(p)-1):volume+=np.dot(p[0],np.cross(p[i],p[i+1]))/6.
            enclosed=abs(volume)<.1 and all(inside(v.co) for v in group)
            remesh_chip=(obj.get('airway_mount_cleanup_pending',False) and len(group)<=100 and abs(volume)<.0005 and np.linalg.norm(np.ptp(co,axis=0))<.5)
            if enclosed or remesh_chip:removed.extend(group)
        if removed:
            bmesh.ops.delete(bm,geom=removed,context='VERTS');bm.to_mesh(obj.data);obj.data.update()
            obj['mount_internal_pocket_vertices_removed']=len(removed)
    finally:bm.free()


def reservoir_ports(item,core,center,base,frame):
    from . import mold_insert_reservoir as compact
    u,r,f=frame;d=compact.dimensions(item);xy=center-u*center.dot(u)
    d.update(flow_frame=frame,xy=xy)
    co=np.array([core.matrix_world@v.co for v in core.data.vertices]);offset=(co-np.array(xy))@np.array(r)
    d['feed_exit']=max(d['radius']+3.,float(-offset.min())+1.);vent_exit=max(d['radius']+3.,float(offset.max())+1.)
    tree=aw.world_tree(base);lo,hi=mi.bounds(base,u)
    heights=lambda sign,length:[mi.column(tree,xy+r*(sign*float(t)),frame,lo,hi)[1] for t in np.linspace(d['distance']*.5,length,32)]
    d['feed_z']=max(heights(-1,d['feed_exit']))+d['feed']+.3;d['vent_z']=max(heights(1,vent_exit))+d['vent']+.3
    d['vent_branch']=[xy+r*t+u*d['vent_z'] for t in (d['distance']*.5,vent_exit)]
    d['boss_top']=max(d['feed_z']+d['feed'],d['vent_z']+d['vent'])+d['wall']
    return d
