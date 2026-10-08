"""Preserve the evaluated Sculpt exterior; construct and validate its fitting side."""
import bpy,bmesh,numpy as np
from mathutils import Vector,Matrix
from mathutils.bvhtree import BVHTree
from . import mesh as mu


def smooth_inner(values,edges,distance,fixed=None):
    """Implicit graph smoothing at a physical scale; never applied to the outer mesh."""
    n=len(values);a,b=edges.T;count=np.bincount(edges.ravel(),minlength=n)
    length=np.median(np.linalg.norm(values[a]-values[b],axis=1))
    weight=(distance/max(length,1e-5))**2/4
    free=np.ones(n,dtype=bool) if fixed is None else ~fixed
    def apply(x):
        total=np.column_stack([np.bincount(a,weights=x[b,i],minlength=n)+np.bincount(b,weights=x[a,i],minlength=n) for i in range(3)])
        return x+weight*(count[:,None]*x-total)
    x=values.copy();residual=values-apply(x);residual[~free]=0
    direction=residual.copy();rr=float(np.sum(residual*residual));initial=max(rr,1e-30)
    for _ in range(200):
        if rr<initial*1e-12:break
        product=apply(direction);product[~free]=0
        alpha=rr/max(float(np.sum(direction*product)),1e-30)
        x+=alpha*direction;residual-=alpha*product
        new_rr=float(np.sum(residual*residual));direction=residual+(new_rr/max(rr,1e-30))*direction;rr=new_rr
    return x


def validate_relief_reference(obj):
    """Folded relief references require the checked reconstruction path."""
    from ..ops.mold_design import has_self_intersections
    if has_self_intersections(obj):
        raise RuntimeError('The offset fitting reference intersects itself')


def build(source,scan,p,col,name='Prototype_building'):
    """Return a new validated mesh; never mutate the Sculpt, scan or old Prototype."""
    obj=None
    before=set(bpy.data.objects)
    reference=None
    try:
        if p.despeckle_enabled and p.despeckle_mm>0:
            dg=bpy.context.evaluated_depsgraph_get()
            me=bpy.data.meshes.new_from_object(source.evaluated_get(dg),preserve_all_data_layers=True,depsgraph=dg)
            reference=bpy.data.objects.new('Prototype_filtered_reference',me);col.objects.link(reference)
            reference.matrix_world=source.matrix_world.copy()
            if source.get('sculpt_temporary_backing'):reference['sculpt_temporary_backing']=True
            if source.get('sculpt_crop_capped'):reference['sculpt_crop_capped']=True
            mu.despeckle(reference,p.despeckle_mm)
            source=reference
        attempt_before=set(bpy.data.objects)
        # Invalid source triangles cannot be retained verbatim in a valid solid.
        from ..ops.mold_design import has_self_intersections
        reason=''
        if has_self_intersections(source):
            reason='Sculpt contains intersecting faces'
        if not reason:
            try:
                obj=_build(source,scan,p,col,name)
            except RuntimeError as exc:
                reason=str(exc)
                for temporary in set(bpy.data.objects)-attempt_before:
                    if temporary.type=='MESH':mu.delete_object(temporary)
        if reason:
            obj=build_recovered(source,scan,p,col,name)
            obj['prototype_recovery_reason']=reason
        return obj
    except Exception:
        # This builder is synchronous. Only its new temporary objects are removed.
        for temporary in set(bpy.data.objects)-before:
            if temporary.type=='MESH':
                if temporary==reference:reference=None
                mu.delete_object(temporary)
        raise
    finally:
        if reference is not None and reference in set(bpy.data.objects):mu.delete_object(reference)


def _build(source,scan,p,col,name):
    dg=bpy.context.evaluated_depsgraph_get()
    me=bpy.data.meshes.new_from_object(source.evaluated_get(dg),preserve_all_data_layers=True,depsgraph=dg)
    obj=bpy.data.objects.new(name,me);col.objects.link(obj);obj.matrix_world=source.matrix_world.copy()
    from ..ops.sculpt_prepare import remove_backing
    if source.get('sculpt_temporary_backing'):
        obj['sculpt_temporary_backing']=True
        if source.get('sculpt_crop_capped'):obj['sculpt_crop_capped']=True
        remove_backing(obj)
    bm=bmesh.new();bm.from_mesh(me)
    try:
        bm.transform(obj.matrix_world);bm.verts.ensure_lookup_table();bm.verts.index_update();bm.normal_update()
        if not bm.faces or any(len(e.link_faces)>2 or not e.link_faces for e in bm.edges):
            raise RuntimeError('Sculpt has branching or loose geometry; repair that area before building')
        border=[e for e in bm.edges if e.is_boundary]
        if not border:raise RuntimeError('Select the open Sculpt surface')
        if any(sum(e.is_boundary for e in v.link_edges)!=2 for e in border for v in e.verts):
            raise RuntimeError('Source boundary branches or touches itself')
        # Adding custom-data layers can invalidate BMesh element handles.
        outer_layer=bm.verts.layers.float.get('anaplast_outer') or bm.verts.layers.float.new('anaplast_outer')
        region=bm.faces.layers.int.get('prototype_region') or bm.faces.layers.int.new('prototype_region')
        bm.verts.ensure_lookup_table();bm.verts.index_update()
        border=[e for e in bm.edges if e.is_boundary]
        # Orient the copy against the fitting scan; source coordinates stay intact.
        tree=mu.oriented_scan_bvh(scan,p.scan_flip,roi=mu.world_bbox(obj))
        faces=list(bm.faces)
        agreement=0.
        for face in faces[::max(1,len(faces)//2000)]:
            _,normal,_,_=tree.find_nearest(face.calc_center_median())
            if normal is not None:agreement+=face.normal.dot(normal)
        if agreement<0:
            for face in faces:face.normal_flip()
            bm.normal_update()
        verts=list(bm.verts);faces=list(bm.faces);face_set=set(faces);n=len(verts)
        pos=np.array([v.co[:] for v in verts]);normals=np.array([v.normal[:] for v in verts])
        edges=np.array([[e.verts[0].index,e.verts[1].index] for e in bm.edges]);a,b=edges.T
        boundary=np.array([v.is_boundary for v in verts]);count=np.bincount(edges.ravel(),minlength=n)
        smooth=smooth_inner(pos,edges,p.shell_thickness*.75)
        # Compute normals on the smoothed inner reference, keeping source points.
        for v,q in zip(verts,smooth):v.co=q
        bm.normal_update();normals=np.array([v.normal[:] for v in verts])
        for v,q in zip(verts,pos):v.co=q
        bm.normal_update()
        if p.feather_width:
            distances=mu.boundary_distances(bm)
            t=np.clip(np.array([distances.get(v.index,float('inf')) for v in verts])/p.feather_width,0,1)
            thickness=p.edge_thickness+(p.shell_thickness-p.edge_thickness)*(t*t*(3-2*t))
        else:thickness=np.full(n,p.shell_thickness)
        # Work on the fitting side only. Original outer faces and custom data stay.
        # Normal-only smoothing avoids pulling the inner perimeter sideways:
        # a shrunken inner perimeter makes the stitched rim cut through pores.
        displacement=np.sum((smooth-pos)*normals,axis=1)
        depth=np.maximum(min(p.edge_thickness,p.shell_thickness)*.5,thickness-displacement)
        inner=[bm.verts.new(Vector(q)) for q in pos-normals*depth[:,None]]
        for v in verts:v[outer_layer]=1.
        for v in inner:v[outer_layer]=0.
        for face in faces:
            new=bm.faces.new([inner[v.index] for v in reversed(face.verts)])
            new.smooth=face.smooth;new.material_index=face.material_index;new[region]=1;face[region]=0
        for edge in border:
            loop=next(l for l in edge.link_loops if l.face in face_set)
            i,j=loop.vert.index,loop.link_loop_next.vert.index
            face=bm.faces.new((verts[j],verts[i],inner[i],inner[j]));face[region]=2
        bmesh.ops.recalc_face_normals(bm,faces=bm.faces)
        bm.transform(obj.matrix_world.inverted());bm.to_mesh(me);me.update()
    finally:bm.free()
    if not mu.is_closed(obj):raise RuntimeError('Preserved surface did not produce a closed shell')
    cutter,temp=mu.solid_copy(scan,p.scan_thickness,p.scan_flip,roi=mu.world_bbox(obj),max_faces=max(150000,len(scan.data.polygons)))
    try:
        if not mu.is_closed(cutter):raise RuntimeError('Fitting scan could not form a closed cutting copy')
        if p.fit_offset>0:
            if not temp:
                cutter=mu.duplicate_object(cutter,'Prototype_relief_copy',col);temp=True
            # Offset only a temporary fitting reference, in world millimetres.
            cb=bmesh.new();cb.from_mesh(cutter.data);cb.transform(cutter.matrix_world);cb.normal_update()
            for v in cb.verts:v.co+=v.normal*p.fit_offset
            cb.transform(cutter.matrix_world.inverted());cb.to_mesh(cutter.data);cb.free();cutter.data.update()
            validate_relief_reference(cutter)
        mu.apply_boolean(obj,cutter,'DIFFERENCE',solver='MANIFOLD')
    finally:
        if temp:mu.delete_object(cutter)
    # Exact repair can create nonmanifold junctions on folded sheets. Use the
    # checked reconstruction path instead when the preserved shell intersects.
    repaired=False
    stats=mu.mesh_stats(obj)
    if not stats[0] or any(stats[1:]):raise RuntimeError('Preserved Prototype is empty or not manifold: '+str(stats))
    from ..ops.mold_design import has_self_intersections
    if has_self_intersections(obj):
        raise RuntimeError('Prototype still contains intersecting surfaces; the previous Prototype was kept')
    obj['prototype_detail_method']='Preserved Sculpt surface'
    obj['prototype_local_repair']=repaired
    obj['prototype_validated']=True
    obj['anaplast_part']='SHELL'
    return obj


def intersection_vertices(obj):
    """Indices implicated by nonadjacent triangle intersections, in world space."""
    me=obj.data;me.calc_loop_triangles()
    points=[obj.matrix_world@v.co for v in me.vertices]
    triangles=[tuple(t.vertices) for t in me.loop_triangles]
    tree=BVHTree.FromPolygons(points,triangles,all_triangles=True)
    bad=set()
    for a,b in tree.overlap(tree):
        if a<b and set(triangles[a]).isdisjoint(triangles[b]):
            bad.update(triangles[a]);bad.update(triangles[b])
    return bad


def coordinates(obj):
    values=np.empty(len(obj.data.vertices)*3)
    obj.data.vertices.foreach_get('co',values)
    return values.reshape(-1,3)


def limit_folds(obj,before):
    """Back off only edits near intersections; baseline must already be valid."""
    after=coordinates(obj);weights=np.ones(len(before));limited=set()
    for step in range(12):
        bad=intersection_vertices(obj)
        if not bad:return len(limited)
        # Include direct neighbours so no needle-shaped transition is left.
        neighbours=set(bad)
        for edge in obj.data.edges:
            a,b=edge.vertices
            if a in bad or b in bad:neighbours.update((a,b))
        ix=np.array(sorted(neighbours));limited.update(neighbours)
        weights[ix]=weights[ix]*.5 if step<8 else 0.
        obj.data.vertices.foreach_set('co',(before+(after-before)*weights[:,None]).ravel())
        obj.data.update()
    # A completely safe baseline is preferable to silently returning a fold.
    obj.data.vertices.foreach_set('co',before.ravel());obj.data.update()
    return len(before)


def build_recovered(source,scan,p,col,name):
    """Resolve the shell first, then refine and recover its outer detail only."""
    obj=mu.duplicate_object(source,name,col)
    # Bake evaluated subdivisions before any operation that replaces the mesh.
    dg=bpy.context.evaluated_depsgraph_get()
    me=bpy.data.meshes.new_from_object(source.evaluated_get(dg),preserve_all_data_layers=True,depsgraph=dg)
    old=obj.data;obj.modifiers.clear();obj.data=me
    if old.users==0:bpy.data.meshes.remove(old)
    from ..ops.sculpt_prepare import remove_backing
    remove_backing(obj)
    if not len(obj.data.polygons):raise RuntimeError('The Sculpt is empty')
    if mu.mesh_stats(obj)[2]:
        mu.voxel_fit(obj,scan,p.scan_flip,wall=p.shell_thickness,
                     clearance=p.shell_clearance,band=0.,voxel=p.voxel_size,
                     scan_thickness=p.scan_thickness,solver=p.boolean_solver,
                     feather=0.,edge_thickness=p.edge_thickness,fit_offset=0.,apply_despeckle=False)
    else:
        # Preserve support for imported closed Sculpts as well as cropped sheets.
        mu.apply_solidify(obj,p.shell_thickness)
        cutter,temp=mu.solid_copy(scan,p.scan_thickness,p.scan_flip,roi=mu.world_bbox(obj))
        try:mu.apply_boolean(obj,cutter,'DIFFERENCE',solver=p.boolean_solver)
        finally:
            if temp:mu.delete_object(cutter)
        mu.voxel_remesh(obj,p.voxel_size)
    bm=bmesh.new();bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm,faces=list(bm.faces))
    bm.to_mesh(obj.data);bm.free();obj.data.update()
    stats=mu.mesh_stats(obj)
    if not stats[0] or any(stats[1:]) or intersection_vertices(obj):
        raise RuntimeError('Could not construct a clean fitting shell; previous Prototype kept')
    print('Prototype: clean shell constructed',flush=True)
    limited=0
    # Label before refining; labels interpolate onto the new outer vertices.
    mu.restore_detail(obj,source,max_dist=max(.6,4*p.voxel_size),snap=False)
    bm=bmesh.new();bm.from_mesh(obj.data);bm.transform(obj.matrix_world)
    # Split planar triangles, so refinement itself cannot change a nonplanar
    # quad's diagonal and introduce folds before any projection occurs.
    bmesh.ops.triangulate(bm,faces=list(bm.faces))
    outer=bm.verts.layers.float.get('anaplast_outer')
    sm=bmesh.new();sm.from_object(source,bpy.context.evaluated_depsgraph_get());sm.transform(source.matrix_world)
    lengths=np.array([e.calc_length() for e in sm.edges]);lengths=lengths[lengths>1e-5]
    target=float(np.percentile(lengths,35)) if len(lengths) else p.voxel_size/2
    target=max(.015,min(.1,target))
    tree=BVHTree.FromBMesh(sm);sm.free()
    try:
        for _ in range(5):
            edges=[e for e in bm.edges if all(v[outer]>.5 for v in e.verts) and e.calc_length()>target*1.5]
            if not edges:break
            if len(bm.faces)+len(edges)*4>4000000:
                raise RuntimeError('Detail recovery exceeds four million faces; reduce the cropped Sculpt area')
            bmesh.ops.subdivide_edges(bm,edges=edges,cuts=1,use_grid_fill=False,use_single_edge=True)
            bmesh.ops.triangulate(bm,faces=[f for f in bm.faces if len(f.verts)>3])
        bm.transform(obj.matrix_world.inverted());bm.to_mesh(obj.data);obj.data.update()
    finally:bm.free()
    print('Prototype: exterior refined',len(obj.data.polygons),'faces',flush=True)
    if intersection_vertices(obj):
        raise RuntimeError('Refinement intersects before detail transfer')
    if p.feather_width>0:
        before=coordinates(obj)
        mu.feather_edge_exact(obj,source,p.feather_width,p.edge_thickness,p.shell_thickness)
        limited+=limit_folds(obj,before)
        print('Prototype: edge taper checked',flush=True)
    before=coordinates(obj);me=obj.data;mw=obj.matrix_world;inv=mw.inverted()
    normal_matrix=mw.to_3x3().inverted().transposed()
    normals=np.empty(len(me.vertices)*3);me.vertices.foreach_get('normal',normals)
    normals=normals.reshape(-1,3)
    labels=np.empty(len(me.vertices));me.attributes['anaplast_outer'].data.foreach_get('value',labels)
    max_shift=max(.15,2*p.voxel_size)
    moved=0
    for i,v in enumerate(me.vertices):
        if labels[i]<.99:continue
        q=mw@Vector(before[i]);n=(normal_matrix@Vector(normals[i])).normalized()
        hits=[]
        for sign in (1.,-1.):
            hit,hn,_,dist=tree.ray_cast(q,n*sign,max_shift)
            if hit is not None and hn.dot(n)>.5:hits.append(((hit-q).length,hit))
        if not hits:continue
        _,hit=min(hits,key=lambda value:value[0])
        v.co=inv@hit;moved+=1
    me.update();limited+=limit_folds(obj,before)
    if p.fit_offset>0:
        before=coordinates(obj)
        mu.resnap_fitting(obj,scan,p.scan_flip,p.fit_offset)
        limited+=limit_folds(obj,before)
    stats=mu.mesh_stats(obj)
    if not stats[0] or any(stats[1:]) or intersection_vertices(obj):
        raise RuntimeError('Detail recovery did not produce a clean solid; previous Prototype kept ('+str(stats)+')')
    obj['anaplast_part']='SHELL'
    obj['prototype_detail_method']='Refined and recovered Sculpt surface'
    obj['prototype_detail_spacing']=target
    obj['prototype_detail_projected']=moved
    obj['prototype_detail_limited']=limited
    obj['prototype_validated']=True
    return obj


def tree_world(obj):
    bm=bmesh.new();bm.from_mesh(obj.data);bm.transform(obj.matrix_world)
    tree=BVHTree.FromBMesh(bm);bm.free();return tree


def deviation(source,result,limit=16000):
    tree=tree_world(result);values=[]
    for i in range(0,len(source.data.vertices),max(1,len(source.data.vertices)//limit)):
        q=source.matrix_world@source.data.vertices[i].co;hit=tree.find_nearest(q)
        values.append(hit[3] if hit[0] is not None else float('inf'))
    return {'samples':len(values),'median_mm':float(np.median(values)),'p95_mm':float(np.percentile(values,95)),'max_mm':float(max(values))}
