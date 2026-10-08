"""Close crops of solid meshes using the actual drawn cutting volume."""
import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from ..utils import mesh as mu


def is_solid(mesh):
    bm=bmesh.new()
    try:
        bm.from_mesh(mesh)
        return bool(bm.faces) and all(e.is_manifold for e in bm.edges) and all(v.link_faces for v in bm.verts)
    finally:bm.free()


def cutting_volume(mesh,projection,outline,axis):
    """Unproject the drawn polygon; cover the mesh's full depth in either view."""
    inv=Matrix(projection).inverted();axis=Vector(axis).normalized()
    co=np.empty(len(mesh.vertices)*3);mesh.vertices.foreach_get('co',co);co=co.reshape(-1,3)
    h=co@np.asarray(axis);padding=max(float(np.linalg.norm(np.ptp(co,axis=0)))*.1,1.)
    near,far=float(h.min()-padding),float(h.max()+padding)
    if len(outline)<3:raise RuntimeError('Draw a crop outline with at least three points')
    from mathutils.geometry import intersect_line_line_2d
    poly=[Vector(p) for p in outline]
    for i,a in enumerate(poly):
        b=poly[(i+1)%len(poly)]
        if (b-a).length<1e-9:raise RuntimeError('The crop outline contains repeated points; redraw the region')
        for j in range(i+2,len(poly)):
            if i==0 and j==len(poly)-1:continue
            if intersect_line_line_2d(a,b,poly[j],poly[(j+1)%len(poly)]) is not None:
                raise RuntimeError('The crop outline crosses itself; draw one non-crossing loop')
    rings=[[],[]]
    for x,y in outline:
        a=inv@Vector((x,y,-.5,1.));b=inv@Vector((x,y,.5,1.))
        if abs(a.w)<1e-12 or abs(b.w)<1e-12:raise RuntimeError('The crop view cannot be projected; redraw the region')
        a=a.xyz/a.w;b=b.xyz/b.w;direction=b-a;denom=direction.dot(axis)
        if abs(denom)<1e-12:raise RuntimeError('The crop rays are parallel to the depth plane; redraw the region')
        for ring,level in zip(rings,(near,far)):ring.append(a+direction*((level-a.dot(axis))/denom))
    bm=bmesh.new()
    try:
        one=[bm.verts.new(v) for v in rings[0]];two=[bm.verts.new(v) for v in rings[1]]
        ends=[bm.faces.new(list(reversed(one))),bm.faces.new(two)]
        for i in range(len(one)):
            j=(i+1)%len(one);bm.faces.new((one[i],one[j],two[j],two[i]))
        bmesh.ops.triangulate(bm,faces=ends)
        bmesh.ops.recalc_face_normals(bm,faces=bm.faces[:])
        layer=bm.faces.layers.int.new('anaplast_crop_cap')
        for f in bm.faces:f[layer]=1
        me=bpy.data.meshes.new('Crop_volume_tmp');bm.to_mesh(me);me.update();return me
    finally:bm.free()


def close_solid(context,source,*,remove=False):
    """Cut a temporary evaluated copy. Never remesh or change the source."""
    if not source.get('crop_projection') or not source.get('crop_outline_ndc'):
        raise RuntimeError('Redraw the crop region once, then use Close mesh; this older preview has no saved outline')
    target=tool=None;result=None
    try:
        target=bpy.data.objects.new('Crop_solid_tmp',source.data.copy());context.scene.collection.objects.link(target)
        target.matrix_world=source.matrix_world.copy();target['anaplast_construction']=True
        if not target.data.attributes.get('anaplast_crop_cap'):
            target.data.attributes.new('anaplast_crop_cap','INT','FACE')
        cutter=cutting_volume(source.data,source['crop_projection'],source['crop_outline_ndc'],source['crop_axis_local'])
        tool=bpy.data.objects.new('Crop_volume_tmp',cutter);context.scene.collection.objects.link(tool)
        tool.matrix_world=source.matrix_world.copy();tool['anaplast_construction']=True
        mod=target.modifiers.new('Close drawn crop','BOOLEAN')
        mod.operation='DIFFERENCE' if remove else 'INTERSECT'
        mod.solver=mu.pick_solver(mod,'EXACT',len(target.data.polygons)+len(cutter.polygons));mod.object=tool
        dg=context.evaluated_depsgraph_get()
        evaluated=bpy.data.meshes.new_from_object(target.evaluated_get(dg),preserve_all_data_layers=True,depsgraph=dg)
        target.modifiers.remove(mod);previous=target.data;target.data=evaluated;bpy.data.meshes.remove(previous)
        stats=mu.mesh_stats(target)
        if not stats[0]:raise RuntimeError('The crop would leave no faces; draw a different region')
        if any(stats[1:]):raise RuntimeError('The cut could not form a closed solid; the source and preview were kept')
        # Face labels propagate through Boolean geometry and let Prototype
        # remove the artificial cut walls from its own working copy later.
        labels=target.data.attributes.get('anaplast_crop_cap')
        if labels is None:raise RuntimeError('The crop lost its closure labels; the original mesh was kept')
        caps=sum(v.value!=0 for v in labels.data)
        result=target.data.copy();return result,caps
    finally:
        if tool is not None:mu.delete_object(tool)
        if target is not None:mu.delete_object(target)
