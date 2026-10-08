"""Keep reference scans intact and prepare a solid working sculpt after cropping."""
import bpy,bmesh,json
import numpy as np
from mathutils import Matrix,Vector
from mathutils.bvhtree import BVHTree
from ..utils import mesh as mu


def begin_crop(context,obj):
    if context.object and context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    if not obj.get('sculpt_crop_work'):
        original=obj
        obj=mu.duplicate_object(original,'Sculpt_Region',mu.get_collection(context.scene,'Anaplast_Sculpt'))
        obj['sculpt_crop_work']=True;obj['crop_reference']=original.name
        obj['anaplast_part']='SCULPT_REGION'
        original.hide_set(True)
    for o in context.selected_objects:o.select_set(False)
    obj.hide_set(False);obj.select_set(True);context.view_layer.objects.active=obj
    context.scene.anaplast.crop_target=obj
    return obj


def transfer_appearance(context,obj,source):
    """Re-sample existing UVs and vertex colors on the new topology."""
    if not source.data.uv_layers and not source.data.color_attributes:return
    mod=obj.modifiers.new('Transfer scan appearance','DATA_TRANSFER');mod.object=source
    mod.use_loop_data=True;mod.loop_mapping='POLYINTERP_NEAREST'
    kinds=set()
    if source.data.uv_layers:kinds.add('UV')
    if any(a.domain=='CORNER' for a in source.data.color_attributes):kinds.add('COLOR_CORNER')
    mod.data_types_loops=kinds
    if any(a.domain=='POINT' for a in source.data.color_attributes):
        mod.use_vert_data=True;mod.vert_mapping='POLYINTERP_NEAREST';mod.data_types_verts={'COLOR_VERTEX'}
    for o in context.selected_objects:o.select_set(False)
    obj.hide_set(False);obj.select_set(True);context.view_layer.objects.active=obj
    bpy.ops.object.datalayout_transfer(modifier=mod.name)
    bpy.ops.object.modifier_apply(modifier=mod.name)


def prepare(context,source):
    p=context.scene.anaplast
    if abs(context.scene.unit_settings.scale_length-.001)>1e-8:raise RuntimeError('Use millimetre case units before preparing the Sculpt')
    if source is None or source.type!='MESH':raise RuntimeError('Choose a cropped mesh first')
    if context.object and context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    if source.modifiers:raise RuntimeError('Apply or bake modifiers on the cropped working region first')
    obj=mu.duplicate_object(source,'Sculpt_Preparing',mu.get_collection(context.scene,'Anaplast_Sculpt'))
    try:
        obj.data.transform(obj.matrix_world);obj.matrix_world=Matrix.Identity(4)
        # A protected reference in world coordinates makes all distances millimetres.
        verts=[source.matrix_world@v.co for v in source.data.vertices]
        faces=[tuple(f.vertices) for f in source.data.polygons]
        tree=BVHTree.FromPolygons(verts,faces,all_triangles=False)
        opened=bool(mu.mesh_stats(obj)[2])
        if opened:
            def solidify(m):
                m.thickness=max(p.sculpt_backing,3*p.sculpt_voxel);m.offset=-1.;m.use_rim=True;m.use_even_offset=False
            mu.apply_modifier(obj,'SOLIDIFY',solidify)
        used=mu.voxel_remesh(obj,p.sculpt_voxel,adaptivity=0.)
        if not len(obj.data.polygons) or any(mu.mesh_stats(obj)[1:]):raise RuntimeError('Could not make a closed Sculpt; check the crop or use a finer voxel size')
        original=np.array([v.co[:] for v in obj.data.vertices]);outer=np.zeros(len(original),np.float32)
        normals=np.array([v.normal[:] for v in obj.data.vertices]);projected=original.copy()
        limit=used*1.5;moved=0
        for i,point in enumerate(original):
            loc,normal,face,dist=tree.find_nearest(Vector(point),limit)
            if loc is None or normal.dot(Vector(normals[i]))<.5:continue
            outer[i]=1.
            if p.sculpt_reproject:
                delta=np.asarray(loc)-point;length=np.linalg.norm(delta)
                projected[i]=point+delta*min(1.,.3*used/max(length,1e-12));moved+=1
        if p.sculpt_reproject:obj.data.vertices.foreach_set('co',projected.ravel())
        obj.data.update()
        from .mold_design import has_self_intersections
        fraction=1. if p.sculpt_reproject else 0.
        intersects=has_self_intersections(obj)
        if p.sculpt_reproject:
            while intersects and fraction>.125:
                fraction*=.5;obj.data.vertices.foreach_set('co',(original+(projected-original)*fraction).ravel());obj.data.update()
                intersects=has_self_intersections(obj)
        rejected=intersects and p.sculpt_reproject
        if rejected:
            obj.data.vertices.foreach_set('co',original.ravel());obj.data.update();moved=0;fraction=0.
            intersects=has_self_intersections(obj)
        if intersects:raise RuntimeError('Remeshed Sculpt has intersecting surfaces; use a finer voxel size or adjust the cropped region')
        attr=obj.data.attributes.get('anaplast_outer') or obj.data.attributes.new('anaplast_outer','FLOAT','POINT')
        attr.data.foreach_set('value',outer if opened else np.ones(len(outer),np.float32))
        transfer_appearance(context,obj,source)
        # Vertex samples in both directions detect lost relief between projected vertices too.
        newtree=BVHTree.FromObject(obj,context.evaluated_depsgraph_get())
        distances=[]
        for v in verts[::max(1,len(verts)//20000)]:
            hit=newtree.find_nearest(v)
            if hit[0] is not None:distances.append(hit[3])
        reference_distances=np.asarray(distances)
        reference_rms=float(np.sqrt(np.mean(reference_distances**2)));reference_max=float(reference_distances.max())
        for v,flag in zip(obj.data.vertices,outer):
            if flag:
                hit=tree.find_nearest(v.co)
                if hit[0] is not None:distances.append(hit[3])
        distances=np.asarray(distances);rms=float(np.sqrt(np.mean(distances**2)));worst=float(distances.max())
        obj.name='Sculpt';obj['anaplast_part']='SCULPT';obj['sculpt_reference']=source.name
        obj['sculpt_temporary_backing']=opened;obj['sculpt_voxel_mm']=used
        obj['sculpt_deviation']=json.dumps({'method':'bidirectional vertex-to-surface samples; not a full surface bound','samples':len(distances),'rms_mm':rms,'max_mm':worst,'reference_to_sculpt_rms_mm':reference_rms,'reference_to_sculpt_max_mm':reference_max,'reprojected_vertices':moved,'reprojection_rejected':rejected,'reprojection_fraction':fraction,'maximum_projection_move_mm':.3*used*fraction})
        # A final prototype extracts this marked outer surface and builds a new fitting face.
        obj.pop('sculpt_crop_work',None)
        for face in obj.data.polygons:face.use_smooth=True
        source.hide_set(True);source.hide_render=True;obj.hide_set(False);obj.hide_render=False
        for selected in context.selected_objects:selected.select_set(False)
        obj.select_set(True);context.view_layer.objects.active=obj
        p.prosthesis_obj=obj;p.source_prosthesis=obj;p.crop_target=obj;p.sculpt_surface=obj;p.sculpt_target='OBJECT'
        mu.mask_inner_surface(obj)
        p.sculpt_remesh_report=f'Sculpt: {used:.3f} mm voxels; reference RMS {reference_rms:.3f}, max {reference_max:.3f} mm'+(' (reprojection skipped: folds)' if rejected else '')
        return obj
    except Exception:
        mu.delete_object(obj);source.hide_set(False);source.select_set(True);context.view_layer.objects.active=source
        raise


def remove_backing(obj):
    """On a build copy only, expose the edited outer surface for prototype fitting."""
    if obj.get('sculpt_crop_capped'):
        bm=bmesh.new();bm.from_mesh(obj.data)
        layer=bm.faces.layers.int.get('anaplast_crop_cap')
        if layer is None:bm.free();raise RuntimeError('Crop closure labels were lost; crop with Keep open before building a Prototype')
        bmesh.ops.delete(bm,geom=[f for f in bm.faces if f[layer]],context='FACES')
        loose=[v for v in bm.verts if not v.link_faces]
        if loose:bmesh.ops.delete(bm,geom=loose,context='VERTS')
        bm.to_mesh(obj.data);bm.free();obj.data.update()
        obj['sculpt_crop_capped']=False;obj['sculpt_temporary_backing']=False
        return
    if not obj.get('sculpt_temporary_backing'):return
    bm=bmesh.new();bm.from_mesh(obj.data)
    layer=bm.verts.layers.float.get('anaplast_outer')
    if layer is None:bm.free();raise RuntimeError('Sculpt surface labels were lost; prepare the Sculpt again before building a prototype')
    discard=[v for v in bm.verts if v[layer]<.5]
    bmesh.ops.delete(bm,geom=discard,context='VERTS')
    if not bm.faces:bm.free();raise RuntimeError('No outer Sculpt surface remains')
    mu.clean_margin(bm,smooth_iters=0);bm.to_mesh(obj.data);bm.free();obj.data.update()
    obj['sculpt_temporary_backing']=False


class ANAPLAST_OT_prepare_sculpt(bpy.types.Operator):
    bl_idname='anaplast.prepare_sculpt';bl_label='Prepare solid Sculpt';bl_options={'REGISTER','UNDO'}
    @classmethod
    def poll(cls,context):
        p=context.scene.anaplast;obj=mu.crop_object(p)
        return obj is not None and obj.type=='MESH' and (obj.get('sculpt_crop_complete') or (obj==p.prosthesis_obj and obj.get('anaplast_part')!='MIRROR'))
    def execute(self,context):
        try:prepare(context,mu.crop_object(context.scene.anaplast))
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        self.report({'INFO'},context.scene.anaplast.sculpt_remesh_report);return {'FINISHED'}
