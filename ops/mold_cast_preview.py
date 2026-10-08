"""Cast previews from the built mold cavity, never from the source Sculpt."""
import json
import bpy,bmesh
from mathutils import Vector
from bpy.props import EnumProperty
from . import mold_inserts as mi,scene_helpers,explode


def clear(scene):
    for name,value in json.loads(scene.get('cast_preview_visibility','{}')).items():
        obj=scene.objects.get(name)
        if obj:obj.hide_set(value[0]);obj.hide_render=value[1]
    if 'cast_preview_visibility' in scene:del scene['cast_preview_visibility']
    for obj in list(scene.objects):
        if obj.get('cast_preview'):mi.mu.delete_object(obj)


def separate_regions(work,wax):
    """Keep detached wax visible on request, without silently filling gaps."""
    bm=bmesh.new();bm.from_mesh(wax.data);remaining=set(bm.verts);groups=[]
    try:
        while remaining:
            todo=[remaining.pop()];group=[]
            while todo:
                v=todo.pop();group.append(v)
                for edge in v.link_edges:
                    other=edge.other_vert(v)
                    if other in remaining:remaining.remove(other);todo.append(other)
            groups.append(group)
        wax['cast_preview_regions']=len(groups)
        if len(groups)<2:return None
        groups.sort(key=len,reverse=True);detached=[v for group in groups[1:] for v in group]
        faces={f for v in detached for f in v.link_faces};mapping={v:i for i,v in enumerate(detached)}
        mesh=bpy.data.meshes.new('Detached_wax_regions');mesh.from_pydata([v.co[:] for v in detached],[],[tuple(mapping[v] for v in face.verts) for face in faces]);mesh.update()
        obj=bpy.data.objects.new('Wax detached regions',mesh);work.col.objects.link(obj);work.add(obj);obj.matrix_world=wax.matrix_world.copy()
        obj['cast_preview_detached']=True;obj.color=(.95,.45,.08,1)
        bmesh.ops.delete(bm,geom=detached,context='VERTS');bm.to_mesh(wax.data);wax.data.update()
        return obj
    finally:bm.free()


def subtract(work,target,source):
    # The Boolean helper may repair its operands. Never hand it a real mold.
    cutter=work.copy(source,'Cast_cutter_work')
    try:mi.aw.cut(target,cutter,'DIFFERENCE')
    finally:work.delete(cutter)


def remove_open_back(obj,up,top):
    """Remove void connected to the outside top of a hollow shell cap."""
    bm=bmesh.new();bm.from_mesh(obj.data)
    try:
        doomed=[];count=0
        for faces in mi.mu.islands(bm):
            verts={v for face in faces for v in face.verts}
            if any((obj.matrix_world@v.co).dot(up)>=top-.001 for v in verts):
                doomed.extend(verts);count+=1
        if doomed:bmesh.ops.delete(bm,geom=list(set(doomed)),context='VERTS')
        bm.to_mesh(obj.data);obj.data.update();obj['cast_external_regions_removed']=count
    finally:bm.free()
    if not obj.data.polygons:raise RuntimeError('No enclosed cast cavity remains; check the cap and its saved blank')


def cavity(work,base,cap):
    # This saved blank is the pre-cavity mold stock, with land features already
    # removed. It bounds the cast and excludes sprues/bolt/pry spaces. Its shape
    # supplies the fitting boundary, not the original anatomy mesh.
    blank=bpy.data.objects.get('Auricular_CapBlank_Reference')
    if blank is None:raise RuntimeError('Rebuild the mold once to save its cavity boundary')
    body=work.copy(blank,'Mold_cavity_work');up=mi.aw.frame_for(base)[0]
    print('Cast preview: extracting actual cap cavity',flush=True)
    subtract(work,body,cap)
    remove_open_back(body,up,mi.bounds(blank,up)[1])
    subtract(work,body,base)
    wedge=bpy.data.objects.get('Mold_Wedge')
    if wedge:subtract(work,body,wedge)
    return body


def finish_body(work,body,stage,keep):
    from . import mold_design
    mi.discard_boolean_dust(body)
    if mold_design.has_self_intersections(body):
        print('Cast preview: repairing visualization copy',stage,flush=True)
        body['cast_preview_spacing_mm']=mi.mu.voxel_remesh(body,.05)
        mi.discard_boolean_dust(body)
    mi.closed(body)
    if mold_design.has_self_intersections(body):raise RuntimeError('The cast preview could not be made cleanly; the mold was not changed')
    detached=separate_regions(work,body)
    if detached:
        detached.name=stage.title()+' detached regions';detached['cast_preview_stage']=stage;keep.append(detached)
    body.name='Wax cast preview' if stage=='WAX' else 'Prosthesis preview'
    body['cast_preview_stage']=stage
    body.color=(.62,.19,.16,1) if stage=='WAX' else (.65,.43,.30,1)
    body.data.materials.clear()
    material=bpy.data.materials.get('Cast preview '+stage) or bpy.data.materials.new('Cast preview '+stage)
    material.diffuse_color=body.color;material.use_nodes=True
    shader=next((n for n in material.node_tree.nodes if n.type=='BSDF_PRINCIPLED'),None)
    if shader:
        shader.inputs['Base Color'].default_value=body.color;shader.inputs['Roughness'].default_value=.5
    body.data.materials.append(material)
    for face in body.data.polygons:face.use_smooth=True
    keep.append(body)


def build(context):
    scene=context.scene;explode.collapse(scene);state=mi.state(scene)
    if state:
        mi.checked_hosts(scene,state)
        if state.get('settings')!=mi.settings_signature(scene):raise RuntimeError('Update the inserts before making the cast previews')
    base=bpy.data.objects.get('Mold_Base');cap=bpy.data.objects.get('Mold_Cap')
    if base is None or cap is None:raise RuntimeError('Build the base and cap first')
    if any(o.get('flipped') for o in (base,cap)):raise RuntimeError('Return the mold to its assembled position first')
    parts=[o for o in scene.objects if o.get('insert_active') and o.get('insert_uid')]
    work=mi.Work(scene);keep=[]
    try:
        body=cavity(work,base,cap)
        # Both casts retain the pedestal, mounting plug and any airway cores.
        # Only wax is formed around the temporary gaze-space holder.
        holders=[]
        for part in parts:
            if part.get('insert_role') in {'FORMER','FORMER_COMBINED'}:holders.append(part)
            elif part.get('compact_passages'):
                envelope=part.get('cast_preview_solid')
                if not isinstance(envelope,bpy.types.Mesh):raise RuntimeError('Build / update all inserts once to exclude the feed and vent sprues from these previews')
                tool=bpy.data.objects.new('Cast_pedestal_exterior',envelope.copy());work.col.objects.link(tool);work.add(tool);tool.matrix_world=part.matrix_world.copy()
                try:subtract(work,body,tool)
                finally:work.delete(tool)
            else:subtract(work,body,part)
        wax=work.copy(body,'Wax_preview_work');prosthesis=body
        for part in holders:subtract(work,wax,part)
        eye=scene.anaplast.ocular_obj
        item=next((i for i in scene.mold_inserts.items if i.kind=='OCULAR' and i.mode=='INSERT' and i.ocular),None)
        if eye is None and item:eye=item.ocular
        if holders and eye is None:raise RuntimeError('The real ocular is missing; restore it before previewing the prosthesis')
        if eye:
            if eye.get('gaze_edge_pending'):raise RuntimeError('Apply gaze and refit the ocular edge before previewing')
            subtract(work,prosthesis,eye)
            if not holders:subtract(work,wax,eye)
        print('Cast preview: validating wax and prosthesis',flush=True)
        finish_body(work,wax,'WAX',keep);finish_body(work,prosthesis,'PROSTHESIS',keep)
        if eye:
            production=scene.ocular_production
            eyes=[o for o in (production.color_obj,production.clear_obj) if o] if eye==scene.anaplast.ocular_obj and production.color_obj else [eye]
            for source in eyes:
                copy=work.copy(source,'Ocular in prosthesis preview');matrix=copy.matrix_world.copy();copy.parent=None;copy.matrix_world=matrix
                copy['cast_preview_eye']=True;copy['cast_preview_stage']='PROSTHESIS';keep.append(copy)
        clear(scene)
        for obj in keep:
            for key in list(obj.keys()):
                if key.startswith(('insert_','workflow_')) or key in ('anaplast_construction','reference_only'):del obj[key]
            obj['cast_preview']=True;obj['anaplast_part']='CAST_PREVIEW';obj['mold_set']=base.get('mold_set','')
            obj['preview_scope']='BUILT_CAP_CAVITY_MINUS_BASE_WEDGE_AND_INSERTS; NO_SPRUES_OR_SHRINKAGE'
            obj.hide_render=True;obj.hide_set(True);obj.hide_viewport=False
            scene_helpers.organize_object(obj,scene,preserve_visibility=True)
        work.finish(keep=keep);return wax,keep
    except Exception:work.finish();raise


class ANAPLAST_OT_cast_preview(bpy.types.Operator):
    bl_idname='anaplast.cast_preview';bl_label='Wax cast preview';bl_options={'REGISTER','UNDO'}
    action:EnumProperty(items=[('BUILD','Create / update previews',''),('WAX','Show wax',''),('OCULAR','Show prosthesis with ocular',''),('ALL','Show detached regions too',''),('MOLD','Return to mold','')])
    def execute(self,context):
        scene=context.scene
        try:
            if self.action=='BUILD':
                wax,_=build(context)
                if wax.get('cast_preview_regions',1)>1:self.report({'WARNING'},f"Wax preview has {wax['cast_preview_regions']} disconnected regions; inspect supports and margins")
            previews=[o for o in scene.objects if o.get('cast_preview')]
            if not previews and self.action!='MOLD':raise RuntimeError('Create the wax preview first')
            if self.action not in {'BUILD','MOLD'} and any('cast_preview_stage' not in o for o in previews):
                raise RuntimeError('Create / update previews to replace the older Sculpt-based preview')
            if self.action=='MOLD':
                for name,value in json.loads(scene.get('cast_preview_visibility','{}')).items():
                    obj=scene.objects.get(name)
                    if obj:obj.hide_set(value[0]);obj.hide_render=value[1]
                if 'cast_preview_visibility' in scene:del scene['cast_preview_visibility']
                for obj in previews:obj.hide_set(True);obj.hide_render=True
            else:
                if not scene.get('cast_preview_visibility'):
                    scene['cast_preview_visibility']=json.dumps({o.name:[o.hide_get(),o.hide_render] for o in scene.objects if o.type=='MESH' and not o.get('cast_preview')})
                stage=scene.get('cast_preview_stage','WAX') if self.action=='ALL' else ('PROSTHESIS' if self.action=='OCULAR' else 'WAX')
                scene['cast_preview_stage']=stage
                for obj in scene.objects:
                    if obj.type!='MESH':continue
                    show=obj in previews and obj.get('cast_preview_stage','WAX')==stage and (not obj.get('cast_preview_detached') or self.action=='ALL')
                    obj.hide_set(not show);obj.hide_render=not show
                wax=next(o for o in previews if o.get('cast_preview_stage','WAX')==stage and not o.get('cast_preview_eye') and not o.get('cast_preview_detached'))
                for obj in context.selected_objects:obj.select_set(False)
                wax.select_set(True);context.view_layer.objects.active=wax
            return {'FINISHED'}
        except (RuntimeError,ValueError) as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}


def register():bpy.utils.register_class(ANAPLAST_OT_cast_preview)
def unregister():bpy.utils.unregister_class(ANAPLAST_OT_cast_preview)
