"""Subdivision the ZBrush way: a Multires modifier on the working surface; Subdivide adds a level,
the Level slider moves between levels, Apply bakes the current level into the mesh for the build."""
import bpy
from ..utils import mesh as mu
from .report import rep

MOD = "Anaplast_Multires"


def resolution_estimate(obj,mod,scene):
    """Base-edge sample scaled by subdivision; no dense evaluated mesh on panel redraw."""
    import numpy as np
    import math
    me=obj.data;levels=mod.total_levels if mod else 0
    current=(mod.sculpt_levels if obj.mode=='SCULPT' else mod.levels) if mod else 0
    count=len(me.polygons)
    faces=count if levels==0 else len(me.loops)*4**(levels-1)
    edges=me.edges;stride=max(1,len(edges)//1024);linear=obj.matrix_world.to_3x3()
    lengths=[(linear@(me.vertices[e.vertices[0]].co-me.vertices[e.vertices[1]].co)).length for e in (edges[i] for i in range(0,len(edges),stride))]
    spacing=float(np.median(lengths))*(scene.unit_settings.scale_length*1000)/2**levels if lengths else 0.
    # The 90th percentile avoids declaring readiness from only the finest parts of an uneven mesh.
    current_spacing=float(np.percentile(lengths,90))*(scene.unit_settings.scale_length*1000)/2**current if lengths else 0.
    goal=scene.anaplast.skin_target_spacing
    needed=max(0,math.ceil(math.log2(current_spacing/goal)-1e-6)) if current_spacing>0 else 0
    ready=bool(lengths) and current_spacing<=goal*(1+1e-6)
    next_faces=faces*4 if levels else len(me.loops)
    return {'faces':faces,'spacing':spacing,'next_faces':next_faces,'current_level':current,
            'current_spacing':current_spacing,'ready':ready,'needed':needed,'sampled_edges':len(lengths),
            'available':max(0,levels-current)}


def target(context):
    from .sculpt_surface import target
    return target(context)


def get_mod(obj, create=False):
    mod = obj.modifiers.get(MOD)
    if mod is None and create:
        mod = obj.modifiers.new(MOD, 'MULTIRES')
    return mod


def level_update(self, context):
    obj = target(context)
    if obj is None:
        return
    mod = get_mod(obj)
    if mod is None:
        return
    lv = max(0, min(self.multires_level, mod.total_levels))
    mod.levels = lv
    mod.sculpt_levels = lv
    mod.render_levels = mod.total_levels
    obj.update_tag()


class ANAPLAST_OT_remesh_surface(bpy.types.Operator):
    """Remesh only the Working surface with even voxels; bake existing subdivision detail first"""
    bl_idname='anaplast.remesh_surface';bl_label='Remesh';bl_options={'REGISTER','UNDO'}
    @classmethod
    def poll(cls,context):return target(context) is not None
    def execute(self,context):
        from .sculpt_surface import activate
        from .sculpt_prepare import prepare,remove_backing
        from .scene_helpers import organize_object
        p=context.scene.anaplast;obj=target(context)
        if obj in (p.face_scan_obj,p.cast_obj,p.cbct_obj) or obj.get('anaplast_part')=='MIRROR':
            self.report({'ERROR'},'Crop a Sculpt first; Scan and Mirror scan remain unchanged');return {'CANCELLED'}
        activate(context,obj)
        pointers={key:getattr(p,key) for key in ('prosthesis_obj','source_prosthesis','crop_target','sculpt_surface')}
        oldmode=p.sculpt_target;reference=None;result=None
        mirror_flags={key:getattr(obj.data,key) for key in ('use_mirror_x','use_mirror_y','use_mirror_z','use_mirror_topology')}
        try:
            # Evaluate the highest stored Multires level before changing topology.
            mres=get_mod(obj);oldlevel=mres.levels if mres else None
            try:
                if mres:mres.levels=mres.total_levels;context.view_layer.update()
                dg=context.evaluated_depsgraph_get()
                mesh=bpy.data.meshes.new_from_object(obj.evaluated_get(dg),preserve_all_data_layers=True,depsgraph=dg)
            finally:
                if mres:mres.levels=oldlevel;context.view_layer.update()
            reference=bpy.data.objects.new(obj.name+' · Remesh reference',mesh)
            mu.get_collection(context.scene,'Anaplast_Sculpt').objects.link(reference)
            reference.matrix_world=obj.matrix_world.copy()
            reference.color=obj.color[:]
            reference['reference_only']=True
            if obj.get('sculpt_temporary_backing'):
                reference['sculpt_temporary_backing']=True;remove_backing(reference)
            result=prepare(context,reference)
            result.data.transform(obj.matrix_world.inverted())
            oldmesh=obj.data;obj.data=result.data
            for key,value in mirror_flags.items():setattr(obj.data,key,value)
            for key in ('sculpt_reference','sculpt_temporary_backing','sculpt_voxel_mm','sculpt_deviation'):
                obj[key]=result[key]
            for mod in list(obj.modifiers):obj.modifiers.remove(mod)
            bpy.data.objects.remove(result,do_unlink=True);result=None
            if oldmesh.users==0:bpy.data.meshes.remove(oldmesh)
            reference.hide_set(True);reference.hide_render=True;organize_object(reference,context.scene)
            p.sculpt_remesh_report=p.sculpt_remesh_report.replace('Sculpt:',obj.name+':',1)
            p.multires_level=0
        except Exception as e:
            if result:bpy.data.objects.remove(result,do_unlink=True)
            if reference:bpy.data.objects.remove(reference,do_unlink=True)
            self.report({'ERROR'},str(e));return {'CANCELLED'}
        finally:
            for key,value in pointers.items():setattr(p,key,value)
            p.sculpt_target=oldmode
            activate(context,obj)
        self.report({'INFO'},p.sculpt_remesh_report);return {'FINISHED'}


class ANAPLAST_OT_multires_subdivide(bpy.types.Operator):
    """Add one subdivision level to the working surface (keeps all lower levels, like ZBrush SDiv)"""
    bl_idname = "anaplast.multires_subdivide"
    bl_label = "Subdivide"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return target(context) is not None

    def execute(self, context):
        p = context.scene.anaplast
        obj = target(context)
        from .sculpt_surface import activate
        activate(context, obj)
        mod = get_mod(obj, create=True)
        try:
            bpy.ops.object.multires_subdivide(modifier=MOD, mode='SIMPLE')
        except Exception as e:
            rep(self, {'ERROR'}, f"Could not subdivide: {e}")
            return {'CANCELLED'}
        p.multires_level = mod.total_levels
        level_update(p, context)
        rep(self, {'INFO'}, f"{obj.name}: now {mod.total_levels} level(s) — slide Level to move between them")
        return {'FINISHED'}


class ANAPLAST_OT_multires_apply(bpy.types.Operator):
    """Bake the highest subdivision level into the mesh (do this before Build Prototype)"""
    bl_idname = "anaplast.multires_apply"
    bl_label = "Apply subdivisions"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = target(context)
        return obj is not None and get_mod(obj) is not None

    def execute(self, context):
        obj = target(context)
        from .sculpt_surface import activate
        activate(context, obj)
        mod = get_mod(obj)
        mod.levels = mod.total_levels
        before = len(obj.data.polygons)
        dg = context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(dg)
        me = bpy.data.meshes.new_from_object(ev, depsgraph=dg)
        obj.modifiers.remove(mod)
        old = obj.data
        obj.data = me
        if old.users == 0:
            bpy.data.meshes.remove(old)
        for pg in obj.data.polygons:
            pg.use_smooth = True
        context.scene.anaplast.multires_level = 0
        rep(self, {'INFO'}, f"{obj.name}: {before:,} → {len(obj.data.polygons):,} faces baked in")
        return {'FINISHED'}


_classes = (ANAPLAST_OT_remesh_surface, ANAPLAST_OT_multires_subdivide, ANAPLAST_OT_multires_apply)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
