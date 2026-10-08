"""Shared sculpt/texture target and patient-frame nasal symmetry."""
import bpy
import numpy as np
from .report import rep


def target(context):
    from .shell import current_prototype, source_surface
    p=context.scene.anaplast
    choice='OBJECT' if p.sculpt_surface else p.sculpt_target
    if choice=='ACTIVE':obj=context.active_object
    elif choice=='OBJECT':obj=p.sculpt_surface or p.source_prosthesis or p.prosthesis_obj
    elif choice=='PROTO':obj=current_prototype(p)
    elif choice=='SOURCE':obj=source_surface(p)
    elif choice=='PROSTHESIS':obj=p.prosthesis_obj
    elif choice=='SCAN':obj=p.face_scan_obj
    elif choice in {'BASE','WEDGE','CAP'}:obj=context.scene.objects.get('Mold_'+choice.title())
    else:obj=None
    return obj if obj is not None and obj.type=='MESH' and obj.name in context.view_layer.objects else None


def activate(context,obj):
    if context.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    # Lock the chosen surface so helper objects cannot redirect subsequent tools.
    context.scene.anaplast.sculpt_surface=obj
    context.scene.anaplast.sculpt_target='OBJECT'
    for other in context.selected_objects:other.select_set(False)
    obj.hide_set(False);obj.select_set(True);context.view_layer.objects.active=obj


class ANAPLAST_OT_surface_selected(bpy.types.Operator):
    bl_idname='anaplast.surface_selected';bl_label='Use selected surface';bl_options={'REGISTER','UNDO'}
    @classmethod
    def poll(cls,context):return context.active_object is not None and context.active_object.type=='MESH'
    def execute(self,context):
        p=context.scene.anaplast;p.sculpt_surface=context.active_object;p.sculpt_target='OBJECT'
        return {'FINISHED'}


class ANAPLAST_OT_nasal_symmetry(bpy.types.Operator):
    """Enable mirrored strokes on the existing Sculpt, using the patient midline without changing its world shape"""
    bl_idname='anaplast.nasal_symmetry';bl_label='Enable midline symmetry';bl_options={'REGISTER','UNDO'}
    @classmethod
    def poll(cls,context):return target(context) is not None
    def execute(self,context):
        from .orient import build_frame
        obj=target(context)
        p=context.scene.anaplast
        if obj==p.face_scan_obj or obj.get('anaplast_part')=='MIRROR' or obj.name.startswith('Mirror_'):
            rep(self,{'ERROR'},'Choose the cropped Sculpt to enable symmetry; the Scan and Mirror scan are references')
            return {'CANCELLED'}
        frame,message=build_frame()
        if frame is None:
            rep(self,{'ERROR'},message+' — confirm the case landmarks first');return {'CANCELLED'}
        array=np.asarray(frame)
        if not np.isfinite(array).all() or abs(np.linalg.det(array[:3,:3])-1)>1e-4:
            rep(self,{'ERROR'},'Case landmarks do not define a valid midline');return {'CANCELLED'}
        delta=frame.inverted()@obj.matrix_world
        reframe=not np.allclose(np.asarray(delta),np.eye(4),atol=1e-6)
        if reframe and (obj.modifiers or obj.constraints or obj.animation_data):
            rep(self,{'ERROR'},'Set up midline symmetry before adding subdivisions or other modifiers; this Sculpt already has a modifier or animation')
            return {'CANCELLED'}
        if context.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        if obj.data.users>1:obj.data=obj.data.copy()
        mesh=obj.data
        if reframe:
            children=[(child,child.matrix_world.copy()) for child in obj.children]
            mesh.transform(delta,shape_keys=True);mesh.update()
            obj.matrix_world=frame
            for child,matrix in children:child.matrix_world=matrix
        obj['nasal_symmetry_frame']=True
        mesh.use_mirror_x=True;mesh.use_mirror_y=False;mesh.use_mirror_z=False
        mesh.use_mirror_topology=False
        activate(context,obj)
        bpy.ops.object.mode_set(mode='SCULPT')
        rep(self,{'INFO'},'Midline symmetry enabled on the same Sculpt. X mirrors brush strokes; the world shape is unchanged.')
        return {'FINISHED'}


_classes=(ANAPLAST_OT_surface_selected,ANAPLAST_OT_nasal_symmetry)
def symmetry_get(obj):return bool(obj.data and obj.type=='MESH' and obj.data.use_mirror_x)
def symmetry_set(obj,value):
    if obj.type=='MESH':
        obj.data.use_mirror_x=value;obj.data.use_mirror_y=False;obj.data.use_mirror_z=False


def register():
    bpy.types.Object.b4d_midline_symmetry=bpy.props.BoolProperty(name="Symmetry",get=symmetry_get,set=symmetry_set,description="Sculpt symmetry across the local X midline")
    for cls in _classes:bpy.utils.register_class(cls)
def unregister():
    del bpy.types.Object.b4d_midline_symmetry
    for cls in reversed(_classes):bpy.utils.unregister_class(cls)
