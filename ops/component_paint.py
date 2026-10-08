"""Paint component patches on an unchanged copy of the fitting scan."""
import json
import bpy
import numpy as np
from mathutils import Vector,Matrix
from . import wedge_marking as wm,substructure as sub
from ..utils import mesh as mu
from .report import rep


def restore(context):
    stored=context.scene.pop('component_visibility',None)
    if stored:
        for name,hidden in json.loads(stored).items():
            obj=bpy.data.objects.get(name)
            if obj and obj.name in context.view_layer.objects:obj.hide_set(hidden)
    obj=context.scene.anaplast.sub_mark_obj
    if obj:obj.hide_set(True)
    if context.area and context.area.type=='VIEW_3D':context.space_data.shading.color_type='OBJECT'


def ensure_marks(context):
    p=context.scene.anaplast;outer,fit=sub.sources(p)
    if fit is None:raise RuntimeError('Assign the fitting scan first')
    signature=wm.geometry_signature(fit);obj=p.sub_mark_obj
    if obj and obj.get('component_source_signature')!=signature:
        raise RuntimeError('The fitting scan changed; use New painting copy before painting again')
    if obj is None:
        obj=mu.duplicate_object(fit,'Components_Marked_Area',mu.get_collection(context.scene,sub.COLLECTION))
        obj['component_paint_copy']=True;obj['component_source_signature']=signature
        wm.write_values(obj,np.zeros(len(obj.data.vertices),np.float32));p.sub_mark_obj=obj
    if context.object and context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    from .explode import ensure_collapsed
    ensure_collapsed(context.scene)
    if 'component_visibility' not in context.scene:context.scene['component_visibility']=json.dumps({o.name:o.hide_get() for o in context.view_layer.objects})
    for o in context.view_layer.objects:
        o.select_set(False)
        if o.type=='MESH':o.hide_set(o!=obj)
    obj.hide_set(False);obj.hide_render=True;obj.select_set(True);context.view_layer.objects.active=obj
    wm.show_colors(obj,wm.values_for(obj))
    if context.area and context.area.type=='VIEW_3D':context.space_data.shading.type='SOLID';context.space_data.shading.color_type='VERTEX'
    p.sub_component_source='PAINT';return obj


def patch_centres(obj):
    selected=wm.values_for(obj)>.5;neighbors=[[] for _ in selected]
    for edge in obj.data.edges:
        a,b=edge.vertices
        if selected[a] and selected[b]:neighbors[a].append(b);neighbors[b].append(a)
    co=np.asarray([obj.matrix_world@v.co for v in obj.data.vertices]);weights=np.zeros(len(co));obj.data.calc_loop_triangles()
    for tri in obj.data.loop_triangles:
        ids=list(tri.vertices);a,b,c=co[ids];area=np.linalg.norm(np.cross(b-a,c-a))/6.
        weights[ids]+=area
    unseen=set(np.flatnonzero(selected));groups=[]
    while unseen:
        seed=unseen.pop();stack=[seed];ids=[seed]
        while stack:
            for v in neighbors[stack.pop()]:
                if v in unseen:unseen.remove(v);stack.append(v);ids.append(v)
        if len(ids)<3:raise RuntimeError('A painted component patch has fewer than three vertices; enlarge it or erase it')
        groups.append(np.average(co[ids],axis=0,weights=np.maximum(weights[ids],1e-12)))
    return sorted(groups,key=lambda q:tuple(q))


def update_markers(context):
    p=context.scene.anaplast;obj=p.sub_mark_obj;_,fit=sub.sources(p)
    if obj is None or fit is None:raise RuntimeError('Paint the component areas on the fitting scan first')
    if obj.get('component_source_signature')!=wm.geometry_signature(fit):raise RuntimeError('The fitting scan changed; create a new painting copy')
    centres=patch_centres(obj)
    if not centres:raise RuntimeError('Paint one separate patch for each component')
    tree=sub.surface_tree(fit);col=mu.get_collection(context.scene,sub.POINTS);new=[]
    try:
        for i,centre in enumerate(centres):
            point,normal,_,_=tree.find_nearest(Vector(centre))
            if point is None:raise RuntimeError('A painted patch could not be located on the fitting scan')
            marker=bpy.data.objects.new('Painted_Component_new',None);col.objects.link(marker);new.append(marker)
            marker.matrix_world=Matrix.Translation(point)@normal.to_track_quat('Z','Y').to_matrix().to_4x4()
            marker.empty_display_type='ARROWS';marker.empty_display_size=3.;marker.show_in_front=True
            marker['generic_component_marker']=True;marker['painted_component']=True;marker['component_scan']=fit.name
        for o in list(context.scene.objects):
            if o.get('painted_component') and o not in new:mu.delete_object(o)
        for i,o in enumerate(new):o.name=f'Painted_Component_{i+1:02d}'
    except Exception:
        for o in new:mu.delete_object(o)
        raise
    p.sub_report=f'{len(new)} painted components; each disconnected patch defines one position'
    return new


class ANAPLAST_OT_component_paint(bpy.types.Operator):
    bl_idname='anaplast.component_paint';bl_label='Paint component areas';bl_options={'REGISTER','UNDO'}
    action:bpy.props.EnumProperty(items=[('START','Paint on fitting scan',''),('NEW','New painting copy',''),('CLEAR','Clear paint',''),('APPLY','Use painted components',''),('VIEW','Return to model','')])
    def execute(self,context):
        try:
            p=context.scene.anaplast
            if self.action=='VIEW':restore(context)
            elif self.action=='APPLY':update_markers(context);restore(context)
            else:
                if self.action=='NEW':
                    if p.sub_mark_obj:p.sub_mark_obj.name='Components_Paint_Previous';p.sub_mark_obj.hide_set(True)
                    p.sub_mark_obj=None
                obj=ensure_marks(context)
                if self.action=='CLEAR':wm.write_values(obj,np.zeros(len(obj.data.vertices)))
        except Exception as e:rep(self,{'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}


def register():bpy.utils.register_class(ANAPLAST_OT_component_paint)
def unregister():bpy.utils.unregister_class(ANAPLAST_OT_component_paint)
