"""Short, repeatable ocular workflow using the existing validated builders."""
import bpy
import numpy as np
from mathutils import Vector
from . import ocular_marking as mark, ocular_production as prod
from ..utils import mesh as mu
from .report import rep


def generate(context):
    from .ocular_contact import finish
    finish(context)
    from .ocular_photos import sync_whole_eye
    sync_whole_eye(context.scene)
    from .mold_inserts import guard_ocular_rebuild
    guard_ocular_rebuild(context.scene)
    from . import ocular_steps as steps
    s=context.scene;w=s.ocular_steps;p=s.ocular_production;ap=s.anaplast
    obj,src,_=steps.validated_marks(context)
    if (mark.values_for(obj,'SCLERA')>.5).sum()<30:
        raise RuntimeError('Mark the white of the eye on both sides of the iris first')
    iris=mark.values_for(obj,'IRIS')>.5
    if iris.sum()<12:raise RuntimeError('Mark the iris, including the pupil, first')
    if not p.photo:raise RuntimeError('Choose an iris photo or use the example photos')
    prod.photo_valid(p)
    if s.ocular_photos.image:prod.pixels(s.ocular_photos.image)
    world=np.array([obj.matrix_world@v.co for v in obj.data.vertices])
    if len(mark.contours(obj,iris,world))!=1:
        raise RuntimeError('Mark one filled iris without gaps or stray islands')
    if context.object and context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    mark.restore(context)
    # Keep the accepted result alive until all four builders succeed.
    originals=set(bpy.data.objects)
    visibility={o:(o.hide_get(),o.hide_render,o.name) for o in context.view_layer.objects}
    pointers=[(w,'ball'),(ap,'ocular_obj'),(p,'orbital_preview'),(p,'color_obj'),(p,'clear_obj')]
    saved=[(group,key,getattr(group,key)) for group,key in pointers]
    fields=('iris_fit','iris_ball_signature','iris_xy','iris_diameter','measured_iris_diameter','report')
    state={key:(tuple(getattr(w,key)) if key=='iris_xy' else getattr(w,key)) for key in fields}
    dimensions=(ap.ocular_iris_diameter,ap.ocular_cornea_diameter)
    marker=bpy.data.objects.get('Ocular_Cornea_Center')
    marker_state=(marker.matrix_world.copy(),marker.empty_display_size) if marker else None
    # The individual builders normally replace these objects immediately.
    w.ball=None;p.orbital_preview=None
    try:
        steps.fit_ball(context)
        steps.fit_iris(context)
        steps.build_shell(context,True)
        core,clear=prod.build(context)
    except Exception:
        for new in set(bpy.data.objects)-originals:mu.delete_object(new)
        for group,key,value in saved:setattr(group,key,value)
        for key,value in state.items():setattr(w,key,value)
        ap.ocular_iris_diameter,ap.ocular_cornea_diameter=dimensions
        if marker_state:marker.matrix_world,marker.empty_display_size=marker_state
        for old,(hidden,render,name) in visibility.items():
            old.name=name;old.hide_set(hidden);old.hide_render=render
        raise
    for group,key,old in saved[:3]:
        if old and old!=getattr(group,key):mu.delete_object(old)
    w.edit_marks=False
    w.report=f'Ready: iris {w.iris_diameter:.2f} mm. '+steps.fit_summary(w.ball)
    p.result='Ocular, clear cornea, photo maps and scan cutout updated.'
    return core,clear


def preview_lighting(context):
    """Supply local softboxes only when the case has no usable scene lighting."""
    s=context.scene;w=s.ocular_steps
    if not w.ball:return
    if any(o.type=='LIGHT' and not o.hide_render and not o.hide_get() and o.data.energy>0 for o in s.objects):return
    u=Vector(w.ball['ocular_up']);r=Vector(w.ball['ocular_right']);f=u.cross(r)
    focus=Vector(w.ball['fit_center_world'])+u*w.ball['fit_radius_mm']
    col=mu.get_collection(s,'Ocular Preview Lighting')
    for i,(offset,energy,size) in enumerate(((u*50-r*28+f*36,26000,12),(u*40+r*42+f*8,18000,22),(r*20-u*5+f*40,14000,15))):
        data=bpy.data.lights.new('Ocular softbox '+str(i),'AREA');light=bpy.data.objects.new(data.name,data);col.objects.link(light)
        light.location=focus+offset;light.rotation_euler=(-offset).to_track_quat('-Z','Y').to_euler()
        data.energy=energy;data.shape='DISK';data.size=size;data.specular_factor=1. if i==0 else 0.
    world=bpy.data.worlds.new('Ocular soft studio');world.use_nodes=True
    world.node_tree.nodes['Background'].inputs['Color'].default_value=(.7,.75,.8,1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value=.45;s.world=world
    s.render.engine='CYCLES';s.cycles.use_denoising=True;s.view_settings.view_transform='AgX'


class OCULAR_OT_generate(bpy.types.Operator):
    bl_idname='anaplast.ocular_generate';bl_label='Generate ocular';bl_options={'REGISTER','UNDO'}
    bl_description='Fit the marked sclera, transfer the iris, build the rounded ocular and clear cornea, apply photos and replace the eye on a scan copy'
    def execute(self,context):
        try:generate(context)
        except Exception as e:rep(self,{'ERROR'},str(e));return {'CANCELLED'}
        # Preview is independent of successful geometry generation.
        try:
            from .ocular_view import rotation_control
            rotation_control(context)
            preview_lighting(context)
            from .overlay import clipping_area
            area=clipping_area(context)
            if area:
                sh=area.spaces.active.shading;sh.type='MATERIAL'
                sh.use_scene_world=True;sh.use_scene_lights=True
                sh.use_scene_world_render=True;sh.use_scene_lights_render=True
        except Exception as e:rep(self,{'WARNING'},'Ocular generated; preview setup: '+str(e))
        rep(self,{'INFO'},context.scene.ocular_steps.report)
        return {'FINISHED'}


class OCULAR_OT_example_photos(bpy.types.Operator):
    bl_idname='anaplast.ocular_example_photos';bl_label='Use example photos';bl_options={'REGISTER','UNDO'}
    bl_description='Fill missing photos with the locally bundled blue/gold iris and real sclera; keep any photos already loaded'
    def execute(self,context):
        p=context.scene.ocular_production;photos=context.scene.ocular_photos
        if not p.photo:
            if bpy.ops.anaplast.ocular_reference(choice='Shillier')!={'FINISHED'}:return {'CANCELLED'}
            photos.iris_exposure=.4
        if not photos.image:
            if bpy.ops.anaplast.ocular_sclera_reference()!={'FINISHED'}:return {'CANCELLED'}
        return {'FINISHED'}


def draw(l,context):
    s=context.scene;w=s.ocular_steps;p=s.ocular_production;photos=s.ocular_photos
    box=l.box();box.label(text='1. Mark the eye')
    box.prop(s.anaplast,'ocular_scan',text='Scan')
    if not s.anaplast.ocular_scan:box.label(text='Uses Mirror_FaceScan when available')
    row=box.row(align=True)
    for region,label in (('SCLERA','Mark sclera'),('IRIS','Mark iris')):
        row.operator('anaplast.ocular_steps',text=label,depress=w.edit_marks and p.mark_region==region).action=region
    if w.edit_marks:
        box.label(text='White only; avoid eyelids.' if p.mark_region=='SCLERA' else 'Include the pupil inside the iris.')
        row=box.row(align=True)
        for label,erase,lasso in [('Paint',False,False),('Erase',True,False),('Lasso',False,True)]:
            op=row.operator('anaplast.wedge_mark_tool',text=label);op.target='OCULAR';op.erase=erase;op.lasso=lasso
        box.prop(p,'mark_radius',text='Brush size (mm)')
        box.label(text='Enter finishes painting; Esc cancels.')
        box.operator('anaplast.ocular_steps',text='Done marking').action='DONE'
    box.prop(w,'use_measured_diameter',text='I know the eyeball diameter')
    if w.use_measured_diameter:box.prop(w,'measured_diameter',text='Diameter (mm)')
    else:box.label(text='Diameter fitted from the sclera marks')
    box=l.box();box.label(text='2. Photo coverage')
    box.label(text='Uses the shared Eye photograph and marks')
    box.label(text='Trace its eye opening for sclera coverage')
    if photos.image:
        from .ocular_photos import draw_extension
        draw_extension(box,photos)
    box=l.box();box.label(text='3. Generate and view')
    box.prop(w,'iris_size_mode')
    if w.iris_size_mode=='MANUAL':box.prop(w,'iris_manual_diameter')
    elif w.measured_iris_diameter:box.label(text=f'Marked iris: {w.measured_iris_diameter:.2f} mm')
    box.prop(w,'cornea_size_mode')
    if w.cornea_size_mode=='MANUAL':box.prop(w,'cornea_manual_diameter')
    else:box.prop(w,'cornea_width',text='Cornea / iris size')
    box.prop(w,'cornea_height',text='Cornea height (mm)')
    from .apparent_pupil import controls
    controls(box,p)
    if not p.apparent_pupil:
        box.prop(p,'pupil_size_mode')
        if p.pupil_size_mode=='MANUAL':box.prop(p,'pupil_diameter')
        else:box.label(text=f'Photo pupil: {100*p.photo_pupil:.1f}% of iris diameter')
    box.prop(p,'limbus_width');box.prop(p,'limbus_photo_detail')
    box.prop(s.anaplast,'ocular_extension',text='Beyond opening (mm)')
    box.prop(p,'pupil_depth')
    box.prop(p,'pupil_underiris')
    row=box.row(align=True);row.prop(p,'pupil_rim_height');row.prop(p,'pupil_rim_width')
    box.prop(p,'pupil_rim_blend')
    row=box.row();row.scale_y=1.5
    row.operator('anaplast.ocular_generate',text='Update ocular' if p.color_obj else 'Generate ocular',icon='MOD_BUILD')
    if p.color_obj and p.clear_obj:
        row=box.row(align=True)
        row.operator('anaplast.ocular_display',text='Photo colors').mode='COLOR'
        row.operator('anaplast.ocular_display',text='Clear cornea').mode='CLEAR'
        if p.orbital_preview:box.prop(p.orbital_preview,'hide_viewport',text='Hide scan preview')
        from .ocular_view import draw as draw_inspection
        draw_inspection(box,context)
        from .ocular_variations import draw_export
        draw_export(box,p)
    if w.ball and 'fit_metrics' in w.ball:
        import json
        metrics=json.loads(w.ball['fit_metrics'])
        box.label(text=f'Estimated globe diameter: {2*w.ball["fit_radius_mm"]:.2f} mm')
        box.label(text=f'Sclera fit RMS: {metrics["rms_mm"]:.2f} mm')
        box.label(text=f'95% of samples within {metrics["p95_mm"]:.2f} mm')
        box.label(text='Surface: '+metrics['surface'])
        if w.ball.get('fit_ignored_vertices',0):box.label(text='Remote paint excluded from eye fit and cutout',icon='INFO')
    elif w.report:box.label(text=w.report)
    l.prop(w,'show_advanced',text='Advanced settings and individual steps',icon='TRIA_DOWN' if w.show_advanced else 'TRIA_RIGHT',emboss=False)
    if w.show_advanced:
        from . import ocular_steps as steps
        steps.draw(l,context);prod.draw_workflow(l,context)
        l.prop(w,'show_legacy')


def draw_marks(box,context):
    s=context.scene;w=s.ocular_steps;p=s.ocular_production
    box.prop(s.anaplast,'ocular_scan',text='Scan')
    if not s.anaplast.ocular_scan:box.label(text='Uses Mirror_FaceScan when available')
    row=box.row(align=True)
    for region,label in (('SCLERA','Mark sclera'),('IRIS','Mark iris')):
        row.operator('anaplast.ocular_steps',text=label,depress=w.edit_marks and p.mark_region==region).action=region
    if w.edit_marks:
        box.label(text='White only; avoid eyelids.' if p.mark_region=='SCLERA' else 'Include the pupil inside the iris.')
        row=box.row(align=True)
        for label,erase,lasso in [('Paint',False,False),('Erase',True,False),('Lasso',False,True)]:
            op=row.operator('anaplast.wedge_mark_tool',text=label);op.target='OCULAR';op.erase=erase;op.lasso=lasso
        box.prop(p,'mark_radius',text='Brush size (mm)')
        box.label(text='Enter finishes painting; Esc cancels.')
        box.operator('anaplast.ocular_steps',text='Done marking').action='DONE'
    box.prop(w,'use_measured_diameter',text='I know the eyeball diameter')
    if w.use_measured_diameter:box.prop(w,'measured_diameter',text='Diameter (mm)')
    else:box.label(text='Diameter fitted from the sclera marks')
