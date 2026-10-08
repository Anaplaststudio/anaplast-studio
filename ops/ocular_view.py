"""Local ocular inspection light and fitted-globe rotation control."""
import math
import bpy
from mathutils import Vector, Matrix
from ..utils import mesh as mu
from .report import rep


def eye_frame(context):
    s=context.scene;src=s.anaplast.ocular_obj;ball=s.ocular_steps.ball
    if src and 'workflow_sphere_center_world' in src:
        # Source vertices and fitted center share the builder's coordinate frame.
        center=src.matrix_world@Vector(src['workflow_sphere_center_world'])
        forward=(src.matrix_world.to_3x3()@Vector(src['ocular_up'])).normalized()
        right=(src.matrix_world.to_3x3()@Vector(src['ocular_right'])).normalized()
        radius=float(src['workflow_sphere_radius_mm'])
    elif ball and 'fit_center_world' in ball:
        center=Vector(ball['fit_center_world']);forward=Vector(ball['ocular_up']);right=Vector(ball['ocular_right'])
        radius=float(ball['fit_radius_mm'])
    else:raise RuntimeError('Fit or generate the ocular first to locate the eyeball center')
    return center,forward,right,forward.cross(right).normalized(),radius


def facial_torch_frame(context,anchor):
    """Use the facial midline and anterior axis, never the ocular's gaze."""
    if context.scene.anaplast.oriented:
        return Vector((0.,anchor.y,anchor.z)),Vector((0.,1.,0.))
    from .orient import build_frame
    frame,message=build_frame()
    if frame is None:raise RuntimeError('Orient the case or place the facial landmarks first: '+message)
    right=frame.to_3x3().col[0].normalized()
    anchor=anchor-right*(anchor-frame.translation).dot(right)
    return anchor,frame.to_3x3().col[1].normalized()


def show_torch(context,light):
    """Expose the light's helper folder and use the torch without softboxes."""
    from .scene_helpers import organize_object
    organize_object(light,context.scene,preserve_visibility=True)
    def reveal(layer):
        contains=light.name in layer.collection.objects
        for child in layer.children:contains=reveal(child) or contains
        if contains:
            layer.exclude=False;layer.hide_viewport=False
            layer.collection.hide_viewport=False;layer.collection.hide_render=False
            layer.collection.hide_select=False
        return contains
    reveal(context.view_layer.layer_collection)
    context.view_layer.update()
    light.hide_set(False);light.hide_select=False
    # Disable area lights (softboxes), including ones in a hidden helper folder
    # that has just been enabled. Retain them for later manual use.
    for obj in context.scene.objects:
        if obj!=light and obj.type=='LIGHT' and (obj.data.type=='AREA' or 'softbox' in obj.name.lower()):
            obj.hide_render=True;obj.hide_viewport=True
            if obj.name in context.view_layer.objects:obj.hide_set(True)
    if context.screen:
        for area in context.screen.areas:
            if area.type=='VIEW_3D':
                area.spaces.active.overlay.show_overlays=True
                area.spaces.active.overlay.show_extras=True


def torch_update(settings,context):
    light=settings.torch
    if not light or light.type!='LIGHT' or not light.get('ocular_torch'):return
    anchor=Vector(light['torch_anchor'])
    if light.get('torch_reference')=='Nasion landmark':
        landmark=context.scene.objects.get('LM_NASION')
        if landmark:anchor=landmark.matrix_world.translation.copy()
    anchor,direction=facial_torch_frame(context,anchor)
    light['torch_anchor']=list(anchor);light['torch_forward']=list(direction)
    position=anchor+direction*settings.torch_distance
    light['torch_target']=list(anchor)
    placement=(anchor-position).to_track_quat('-Z','Y').to_matrix().to_4x4()
    placement.translation=position;light.matrix_world=placement
    # Keep incident illumination approximately constant when changing distance.
    light.data.energy=180000.*settings.torch_brightness*(settings.torch_distance/300.)**2
    light.data.shadow_soft_size=settings.torch_size*.5
    light.hide_render=not settings.torch_on
    light.hide_viewport=not settings.torch_on
    if settings.torch_on:show_torch(context,light)


def place_torch(context,at_cursor=False):
    s=context.scene;settings=s.ocular_view
    from .orient import repair_legacy_landmark_frame
    if not at_cursor:repair_legacy_landmark_frame(s)
    landmark=s.objects.get('LM_NASION')
    if not at_cursor and landmark is None:
        raise RuntimeError('Place the Nasion landmark first, or put the 3D cursor at the nasion and use Torch at cursor')
    anchor=s.cursor.location.copy() if at_cursor else landmark.matrix_world.translation.copy()
    anchor,forward=facial_torch_frame(context,anchor)
    light=settings.torch
    if not light or light.name not in s.objects or light.type!='LIGHT' or not light.get('ocular_torch'):
        data=bpy.data.lights.new('Ocular torch','SPOT');light=bpy.data.objects.new('Ocular torch',data)
        mu.get_collection(s,'Ocular Preview Lighting').objects.link(light);settings.torch=light
    light['ocular_torch']=True;light['torch_anchor']=list(anchor);light['torch_forward']=list(forward)
    light['torch_target']=list(anchor)
    light['torch_reference']='3D cursor' if at_cursor else 'Nasion landmark'
    light.data.type='SPOT';light.data.spot_size=math.radians(50);light.data.spot_blend=.5
    light.data.color=(1.,.97,.92);light.data.specular_factor=1.
    settings.torch_on=True;torch_update(settings,context)
    light.hide_set(False)
    s.render.engine='CYCLES';s.cycles.use_denoising=True
    from .overlay import clipping_area
    area=clipping_area(context) if context.screen else None
    if area:
        sh=area.spaces.active.shading;sh.type='RENDERED'
        sh.use_scene_lights=True;sh.use_scene_world=True
        sh.use_scene_lights_render=True;sh.use_scene_world_render=True
    for obj in (s.ocular_production.color_obj,s.ocular_production.clear_obj):
        if obj:obj.hide_viewport=False;obj.hide_set(False)
    return light


def rotation_control(context):
    """Parent the complete ocular to a globe-center handle, preserving mesh data."""
    s=context.scene;v=s.ocular_view;p=s.ocular_production;src=s.anaplast.ocular_obj
    parts=[src,p.color_obj,p.clear_obj]
    if any(o is None for o in parts):raise RuntimeError('Generate the complete ocular first')
    if any(not all(abs(a-b)<1e-5 for row1,row2 in zip(o.matrix_world,src.matrix_world) for a,b in zip(row1,row2)) for o in parts):
        raise RuntimeError('The ocular parts moved apart; restore their shared position before rotating')
    center,forward,right,up,radius=eye_frame(context)
    pivot=v.pivot
    if not pivot or not pivot.get('ocular_rotation_control') or src.parent!=pivot:
        old=pivot
        pivot=bpy.data.objects.new('Ocular rotation',None)
        mu.get_collection(s,'Anaplast_Ocular').objects.link(pivot)
        pivot['ocular_rotation_control']=True;pivot['anaplast_part']='OCULAR'
        pivot.empty_display_type='PLAIN_AXES';pivot.empty_display_size=radius*.35
        pivot.matrix_world=Matrix((right,up,forward)).transposed().to_4x4();pivot.location=center
        pivot.lock_location=(True,True,True);pivot.lock_scale=(True,True,True)
        v.pivot=pivot
        context.view_layer.update()
        if old and old.get('ocular_rotation_control') and not old.children:mu.delete_object(old)
    inverse=pivot.matrix_world.inverted()
    for obj in parts:
        if obj.parent==pivot:continue
        world=obj.matrix_world.copy();obj.parent=pivot;obj.matrix_parent_inverse=inverse;obj.matrix_world=world
    context.view_layer.update()
    return pivot


def _matrix(values):
    return Matrix([values[i:i+4] for i in range(0,16,4)])


def prepare_gaze(context):
    from .mold_inserts import guard_ocular_rebuild
    guard_ocular_rebuild(context.scene)
    p=context.scene.ocular_production
    if any(o and o.mode!='OBJECT' for o in (p.color_obj,p.clear_obj,context.scene.anaplast.ocular_obj)):
        raise RuntimeError('Return the ocular to Object Mode before adjusting gaze')
    pivot=rotation_control(context)
    if 'gaze_rest_matrix' not in pivot:
        if context.scene.anaplast.oriented:
            x,z=Vector((1,0,0)),Vector((0,0,1))
        else:
            from .orient import build_frame
            frame,message=build_frame()
            if frame is None:raise RuntimeError('Orient the case before adjusting X/Z gaze: '+message)
            x,z=frame.to_3x3().col[0],frame.to_3x3().col[2]
        # An existing hand-adjusted pose becomes zero, without moving the eye.
        pivot['gaze_rest_matrix']=[value for row in pivot.matrix_world for value in row]
        pivot['gaze_x_axis']=list(x.normalized());pivot['gaze_z_axis']=list(z.normalized())
        pivot['gaze_angle_x']=0.;pivot['gaze_angle_z']=0.
    no_rotation_gizmo(context)
    return pivot


def gaze_matrix(pivot,x,z):
    rest=_matrix(pivot['gaze_rest_matrix']);center=rest.translation
    rotation=Matrix.Rotation(z,4,Vector(pivot['gaze_z_axis']))@Matrix.Rotation(x,4,Vector(pivot['gaze_x_axis']))
    return Matrix.Translation(center)@rotation@Matrix.Translation(-center)@rest


def apply_gaze(context,x,z,pivot=None):
    from .ocular_contact import finish
    finish(context)
    pivot=pivot or prepare_gaze(context)
    limit=math.radians(60);x=max(-limit,min(limit,x));z=max(-limit,min(limit,z))
    from .ocular_gaze_edge import restore_unfitted
    restore_unfitted(context.scene)
    pivot.matrix_world=gaze_matrix(pivot,x,z)
    pivot['gaze_angle_x']=x;pivot['gaze_angle_z']=z
    context.scene.ocular_view.gaze_message=''
    return pivot


def gaze_x_get(v):return float(v.pivot.get('gaze_angle_x',0.)) if v.pivot else 0.
def gaze_z_get(v):return float(v.pivot.get('gaze_angle_z',0.)) if v.pivot else 0.


def set_slider(v,value,axis):
    try:
        if bpy.context.scene!=v.id_data:raise RuntimeError('Switch to this case before adjusting gaze')
        apply_gaze(bpy.context,value if axis=='X' else v.gaze_x,value if axis=='Z' else v.gaze_z)
    except Exception as error:v.gaze_message=str(error)


def gaze_x_set(v,value):set_slider(v,value,'X')
def gaze_z_set(v,value):set_slider(v,value,'Z')


def no_rotation_gizmo(context):
    from .overlay import clipping_area
    area=clipping_area(context) if context.screen else None
    if area:
        area.spaces.active.show_gizmo_object_rotate=False
        with context.temp_override(area=area):bpy.ops.wm.tool_set_by_id(name='builtin.select_box')


def mouse_gaze(x,z,dx,dy,fine=False):
    speed=math.radians(.024 if fine else .18)
    limit=math.radians(60)
    return max(-limit,min(limit,x+dy*speed)),max(-limit,min(limit,z+dx*speed))


class OcularView(bpy.types.PropertyGroup):
    fit_active:bpy.props.BoolProperty(default=False)
    fit_report:bpy.props.StringProperty()
    fit_visibility:bpy.props.StringProperty()
    fit_shading:bpy.props.StringProperty()
    fit_tolerance:bpy.props.FloatProperty(name='Contact tolerance (mm)',default=.15,min=.01,soft_max=1.)

    gaze_x:bpy.props.FloatProperty(name='Look up / down (X)',subtype='ANGLE',min=-math.pi/3,max=math.pi/3,soft_min=-math.pi/9,soft_max=math.pi/9,get=gaze_x_get,set=gaze_x_set,description='Rotate about the facial X axis through the eyeball center; positive looks up')
    gaze_z:bpy.props.FloatProperty(name='Look left / right (Z)',subtype='ANGLE',min=-math.pi/3,max=math.pi/3,soft_min=-math.pi/9,soft_max=math.pi/9,get=gaze_z_get,set=gaze_z_set,description='Rotate about the facial Z axis through the eyeball center; no Y rotation control')
    gaze_message:bpy.props.StringProperty()
    pivot:bpy.props.PointerProperty(type=bpy.types.Object)
    torch:bpy.props.PointerProperty(type=bpy.types.Object)
    torch_on:bpy.props.BoolProperty(name='Torch on',default=True,update=torch_update)
    torch_distance:bpy.props.FloatProperty(name='Distance from nasion (mm)',default=300,min=50,max=2000,update=torch_update)
    torch_brightness:bpy.props.FloatProperty(name='Brightness',default=1.,min=0,max=10,update=torch_update)
    torch_size:bpy.props.FloatProperty(name='Light diameter (mm)',default=6,min=.1,max=50,update=torch_update)


class OCULAR_OT_torch(bpy.types.Operator):
    bl_idname='anaplast.ocular_torch';bl_label='Torch at nasion';bl_options={'REGISTER','UNDO'}
    bl_description='Place a small inspection light at the exact nasion height (same Z), pointing horizontally toward the nasion'
    at_cursor:bpy.props.BoolProperty(default=False,options={'HIDDEN'})
    def execute(self,context):
        try:place_torch(context,self.at_cursor)
        except Exception as e:rep(self,{'ERROR'},str(e));return {'CANCELLED'}
        rep(self,{'INFO'},'Torch ready; use Clear cornea to see its reflection')
        return {'FINISHED'}


class OCULAR_OT_eye_center(bpy.types.Operator):
    bl_idname='anaplast.ocular_eye_center';bl_label='Gaze controls';bl_options={'REGISTER','UNDO'}
    bl_description='Prepare the X/Z gaze controls without showing a rotation gizmo'
    def execute(self,context):
        try:prepare_gaze(context);no_rotation_gizmo(context)
        except Exception as e:rep(self,{'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}


class OCULAR_OT_gaze_reset(bpy.types.Operator):
    bl_idname='anaplast.ocular_gaze_reset';bl_label='Reset gaze';bl_options={'REGISTER','UNDO'}
    bl_description='Return X and Z adjustments to zero, preserving the pose that was present when these controls were first used'
    def execute(self,context):
        try:apply_gaze(context,0.,0.);no_rotation_gizmo(context)
        except Exception as e:rep(self,{'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}


class OCULAR_OT_gaze_mouse(bpy.types.Operator):
    bl_idname='anaplast.ocular_gaze_mouse';bl_label='Adjust gaze with mouse';bl_options={'REGISTER','UNDO','BLOCKING'}
    bl_description='Click in the scene to start, then move horizontally for Z and vertically for X; Shift is finer, click again confirms, Esc or right-click cancels'
    @classmethod
    def poll(cls,context):return context.area is not None and context.area.type=='VIEW_3D'
    def invoke(self,context,event):
        try:
            self._pivot=prepare_gaze(context);no_rotation_gizmo(context)
        except Exception as e:rep(self,{'ERROR'},str(e));return {'CANCELLED'}
        self._start=self._pivot.matrix_world.copy()
        self._angles=(context.scene.ocular_view.gaze_x,context.scene.ocular_view.gaze_z)
        from .ocular_gaze_edge import parts
        self._mesh_state=[(o,o.data,o.get('built_geometry'),o.get('gaze_edge_fitted',False),o.get('gaze_edge_pending',False)) for o in parts(context.scene)]
        self._area=context.area;self._window=context.window;self._last=None;self._started=False
        self._region=next((r for r in self._area.regions if r.type=='WINDOW'),None)
        if not self._region:return {'CANCELLED'}
        self._area.header_text_set('Click inside the scene to start gaze adjustment | Esc: cancel')
        context.window.cursor_modal_set('SCROLL_XY');context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    def _inside(self,event):
        def contains(region):return region.x<=event.mouse_x<region.x+region.width and region.y<=event.mouse_y<region.y+region.height
        return contains(self._region) and not any(contains(r) for r in self._area.regions if r.type in {'UI','TOOLS'})
    def _finish(self,cancel=False):
        if cancel:
            self._pivot.matrix_world=self._start
            self._pivot['gaze_angle_x'],self._pivot['gaze_angle_z']=self._angles
            for obj,mesh,signature,fitted,pending in getattr(self,'_mesh_state',[]):
                obj.data=mesh
                if signature is not None:obj['built_geometry']=signature
                obj['gaze_edge_fitted']=fitted;obj['gaze_edge_pending']=pending
        self._area.header_text_set(None);self._window.cursor_modal_restore();self._area.tag_redraw()
    def modal(self,context,event):
        if event.type in {'ESC','RIGHTMOUSE'} and event.value=='PRESS':
            self._finish(True);return {'CANCELLED'}
        if event.type in {'LEFTMOUSE','RET','NUMPAD_ENTER'} and event.value=='PRESS':
            if not self._started:
                if event.type=='LEFTMOUSE' and self._inside(event):
                    self._started=True;self._last=(event.mouse_x,event.mouse_y)
                    self._area.header_text_set('Move to adjust gaze | Shift: fine | Click: confirm | Esc: cancel')
                return {'RUNNING_MODAL'}
            if event.type=='LEFTMOUSE' and not self._inside(event):return {'RUNNING_MODAL'}
            self._finish();return {'FINISHED'}
        if event.type in {'MIDDLEMOUSE','WHEELUPMOUSE','WHEELDOWNMOUSE','TRACKPADPAN','TRACKPADZOOM'}:
            self._last=None;return {'PASS_THROUGH'}
        if event.type=='MOUSEMOVE':
            if not self._started:return {'RUNNING_MODAL'}
            if not self._inside(event):self._last=None;return {'RUNNING_MODAL'}
            point=(event.mouse_x,event.mouse_y)
            if self._last is not None:
                v=context.scene.ocular_view
                x,z=mouse_gaze(v.gaze_x,v.gaze_z,point[0]-self._last[0],point[1]-self._last[1],event.shift)
                apply_gaze(context,x,z,self._pivot)
            self._last=point;self._area.tag_redraw()
        return {'RUNNING_MODAL'}
    def cancel(self,context):self._finish(True)


class OCULAR_OT_gaze_edge(bpy.types.Operator):
    bl_idname='anaplast.ocular_gaze_edge';bl_label='Apply gaze & refit edge';bl_options={'REGISTER','UNDO'}
    bl_description='Keep the iris and cornea at the chosen gaze and refit the rounded outer border to the original marked opening and extension'
    def execute(self,context):
        try:
            from .ocular_gaze_edge import refit
            refit(context)
        except Exception as e:
            context.scene.ocular_view.gaze_message=str(e);rep(self,{'ERROR'},str(e));return {'CANCELLED'}
        rep(self,{'INFO'},'Ocular border refitted; iris/cornea gaze and original extension retained')
        return {'FINISHED'}


def draw(layout,context):
    v=context.scene.ocular_view
    box=layout.box();box.label(text='Eye inspection')
    box.prop(v,'fit_tolerance')
    row=box.row(align=True);row.operator('anaplast.ocular_fit_check')
    if v.fit_active:
        row.operator('anaplast.ocular_fit_check',text='Return to colors').restore=True
        box.label(text='Green: contact; blue: gap; red: below Sculpt')
        box.label(text='Gray: outside painted sclera; no contact')
        box.label(text='Reference: original Sculpt sclera (wire)')
        import textwrap
        for line in textwrap.wrap(v.fit_report,48):box.label(text=line)

    row=box.row(align=True);row.operator('anaplast.ocular_torch')
    if not context.scene.objects.get('LM_NASION'):
        box.label(text='Place cursor at nasion for a manual light position.')
        box.operator('anaplast.ocular_torch',text='Torch at cursor').at_cursor=True
    if v.torch:
        box.prop(v,'torch_on');box.prop(v,'torch_distance');box.prop(v,'torch_brightness');box.prop(v,'torch_size')
    box.label(text='Gaze - X and Z only')
    box.prop(v,'gaze_x',slider=True);box.prop(v,'gaze_z',slider=True)
    row=box.row(align=True);row.operator('anaplast.ocular_gaze_mouse');row.operator('anaplast.ocular_gaze_reset')
    box.operator('anaplast.ocular_gaze_edge')
    core=context.scene.ocular_production.color_obj
    if core and core.get('gaze_edge_pending'):box.label(text='Apply gaze to refit the border before export.',icon='INFO')
    if v.gaze_message:box.label(text=v.gaze_message,icon='INFO' if v.gaze_message.startswith('Border refitted') else 'ERROR')


from .ocular_contact import OCULAR_OT_fit

_classes=(OCULAR_OT_fit,OcularView,OCULAR_OT_torch,OCULAR_OT_eye_center,OCULAR_OT_gaze_reset,OCULAR_OT_gaze_mouse,OCULAR_OT_gaze_edge)
def register():
    for c in _classes:bpy.utils.register_class(c)
    bpy.types.Scene.ocular_view=bpy.props.PointerProperty(type=OcularView)


def unregister():
    del bpy.types.Scene.ocular_view
    for c in reversed(_classes):bpy.utils.unregister_class(c)
