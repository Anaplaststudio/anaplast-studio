"""Reference photo overlay: a see-through picture of the patient (or a donor) floated over the models,
sized in mm, faded with a slider, facing the view you are working in."""
import os
import math
import bpy
from mathutils import Vector, Quaternion
from ..utils import mesh as mu
from .report import rep

COL = "Anaplast_Photos"


def photos(context):
    return [o for o in context.scene.objects if o.type == 'EMPTY' and o.empty_display_type == 'IMAGE']


def selected_photo(context):
    o=context.active_object
    return o if o is not None and o.select_get() and o in photos(context) else None


def scale_get(o):return float(o.scale.x*100)
def scale_set(o,value):
    old=o.scale.x
    if abs(old)>1e-9:o.scale*=value/(100*old)
    else:o.scale=(value/100,)*3


def opacity_get(o):return float(o.color[3])
def opacity_set(o,value):o.color[3]=value;o.use_empty_image_alpha=True


_timer_on = False


def photo_area(context,area_id=None):
    from .overlay import clipping_area
    if area_id:
        for win in context.window_manager.windows:
            for area in win.screen.areas:
                if area.type=='VIEW_3D' and area.as_pointer()==area_id:return area
    return clipping_area(context)


def reveal_photo(context,obj,area):
    """Image empties also require Extras, object-type visibility and Local View membership."""
    space=area.spaces.active
    space.overlay.show_overlays=True
    space.overlay.show_extras=True
    space.show_object_viewport_empty=True
    space.show_object_select_empty=True
    obj.hide_viewport=False
    obj.show_empty_image_perspective=True
    obj.show_empty_image_orthographic=True
    obj.show_empty_image_only_axis_aligned=False
    obj.empty_image_depth='FRONT';obj.empty_image_side='DOUBLE_SIDED'
    obj.show_in_front=True;obj.use_empty_image_alpha=True
    if obj.color[3]<.01:obj.color[3]=.7
    def reveal_branch(layer):
        contains=obj.name in layer.collection.all_objects
        if contains:
            layer.exclude=False;layer.hide_viewport=False;layer.collection.hide_viewport=False
            for child in layer.children:reveal_branch(child)
    reveal_branch(context.view_layer.layer_collection)
    context.view_layer.update()
    if space.local_view:obj.local_view_set(space,True)
    obj.hide_set(False)
    area.tag_redraw()


def save_position(obj,area):
    rv=area.spaces.active.region_3d
    obj['view_rot']=list(rv.view_rotation);obj['view_persp']=rv.view_perspective
    obj['view_dist']=rv.view_distance;obj['view_loc']=list(rv.view_location)
    obj['photo_saved_matrix']=[v for row in obj.matrix_world for v in row]
    obj['photo_saved_size']=obj.empty_display_size
    obj['photo_saved_lens']=area.spaces.active.lens


def update_visibility(context,area):
    if not area:return
    rv=area.spaces.active.region_3d
    for obj in photos(context):
        if 'view_rot' not in obj:continue
        angle=Quaternion(obj['view_rot']).rotation_difference(rv.view_rotation).angle
        angle=min(angle,abs(2*math.pi-angle))
        hidden=math.degrees(angle)>context.scene.anaplast.photo_tolerance
        if obj.hide_get()!=hidden:obj.hide_set(hidden)


_area_id=None
def _watch_photos():
    try:update_visibility(bpy.context,photo_area(bpy.context,_area_id))
    except (ReferenceError,RuntimeError,AttributeError):pass
    return 0.25


def start_watcher(area):
    global _area_id
    if area:_area_id=area.as_pointer()
    if not bpy.app.background and not bpy.app.timers.is_registered(_watch_photos):
        bpy.app.timers.register(_watch_photos,first_interval=.25,persistent=True)


@bpy.app.handlers.persistent
def photos_load(_):
    if photos(bpy.context):start_watcher(photo_area(bpy.context))


def apply_photo_settings(context):
    p = context.scene.anaplast
    act = context.active_object
    targets = [act] if selected_photo(context) else []
    for o in targets:
        o.color = (1.0, 1.0, 1.0, p.photo_opacity)
        o.use_empty_image_alpha = True
        o.empty_display_size = p.photo_width
        o.show_in_front = True
        try:
            o.empty_image_depth = 'FRONT'
        except Exception:
            pass


class ANAPLAST_OT_photo_add(bpy.types.Operator):
    """Load a photo and float it over the models, facing the current view (front view for a frontal photo, side view for a profile). Then fade it and move / scale it with the normal G / S / R keys"""
    bl_idname = "anaplast.photo_add"
    bl_label = "Add reference photo…"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH', default="")
    filter_glob: bpy.props.StringProperty(default="*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp;*.heic", options={'HIDDEN'})

    def invoke(self, context, event):
        area=photo_area(context)
        self._source_area=area.as_pointer() if area else None
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        p=context.scene.anaplast;area=photo_area(context,getattr(self,'_source_area',None))
        if area is None:
            rep(self,{'ERROR'},'Open a 3D viewport before adding a reference photo');return {'CANCELLED'}
        try:
            img=bpy.data.images.load(self.filepath,check_existing=True)
            img.pack()
        except RuntimeError as e:
            rep(self,{'ERROR'},f'Could not load photo: {e}');return {'CANCELLED'}
        if not all(img.size):
            rep(self,{'ERROR'},'This image has no readable pixels; save it as PNG or JPEG and try again');return {'CANCELLED'}
        if context.object and context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        col=mu.get_collection(context.scene,COL)
        emp=bpy.data.objects.new('Photo · '+os.path.basename(self.filepath),None)
        emp.empty_display_type='IMAGE';emp.data=img;col.objects.link(emp)
        rv=area.spaces.active.region_3d
        emp.rotation_euler=rv.view_rotation.to_euler();emp.location=rv.view_location
        emp.empty_image_offset=(-.5,-.5)
        emp.empty_image_depth='FRONT';emp.empty_image_side='DOUBLE_SIDED'
        emp.show_empty_image_perspective=True;emp.show_empty_image_orthographic=True
        emp.show_in_front=True;emp.hide_render=True
        emp.color=(1,1,1,.7);emp.use_empty_image_alpha=True
        # Fit the photo at view-center depth, preserving its aspect ratio.
        from .viewport_tools import visible_rect
        region=next((v for v in area.regions if v.type=='WINDOW'),None)
        width=max(1,region.width);height=max(1,region.height)
        l,r,b,t=visible_rect(area,region)
        pixel=rv.view_distance*72/area.spaces.active.lens/max(width,height)
        ratio=img.size[0]/max(1,img.size[1])
        emp.empty_display_size=.8*min((r-l)*pixel,(t-b)*pixel*ratio)/min(1,ratio)
        offset=Vector(((l+r-width)*.5*pixel,(b+t-height)*.5*pixel,0))
        emp.location+=rv.view_rotation@offset
        reveal_photo(context,emp,area)
        context.view_layer.update()
        for obj in context.selected_objects:obj.select_set(False)
        emp.hide_set(False);emp.select_set(True);context.view_layer.objects.active=emp
        context.view_layer.update();save_position(emp,area);start_watcher(area);update_visibility(context,area)
        rep(self,{'INFO'},'Photo added. Move with G, scale with S or the slider, then Save photo position. Each photo has its own Return button.')
        return {'FINISHED'}


class ANAPLAST_OT_photo_face_view(bpy.types.Operator):
    """Save the selected photo's position, scale and current viewing angle"""
    bl_idname='anaplast.photo_face_view';bl_label='Save photo position'
    @classmethod
    def poll(cls,context):return selected_photo(context) is not None
    def execute(self,context):
        obj=selected_photo(context);area=photo_area(context)
        if not obj or not area:return {'CANCELLED'}
        context.view_layer.update();save_position(obj,area);start_watcher(area)
        rep(self,{'INFO'},f'Saved position and view for {obj.name}')
        return {'FINISHED'}


class ANAPLAST_OT_photo_view(bpy.types.Operator):
    """Return to this photo's saved position and view without changing other photos"""
    bl_idname='anaplast.photo_view';bl_label='Return to photo'
    photo_name:bpy.props.StringProperty()
    def execute(self,context):
        obj=context.scene.objects.get(self.photo_name) if self.photo_name else selected_photo(context)
        if obj not in photos(context):return {'CANCELLED'}
        if 'view_rot' not in obj:
            rep(self,{'ERROR'},'Select the photo and Save photo position first');return {'CANCELLED'}
        area=photo_area(context)
        if not area:return {'CANCELLED'}
        if context.object and context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        from mathutils import Matrix
        rv=area.spaces.active.region_3d
        rv.view_rotation=Quaternion(obj['view_rot']);rv.view_location=Vector(obj['view_loc'])
        rv.view_distance=float(obj['view_dist']);rv.view_perspective=str(obj.get('view_persp','ORTHO'))
        if 'photo_saved_lens' in obj:area.spaces.active.lens=float(obj['photo_saved_lens'])
        if 'photo_saved_matrix' in obj:
            v=list(obj['photo_saved_matrix']);obj.matrix_world=Matrix([v[i:i+4] for i in range(0,16,4)])
            obj.empty_display_size=float(obj['photo_saved_size'])
        for other in context.selected_objects:other.select_set(False)
        reveal_photo(context,obj,area)
        obj.select_set(True);context.view_layer.objects.active=obj
        start_watcher(area);update_visibility(context,area);area.tag_redraw()
        return {'FINISHED'}


class ANAPLAST_OT_photo_scale_two_points(bpy.types.Operator):
    """Scale the photo to life size: click two points on the PHOTO that are a known distance apart (e.g. the pupils, or a ruler in the picture), then type that distance"""
    bl_idname = "anaplast.photo_scale"
    bl_label = "Scale photo by 2 points"
    bl_options = {'REGISTER', 'UNDO'}

    known_mm: bpy.props.FloatProperty(name="Real distance (mm)", default=63.0, min=1.0, soft_max=300.0,
                                      description="Adult inter-pupillary distance is ~63 mm")
    _pts = None

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == 'EMPTY' and o.empty_display_type == 'IMAGE' and context.area is not None and context.area.type == 'VIEW_3D'

    def invoke(self, context, event):
        self._pts = []
        context.window_manager.modal_handler_add(self)
        context.window.cursor_modal_set('CROSSHAIR')
        context.area.header_text_set("Click the FIRST reference point on the photo · ESC cancel")
        return {'RUNNING_MODAL'}

    def _on_photo(self, context, event):
        from bpy_extras import view3d_utils
        from mathutils.geometry import intersect_line_plane
        emp = context.active_object
        region, rv3d = context.region, context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        normal = emp.matrix_world.to_3x3() @ Vector((0, 0, 1))
        hit = intersect_line_plane(origin, origin + direction * 1e5, emp.matrix_world.translation, normal)
        return hit

    def modal(self, context, event):
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM'}:
            return {'PASS_THROUGH'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            hit = self._on_photo(context, event)
            if hit is None:
                return {'RUNNING_MODAL'}
            self._pts.append(hit)
            if len(self._pts) == 1:
                context.area.header_text_set("Click the SECOND reference point · ESC cancel")
                return {'RUNNING_MODAL'}
            emp = context.active_object
            d = (self._pts[1] - self._pts[0]).length
            if d < 1e-3:
                self._finish(context)
                return {'CANCELLED'}
            factor = self.known_mm / d
            emp.empty_display_size *= factor
            context.scene.anaplast.photo_width = emp.empty_display_size
            self._finish(context)
            rep(self, {'INFO'}, f"Photo scaled ×{factor:.3f} so those points are {self.known_mm:.1f} mm apart (now {emp.empty_display_size:.0f} mm wide)")
            return {'FINISHED'}
        if event.type == 'ESC':
            self._finish(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        context.area.header_text_set(None)
        context.window.cursor_modal_restore()


class ANAPLAST_OT_photo_remove(bpy.types.Operator):
    """Remove only the selected reference photo"""
    bl_idname='anaplast.photo_remove';bl_label='Remove selected photo';bl_options={'REGISTER','UNDO'}
    @classmethod
    def poll(cls,context):return selected_photo(context) is not None
    def execute(self,context):
        obj=selected_photo(context)
        if not obj:return {'CANCELLED'}
        bpy.data.objects.remove(obj,do_unlink=True)
        return {'FINISHED'}


_classes = (ANAPLAST_OT_photo_add, ANAPLAST_OT_photo_face_view, ANAPLAST_OT_photo_view, ANAPLAST_OT_photo_remove)


def register():
    bpy.app.handlers.load_post.append(photos_load)
    bpy.types.Object.b4d_photo_scale=bpy.props.FloatProperty(name='Scale (%)',min=1,soft_max=300,get=scale_get,set=scale_set)
    bpy.types.Object.b4d_photo_opacity=bpy.props.FloatProperty(name='Opacity',min=0,max=1,get=opacity_get,set=opacity_set)
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    if photos_load in bpy.app.handlers.load_post:bpy.app.handlers.load_post.remove(photos_load)
    if bpy.app.timers.is_registered(_watch_photos):bpy.app.timers.unregister(_watch_photos)
    del bpy.types.Object.b4d_photo_scale
    del bpy.types.Object.b4d_photo_opacity
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
