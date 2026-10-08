"""One eye photograph and source outlines shared by ocular and iris button."""
import json
import bpy
import numpy as np
from bpy.props import StringProperty,EnumProperty
from . import ocular_production as prod,ocular_pupil

PHOTO_FIELDS=('photo','photo_center','photo_radius','iris_rotation','photo_pupil',
              'photo_pupil_outline','pupil_outline_image')


def source(scene):
    """Read legacy inputs without deleting their photos or manufacturing settings."""
    p=scene.ocular_production;w=scene.ocular_photos
    if w.single_image and w.image:
        data={k:getattr(p,k) for k in PHOTO_FIELDS}
        data.update(photo=w.image,photo_center=tuple(w.center),photo_radius=tuple(w.radius),iris_rotation=w.rotation,mirror=w.mirror)
        return data
    if not p.photo:p=scene.iris_button.photo
    data={k:getattr(p,k) for k in PHOTO_FIELDS};data['mirror']=False
    return data


def adopt(scene):
    """Bring an existing input into the shared editor; no sclera trace is needed for a button."""
    data=source(scene)
    if not data['photo']:raise RuntimeError('Load an eye photograph first')
    w=scene.ocular_photos;p=scene.ocular_production
    if w.image!=data['photo']:w.border='[]';w.extended_image=None
    w.image=data['photo'];w.center=data['photo_center'];w.radius=data['photo_radius']
    w.rotation=data['iris_rotation'];w.mirror=data['mirror'];w.single_image=True
    for k in PHOTO_FIELDS:setattr(p,k,data[k])
    scene.iris_button.editing_photo=False
    return w.image


def load(scene,image):
    prod.pixels(image);image.pack()
    if source(scene)['photo']==image:
        adopt(scene);return
    w=scene.ocular_photos;p=scene.ocular_production
    w.image=image;w.single_image=True;w.center=(.5,.5);w.radius=(.15,.3);w.rotation=0;w.mirror=False
    w.border='[]';w.extended_image=None;w.extend_photo=True;w.extension_method='STRETCH'
    p.photo=image;p.photo_center=w.center;p.photo_radius=w.radius;p.iris_rotation=0
    p.photo_pupil_outline='[]';p.pupil_outline_image=None;p.photo_pupil=.3
    scene.iris_button.editing_photo=False


class EYE_OT_load(bpy.types.Operator):
    bl_idname='anaplast.eye_photo_load';bl_label='Load eye photograph';bl_options={'REGISTER','UNDO'}
    filepath:StringProperty(subtype='FILE_PATH')
    filter_glob:StringProperty(default='*.png;*.jpg;*.jpeg;*.tif;*.tiff',options={'HIDDEN'})
    def invoke(self,c,event):c.window_manager.fileselect_add(self);return {'RUNNING_MODAL'}
    def execute(self,c):
        try:load(c.scene,bpy.data.images.load(self.filepath,check_existing=True))
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class EYE_OT_view(bpy.types.Operator):
    bl_idname='anaplast.eye_photo_view';bl_label='Mark iris and pupil';bl_options={'REGISTER'}
    def execute(self,c):
        try:image=adopt(c.scene)
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        c.area.type='IMAGE_EDITOR';c.area.spaces.active.image=image;c.area.spaces.active.show_region_ui=True
        region=next((r for r in c.area.regions if r.type=='WINDOW'),None)
        if region:
            with c.temp_override(region=region):bpy.ops.image.view_all(fit_view=True)
        return {'FINISHED'}


def iris_oval(scene,a,b):
    lo=np.minimum(a,b);hi=np.maximum(a,b)
    if np.min(hi-lo)<.005:raise RuntimeError('Drag a larger oval around the complete iris')
    w=scene.ocular_photos;p=scene.ocular_production
    w.center=(lo+hi)*.5;w.radius=(hi-lo)*.5
    p.photo=w.image;p.photo_center=w.center;p.photo_radius=w.radius;p.iris_rotation=w.rotation


def pupil_stroke(scene,points):
    """Remove hand jitter and evenly sample a closed, noncircular source outline."""
    pts=np.asarray(points,dtype=float)
    if len(pts)<6:raise RuntimeError('Draw all the way around the pupil')
    delta=np.linalg.norm(np.roll(pts,-1,axis=0)-pts,axis=1)
    pts=pts[delta>1e-6]
    if len(pts)<6:raise RuntimeError('Draw a larger pupil outline')
    # Two gentle smoothing passes preserve asymmetry and centre offset.
    for _ in range(2):pts=.75*pts+.125*(np.roll(pts,1,axis=0)+np.roll(pts,-1,axis=0))
    delta=np.linalg.norm(np.roll(pts,-1,axis=0)-pts,axis=1)
    t=np.r_[0,np.cumsum(delta)];closed=np.vstack((pts,pts[0]))
    samples=np.linspace(0,t[-1],48,endpoint=False)
    pts=np.column_stack([np.interp(samples,t,closed[:,i]) for i in range(2)])
    p=scene.ocular_production;w=scene.ocular_photos
    before=(p.photo_pupil_outline,p.pupil_outline_image)
    p.photo=w.image;p.photo_center=w.center;p.photo_radius=w.radius;p.iris_rotation=w.rotation
    p.photo_pupil_outline=json.dumps(pts.tolist());p.pupil_outline_image=w.image
    try:model=ocular_pupil.contour(p)
    except Exception:
        p.photo_pupil_outline,p.pupil_outline_image=before;raise
    p.photo_pupil=model['equivalent']


class EYE_OT_outline(bpy.types.Operator):
    bl_idname='anaplast.eye_photo_outline';bl_label='Draw photo outline';bl_options={'REGISTER','UNDO'}
    kind:EnumProperty(items=[('IRIS','Drag iris oval',''),('PUPIL','Draw pupil outline','')])
    @classmethod
    def poll(cls,c):return c.area and c.area.type=='IMAGE_EDITOR' and c.space_data.image is not None and c.space_data.image==c.scene.ocular_photos.image
    def invoke(self,c,event):
        self.dragging=False;self.points=[]
        w=c.scene.ocular_photos;p=c.scene.ocular_production
        self.before=(tuple(w.center),tuple(w.radius),tuple(p.photo_center),tuple(p.photo_radius),p.photo_pupil_outline,p.pupil_outline_image,p.photo,p.iris_rotation)
        c.window_manager.modal_handler_add(self);c.window.cursor_modal_set('CROSSHAIR')
        c.area.header_text_set('Drag from one corner of the iris oval to its opposite corner; release to finish; Esc cancel' if self.kind=='IRIS' else 'Draw around pupil and release near the starting point; Esc cancel')
        return {'RUNNING_MODAL'}
    def finish(self,c):c.window.cursor_modal_restore();c.area.header_text_set(None);c.area.tag_redraw()
    def restore(self,c):
        w=c.scene.ocular_photos;p=c.scene.ocular_production
        w.center,w.radius,p.photo_center,p.photo_radius,p.photo_pupil_outline,p.pupil_outline_image,p.photo,p.iris_rotation=self.before
    def modal(self,c,event):
        if event.type in {'ESC','RIGHTMOUSE'}:
            self.restore(c);self.finish(c);return {'CANCELLED'}
        region=next(r for r in c.area.regions if r.type=='WINDOW')
        x,y=event.mouse_x-region.x,event.mouse_y-region.y
        inside=0<=x<region.width and 0<=y<region.height
        uv=region.view2d.region_to_view(x,y)
        valid=inside and all(0<=v<=1 for v in uv)
        if event.type=='LEFTMOUSE' and event.value=='PRESS' and valid:
            self.dragging=True;self.start=uv;self.start_pixel=(x,y);self.last_pixel=(x,y);self.points=[list(uv)]
            return {'RUNNING_MODAL'}
        if event.type=='MOUSEMOVE' and self.dragging and valid:
            if self.kind=='IRIS':
                try:iris_oval(c.scene,self.start,uv)
                except RuntimeError:pass
            elif np.linalg.norm(np.array((x,y))-self.last_pixel)>=3:
                self.points.append(list(uv));self.last_pixel=(x,y)
                p=c.scene.ocular_production;p.photo_pupil_outline=json.dumps(self.points);p.pupil_outline_image=c.scene.ocular_photos.image
            c.area.tag_redraw();return {'RUNNING_MODAL'}
        if event.type=='LEFTMOUSE' and event.value=='RELEASE' and self.dragging:
            try:
                if not valid:raise RuntimeError('Finish the outline inside the photograph')
                if self.kind=='IRIS':iris_oval(c.scene,self.start,uv)
                else:
                    if np.linalg.norm(np.array((x,y))-self.start_pixel)>30:raise RuntimeError('Finish close to the start to close the pupil outline')
                    # Restore stored outline first so a rejected stroke has a valid rollback.
                    c.scene.ocular_production.photo_pupil_outline=self.before[4]
                    c.scene.ocular_production.pupil_outline_image=self.before[5]
                    pupil_stroke(c.scene,self.points)
            except Exception as exc:
                self.restore(c);self.report({'WARNING'},str(exc));self.finish(c);return {'CANCELLED'}
            self.finish(c);return {'FINISHED'}
        return {'PASS_THROUGH'}


def draw(layout,context):
    data=source(context.scene)
    layout.operator('anaplast.eye_photo_load',icon='IMAGE_DATA')
    if data['photo']:
        layout.label(text=data['photo'].name)
        layout.operator('anaplast.eye_photo_view',icon='GREASEPENCIL')
    layout.label(text='One design; choose the output at export')


_classes=(EYE_OT_load,EYE_OT_view,EYE_OT_outline)
def register():
    for cls in _classes:bpy.utils.register_class(cls)
def unregister():
    for cls in reversed(_classes):bpy.utils.unregister_class(cls)
