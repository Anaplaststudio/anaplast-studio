"""Local scleral photo registration and independent color layers."""
import json,math
from pathlib import Path
import numpy as np
import bpy
from . import ocular_production as prod


def sync_whole_eye(scene):
    """Use one image and one iris registration for both independent layers."""
    w=scene.ocular_photos;p=scene.ocular_production
    if not w.image and not p.photo and getattr(scene,'iris_button',None) and scene.iris_button.photo.photo:
        from .eye_photo import adopt
        adopt(scene)
    if not w.single_image:return
    if not w.image:raise RuntimeError('Load one whole-eye photograph')
    if len(json.loads(w.border or '[]'))<3:raise RuntimeError('Align the whole-eye photo and trace its eye opening to exclude eyelids')
    prod.pixels(w.image)
    if w.extend_photo and w.extension_method=='PATCH':
        lo=np.minimum(w.patch_min,w.patch_max);hi=np.maximum(w.patch_min,w.patch_max)
        if np.any(hi-lo<.005):raise RuntimeError('Choose a clean sclera patch with two corners')
        nearest=np.clip(np.array(w.center),lo,hi)
        if np.sum(((nearest-np.array(w.center))/np.array(w.radius))**2)<1.05**2:
            raise RuntimeError('Move the clean sclera patch outside the iris')
    p.photo=w.image;p.photo_center=w.center;p.photo_radius=w.radius;p.iris_rotation=w.rotation


def extend_patch(w,pixels,uv):
    """Cover unphotographed areas with overlapping samples of a clean photo patch."""
    lo=np.minimum(w.patch_min,w.patch_max);hi=np.maximum(w.patch_min,w.patch_max);span=hi-lo
    if np.any(span<.005) or np.any(lo<0) or np.any(hi>1):raise RuntimeError('Choose two valid corners around a clean scleral patch, without iris, lids or lashes')
    tile=(np.asarray(uv)-lo)/(span*.35);cell=np.floor(tile)
    result=np.zeros((*tile.shape[:-1],3))
    for i,j in ((0,0),(1,0),(0,1),(1,1)):
        d=tile-cell-np.array([i,j]);weights=(1-abs(d[...,0]))*(1-abs(d[...,1]))
        seed=np.mod((cell[...,0]+i)*73856093+(cell[...,1]+j)*19349663,15485863)
        sx=np.where(np.mod(seed,2)<1,-1.,1.);sy=np.where(np.mod(np.floor(seed/2),2)<1,-1.,1.)
        sample_uv=lo+span*np.stack((.5+.35*d[...,0]*sx,.5+.35*d[...,1]*sy),axis=-1)
        result+=prod.sample(pixels,sample_uv)[...,:3]*weights[...,None]
    return result


def sclera_rgb(w,xy,cc,radius,fallback):
    q=(np.asarray(xy)-cc)/radius;a=math.radians(w.rotation)
    mapped=np.stack((q[...,0]*math.cos(a)-q[...,1]*math.sin(a),q[...,0]*math.sin(a)+q[...,1]*math.cos(a)),axis=-1)
    if w.mirror:mapped[...,0]*=-1
    uv=np.array(w.center)+mapped*np.array(w.radius)
    c=np.array(w.center);r=np.array(w.radius)
    if np.any(c-r<0) or np.any(c+r>1):raise RuntimeError('The iris reference ellipse exceeds the sclera photo; align its center and edges')
    pixels=prod.pixels(w.image)
    exposure=w.iris_exposure if w.link_exposure and w.image==w.id_data.ocular_production.photo else w.exposure
    if w.preserve_photo:exposure=0.
    if w.extend_photo and w.extension_method in {'STRETCH','AI_IMAGE'}:
        from .ocular_photo_extend import extend,ai_extension
        result=extend(w,pixels,mapped,radius,0 if w.extension_method=='AI_IMAGE' else None)
        if w.extension_method=='AI_IMAGE':
            # Keep the original photographed sclera inside its traced opening.
            from .ocular import signed_distance
            border=np.array(json.loads(w.border or '[]'))
            distance=signed_distance(uv.reshape(-1,2),border).reshape(uv.shape[:-1])
            rho=np.linalg.norm(mapped,axis=-1)
            source=prod.sample(pixels,uv)[...,:3]
            keep=(distance<=0)&(rho>=1.01)
            result=np.where(keep[...,None],source,result)
            result=ai_extension(w,pixels,uv,result)
        return np.clip(result*(2**exposure),0,1)
    # At the limbus, use photographed color just outside the marked iris.
    # The previous exclusion ramp inserted the pale fallback here, making a halo.
    rho=np.linalg.norm(mapped,axis=-1)
    safe=mapped*np.maximum(1,1.01/np.maximum(rho,1e-8))[...,None]
    photo=prod.sample(pixels,np.array(w.center)+safe*np.array(w.radius))
    valid=np.clip(np.minimum(uv,1-uv).min(axis=-1)/.015,0,1)
    border=json.loads(w.border or '[]')
    if border:
        if len(border)<3:raise RuntimeError('Finish the sclera photo boundary with at least three points')
        from .ocular import signed_distance
        distance=signed_distance(uv.reshape(-1,2),np.array(border)).reshape(uv.shape[:-1]);valid*=np.clip(-distance/w.feather,0,1)
    if w.extend_photo:fallback=np.clip(extend_patch(w,pixels,uv)*(2**exposure),0,1)
    weight=valid*photo[...,3];rgb=np.clip(photo[...,:3]*(2**exposure),0,1)
    return rgb*weight[...,None]+fallback*(1-weight[...,None])


def set_point(w,uv):
    if w.point=='CENTER':w.center=uv
    elif w.point=='HORIZONTAL':w.radius[0]=max(.001,abs(uv[0]-w.center[0]))
    elif w.point=='VERTICAL':w.radius[1]=max(.001,abs(uv[1]-w.center[1]))
    elif w.point=='PATCH_MIN':w.patch_min=uv
    elif w.point=='PATCH_MAX':w.patch_max=uv
    else:
        points=json.loads(w.border or '[]');points.append(list(uv));w.border=json.dumps(points)


def record_settings(core,p,w):
    exposure=w.iris_exposure if w.link_exposure and w.image==p.photo else w.exposure
    if w.preserve_photo:exposure=0.
    core['photo_layer_alignment']=json.dumps(dict(iris_photo_center=list(p.photo_center),iris_photo_radius=list(p.photo_radius),iris_rotation=p.iris_rotation,iris_exposure=w.iris_exposure,sclera_photo_center=list(w.center),sclera_photo_radius=list(w.radius),sclera_rotation=w.rotation,sclera_mirror=w.mirror,sclera_exposure=exposure,sclera_boundary=json.loads(w.border or '[]')))
    core['sclera_extension']=json.dumps(dict(method=w.extension_method,enabled=w.extend_photo,overlap_mm=w.photo_extension_mm,source_edge_inset_mm=w.photo_edge_inset_mm,linked_exposure=w.link_exposure,ai_image=w.extended_image.name if w.extended_image else None))
    if w.image:
        core['photo_appearance']='Iris and sclera photographs; '+('externally AI-extended sclera outside the traced opening' if w.extend_photo and w.extension_method=='AI_IMAGE' else 'complete traced sclera fitted radially to the front edge; fixed iris; separate mirrored back' if w.extend_photo and w.extension_method=='STRETCH' else 'photo border blended with sampled or base colors')
    core['source_photo_colors_preserved']=w.preserve_photo
    core['mirrored_sclera_back']=w.mirror_back
    core['effective_photo_exposure_stops']=exposure
    core['effective_iris_repair_degrees']=0. if w.preserve_photo else p.upper_repair


class OcularPhotos(bpy.types.PropertyGroup):
    show_photo_fine:bpy.props.BoolProperty(name='Fine adjustments',default=False)
    preserve_photo:bpy.props.BoolProperty(name='Use original photo colors and detail',default=True,description='Bypass exposure and iris-repair edits; retain source colors and vessels. Photo stretch fits the full traced sclera radially to the front edge. Lighting can still change the rendered appearance')
    mirror_back:bpy.props.BoolProperty(name='Mirrored sclera photograph on back',default=True,description='The sclera looks left/right mirrored compared with the front when viewed from behind. Fit radially to the edge; replace iris and pupil with a smooth scleral fill')
    single_image:bpy.props.BoolProperty(name='Use one whole-eye photo',default=False,description='Use the same photograph and iris alignment for sclera and iris; trace the photo opening to exclude lids')
    image:bpy.props.PointerProperty(type=bpy.types.Image)
    center:bpy.props.FloatVectorProperty(name='Iris center in sclera photo',size=2,default=(.5,.5),min=0,max=1)
    radius:bpy.props.FloatVectorProperty(name='Iris half-width / height in photo',size=2,default=(.15,.3),min=.001,max=1)
    rotation:bpy.props.FloatProperty(name='Sclera photo rotation',default=0,min=-180,max=180)
    mirror:bpy.props.BoolProperty(name='Mirror sclera photo',default=False)
    exposure:bpy.props.FloatProperty(name='Sclera exposure (stops)',default=0,min=-3,max=3)
    feather:bpy.props.FloatProperty(name='Photo border blend',default=.025,min=.002,max=.08,description='Blend width as a fraction of the photograph')
    extend_photo:bpy.props.BoolProperty(name='Cover entire ocular with photo detail',default=False,description='Extend photographed sclera beyond its traced outline using photo stretch, an imported AI extension, or a legacy patch')
    extension_method:bpy.props.EnumProperty(name='Sclera extension',items=[('STRETCH','Photo stretch','Fit the full traced sclera radially between the unchanged iris and ocular front edge. The mirrored back has its own map'),('AI_IMAGE','AI-extended photo','Load a same-canvas AI result; keep the original photo inside its traced opening'),('PATCH','Sample patch (legacy)','Repeat a selected clean sclera patch')],default='STRETCH')
    photo_extension_mm:bpy.props.FloatProperty(name='Minimum photo overlap (mm)',default=1.,min=0,max=10,description='Minimum extension beyond the photo border; automatically increases as needed to cover the ocular texture footprint. Does not change the physical ocular')
    photo_edge_inset_mm:bpy.props.FloatProperty(name='Avoid lid edge (mm)',default=.5,min=0,max=2,description='Keep extension samples this far inside the traced photo border, to avoid stretching eyelid or tear-edge colors')
    link_exposure:bpy.props.BoolProperty(name='Match iris and sclera brightness',default=True,description='When both use the same source photo, use Iris exposure for both to avoid a brightness seam')
    extended_image:bpy.props.PointerProperty(type=bpy.types.Image)
    patch_min:bpy.props.FloatVectorProperty(name='Patch lower-left',size=2,default=(.211,.463),min=0,max=1)
    patch_max:bpy.props.FloatVectorProperty(name='Patch upper-right',size=2,default=(.290,.616),min=0,max=1)
    iris_exposure:bpy.props.FloatProperty(name='Iris exposure (stops)',default=0,min=-3,max=3)
    border:bpy.props.StringProperty(default='[]')
    point:bpy.props.EnumProperty(name='Photo landmark',items=[('CENTER','Iris center',''),('HORIZONTAL','Iris left/right edge',''),('VERTICAL','Iris top/bottom edge',''),('BORDER','Eye opening boundary','Click around the opening to exclude eyelids; Enter finishes'),('PATCH_MIN','Clean patch lower-left',''),('PATCH_MAX','Clean patch upper-right','')])
    iris_map:bpy.props.PointerProperty(type=bpy.types.Image)
    sclera_map:bpy.props.PointerProperty(type=bpy.types.Image)
    sclera_wrap:bpy.props.PointerProperty(type=bpy.types.Image)


class OCULAR_OT_sclera_load(bpy.types.Operator):
    bl_idname='anaplast.ocular_sclera_load';bl_label='Load sclera / whole-eye photo';bl_options={'REGISTER','UNDO'}
    filepath:bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob:bpy.props.StringProperty(default='*.png;*.jpg;*.jpeg;*.tif;*.tiff',options={'HIDDEN'})
    def invoke(self,context,event):context.window_manager.fileselect_add(self);return {'RUNNING_MODAL'}
    def execute(self,context):
        try:
            image=bpy.data.images.load(self.filepath,check_existing=True);prod.pixels(image);image.pack();w=context.scene.ocular_photos
            w.image=image;w.border='[]';w.center=(.5,.5);w.radius=(.15,.3);w.extend_photo=False;w.extended_image=None
            if w.single_image:
                w.extend_photo=True;w.extension_method='STRETCH'
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}


class OCULAR_OT_sclera_reference(bpy.types.Operator):
    bl_idname='anaplast.ocular_sclera_reference';bl_label='Load real sclera reference';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        path=Path(__file__).resolve().parent.parent/'assets/ocular/Rnbc_sclera.jpg'
        result=bpy.ops.anaplast.ocular_sclera_load(filepath=str(path))
        if result!={'FINISHED'}:return result
        w=context.scene.ocular_photos;p=context.scene.ocular_production
        w.center=(.470,.567);w.radius=(.165,.219);w.rotation=0;w.mirror=False;w.exposure=-.1
        border=[(345,545),(465,460),(640,460),(900,470),(1125,505),(1240,575),(1285,660),(1260,735),(1190,808),(1050,885),(865,914),(650,892),(490,842),(365,772),(329,669)]
        w.border=json.dumps([(x/1824,1-y/1368) for x,y in border])
        w.feather=.035
        w.extend_photo=True;w.patch_min=(385/1824,1-735/1368);w.patch_max=(530/1824,1-525/1368)
        p.vein_strength=0;p.sclera=(.94,.93,.90)
        w.image['source']='Brown human eye.jpg, Rnbc, Wikimedia Commons, CC BY-SA 3.0'
        credit=(Path(__file__).resolve().parent.parent/'assets/ocular/Sclera_Attribution.txt').read_text(encoding='utf-8')
        if credit not in p.photo_credit:p.photo_credit+='\n\n'+credit
        self.report({'INFO'},'Real sclera photograph loaded locally; review alignment and boundary')
        return {'FINISHED'}


class OCULAR_OT_sclera_view(bpy.types.Operator):
    bl_idname='anaplast.ocular_sclera_view';bl_label='Align sclera photo'
    def execute(self,context):
        if not context.scene.ocular_photos.image:self.report({'ERROR'},'Load a sclera photograph first');return {'CANCELLED'}
        context.area.type='IMAGE_EDITOR';context.area.spaces.active.image=context.scene.ocular_photos.image;context.area.spaces.active.show_region_ui=True
        return {'FINISHED'}


class OCULAR_OT_sclera_point(bpy.types.Operator):
    bl_idname='anaplast.ocular_sclera_point';bl_label='Click selected photo landmark';bl_options={'REGISTER','UNDO'}
    replace_border:bpy.props.BoolProperty(default=False,options={'SKIP_SAVE'})
    def invoke(self,context,event):
        if context.area.type!='IMAGE_EDITOR' or context.space_data.image!=context.scene.ocular_photos.image:return {'CANCELLED'}
        self.before=context.scene.ocular_photos.border
        if self.replace_border:context.scene.ocular_photos.point='BORDER';context.scene.ocular_photos.border='[]'
        context.window_manager.modal_handler_add(self);context.window.cursor_modal_set('CROSSHAIR');context.area.header_text_set('Click landmark. Boundary: click points, Enter finish, Backspace undo, Esc cancel.');return {'RUNNING_MODAL'}
    def finish(self,context):context.window.cursor_modal_restore();context.area.header_text_set(None)
    def modal(self,context,event):
        w=context.scene.ocular_photos
        if event.type in {'ESC','RIGHTMOUSE'}:
            w.border=self.before;self.finish(context);return {'CANCELLED'}
        if event.type in {'RET','NUMPAD_ENTER'}:
            self.finish(context);return {'FINISHED'}
        if event.type=='BACK_SPACE' and event.value=='PRESS':
            pts=json.loads(w.border);w.border=json.dumps(pts[:-1]);context.area.tag_redraw()
        if event.type=='LEFTMOUSE' and event.value=='PRESS':
            region=next(r for r in context.area.regions if r.type=='WINDOW');x,y=event.mouse_x-region.x,event.mouse_y-region.y
            if not(0<=x<region.width and 0<=y<region.height):return {'RUNNING_MODAL'}
            uv=region.view2d.region_to_view(x,y)
            if not all(0<=v<=1 for v in uv):return {'RUNNING_MODAL'}
            set_point(w,uv);context.area.tag_redraw()
            if w.point!='BORDER':self.finish(context);return {'FINISHED'}
        return {'RUNNING_MODAL'}


class OCULAR_OT_clear_border(bpy.types.Operator):
    bl_idname='anaplast.ocular_clear_photo_border';bl_label='Clear photo boundary';bl_options={'REGISTER','UNDO'}
    def execute(self,context):context.scene.ocular_photos.border='[]';return {'FINISHED'}


class OCULAR_OT_extended_photo(bpy.types.Operator):
    bl_idname='anaplast.ocular_extended_photo';bl_label='Load AI-extended photo';bl_options={'REGISTER','UNDO'}
    filepath:bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob:bpy.props.StringProperty(default='*.png;*.jpg;*.jpeg;*.tif;*.tiff',options={'HIDDEN'})
    def invoke(self,context,event):context.window_manager.fileselect_add(self);return {'RUNNING_MODAL'}
    def execute(self,context):
        w=context.scene.ocular_photos
        try:
            if not w.image:raise RuntimeError('Load the original eye photo first')
            image=bpy.data.images.load(self.filepath,check_existing=True)
            if tuple(image.size)!=tuple(w.image.size):raise RuntimeError('Keep the original image dimensions and alignment in the AI result')
            image.pack();w.extended_image=image;w.extension_method='AI_IMAGE';w.extend_photo=True
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}


class OCULAR_OT_extension_guide(bpy.types.Operator):
    bl_idname='anaplast.ocular_extension_guide';bl_label='Save AI extension guide'
    directory:bpy.props.StringProperty(subtype='DIR_PATH')
    def invoke(self,context,event):context.window_manager.fileselect_add(self);return {'RUNNING_MODAL'}
    def execute(self,context):
        try:
            from .ocular_photo_extend import save_guide
            path=save_guide(context.scene,self.directory)
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        self.report({'INFO'},'Saved source, edit mask and prompt: '+str(path));return {'FINISHED'}


class OCULAR_OT_update_photo_colors(bpy.types.Operator):
    bl_idname='anaplast.ocular_update_photo_colors';bl_label='Update photo colors';bl_options={'REGISTER','UNDO'}
    bl_description='Update color maps without moving the ocular or changing its shape, gaze, key or cornea'
    def execute(self,context):
        try:
            from .ocular_photo_extend import update_colors
            update_colors(context.scene)
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        self.report({'INFO'},'Photo colors updated; geometry and gaze unchanged');return {'FINISHED'}


def draw_extension(l,p):
    l.prop(p,'preserve_photo');l.prop(p,'mirror_back')
    l.prop(p,'extend_photo',text='Extend sclera photo')
    if p.extend_photo:
        l.prop(p,'extension_method')
        if p.extension_method=='PATCH':l.prop(p,'patch_min');l.prop(p,'patch_max')
        else:
            l.prop(p,'photo_extension_mm')
            if not p.preserve_photo:l.prop(p,'photo_edge_inset_mm')
            if p.extension_method=='AI_IMAGE':
                l.operator('anaplast.ocular_extension_guide');l.operator('anaplast.ocular_extended_photo')
                l.label(text='Create extension in your AI tool, then load it')
                if p.extended_image:l.label(text=p.extended_image.name)
    if p.image and p.image==p.id_data.ocular_production.photo:
        l.prop(p,'link_exposure')
        if p.link_exposure:
            row=l.row();row.enabled=not p.preserve_photo;row.prop(p,'iris_exposure',text='Photo exposure (stops)')
    if p.id_data.ocular_production.color_obj:l.operator('anaplast.ocular_update_photo_colors')


def draw_controls(l,p):
    l.operator('anaplast.eye_photo_outline',text='1. Drag iris oval').kind='IRIS'
    l.operator('anaplast.eye_photo_outline',text='2. Draw pupil outline').kind='PUPIL'
    l.label(text='Marks are shared by ocular and iris button')
    l.separator()
    l.label(text='Ocular only: trace the visible eye opening')
    l.operator('anaplast.ocular_sclera_point',text='Trace eye opening').replace_border=True
    l.prop(p,'show_photo_fine')
    if p.show_photo_fine:
        l.prop(p,'point');l.operator('anaplast.ocular_sclera_point')
        l.prop(p,'center');l.prop(p,'radius');l.prop(p,'rotation');l.prop(p,'mirror')
        l.operator('anaplast.ocular_clear_photo_border');l.prop(p,'feather')
        iris=p.id_data.ocular_production
        from .ocular_pupil import controls
        controls(l,iris)
        l.prop(iris,'photo_pupil',text='Pupil / iris radius in photo')
    l.operator('anaplast.ocular_photo_view',text='Return to model').back=True


def draw_overlay(context):
    w=context.scene.ocular_photos
    if not w.image or context.space_data.image!=w.image:return False
    import gpu
    from gpu_extras.batch import batch_for_shader
    a=np.arange(97)*2*np.pi/96;points=np.array(w.center)+np.array(w.radius)*np.c_[np.cos(a),np.sin(a)]
    lines=[]
    sequences=[points,json.loads(w.border or '[]')]
    if w.extend_photo and w.extension_method=='PATCH':
        x0,y0=w.patch_min;x1,y1=w.patch_max;sequences.append([(x0,y0),(x1,y0),(x1,y1),(x0,y1)])
    for sequence in sequences:
        pos=[context.region.view2d.view_to_region(*uv,clip=False) for uv in sequence]
        if len(pos)>1:
            for x,y in zip(pos,pos[1:]+pos[:1]):lines.extend([x,y])
    shader=gpu.shader.from_builtin('UNIFORM_COLOR');shader.bind();shader.uniform_float('color',(.1,1,.7,1));batch_for_shader(shader,'LINES',{'pos':lines}).draw(shader)
    return True


_classes=(OcularPhotos,OCULAR_OT_sclera_load,OCULAR_OT_sclera_reference,OCULAR_OT_sclera_view,OCULAR_OT_sclera_point,OCULAR_OT_clear_border,OCULAR_OT_extended_photo,OCULAR_OT_extension_guide,OCULAR_OT_update_photo_colors)
def register():
    for c in _classes:bpy.utils.register_class(c)
    bpy.types.Scene.ocular_photos=bpy.props.PointerProperty(type=OcularPhotos)
def unregister():
    del bpy.types.Scene.ocular_photos
    for c in reversed(_classes):bpy.utils.unregister_class(c)
