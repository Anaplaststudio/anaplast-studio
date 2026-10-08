"""One eye design, with ocular and standalone iris-button manufacturing outputs."""
import hashlib,json
from pathlib import Path
import bpy
from bpy.app.handlers import persistent
from bpy.props import EnumProperty,StringProperty
from . import eye_photo,apparent_pupil

BAND=('apparent_pupil','pupil_min','pupil_max','pupil_darkness','pupil_outer_darkness','pupil_outer_fade')
COMMON=('pupil_size_mode','pupil_diameter','pupil_depth','pupil_underiris','pupil_rim_height','pupil_rim_width','pupil_rim_blend','relief_height','relief_spacing','relief_smooth','relief_invert','texture_size')

def initialize(scene):
    p=scene.ocular_production;b=scene.iris_button;w=scene.ocular_steps
    if p.eye_design_shared:return
    # Only migrate a standalone design when there is no existing full ocular.
    if (b.color_obj or b.photo.photo) and not (p.color_obj or scene.anaplast.ocular_obj):
        for key in COMMON:setattr(p,key,getattr(b.photo,key))
        for key in BAND:setattr(p,key,getattr(b,key))
        if not p.photo:
            for key in eye_photo.PHOTO_FIELDS:setattr(p,key,getattr(b.photo,key))
        w.iris_size_mode='MANUAL';w.iris_manual_diameter=b.diameter
        w.cornea_height=min(b.dome_height,4.);p.clear_thickness=b.cover
    p.eye_fit_scan=bool(p.color_obj or scene.anaplast.ocular_obj or scene.anaplast.ocular_scan)
    p.eye_design_shared=True

@persistent
def migrate(_):
    for scene in getattr(bpy.data,'scenes',()):
        if hasattr(scene,'iris_button'):initialize(scene)

def migrate_after_enable():
    migrate(None)
    return None

def diameter(scene):
    w=scene.ocular_steps;p=scene.ocular_production
    if not p.eye_fit_scan or w.iris_size_mode=='MANUAL':return w.iris_manual_diameter
    if not w.measured_iris_diameter and not w.iris_fit:
        raise RuntimeError('Fit the scan iris first, or choose Set diameter')
    value=w.measured_iris_diameter or w.iris_diameter
    if value<=0:raise RuntimeError('Fit the scan iris first, or choose Set diameter')
    return value

class ButtonSettings:
    """Share appearance without overwriting retained legacy button settings."""
    def __init__(self,scene):object.__setattr__(self,'scene',scene)
    def __getattr__(self,key):
        s=self.scene;p=s.ocular_production
        if key=='photo':return p
        if key in BAND:return getattr(p,key)
        if key=='diameter':return diameter(s)
        if key=='dome_height':return s.ocular_steps.cornea_height
        if key=='cover':return p.clear_thickness
        return getattr(s.iris_button,key)
    def __setattr__(self,key,value):setattr(self.scene.iris_button,key,value)

def signature(scene):
    p=scene.ocular_production;w=scene.ocular_steps
    data={key:getattr(p,key) for key in COMMON+BAND}
    source=eye_photo.source(scene)
    data['photo']={key:(value.name if hasattr(value,'name') else list(value) if not isinstance(value,(str,float,int,bool,type(None))) else value) for key,value in source.items()}
    data['dimensions']=[w.iris_size_mode,w.iris_manual_diameter,w.cornea_height,p.clear_thickness]
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()

class EYE_OT_preview(bpy.types.Operator):
    bl_idname='anaplast.eye_design_preview';bl_label='Update eye preview';bl_options={'REGISTER','UNDO'}
    def execute(self,c):
        initialize(c.scene)
        if c.scene.ocular_production.eye_fit_scan:return bpy.ops.anaplast.ocular_generate()
        return bpy.ops.anaplast.iris_button_build()

class EYE_OT_export(bpy.types.Operator):
    bl_idname='anaplast.eye_design_export';bl_label='Export eye design'
    target:EnumProperty(items=[('OCULAR','Ocular',''),('BUTTON','Iris button','')])
    directory:StringProperty(subtype='DIR_PATH')
    def invoke(self,c,event):c.window_manager.fileselect_add(self);return {'RUNNING_MODAL'}
    def execute(self,c):
        try:
            initialize(c.scene);p=c.scene.ocular_production
            if self.target=='BUTTON':
                from . import iris_button
                iris_button.build(c)
                root=Path(self.directory);root.mkdir(parents=True,exist_ok=True)
                path=root/'Iris_button.zip';i=1
                while path.exists():path=root/f'Iris_button_{i:03d}.zip';i+=1
                iris_button.export_button(c.scene,str(path))
            else:
                if not p.color_obj or p.color_obj.get('eye_design_signature')!=signature(c.scene):
                    raise RuntimeError('Update eye preview with Fit to scan enabled before exporting the ocular')
                if p.export_variations:
                    from .ocular_variations import export_variations
                    path=export_variations(c,self.directory)
                else:
                    from .ocular_production import export
                    path=export(c,self.directory)
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        self.report({'INFO'},'Eye design exported');return {'FINISHED'}

def draw(layout,c):
    from . import ocular_simple,ocular_photos
    s=c.scene;p=s.ocular_production;w=s.ocular_steps;b=s.iris_button
    box=layout.box();box.label(text='1. Photograph and marks');eye_photo.draw(box,c)
    box=layout.box();box.label(text='2. Iris, pupil and cornea')
    if p.eye_fit_scan:
        box.prop(w,'iris_size_mode')
    if not p.eye_fit_scan or w.iris_size_mode=='MANUAL':box.prop(w,'iris_manual_diameter')
    else:box.label(text=f'Marked iris: {w.measured_iris_diameter or w.iris_diameter:.2f} mm')
    apparent_pupil.controls(box,p)
    if not p.apparent_pupil:
        box.prop(p,'pupil_size_mode')
        if p.pupil_size_mode=='MANUAL':box.prop(p,'pupil_diameter')
    box.prop(w,'cornea_height',text='Corneal rise (mm)')
    box.prop(p,'clear_thickness');box.prop(p,'relief_height')
    box.prop(p,'eye_details',text='Pupil chamber and iris detail',icon='TRIA_DOWN' if p.eye_details else 'TRIA_RIGHT',emboss=False)
    if p.eye_details:
        for key in ('pupil_depth','pupil_underiris','pupil_rim_height','pupil_rim_width','pupil_rim_blend','relief_spacing','relief_smooth','relief_invert','texture_size'):box.prop(p,key)
    box=layout.box();box.prop(p,'eye_fit_scan',text='3. Fit to scan — full ocular')
    if p.eye_fit_scan:
        ocular_simple.draw_marks(box,c)
        ocular_photos.draw_extension(box,s.ocular_photos)
        box.prop(w,'cornea_size_mode')
        if w.cornea_size_mode=='MANUAL':box.prop(w,'cornea_manual_diameter')
        else:box.prop(w,'cornea_width')
        box.prop(s.anaplast,'ocular_extension',text='Beyond opening (mm)')
        box.prop(p,'limbus_width');box.prop(p,'limbus_photo_detail')
        box.prop(w,'show_advanced',text='Ocular fitting details')
        if w.show_advanced:
            for key in ('thickness','thickness_basis','inner_flatten','edge_radius','back_blend_depth'):box.prop(w,key)
            box.prop(p,'flat_iris');box.prop(p,'iris_depth')
            if p.flat_iris:box.prop(p,'iris_blend_width')
            row=box.row(align=True)
            row.operator('anaplast.ocular_steps',text='Use sculpt mask').action='CAPTURE'
            row.operator('anaplast.ocular_steps',text='Fit ball').action='BALL'
            if s.anaplast.ball_obj:
                box.prop(s.anaplast,'ball_diameter');box.prop(s.anaplast,'ball_sink')
            box.operator('anaplast.fit_sphere',text='Fit ball to selected surface')
            box.operator('anaplast.ball_new',text='Add fitting ball')
        if p.orbital_preview:box.prop(p.orbital_preview,'hide_viewport',text='Hide scan preview')
        if w.report:box.label(text=w.report)
    box=layout.box();box.prop(p,'eye_button_details',text='Iris button — peg and backing',icon='TRIA_DOWN' if p.eye_button_details else 'TRIA_RIGHT',emboss=False)
    if p.eye_button_details:
        for key in ('peg_diameter','peg_height','peg_taper','back','backing'):box.prop(b,key)
        if b.back=='CONCAVE':box.prop(b,'back_sag')
    box=layout.box();box.label(text='4. Preview and export')
    box.operator('anaplast.eye_design_preview',icon='MESH_UVSPHERE')
    if p.eye_fit_scan and p.color_obj and p.clear_obj:
        row=box.row(align=True)
        row.operator('anaplast.ocular_display',text='Photo colors').mode='COLOR'
        row.operator('anaplast.ocular_display',text='Clear cornea').mode='CLEAR'
        from .ocular_view import draw as inspection
        inspection(box,c)
    row=box.row(align=True)
    row.operator('anaplast.eye_design_export',text='Export ocular',icon='EXPORT').target='OCULAR'
    row.operator('anaplast.eye_design_export',text='Export iris button',icon='EXPORT').target='BUTTON'
    if p.eye_fit_scan:
        box.prop(p,'export_variations')
        if p.export_variations:box.prop(p,'variation_region');box.prop(p,'variation_strength')

_classes=(EYE_OT_preview,EYE_OT_export)
def register():
    for cls in _classes:bpy.utils.register_class(cls)
    bpy.app.handlers.load_post.append(migrate)
    if hasattr(bpy.data,'scenes'):migrate(None)
    else:bpy.app.timers.register(migrate_after_enable,first_interval=.1)
def unregister():
    if bpy.app.timers.is_registered(migrate_after_enable):bpy.app.timers.unregister(migrate_after_enable)
    if migrate in bpy.app.handlers.load_post:bpy.app.handlers.load_post.remove(migrate)
    for cls in reversed(_classes):bpy.utils.unregister_class(cls)
