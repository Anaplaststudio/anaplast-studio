import bpy
from . import silicones
from bpy.props import (StringProperty, EnumProperty, PointerProperty,
                       FloatProperty, FloatVectorProperty, IntProperty, BoolProperty)

PROSTHESIS_TYPES = [
    ("AURICULAR", "Auricular", ""),
    ("NASAL", "Nasal", ""),
    ("ORBITAL", "Orbital", ""),
    ("OCULAR", "Ocular", ""),
    ("MIDFACIAL", "Midfacial", ""),
]

SIDES = [("LEFT", "Left", ""), ("RIGHT", "Right", ""), ("MIDLINE", "Midline", "")]


def _live(self, context):
    try:
        from .ops.xyz_live import sync, update_ring
        sync(context)
        update_ring(context)
    except Exception:
        pass


def _ball(self, context):
    try:
        from .ops.sphere import ball_update
        ball_update(self, context)
    except Exception:
        pass


def _photo(self, context):
    try:
        from .ops.photo import apply_photo_settings
        apply_photo_settings(context)
    except Exception:
        pass


def _mres(self, context):
    try:
        from .ops.multires import level_update
        level_update(self, context)
    except Exception:
        pass


def _light(self, context):
    try:
        from .ops.overlay import apply_light
        apply_light(context)
    except Exception:
        pass


def _section(self, context):
    try:
        from .ops.section import apply_cutaway
        apply_cutaway(context)
    except Exception:
        pass


def _is_mesh(self, obj):
    from .ops.viewport_tools import surface_object
    from .ops.scene_helpers import is_construction
    return surface_object(obj) and not is_construction(obj,self.id_data)


def _case_type(self,context):
    if self.prosthesis_type in {'NASAL','MIDFACIAL'}:self.side='MIDLINE'


@bpy.app.handlers.persistent
def _load_case_types(_):
    for scene in bpy.data.scenes:
        if hasattr(scene,'anaplast'):_case_type(scene.anaplast,None)
        if scene.get('b4d_grid_version',0)<2:
            if scene==bpy.context.scene and hasattr(scene,'anaplast') and scene.anaplast.grid_front:
                for screen in bpy.data.screens:
                    for area in screen.areas:
                        if area.type=='VIEW_3D':area.spaces.active.shading.show_xray=False
            scene['b4d_grid_version']=2


_enum_strings={}
_enum_object_ids={}
def _enum_text(value):
    return _enum_strings.setdefault(value,value)
def _enum_number(obj):
    key=obj.session_uid if hasattr(obj,'session_uid') else obj.as_pointer()
    if key not in _enum_object_ids:_enum_object_ids[key]=len(_enum_object_ids)+1
    return _enum_object_ids[key]

_crop_items=[]

def _helpers_visibility(self,context):
    from .ops.scene_helpers import organize
    organize(self.id_data)

def _landmark_visibility(self,context):
    from .ops.scene_helpers import marker_visibility
    marker_visibility(self.id_data)

def _crop_choices(self,context):
    global _crop_items
    from .ops.viewport_tools import surface_object
    from .ops.scene_helpers import is_construction
    _crop_items=[('NONE','Choose mesh','',0)]+[(_enum_text(o.name),_enum_text(o.name),'',_enum_number(o)) for i,o in enumerate(sorted(self.id_data.objects,key=lambda o:o.name)) if surface_object(o) and not is_construction(o,self.id_data)]
    return _crop_items

def _crop_get(self):
    return next((i for key,label,tip,i in _crop_choices(self,None) if self.crop_target and key==self.crop_target.name),0)

def _crop_set(self,value):
    key=next((key for key,label,tip,i in _crop_choices(self,None) if i==value),'NONE')
    self.crop_target=self.id_data.objects.get(key) if key!='NONE' else None


_working_items=[]
def _working_choices(self,context):
    global _working_items
    from .ops.viewport_tools import surface_object
    from .ops.scene_helpers import is_construction
    from .ops.naming import surface_label
    _working_items=[('NONE','Choose surface','',0)]+[(_enum_text(o.name),_enum_text(surface_label(o,self)),'',_enum_number(o)) for i,o in enumerate(sorted(self.id_data.objects,key=lambda o:o.name)) if surface_object(o) and not is_construction(o,self.id_data)]
    return _working_items

def _working_get(self):
    obj=self.sculpt_surface or self.source_prosthesis or self.prosthesis_obj
    return next((i for name,label,tip,i in _working_choices(self,None) if obj and name==obj.name),0)

def _working_set(self,value):
    name=next((n for n,label,tip,i in _working_choices(self,None) if i==value),'NONE')
    self.sculpt_surface=self.id_data.objects.get(name) if name!='NONE' else None
    self.sculpt_target='OBJECT'


def _working_update(self,context):
    # A viewport brush continues acting on the active Sculpt Mode object.
    # Merely changing the dropdown used to leave it on the previous surface.
    if context and context.scene==self.id_data and context.mode=='SCULPT':
        from .ops.sculpt_surface import activate
        obj=self.sculpt_surface
        if obj is not None and obj!=context.active_object and obj.name in context.view_layer.objects:
            activate(context,obj)
            bpy.ops.object.mode_set(mode='SCULPT')


class ANAPLAST_CaseProps(bpy.types.PropertyGroup):
    ocular_scan: PointerProperty(name="Eye scan",type=bpy.types.Object,poll=_is_mesh,description="Defaults to Mirror_FaceScan")
    ocular_trace: PointerProperty(name="Aperture trace",type=bpy.types.Object)
    ocular_extension: FloatProperty(name="Beyond lids (mm)",default=3.,min=0.,soft_max=6.,description="Outward offset of the traced aperture in the tracing view")
    ocular_thickness: FloatProperty(name="Shell thickness (mm)",default=4.,min=.1,soft_max=5.,description="Nominal inner offset perpendicular to the generated front; an editable starting value, not impression fit")
    ocular_resolution: FloatProperty(name="Surface spacing (mm)",default=.25,min=.1,max=1.)
    ocular_detail: FloatProperty(name="Retain scan detail",default=1.,min=0.,max=1.,subtype='FACTOR')
    ocular_cornea_diameter: FloatProperty(name="Corneal diameter (mm)",default=10.,min=1.,soft_max=15.)
    ocular_cornea_projection: FloatProperty(name="Corneal projection (mm)",default=0.,min=0.,soft_max=3.,description="Modelled definition missing from the scan; zero adds no projection")
    ocular_iris_diameter: FloatProperty(name="Iris guide diameter (mm)",default=10.,min=1.,soft_max=15.,description="Visual iris region on the front, centered under the modelled cornea; no engraved edge")
    ocular_report: StringProperty(name="Ocular result")
    # --- case ---
    case_id: StringProperty(name="Case ID", default="CASE")
    patient_first: StringProperty(name="First name", default="")
    patient_last: StringProperty(name="Last name", default="")
    case_date: StringProperty(name="Date", default="")
    mrn: StringProperty(name="MRN", default="", description="Medical record number — used in file names when set")
    prosthesis_type: EnumProperty(name="Type", items=PROSTHESIS_TYPES, default="AURICULAR",update=_case_type)
    side: EnumProperty(name="Side", items=SIDES, default="RIGHT")
    prosthesis_obj: PointerProperty(name="Sculpt", type=bpy.types.Object, poll=_is_mesh)
    face_scan_obj: PointerProperty(name="Scan", type=bpy.types.Object, poll=_is_mesh)
    cast_obj: PointerProperty(name="Cast / defect scan", type=bpy.types.Object, poll=_is_mesh)
    cbct_obj: PointerProperty(name="CBCT bone", type=bpy.types.Object, poll=_is_mesh)
    align_moving: PointerProperty(name="Moving", type=bpy.types.Object, poll=_is_mesh)
    align_target: PointerProperty(name="Target", type=bpy.types.Object, poll=_is_mesh)
    last_rms: FloatProperty(name="Last ICP RMS", default=0.0)
    icp_region: FloatProperty(name="Favoured radius (mm)", default=10.0, min=5.0, soft_max=150.0,
                              description="How far around the clicked spot the fit is favoured")
    favour_centre: FloatVectorProperty(name="Favour centre", size=3, default=(0.0, 0.0, 0.0), subtype='TRANSLATION')
    favour_on: BoolProperty(name="Favour a spot", default=False)
    keep_ortho: BoolProperty(name="Keep orthographic view", default=True,
                             description="Framing the case does not switch the viewport to perspective")
    deviation_range: FloatProperty(name="Red at (mm)", default=2.0, min=0.05, soft_max=10.0)
    exploded: BoolProperty(default=False)
    explode_factor: FloatProperty(name="Explode", default=0.6, min=0.1, soft_max=3.0)
    oriented: BoolProperty(name="Oriented", default=False)
    symmetry_rms: FloatProperty(name="Symmetry RMS", default=0.0)
    scan_thickness: FloatProperty(name="Scan thickness for booleans (mm)", default=15.0, min=1.0, soft_max=60.0,
                                  description="Open scan surfaces are thickened inward by this much (on a temporary copy) so subtract/intersect operations have a solid to work against")
    simplify_prosthesis: BoolProperty(name="Simplify sculpt for mold booleans", default=False,
                                      description="OFF: the mold cavity keeps every face of the sculpt (big ones use the Fast solver). ON: a decimated copy is used for the boolean only")
    boolean_solver: EnumProperty(name="Boolean solver", items=[("AUTO", "Auto", "Manifold solver on Blender 5.x (fast, robust for closed solids); Exact on 4.x"),
                                                               ("MANIFOLD", "Manifold", "Blender 5.x only; needs closed inputs"),
                                                               ("EXACT", "Exact", "Robust but crashes on very large inputs"),
                                                               ("FAST", "Fast / Float", "Handles big meshes; less tolerant of touching surfaces")], default="AUTO")
    boolean_max_faces: IntProperty(name="Max faces for booleans", default=150000, min=20000, soft_max=600000,
                                   description="The scan is clipped to the sculpt area and decimated to this many faces before any subtract — keeps Blender's exact boolean from choking on multi-million-face scans")
    scan_flip: BoolProperty(name="Flip scan inside/outside", default=False,
                            description="Tick if the thickened scan grows toward the camera instead of into the head")

    # --- mold block (Tool 2). Placeholders pending bench test B1. ---
    block_shape: EnumProperty(name="Shape", items=[("BOX", "Box", ""), ("CYLINDER", "Cylinder", "")], default="BOX")
    block_offset: FloatProperty(name="Offset (mm)", default=10.0, min=1.0, soft_max=40.0,
                                description="Placeholder pending bench test B1")
    include_cast: BoolProperty(name="Seat on cast", default=True,
                               description="Subtract the cast so the mold seats on it at the margin")

    # --- keys (Tool 4). Placeholders pending bench test B2. ---
    key_diameter: FloatProperty(name="Key Ø (mm)", default=4.0, min=1.0, soft_max=12.0,
                                description="Placeholder pending bench test B2")
    key_count: IntProperty(name="Keys per plane", default=3, min=2, max=8)
    key_clearance: FloatProperty(name="Key clearance (mm)", default=0.1, min=0.0, soft_max=0.5,
                                 description="Socket oversize; placeholder pending bench test B2")

    # --- pathway A: prototype shell. Placeholder thickness pending bench test. ---
    shell_thickness: FloatProperty(name="Shell wall (mm)", default=1.0, min=0.3, soft_max=6.0)
    shell_trim: EnumProperty(name="Subtract", items=[("CAST", "Cast / stone", ""), ("FACE", "Scan", "")], default="CAST",
                             description="Which tissue surface is boolean-subtracted from the sculpt")
    drain_diameter: FloatProperty(name="Drain hole Ø (mm)", default=3.0, min=1.0, soft_max=8.0)
    build_method: EnumProperty(name="Build", items=[("VOXEL", "Extract → Voxel remesh → Boolean", "ZBrush-style: extract a shell toward the tissue, voxel-remesh both parts, subtract"),
                                                   ("BOOLEAN", "Boolean (prism)", "Close the surface into a prism along one direction and subtract the tissue prism"),
                                                   ("GEOMETRIC", "Geometric", "Project the inner surface directly; no solver")], default="VOXEL")
    despeckle_mm: FloatProperty(name="Despeckle spikes above (mm)", default=0.3, min=0.0, soft_max=2.0,
                                description="Move vertices differing from a smoothed surface by more than this threshold; can also alter real anatomical features")
    despeckle_enabled: BoolProperty(name="Despeckle during prototype build",default=False)
    voxel_size: FloatProperty(name="Detail (voxel mm)", default=0.15, min=0.05, soft_max=0.5,
                              description="Prototype voxel sampling size. Smaller values retain finer detail at greater memory cost; not a guaranteed accuracy tolerance")
    extrude_dir: EnumProperty(name="Extrude into head", items=[("AUTO", "Auto (average normal)", ""), ("NEG_X", "−X (patient's left)", ""), ("POS_X", "+X (patient's right)", ""),
                                                             ("NEG_Y", "−Y (posterior)", ""), ("NEG_Z", "−Z (inferior)", ""), ("POS_Z", "+Z (superior)", "")], default="AUTO")
    feather_width: FloatProperty(name="Feather width (mm)", default=10.0, min=0.0, soft_max=15.0,
                                 description="Over this distance in from the edge the wall tapers from Edge thickness up to Shell wall")
    edge_thickness: FloatProperty(name="Edge thickness (mm)", default=0.15, min=0.05, soft_max=1.0,
                                  description="Wall thickness at the very margin. Anything thinner than the voxel size (0.15) cannot be represented")
    margin_band: FloatProperty(name="Margin adaptation band (mm)", default=10.0, min=0.0, soft_max=20.0,
                               description="Within this distance of the open edge, a sculpt surface floating above the tissue is pulled down to skin clearance, blending smoothly inward")
    adapt_margins: BoolProperty(name="Adapt open margins to tissue", default=True,
                                description="Applied automatically when building the cap / fitted sculpt from an open surface")
    dynamesh_on_import: BoolProperty(name="Legacy remesh on import", default=False,options={'HIDDEN'},description="Retained for old cases; imported scan geometry is now preserved")
    sculpt_remesh_after_crop: BoolProperty(name="Prepare solid Sculpt after cropping",default=False,options={'HIDDEN'})
    sculpt_voxel: FloatProperty(name="Sculpt voxel size (mm)",default=.15,min=.05,soft_max=.5)
    sculpt_backing: FloatProperty(name="Temporary backing (mm)",default=.6,min=.3,soft_max=5.,description="Thickness behind an open cropped region for solid sculpting; at least three voxels. This is not the final fitting surface")
    sculpt_reproject: BoolProperty(name="Preserve surface detail",default=True,description="Bring outer vertices toward the cropped reference; fine detail remains limited by mesh density")
    sculpt_remesh_report: StringProperty()
    crop_mesh: EnumProperty(name="Crop object",items=_crop_choices,get=_crop_get,set=_crop_set)
    crop_shape: EnumProperty(name="Shape",items=[('LASSO','Lasso','Draw a freehand outline'),('OVAL','Oval','Drag an oval around the region'),('RECTANGLE','Rectangle','Drag a rectangular region')],default='LASSO')
    crop_finish: EnumProperty(name="Mesh edge",items=[('OPEN','Keep open','Leave cut boundaries open'),('CLOSE','Close mesh','Fill boundary loops without remeshing the surface')],default='OPEN')
    crop_create_sculpt: BoolProperty(name="Create Sculpt region",default=False,description="Create a separate cropped Sculpt and leave the source mesh unchanged")
    crop_preview_obj: PointerProperty(type=bpy.types.Object,options={'HIDDEN'})
    crop_hide_helpers: BoolProperty(name="Hide construction objects in list",default=True)
    show_construction_helpers: BoolProperty(name="Construction helpers",default=False,update=_helpers_visibility)
    show_landmark_markers: BoolProperty(name="Landmark markers",default=False,update=_landmark_visibility)
    landmarks_placing: BoolProperty(default=False,options={'HIDDEN','SKIP_SAVE'})
    solid_on_import: BoolProperty(name="Make solid twin on import", default=False,
                                  description="Also create a hidden, full-resolution closed copy (<name>_Solid) of each imported open scan for manual booleans. Doubles memory; multi-million-face scans get huge")
    shell_clearance: FloatProperty(name="Skin clearance (mm)", default=0.0, min=0.0, soft_max=3.0,
                                   description="Any part of the sculpt surface lying below the tissue is lifted to this height above it before shelling — placeholder pending bench test")

    # --- skin texture. All placeholders pending bench test on printed molds. ---
    tex_subdiv: IntProperty(name="Densify (subdiv levels)", default=1, min=0, max=3,
                            description="Simple subdivision before displacing; 0 = use mesh as is")
    tex_gain: FloatProperty(name="Detail gain", default=1.0, min=0.0, soft_max=4.0,
                            description="1.0 doubles the scan's fine detail; 2.0 triples it")
    tex_radius: IntProperty(name="Detail radius (smooth iters)", default=6, min=1, max=40,
                            description="Larger = broader features count as detail")
    tex_pore_size: FloatProperty(name="Pore spacing (mm)", default=0.5, min=0.05, soft_max=3.0)
    tex_pore_depth: FloatProperty(name="Pore depth (mm)", default=0.06, min=0.0, soft_max=0.5)
    tex_crease_size: FloatProperty(name="Crease scale (mm)", default=4.0, min=0.2, soft_max=20.0)
    tex_crease_depth: FloatProperty(name="Crease depth (mm)", default=0.15, min=0.0, soft_max=1.0)
    skin_scale: FloatProperty(name="Skin scale", default=1.0, min=0.3, soft_max=3.0, description="1.0 ≈ cheek; ~0.7 for ear/nose tip; ~1.5 for older / coarser skin")
    skin_depth: FloatProperty(name="Skin depth (mm)", default=0.08, min=0.01, soft_max=0.4, description="Relief height of the micro-pattern; printed molds soften ~30–50%, so overshoot")
    skin_line_angle: FloatProperty(name="Crease line angle (°)", default=20.0, min=-90.0, max=90.0, description="Direction of the fine crease lines (follow the local Langer's lines)")
    tex_alpha_tiles: IntProperty(name="Alpha tiles across", default=6, min=1, max=64)
    tex_alpha_depth: FloatProperty(name="Alpha depth (mm)", default=0.2, min=0.0, soft_max=2.0)

    # --- statistical shape model ---
    ssm_dataset_dir: StringProperty(name="Dataset folder", subtype='DIR_PATH', default="")
    mpfb_gender: EnumProperty(name="Gender", items=[("ANY", "Mixed", ""), ("F", "Female", ""), ("M", "Male", "")], default="ANY")
    mpfb_age: EnumProperty(name="Age", items=[("ANY", "Any adult", ""), ("YOUNG", "Young (20–35)", ""),
                                              ("MID", "Middle (35–55)", ""), ("OLD", "Older (55+)", "")], default="ANY")
    mpfb_subdiv: IntProperty(name="Resolution (subdivisions)", default=2, min=0, max=3,
                             description="MakeHuman's base mesh is coarse (~3k vertices per head). Each subdivision quadruples it; 2 gives ~50k faces for a head, smooth enough for prosthetics. Applied identically to every head so correspondence is kept")
    mpfb_keep: EnumProperty(name="Keep", items=[("HEAD", "Head only", "Everything above the neck"),
                                                ("FACE", "Face only", "Front of the head — the region prosthetics come from"),
                                                ("BODY", "Whole body", "")], default="HEAD")
    ssm_model_path: StringProperty(name="Model file (.npz)", subtype='FILE_PATH', default="",
                                   description="A trained shape model written by Train Model (.npz) — not a scan or OBJ")
    ssm_region: StringProperty(name="Region (internal)", default="")
    ssm_keep_variance: FloatProperty(name="Keep variance (%)", default=98.0, min=80.0, max=100.0)
    ssm_iterations: IntProperty(name="Fit iterations", default=15, min=1, max=100)
    ssm_max_dist: FloatProperty(name="Max pair distance (mm)", default=6.0, min=0.5, soft_max=30.0,
                                description="Model vertices farther than this from the scan are treated as defect and filled from statistics")
    ssm_regularisation: FloatProperty(name="Regularisation", default=20.0, min=0.0, soft_max=50.0,
                                      description="Higher = stays closer to the mean shape; lower = follows the scan more freely")

    # --- surgical guide. Placeholders — set from the implant system in use. ---
    implant_diameter: FloatProperty(name="Implant Ø (mm)", default=3.75, min=1.5, soft_max=6.0)
    implant_length: FloatProperty(name="Implant length (mm)", default=4.0, min=2.0, soft_max=15.0)
    guide_support: EnumProperty(name="Guide sits on", items=[("TISSUE", "Skin / cast", ""), ("BONE", "Bone (CBCT)", "")], default="TISSUE")
    guide_thickness: FloatProperty(name="Guide thickness (mm)", default=3.0, min=1.0, soft_max=8.0)
    guide_radius: FloatProperty(name="Guide reach around implants (mm)", default=12.0, min=4.0, soft_max=40.0)
    sleeve_diameter: FloatProperty(name="Sleeve Ø (mm)", default=4.2, min=1.5, soft_max=8.0)

    # --- bar & retention ---
    retention: EnumProperty(name="Retention", items=[("BAR", "Bar + clips", ""), ("MAGNET", "Magnets", "")], default="BAR")
    bar_diameter: FloatProperty(name="Bar Ø (mm)", default=1.9, min=1.0, soft_max=4.0)
    bar_height: FloatProperty(name="Bar height above implant (mm)", default=4.0, min=1.0, soft_max=12.0)
    abutment_diameter: FloatProperty(name="Abutment Ø (mm)", default=4.0, min=2.0, soft_max=7.0)
    housing_length: FloatProperty(name="Clip housing length (mm)", default=5.0, min=2.0, soft_max=12.0)
    housing_wall: FloatProperty(name="Clip housing wall (mm)", default=1.0, min=0.3, soft_max=3.0)
    magnet_diameter: FloatProperty(name="Magnet Ø (mm)", default=5.0, min=2.0, soft_max=10.0)
    magnet_height: FloatProperty(name="Magnet height (mm)", default=2.0, min=0.5, soft_max=5.0)
    sub_outer_obj: PointerProperty(name="Outer sculpt",type=bpy.types.Object,poll=_is_mesh)
    sub_fit_obj: PointerProperty(name="Fitting scan",type=bpy.types.Object,poll=_is_mesh)
    sub_component_source: EnumProperty(name="Component positions",items=[('PAINT','Painted scan','One connected painted patch per component'),('MARKERS','Existing markers','Use manually positioned component markers')],default='PAINT')
    sub_mark_obj: PointerProperty(name="Painted fitting scan",type=bpy.types.Object)
    sub_mark_radius: FloatProperty(name="Paint radius (mm)",default=2.,min=.2,max=20.)
    sub_layout: EnumProperty(name="Substructure layout",items=[('CONNECTED','Connected plate','Fill between components while following the available space inside the sculpt'),('SEPARATE','Separate pieces','Divide the shared fitted plate midway between implant positions; footprint size only controls the outside extent'),('FULL','Whole available interior','Legacy full interior without component footprint limits')],default='CONNECTED')
    sub_width: FloatProperty(name="Footprint width (mm)",default=20.,min=3.,soft_max=50.)
    sub_length: FloatProperty(name="Footprint length (mm)",default=20.,min=3.,soft_max=50.)
    sub_height: FloatProperty(name="Maximum height (mm)",default=8.,min=.5,soft_max=30.,description="Maximum thickness above the relieved fitting surface; limited further by the silicone cover")
    sub_piece_gap: FloatProperty(name="Separate-piece gap (mm)",default=1.,min=.2,soft_max=5.)
    sub_ocular_relief: BoolProperty(name="Relief around ocular",default=False)
    sub_ocular_obj: PointerProperty(name="Ocular for relief",type=bpy.types.Object,poll=_is_mesh)
    sub_ocular_gap: FloatProperty(name="Ocular clearance (mm)",default=1.,min=0.,soft_max=5.)
    sub_relief: FloatProperty(name="Fitting relief (mm)",default=1.5,min=0.,soft_max=5.)
    sub_silicone: FloatProperty(name="Silicone cover (mm)",default=3.,min=0.,soft_max=8.,description="Minimum setback from the outer sculpt surface, including its edge")
    sub_resolution: FloatProperty(name="Detail (mm)",default=.2,min=.1,soft_max=.6)
    sub_smooth: FloatProperty(name="Substructure smoothing",default=.5,min=0.,max=1.,subtype='FACTOR')
    sub_pockets: BoolProperty(name="Space for manual components",default=True)
    sub_component_diameter: FloatProperty(name="Component space diameter (mm)",default=6.,min=1.,soft_max=15.,description="Generic space; enter your actual magnet housing / analog envelope")
    sub_component_depth: FloatProperty(name="Component space depth (mm)",default=4.,min=.5,soft_max=12.,description="Depth above the relieved fitting level at each marker")
    sub_component_clearance: FloatProperty(name="Component radial clearance (mm)",default=.2,min=0.,soft_max=1.)
    sub_report: StringProperty(name="Substructure result",default="")

    # --- streamlined prototype ---
    prototype_kind: EnumProperty(name="Prototype", items=[("HOLLOW", "Fitted, hollow", "Seats on the stone; thin walls; drain holes"),
                                                          ("FITTED", "Fitted, solid", "Seats on the stone; solid"),
                                                          ("CAP", "Thin cap", "Outer skin only, rests on the margin")], default="HOLLOW")
    source_prosthesis: PointerProperty(name="Sculpt", type=bpy.types.Object, poll=_is_mesh)
    photo_opacity: FloatProperty(name="Opacity", default=0.5, min=0.0, max=1.0, update=lambda self, ctx: _photo(self, ctx))
    photo_width: FloatProperty(name="Width (mm)", default=180.0, min=10.0, soft_max=600.0, update=lambda self, ctx: _photo(self, ctx))
    photo_tolerance: FloatProperty(name="Hide beyond (°)", default=2.0, min=0.5, max=45.0,
                                   description="The photo hides once the view is turned more than this from where it was placed")
    polyframe: BoolProperty(name="Polyframe", default=False)
    ball_obj: PointerProperty(name="Ball", type=bpy.types.Object, poll=_is_mesh)
    ball_diameter: FloatProperty(name="Diameter override (mm)", default=0.0, min=0.0, soft_max=40.0,
                                 description="0 = fitted diameter; e.g. 24 for an adult globe", update=lambda self, ctx: _ball(self, ctx))
    ball_sink: FloatProperty(name="Sink in (mm)", default=0.0, min=-10.0, soft_max=10.0,
                             description="Deeper along the dome axis (negative = out)", update=lambda self, ctx: _ball(self, ctx))
    lasso_smooth: FloatProperty(name="Lasso smoothing (px)", default=24.0, min=0.0, soft_max=80.0,
                                description="How much the lasso line is smoothed as you draw — 0 follows your hand exactly")
    m_skin: FloatProperty(name="Skin band (mm)", default=3.0, min=0.0, soft_max=20.0)
    m_land: FloatProperty(name="Land (mm)", default=15.0, min=3.0, soft_max=40.0)
    m_land_transition: FloatProperty(name="Transition width (mm)", default=14.0, min=1.0, soft_max=40.0,
                                    description="Distance over which the land eases away from the skin; limited to leave a settled outer rim")
    m_land_follow: FloatProperty(name="Follow skin height",default=1.,min=0.,max=1.,subtype='FACTOR',description="Keep broad changes in skin height around the rim; zero uses one level, one follows the smoothed skin boundary")
    m_skin_smoothing: FloatProperty(name="Skin-band smoothing", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
                                   description="Optional smoothing of the shared skin band on both halves; zero leaves the scan unsmoothed")
    m_fit_smoothing: FloatProperty(name="Base fitting smoothing", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
                                  description="Optional smoothing beneath the sculpt, on a copy of the scan; the sculpt cavity is never smoothed")
    m_base_depth: FloatProperty(name="Base depth (mm)", default=12.0, min=3.0, soft_max=60.0)
    m_ocular_impression: BoolProperty(name="Include ocular impression", default=False, description="Form an exact seat for the ocular front and clear cornea at the final gaze, using the cut Sculpt")
    m_cap_height: FloatProperty(name="Cap height (mm)", default=10.0, min=3.0, soft_max=60.0)
    m_key_type: EnumProperty(name="Key type", items=[("CAPSULE", "Round keys", "Rounded pegs on the land with sockets in the cap"),
                                                     ("WEDGE", "Bullet wedge", "Rounded inner nose outside the ring, widening toward the rim; matching groove in the base"),
                                                     ("NONE", "None", "")], default="CAPSULE")
    m_key_count: IntProperty(name="Keys", default=4, min=2, max=8)
    m_bolts: IntProperty(name="Clamp bolts", default=0, min=0, max=8, description="Through-holes for clamping bolts, on the land")
    m_bolt_diameter: FloatProperty(name="Bolt hole Ø (mm)", default=4.5, min=2.0, soft_max=10.0)
    m_bolt_sleeve_length: EnumProperty(name="Bolt sleeve", items=[('SHORT','Short sleeve','Flat seat just above the local shell backing'),('FULL','Full-depth sleeve','Retain the original mold block depth for the sleeve; seat remains clear of the shell')], default='FULL')
    m_bolt_sleeve_wall: FloatProperty(name="Sleeve wall (mm)", default=2.0, min=0.5, soft_max=8.0, description="Radial sleeve thickness measured outward from the bolt hole; outer diameter is hole diameter plus twice this value")
    m_bolt_seat_raise: FloatProperty(name="Seat height above shell (mm)", default=1.0, min=0.1, soft_max=5.0, description="Flat seat perpendicular to the bolt. Short sleeve: height above the highest shell surface under the sleeve. Full-depth: minimum clearance above that surface")
    m_base_type: EnumProperty(name="Mold body", items=[("FULL", "Full", "Solid mold halves"), ("SHELL", "Shell", "Thicken each inner mold surface outward, without enclosing walls or a flat base")], default="FULL")
    wedge_body: EnumProperty(name="Three-piece body", items=[('FULL','Full','Solid base, wedge and cap'),('SHELL','Shell','Open backing on base and cap; wedge hollowed from the outside rim')], default='FULL')
    wedge_base_wall: FloatProperty(name="Base wall (mm)", default=4.0, min=1.0, soft_max=12.0, description="Nominal material behind the fitting surface and land")
    wedge_wall: FloatProperty(name="Wedge wall (mm)", default=3.0, min=1.0, soft_max=12.0, description="Nominal material against the ear and both mating faces; opens at the outside mold edge")
    wedge_cap_wall: FloatProperty(name="Cap wall (mm)", default=4.0, min=1.0, soft_max=12.0, description="Nominal material behind the cavity and mating land")
    m_shell_wall: FloatProperty(name="Shell wall (mm)", description="Thickness along the local inner-surface normals, away from the cavity; voxel resolution and tight curves affect the final thickness", default=4.0, min=2.0, soft_max=12.0)
    m_breakable_cap: BoolProperty(name="Breakable cap", default=False, description="Thin cap with a separate wall thickness; omits spillways, ring, bolts and pry slots; keeps the keys")
    m_cap_wall: FloatProperty(name="Cap wall (mm)", default=0.1, min=0.1, max=1.0, precision=3, description="Nominal cap thickness. 0.1 mm is experimental and below Formlabs published 0.2 mm supported-wall guidance")
    m_break_pattern: EnumProperty(name="Break pattern", items=[('RADIAL','Radial grooves','Outside grooves dividing the cap into sections'),('GRID','Crossed grooves','Outside grooves forming smaller panels'),('STARTERS','Edge starters','Short weak lines at the perimeter'),('NONE','None','Uniform thin cap')], default='RADIAL')
    m_break_count: IntProperty(name="Sections", default=4, min=2, max=12)
    m_break_width: FloatProperty(name="Groove width (mm)", default=1.2, min=0.3, soft_max=4.)
    m_break_remaining: FloatProperty(name="Wall at groove (mm)", default=0.05, min=0.04, max=0.9, precision=3, description="Nominal sealed wall at the groove; must be less than cap wall. Fine distance grid affects actual thickness")
    m_break_spacing: FloatProperty(name="Panel spacing (mm)", default=20., min=5., soft_max=40.)
    m_break_rotation: FloatProperty(name="Pattern rotation", default=0.7853981634, subtype='ANGLE')
    m_spout_diameter: FloatProperty(name="Spout Ø (mm)", default=6.0, min=2.0, soft_max=15.0)
    m_key_diameter: FloatProperty(name="Key Ø (mm)", default=5.0, min=2.0, soft_max=12.0)
    m_spillways: IntProperty(name="Spillways", default=4, min=0, max=12)
    m_spillway_width: FloatProperty(name="Spillway width (mm)", default=2.5, min=1.0, soft_max=6.0)
    m_ring: BoolProperty(name="Ring around the land", default=True)
    m_pry: IntProperty(name="Pry points", default=2, min=0, max=6)
    m_pry_width: FloatProperty(name="Pry width (mm)", default=15.0, min=2.0, soft_max=30.0, description="Width along the outer rim")
    m_pry_depth: FloatProperty(name="Pry depth (mm)", default=3.0, min=0.5, soft_max=10.0, description="Reach from the perimeter when there is no ring spillway; with a ring, reach is automatic and stops 1 mm before its outer edge")
    m_pry_height: FloatProperty(name="Pry opening (mm)", default=3.0, min=0.5, soft_max=10.0, description="Total height of the opening across the parting line")
    m_grid: FloatProperty(name="Surface grid (mm)", default=0.6, min=0.25, soft_max=2.0)
    m_depth_limit: FloatProperty(name="Ignore tissue deeper than (mm)", default=40.0, min=5.0, soft_max=80.0,
                                 description="Below the sculpt's lowest point. Rays that miss the ear or nose and hit the far side of the head are ignored, so a stray hit can't stretch the block")
    m_voxel: FloatProperty(name="Mold resolution (mm)", default=0.15, min=0.08, soft_max=0.5,
                           description="Repair resolution for open scans or intersecting sculpt cutters; smaller values retain more detail at higher cost. Intact source surfaces are retained. Shell backing normally uses twice this spacing")
    mold_steps: StringProperty(name="Mold steps", default="[]")
    mold_silicone: EnumProperty(name="Silicone", items=silicones.ITEMS, default='VST50')
    mold_product_density: FloatProperty(name="Density (g/mL)", min=0,soft_max=3,precision=3,
        get=silicones.density_get,set=silicones.density_set,
        description='Supplier or measured value, stored separately for this material; zero means unknown')
    mold_silicone_density: FloatProperty(name="Density (g/mL)", default=1.11, min=0.01, soft_max=3., precision=3,
        description="Mixed silicone mass per mL; changes weight estimates without rebuilding the mold")
    m_spout_outer: FloatProperty(name="Spout outer Ø (mm)", default=12.0, min=2.0, soft_max=30.0)
    wedge_selection: EnumProperty(name="Rear region", items=[('MARK','Marked area','Use the dedicated editable wedge marking copy'),('MASK','Sculpt mask','Use the mask painted on the source ear'),('TRACE','Traced boundary','Use the visible surface inside the traced boundary'),('AUTO','Undercut suggestion','Use the editable suggestion copy')], default='MARK')
    wedge_mark_obj: PointerProperty(name="Marked ear",type=bpy.types.Object)
    wedge_mark_radius: FloatProperty(name="Brush radius (mm)",default=2.5,min=.2,max=20.)
    wedge_trace: PointerProperty(name="Wedge boundary",type=bpy.types.Object)
    wedge_suggestion: PointerProperty(name="Suggestion copy",type=bpy.types.Object)
    wedge_draft: FloatProperty(name="Draft threshold",default=0.05235988,min=0.,max=0.52359878,subtype='ANGLE')
    wedge_margin: FloatProperty(name="Region margin (mm)",default=1.0,min=0.,soft_max=5.)
    wedge_auto_exit: BoolProperty(name="Automatic access direction",default=True)
    wedge_exit_angle: FloatProperty(name="Access direction",default=0.,subtype='ANGLE')
    wedge_access_width: FloatProperty(name="Access width (mm)",default=18.,min=10.,soft_max=35.)
    wedge_height_offset: FloatProperty(name="Rear surface overlap (mm)",default=1.,min=0.,max=5.,description="Carry the divider into the masked ear before subtracting the unchanged sculpt; the final cavity remains on the sculpt")
    wedge_top_smoothing: FloatProperty(name="Top smoothing (mm)",default=5.,min=0.,soft_max=12.,description="Smooth broad changes between rear sections on the dividing roof only; the original sculpt still defines the impression")
    wedge_transition: FloatProperty(name="Roof transition (mm)",default=14.,min=1.,soft_max=40.,description="Distance over which the under-helix angle gently approaches a roof parallel to the local base; measured starting thickness varies along the ear")
    wedge_land_height: FloatProperty(name="Wedge height on land (mm)",default=5.,min=3.,soft_max=12.)
    wedge_keys: BoolProperty(name="Keys at both interfaces",default=True)
    wedge_key_count: IntProperty(name="Keys per wedge interface",default=3,min=3,max=8,description="Matched keys on both the base/wedge and wedge/cap interfaces; placed clear of the ring, outlets and pry points")
    wedge_end1_mode: EnumProperty(name="End 1",items=[('DISTANCE','Distance','Offset along the rim from its shortest-path endpoint'),('POINT','Picked point','Use the chosen position on the outer rim')],default='DISTANCE')
    wedge_end1_extra: FloatProperty(name="End 1 extra (mm)",default=3.,min=-20.,max=20.,description="Distance along the rim: positive toward the cap side, negative toward the wedge; zero restores this shortest endpoint")
    wedge_end1_point: StringProperty(default='')
    wedge_end2_mode: EnumProperty(name="End 2",items=[('DISTANCE','Distance','Offset along the rim from its shortest-path endpoint'),('POINT','Picked point','Use the chosen position on the outer rim')],default='DISTANCE')
    wedge_end2_extra: FloatProperty(name="End 2 extra (mm)",default=3.,min=-20.,max=20.,description="Distance along the rim: positive toward the cap side, negative toward the wedge; zero restores this shortest endpoint")
    wedge_end2_point: StringProperty(default='')
    wedge_border_detail: FloatProperty(name="Border refinement (mm)",default=.8,min=0.,max=3.,precision=2,description="Replace paint-edge teeth and nicks with smoother lines and curves on the unchanged ear; zero uses the raw painted border")
    wedge_end_blend: FloatProperty(name="End transition (mm)",default=4.,min=1.,soft_max=10.,description="Ease the roof into the sloped terminal surfaces without a sharp crease")
    wedge_spillway: BoolProperty(name="Wedge/cap spillway",default=True)
    wedge_pry: BoolProperty(name="Pry points at all three interfaces",default=True)
    wedge_report: StringProperty(name="Wedge result",default='')
    m_key_flare: FloatProperty(name="Key flare", default=0.20943951, min=0.0, max=0.78539816, subtype='ANGLE', description="Widen the key root; all profiles narrow smoothly toward the tip")
    m_key_blend: FloatProperty(name="Root blend (mm)", default=0.7, min=0.1, soft_max=3.0, description="Additional width for the rounded transition into the land")
    m_wedge_inner: FloatProperty(name="Inner diameter (mm)", default=3.0, min=0.5, soft_max=15.0,
                                 description="Nominal diameter behind the rounded nose of the wedge groove")
    m_wedge_ring_gap: FloatProperty(name="Ring clearance (mm)", default=1.0, min=0.2, soft_max=5.0,
                                    description="Clear space from the outside of the ring spillway to the rounded inner tip")
    ocular_obj: PointerProperty(name="Ocular", type=bpy.types.Object, poll=_is_mesh)
    crop_target: PointerProperty(name="Crop", type=bpy.types.Object, poll=_is_mesh,
                                 description="The model you are cropping: the mirrored scan, a donor nose/ear, an SSM fit, or anything imported")

    # --- skin micro-relief on the built prototype ---
    skin_preset: EnumProperty(name="Skin", items=[("FINE", "Fine (eyelid / young)", ""), ("MEDIUM", "Medium (cheek / ear)", ""), ("COARSE", "Coarse (nose / older)", "")], default="MEDIUM")
    skin_strength: FloatProperty(name="Strength", default=1.0, min=0.0, soft_max=3.0)
    proto_exploded: BoolProperty(default=False)
    fit_range: FloatProperty(name="Fit colour range (mm)", default=1.0, min=0.1, soft_max=5.0,
                             description="Red at this gap or more between the fitting surface and the tissue")
    thick_min: FloatProperty(name="Red at or below (mm)", default=0.15, min=0.0, soft_max=3.0, description="Thickness shown as red")
    thick_max: FloatProperty(name="Green at (mm)", default=1.0, min=0.05, soft_max=6.0, description="Thickness shown as green; above this goes blue")
    section_on: BoolProperty(name="Section view", default=False)
    section_axis: EnumProperty(update=_section, name="Plane", items=[("SAG", "Sagittal (X)", ""), ("COR", "Coronal (Y)", ""), ("AXI", "Axial (Z)", "")], default="SAG")
    section_pos: FloatProperty(update=_section,name="Position (mm)", default=0.0, soft_min=-120.0, soft_max=120.0)
    section_flip: BoolProperty(update=_section, name="Flip side", default=False)
    slice_slab: FloatProperty(update=_section, name="Slab thickness (mm)", default=0.0, min=0.0, soft_max=40.0,
                              description="0 = cut away everything in front. Above 0 = show only a slice of this thickness, like a CT slab")
    show_texture: BoolProperty(name="Scan colour/texture", default=False)
    light_rot: FloatProperty(name="Light direction (°)", default=0.0, min=-180.0, max=180.0,
                             description="Rotate the studio light around the case — like moving a lamp around the bench",
                             update=lambda self, ctx: _light(self, ctx))
    light_tilt: FloatProperty(name="Light height (°)", default=0.0, min=-90.0, max=90.0,
                              update=lambda self, ctx: _light(self, ctx))
    light_zero: FloatProperty(name="Front calibration (°)", default=180.0, min=-180.0, max=180.0,
                              description="Rotation at which the light shines onto the patient's face. Blender's studio key light comes from −Y, the patient faces +Y, so 180° by default. Use 'This is front' to recalibrate")
    crosshair: BoolProperty(name="Crosshair guides", default=False)
    restore_detail: BoolProperty(name="Reproject surface detail", default=False,
                                 description="Project outer vertices onto the reference surface; cannot recover detail finer than the mesh can represent")
    lock_fitting: BoolProperty(name="Lock fitting surface while sculpting", default=True,
                               description="Masks the inner/fitting surface so Move and other brushes cannot disturb it")
    brush_strength: FloatProperty(name="Brush strength", default=0.2, min=0.01, max=1.0)
    brush_frontface: BoolProperty(name="Front faces only", default=True,
                                  description="Brushes affect only faces facing you — the back of a thin sheet is never touched")
    source_edge: FloatProperty(name="Target edge (mm)", default=0.25, min=0.05, soft_max=2.0,
                               description="Even out the cropped surface to roughly this edge length — uniform topology for sculpting. Blender's own voxel field shows metres, so 0.5 mm = 0.0005 there")
    auto_even: BoolProperty(name="Even out after cropping", default=False,
                            description="OFF keeps the scan's own resolution. Only turn it on if you want uniform topology and accept coarser detail")
    subdiv_levels: IntProperty(name="Subdivision levels", default=1, min=1, max=3,
                               description="Applied after sculpting, before texturing, so the mesh can carry fine relief")
    multires_level: IntProperty(name="Level", default=0, min=0, soft_max=6, update=lambda self, ctx: _mres(self, ctx),
                                description="Move between subdivision levels (0 = base mesh)")
    skin_target_spacing: FloatProperty(name="Skin detail spacing (mm)",default=.05,min=.005,soft_max=.15,precision=3,
        description="Suggested mesh spacing for skin relief; 0.05 mm is a starting point, 0.025 mm for finer detail. This measures mesh density, not printed texture quality")
    grid_front: BoolProperty(name="Millimetre grid", default=False)
    grid_spacing: FloatProperty(name="Spacing (mm)",default=5.,min=.1,soft_max=20)
    grid_opacity: FloatProperty(name="Grid opacity",default=.25,min=.02,max=1.)
    sculpt_target: EnumProperty(name="Surface", items=[("ACTIVE", "Selected mesh", "Use the active mesh", 3),
        ("OBJECT", "Choose surface", "Any mesh in the scene", 4), ("PROSTHESIS", "Sculpt", "Assigned sculpt", 5),
        ("PROTO", "Prototype", "", 0), ("SOURCE", "Sculpt", "", 1), ("SCAN", "Scan", "", 2),
        ("BASE", "Mold Base", "", 6), ("WEDGE", "Mold Wedge", "", 7), ("CAP", "Mold Cap", "", 8)], default="OBJECT")
    working_surface: EnumProperty(name="Working surface",items=_working_choices,get=_working_get,set=_working_set,update=_working_update)
    sculpt_surface: PointerProperty(name="Sculpt / texture surface", type=bpy.types.Object, poll=_is_mesh)
    xyz_depth_r: FloatProperty(update=_live,name="R coarse (mm)", default=0.12, min=0.0, soft_max=1.0)
    xyz_depth_g: FloatProperty(update=_live,name="G medium (mm)", default=0.06, min=0.0, soft_max=1.0)
    xyz_depth_b: FloatProperty(update=_live,name="B fine (mm)", default=0.03, min=0.0, soft_max=1.0)
    xyz_width: FloatProperty(update=_live,name="Map width on skin (mm)", default=180.0, min=5.0, soft_max=400.0,
                             description="How many mm of skin the full map width covers (a full-face map ≈ 180)")
    xyz_off_u: FloatProperty(update=_live,name="Shift U", default=0.0, min=-1.0, max=1.0)
    xyz_off_v: FloatProperty(update=_live,name="Shift V", default=0.0, min=-1.0, max=1.0)
    xyz_rotation: FloatProperty(update=_live,name="Rotate (°)", default=0.0, min=-180.0, max=180.0)
    xyz_midlevel: FloatProperty(update=_live,name="Mid level", default=0.5, min=0.0, max=1.0)
    xyz_subdivide: BoolProperty(name="Subdivide first", default=True)
    xyz_image: StringProperty(name="Loaded map", default="")
    xyz_bundled: StringProperty(name="Library map", default="")
    stamp_mode: BoolProperty(update=_live, name="Stamp mode", default=False,
                             description="Apply the map only inside a circle you place, with a soft edge — like a ZBrush alpha stamp")
    stamp_radius: FloatProperty(update=_live, name="Stamp radius (mm)", default=12.0, min=1.0, soft_max=60.0)
    stamp_centre: FloatVectorProperty(update=_live, name="Stamp centre", size=3, default=(0.0, 0.0, 0.0), subtype='TRANSLATION')
    slice_outline: BoolProperty(name="Cut outline", default=False)
    fit_offset: FloatProperty(name="Fitting-surface relief (mm)", default=0.0, min=0.0, soft_max=1.0,
                              description="Gap left between the sculpt fitting surface and the tissue, like die relief")
    xyz_mode: EnumProperty(name="Map type", items=[("ALPHA", "Alpha (grayscale)", "AlphaSkin and similar single-channel height maps"),
                                                  ("RGB", "Multi-channel (R/G/B)", "Texturing.xyz Multichannel: R coarse, G medium, B fine")], default="ALPHA")
    xyz_depth: FloatProperty(update=_live,name="Depth (mm)", default=0.15, min=0.0, soft_max=1.0,
                             description="Peak-to-trough displacement for a grayscale alpha map")

    # --- export (Tool 8) ---
    export_dir: StringProperty(name="Export folder", subtype='DIR_PATH', default="")
    export_format: EnumProperty(name="Format", items=[("STL", "STL", ""), ("OBJ", "OBJ", "")], default="STL")


def register():
    bpy.utils.register_class(ANAPLAST_CaseProps)
    bpy.types.Scene.anaplast = PointerProperty(type=ANAPLAST_CaseProps)
    bpy.app.handlers.load_post.append(_load_case_types)


def unregister():
    if _load_case_types in bpy.app.handlers.load_post:bpy.app.handlers.load_post.remove(_load_case_types)
    del bpy.types.Scene.anaplast
    bpy.utils.unregister_class(ANAPLAST_CaseProps)
