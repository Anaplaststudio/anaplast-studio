import bpy



class ANAPLAST_PT_case(bpy.types.Panel):

    bl_label = "Case"

    bl_space_type = 'VIEW_3D'

    bl_region_type = 'UI'

    bl_category = "Anaplast"

    def draw(self, context):

        from .ops.import_slot import SLOTS

        p = context.scene.anaplast

        l = self.layout

        notice = l.column(align=True)
        notice.label(text='v1.0.0 · Experimental preview', icon='INFO')
        notice.label(text='Not clinically validated')

        row = l.row(align=True)

        row.operator("anaplast.case_setup", icon='PREFERENCES')

        row.operator("anaplast.frame_case", text="", icon='VIEWZOOM')

        row = l.row(align=True); row.prop(p, "patient_first", text=""); row.prop(p, "patient_last", text="")

        l.prop(p, "mrn")

        l.prop(p, "case_date")

        row = l.row(align=True)

        row.prop(p, "prosthesis_type", text="")

        if p.prosthesis_type not in {"NASAL", "MIDFACIAL"}: row.prop(p, "side", text="")

        for key, label, _ in SLOTS:

            row = l.row(align=True)

            row.prop(p, key, text=label)

            row.operator("anaplast.import_slot", text="", icon='IMPORT').slot = key

class ANAPLAST_PT_eye_photo(bpy.types.Panel):

    bl_label='Eye photograph';bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='Anaplast';bl_options={'DEFAULT_CLOSED'}

    def draw(self,context):

        from .ops.eye_photo import draw

        draw(self.layout,context)



class ANAPLAST_PT_iris_button(bpy.types.Panel):

    bl_label='Iris button';bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='Anaplast';bl_options={'DEFAULT_CLOSED'}

    def draw(self,context):

        from .ops.iris_button import draw

        draw(self.layout,context)



class ANAPLAST_PT_ocular(bpy.types.Panel):

    bl_label='8 Â· Eye design';bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='Anaplast';bl_options={'DEFAULT_CLOSED'}

    def draw(self,context):

        from .ops.eye_design import draw

        draw(self.layout,context)

class ANAPLAST_PT_orient(bpy.types.Panel):

    bl_label = "1 Â· Orient, Mirror & Align"

    bl_space_type = 'VIEW_3D'

    bl_region_type = 'UI'

    bl_category = "Anaplast"

    def draw(self, context):

        from .ops.orient import LANDMARKS, get_landmark

        from .ops.align import _points

        p = context.scene.anaplast

        l = self.layout

        l.operator("anaplast.place_landmarks", icon='RESTRICT_SELECT_OFF')

        l.label(text="Tab: skip Â· Backspace: back Â· Enter: finish")

        l.label(text="or place them one by one:")

        col = l.column(align=True)

        for key, label, _ in LANDMARKS:

            placed = get_landmark(key) is not None

            col.operator("anaplast.add_landmark", text=label,

                         icon='CHECKMARK' if placed else 'RADIOBUT_OFF').key = key

        l.operator("anaplast.orient_case", icon='ORIENTATION_GLOBAL')

        row = l.row()

        row.enabled = p.oriented

        row.operator("anaplast.auto_symmetry", icon='MOD_MIRROR')

        if p.symmetry_rms > 0:

            l.label(text=f"Symmetric skin RMS: {p.symmetry_rms:.2f} mm", icon='INFO')

        row = l.row()

        row.enabled = p.oriented

        row.operator("anaplast.mirror_contralateral", icon='MOD_MIRROR')

        box = l.box()

        box.label(text="Align objects", icon='SNAP_ON')

        box.prop(p, "align_moving")

        box.prop(p, "align_target")

        box.label(text=f"Pairs: {len(_points('AL_S_'))} source / {len(_points('AL_T_'))} target")

        box.operator("anaplast.side_by_side", icon='ARROW_LEFTRIGHT')

        box.operator("anaplast.place_align_points", icon='RESTRICT_SELECT_OFF')

        row = box.row(align=True)

        row.operator("anaplast.clear_align_points", text="Clear pairs", icon='X')

        row.operator("anaplast.align_points", icon='SNAP_ON')

        row = box.row(align=True)

        row.operator("anaplast.favour_here", icon='RESTRICT_SELECT_OFF')

        row.operator("anaplast.favour_clear", text="", icon='X')

        box.prop(p, "icp_region")

        box.operator("anaplast.align_icp", icon='MOD_SHRINKWRAP')

        if p.last_rms > 0:

            box.label(text=f"RMS: {p.last_rms:.3f} mm")

        row = box.row(align=True)

        row.prop(p, "deviation_range")

        row.operator("anaplast.deviation_map", text="Deviation")

        row.operator("anaplast.deviation_reset", text="Clear deviation")

class ANAPLAST_PT_sculpt(bpy.types.Panel):

    bl_label = "2 Â· Sculpt & Texture"

    bl_space_type = 'VIEW_3D'

    bl_region_type = 'UI'

    bl_category = "Anaplast"

    def draw(self, context):

        from .ops.shell import source_surface

        from .ops.view import BRUSHES

        from .ops.multires import target, get_mod

        p = context.scene.anaplast

        l = self.layout

        box = l.box()

        box.label(text="Crop", icon='SELECT_SET')

        row = box.row(align=True)

        row.prop(p, "crop_mesh", text="Object")

        row.operator("anaplast.import_slot", text="", icon='IMPORT').slot = "prosthesis_obj"

        box.prop(p,"crop_shape",expand=True)

        box.operator("anaplast.smooth_lasso", text="Draw crop region", icon='GP_SELECT_STROKES')

        if p.crop_shape=='LASSO':box.prop(p,"lasso_smooth",text="Lasso smoothing")

        box.prop(p,"crop_finish",expand=True)

        box.prop(p,"crop_create_sculpt")

        row = box.row(align=True)

        row.operator("anaplast.crop_keep_selected", text="Keep").remove = False

        row.operator("anaplast.crop_keep_selected", text="Remove").remove = True

        if p.crop_preview_obj:box.operator('anaplast.crop_cancel',text='Clear crop region')

        src = source_surface(p)

        from .ops.sculpt_surface import target as surface_target

        surface = surface_target(context)

        l.prop(p, "working_surface", text="Working surface")

        l.label(text=f"Sculpt edge surface: {src.name}" if src else "No surface yet â€” Mirror + crop, or import one",

                icon='MESH_DATA' if src else 'ERROR')

        row = l.row(align=True)

        row.label(text="Subtract:")

        row.prop(p, "shell_trim", expand=True)

        box = l.box()

        box.label(text="Step 1 Â· Sculpt edges", icon='MOD_SHRINKWRAP')

        box.enabled = surface is not None and surface == src

        box.prop(p, "shell_clearance")

        box.prop(p, "margin_band")

        row = box.row(align=True); row.prop(p, "feather_width"); row.prop(p, "edge_thickness")

        row = box.row(align=True)

        row.operator("anaplast.adapt_margins", text="Seat the margins")

        box = l.box()

        box.label(text="Step 2 Â· Sculpt", icon='BRUSH_DATA')

        box.prop(p, "brush_strength")

        box.operator("anaplast.nasal_symmetry", icon='MOD_MIRROR')

        if surface is not None:

            row = box.row(align=True)

            row.prop(surface, "b4d_midline_symmetry", text="Symmetry", toggle=True)

        box.prop(p, "brush_frontface")

        col = box.column(align=True)

        for i in range(0, len(BRUSHES), 2):

            row = col.row(align=True)

            for key, label, _ in BRUSHES[i:i + 2]:

                row.operator("anaplast.sculpt", text=label).brush = key

        row = box.row(align=True)

        row = box.row(align=True)

        row.operator("anaplast.clear_masks", text="Clear surface mask")

        row.operator("anaplast.sculpt_exit", text="Done", icon='CHECKMARK')

        box = l.box()

        box.label(text="Step 3 Â· Remesh / Subdivide", icon='MOD_SUBSURF')

        box.prop(p,"sculpt_voxel")

        box.prop(p,"sculpt_reproject")

        box.prop(p,"sculpt_backing")

        box.operator("anaplast.remesh_surface",icon='MOD_REMESH')

        if p.sculpt_remesh_report:

            for part in p.sculpt_remesh_report.split('; '):box.label(text=part)

        box.separator()

        obj = target(context)

        mod = get_mod(obj) if obj else None

        row = box.row(align=True)

        row.operator("anaplast.multires_subdivide", icon='ADD')

        row.operator("anaplast.multires_apply", icon='CHECKMARK')

        if mod is not None:

            box.prop(p, "multires_level", slider=True, text=f"Level (of {mod.total_levels})")

        from .ops.multires import resolution_estimate

        box.prop(p,"skin_target_spacing")

        if obj:

            stats=resolution_estimate(obj,mod,context.scene)

            if not stats['sampled_edges']:

                box.label(text="Cannot assess density: no mesh edges",icon='ERROR')

            elif stats['ready']:

                box.label(text="Ready for skin texture",icon='CHECKMARK')

            elif stats['needed']<=stats['available']:

                box.label(text=f"Use existing level {stats['current_level']+stats['needed']} for skin texture",icon='INFO')

            else:

                extra=stats['needed']-stats['available']

                box.label(text=f"Add {extra} more subdivision level(s)",icon='INFO')

            box.label(text=f"Current level spacing: ~{stats['current_spacing']:.3f} mm")

            box.label(text="Readiness estimates mesh density only")

            box.label(text=f"Highest level faces: {stats['faces']:,}")

            box.label(text=f"Highest level spacing: {stats['spacing']:.3f} mm")

            box.label(text=f"Next level: about {stats['next_faces']:,} faces")

        box.label(text="Skin relief: start near 0.05 mm spacing")

        box.label(text="Fine detail: 0.02â€“0.03 mm where needed")

        box.label(text="More faces provide room for detail; add texture separately")

class ANAPLAST_PT_texture_more(bpy.types.Panel):

    bl_label = "Texture"

    bl_parent_id = "ANAPLAST_PT_sculpt"

    bl_space_type = 'VIEW_3D'

    bl_region_type = 'UI'

    bl_category = "Anaplast"

    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):

        p = context.scene.anaplast

        l = self.layout

        from .ops.sculpt_surface import target

        obj = target(context)

        l.label(text="Texture on: " + (obj.name if obj else "Choose a surface above"))

        box = l.box()

        box.label(text="Enhance scan detail", icon='SHARPCURVE')

        box.prop(p, "tex_gain"); box.prop(p, "tex_radius")

        box.operator("anaplast.texture_enhance")

        box = l.box()

        box.label(text="Procedural pores + creases", icon='TEXTURE')

        row = box.row(align=True); row.prop(p, "tex_pore_size"); row.prop(p, "tex_pore_depth")

        row = box.row(align=True); row.prop(p, "tex_crease_size"); row.prop(p, "tex_crease_depth")

        box.operator("anaplast.texture_procedural")

class ANAPLAST_PT_prototype(bpy.types.Panel):

    bl_label = "3 Â· Prototype"

    bl_space_type = 'VIEW_3D'

    bl_region_type = 'UI'

    bl_category = "Anaplast"

    def draw(self, context):

        from .ops.shell import source_surface, current_prototype

        p = context.scene.anaplast

        l = self.layout

        src = source_surface(p)

        proto = current_prototype(p)

        l.label(text=f"From: {src.name}" if src else "No surface yet", icon='MESH_DATA' if src else 'ERROR')

        l.prop(p, "shell_thickness")

        l.prop(p, "fit_offset")

        l.label(text="Preserve Sculpt detail automatically", icon='SHARPCURVE')

        l.prop(p, "voxel_size", text="Repair spacing (mm)")

        l.prop(p, "despeckle_enabled")

        if p.despeckle_enabled:l.prop(p, "despeckle_mm")

        l.operator("anaplast.build_prototype", icon='PLAY')

        if proto is not None:

            if proto.get("prototype_detail_method"):

                l.label(text=proto["prototype_detail_method"], icon='INFO')

            box = l.box()

            box.label(text="Check", icon='ZOOM_SELECTED')

            row = box.row(align=True); row.prop(p, "thick_min"); row.prop(p, "thick_max")

            box.label(text=f"red â‰¤ {p.thick_min:.2f} Â· yellow {(p.thick_min + p.thick_max) / 2:.2f} Â· green {p.thick_max:.2f} Â· blue > {p.thick_max * 2:.2f} mm", icon='INFO')

            row = box.row(align=True)

            row.operator("anaplast.thickness_map", text="Thickness", icon='COLOR')

            row.operator("anaplast.fit_map", text="Fit", icon='COLOR')

            row.operator("anaplast.deviation_reset", text="Clear deviation")

            box.prop(p, "fit_range")

            box.operator("anaplast.lift_prototype", icon='ORIENTATION_VIEW')

def draw_view_contents(l, context):

    from .ops.view import MODES

    p = context.scene.anaplast

    row = l.row(align=True)

    row.operator("anaplast.fit_view", text="Fit case", icon='ZOOM_ALL').selected_only = False

    row.operator("anaplast.fit_view", text="Fit selected").selected_only = True

    row = l.row(align=True)

    row.operator("anaplast.grid_front", text="Millimetre grid", depress=p.grid_front)

    row.operator("anaplast.polyframe", text="Polyframe", depress=p.polyframe)

    l.operator("anaplast.crosshair",text="Mouse guides",depress=p.crosshair)

    l.operator("anaplast.fidelity_report",icon='RNA')

    if p.grid_front:

        row=l.row(align=True);row.prop(p,"grid_spacing");row.prop(p,"grid_opacity",slider=True)

        l.label(text="At view-center depth; spacing coarsens when zoomed out")

    row = l.row(align=True)

    row.operator("anaplast.clip_region", icon='MOD_BEVEL')

    row.operator("anaplast.measure_tool", text="Measure", icon='DRIVER_DISTANCE')

    row.operator("anaplast.measure_clear", text="", icon='X')

    box = l.box()

    box.label(text="Reference photo", icon='IMAGE_REFERENCE')

    row = box.row(align=True)

    row.operator("anaplast.photo_add", icon='FILE_IMAGE')

    row.operator("anaplast.photo_remove", text="", icon='X')

    from .ops.photo import selected_photo

    photo=selected_photo(context)

    if photo:

        box.label(text="Selected: "+photo.name)

        box.prop(photo,"b4d_photo_scale",slider=True)

        box.prop(photo,"b4d_photo_opacity",slider=True)

    else:box.label(text="Select a reference photo to adjust it")

    box.prop(p,"photo_tolerance")

    row=box.row(align=True)

    if photo:row.operator("anaplast.photo_view", icon='CAMERA_DATA').photo_name=photo.name

    row.operator("anaplast.photo_face_view",text="Save photo position")

    from .ops.photo import photos

    for item in photos(context):

        row=box.row(align=True)

        row.label(text=item.name)

        row.operator("anaplast.photo_view",text="Return",icon='CAMERA_DATA').photo_name=item.name

    box = l.box()

    box.label(text="Light", icon='LIGHT_SUN')

    row=box.row(align=True)

    row.operator("anaplast.light_drag",icon='MOUSE_MOVE')

    row.operator("anaplast.light_orbit",icon='FILE_REFRESH')

    l.label(text="Surface Appearance:")

    for start in (0,3):

        row = l.row(align=True)

        for key, label, _ in MODES[start:start+3]:

            row.operator("anaplast.display_mode", text=label).mode = key

    box=l.box()

    box.label(text="Object Appearance")

    box.prop(p,"working_surface",text="Object")

    from .ops.sculpt_surface import target as appearance_target

    from .ops.view import scan_color_type

    obj=appearance_target(context)

    if obj:

        box.prop(obj,"b4d_scan_color")

        if obj.b4d_scan_color and not scan_color_type(obj):

            box.label(text="No scan colors on this surface",icon='INFO')

        box.prop(obj,"b4d_surface_tint",text="Tint")

        box.prop(obj,"b4d_surface_opacity",slider=True)

        box.label(text="100% opaque Â· 0% invisible")

    l.label(text="Show / hide:")

    row = l.row(align=True)

    row.operator("anaplast.toggle_visible", text="Scan").which = 'SCAN'

    row.operator("anaplast.toggle_visible", text="Cast").which = 'CAST'

    row = l.row(align=True)

    row.operator("anaplast.toggle_visible", text="Prototype").which = 'PROTO'

    row.operator("anaplast.toggle_visible", text="Sculpt").which = 'SOURCE'

    l.prop(p,"show_landmark_markers")

    l.prop(p,"show_construction_helpers")

class ANAPLAST_PT_view(bpy.types.Panel):

    bl_label = "View"

    bl_space_type = 'VIEW_3D'

    bl_region_type = 'UI'

    bl_category = "Anaplast"

    def draw(self, context):

        draw_view_contents(self.layout, context)

class ANAPLAST_PT_view_popover(bpy.types.Panel):

    """The same View tools as a pop-up: opens from the 'View' button in the 3D viewport header, or press V while hovering the viewport"""

    bl_label = "Anaplast view"

    bl_space_type = 'VIEW_3D'

    bl_region_type = 'HEADER'

    bl_ui_units_x = 18

    def draw(self, context):

        draw_view_contents(self.layout, context)

def _header_button(self, context):

    if context.space_data and context.space_data.type == 'VIEW_3D':

        self.layout.popover(panel="ANAPLAST_PT_view_popover", text="Anaplast view", icon='HIDE_OFF')

class ANAPLAST_PT_ssm(bpy.types.Panel):

    bl_label = "4 Â· Legacy shape-model training (SSM)"

    bl_space_type = 'VIEW_3D'

    bl_region_type = 'UI'

    bl_category = "Anaplast"

    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):

        import json

        p = context.scene.anaplast

        l = self.layout

        box = l.box()

        box.label(text="Train", icon='RNA')

        box.prop(p, "ssm_dataset_dir", text="Dataset")

        row = box.row(align=True); row.prop(p, "mpfb_gender", text=""); row.prop(p, "mpfb_age", text="")

        box.prop(p, "mpfb_keep", expand=True)

        box.prop(p, "mpfb_subdiv")

        box.operator("anaplast.ssm_generate_mpfb", icon='COMMUNITY')

        reg = json.loads(p.ssm_region)["region"] if p.ssm_region else None

        box.label(text=f"Region: {len(reg)} verts" if reg else "Region: import one dataset mesh, lasso it in Edit Mode, then:", icon='CHECKMARK' if reg else 'INFO')

        box.operator("anaplast.ssm_set_region")

        box.prop(p, "ssm_keep_variance")

        box.prop(p, "ssm_model_path", text="Model")

        box.operator("anaplast.ssm_train")

        box = l.box()

        box.label(text="Fit to patient", icon='MOD_MESHDEFORM')

        box.operator("anaplast.ssm_place_mean")

        box.label(text="then align SSM_Mean with the Align panel", icon='INFO')

        box.prop(p, "ssm_max_dist")

        box.prop(p, "ssm_regularisation")

        box.prop(p, "ssm_iterations")

        box.operator("anaplast.ssm_fit")

class ANAPLAST_PT_guide(bpy.types.Panel):

    bl_label = "6 Â· Surgical Guide"

    bl_space_type = 'VIEW_3D'

    bl_region_type = 'UI'

    bl_category = "Anaplast"

    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):

        from .ops.guide import implants

        from .ops.align import _points

        p = context.scene.anaplast

        l = self.layout

        box = l.box()

        box.label(text="CBCT â†’ scan alignment", icon='SNAP_ON')

        box.operator("anaplast.align_cbct_setup")

        box.label(text=f"Pairs: {len(_points('AL_S_'))} / {len(_points('AL_T_'))}   Moving: {p.align_moving.name if p.align_moving else 'â€”'}")

        box.operator("anaplast.place_align_points", icon='RESTRICT_SELECT_OFF')

        box.operator("anaplast.clear_align_points", text="Clear pairs", icon='X')

        row = box.row(align=True)

        row.operator("anaplast.align_points")

        row.operator("anaplast.align_icp")

        box = l.box()

        box.label(text=f"Implants ({len(implants())})", icon='EMPTY_AXIS')

        row = box.row(align=True); row.prop(p, "implant_diameter"); row.prop(p, "implant_length")

        box.label(text="Shift+Right-click the bone/skin, then:", icon='INFO')

        box.operator("anaplast.add_implant")

        box = l.box()

        box.label(text="Guide", icon='MOD_CAST')

        box.prop(p, "guide_support", expand=True)

        box.prop(p, "guide_thickness")

        box.prop(p, "guide_radius")

        box.prop(p, "sleeve_diameter")

        box.operator("anaplast.make_guide")

class ANAPLAST_PT_bar(bpy.types.Panel):

    bl_label = "7 Â· Bar & Substructure"

    bl_space_type = 'VIEW_3D'

    bl_region_type = 'UI'

    bl_category = "Anaplast"

    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):

        from .ops.guide import implants

        p = context.scene.anaplast

        l = self.layout

        l.label(text=f"Uses the {len(implants())} implant(s) from the Guide tab", icon='INFO')

        l.prop(p, "retention", expand=True)

        row = l.row(align=True); row.prop(p, "abutment_diameter"); row.prop(p, "bar_height")

        if p.retention == 'BAR':

            l.prop(p, "bar_diameter")

            row = l.row(align=True); row.prop(p, "housing_length"); row.prop(p, "housing_wall")

        else:

            row = l.row(align=True); row.prop(p, "magnet_diameter"); row.prop(p, "magnet_height")

        l.operator("anaplast.make_bar")

        box=l.box();box.label(text="Generic substructure for manual components")

        box.prop(p,"sub_outer_obj");box.prop(p,"sub_fit_obj")

        from .ops.substructure import sources

        outer,_=sources(p)

        visible=outer is not None and outer.name in context.view_layer.objects and outer.visible_get()

        box.operator("anaplast.substructure_view",text="Hide sculpt" if visible else "Show sculpt",icon='HIDE_OFF' if visible else 'HIDE_ON')

        box.label(text="Blank fields use the selected Sculpt and Scan")

        row=box.row(align=True);row.prop(p,"sub_relief");row.prop(p,"sub_silicone")

        row=box.row(align=True);row.prop(p,"sub_smooth");row.prop(p,"sub_resolution")

        box.prop(p,"sub_component_source",expand=True)

        if p.sub_component_source=='PAINT':

            box.operator("anaplast.component_paint",text="Paint on fitting scan").action='START'

            if p.sub_mark_obj:

                row=box.row(align=True)

                op=row.operator("anaplast.wedge_mark_tool",text="Paint");op.target='COMPONENT';op.erase=False

                op=row.operator("anaplast.wedge_mark_tool",text="Erase");op.target='COMPONENT';op.erase=True

                box.prop(p,"sub_mark_radius")

                row=box.row(align=True)

                op=row.operator("anaplast.wedge_mark_tool",text="Lasso add");op.target='COMPONENT';op.lasso=True

                op=row.operator("anaplast.wedge_mark_tool",text="Lasso remove");op.target='COMPONENT';op.lasso=True;op.erase=True

                row=box.row(align=True);row.operator("anaplast.component_paint",text="Clear paint").action='CLEAR';row.operator("anaplast.component_paint",text="New painting copy").action='NEW'

                row=box.row(align=True);row.operator("anaplast.component_paint",text="Use painted components").action='APPLY';row.operator("anaplast.component_paint",text="Return to model").action='VIEW'

            box.label(text="Paint one separate patch per component")

        else:box.operator("anaplast.substructure_marker")

        from .ops.substructure import markers

        box.label(text=f"Component positions: {len(markers())}")

        if p.sub_component_source=='MARKERS':box.label(text="Rotate marker Z to the component axis")

        box.prop(p,"sub_layout")

        if p.sub_layout!='FULL':

            row=box.row(align=True);row.prop(p,"sub_width");row.prop(p,"sub_length")

        box.prop(p,"sub_height")

        if p.sub_layout=='SEPARATE':

            box.prop(p,"sub_piece_gap")

            box.label(text="Splits lie midway between implants")

            box.label(text="Footprint controls the outer size")

        box.prop(p,"sub_ocular_relief")

        if p.sub_ocular_relief:box.prop(p,"sub_ocular_obj");box.prop(p,"sub_ocular_gap")

        box.label(text="Size is limited by cover, fitting and ocular relief")

        box.prop(p,"sub_pockets")

        if p.sub_pockets:

            box.prop(p,"sub_component_diameter");box.prop(p,"sub_component_depth");box.prop(p,"sub_component_clearance")

        box.label(text="Generic spaces; no implant threads or proprietary interfaces")

        box.operator("anaplast.substructure_build")

        box.operator("anaplast.substructure_export")

        if p.sub_report:box.label(text=p.sub_report)

class ANAPLAST_PT_export(bpy.types.Panel):

    bl_label = "9 Â· Export & Save"

    bl_space_type = 'VIEW_3D'

    bl_region_type = 'UI'

    bl_category = "Anaplast"

    def draw(self, context):

        p = context.scene.anaplast

        l = self.layout

        from .ops import mold_volume

        mold_volume.draw(l,context.scene)

        l.prop(p, "export_dir", text="Folder")

        row = l.row(align=True); row.label(text="Format:"); row.prop(p, "export_format", expand=True)

        l.operator("anaplast.export_prototype", text="Sculpt", icon='EXPORT')

        row = l.row(align=True)

        row.operator("anaplast.export_check", text="Mold (all parts)", icon='MOD_BOOLEAN').what = 'MOLD'

        l.operator("anaplast.export_combined", text="Face + sculpt (one mesh)", icon='EXPORT')

        l.separator()

        l.label(text="Snapshots (8 views):", icon='RESTRICT_RENDER_OFF')

        row = l.row(align=True)

        row.operator("anaplast.snapshots", text="With").mode = 'WITH'

        row.operator("anaplast.snapshots", text="Without").mode = 'WITHOUT'

        row.operator("anaplast.snapshots", text="Side by side").mode = 'COMPARE'

        row.operator("anaplast.snapshots", text="Sculpt").mode = 'PROTO'

        l.separator()

        l.operator("anaplast.save_case", icon='FILE_TICK')

        l.operator("anaplast.bug_report", icon='TEXT')

class ANAPLAST_PT_mold(bpy.types.Panel):

    bl_label = "5 Â· Mold"

    bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='Anaplast'

    def draw(self,context):

        from .ops import mold_workflow as flow,mold_inserts

        from types import SimpleNamespace

        p=context.scene.anaplast;w=context.scene.mold_workflow;l=self.layout

        def build(layout,part,text):

            layout.operator('anaplast.mold_workflow',text=text,icon='PLAY').part=part

        box=l.box();box.label(text='Active mold')

        if w.active_id:box.label(text=next((c.name for c in bpy.data.collections if c.get('mold_set')==w.active_id),'No mold yet'))

        row=box.row();row.prop(w,'selection',text='');row.operator('anaplast.mold_workflow',text='Switch').part='SWITCH'

        box.operator('anaplast.mold_workflow',text='Organize scene collections').part='ORGANIZE'

        box=l.box();box.label(text='Mold features')

        box.prop(w,'show_keys',icon='TRIA_DOWN' if w.show_keys else 'TRIA_RIGHT',emboss=False)

        if w.show_keys:

            sub=box.box();sub.label(text='Baseâ€“Cap');sub.prop(p,'m_key_type',text='Shape')

            if p.m_key_type!='NONE':

                row=sub.row(align=True);row.prop(p,'m_key_count');row.prop(p,'m_key_diameter',text='Outer diameter')

                row=sub.row(align=True);row.prop(p,'m_key_flare');row.prop(p,'m_key_blend');sub.prop(p,'key_clearance')

                if p.m_key_type=='WEDGE':sub.prop(p,'m_wedge_inner');sub.prop(p,'m_wedge_ring_gap')

            sub=box.box();sub.label(text='Baseâ€“Wedge and Wedgeâ€“Cap');sub.prop(p,'wedge_keys',text='Registration keys')

            if p.wedge_keys:sub.prop(p,'wedge_key_count',text='Keys on each connection')

        box.prop(w,'show_pry',icon='TRIA_DOWN' if w.show_pry else 'TRIA_RIGHT',emboss=False)

        if w.show_pry:

            sub=box.box();sub.prop(p,'m_pry',text='Baseâ€“Cap points');sub.prop(p,'wedge_pry',text='Wedge connections')

            sub.prop(p,'m_pry_width');sub.prop(p,'m_pry_height')

            if p.m_ring and p.m_spillways:sub.label(text='Reach: stop 1 mm before ring spillway')

            else:sub.prop(p,'m_pry_depth')

        box.prop(w,'show_flow',icon='TRIA_DOWN' if w.show_flow else 'TRIA_RIGHT',emboss=False)

        if w.show_flow:

            sub=box.box();sub.prop(p,'m_ring');sub.prop(p,'m_spillways',text='Feed channels');sub.prop(p,'m_spillway_width');sub.prop(p,'wedge_spillway',text='Wedge channels')

        box.prop(w,'show_bolts',icon='TRIA_DOWN' if w.show_bolts else 'TRIA_RIGHT',emboss=False)

        if w.show_bolts:

            sub=box.box();sub.prop(p,'m_bolts');sub.prop(p,'m_bolt_diameter');sub.prop(p,'m_bolt_sleeve_length',expand=True)

            sub.prop(p,'m_bolt_sleeve_wall');sub.prop(p,'m_bolt_seat_raise');sub.label(text='Clamp bolts currently support Baseâ€“Cap')

        box=l.box();box.label(text='Base')

        box.label(text='Opening direction: current view when building Base')

        box.prop(p,'face_scan_obj',text='Fitting scan');box.prop(p,'source_prosthesis',text='Sculpt')

        box.prop(w,'base_body',expand=True)

        if w.base_body=='SHELL':box.prop(w,'base_wall')

        else:box.prop(p,'m_base_depth')

        row=box.row(align=True);row.prop(p,'m_skin');row.prop(p,'m_land')

        box.prop(p,'m_land_transition');box.prop(p,'m_land_follow')

        box.prop(p,'m_fit_smoothing',text='Fitting surface smoothing');box.prop(p,'m_skin_smoothing',text='Skin band smoothing')

        build(box,'BASE','Build baseâ€¦')

        box=l.box();box.label(text='Wedge');box.label(text='Sculpt contact area')

        box.operator('anaplast.wedge_curve',text='Draw smooth contact boundary').action='DRAW'

        row=box.row(align=True);row.operator('anaplast.wedge_curve',text='Edit curve').action='EDIT';row.operator('anaplast.wedge_curve',text='Preview contact').action='PREVIEW'

        row=box.row(align=True);row.operator('anaplast.wedge_mark_start',text='Paint contact area');row.operator('anaplast.wedge_suggest',text='Suggest rear undercut')

        if p.wedge_mark_obj and p.wedge_selection!='TRACE':

            row=box.row(align=True)

            for erase,label in ((False,'Paint'),(True,'Erase')):

                op=row.operator('anaplast.wedge_mark_tool',text=label);op.erase=erase;op.lasso=False

            box.prop(p,'wedge_mark_radius');box.prop(p,'wedge_border_detail',text='Paint boundary refinement')

        box.operator('anaplast.wedge_mark_action',text='Return to mold').action='VIEW'

        box.prop(w,'wedge_body',expand=True)

        if w.wedge_body=='SHELL':box.prop(w,'wedge_wall')

        for index,label in ((1,'Upper end'),(2,'Lower end')):

            sub=box.box();sub.label(text=label);sub.prop(p,f'wedge_end{index}_mode',expand=True)

            if getattr(p,f'wedge_end{index}_mode')=='DISTANCE':sub.prop(p,f'wedge_end{index}_extra',text='Extra distance')

            else:sub.operator('anaplast.wedge_endpoint',text='Choose point on mold rim').end=index

        box.prop(p,'wedge_transition');box.prop(p,'wedge_top_smoothing');box.prop(p,'wedge_end_blend')

        build(box,'WEDGE','Build wedgeâ€¦')

        box=l.box();box.label(text='Inserts')

        box.label(text='Set positions here; build inserts after the Cap')

        mold_inserts.ANAPLAST_PT_inserts.draw(SimpleNamespace(layout=box),context)

        box=l.box();box.label(text='Cap');box.prop(w,'cap_support',expand=True);box.prop(w,'cap_body',expand=True)

        if w.cap_body=='SHELL':box.prop(w,'cap_wall')

        else:box.prop(p,'m_cap_height')

        box.prop(p,'m_ocular_impression')

        build(box,'CAP','Build cap')

        sub=box.box();sub.label(text='Separate breakable shell â€” over Base')

        sub.prop(p,'m_cap_wall');sub.prop(p,'m_break_pattern')

        if p.m_break_pattern!='NONE':

            sub.prop(p,'m_break_width');sub.prop(p,'m_break_remaining')

            sub.prop(p,'m_break_spacing' if p.m_break_pattern=='GRID' else 'm_break_count');sub.prop(p,'m_break_rotation')

        build(sub,'BREAK','Build separate breakable shell')

        if bpy.data.objects.get('Mold_BreakableShell'):sub.operator('anaplast.mold_workflow',text='Show / hide breakable shell').part='SHOW_BREAK'

        box=l.box();box.label(text='Inspect')

        box.operator('anaplast.explode_view',text='Assemble' if p.exploded else 'Explode')

        box.operator('anaplast.mold_contact',text='Check fit')

        box.operator('anaplast.wedge_mark_action',text='Check wedge contact coverage').action='COVERAGE'

        box.operator('anaplast.wedge_coverage',text='Show base boundary')

        if w.report:

            import textwrap

            for line in textwrap.wrap(w.report,max(28,int(context.region.width/7)-6)):box.label(text=line)

        if p.wedge_report:

            import textwrap

            for line in textwrap.wrap(p.wedge_report,max(28,int(context.region.width/7)-6)):box.label(text=line)

class ANAPLAST_PT_mold_steps(bpy.types.Panel):

    bl_label = "Build report"

    bl_space_type = 'VIEW_3D'

    bl_region_type = 'UI'

    bl_category = "Anaplast"

    bl_parent_id = "ANAPLAST_PT_mold"

    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):

        import json

        l = self.layout

        try:

            steps = json.loads(context.scene.anaplast.mold_steps or "[]")

        except Exception:

            steps = []

        if not steps:

            l.label(text="No build yet", icon='INFO')

            return

        total = sum(st.get("s", 0) for st in steps)

        col = l.column(align=True)

        for i, st in enumerate(steps, 1):

            row = col.row(align=True)

            row.label(text=f"{i}. {st['step']}", icon='CHECKMARK' if st.get("ok") else 'CANCEL')

            row.label(text=f"{st.get('s', 0):.0f}s" if st.get("ok") else "FAILED")

            if st.get("info"):

                sub = col.row(); sub.alignment = 'LEFT'; sub.scale_y = 0.8

                sub.label(text="      " + st["info"][:90])

        l.label(text=f"total {total:.0f} s")

_classes = (ANAPLAST_PT_case, ANAPLAST_PT_view, ANAPLAST_PT_orient, ANAPLAST_PT_sculpt, ANAPLAST_PT_texture_more, ANAPLAST_PT_prototype, ANAPLAST_PT_view_popover, ANAPLAST_PT_ssm, ANAPLAST_PT_mold, ANAPLAST_PT_mold_steps, ANAPLAST_PT_guide, ANAPLAST_PT_bar, ANAPLAST_PT_ocular, ANAPLAST_PT_export)



_keymaps = []



def register():

    for c in _classes:

        bpy.utils.register_class(c)



    bpy.types.VIEW3D_HT_header.append(_header_button)

    try:

        wm = bpy.context.window_manager

        km = wm.keyconfigs.addon.keymaps.new(name="3D View", space_type='VIEW_3D')

        kmi = km.keymap_items.new("wm.call_panel", type='V', value='PRESS')

        kmi.properties.name = "ANAPLAST_PT_view_popover"

        kmi.properties.keep_open = True

        _keymaps.append((km, kmi))

        clip=km.keymap_items.new("anaplast.clip_region",type="B",value="PRESS",alt=True)

        _keymaps.append((km,clip))

    except Exception:

        pass

def unregister():



    try:

        bpy.types.VIEW3D_HT_header.remove(_header_button)

    except Exception:

        pass

    for km, kmi in _keymaps:

        try:

            km.keymap_items.remove(kmi)

        except Exception:

            pass

    _keymaps.clear()

    for c in reversed(_classes):

        bpy.utils.unregister_class(c)

