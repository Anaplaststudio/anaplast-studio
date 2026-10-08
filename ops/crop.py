"""General mesh cropping with optional closure and a separate Sculpt region."""
import bpy
import bmesh
from mathutils import Matrix, Vector
from ..utils import mesh as mu
from .report import rep

CROP = "Crop_Box"


def _view_perf(context, on):
    """Dense-scan viewport mode: overlays off and X-ray on while cropping in Edit Mode; restore after."""
    sd = getattr(context, "space_data", None)
    if sd is None or not hasattr(sd, "shading"):
        return
    sd.shading.show_xray = on
    sd.overlay.show_overlays = not on


class ANAPLAST_OT_lasso_crop_start(bpy.types.Operator):
    """Sculpt-mode mask on the Sculpt (fast even on millions of faces): lasso = drag, polyline = click points + Enter. Then Keep Selected"""
    bl_idname = "anaplast.lasso_crop_start"
    bl_label = "Lasso: Start"
    bl_options = {'REGISTER'}

    tool: bpy.props.EnumProperty(items=[("LASSO", "Lasso", ""), ("POLYLINE", "Polyline", "")], default="LASSO")

    @classmethod
    def poll(cls, context):
        return mu.crop_object(context.scene.anaplast) is not None

    def execute(self, context):
        obj = mu.crop_object(context.scene.anaplast)
        from .sculpt_prepare import begin_crop
        obj=begin_crop(context,obj)
        for o in context.view_layer.objects:
            o.select_set(False)
        obj.hide_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        # sculpt mode draws with its own fast structure; edit mode redraws every edge and point
        try:
            bpy.ops.object.mode_set(mode='SCULPT')
            bpy.ops.wm.tool_set_by_id(name="builtin.polyline_mask" if self.tool == 'POLYLINE' else "builtin.lasso_mask")
            mode = "SCULPT"
        except Exception:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_mode(type='FACE')             # face mode: no vertex dots to draw
            bpy.ops.mesh.select_all(action='DESELECT')
            try:
                bpy.ops.wm.tool_set_by_id(name="builtin.select_lasso")
            except Exception:
                pass
            mode = "EDIT"
        _view_perf(context, mode == "EDIT")          # sculpt draws fast: keep overlays so the mask shows
        if mode == "SCULPT":
            sd = getattr(context, "space_data", None)
            if sd is not None and hasattr(sd, "overlay"):
                sd.overlay.show_sculpt_mask = True
                sd.overlay.sculpt_mode_mask_opacity = 0.8
                sd.shading.show_xray = True
            rep(self, {'INFO'}, "Sculpt Lasso Mask: drag around the region to KEEP (Ctrl+drag removes mask), then Keep Selected")
        else:
            rep(self, {'INFO'}, "Drag a lasso around the region to keep (Shift+drag adds), then Keep Selected")
        return {'FINISHED'}


def clear_preview(context):
    p=context.scene.anaplast
    preview=p.crop_preview_obj
    if preview is None:return
    if context.object and context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    source=bpy.data.objects.get(preview.get('crop_source',''))
    if source:
        source.hide_set(bool(preview.get('crop_source_hidden',False)))
        source.select_set(True);context.view_layer.objects.active=source
    p.crop_preview_obj=None
    mu.delete_object(preview)


def crop_mesh_data(mesh,selected,*,remove=False,close=False,view_dir=None,matrix=None):
    """Crop a temporary BMesh; source data stays intact until validation succeeds."""
    bm=bmesh.new();bm.from_mesh(mesh)
    try:
        bm.verts.ensure_lookup_table();bm.verts.index_update()
        marked=set(selected)
        if not marked:raise RuntimeError('Draw a crop region first')
        discard=[v for v in bm.verts if (v.index in marked)==remove]
        bmesh.ops.delete(bm,geom=discard,context='VERTS')
        loose=[v for v in bm.verts if not v.link_faces]
        if loose:bmesh.ops.delete(bm,geom=loose,context='VERTS')
        if not bm.faces:raise RuntimeError('The crop would leave no faces; draw a larger region')
        if view_dir is not None and not remove:
            comps=mu.islands(bm)
            if len(comps)>1:
                big=[c for c in comps if len(c)>=.05*len(bm.faces)] or comps
                keep=min(big,key=lambda c:sum((matrix@f.calc_center_median()).dot(view_dir) for f in c)/len(c))
                bmesh.ops.delete(bm,geom=[f for c in comps if c is not keep for f in c],context='FACES')
        cap_layer=bm.faces.layers.int.get('anaplast_crop_cap') or bm.faces.layers.int.new('anaplast_crop_cap')
        caps=0
        if close:
            boundary=[e for e in bm.edges if e.is_boundary]
            if any(sum(e.is_boundary for e in v.link_edges)!=2 for e in boundary for v in e.verts):
                raise RuntimeError('The boundary branches; use Keep open or repair the boundary first')
            if boundary:
                new=bmesh.ops.holes_fill(bm,edges=boundary,sides=0)['faces'];caps=len(new)
                for face in new:face[cap_layer]=1
                bmesh.ops.triangulate(bm,faces=[f for f in new if len(f.verts)>3])
                bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
            if any(not e.is_manifold for e in bm.edges):raise RuntimeError('Could not close all boundaries; the original mesh was kept')
            if abs(bm.calc_volume(signed=True))<1e-8:raise RuntimeError('This flat region needs thickness to form a closed mesh; use Keep open')
        output=bpy.data.meshes.new(mesh.name+'_crop');bm.to_mesh(output);output.update()
        for material in mesh.materials:output.materials.append(material)
        if mesh.uv_layers.active and output.uv_layers.get(mesh.uv_layers.active.name):
            output.uv_layers.active=output.uv_layers[mesh.uv_layers.active.name]
        if mesh.color_attributes.active_color and output.color_attributes.get(mesh.color_attributes.active_color.name):
            output.color_attributes.active_color=output.color_attributes[mesh.color_attributes.active_color.name]
        return output,caps
    finally:bm.free()


class ANAPLAST_OT_crop_cancel(bpy.types.Operator):
    bl_idname='anaplast.crop_cancel';bl_label='Clear crop region';bl_options={'REGISTER','UNDO'}
    @classmethod
    def poll(cls,context):return context.scene.anaplast.crop_preview_obj is not None
    def execute(self,context):
        clear_preview(context);return {'FINISHED'}


class ANAPLAST_OT_crop_keep_selected(bpy.types.Operator):
    """Keep or remove the marked region; optionally make a separate Sculpt"""
    bl_idname='anaplast.crop_keep_selected';bl_label='Apply Crop';bl_options={'REGISTER','UNDO'}
    remove:bpy.props.BoolProperty(name='Remove region',default=False)
    @classmethod
    def poll(cls,context):
        return context.scene.anaplast.crop_preview_obj is not None or context.mode in {'EDIT_MESH','SCULPT'}
    def execute(self,context):
        import numpy as np
        p=context.scene.anaplast;preview=p.crop_preview_obj
        target=preview or context.object
        if not target or target.type!='MESH':return {'CANCELLED'}
        source=bpy.data.objects.get(target.get('crop_source','')) if preview else target
        if source is None:
            rep(self,{'ERROR'},'The crop source no longer exists');return {'CANCELLED'}
        if preview and p.crop_target!=source:
            rep(self,{'ERROR'},'The crop object changed; clear the preview and draw a region on the new object');return {'CANCELLED'}
        mode=context.mode
        if mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        mask=target.data.attributes.get('.sculpt_mask')
        if preview or mode=='SCULPT':
            values=np.zeros(len(target.data.vertices))
            if mask:mask.data.foreach_get('value',values)
            marked=np.flatnonzero(values>.5).tolist()
        else:marked=[v.index for v in target.data.vertices if v.select]
        rd=getattr(context,'region_data',None)
        view_dir=Vector(preview['crop_view_direction']) if preview else (rd.view_rotation@Vector((0,0,-1)) if rd else Vector((0,-1,0)))
        result=None
        try:
            from .crop_closed import is_solid,close_solid
            solid_crop=p.crop_finish=='CLOSE' and preview is not None and is_solid(target.data)
            if solid_crop:
                result,caps=close_solid(context,target,remove=self.remove)
            else:
                result,caps=crop_mesh_data(target.data,marked,remove=self.remove,close=p.crop_finish=='CLOSE',
                    view_dir=view_dir if p.crop_create_sculpt else None,matrix=target.matrix_world)
            if p.crop_finish=='CLOSE':
                from .mold_design import has_self_intersections
                check=bpy.data.objects.new('Crop_check',result)
                try:
                    if has_self_intersections(check):raise RuntimeError('The closure intersects the surface; use Keep open or change the crop')
                finally:bpy.data.objects.remove(check)
        except Exception as exc:
            if result is not None and result.users==0:bpy.data.meshes.remove(result)
            rep(self,{'ERROR'},str(exc));return {'CANCELLED'}
        if p.crop_create_sculpt:
            from .scene_helpers import group,GROUPS
            obj=bpy.data.objects.new('Sculpt',result);group(context.scene,GROUPS[3]).objects.link(obj)
            obj.matrix_world=target.matrix_world.copy()
            obj.color=source.color
            obj['anaplast_part']='SCULPT';obj['sculpt_crop_complete']=True;obj['crop_reference']=source.name
            if caps:
                obj['sculpt_crop_capped']=True;obj['sculpt_temporary_backing']=True
            p.prosthesis_obj=p.source_prosthesis=p.sculpt_surface=obj;p.sculpt_target='OBJECT'
        else:
            obj=source;old=obj.data;obj.data=result
            if preview:obj.modifiers.clear()
            if old.users==0:bpy.data.meshes.remove(old)
            if obj.get('anaplast_part')=='SCULPT' and caps:
                obj['sculpt_crop_capped']=True;obj['sculpt_temporary_backing']=True
        mask=result.attributes.get('.sculpt_mask')
        if mask:result.attributes.remove(mask)
        clear_preview(context)
        if p.crop_create_sculpt:source.hide_set(True)
        for o in context.selected_objects:o.select_set(False)
        obj.hide_set(False);obj.select_set(True);context.view_layer.objects.active=obj;p.crop_target=obj
        from .scene_helpers import organize_object
        organize_object(obj,context.scene,preserve_visibility=True)
        rep(self,{'INFO'},f"{obj.name}: {len(result.polygons):,} faces; {'closed mesh' if p.crop_finish=='CLOSE' else 'cut edges left open'}")
        return {'FINISHED'}


class ANAPLAST_OT_protect_paint(bpy.types.Operator):
    """Paint areas of the Sculpt that the build must NOT change (margin adaptation, despeckle). Sculpt mask brush; painted masking remains on the mesh"""
    bl_idname = "anaplast.protect_paint"
    bl_label = "Paint Protected Areas"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        from .sculpt_surface import target, activate
        return target(context) is not None

    def execute(self, context):
        from .sculpt_surface import target, activate
        obj = target(context)
        activate(context, obj)
        bpy.ops.object.mode_set(mode='SCULPT')
        ok = False
        for fname in ("essentials_brushes-mesh_sculpt.blend", "essentials_sculpt.blend"):
            try:
                bpy.ops.wm.tool_set_by_id(name="builtin.brush")
                bpy.ops.brush.asset_activate(asset_library_type='ESSENTIALS', asset_library_identifier="",
                                             relative_asset_identifier=f"brushes/{fname}/Brush/Mask")
                ok = True
                break
            except Exception:
                continue
        if not ok:
            try:
                bpy.ops.wm.tool_set_by_id(name="builtin_brush.mask")
            except Exception:
                pass
        sd = getattr(context, "space_data", None)
        if sd is not None and hasattr(sd, "overlay"):
            sd.overlay.show_sculpt_mask = True
            sd.overlay.sculpt_mode_mask_opacity = 0.75
        rep(self, {'INFO'}, "Mask brush: paint the areas to protect (Ctrl+drag erases), the mask remains on the surface")
        return {'FINISHED'}


class ANAPLAST_OT_protect_save(bpy.types.Operator):
    """Store the painted mask as protected, and return to Object Mode"""
    bl_idname = "anaplast.protect_save"
    bl_label = "Save Protected"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'SCULPT' and context.sculpt_object is not None

    def execute(self, context):
        obj = context.sculpt_object
        bpy.ops.object.mode_set(mode='OBJECT')
        n = mu.bake_sculpt_mask_to_protect(obj)
        rep(self, {'INFO'}, f"{n:,} vertices protected on {obj.name}" if n else "Nothing painted — no areas protected")
        return {'FINISHED'}


class ANAPLAST_OT_protect_clear(bpy.types.Operator):
    """Remove all protected areas"""
    bl_idname = "anaplast.protect_clear"
    bl_label = "Clear Protected"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from .sculpt_surface import target, activate
        obj = target(context)
        if obj is not None:
            a = obj.data.attributes.get(mu.PROTECT)
            if a:
                obj.data.attributes.remove(a)
        return {'FINISHED'}


class ANAPLAST_OT_fill_holes(bpy.types.Operator):
    """Fill every internal hole in the working surface (scan dropouts, torn spots) so it is one clean sheet with a single outer margin. Runs automatically before seating, building and molding too"""
    bl_idname = "anaplast.fill_holes"
    bl_label = "Fill holes"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        from .shell import source_surface
        return source_surface(context.scene.anaplast) is not None

    def execute(self, context):
        from .shell import source_surface
        obj = source_surface(context.scene.anaplast)
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        holes, edges = mu.fill_internal_holes(obj)
        n, nonman, boundary, loose = mu.mesh_stats(obj)
        rep(self, {'INFO'}, f"{obj.name}: {holes} hole(s) filled ({edges} edges); one outer margin of {boundary} edges remains" if holes
            else f"{obj.name}: no internal holes — one outer margin of {boundary} edges")
        return {'FINISHED'}


_classes = (ANAPLAST_OT_crop_cancel, ANAPLAST_OT_fill_holes, ANAPLAST_OT_lasso_crop_start, ANAPLAST_OT_crop_keep_selected, ANAPLAST_OT_protect_paint, ANAPLAST_OT_protect_save, ANAPLAST_OT_protect_clear,)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)


class ANAPLAST_OT_even_out(bpy.types.Operator):
    """Even out the cropped surface to a uniform edge length — clean topology for sculpting, margin kept open"""
    bl_idname = "anaplast.even_out"
    bl_label = "Even out mesh"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        from .shell import source_surface
        return source_surface(context.scene.anaplast) is not None

    def execute(self, context):
        from .shell import source_surface
        p = context.scene.anaplast
        obj = source_surface(p)
        b0, b1 = mu.even_out(obj, p.source_edge)
        n, nonman, boundary, loose = mu.mesh_stats(obj)
        rep(self, {'INFO'}, f"{obj.name}: {b0:,} → {b1:,} faces at ~{p.source_edge:.2f} mm; margin still open ({boundary} edges)")
        return {'FINISHED'}


from .sculpt_prepare import ANAPLAST_OT_prepare_sculpt
_even = (ANAPLAST_OT_even_out,ANAPLAST_OT_prepare_sculpt)
_reg_e, _unreg_e = register, unregister


def register():
    _reg_e()
    for c in _even:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_even):
        bpy.utils.unregister_class(c)
    _unreg_e()


class ANAPLAST_OT_subdivide(bpy.types.Operator):
    """Subdivide the working surface so it can carry fine texture — do this after sculpting, before texturing"""
    bl_idname = "anaplast.subdivide"
    bl_label = "Subdivide for texture"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        from .shell import current_prototype, source_surface
        p = context.scene.anaplast
        return (current_prototype(p) or source_surface(p)) is not None

    def execute(self, context):
        from .shell import current_prototype, source_surface
        p = context.scene.anaplast
        obj = current_prototype(p) or source_surface(p)
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        before = len(obj.data.polygons)
        levels = p.subdiv_levels

        def setup(m):
            m.levels = levels
            m.render_levels = levels
            m.subdivision_type = 'SIMPLE'
        mu.apply_modifier(obj, 'SUBSURF', setup)
        for pg in obj.data.polygons:
            pg.use_smooth = True
        n, mean, lo, hi = mu.fidelity(obj)
        rep(self, {'INFO'}, f"{obj.name}: {before:,} → {n:,} faces, mean edge {mean:.3f} mm — ready for texture")
        return {'FINISHED'}


_sub = (ANAPLAST_OT_subdivide,)
_reg_s, _unreg_s = register, unregister


def register():
    _reg_s()
    for c in _sub:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_sub):
        bpy.utils.unregister_class(c)
    _unreg_s()
