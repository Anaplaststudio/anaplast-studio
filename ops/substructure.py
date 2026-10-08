"""Generic manually retained substructure bounded by both source surfaces."""
import os
import json
import bpy
import bmesh
import numpy as np
from mathutils import Vector,Matrix
from mathutils.bvhtree import BVHTree
from ..utils import mesh as mu
from .report import rep

COLLECTION='Anaplast_Substructure'
POINTS='Anaplast_Substructure_Points'


def sources(p):
    from .shell import source_surface
    return p.sub_outer_obj or source_surface(p),p.sub_fit_obj or p.face_scan_obj


def markers():
    col=bpy.data.collections.get(POINTS)
    p=bpy.context.scene.anaplast
    return sorted([o for o in col.objects if o.name in bpy.context.scene.objects and o.get('generic_component_marker') and (bool(o.get('painted_component'))==(p.sub_component_source=='PAINT'))],key=lambda o:o.name) if col else []


def offset_solid(obj,distance,voxel,*,surface_level=.1):
    """Offset a solid; callers can request the exact zero SDF isosurface.

    The legacy default is retained for existing substructure workflows.
    Airway construction explicitly uses zero to avoid cumulative dilation.
    """
    group=bpy.data.node_groups.new('Substructure offset','GeometryNodeTree')
    try:
        group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
        group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
        a=group.nodes.new('NodeGroupInput');z=group.nodes.new('NodeGroupOutput')
        grid=group.nodes.new('GeometryNodeMeshToSDFGrid');grid.inputs['Voxel Size'].default_value=voxel
        off=group.nodes.new('GeometryNodeSDFGridOffset');off.inputs['Distance'].default_value=distance
        mesh=group.nodes.new('GeometryNodeGridToMesh');mesh.inputs['Adaptivity'].default_value=0.
        mesh.inputs['Threshold'].default_value=surface_level
        for src,dst in ((a.outputs['Geometry'],grid.inputs['Mesh']),(grid.outputs['SDF Grid'],off.inputs['Grid']),(off.outputs['Grid'],mesh.inputs['Grid']),(mesh.outputs['Mesh'],z.inputs['Geometry'])):group.links.new(src,dst)
        mu.apply_modifier(obj,'NODES',lambda m:setattr(m,'node_group',group))
    finally:bpy.data.node_groups.remove(group)


def frame_for(context):
    from .mold_auto import view_axes
    mold=bpy.data.objects.get('Mold_Base')
    if mold and 'mold_up' in mold:
        u=Vector(mold['mold_up']);r=Vector(mold['mold_right']);return u,r,u.cross(r).normalized()
    return view_axes(context)


def surface_tree(obj):
    bm=bmesh.new()
    try:
        bm.from_mesh(obj.data);bm.transform(obj.matrix_world)
        return BVHTree.FromBMesh(bm)
    finally:bm.free()


def verify_clearance(obj,outer,fit):
    # Check every vertex and every triangle centre, including pocket walls.
    obj.data.calc_loop_triangles()
    points=[obj.matrix_world@v.co for v in obj.data.vertices]
    points.extend(obj.matrix_world@t.center for t in obj.data.loop_triangles)
    result=[]
    for ref in (outer,fit):
        tree=surface_tree(ref)
        result.append(min(tree.find_nearest(p)[3] for p in points))
    return result


def clean_grid_fragments(obj,voxel):
    """Discard isolated few-vertex remnants smaller than one grid cell."""
    bm=bmesh.new()
    try:
        bm.from_mesh(obj.data);unseen=set(bm.verts);discard=[];components=0;removed=0
        while unseen:
            verts={unseen.pop()};stack=list(verts)
            while stack:
                v=stack.pop()
                for edge in v.link_edges:
                    other=edge.other_vert(v)
                    if other in unseen:unseen.remove(other);verts.add(other);stack.append(other)
            span=max(max(v.co[i] for v in verts)-min(v.co[i] for v in verts) for i in range(3))
            if len(verts)<=8 and span<voxel:
                discard.extend(verts);removed+=1
            else:components+=1
        if discard:
            bmesh.ops.delete(bm,geom=discard,context='VERTS');bm.to_mesh(obj.data);obj.data.update()
        return components,removed
    finally:bm.free()


def footprint_polygons(component_points,frame,width,length,separate,gap):
    from .mold_auto import convex_hull_2d
    u,r,f=map(np.asarray,frame);centres=np.array([[q.dot(Vector(r)),q.dot(Vector(f))] for q in component_points])
    rectangles=[c+np.array([[-width/2,-length/2],[width/2,-length/2],[width/2,length/2],[-width/2,length/2]]) for c in centres]
    envelope=convex_hull_2d(np.vstack(rectangles))
    if not separate:return [envelope]
    result=[]
    # Divide the shared fitted envelope, not individual component squares.
    # Footprint size changes the outside boundary, never the split planes.
    for i in range(len(centres)):
        poly=np.asarray(envelope,float)
        for j,c in enumerate(centres):
            if i==j:continue
            direction=c-centres[i];distance=np.linalg.norm(direction)
            if distance<=gap:raise RuntimeError('Component centres are too close for separate pieces; move the paint patches or reduce the gap')
            normal=direction/distance;limit=(c+centres[i])*.5-normal*gap*.5;clipped=[]
            for a,b in zip(poly,np.roll(poly,-1,axis=0)):
                da=(limit-a)@normal;db=(limit-b)@normal
                if da>=0:clipped.append(a)
                if (da>=0)!=(db>=0):clipped.append(a+(b-a)*da/(da-db))
            poly=np.asarray(clipped)
            if len(poly)<3:raise RuntimeError('No footprint remains for a separate component')
        result.append(poly.tolist())
    return result


def results():
    return sorted([o for o in bpy.context.scene.objects if o.get('anaplast_part')=='SUBSTRUCTURE'],key=lambda o:o.name)


def make_substructure(context):
    from . import mold_auto as ma
    p=context.scene.anaplast;outer,fit=sources(p)
    if outer is None or fit is None:raise RuntimeError('Assign the aligned outer sculpt and fitting scan')
    if outer==fit:raise RuntimeError('Outer sculpt and fitting scan must be different surfaces')
    from . import component_paint
    if p.sub_component_source=='PAINT':component_paint.update_markers(context);component_paint.restore(context)
    positions=markers()
    if p.sub_layout!='FULL' and not positions:raise RuntimeError('Paint component patches or choose existing component markers')
    ocular=p.sub_ocular_obj or p.ocular_obj if p.sub_ocular_relief else None
    if p.sub_ocular_relief and ocular is None:raise RuntimeError('Assign an ocular model for the relief')
    if ocular and not mu.is_closed(ocular):raise RuntimeError('Ocular relief needs a closed ocular model')
    col=mu.get_collection(context.scene,COLLECTION);temp=[];result=None
    try:
        frame=frame_for(context);up=frame[0]
        lo,hi=mu.world_bbox(outer);reach=(hi-lo).length+30.
        # The closed head/scan has an underside. Restrict the candidate to the
        # facing tissue region so subtraction cannot leave a second core below
        # that underside (particularly with finite cast scans).
        u,r,f=map(np.asarray,frame)
        outer_world=np.array([outer.matrix_world@v.co for v in outer.data.vertices])
        step=max(1,len(outer_world)//6000)
        hull=ma.convex_hull_2d(list(zip((outer_world@r)[::step],(outer_world@f)[::step])))
        fit_world=np.array([fit.matrix_world@v.co for v in fit.data.vertices])
        normal_matrix=fit.matrix_world.to_3x3().inverted().transposed()
        normals=np.array([normal_matrix@v.normal for v in fit.data.vertices])
        faces=[fit.data.polygons[i] for i in range(0,len(fit.data.polygons),max(1,len(fit.data.polygons)//20000))]
        fit_world=np.vstack((fit_world,np.array([fit.matrix_world@face.center for face in faces])))
        normals=np.vstack((normals,np.array([normal_matrix@face.normal for face in faces])))
        d=ma.dist_to_polygon(np.column_stack((fit_world@r,fit_world@f)),hull)
        height=fit_world@u
        valid=(d<=5.)&(height>=float((outer_world@u).min())-p.m_depth_limit)&(normals@u>-.1)
        if not valid.any():raise RuntimeError('No facing fitting surface beneath the aligned sculpt')
        floor=float(height[valid].min())-.01
        # Only temporary solids are repaired or smoothed. The sculpt stays intact.
        core=ma.sculpt_solid_toward('Sub_Work_outer',outer,up,reach,(lo,hi),col,margin=10.,max_faces=max(250000,len(outer.data.polygons)*2));temp.append(core)
        stats=mu.mesh_stats(fit)
        if stats[0] and not any(stats[1:]):
            relief=mu.duplicate_object(fit,'Sub_Work_fit',col)
        else:
            bm=mu.scan_region_bmesh(fit,(lo,hi),15.,p.scan_flip)
            patch=mu.new_mesh_object('Sub_Work_patch',bm,col);temp.append(patch)
            relief=ma.thicken_into(patch,up,reach,'Sub_Work_fit',col)
        temp.append(relief)
        ceiling=mu.duplicate_object(relief,'Sub_Work_height',col);temp.append(ceiling)
        voxel=max(mu.safe_voxel(core,p.sub_resolution),mu.safe_voxel(relief,p.sub_resolution))
        # Reserve a numerical allowance for the sampled grid, then measure the
        # final result against the ORIGINAL references below before publishing.
        allowance=voxel
        offset_solid(core,-(p.sub_silicone+allowance),voxel)
        offset_solid(relief,p.sub_relief+allowance,voxel)
        offset_solid(ceiling,p.sub_relief+p.sub_height-allowance,voxel)
        if not len(core.data.polygons):raise RuntimeError('No space remains inside the requested silicone cover')
        crop=ma.prism('Sub_Work_crop',ma.offset_polygon(hull,10.),floor,float((outer_world@u).max())+10.,frame,col);temp.append(crop)
        ma.cut(core,crop,'INTERSECT',voxel)
        allowed=mu.duplicate_object(core,'Sub_Work_allowed',col);temp.append(allowed)
        ma.cut(allowed,relief,'DIFFERENCE',voxel)
        ma.cut(allowed,ceiling,'INTERSECT',voxel)
        if ocular:
            ocular_tool=mu.duplicate_object(ocular,'Sub_Work_ocular',col);temp.append(ocular_tool)
            offset_solid(ocular_tool,p.sub_ocular_gap+allowance,voxel);ma.cut(allowed,ocular_tool,'DIFFERENCE',voxel)
        if not len(allowed.data.polygons):raise RuntimeError('No room between fitting relief and silicone cover; check the alignment or reduce the clearances')
        if p.sub_smooth:
            def smooth(m):m.factor=p.sub_smooth;m.iterations=15
            smoothed=mu.duplicate_object(allowed,'Sub_Work_smooth',col);temp.append(smoothed)
            mu.apply_modifier(smoothed,'SMOOTH',smooth);ma.cut(smoothed,allowed,'INTERSECT',voxel);allowed=smoothed
        polygons=footprint_polygons([o.matrix_world.translation for o in positions],frame,p.sub_width,p.sub_length,p.sub_layout=='SEPARATE',p.sub_piece_gap) if p.sub_layout!='FULL' else [None]
        pending=[];records=[]
        def volume(obj):
            bm=bmesh.new()
            try:bm.from_mesh(obj.data);return abs(bm.calc_volume())
            finally:bm.free()
        for number,polygon in enumerate(polygons):
            result=mu.duplicate_object(allowed,'Sub_Work_result',col);temp.append(result)
            if polygon is not None:
                footprint=ma.prism('Sub_Work_footprint',polygon,floor-1.,float((outer_world@u).max())+10.,frame,col);temp.append(footprint)
                ma.cut(result,footprint,'INTERSECT',voxel)
            if not len(result.data.polygons):raise RuntimeError(f'Component {number+1} has no space inside the requested footprint, height and clearances')
            pocket_count=0;local_markers=[positions[number]] if p.sub_layout=='SEPARATE' else positions
            if p.sub_pockets:
                for marker in local_markers:
                    axis=(marker.matrix_world.to_3x3()@Vector((0,0,1))).normalized();point=marker.matrix_world.translation
                    start=point+axis*(p.sub_relief-2.);end=point+axis*(p.sub_relief+p.sub_component_depth)
                    cutter=ma.cylinder('Sub_Work_pocket',(start+end)*.5,axis,(p.sub_component_diameter+2*p.sub_component_clearance)*.5,(end-start).length,col,segments=64);temp.append(cutter)
                    before_volume=volume(result);ma.cut(result,cutter,'DIFFERENCE',voxel)
                    if before_volume-volume(result)<1e-5:raise RuntimeError(f'{marker.name} does not cut the substructure; adjust its painted position, height, or space depth')
                    pocket_count+=1
            components,removed=clean_grid_fragments(result,voxel);stats=mu.mesh_stats(result)
            if not stats[0] or any(stats[1:]):raise RuntimeError('Substructure does not form a closed mesh; no previous result was replaced')
            if p.sub_layout!='FULL' and components!=1:raise RuntimeError('The available space splits this substructure. Increase its footprint/height or choose Separate pieces; no previous result was replaced')
            cover,clearance=verify_clearance(result,outer,fit)
            if cover+1e-4<p.sub_silicone or clearance+1e-4<p.sub_relief:raise RuntimeError(f'Clearance verification failed: cover {cover:.3f}, fitting {clearance:.3f} mm; reduce Detail spacing and rebuild')
            ocular_distance=None
            if ocular:
                ocular_distance=verify_clearance(result,ocular,ocular)[0]
                if ocular_distance+1e-4<p.sub_ocular_gap:raise RuntimeError('Ocular clearance check failed; reduce Detail spacing and rebuild')
            result['anaplast_part']='SUBSTRUCTURE';result['generic_manual_components']=True
            result['fitting_relief_mm']=p.sub_relief;result['silicone_cover_mm']=p.sub_silicone
            result['checked_min_cover_mm']=cover;result['checked_min_relief_mm']=clearance
            result['component_spaces']=pocket_count;result['disconnected_regions']=components;result['removed_grid_fragments']=removed
            result['source_outer']=outer.name;result['source_fit']=fit.name;result['layout']=p.sub_layout
            if p.sub_layout=='SEPARATE':result['separation_mode']='IMPLANT_MIDPLANES'
            result['requested_footprint_mm']=[p.sub_width,p.sub_length];result['maximum_height_mm']=p.sub_height
            result['ocular_relief_enabled']=bool(ocular)
            if ocular:result['checked_ocular_clearance_mm']=ocular_distance;result['ocular_clearance_mm']=p.sub_ocular_gap
            for face in result.data.polygons:face.use_smooth=True
            result.color=(.3,.65,.9,1)
            pending.append(result);records.append({'cover':cover,'relief':clearance,'spaces':pocket_count,'regions':components,'ocular_clearance':ocular_distance})
        # Publish every piece together only after all requested components pass.
        for old in results():
            if old not in pending:mu.delete_object(old)
        for i,result in enumerate(pending):
            result.name='Substructure' if i==0 else f'Substructure_{i+1:02d}';result.data.name=result.name;temp.remove(result)
        p.sub_report=f'{len(pending)} piece(s); cover {min(v["cover"] for v in records):.2f} mm; relief {min(v["relief"] for v in records):.2f} mm. Footprint and height clipped to available space.'
        for o in context.selected_objects:o.select_set(False)
        for result in pending:result.hide_set(False);result.select_set(True)
        context.view_layer.objects.active=pending[0]
        return pending[0]
    finally:
        for obj in reversed(temp):
            if obj.name in bpy.data.objects:mu.delete_object(obj)


class ANAPLAST_OT_substructure_marker(bpy.types.Operator):
    bl_idname='anaplast.substructure_marker';bl_label='Mark Component at Cursor';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        outer,fit=sources(context.scene.anaplast)
        if fit is None:rep(self,{'ERROR'},'Assign the fitting scan');return {'CANCELLED'}
        point=context.scene.cursor.location.copy();hit=surface_tree(fit).find_nearest(point)
        if hit[0] is None:rep(self,{'ERROR'},'No fitting surface near the cursor');return {'CANCELLED'}
        col=mu.get_collection(context.scene,POINTS)
        marker=bpy.data.objects.new('Component_Position',None);col.objects.link(marker)
        marker.empty_display_type='ARROWS';marker.empty_display_size=4.
        marker.matrix_world=Matrix.Translation(hit[0])@hit[1].to_track_quat('Z','Y').to_matrix().to_4x4()
        marker['generic_component_marker']=True;marker.show_in_front=True
        for o in context.selected_objects:o.select_set(False)
        marker.select_set(True);context.view_layer.objects.active=marker
        rep(self,{'INFO'},'Generic position marked. Move or rotate its local Z axis to match the component insertion direction')
        return {'FINISHED'}


class ANAPLAST_OT_substructure_view(bpy.types.Operator):
    bl_idname='anaplast.substructure_view';bl_label='Show / hide sculpt';bl_options={'REGISTER','UNDO'}
    target:bpy.props.EnumProperty(items=[('SUB','Substructure',''),('WEDGE','Wedge boundary','')],default='SUB')
    def execute(self,context):
        from . import mold_auricular
        outer=mold_auricular.reference(context.scene.anaplast) if self.target=='WEDGE' else sources(context.scene.anaplast)[0]
        if outer is None:rep(self,{'ERROR'},'Assign the outer sculpt first');return {'CANCELLED'}
        if outer.name not in context.view_layer.objects:rep(self,{'ERROR'},'Enable the collection containing the sculpt first');return {'CANCELLED'}
        hidden=outer.visible_get()
        if not hidden:outer.hide_viewport=False
        outer.hide_set(hidden)
        # Preserve an explicit visibility choice when painting later restores
        # the scene. Source shape, selection and painting target are unchanged.
        for key in ('component_visibility','wedge_coverage_visibility'):
            stored=context.scene.get(key)
            if stored:
                visibility=json.loads(stored);visibility[outer.name]=hidden;context.scene[key]=json.dumps(visibility)
        return {'FINISHED'}


class ANAPLAST_OT_substructure_build(bpy.types.Operator):
    bl_idname='anaplast.substructure_build';bl_label='Build Generic Substructure';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:make_substructure(context)
        except Exception as e:rep(self,{'ERROR'},str(e));return {'CANCELLED'}
        rep(self,{'INFO'},context.scene.anaplast.sub_report);return {'FINISHED'}


class ANAPLAST_OT_substructure_export(bpy.types.Operator):
    bl_idname='anaplast.substructure_export';bl_label='Export Substructure STL'
    @classmethod
    def poll(cls,context):return bool(results())
    def execute(self,context):
        objects=results()
        if any(any(mu.mesh_stats(o)[1:]) for o in objects):rep(self,{'ERROR'},'Substructure has open or non-manifold geometry');return {'CANCELLED'}
        folder=bpy.path.abspath(context.scene.anaplast.export_dir)
        if not folder:rep(self,{'ERROR'},'Choose the export folder in Export & Save');return {'CANCELLED'}
        try:
            os.makedirs(folder,exist_ok=True)
            for obj in objects:
                path=os.path.join(folder,obj.name+'.stl');mu.write_binary_stl(obj,path)
        except OSError as e:rep(self,{'ERROR'},str(e));return {'CANCELLED'}
        rep(self,{'INFO'},f'Exported {len(objects)} substructure STL file(s) to '+folder);return {'FINISHED'}


_classes=(ANAPLAST_OT_substructure_view,ANAPLAST_OT_substructure_marker,ANAPLAST_OT_substructure_build,ANAPLAST_OT_substructure_export)
def register():
    for cls in _classes:bpy.utils.register_class(cls)
def unregister():
    for cls in reversed(_classes):bpy.utils.unregister_class(cls)
