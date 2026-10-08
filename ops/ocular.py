"""Traced scan-based ocular shells with explicitly modelled corneal definition."""
import math
import numpy as np
import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from ..utils import mesh as mu
from .report import rep

COLLECTION='Anaplast_Ocular'

def reference(p):
    return p.ocular_scan or bpy.data.objects.get('Mirror_FaceScan')

def signed_distance(points,poly):
    from .lasso import points_in_polygon
    points=np.asarray(points);poly=np.asarray(poly);distance=np.full(len(points),np.inf)
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        e=b-a;t=np.clip((points-a)@e/max(e@e,1e-12),0,1)
        distance=np.minimum(distance,np.linalg.norm(points-a-t[:,None]*e,axis=1))
    return np.where(points_in_polygon(points[:,0],points[:,1],poly),-distance,distance)

def radial_outline(poly,extension,count=256,center=None,directions=None):
    poly=np.asarray(poly,float);center=poly.mean(axis=0) if center is None else np.asarray(center,float)
    if signed_distance(center[None],poly)[0]>=0:raise RuntimeError('Trace must surround its center; retrace the eyelid opening')
    # A single crossing per ray is required. Do not silently replace a traced
    # eyelid with its convex hull or bridge a folded/self-crossing trace.
    directions=np.column_stack((np.cos(np.arange(count)*2*math.pi/count),np.sin(np.arange(count)*2*math.pi/count))) if directions is None else np.asarray(directions,float)
    count=len(directions)
    e=np.roll(poly,-1,axis=0)-poly;a=poly-center
    crossings=np.zeros(count,dtype=int)
    for origin,edge in zip(a,e):
        det=directions[:,0]*edge[1]-directions[:,1]*edge[0]
        safe=np.abs(det)>1e-10
        t=np.divide(origin[0]*edge[1]-origin[1]*edge[0],det,out=np.zeros(count),where=safe)
        q=np.divide(origin[0]*directions[:,1]-origin[1]*directions[:,0],det,out=np.zeros(count),where=safe)
        crossings+=safe&(t>0)&(q>=-1e-9)&(q<1-1e-9)
    if np.any(crossings!=1):raise RuntimeError('The aperture trace folds across itself or has a deep indentation; adjust the trace')
    lo=np.zeros(count);hi=np.full(count,np.linalg.norm(a,axis=1).max()+extension+1.)
    for _ in range(35):
        radius=(lo+hi)*.5;d=signed_distance(center+directions*radius[:,None],poly)
        lo=np.where(d<extension,radius,lo);hi=np.where(d>=extension,radius,hi)
    return center,center+directions*((lo+hi)*.5)[:,None]

def build(context):
    if context.scene.ocular_production.surface_method=='COPY' and context.scene.anaplast.ocular_trace and context.scene.anaplast.ocular_trace.get('ocular_marked_source'):
        from .ocular_extract import build as copy_surface
        return copy_surface(context)
    p=context.scene.anaplast;scan=reference(p);trace=p.ocular_trace
    smooth_mode=context.scene.ocular_production.surface_method=='SMOOTH' and trace is not None and bool(trace.get('ocular_marked_source'))
    fit_metrics={}
    if smooth_mode and context.scene.ocular_production.match_scan_iris:
        if not trace or not trace.get('iris_fit_diameter_mm'):raise RuntimeError('Fit the highlighted iris first to record its scan diameter')
        p.ocular_iris_diameter=trace['iris_fit_diameter_mm'];p.ocular_cornea_diameter=p.ocular_iris_diameter
    if scan is None or scan.type!='MESH':raise RuntimeError('Assign the mirror eye scan')
    if trace is None or trace.type!='CURVE' or not trace.data.splines:raise RuntimeError('Trace the eyelid aperture first')
    spline=trace.data.splines[0]
    if spline.type!='POLY' or not spline.use_cyclic_u:raise RuntimeError('Use a closed polygon aperture trace')
    u=Vector(trace['ocular_up']);r=Vector(trace['ocular_right']);f=u.cross(r).normalized()
    world=np.array([trace.matrix_world@Vector(v.co[:3]) for v in spline.points]);poly=np.column_stack((world@r,world@f))
    center,boundary=radial_outline(poly,p.ocular_extension)
    marker=bpy.data.objects.get('Ocular_Cornea_Center')
    cc=np.array((marker.matrix_world.translation.dot(r),marker.matrix_world.translation.dot(f))) if marker else center
    if smooth_mode and context.scene.ocular_production.match_scan_iris:
        measured=Vector(trace['iris_fit_center_world']);cc=np.array((measured.dot(r),measured.dot(f)))
    from mathutils.geometry import delaunay_2d_cdt
    lo=boundary.min(axis=0);hi=boundary.max(axis=0);step=p.ocular_resolution
    X,Y=np.meshgrid(np.arange(lo[0],hi[0]+step,step),np.arange(lo[1],hi[1]+step,step))
    grid=np.column_stack((X.ravel(),Y.ravel()));grid=grid[signed_distance(grid,boundary)<-.6*step]
    # Give the visual iris a circular mesh boundary instead of coloring whole
    # triangles across its edge, which produced a visibly jagged guide.
    ir=p.ocular_iris_diameter*.5;angles=np.arange(128)*2*math.pi/128
    iris=cc+ir*np.column_stack((np.cos(angles),np.sin(angles)))
    if np.any(signed_distance(iris,boundary)>=0):raise RuntimeError('The iris guide extends beyond the ocular outline; reduce its diameter or move its center')
    grid=grid[np.abs(np.linalg.norm(grid-cc,axis=1)-ir)>.45*step]
    start=len(boundary);edges=[(start+i,start+(i+1)%128) for i in range(128)]
    points=np.vstack((boundary,iris,grid))
    output=delaunay_2d_cdt([Vector(q) for q in points],edges,[list(range(len(boundary)))],1,1e-6,False)
    xy=np.array(output[0]);triangles=output[2]
    bm=bmesh.new();bm.from_mesh(scan.data);bm.transform(scan.matrix_world);tree=BVHTree.FromBMesh(bm)
    zs=[v.co.dot(u) for v in bm.verts];top=max(zs)+10;reach=top-min(zs)+10;bm.free()
    distances=signed_distance(xy,poly);sampled=np.full(len(xy),np.nan)
    for i in np.flatnonzero(distances<=0):
        hit=tree.ray_cast(r*xy[i,0]+f*xy[i,1]+u*top,-u,reach)[0]
        if hit is not None:sampled[i]=hit.dot(u)
    inside=distances<-.3;valid=inside&np.isfinite(sampled)
    if valid.sum()<30 or valid.sum()<.9*inside.sum():raise RuntimeError('Too little scan surface inside the aperture; check the scan, trace, and viewing direction')
    if smooth_mode:
        from .ocular_smooth import marked_fit
        height,fit_metrics=marked_fit(context,xy,np.array(u),np.array(r),np.array(f),cc)
        fitted=height
    else:
        q=xy-center;x,y=q.T
        A=np.column_stack((np.ones(len(x)),x,y,x*x,x*y,y*y));coef=np.linalg.lstsq(A[valid],sampled[valid],rcond=None)[0]
        # Use a convex, smooth continuation beneath the lids. Scan detail remains
        # inside the opening; the 3 mm hidden extension does not copy the eyelids.
        curvature=np.array([[2*coef[3],coef[4]],[coef[4],2*coef[5]]]);eig,rot=np.linalg.eigh(curvature)
        curvature=rot@np.diag(np.minimum(eig,-.002))@rot.T
        coef[3]=curvature[0,0]*.5;coef[4]=curvature[0,1];coef[5]=curvature[1,1]*.5
        coef[:3]=np.linalg.lstsq(A[valid,:3],sampled[valid]-A[valid,3:]@coef[3:],rcond=None)[0]
        fitted=A@coef;blend=np.clip(-distances/1.,0,1);blend=blend*blend*(3-2*blend)
        height=fitted+np.where(np.isfinite(sampled),np.nan_to_num(sampled)-fitted,0)*blend*p.ocular_detail
        rho=np.linalg.norm(xy-cc,axis=1);radius=p.ocular_cornea_diameter*.5
        if p.ocular_cornea_projection>0 and signed_distance(cc[None],boundary)[0]>-radius:
            raise RuntimeError('The corneal footprint crosses the extended ocular outline; adjust extension, center or diameter')
        height+=p.ocular_cornea_projection*np.maximum(0,1-(rho/radius)**2)**2
    coords=xy[:,0,None]*np.array(r)+xy[:,1,None]*np.array(f)+height[:,None]*np.array(u)
    bm=bmesh.new();pair=bm.verts.layers.int.new('ocular_back_index');verts=[bm.verts.new(v) for v in coords]
    for triangle in triangles:bm.faces.new([verts[i] for i in triangle])
    bm.normal_update()
    layer=bm.faces.layers.int.new('ocular_front')
    for face in bm.faces:face[layer]=1
    rim=[tuple(e.verts) for e in bm.edges if e.is_boundary]
    inner=[bm.verts.new(v.co-v.normal*p.ocular_thickness) for v in verts]
    bm.verts.index_update();mapping={v:i for i,v in enumerate(verts)}
    for face in list(bm.faces):bm.faces.new([inner[mapping[v]] for v in reversed(face.verts)])
    for va,vb in rim:
        a=mapping[va];b=mapping[vb];bm.faces.new((verts[a],inner[a],inner[b],verts[b]))
    bm.verts.index_update()
    for a,b in zip(verts,inner):a[pair]=b.index+1
    bmesh.ops.recalc_face_normals(bm,faces=bm.faces)
    bmesh.ops.triangulate(bm,faces=bm.faces)
    obj=mu.new_mesh_object('Ocular_Work',bm,mu.get_collection(context.scene,COLLECTION))
    try:
        from .mold_design import has_self_intersections
        if any(mu.mesh_stats(obj)[1:]) or has_self_intersections(obj):raise RuntimeError('Shell folds into itself; reduce thickness or corneal projection, or lower scan-detail retention')
        old=bpy.data.objects.get('Ocular_Shell')
        if old:mu.delete_object(old)
        obj.name='Ocular_Shell';obj['anaplast_part']='OCULAR';obj['source_scan']=scan.name
        context.scene.anaplast.ocular_obj=obj
        for name,color in (('Ocular sclera',(.8,.8,.74,1)),('Ocular iris guide',(.16,.30,.32,1))):
            material=bpy.data.materials.get(name) or bpy.data.materials.new(name);material.diffuse_color=color;obj.data.materials.append(material)
        for face in obj.data.polygons:
            point=face.center;d=np.linalg.norm(np.array((point.dot(r),point.dot(f)))-cc)
            if face.normal.dot(u)>0 and d<=p.ocular_iris_diameter*.5:face.material_index=1
        obj['aperture_extension_mm']=p.ocular_extension;obj['normal_offset_mm']=p.ocular_thickness
        obj['cornea_projection_mm']=0. if smooth_mode else p.ocular_cornea_projection;obj['cornea_diameter_mm']=p.ocular_cornea_diameter
        obj['iris_diameter_mm']=p.ocular_iris_diameter
        obj['impression_fitted']=False;obj['scan_detail']=0. if smooth_mode else p.ocular_detail
        if smooth_mode:
            for key,value in fit_metrics.items():obj[key]=value
            obj['front_method']='Smooth sphere fitted to the marked scan'
            obj['iris_matches_marked_scan']=context.scene.ocular_production.match_scan_iris
        obj['ocular_up']=list(u);obj['ocular_right']=list(r)
        obj['ocular_iris_center']=list(cc[0]*np.array(r)+cc[1]*np.array(f))
        obj['continuation_fit_rms_mm']=float(np.sqrt(np.mean((fitted[valid]-sampled[valid])**2)))
        for face in obj.data.polygons:face.use_smooth=True
        obj.color=(.88,.88,.8,1)
        for other in context.selected_objects:other.select_set(False)
        obj.select_set(True);context.view_layer.objects.active=obj
        p.ocular_report=f'Closed shell; {p.ocular_extension:g} mm extension; {p.ocular_thickness:g} mm nominal offset'
        if smooth_mode:p.ocular_report=f"Smooth scan fit: radius {fit_metrics['fitted_sclera_radius_mm']:.2f} mm; iris {p.ocular_iris_diameter:.2f} mm"
        return obj
    except Exception:
        mu.delete_object(obj);raise

def save_trace(context,points,frame):
    p=context.scene.anaplast;old=p.ocular_trace
    cu=bpy.data.curves.new('Ocular_Aperture','CURVE');cu.dimensions='3D';cu.bevel_depth=.08
    spline=cu.splines.new('POLY');spline.points.add(len(points)-1)
    for v,co in zip(spline.points,points):v.co=(*co,1)
    spline.use_cyclic_u=True
    obj=bpy.data.objects.new('Ocular_Aperture',cu);mu.get_collection(context.scene,COLLECTION).objects.link(obj)
    obj['ocular_up']=list(frame[0]);obj['ocular_right']=list(frame[1]);obj.show_in_front=True
    p.ocular_trace=obj
    if old and old.get('ocular_up') is not None:mu.delete_object(old)
    obj.name='Ocular_Aperture'
    return obj

class ANAPLAST_OT_ocular_trace(bpy.types.Operator):
    """Click around the visible eyelid opening; Enter closes it, Backspace removes a point, Esc cancels"""
    bl_idname='anaplast.ocular_trace';bl_label='Trace Eyelid Aperture';bl_options={'REGISTER','UNDO'}
    def invoke(self,context,event):
        from .mold_auto import view_axes
        scan=reference(context.scene.anaplast)
        if scan is None:rep(self,{'ERROR'},'Assign the mirror eye scan');return {'CANCELLED'}
        self.region=next((r for r in context.area.regions if r.type=='WINDOW'),None)
        self.rv=context.space_data.region_3d
        q=self.rv.view_rotation;self.frame=(q@Vector((0,0,1)),q@Vector((1,0,0)))
        self.scan=scan;self.points=[];self.screen=[]
        self.handle=bpy.types.SpaceView3D.draw_handler_add(self.draw_trace,(), 'WINDOW','POST_PIXEL')
        context.area.header_text_set('Click aperture boundary · Enter finish · Backspace undo point · Esc cancel')
        context.window.cursor_modal_set('CROSSHAIR');context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    def draw_trace(self):
        if len(self.screen)<2:return
        import gpu
        from gpu_extras.batch import batch_for_shader
        shader=gpu.shader.from_builtin('UNIFORM_COLOR');shader.bind();shader.uniform_float('color',(1,.5,.1,1))
        batch_for_shader(shader,'LINE_STRIP',{'pos':self.screen+[self.screen[0]]}).draw(shader)
    def finish(self,context):
        bpy.types.SpaceView3D.draw_handler_remove(self.handle,'WINDOW');context.area.header_text_set(None)
        context.window.cursor_modal_restore();context.area.tag_redraw()
    def modal(self,context,event):
        from bpy_extras import view3d_utils
        if event.type in {'ESC','RIGHTMOUSE'}:
            self.finish(context);return {'CANCELLED'}
        if event.type=='BACK_SPACE' and event.value=='PRESS':
            if self.points:self.points.pop();self.screen.pop()
        if event.type in {'RET','NUMPAD_ENTER'} and event.value=='PRESS':
            if len(self.points)<6:rep(self,{'WARNING'},'Place at least six points around the opening');return {'RUNNING_MODAL'}
            projected=np.array([[v.dot(self.frame[1]),v.dot(self.frame[0].cross(self.frame[1]))] for v in self.points])
            try:radial_outline(projected,0)
            except RuntimeError as e:rep(self,{'WARNING'},str(e));return {'RUNNING_MODAL'}
            save_trace(context,self.points,self.frame);self.finish(context);return {'FINISHED'}
        if event.type=='LEFTMOUSE' and event.value=='PRESS':
            co=(event.mouse_x-self.region.x,event.mouse_y-self.region.y)
            if not(0<=co[0]<self.region.width and 0<=co[1]<self.region.height):return {'RUNNING_MODAL'}
            origin=view3d_utils.region_2d_to_origin_3d(self.region,self.rv,co);direction=view3d_utils.region_2d_to_vector_3d(self.region,self.rv,co)
            inv=self.scan.matrix_world.inverted();local=inv.to_3x3()@direction
            hit,point,normal,face=self.scan.ray_cast(inv@origin,local.normalized())
            if hit:self.points.append(self.scan.matrix_world@point);self.screen.append(co)
            else:rep(self,{'WARNING'},'That point did not hit the eye scan')
        context.area.tag_redraw();return {'RUNNING_MODAL'}

class ANAPLAST_OT_ocular_center(bpy.types.Operator):
    """Put the 3D cursor at the intended corneal center, then click; move the marker to adjust"""
    bl_idname='anaplast.ocular_center';bl_label='Corneal Center at Cursor';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        obj=bpy.data.objects.get('Ocular_Cornea_Center')
        if not obj:
            obj=bpy.data.objects.new('Ocular_Cornea_Center',None);mu.get_collection(context.scene,COLLECTION).objects.link(obj)
        obj.location=context.scene.cursor.location;obj.empty_display_type='CIRCLE';obj.empty_display_size=.5;obj.show_in_front=True
        return {'FINISHED'}

class ANAPLAST_OT_ocular_build(bpy.types.Operator):
    bl_idname='anaplast.ocular_build';bl_label='Build Ocular Shell';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:build(context)
        except Exception as e:rep(self,{'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}

class ANAPLAST_OT_ocular_export(bpy.types.Operator):
    bl_idname='anaplast.ocular_export';bl_label='Export Ocular STL'
    filepath:bpy.props.StringProperty(subtype='FILE_PATH')
    def invoke(self,context,event):
        self.filepath='Ocular_Shell.stl';context.window_manager.fileselect_add(self);return {'RUNNING_MODAL'}
    def execute(self,context):
        obj=bpy.data.objects.get('Ocular_Shell')
        if not obj:rep(self,{'ERROR'},'Build the ocular shell first');return {'CANCELLED'}
        mu.write_binary_stl(obj,bpy.path.ensure_ext(self.filepath,'.stl'));return {'FINISHED'}

_classes=(ANAPLAST_OT_ocular_trace,ANAPLAST_OT_ocular_center,ANAPLAST_OT_ocular_build,ANAPLAST_OT_ocular_export)
def register():
    for c in _classes:bpy.utils.register_class(c)
def unregister():
    for c in reversed(_classes):bpy.utils.unregister_class(c)
