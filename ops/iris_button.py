"""Independent photograph-based iris/corneal button. Dimensions below are mm."""
import json
import math
from pathlib import Path
import tempfile
import zipfile
from types import SimpleNamespace
import bpy
import bmesh
import numpy as np
from bpy.props import PointerProperty, FloatProperty, EnumProperty, StringProperty, BoolProperty
from . import ocular_production as prod, ocular_pupil, scene_helpers

_handle = None


class PhotoSettings:
    """Reuse photo sampling without reading/changing the orbital workflow."""
    def __init__(self, settings, scene=None):
        self.settings = settings
        if scene:
            from .eye_photo import source
            self.source=source(scene)
        else:self.source={}
        self.id_data = SimpleNamespace(ocular_photos=SimpleNamespace(
            single_image=True, mirror=self.source.get('mirror',False), preserve_photo=True))

    def __getattr__(self, name):
        if name in self.source:return self.source[name]
        return getattr(self.settings, name)


class IrisButton(bpy.types.PropertyGroup):
    photo: PointerProperty(type=prod.OcularProduction)
    diameter: FloatProperty(name='Iris / button diameter (mm)', default=12, min=6, max=18)
    dome_height: FloatProperty(name='Corneal rise (mm)', default=2.5, min=.5, max=6)
    cover: FloatProperty(name='Clear cover at rim (mm)', default=.35, min=.15, max=2)
    back: EnumProperty(name='Back shape', items=[('FLAT','Flat',''),('CONCAVE','Shallow concave','')])
    back_sag: FloatProperty(name='Back concavity (mm)', default=.4, min=.05, max=1.5)
    backing: FloatProperty(name='Backing below chamber (mm)', default=.5, min=.25, max=3)
    peg_diameter: FloatProperty(name='Peg diameter (mm)', default=2, min=.6, max=4,
        description='Adjustable design starting point, not a supplier peg standard')
    peg_height: FloatProperty(name='Peg above dome (mm)', default=5, min=1, max=15)
    peg_taper: FloatProperty(name='Peg taper (%)', default=20, min=0, max=60,
        description='Reduction in shaft diameter at the tip; zero keeps parallel sides')
    apparent_pupil: BoolProperty(name='Apparent pupil band', default=False,
        description='Fixed dark iris band fading outward from the real pupil; the opening does not physically change')
    pupil_min: FloatProperty(name='Minimum / real pupil (mm)', default=2, min=.5, max=10)
    pupil_max: FloatProperty(name='Maximum / dark band (mm)', default=4, min=.5, max=14)
    pupil_darkness: FloatProperty(name='Inner iris darkening', default=.65, min=0, max=1, subtype='FACTOR',
        description='Darken the photo near the pupil while retaining iris detail, fading to unchanged photo colors')
    pupil_outer_darkness: FloatProperty(name='Darkening at maximum', default=.30, min=0, max=1, subtype='FACTOR',
        description='Darkening still present at the maximum band diameter, before the final soft edge')
    pupil_outer_fade: FloatProperty(name='Outer soft edge (mm)', default=.25, min=0, max=1,
        description='Zero gives a defined ring at the maximum diameter; positive values soften it outward')
    color_obj: PointerProperty(type=bpy.types.Object)
    clear_obj: PointerProperty(type=bpy.types.Object)
    result: StringProperty()
    editing_photo: BoolProperty(default=False,options={'SKIP_SAVE'})


def mesh_object(name, coords, faces, materials, slots, scene):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(coords, [], faces)
    for mat in materials: mesh.materials.append(mat)
    for poly, slot in zip(mesh.polygons, slots):
        poly.material_index = slot
        poly.use_smooth = True
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        if any(not e.is_manifold for e in bm.edges):
            raise RuntimeError('Iris button contains an open or non-manifold edge')
        if bm.calc_volume(signed=True) < 0:
            bmesh.ops.reverse_faces(bm, faces=list(bm.faces))
        if bm.calc_volume() < .01: raise RuntimeError('Iris button has no usable volume')
        bm.to_mesh(mesh)
    except Exception:
        bpy.data.meshes.remove(mesh)
        raise
    finally:
        bm.free()
    obj = bpy.data.objects.new(name, mesh)
    scene_helpers.group(scene, scene_helpers.GROUPS[5]).objects.link(obj)
    obj['anaplast_part'] = 'IRIS_BUTTON'
    obj['iris_button_role'] = 'CLEAR' if 'Clear' in name else 'COLOR'
    return obj


def pupil_fade(b,p,rgb,xy,radius):
    from .apparent_pupil import apply
    return apply(b,p,rgb,xy,radius)


def build(context):
    scene = context.scene
    from .eye_design import initialize,ButtonSettings
    initialize(scene)
    b = ButtonSettings(scene)
    p = PhotoSettings(b.photo,scene)
    prod.photo_valid(p)
    R = b.diameter / 2
    if b.dome_height < .5:raise RuntimeError('For an iris button, set Corneal rise to at least 0.5 mm')
    if b.dome_height >= R: raise RuntimeError('Corneal rise must be smaller than half the button diameter')
    model = ocular_pupil.contour(p)
    # Resolve photo sizing locally; the saved ocular settings remain untouched.
    if b.apparent_pupil:
        p.pupil_diameter=b.pupil_min
        if b.pupil_max<=b.pupil_min or b.pupil_max+2*b.pupil_outer_fade>=b.diameter*.9:
            raise RuntimeError('Set maximum pupil fade larger than minimum and smaller than 90% of the iris diameter')
        if b.pupil_outer_darkness>b.pupil_darkness:
            raise RuntimeError('Darkening at maximum must not exceed inner iris darkening')
    elif p.pupil_size_mode == 'PHOTO':
        p.pupil_diameter = b.diameter * (model['equivalent'] if model else p.photo_pupil)
    if p.pupil_depth < .45: raise RuntimeError('Pupil chamber needs at least 0.45 mm depth')
    if p.pupil_diameter >= b.diameter * .8: raise RuntimeError('Reduce pupil diameter to leave room for the iris')
    if p.relief_height * .5 + p.pupil_rim_height >= b.cover:
        raise RuntimeError('Increase clear cover or reduce iris relief / rim height')
    n = 256
    angles = np.arange(n) * (2 * math.pi / n)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    shaped = ocular_pupil.shape(p, np.zeros(2), R)
    if shaped:
        _, _, center, boundary = shaped
        delta = boundary - center
        theta = np.mod(np.arctan2(delta[:,1],delta[:,0]),2*math.pi)
        order = np.argsort(theta)
        pupil_r = np.interp(angles, theta[order], np.linalg.norm(delta,axis=1)[order], period=2*math.pi)
    else:
        center = np.zeros(2)
        pupil_r = np.full(n, p.pupil_diameter/2)
    if np.max(np.linalg.norm(center + directions * (pupil_r+p.pupil_underiris)[:,None],axis=1)) > R-.4:
        raise RuntimeError('Under-iris chamber is too wide; reduce pupil size or chamber widening')
    # Rays from an off-centre pupil terminate on the circular outer iris.
    dot = directions @ center
    outer_r = -dot + np.sqrt(dot*dot + R*R - center@center)
    points, faces, labels = [], [], []

    def ring(xy, z, label=0):
        start = len(points)
        z = np.broadcast_to(z, (n,))
        points.extend(np.column_stack((xy,z)).tolist())
        if start:
            prev = start-n
            for k in range(n):
                j=(k+1)%n
                faces.append((prev+k,prev+j,start+j,start+k)); labels.append(label)
        return start

    def center_cap(xy, z, label):
        start=len(points)-n; tip=len(points);points.append([*xy,z])
        for k in range(n):
            faces.append((start+k,start+(k+1)%n,tip));labels.append(label)

    # Shared interface: textured annulus, rounded pupil lip, flared black chamber.
    iris_points=[];iris_distances=[]
    for t in np.linspace(0,1,81):
        r=outer_r*(1-t)+pupil_r*t
        xy=center+directions*r[:,None]
        distance=(r-pupil_r)
        iris_points.extend(xy);iris_distances.extend(distance)
        ring(xy,0,0)
    base=np.array(points)
    base[:,2]+=prod.relief_values(p,np.array(iris_points),np.zeros(2),R)
    base=prod.rolled_iris_rim(base,np.array(iris_points),center,p.pupil_diameter/2,
        p.pupil_rim_height,p.pupil_rim_width,np.array([0,0,1]),p.pupil_rim_blend,np.array(iris_distances))
    points[:]=base.tolist()
    ring(center+directions*pupil_r[:,None],-.12,2)
    for a in np.linspace(0,math.pi,41)[1:]:
        r=pupil_r+p.pupil_underiris*math.sin(a)
        z=-.12-(p.pupil_depth-.12)*(1-math.cos(a))*.5
        ring(center+directions*r[:,None],z,2)
    for t in np.linspace(1,0,12)[1:-1]:
        ring(center+directions*(pupil_r*t)[:,None],-p.pupil_depth,2)
    center_cap(center,-p.pupil_depth,2)
    interface=np.array(points)
    # Core backside: concavity is away from the chamber and never thins its floor.
    back_points=interface.tolist(); back_faces=list(faces); back_labels=list(labels)
    def back_z(r):
        return -p.pupil_depth-b.backing-(b.back_sag*(r/R)**2 if b.back=='CONCAVE' else 0)
    prev=0
    for r in np.linspace(R,0,49)[:-1]:
        start=len(back_points)
        back_points.extend(np.column_stack((directions*r,np.full(n,back_z(r)))).tolist())
        # Outer interface ring uses rays from the pupil centre; preserve ordering.
        if prev==0:
            xy=interface[:n,:2]
            back_points[start:start+n]=np.column_stack((xy,np.full(n,back_z(R)))).tolist()
        for k in range(n):
            j=(k+1)%n;back_faces.append((prev+k,start+k,start+j,prev+j));back_labels.append(1)
        prev=start
    tip=len(back_points);back_points.append([0,0,back_z(0)])
    for k in range(n):back_faces.append((prev+k,tip,prev+(k+1)%n));back_labels.append(1)

    # Complementary clear volume, with a continuous dome-to-peg outer skin.
    clear_points=interface.tolist();clear_faces=[tuple(reversed(f)) for f in faces]
    clear_labels=[1 if x==0 else 2 for x in labels]
    prev=0
    sphere_radius=(R*R+b.dome_height*b.dome_height)/(2*b.dome_height)
    def dome(r):return b.cover+np.sqrt(sphere_radius*sphere_radius-r*r)-(sphere_radius-b.dome_height)
    def outer_ring(r,z,xy=None):
        nonlocal prev
        start=len(clear_points)
        xy=directions*r if xy is None else xy
        clear_points.extend(np.column_stack((xy,np.full(n,z))).tolist())
        for k in range(n):
            j=(k+1)%n;clear_faces.append((prev+k,prev+j,start+j,start+k));clear_labels.append(0)
        prev=start
    outer_ring(R,b.cover,interface[:n,:2])
    peg=b.peg_diameter/2
    if peg+.3>=R:raise RuntimeError('Peg is too wide for this button')
    # Match the first ring's angular distribution, then relax into centred rings.
    outer_angles=np.arctan2(interface[:n,1],interface[:n,0])
    outer_dirs=np.column_stack((np.cos(outer_angles),np.sin(outer_angles)))
    for r in np.linspace(R,peg+.25,70)[1:]:outer_ring(r,dome(r),outer_dirs*r)
    # Small rounded shoulder and rounded tip on the sacrificial positioning stem.
    shoulder_z=dome(peg+.25)
    for a in np.linspace(0,math.pi/2,12)[1:]:
        r=peg+.25-.25*math.sin(a);z=shoulder_z+.25*(1-math.cos(a))
        outer_ring(r,z,outer_dirs*r)
    tip_radius=peg*(1-b.peg_taper/100)
    top=dome(0)+b.peg_height; rounding=min(.2,tip_radius*.25)
    shaft_start=shoulder_z+.25
    # Gentle convergence, tangent to the rounded shoulder and rounded tip.
    for t in np.linspace(0,1,25)[1:]:
        r=peg+(tip_radius-peg)*t*t*(3-2*t)
        z=shaft_start+(top-rounding-shaft_start)*t
        outer_ring(r,z,outer_dirs*r)
    for a in np.linspace(0,math.pi/2,10)[1:]:
        r=tip_radius-rounding+rounding*math.cos(a);z=top-rounding+rounding*math.sin(a)
        outer_ring(r,z,outer_dirs*r)
    tip=len(clear_points);clear_points.append([0,0,top])
    for k in range(n):clear_faces.append((prev+k,prev+(k+1)%n,tip));clear_labels.append(0)

    size=p.texture_size
    axis=np.linspace(-R,R,size);X,Y=np.meshgrid(axis,axis)
    rgb,_,_=prod.iris_photo(p,np.stack((X,Y),-1),np.zeros(2),R)
    rgb=pupil_fade(b,p,rgb,np.stack((X,Y),-1),R)
    rgba=np.concatenate((rgb,np.ones((*rgb.shape[:2],1))),axis=-1).astype(np.float32)
    created=[];mats=[];image=None
    try:
        image=bpy.data.images.new('Iris button photo colors',width=size,height=size)
        image.pixels.foreach_set(rgba.ravel());image.pack()
        mats=[prod.material('Iris button photo',image=image),prod.material('Iris button backing',color=(.003,.003,.003)),
              prod.material('Iris button pupil lining',color=(.003,.003,.003)),prod.material('Iris button clear',clear=True)]
        core=mesh_object('Iris_Button_Color',back_points,back_faces,mats[:3],back_labels,scene);created.append(core)
        clear=mesh_object('Iris_Button_Clear',clear_points,clear_faces,[mats[3],mats[0],mats[2]],clear_labels,scene);created.append(clear)
        for obj in created:
            uv=obj.data.uv_layers.new(name='Iris photo')
            for loop in obj.data.loops:
                v=obj.data.vertices[loop.vertex_index].co
                uv.data[loop.index].uv=((v.x/R+1)/2,(v.y/R+1)/2)
            scale=.001/scene.unit_settings.scale_length
            obj.scale=(scale,)*3
            obj['iris_button_diameter_mm']=b.diameter
            obj['pupil_diameter_mm']=p.pupil_diameter
            obj['pupil_chamber_depth_mm']=p.pupil_depth
            obj['pupil_underiris_mm']=p.pupil_underiris
            obj['peg_diameter_mm']=b.peg_diameter;obj['peg_height_mm']=b.peg_height
            obj['peg_taper_percent']=b.peg_taper
            obj['peg_tip_shaft_diameter_mm']=2*tip_radius
            obj['apparent_pupil']=b.apparent_pupil
            obj['apparent_pupil_max_mm']=b.pupil_max if b.apparent_pupil else p.pupil_diameter
            obj['apparent_pupil_darkness']=b.pupil_darkness if b.apparent_pupil else 0.
            obj['apparent_pupil_outer_darkness']=b.pupil_outer_darkness if b.apparent_pupil else 0.
            obj['apparent_pupil_outer_fade_mm']=b.pupil_outer_fade if b.apparent_pupil else 0.
            obj['back_shape']=b.back;obj['units']='mm in mesh coordinates'
        from .mold_design import has_self_intersections
        for obj in created:
            if has_self_intersections(obj):raise RuntimeError('Button surfaces intersect; reduce chamber size or iris relief')
    except Exception:
        for obj in created:
            mesh=obj.data;bpy.data.objects.remove(obj,do_unlink=True)
            if mesh.users==0:bpy.data.meshes.remove(mesh)
        for mat in mats:
            if mat.users==0:bpy.data.materials.remove(mat)
        if image and image.users==0:bpy.data.images.remove(image)
        raise
    old=(b.color_obj,b.clear_obj)
    old_materials={mat for obj in old if obj and obj.get('anaplast_part')=='IRIS_BUTTON' for mat in obj.data.materials if mat}
    old_images={node.image for mat in old_materials if mat.use_nodes for node in mat.node_tree.nodes if node.type=='TEX_IMAGE' and node.image}
    # Preserve placement across updates; old results survive all validation failures.
    if b.color_obj:
        for obj in created:
            obj.location=b.color_obj.location.copy();obj.rotation_euler=b.color_obj.rotation_euler.copy()
    b.color_obj,b.clear_obj=created
    for obj in old:
        if obj and obj.get('anaplast_part')=='IRIS_BUTTON':
            mesh=obj.data;bpy.data.objects.remove(obj,do_unlink=True)
            if mesh.users==0:bpy.data.meshes.remove(mesh)
    for mat in old_materials:
        if mat.users==0:bpy.data.materials.remove(mat)
    for old_image in old_images:
        if old_image.users==0 and old_image.name.startswith('Iris button photo colors'):
            bpy.data.images.remove(old_image)
    core.name='Iris_Button_Color';clear.name='Iris_Button_Clear'
    b.result=f'{b.diameter:g} mm iris button • {p.pupil_diameter:g} mm pupil • clear dome + peg'
    return core,clear


class IRIS_OT_build(bpy.types.Operator):
    bl_idname='anaplast.iris_button_build';bl_label='Build iris button';bl_options={'REGISTER','UNDO'}
    def execute(self,c):
        try:objects=build(c)
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        if c.object and c.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        for obj in c.selected_objects:obj.select_set(False)
        for obj in objects:obj.select_set(True)
        c.view_layer.objects.active=objects[0]
        if c.area and c.area.type=='VIEW_3D':
            c.space_data.shading.type='MATERIAL'
            region=next((r for r in c.area.regions if r.type=='WINDOW'),None)
            if region:
                with c.temp_override(region=region):bpy.ops.view3d.view_selected(use_all_regions=False)
        self.report({'INFO'},c.scene.iris_button.result);return {'FINISHED'}


class IRIS_OT_photo(bpy.types.Operator):
    bl_idname='anaplast.iris_button_photo';bl_label='Load iris photograph';bl_options={'REGISTER','UNDO'}
    filepath:StringProperty(subtype='FILE_PATH')
    filter_glob:StringProperty(default='*.png;*.jpg;*.jpeg;*.tif;*.tiff',options={'HIDDEN'})
    def invoke(self,c,event):c.window_manager.fileselect_add(self);return {'RUNNING_MODAL'}
    def execute(self,c):
        return bpy.ops.anaplast.eye_photo_load(filepath=self.filepath)


class IRIS_OT_view_photo(bpy.types.Operator):
    bl_idname='anaplast.iris_button_view_photo';bl_label='Align iris / trace pupil'
    def execute(self,c):
        return bpy.ops.anaplast.eye_photo_view()


class IRIS_OT_return(bpy.types.Operator):
    bl_idname='anaplast.iris_button_return';bl_label='Return to model'
    def execute(self,c):c.scene.iris_button.editing_photo=False;c.area.type='VIEW_3D';return {'FINISHED'}


def export_button(scene, filepath):
    """Separate, aligned color and clear manufacturing solids, always in mm."""
    b=scene.iris_button
    if not b.color_obj or not b.clear_obj:raise RuntimeError('Build an iris button first')
    image=next((node.image for mat in b.color_obj.data.materials if mat and mat.use_nodes
        for node in mat.node_tree.nodes if node.type=='TEX_IMAGE' and node.image),None)
    if not image:raise RuntimeError('Button photo texture is missing; rebuild the button')
    with tempfile.TemporaryDirectory(prefix='anaplast_iris_button_') as temporary:
        folder=Path(temporary)
        copy=image.copy()
        try:
            copy.filepath_raw=str(folder/'Iris.png');copy.file_format='PNG';copy.save()
        finally:bpy.data.images.remove(copy)
        (folder/'Iris_button.mtl').write_text(
            'newmtl Iris\nKd 1 1 1\nmap_Kd Iris.png\n\n'
            'newmtl Black\nKd 0.003 0.003 0.003\n\n'
            'newmtl Clear\nKd 1 1 1\nNi 1.49\nd 0.15\nillum 4\n',encoding='utf-8')
        for obj,name in [(b.color_obj,'Iris_button_color'),(b.clear_obj,'Iris_button_clear')]:
            mesh=obj.data;uv=mesh.uv_layers.active
            if not uv:raise RuntimeError('Button UV map is missing; rebuild the button')
            lines=['# Anaplast Studio iris button; coordinates in millimetres','mtllib Iris_button.mtl',f'o {name}']
            for v in mesh.vertices:
                co=(obj.matrix_world@v.co)*(scene.unit_settings.scale_length*1000)
                lines.append('v '+' '.join(f'{x:.8f}' for x in co))
            lines.extend('vt '+' '.join(f'{x:.8f}' for x in entry.uv) for entry in uv.data)
            for face in mesh.polygons:
                material='Clear' if obj==b.clear_obj else ('Iris' if face.material_index==0 else 'Black')
                lines.append('usemtl '+material)
                lines.append('f '+' '.join(f'{mesh.loops[i].vertex_index+1}/{i+1}' for i in face.loop_indices))
            (folder/(name+'.obj')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
        (folder/'Read_me.txt').write_text(
            'Anaplast Studio — standalone iris button\n\n'
            'Two aligned, closed material solids: color/black backing and clear cornea/peg/chamber.\n'
            'Keep their shared coordinates. Dimensions are millimetres.\n'
            'Iris.png is the photo color map. Keep OBJ, MTL and PNG files together.\n'
            'Assign the clear solid to a clear printing material; OBJ transparency alone is a viewer hint.\n'
            'The positioning peg is joined to the clear cornea for subsequent removal and finishing.\n',encoding='utf-8')
        with zipfile.ZipFile(filepath,'w',zipfile.ZIP_DEFLATED) as archive:
            for path in folder.iterdir():archive.write(path,path.name)


class IRIS_OT_export(bpy.types.Operator):
    bl_idname='anaplast.iris_button_export';bl_label='Export iris button';bl_options={'REGISTER'}
    filepath:StringProperty(subtype='FILE_PATH',default='Iris_button.zip')
    filter_glob:StringProperty(default='*.zip',options={'HIDDEN'})
    def invoke(self,c,event):c.window_manager.fileselect_add(self);return {'RUNNING_MODAL'}
    def execute(self,c):
        try:export_button(c.scene,bpy.path.ensure_ext(self.filepath,'.zip'))
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        self.report({'INFO'},'Exported aligned color and clear parts, with iris texture');return {'FINISHED'}


class IRIS_OT_mark(bpy.types.Operator):
    bl_idname='anaplast.iris_button_mark';bl_label='Mark photograph';bl_options={'REGISTER','UNDO'}
    kind:EnumProperty(items=[('CENTER','Iris center',''),('HORIZONTAL','Iris side',''),('VERTICAL','Iris top/bottom',''),('PUPIL','Trace pupil border','')])
    @classmethod
    def poll(cls,c):return c.area and c.area.type=='IMAGE_EDITOR' and c.scene.iris_button.photo.photo==c.space_data.image and c.space_data.image is not None
    def invoke(self,c,event):
        p=c.scene.iris_button.photo
        self.before=(p.photo_pupil_outline,p.pupil_outline_image)
        if self.kind=='PUPIL':p.photo_pupil_outline='[]';p.pupil_outline_image=p.photo
        c.window_manager.modal_handler_add(self);c.window.cursor_modal_set('CROSSHAIR')
        c.area.header_text_set('Click photo landmark' if self.kind!='PUPIL' else 'Click around pupil • Enter finish • Backspace undo • Esc cancel')
        return {'RUNNING_MODAL'}
    def finish(self,c):c.window.cursor_modal_restore();c.area.header_text_set(None);c.area.tag_redraw()
    def modal(self,c,event):
        p=c.scene.iris_button.photo
        if event.type in {'ESC','RIGHTMOUSE'}:
            if self.kind=='PUPIL':p.photo_pupil_outline,p.pupil_outline_image=self.before
            self.finish(c);return {'CANCELLED'}
        if event.value!='PRESS':return {'PASS_THROUGH'}
        if event.type in {'RET','NUMPAD_ENTER'} and self.kind=='PUPIL':
            try:
                model=ocular_pupil.contour(p)
                if model is None:raise RuntimeError('Trace at least six pupil border points')
            except RuntimeError as exc:self.report({'WARNING'},str(exc));return {'RUNNING_MODAL'}
            p.photo_pupil=model['equivalent'];p.pupil_size_mode='PHOTO';self.finish(c);return {'FINISHED'}
        if event.type=='BACK_SPACE' and self.kind=='PUPIL':
            p.photo_pupil_outline=json.dumps(json.loads(p.photo_pupil_outline)[:-1]);c.area.tag_redraw();return {'RUNNING_MODAL'}
        if event.type!='LEFTMOUSE':return {'PASS_THROUGH'}
        region=next(r for r in c.area.regions if r.type=='WINDOW')
        x,y=event.mouse_x-region.x,event.mouse_y-region.y
        if not (0<=x<region.width and 0<=y<region.height):return {'RUNNING_MODAL'}
        uv=region.view2d.region_to_view(x,y)
        if not all(0<=v<=1 for v in uv):return {'RUNNING_MODAL'}
        if self.kind=='PUPIL':
            points=json.loads(p.photo_pupil_outline);points.append(list(uv));p.photo_pupil_outline=json.dumps(points)
            c.area.tag_redraw();return {'RUNNING_MODAL'}
        if self.kind=='CENTER':p.photo_center=uv
        elif self.kind=='HORIZONTAL':p.photo_radius=(max(.001,abs(uv[0]-p.photo_center[0])),p.photo_radius[1])
        else:p.photo_radius=(p.photo_radius[0],max(.001,abs(uv[1]-p.photo_center[1])))
        self.finish(c);return {'FINISHED'}


def draw(layout,context):
    b=context.scene.iris_button;p=b.photo
    layout.label(text='Standalone iris, pupil chamber, cornea and peg')
    layout.label(text='Uses the shared Eye photograph and marks')
    layout.prop(b,'diameter');layout.prop(b,'apparent_pupil')
    if b.apparent_pupil:
        layout.prop(b,'pupil_min');layout.prop(b,'pupil_max');layout.prop(b,'pupil_darkness')
        layout.prop(b,'pupil_outer_darkness');layout.prop(b,'pupil_outer_fade')
        layout.label(text='Fixed opening; dark iris fades to photo color')
    else:
        layout.prop(p,'pupil_size_mode')
        if p.pupil_size_mode=='MANUAL':layout.prop(p,'pupil_diameter')
    layout.prop(b,'dome_height');layout.prop(b,'cover')
    layout.prop(b,'back')
    if b.back=='CONCAVE':layout.prop(b,'back_sag')
    layout.prop(b,'backing');layout.prop(p,'pupil_depth');layout.prop(p,'pupil_underiris')
    layout.prop(b,'peg_diameter');layout.prop(b,'peg_height');layout.prop(b,'peg_taper')
    layout.prop(p,'relief_height');layout.prop(p,'pupil_rim_height');layout.prop(p,'pupil_rim_blend')
    layout.operator('anaplast.iris_button_build',text='Update iris button' if b.color_obj else 'Build iris button',icon='MESH_UVSPHERE')
    if b.color_obj and b.clear_obj:layout.operator('anaplast.iris_button_export',icon='EXPORT')
    if b.result:layout.label(text=b.result)


class IRIS_PT_photo(bpy.types.Panel):
    bl_label='Iris button photograph';bl_space_type='IMAGE_EDITOR';bl_region_type='UI';bl_category='Iris button'
    def draw(self,c):
        l=self.layout;p=c.scene.iris_button.photo
        for kind,text in [('CENTER','1 · Mark iris center'),('HORIZONTAL','2 · Mark iris side'),('VERTICAL','3 · Mark iris top/bottom'),('PUPIL','4 · Trace pupil border')]:
            l.operator('anaplast.iris_button_mark',text=text).kind=kind
        l.prop(p,'photo_center');l.prop(p,'photo_radius');l.prop(p,'iris_rotation')
        l.prop(p,'photo_pupil');l.label(text='Traced pupil overrides circular photo ratio')
        l.operator('anaplast.iris_button_return')


def overlay():
    c=bpy.context
    if not c.area or c.area.type!='IMAGE_EDITOR':return
    p=c.scene.iris_button.photo
    if not c.scene.iris_button.editing_photo or not p.photo or c.space_data.image!=p.photo:return
    import gpu
    from gpu_extras.batch import batch_for_shader
    a=np.linspace(0,2*math.pi,129)
    iris=np.array(p.photo_center)+np.array(p.photo_radius)*np.column_stack((np.cos(a),np.sin(a)))
    contours=[iris]
    try:
        model=ocular_pupil.contour(p)
        if model:contours.append(np.array(p.photo_center)+model['curve']*np.array(p.photo_radius))
        else:contours.append(np.array(p.photo_center)+(iris-np.array(p.photo_center))*p.photo_pupil)
    except RuntimeError:contours.append(json.loads(p.photo_pupil_outline))
    lines=[]
    for contour in contours:
        positions=[c.region.view2d.view_to_region(*uv,clip=False) for uv in contour]
        for x,y in zip(positions,positions[1:]+positions[:1]):lines.extend((x,y))
    if lines:
        shader=gpu.shader.from_builtin('UNIFORM_COLOR');shader.bind();shader.uniform_float('color',(.1,1,.5,1))
        batch_for_shader(shader,'LINES',{'pos':lines}).draw(shader)


_classes=(IrisButton,IRIS_OT_build,IRIS_OT_photo,IRIS_OT_view_photo,IRIS_OT_return,IRIS_OT_export,IRIS_OT_mark)

def register():
    global _handle
    for cls in _classes:bpy.utils.register_class(cls)
    bpy.types.Scene.iris_button=PointerProperty(type=IrisButton)

def unregister():
    global _handle
    if _handle is not None:bpy.types.SpaceImageEditor.draw_handler_remove(_handle,'WINDOW');_handle=None
    del bpy.types.Scene.iris_button
    for cls in reversed(_classes):bpy.utils.unregister_class(cls)
