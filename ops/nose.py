"""Local nasal reconstruction prototype. Model and photos are read from disk only."""
import json
from pathlib import Path
import numpy as np
import bpy
from mathutils import Matrix, Vector
from bpy.props import (StringProperty, FloatProperty, IntProperty, BoolProperty,
                       EnumProperty, PointerProperty, CollectionProperty)
from . import nose_core as core
from ..utils import mesh as mu

COL='Anaplast_Nose_Prototype'
KEYS=[('NASION','Bridge root',''),('SUBNASALE','Columella base',''),
      ('TRAGION_R','Tragion R',''),('TRAGION_L','Tragion L',''),
      ('ORBITALE_R','Orbitale R',''),('ORBITALE_L','Orbitale L',''),
      ('EYE_IN_R','Inner eye corner R',''),('EYE_IN_L','Inner eye corner L',''),
      ('EXOCANTHION_R','Outer eye corner R',''),('EXOCANTHION_L','Outer eye corner L',''),
      ('TIP','Nose tip',''),('ALA_R','Alar outer edge R',''),('ALA_L','Alar outer edge L',''),
      ('MOUTH_R','Mouth corner R',''),('MOUTH_L','Mouth corner L',''),('CHIN','Chin bottom','')]
_cache={}
_draw_handle=None


def draw_photo_points():
    """Observed dots and fitted crosses in the image; coordinates stay editable."""
    context=bpy.context
    if not context.area or context.area.type!='IMAGE_EDITOR':return
    try:
        p=context.scene.nose_local;ph=current_photo(p)
        if context.space_data.image!=ph.image:return
        import gpu,blf
        from gpu_extras.batch import batch_for_shader
        shader=gpu.shader.from_builtin('UNIFORM_COLOR')
        def segments(lines,color):
            if not lines:return
            batch=batch_for_shader(shader,'LINES',{'pos':lines});shader.bind();shader.uniform_float('color',color);batch.draw(shader)
        marks=json.loads(ph.points);lines=[]
        for key,uv in marks.items():
            x,y=context.region.view2d.view_to_region(*uv,clip=False)
            lines.extend([(x-4,y),(x+4,y),(x,y-4),(x,y+4)])
            blf.position(0,x+7,y+4,0);blf.size(0,11);blf.color(0,.2,1.,.4,1.);blf.draw(0,key)
        segments(lines,(.2,1.,.4,1.))
        saved=json.loads(p.cameras)
        if not isinstance(saved,dict) or p.photo_index-1 not in saved['indices']:return
        mean,basis,*_=data(p);co=coefficient_shape(p);ids=mapping(p)
        cam=np.asarray(saved['cameras'][saved['indices'].index(p.photo_index-1)])
        keys=[k for k in marks if k in ids];vertices=np.array([ids[k] for k in keys],int)
        xy=core.project((co[vertices]-mean.mean(0))/np.ptp(mean,axis=0).max(),cam)
        dims=np.array(ph.image.size);uvs=xy*dims.max()/dims;lines=[]
        for key,uv in zip(keys,uvs):
            x,y=context.region.view2d.view_to_region(*uv,clip=False)
            tx,ty=context.region.view2d.view_to_region(*marks[key],clip=False)
            lines.extend([(x-4,y-4),(x+4,y+4),(x-4,y+4),(x+4,y-4),(x,y),(tx,ty)])
        segments(lines,(1.,.5,.12,1.))
    except (ValueError,KeyError,ReferenceError,OSError):return


def data(p):
    path=Path(bpy.path.abspath(p.model_path))
    key=(str(path),path.stat().st_mtime_ns,path.stat().st_size)
    if key not in _cache:
        _cache.clear();_cache[key]=core.read_model(path)
    return _cache[key]


def mapping(p):return json.loads(p.model_points or '{}')


def automatic_points(p):
    """Known vertex correspondence, never landmark inference on an unknown scan."""
    mean,basis,faces,stored,_=data(p)
    result=dict(stored)
    prepared=Path(__file__).parent.parent/'assets'/'FLAME2023Open_Nose.npz'
    if prepared.exists():
        with np.load(prepared,allow_pickle=False) as d:
            # Same mesh layout and neutral template, not merely the same vertex count.
            if mean.shape==d['mean'].shape and np.array_equal(faces,d['faces']) and np.allclose(mean,d['mean'],atol=1e-6,rtol=0):
                result={**json.loads(str(d['landmarks'])),**result}
    return result


def sync_model_markers(p):
    """Parent labels to both generated meshes and update after shape adjustments."""
    labels={k:n for k,n,_ in KEYS}
    ids=mapping(p)
    for obj in (p.template,p.proposal):
        if obj is None:continue
        for child in list(obj.children):
            if child.get('nose_landmark') and child['nose_landmark'] not in ids:
                bpy.data.objects.remove(child,do_unlink=True)
        for key,index in ids.items():
            if not 0<=index<len(obj.data.vertices):continue
            marker=next((o for o in obj.children if o.get('nose_landmark')==key),None)
            if marker is None:
                marker=bpy.data.objects.new(labels.get(key,key),None)
                mu.get_collection(bpy.context.scene,COL).objects.link(marker)
                marker.parent=obj;marker['nose_landmark']=key
                marker.empty_display_type='SPHERE';marker.show_name=True;marker.show_in_front=True
                marker.hide_select=True  # correction goes through the named point picker
            marker.location=obj.data.vertices[index].co
            marker.empty_display_size=p.marker_size
            marker.color=(.95,.55,.1,1.)
            marker.hide_render=True
            marker.hide_set(not p.show_model_marks or obj.hide_get())


def markers_update(self,context):
    sync_model_markers(context.scene.nose_local)


def patient_pose(context,p,co):
    ids=mapping(p)
    pairs=[(i,bpy.data.objects.get('LM_'+key)) for key,i in ids.items() if bpy.data.objects.get('LM_'+key)]
    if len(pairs)<3:raise ValueError('Need at least three confirmed patient landmarks matching the model marks')
    src=np.array([co[i] for i,o in pairs]);target=np.array([o.matrix_world.translation[:] for i,o in pairs])
    mm=context.scene.unit_settings.scale_length*1000 if context.scene.unit_settings.system!='NONE' else 1.
    linear,shift=core.similarity(src,target*mm)
    matrix=np.eye(4);matrix[:3,:3]=linear/mm;matrix[:3,3]=shift/mm
    rms=float(np.sqrt(np.mean(np.sum((src@linear.T+shift-target*mm)**2,axis=1))))
    return Matrix(matrix.tolist()),rms,len(pairs)


def make_object(name,co,faces):
    col=mu.get_collection(bpy.context.scene,COL)
    me=bpy.data.meshes.new(name);me.from_pydata(co.tolist(),[],faces.tolist());me.update()
    obj=bpy.data.objects.new(name,me);col.objects.link(obj)
    for f in me.polygons:f.use_smooth=True
    return obj


def activate(context,obj):
    for o in context.selected_objects:o.select_set(False)
    obj.hide_set(False);obj.select_set(True);context.view_layer.objects.active=obj


def coefficient_shape(p):
    mean,basis,*_=data(p)
    b=np.asarray(json.loads(p.coefficients or '[]'),dtype=float)
    co=mean.copy()
    if len(b):co+=np.einsum('nck,k->nc',basis[:,:,:len(b)],b)
    ids=np.asarray(json.loads(p.region or '[]'),dtype=int)
    points=mapping(p)
    if len(ids) and 'NASION' in points and 'SUBNASALE' in points and (p.width,p.projection,p.height)!=(1.,1.,1.):
        n,s=co[points['NASION']],co[points['SUBNASALE']]
        up=n-s;up/=np.linalg.norm(up)
        if 'TRAGION_R' in points and 'TRAGION_L' in points:
            right=co[points['TRAGION_R']]-co[points['TRAGION_L']]
        else:right=np.array([1.,0.,0.])
        right-=up*np.dot(up,right);right/=np.linalg.norm(right);forward=np.cross(right,up)
        # FLAME source space is X right, Y up, Z forward. Preserve outside region.
        rel=co[ids]-s;frame=np.stack((right,forward,up),axis=1);q=rel@frame
        q[:,0]*=p.width;q[:,1]*=p.projection;q[:,2]*=p.height
        # Fade controls to zero at the region boundary to avoid a step into skin.
        import heapq
        faces=data(p)[2];inside=np.zeros(len(mean),bool);inside[ids]=True
        edges=np.unique(np.sort(np.concatenate((faces[:,:2],faces[:,1:],faces[:,[0,2]])),axis=1),axis=0)
        boundary=edges[inside[edges[:,0]]!=inside[edges[:,1]]].ravel()
        seeds=np.unique(boundary[inside[boundary]])
        distance=np.full(len(mean),np.inf);distance[seeds]=0.;adj=[[] for _ in mean]
        for a,b in edges[inside[edges].all(1)]:
            length=float(np.linalg.norm(mean[a]-mean[b]));adj[a].append((b,length));adj[b].append((a,length))
        queue=[(0.,int(i)) for i in seeds];heapq.heapify(queue)
        while queue:
            d,i=heapq.heappop(queue)
            if d>distance[i] or d>10.:continue
            for j,w in adj[i]:
                if d+w<distance[j]:distance[j]=d+w;heapq.heappush(queue,(d+w,j))
        t=np.clip(distance[ids]/10.,0,1);weight=t*t*(3-2*t)
        co[ids]+=weight[:,None]*(q@frame.T+s-co[ids])
    return co


def refresh(self,context):
    p=context.scene.nose_local
    if p.proposal is None:return
    try:
        co=coefficient_shape(p)
        if len(co)!=len(p.proposal.data.vertices):return
        p.proposal.data.vertices.foreach_set('co',co.ravel());p.proposal.data.update()
        sync_model_markers(p)
    except Exception as e:p.status='Preview not updated: '+str(e)


class NosePhoto(bpy.types.PropertyGroup):
    image: PointerProperty(type=bpy.types.Image)
    points: StringProperty(default='{}')


class NoseSettings(bpy.types.PropertyGroup):
    model_path: StringProperty(subtype='FILE_PATH')
    template: PointerProperty(type=bpy.types.Object)
    proposal: PointerProperty(type=bpy.types.Object)
    model_points: StringProperty(default='{}')
    region: StringProperty(default='[]')
    coefficients: StringProperty(default='[]')
    cameras: StringProperty(default='[]')
    status: StringProperty(default='Load FLAME 2023 Open to begin')
    show_model_marks: BoolProperty(name='Show automatic marks',default=True,update=markers_update)
    marker_size: FloatProperty(name='Mark size',default=.8,min=.2,max=3.,update=markers_update)
    mark_model: EnumProperty(name='Correct points on',items=[('TEMPLATE','Template',''),('PROPOSAL','Generated proposal','')],default='TEMPLATE')
    key: EnumProperty(items=KEYS)
    photos: CollectionProperty(type=NosePhoto)
    photo_index: IntProperty(default=1,min=1)
    modes: IntProperty(name='Shape flexibility',default=30,min=5,max=80)
    width: FloatProperty(name='Nose width',default=1.,min=.7,max=1.3,update=refresh)
    projection: FloatProperty(name='Projection',default=1.,min=.7,max=1.3,update=refresh)
    height: FloatProperty(name='Nose height',default=1.,min=.7,max=1.3,update=refresh)


def current_photo(p):
    if not len(p.photos):raise ValueError('Add a photograph first')
    return p.photos[min(p.photo_index-1,len(p.photos)-1)]


class NOSE_OT_load(bpy.types.Operator):
    bl_idname='anaplast.nose_load';bl_label='Load local shape model';bl_options={'REGISTER','UNDO'}
    filepath:StringProperty(subtype='FILE_PATH')
    filter_glob:StringProperty(default='*.zip;*.pkl;*.npz',options={'HIDDEN'})
    def invoke(self,context,event):context.window_manager.fileselect_add(self);return {'RUNNING_MODAL'}
    def execute(self,context):
        p=context.scene.nose_local
        try:mean,basis,faces,points,region=core.read_model(self.filepath)
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        obj=make_object('Nose_Template',mean,faces)
        p.model_path=self.filepath;p.template=obj;p.proposal=None;p.model_points=json.dumps(points)
        p.mark_model='TEMPLATE'
        p.region=json.dumps(region.tolist());p.coefficients='[]';p.cameras='[]'
        p.width=p.projection=p.height=1.
        p.model_points=json.dumps({**automatic_points(p),**points})
        obj['nose_source']='FLAME 2023 Open or user-prepared library; neutral identity shape'
        # Prepared model points refer to original mesh coordinates, never screen positions.
        activate(context,obj)
        sync_model_markers(p)
        p.status=f'Loaded {len(mean):,} vertices. Confirm template landmarks before placement.'
        self.report({'INFO'},p.status);return {'FINISHED'}


class NOSE_OT_prepared(bpy.types.Operator):
    bl_idname='anaplast.nose_prepared';bl_label='Load prepared FLAME nose template';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        path=Path(__file__).parent.parent/'assets'/'FLAME2023Open_Nose.npz'
        if not path.exists():self.report({'ERROR'},'Prepared template is missing; use Load local shape model');return {'CANCELLED'}
        return bpy.ops.anaplast.nose_load(filepath=str(path))


class NOSE_OT_model_point(bpy.types.Operator):
    bl_idname='anaplast.nose_model_point';bl_label='Click to correct selected mark';bl_options={'REGISTER','UNDO'}
    @classmethod
    def poll(cls,context):return context.area is not None and context.area.type=='VIEW_3D' and context.scene.nose_local.template is not None
    def invoke(self,context,event):
        p=context.scene.nose_local;self.key=p.key
        self.obj=p.proposal if p.mark_model=='PROPOSAL' else p.template
        if self.obj is None:
            self.report({'ERROR'},'Generate or load the chosen model first');return {'CANCELLED'}
        self.obj.hide_set(False);sync_model_markers(p)
        context.window_manager.modal_handler_add(self);context.window.cursor_modal_set('CROSSHAIR')
        context.area.header_text_set('Click '+dict((k,n) for k,n,_ in KEYS)[self.key]+' on '+self.obj.name+' · Esc cancel')
        return {'RUNNING_MODAL'}
    def finish(self,context):context.area.header_text_set(None);context.window.cursor_modal_restore()
    def modal(self,context,event):
        if event.type in {'ESC','RIGHTMOUSE'}:
            self.finish(context);return {'CANCELLED'}
        if event.type=='LEFTMOUSE' and event.value=='PRESS':
            from bpy_extras import view3d_utils
            p=context.scene.nose_local;obj=self.obj
            region=next(r for r in context.area.regions if r.type=='WINDOW');rv=context.space_data.region_3d
            xy=(event.mouse_x-region.x,event.mouse_y-region.y)
            origin=view3d_utils.region_2d_to_origin_3d(region,rv,xy)
            direction=view3d_utils.region_2d_to_vector_3d(region,rv,xy)
            w,_,_=mu.ray_cast_clipped(context,origin,direction,obj=obj)
            if w is None:return {'RUNNING_MODAL'}
            q=np.asarray(obj.matrix_world.inverted()@w);co=np.array([v.co[:] for v in obj.data.vertices])
            ids=mapping(p);ids[self.key]=int(np.argmin(np.linalg.norm(co-q,axis=1)));p.model_points=json.dumps(ids)
            sync_model_markers(p)
            p.status='Model point saved: '+self.key;self.finish(context);return {'FINISHED'}
        return {'PASS_THROUGH'}


class NOSE_OT_place(bpy.types.Operator):
    bl_idname='anaplast.nose_place';bl_label='Position on patient landmarks';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        p=context.scene.nose_local
        try:
            mean,basis,faces,*_=data(p);co=coefficient_shape(p)
            matrix,rms,count=patient_pose(context,p,co)
            obj=make_object('Nose_Proposal',co,faces);obj.matrix_world=matrix
            obj['nose_prototype']=True;obj['not_final_prosthesis']=True
            obj['nose_patient_aligned']=True
            p.proposal=obj;activate(context,obj)
            p.mark_model='PROPOSAL'
            if p.template:p.template.hide_set(True)
            sync_model_markers(p)
            p.status=f'Positioned using {count} landmarks; RMS {rms:.2f} mm. Review placement.'
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        self.report({'INFO'},p.status);return {'FINISHED'}


class NOSE_OT_show_template(bpy.types.Operator):
    bl_idname='anaplast.nose_show_template';bl_label='Show template for marking'
    def execute(self,context):
        p=context.scene.nose_local
        if p.template is None:return {'CANCELLED'}
        p.mark_model='TEMPLATE'
        activate(context,p.template)
        sync_model_markers(p)
        # Display away from the case, without changing the stored model coordinates.
        case=context.scene.anaplast.face_scan_obj
        if case:
            p.template.location=case.location+Vector((float(case.dimensions.x)*1.2,0.,0.))
        if context.area and context.area.type=='VIEW_3D':
            for r in context.area.regions:
                if r.type=='WINDOW':
                    with context.temp_override(region=r):bpy.ops.view3d.view_selected(use_all_regions=False)
                    break
        return {'FINISHED'}


class NOSE_OT_region(bpy.types.Operator):
    bl_idname='anaplast.nose_region';bl_label='Use selected template vertices as nose';bl_options={'REGISTER','UNDO'}
    @classmethod
    def poll(cls,context):return context.mode=='EDIT_MESH' and context.edit_object==context.scene.nose_local.template
    def execute(self,context):
        import bmesh
        bm=bmesh.from_edit_mesh(context.edit_object.data);bm.verts.ensure_lookup_table()
        ids=[v.index for v in bm.verts if v.select]
        if len(ids)<10:self.report({'ERROR'},'Select the nose and its margin first');return {'CANCELLED'}
        context.scene.nose_local.region=json.dumps(ids)
        bpy.ops.object.mode_set(mode='OBJECT');return {'FINISHED'}


class NOSE_OT_auto_marks(bpy.types.Operator):
    """Fill missing model marks from the prepared library; retain manual corrections"""
    bl_idname='anaplast.nose_auto_marks';bl_label='Auto-mark model';bl_options={'REGISTER','UNDO'}
    reset: BoolProperty(default=False,options={'HIDDEN'})
    def execute(self,context):
        p=context.scene.nose_local
        if p.template is None:self.report({'ERROR'},'Load a model first');return {'CANCELLED'}
        try:defaults=automatic_points(p)
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        if not defaults:
            self.report({'ERROR'},'This model has no known landmark map. Mark it manually and save it to your library.');return {'CANCELLED'}
        existing=mapping(p)
        # Reset only known defaults, retaining any additional custom named marks.
        ids={**existing,**defaults} if self.reset else {**defaults,**existing}
        p.model_points=json.dumps(ids);p.show_model_marks=True
        obj=p.proposal if p.mark_model=='PROPOSAL' else p.template
        if obj is not None:
            obj.hide_set(False)
            # Show the full generated head while checking facial alignment marks.
            preview=obj.modifiers.get('Nose preview')
            if preview:preview.show_viewport=False
        sync_model_markers(p)
        matches=sum(context.scene.objects.get('LM_'+key) is not None for key in ids)
        p.status=f'{len(ids)} model marks ready; {matches} matching patient marks. Review, then Position on patient landmarks.'
        self.report({'INFO'},p.status);return {'FINISHED'}


class NOSE_OT_library(bpy.types.Operator):
    bl_idname='anaplast.nose_library';bl_label='Save prepared template to library'
    filepath:StringProperty(subtype='FILE_PATH')
    filter_glob:StringProperty(default='*.npz',options={'HIDDEN'})
    def invoke(self,context,event):self.filepath='Nose_template.npz';context.window_manager.fileselect_add(self);return {'RUNNING_MODAL'}
    def execute(self,context):
        p=context.scene.nose_local
        try:
            mean,basis,faces,*_=data(p)
            path=Path(self.filepath).with_suffix('.npz')
            if path.exists():raise ValueError('Choose a new filename to preserve the existing library entry')
            np.savez_compressed(path,mean=mean,basis=basis,faces=faces,landmarks=p.model_points,
                                region=np.array(json.loads(p.region)),source='User prepared; retain source model license')
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        self.report({'INFO'},'Template and landmark setup saved for reuse');return {'FINISHED'}


class NOSE_OT_photo_add(bpy.types.Operator):
    bl_idname='anaplast.nose_photo_add';bl_label='Add pre-surgery photograph';bl_options={'REGISTER','UNDO'}
    filepath:StringProperty(subtype='FILE_PATH')
    filter_glob:StringProperty(default='*.jpg;*.jpeg;*.png;*.tif;*.tiff',options={'HIDDEN'})
    def invoke(self,context,event):context.window_manager.fileselect_add(self);return {'RUNNING_MODAL'}
    def execute(self,context):
        try:img=bpy.data.images.load(self.filepath,check_existing=True)
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        p=context.scene.nose_local;ph=p.photos.add();ph.image=img;p.photo_index=len(p.photos)
        self.report({'INFO'},'Photo loaded locally. Open it and mark visible matching landmarks.');return {'FINISHED'}


class NOSE_OT_photo_open(bpy.types.Operator):
    bl_idname='anaplast.nose_photo_open';bl_label='Open selected photo for marking'
    def execute(self,context):
        try:ph=current_photo(context.scene.nose_local)
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        area=context.area;area.type='IMAGE_EDITOR'
        def show():
            space=next((s for s in area.spaces if s.type=='IMAGE_EDITOR'),None)
            if space:
                space.image=ph.image;space.show_region_ui=True
            return None
        # Editor spaces are allocated on the next UI redraw on some Blender builds.
        if any(s.type=='IMAGE_EDITOR' for s in area.spaces):show()
        elif not bpy.app.background:bpy.app.timers.register(show,first_interval=.05)
        return {'FINISHED'}


class NOSE_OT_photo_point(bpy.types.Operator):
    bl_idname='anaplast.nose_photo_point';bl_label='Click this point in photograph';bl_options={'REGISTER','UNDO'}
    @classmethod
    def poll(cls,context):return context.area and context.area.type=='IMAGE_EDITOR' and len(context.scene.nose_local.photos)>0
    def invoke(self,context,event):
        p=context.scene.nose_local;ph=current_photo(p)
        if context.space_data.image!=ph.image:self.report({'ERROR'},'Open the selected photo first');return {'CANCELLED'}
        if p.key not in mapping(p):self.report({'ERROR'},'Mark this landmark on the template first');return {'CANCELLED'}
        self.key=p.key;self.photo_index=min(p.photo_index-1,len(p.photos)-1)
        context.window_manager.modal_handler_add(self);context.window.cursor_modal_set('CROSSHAIR')
        context.area.header_text_set('Click '+dict((k,n) for k,n,_ in KEYS)[self.key]+' · Esc cancel')
        return {'RUNNING_MODAL'}
    def finish(self,context):context.area.header_text_set(None);context.window.cursor_modal_restore()
    def modal(self,context,event):
        if event.type in {'ESC','RIGHTMOUSE'}:self.finish(context);return {'CANCELLED'}
        if event.type=='LEFTMOUSE' and event.value=='PRESS':
            region=next(r for r in context.area.regions if r.type=='WINDOW')
            uv=region.view2d.region_to_view(event.mouse_x-region.x,event.mouse_y-region.y)
            if not all(0<=x<=1 for x in uv):return {'RUNNING_MODAL'}
            ph=context.scene.nose_local.photos[self.photo_index];points=json.loads(ph.points)
            points[self.key]=list(uv);ph.points=json.dumps(points)
            self.finish(context);return {'FINISHED'}
        return {'PASS_THROUGH'}


class NOSE_OT_photo_remove_point(bpy.types.Operator):
    bl_idname='anaplast.nose_photo_remove_point';bl_label='Remove selected photo point';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        p=context.scene.nose_local
        try:ph=current_photo(p);pts=json.loads(ph.points);pts.pop(p.key,None);ph.points=json.dumps(pts)
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}


class NOSE_OT_photo_remove(bpy.types.Operator):
    bl_idname='anaplast.nose_photo_remove';bl_label='Remove photo from fitting';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        p=context.scene.nose_local
        if not len(p.photos):return {'CANCELLED'}
        p.photos.remove(min(p.photo_index-1,len(p.photos)-1));p.photo_index=max(1,min(p.photo_index,len(p.photos)))
        p.cameras='[]';p.status='Photo removed from fitting; source image file unchanged. Fit again.'
        return {'FINISHED'}


class NOSE_OT_fit(bpy.types.Operator):
    bl_idname='anaplast.nose_fit';bl_label='Fit one shape to all marked photographs';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        p=context.scene.nose_local
        try:
            mean,basis,faces,*_=data(p);points=mapping(p);photos=[];used=[]
            for i,ph in enumerate(p.photos):
                marks=json.loads(ph.points)
                if not marks:continue
                if ph.image is None:raise ValueError('A marked photograph is missing')
                keys=[k for k in marks if k in points]
                dims=np.array(ph.image.size[:],dtype=float)
                if dims.min()<=0:raise ValueError('Photo could not be read')
                xy=np.array([marks[k] for k in keys])*dims/dims.max()
                photos.append((np.array([points[k] for k in keys],dtype=int),xy));used.append(i)
            co,b,cams,report=core.fit_photos(mean,basis[:,:,:p.modes],photos)
            pose=patient_pose(context,p,co)[0] if p.proposal and p.proposal.get('nose_patient_aligned') else None
            p.coefficients=json.dumps(b.tolist());p.cameras=json.dumps({'indices':used,'cameras':cams.tolist()})
            p.width=p.projection=p.height=1.
            refresh(p,context)
            if p.proposal is None:
                obj=make_object('Nose_Photo_Proposal',co,faces);obj['nose_prototype']=True;obj['not_final_prosthesis']=True
                if p.template:obj.matrix_world=p.template.matrix_world.copy();p.template.hide_set(True)
                p.proposal=obj;p.mark_model='PROPOSAL';activate(context,obj)
            if pose is not None:p.proposal.matrix_world=pose
            sync_model_markers(p)
            errors=[r*max(p.photos[i].image.size) for i,r in zip(used,report['photo_rms_fraction'])]
            p.status='Photo fit: '+', '.join(f'{e:.1f} px' for e in errors)+' RMS. This is 2D agreement, not 3D accuracy.'
            if p.proposal:p.proposal['nose_photo_fit']=json.dumps(report)
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        self.report({'INFO'},p.status);return {'FINISHED'}


class NOSE_OT_extract(bpy.types.Operator):
    bl_idname='anaplast.nose_extract';bl_label='Create editable nose patch';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        p=context.scene.nose_local
        try:
            if p.proposal is None:raise ValueError('Position a proposal first')
            mean,basis,faces,*_=data(p);ids=np.array(json.loads(p.region),dtype=int)
            keep=np.zeros(len(mean),bool);keep[ids]=True;f=faces[keep[faces].all(1)]
            if len(f)<10:raise ValueError('Select the nose region on the template first')
            used=np.unique(f);lookup=np.full(len(mean),-1,int);lookup[used]=np.arange(len(used))
            obj=make_object('Nose_Patch_REVIEW',coefficient_shape(p)[used],lookup[f]);obj.matrix_world=p.proposal.matrix_world.copy()
            obj['not_final_prosthesis']=True;obj['nose_note']='Open anatomical patch; fit margin and create fitting surface before mold workflow'
            activate(context,obj);p.proposal.hide_set(True)
            sync_model_markers(p)
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        self.report({'INFO'},'Editable open nose patch created; not a finished sculpt');return {'FINISHED'}


class NOSE_OT_review(bpy.types.Operator):
    bl_idname='anaplast.nose_review';bl_label='Show live nose controls';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        p=context.scene.nose_local;obj=p.proposal
        if obj is None:self.report({'ERROR'},'Position or fit a proposal first');return {'CANCELLED'}
        ids=json.loads(p.region)
        if ids:
            group=obj.vertex_groups.get('Nose preview') or obj.vertex_groups.new(name='Nose preview')
            group.remove(list(range(len(obj.data.vertices))));group.add(ids,1.,'REPLACE')
            mod=obj.modifiers.get('Nose preview') or obj.modifiers.new('Nose preview','MASK');mod.vertex_group=group.name
            mod.show_viewport=True
        col=bpy.data.collections.get(COL)
        for o in context.scene.objects:
            if (col and o.name in col.objects) or COL in json.loads(o.get('anaplast_legacy_collections','[]')):o.hide_set(o!=obj)
        activate(context,obj);sync_model_markers(p);return {'FINISHED'}


class NOSE_OT_back(bpy.types.Operator):
    bl_idname='anaplast.nose_back';bl_label='Back to 3D view'
    def execute(self,context):context.area.type='VIEW_3D';return {'FINISHED'}


class NOSE_PT_main(bpy.types.Panel):
    bl_label='4 · Nose from photographs — prototype';bl_idname='NOSE_PT_main'
    bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='Anaplast';bl_options={'DEFAULT_CLOSED'}
    def draw(self,context):
        p=context.scene.nose_local;l=self.layout
        l.label(text='Local files only · no online generation',icon='LOCKED')
        box=l.box();box.label(text='1. Prepare reusable template')
        box.operator('anaplast.nose_prepared')
        box.operator('anaplast.nose_load');box.prop(p,'model_path',text='Model')
        if p.template:
            row=box.row(align=True);row.operator('anaplast.nose_auto_marks',icon='TRACKER')
            row.operator('anaplast.nose_auto_marks',text='Reset defaults').reset=True
            box.prop(p,'show_model_marks');box.prop(p,'marker_size')
            box.operator('anaplast.nose_show_template')
            box.prop(p,'mark_model')
            box.prop(p,'key',text='Landmark');box.operator('anaplast.nose_model_point')
            box.label(text=f'{len(mapping(p))} model landmarks ready')
            box.operator('anaplast.nose_region');box.operator('anaplast.nose_library')
        box=l.box();box.label(text='2. Position on patient')
        box.label(text='Confirm surviving LM_ landmarks first')
        box.operator('anaplast.nose_place')
        box=l.box();box.label(text='3. Fit photographs')
        box.operator('anaplast.nose_photo_add')
        if len(p.photos):
            box.prop(p,'photo_index',text='Photo number');box.operator('anaplast.nose_photo_open')
        box.prop(p,'modes');box.operator('anaplast.nose_fit')
        box.label(text='Distant photos; perspective selfies unsupported')
        box=l.box();box.label(text='4. Review nose')
        box.operator('anaplast.nose_review')
        box.prop(p,'width');box.prop(p,'projection');box.prop(p,'height');box.operator('anaplast.nose_extract')
        for start in range(0,len(p.status),55):l.label(text=p.status[start:start+55])


class NOSE_PT_photo(bpy.types.Panel):
    bl_label='Nose photo points';bl_idname='NOSE_PT_photo'
    bl_space_type='IMAGE_EDITOR';bl_region_type='UI';bl_category='Nose'
    def draw(self,context):
        p=context.scene.nose_local;l=self.layout
        l.prop(p,'photo_index',text='Photo number');l.operator('anaplast.nose_photo_open');l.operator('anaplast.nose_photo_remove')
        l.prop(p,'key',text='Point');l.operator('anaplast.nose_photo_point');l.operator('anaplast.nose_photo_remove_point')
        if len(p.photos):
            ph=current_photo(p);marks=json.loads(ph.points)
            l.label(text=f'{len(marks)} points marked; at least 8 required')
            for key in marks:l.label(text=dict((k,n) for k,n,_ in KEYS).get(key,key),icon='CHECKMARK')
        l.label(text='Only mark visible landmarks; never guess hidden points')
        l.label(text='Green: marked · Orange: model projection')
        l.operator('anaplast.nose_fit');l.operator('anaplast.nose_back')


_classes=(NosePhoto,NoseSettings,NOSE_OT_load,NOSE_OT_prepared,NOSE_OT_model_point,NOSE_OT_place,NOSE_OT_show_template,NOSE_OT_region,NOSE_OT_auto_marks,
          NOSE_OT_library,NOSE_OT_photo_add,NOSE_OT_photo_open,NOSE_OT_photo_point,NOSE_OT_photo_remove_point,NOSE_OT_photo_remove,
          NOSE_OT_fit,NOSE_OT_extract,NOSE_OT_review,NOSE_OT_back,NOSE_PT_main,NOSE_PT_photo)


def register():
    global _draw_handle
    for c in _classes:bpy.utils.register_class(c)
    bpy.types.Scene.nose_local=PointerProperty(type=NoseSettings)
    if not bpy.app.background:_draw_handle=bpy.types.SpaceImageEditor.draw_handler_add(draw_photo_points,(),'WINDOW','POST_PIXEL')


def unregister():
    global _draw_handle
    if _draw_handle is not None:bpy.types.SpaceImageEditor.draw_handler_remove(_draw_handle,'WINDOW');_draw_handle=None
    del bpy.types.Scene.nose_local
    for c in reversed(_classes):bpy.utils.unregister_class(c)
    _cache.clear()
