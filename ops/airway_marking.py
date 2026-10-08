"""Surface outlines for draining or axially releasable nasal casting cores."""
import json, math
import bpy
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree
from ..utils import mesh as mu
from . import mold_inserts as mi, mold_auricular as aw
from .ocular_marking import contours, smooth_loop


def outline(item, mark):
    raw=json.loads(item.airway_outlines).get(mark)
    if not raw:raise RuntimeError('Draw or paint the '+mi.mark_label(item,mark)+' first')
    return smooth_loop(np.asarray(raw,float),item.airway_smoothing,256)


def radial_profile(points, origin, frame):
    """Intersect rays with the actual polygon, rejecting folds and extra crossings."""
    u,r,f=map(np.asarray,frame);points=np.asarray(points)
    xy=np.column_stack((points@r,points@f))-origin
    nxt=np.roll(xy,-1,axis=0);edges=nxt-xy
    angles=np.arange(256)*2*np.pi/256+.000013
    directions=np.column_stack((np.cos(angles),np.sin(angles)))
    radii=[];heights=[]
    for d in directions:
        den=d[0]*edges[:,1]-d[1]*edges[:,0]
        valid=abs(den)>1e-10
        t=np.zeros(len(xy));v=np.zeros(len(xy))
        t[valid]=(xy[valid,0]*edges[valid,1]-xy[valid,1]*edges[valid,0])/den[valid]
        v[valid]=(xy[valid,0]*d[1]-xy[valid,1]*d[0])/den[valid]
        hits=np.flatnonzero(valid&(t>1e-6)&(v>=0)&(v<1))
        if len(hits)!=1:
            raise RuntimeError('Opening folds back or does not surround the nostril along the withdrawal direction. Smooth or reposition its outline')
        i=hits[0];radii.append(t[i]);heights.append((points[i]*(1-v[i])+points[(i+1)%len(points)]*v[i])@u)
    return directions,np.array(radii),np.array(heights)


def shaped_rings(root, tip, frame, draft):
    u,r,f=map(np.asarray,frame)
    origin=np.array([np.mean(tip@r),np.mean(tip@f)])
    directions,rb,zb=radial_profile(root,origin,frame)
    _,rt,zt=radial_profile(tip,origin,frame)
    height=zt-zb;slope=math.tan(math.radians(draft))
    if height.min()<3:raise RuntimeError('Nostril outline must be at least 3 mm above the patient opening along the withdrawal direction')
    extra=rb-rt-height*slope
    if extra.min()<.02:
        raise RuntimeError(f'Patient opening is too narrow for straight removal: allow at least {max(0.,.02-extra.min()):.2f} mm more locally, or reposition the outlines')
    # Every angular section narrows monotonically toward the nostril. Smoothstep
    # removes the abrupt shoulder while the linear term retains release draft.
    def ring(rad,z):
        xy=origin+directions*rad[:,None]
        return xy[:,0,None]*r+xy[:,1,None]*f+z[:,None]*u
    rings=[ring(rb+.6*slope,zb-.6)]
    count=max(32,min(256,math.ceil(height.max()/.2)))
    for t in np.linspace(0,1,count+1):
        ease=t*t*(3-2*t)
        rings.append(ring(rt+height*slope*(1-t)+extra*(1-ease),zb+height*t))
    if rt.min()<=.6*slope+.1:raise RuntimeError('The nostril outline is too narrow; enlarge it slightly')
    rings.append(ring(rt-.6*slope,zt+.6))
    return rings


def draining_rings(root, tip, frame):
    """Loft the actual 3D borders without projecting them along the mold axis.

    World Z is the case's upright axis. Corresponding wall vertices descend
    toward the nostril; checking every vertex also checks the section floor.
    This is a geometry check, not a fluid-flow simulation.
    """
    root=np.asarray(root,float).copy();tip=np.asarray(tip,float).copy()
    delta=tip.mean(0)-root.mean(0)
    length=np.linalg.norm(delta)
    if length<3:raise RuntimeError('Place the nostril at least 3 mm from the patient opening')
    direction=delta/length
    for loop in (root,tip):
        centered=loop-loop.mean(0)
        normal=np.cross(centered,np.roll(centered,-1,axis=0)).sum(0)
        if np.linalg.norm(normal)<.01:raise RuntimeError('Opening has no usable area; draw a closed border again')
        if normal@direction<0:loop[:]=loop[::-1].copy()
    # Match perimeter order and phase, including tilted/non-circular nostrils.
    a=root-root.mean(0);a/=max(np.linalg.norm(a,axis=1).mean(),1e-6)
    choices=[]
    for shift in range(len(tip)):
        q=np.roll(tip,shift,axis=0);b=q-q.mean(0);b/=max(np.linalg.norm(b,axis=1).mean(),1e-6)
        rise=float(np.maximum(q[:,2]-root[:,2],0).max())
        choices.append((rise,float(((a-b)**2).sum()),shift))
    rise,_,shift=min(choices)
    if rise>.02:
        raise RuntimeError(f'The marked openings require an uphill wall ({rise:.2f} mm) toward the nostril in upright view; move the low part of the patient opening upward or the nostril downward')
    tip=np.roll(tip,shift,axis=0)
    count=max(32,min(256,math.ceil(length/.25)))
    rings=[root-np.asarray(frame[0])*.6]
    # Linear interpolation gives monotone wall heights, including non-planar
    # rims. The smoothed drawn borders determine the cross-sectional shape.
    rings.extend(root*(1-t)+tip*t for t in np.linspace(0,1,count+1))
    rings.append(tip+direction*.6)
    return rings


def cores(work,item,frame):
    roots=[outline(item,'BASE1'),outline(item,'BASE2' if item.chambers=='SEPARATE' else 'BASE1')]
    tips=[outline(item,'TIP1'),outline(item,'TIP2')]
    result=[]
    for n,(root,tip) in enumerate(zip(roots,tips),1):
        rings=draining_rings(root,tip,frame) if item.airway_path=='DRAIN' else shaped_rings(root,tip,frame,item.draft)
        obj=mi.loft(work,f'Airway_{n}_work',rings)
        for poly in obj.data.polygons:poly.use_smooth=True
        mi.closed(obj,connected=True);result.append(obj)
        if item.airway_path=='DRAIN':
            obj['airway_floor_drop_mm']=float(np.min(rings[1][:,2])-np.min(rings[-2][:,2]))
            obj['airway_path']='DRAIN'
    if item.chambers=='SEPARATE' and mi.intersection_volume(work,*result)>.001:
        raise RuntimeError('The two marked airways overlap; separate the outlines or choose a shared chamber')
    center=Vector(np.mean(np.vstack(roots),axis=0));u=np.asarray(frame[0])
    delta=np.vstack(roots)-np.array(center);delta-=np.outer(delta@u,u)
    width=item.dock_diameter if item.airway_path=='DRAIN' else max(item.dock_diameter,2*(np.linalg.norm(delta,axis=1).max()+1)/.65)
    return center,width,result,[Vector(q.mean(0)) for q in tips]


def marker(context,item,mark):
    points=outline(item,mark);name=f'InsertMark_{item.uid[:6]}_{mark}'
    old=bpy.data.objects.get(name)
    if old:mu.delete_object(old)
    curve=bpy.data.curves.new(name,'CURVE');curve.dimensions='3D';curve.bevel_depth=.055;curve.bevel_resolution=3
    sp=curve.splines.new('POLY');sp.points.add(len(points)-1);sp.use_cyclic_u=True
    for p,co in zip(sp.points,points):p.co=(*co,1)
    obj=bpy.data.objects.new(name,curve);mu.get_collection(context.scene,mi.COL).objects.link(obj)
    obj['anaplast_construction']=True;obj.show_in_front=True;obj.hide_render=True;obj.color=(1,.35,.02,1)
    curve.materials.append(mu.get_material('Airway outline',(1,.35,.02)))
    return obj


def save(context,item,mark,points,target):
    clean=smooth_loop(np.asarray(points,float),item.airway_smoothing,256)
    if not np.isfinite(clean).all():raise RuntimeError('Outline contains an invalid point; draw it again')
    data=json.loads(item.airway_outlines);data[mark]=np.asarray(points).tolist()
    item.airway_outlines=json.dumps(data);item.airway_shape='OUTLINE'
    setattr(item,mark.lower(),clean.mean(0))
    marks=set(json.loads(item.marks));marks.add(mark);item.marks=json.dumps(sorted(marks))
    sources=json.loads(item.mark_sources);sources[mark]=mi.marking_source(context.scene,target);item.mark_sources=json.dumps(sources)
    for old in context.scene.objects:
        if old.get('airway_preview')==item.uid:old.hide_set(True)
    marker(context,item,mark)


def validate_sources(item):
    mi.validate_marks(item)
    cache={}
    for mark in mi.required_marks(item):
        info=json.loads(item.mark_sources).get(mark)
        obj=bpy.data.objects.get(info['name']) if info else None
        if obj:obj=mi.marking_target(item.id_data,obj)
        if info is None or not mi.mark_matches(item,mark,obj,info,cache):
            raise RuntimeError('The marked surface changed; mark the '+mi.mark_label(item,mark)+' again')


class ANAPLAST_OT_airway_area(bpy.types.Operator):
    bl_idname='anaplast.airway_area';bl_label='Mark airway opening';bl_options={'REGISTER','UNDO'}
    mark:bpy.props.EnumProperty(items=mi.MARKS[:4])
    action:bpy.props.EnumProperty(items=[('LASSO','Smooth lasso',''),('PAINT','Paint / erase',''),('PREVIEW','Preview airway',''),('HIDE','Hide preview','')])
    @classmethod
    def poll(cls,context):
        item=mi.active(context.scene)
        return item is not None and item.kind=='AIRWAY'
    def execute(self,context):
        item=mi.active(context.scene);work=mi.Work(context.scene)
        try:
            if self.action=='HIDE':
                for o in context.scene.objects:
                    if o.get('airway_preview')==item.uid:o.hide_set(True)
                return {'FINISHED'}
            if self.action!='PREVIEW':raise RuntimeError('Draw or paint in the 3D view')
            mi.checked_hosts(context.scene,mi.state(context.scene))
            validate_sources(item)
            base=bpy.data.objects.get('Mold_Base')
            if base is None:raise RuntimeError('Build the Mold Base first')
            _,_,objects,_=mi.airway(work,item,aw.frame_for(base))
            for old in list(context.scene.objects):
                if old.get('airway_preview')==item.uid:mu.delete_object(old)
            for i,obj in enumerate(objects,1):
                obj.name=f'Airway preview {i}';obj['airway_preview']=item.uid;obj['anaplast_construction']=True
                obj.color=(.12,.7,.85,1);obj.show_in_front=True;obj.hide_render=True
                obj.data.materials.append(mu.get_material('Airway preview',(.12,.7,.85)))
            work.finish(objects)
            for mark in mi.required_marks(item):
                if mark in json.loads(item.airway_outlines):marker(context,item,mark)
            checks=[json.loads(o['airway_cavity_report']) for o in objects if o.get('airway_cavity_report')]
            if checks:
                gap=min(c['sampled_wall_mm'] for c in checks)
                self.report({'WARNING'} if gap<item.airway_wall else {'INFO'},f'Curved airway test preview: sampled wall {gap:.2f} mm; target {item.airway_wall:.1f} mm. Removal not validated')
            else:self.report({'INFO'},'Smooth airway preview ready; Build / update inserts applies it to the mold')
        except Exception as exc:
            work.finish();self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}
    def invoke(self,context,event):
        if self.action in {'PREVIEW','HIDE'}:return self.execute(context)
        self.item=mi.active(context.scene)
        if context.area is None or context.area.type!='VIEW_3D':return {'CANCELLED'}
        base=bpy.data.objects.get('Mold_Base')
        self.target=base if self.mark.startswith('BASE') else aw.reference(context.scene.anaplast)
        if base is None or self.target is None:self.report({'ERROR'},'Build the base and choose the Sculpt first');return {'CANCELLED'}
        try:self.target=mi.marking_target(context.scene,self.target)
        except RuntimeError as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        mi.explode.collapse(context.scene)
        if context.object and context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        self.region=next(r for r in context.area.regions if r.type=='WINDOW');self.rv=context.space_data.region_3d
        self.tree=aw.world_tree(self.target);self.up=aw.frame_for(base)[0]
        self.points=[];self.drawing=False;self.display=[];self.visibility={o.name:o.hide_get() for o in context.view_layer.objects if o.type=='MESH'}
        for name in self.visibility:bpy.data.objects[name].hide_set(bpy.data.objects[name]!=self.target)
        if self.action=='PAINT':
            self.co=np.array([self.target.matrix_world@v.co for v in self.target.data.vertices]);self.mask=np.zeros(len(self.co),bool)
            normal_matrix=self.target.matrix_world.to_3x3().inverted().transposed()
            self.normals=np.array([(normal_matrix@v.normal).normalized() for v in self.target.data.vertices])
            self.kd=KDTree(len(self.co))
            for i,co in enumerate(self.co):self.kd.insert(co,i)
            self.kd.balance()
        self.handle=bpy.types.SpaceView3D.draw_handler_add(self.draw_overlay,(),'WINDOW','POST_VIEW')
        context.area.header_text_set('Draw a closed outline' if self.action=='LASSO' else 'Paint filled opening; Shift = erase; Ctrl + wheel = brush size')
        self.report({'INFO'},'Enter accepts a smooth outline; Backspace clears; Esc cancels. Mark only the opening')
        context.window.cursor_modal_set('CROSSHAIR');context.window_manager.modal_handler_add(self);return {'RUNNING_MODAL'}
    def draw_overlay(self):
        import gpu
        from gpu_extras.batch import batch_for_shader
        positions=self.points if self.action=='LASSO' else self.display
        if len(positions)<2:return
        shader=gpu.shader.from_builtin('UNIFORM_COLOR');shader.bind();shader.uniform_float('color',(1,.35,.02,1))
        gpu.state.point_size_set(4)
        try:batch_for_shader(shader,'LINE_STRIP' if self.action=='LASSO' else 'POINTS',{'pos':positions+[positions[0]] if self.action=='LASSO' else positions}).draw(shader)
        finally:gpu.state.point_size_set(1)
    def finish(self,context):
        bpy.types.SpaceView3D.draw_handler_remove(self.handle,'WINDOW')
        for name,hidden in self.visibility.items():
            o=bpy.data.objects.get(name)
            if o is not None:o.hide_set(hidden)
        context.area.header_text_set(None);context.window.cursor_modal_restore();context.area.tag_redraw()
    def modal(self,context,event):
        if event.type in {'ESC','RIGHTMOUSE'} and event.value=='PRESS':self.finish(context);return {'CANCELLED'}
        if event.type in {'RET','NUMPAD_ENTER'} and event.value=='PRESS':
            try:
                if self.action=='PAINT':
                    loops=contours(self.target,self.mask,self.co)
                    if len(loops)!=1:raise RuntimeError('Paint one filled opening; erase detached spots and fill any unpainted holes')
                    points=loops[0]
                else:
                    if len(self.points)<8:raise RuntimeError('Draw a complete outline with at least eight samples')
                    points=self.points
                save(context,self.item,self.mark,points,self.target)
            except Exception as exc:self.report({'WARNING'},str(exc));return {'RUNNING_MODAL'}
            self.finish(context);return {'FINISHED'}
        if event.type=='BACK_SPACE' and event.value=='PRESS':
            self.points=[];self.display=[]
            if self.action=='PAINT':self.mask[:]=False
        if self.action=='PAINT' and event.ctrl and event.type in {'WHEELUPMOUSE','WHEELDOWNMOUSE'}:
            self.item.airway_brush=max(.2,min(10,self.item.airway_brush*(1.15 if event.type=='WHEELUPMOUSE' else 1/1.15)))
            context.area.header_text_set(f'Paint radius {self.item.airway_brush:.1f} mm; Shift erase; Enter finish')
            return {'RUNNING_MODAL'}
        xy=(event.mouse_x-self.region.x,event.mouse_y-self.region.y)
        inside=0<=xy[0]<self.region.width and 0<=xy[1]<self.region.height
        if event.type=='LEFTMOUSE':
            self.drawing=event.value=='PRESS' and inside and not event.alt
            if self.drawing and self.action=='LASSO':self.points=[]
        if self.drawing and event.type in {'LEFTMOUSE','MOUSEMOVE'} and inside:
            from bpy_extras import view3d_utils as vu
            origin=vu.region_2d_to_origin_3d(self.region,self.rv,xy);direction=vu.region_2d_to_vector_3d(self.region,self.rv,xy)
            hit,normal,_,_=self.tree.ray_cast(origin,direction)
            if hit is not None and (not self.mark.startswith('BASE') or normal.dot(self.up)>.15):
                if self.action=='LASSO':
                    if not self.points or (hit-self.points[-1]).length>.08:self.points.append(hit)
                else:
                    ids=np.array([i for _,i,_ in self.kd.find_range(hit,self.item.airway_brush)],int)
                    ids=ids[(self.normals[ids]@np.array(direction)<-.05)&(abs((self.co[ids]-np.array(hit))@np.array(direction))<self.item.airway_brush*.5)]
                    self.mask[ids]=not event.shift
                    selected=np.flatnonzero(self.mask);self.display=self.co[selected[::max(1,len(selected)//50000)]].tolist()
        context.area.tag_redraw()
        return {'RUNNING_MODAL'} if event.type in {'LEFTMOUSE','BACK_SPACE'} and not event.alt or self.drawing else {'PASS_THROUGH'}


def register():bpy.utils.register_class(ANAPLAST_OT_airway_area)
def unregister():bpy.utils.unregister_class(ANAPLAST_OT_airway_area)
