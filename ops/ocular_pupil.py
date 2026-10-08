"""Traced, off-centre pupil contours shared by artwork and material geometry."""
import json,math
import numpy as np
import bpy


def contour(p):
    raw=json.loads(p.photo_pupil_outline or '[]')
    if not raw:return None
    if p.pupil_outline_image!=p.photo:
        raise RuntimeError('The pupil outline belongs to another photograph; trace the pupil in this photo')
    points=np.asarray(raw,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<6 or not np.isfinite(points).all():
        raise RuntimeError('Trace at least six points around the pupil, then press Enter')
    q=(points-np.array(p.photo_center))/np.array(p.photo_radius)
    # Smooth the clicked contour with a periodic interpolating curve. The
    # stored clicks remain editable and are never replaced by a fitted circle.
    lengths=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
    if np.any(lengths<1e-5):raise RuntimeError('Remove repeated pupil points')
    t=np.r_[0,np.cumsum(lengths)]
    samples=np.linspace(0,t[-1],256,endpoint=False)
    ids=np.searchsorted(t,samples,side='right')-1;next_ids=(ids+1)%len(q)
    v=((samples-t[ids])/lengths[ids])[:,None]
    tangent=(np.roll(q,-1,axis=0)-np.roll(q,1,axis=0))/(lengths+np.roll(lengths,1))[:,None]
    curve=(2*v**3-3*v*v+1)*q[ids]+(v**3-2*v*v+v)*lengths[ids,None]*tangent[ids]+(-2*v**3+3*v*v)*q[next_ids]+(v**3-v*v)*lengths[ids,None]*tangent[next_ids]
    if np.max(np.linalg.norm(curve,axis=1))>=.92:
        raise RuntimeError('Keep the traced pupil inside the aligned iris')
    cross=curve[:,0]*np.roll(curve[:,1],-1)-np.roll(curve[:,0],-1)*curve[:,1]
    area=cross.sum()*.5
    if abs(area)<.001:raise RuntimeError('The pupil outline is too small or crosses itself')
    center=((curve+np.roll(curve,-1,axis=0))*cross[:,None]).sum(0)/(6*area)
    radial=curve-center
    angles=np.unwrap(np.arctan2(np.r_[radial[:,1],radial[0,1]],np.r_[radial[:,0],radial[0,0]]))
    steps=np.diff(angles)
    if not (np.all(steps>1e-7) or np.all(steps<-1e-7)) or abs(abs(angles[-1]-angles[0])-2*np.pi)>.001:
        raise RuntimeError('Trace once around the pupil without crossing or doubling back')
    theta=np.mod(np.arctan2(radial[:,1],radial[:,0]),2*np.pi);order=np.argsort(theta)
    return dict(center=center,equivalent=math.sqrt(abs(area)/math.pi),theta=theta[order],radii=np.linalg.norm(radial,axis=1)[order],curve=curve)


def radial(model,theta):
    return np.interp(np.mod(theta,2*np.pi),model['theta'],model['radii'],period=2*np.pi)


def to_photo(p,q):
    a=math.radians(p.iris_rotation);c=math.cos(a);s=math.sin(a)
    result=np.stack((q[...,0]*c-q[...,1]*s,q[...,0]*s+q[...,1]*c),axis=-1)
    if p.id_data.ocular_photos.single_image and p.id_data.ocular_photos.mirror:result[...,0]*=-1
    return result


def from_photo(p,q):
    q=np.array(q,copy=True)
    if p.id_data.ocular_photos.single_image and p.id_data.ocular_photos.mirror:q[...,0]*=-1
    a=math.radians(p.iris_rotation);c=math.cos(a);s=math.sin(a)
    return np.stack((q[...,0]*c+q[...,1]*s,-q[...,0]*s+q[...,1]*c),axis=-1)


def shape(p,cc,radius):
    model=contour(p)
    if model is None:return None
    scale=p.pupil_diameter/(2*radius*model['equivalent'])
    boundary=model['center']+(model['curve']-model['center'])*scale
    if np.max(np.linalg.norm(boundary,axis=1))>=.85:raise RuntimeError('The enlarged traced pupil is too close to the iris edge')
    return model,scale,np.asarray(cc)+from_photo(p,model['center'])*radius,np.asarray(cc)+from_photo(p,boundary)*radius


def distance(p,xy,cc,radius):
    model=contour(p)
    if model is None:return np.linalg.norm(np.asarray(xy)-cc,axis=-1)-p.pupil_diameter*.5
    q=to_photo(p,(np.asarray(xy)-cc)/radius)-model['center']
    rho=np.linalg.norm(q,axis=-1);theta=np.arctan2(q[...,1],q[...,0])
    return rho*radius-radial(model,theta)*p.pupil_diameter/(2*model['equivalent'])


def source_uv(p,xy,cc,radius):
    model=contour(p)
    if model is None:return None
    q=to_photo(p,(np.asarray(xy)-cc)/radius)
    delta=q-model['center'];rho=np.linalg.norm(delta,axis=-1)
    direction=delta/np.maximum(rho[...,None],1e-12)
    theta=np.arctan2(delta[...,1],delta[...,0]);source=radial(model,theta)
    target=source*p.pupil_diameter/(2*radius*model['equivalent'])
    dot=np.sum(direction*model['center'],axis=-1)
    outer=-dot+np.sqrt(np.maximum(0,dot*dot+1-np.sum(model['center']**2)))
    if np.any(target>=outer-.02):raise RuntimeError('Traced pupil leaves too little iris; reduce pupil size')
    travel=np.where(rho<=target,rho*source/np.maximum(target,1e-8),source+(rho-target)*(outer-source)/np.maximum(outer-target,1e-8))
    source_q=model['center']+direction*travel[...,None]
    source_q=np.where((rho>=outer)[...,None],q,source_q)
    return np.asarray(p.photo_center)+source_q*np.asarray(p.photo_radius)


class OCULAR_OT_trace_pupil(bpy.types.Operator):
    bl_idname='anaplast.ocular_trace_pupil';bl_label='Trace pupil border';bl_options={'REGISTER','UNDO'}
    @classmethod
    def poll(cls,c):return c.area and c.area.type=='IMAGE_EDITOR'
    def invoke(self,c,event):
        p=c.scene.ocular_production;w=c.scene.ocular_photos
        if w.single_image:
            p.photo=w.image;p.photo_center=w.center;p.photo_radius=w.radius;p.iris_rotation=w.rotation
        if not p.photo or c.space_data.image!=p.photo:self.report({'ERROR'},'Open the eye photograph first');return {'CANCELLED'}
        self.before=(p.photo_pupil_outline,p.pupil_outline_image)
        p.photo_pupil_outline='[]';p.pupil_outline_image=p.photo
        c.window_manager.modal_handler_add(self);c.window.cursor_modal_set('CROSSHAIR')
        c.area.header_text_set('Trace pupil: click around its border; Enter finish; Backspace undo; Esc restore previous outline')
        return {'RUNNING_MODAL'}
    def finish(self,c):c.window.cursor_modal_restore();c.area.header_text_set(None);c.area.tag_redraw()
    def modal(self,c,event):
        p=c.scene.ocular_production
        if event.type in {'ESC','RIGHTMOUSE'}:
            p.photo_pupil_outline,p.pupil_outline_image=self.before;self.finish(c);return {'CANCELLED'}
        if event.value!='PRESS':return {'PASS_THROUGH'}
        if event.type in {'RET','NUMPAD_ENTER'}:
            try:
                model=contour(p)
                if model is None:raise RuntimeError('Click at least six points around the pupil')
            except RuntimeError as exc:self.report({'WARNING'},str(exc));return {'RUNNING_MODAL'}
            p.photo_pupil=model['equivalent'];p.pupil_size_mode='PHOTO'
            self.finish(c);return {'FINISHED'}
        points=json.loads(p.photo_pupil_outline)
        if event.type=='BACK_SPACE':points=points[:-1]
        elif event.type=='LEFTMOUSE':
            region=next(r for r in c.area.regions if r.type=='WINDOW');x,y=event.mouse_x-region.x,event.mouse_y-region.y
            if not (0<=x<region.width and 0<=y<region.height):return {'RUNNING_MODAL'}
            uv=region.view2d.region_to_view(x,y)
            if not all(0<=v<=1 for v in uv):return {'RUNNING_MODAL'}
            points.append(list(uv))
        else:return {'PASS_THROUGH'}
        p.photo_pupil_outline=json.dumps(points);c.area.tag_redraw();return {'RUNNING_MODAL'}


def controls(layout,p):
    layout.operator('anaplast.ocular_trace_pupil')
    if p.photo_pupil_outline!='[]':layout.label(text='Traced pupil: shape and position preserved')


def overlay(c):
    p=c.scene.ocular_production
    if not p.photo or c.space_data.image!=p.photo:return
    points=json.loads(p.photo_pupil_outline or '[]')
    if len(points)<2:return
    try:
        model=contour(p)
        if model is not None:points=np.asarray(p.photo_center)+model['curve']*np.asarray(p.photo_radius)
    except RuntimeError:pass
    import gpu
    from gpu_extras.batch import batch_for_shader
    pos=[c.region.view2d.view_to_region(*uv,clip=False) for uv in points];lines=[]
    for a,b in zip(pos,pos[1:]+pos[:1]):lines.extend((a,b))
    shader=gpu.shader.from_builtin('UNIFORM_COLOR');shader.bind();shader.uniform_float('color',(1,.35,.1,1));batch_for_shader(shader,'LINES',{'pos':lines}).draw(shader)


def register():bpy.utils.register_class(OCULAR_OT_trace_pupil)
def unregister():bpy.utils.unregister_class(OCULAR_OT_trace_pupil)
