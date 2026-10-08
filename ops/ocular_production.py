"""Offline photo-colored ocular insert with a complementary printed clear region.
The shell is the final envelope. This module never changes the source shell.
"""
import json, math, hashlib
from pathlib import Path
import numpy as np
import bpy, bmesh
from mathutils import Vector
from bpy.props import (PointerProperty, FloatProperty, FloatVectorProperty,
                       StringProperty, EnumProperty, IntProperty, BoolProperty)
from ..utils import mesh as mu
from . import ocular_pupil

COL='Anaplast_Ocular_Production'
_handle=None


def pixels(image):
    w,h=image.size
    if w<8 or h<8:raise RuntimeError('Load an eye photograph first')
    a=np.empty(w*h*4,np.float32);image.pixels.foreach_get(a)
    return a.reshape(h,w,4)


def sample(a,uv):
    h,w=a.shape[:2];xy=np.clip(uv,0,1)*np.array([w-1,h-1])
    ij=np.floor(xy).astype(int);f=xy-ij;hi=np.minimum(ij+1,[w-1,h-1])
    return (a[ij[...,1],ij[...,0]]*(1-f[...,0,None])*(1-f[...,1,None])+
            a[ij[...,1],hi[...,0]]*f[...,0,None]*(1-f[...,1,None])+
            a[hi[...,1],ij[...,0]]*(1-f[...,0,None])*f[...,1,None]+
            a[hi[...,1],hi[...,0]]*f[...,0,None]*f[...,1,None])


def photo_valid(p):
    a=pixels(p.photo);c=np.array(p.photo_center);r=np.array(p.photo_radius)
    if np.any(c-r<0) or np.any(c+r>1):raise RuntimeError('The iris outline extends outside the photograph; adjust the center and edges')
    return a


def iris_photo(p,xy,cc,iris_radius,limbal=False):
    a=photo_valid(p);q=(np.asarray(xy)-cc)/iris_radius
    theta=np.arctan2(q[...,1],q[...,0])+math.radians(p.iris_rotation)
    rho=np.linalg.norm(q,axis=-1);target=p.pupil_diameter/(2*iris_radius)
    if target>=.85:raise RuntimeError('Pupil must be smaller than the iris')
    # Resample iris annulus independently of the pupil, preserving iris fibers.
    sr=p.photo_pupil+(rho-target)*(1-p.photo_pupil)/(1-target)
    sr=np.clip(sr,0,1)+(np.clip(sr-1,0,.25)*p.limbus_photo_detail if limbal else 0)
    def photo_uv(angle):
        direction=np.stack((np.cos(angle),np.sin(angle)),axis=-1)*np.array(p.photo_radius)
        photos=p.id_data.ocular_photos
        if photos.single_image and photos.mirror:direction[...,0]*=-1
        uv=np.array(p.photo_center)+direction*sr[...,None]
        # A tightly cropped photograph may have no sclera outside the limbus.
        # Use its real edge colour there, never repeated/clamped border pixels.
        fallback=np.array(p.photo_center)+direction*np.minimum(sr,1)[...,None]
        return np.where(np.any((uv<0)|(uv>1),axis=-1)[...,None],fallback,uv)
    source_theta=theta.copy();weight=np.zeros(rho.shape)
    if p.upper_repair>0:
        delta=np.arctan2(np.sin(theta-np.pi/2),np.cos(theta-np.pi/2))
        weight=np.clip((math.radians(p.upper_repair)*.5-np.abs(delta))/.15,0,1)*np.clip((rho-.62)/.18,0,1)
        # Blend sampled colors rather than sweeping through unrelated iris angles.
    uv=photo_uv(source_theta)
    traced_uv=ocular_pupil.source_uv(p,xy,cc,iris_radius)
    if traced_uv is not None:uv=traced_uv
    rgb=sample(a,uv)[...,:3]
    preserve=p.id_data.ocular_photos.preserve_photo
    if p.upper_repair>0 and not preserve and traced_uv is None:
        mirror_uv=photo_uv(-theta)
        rgb=rgb*(1-weight[...,None])+sample(a,mirror_uv)[...,:3]*weight[...,None]
    # Photographic specular spots can be repaired from nearby iris angles.
    if p.highlight_repair>0 and not preserve and traced_uv is None:
        bright=(rgb.min(axis=-1)>.8)&(rho<1)&(rho>target)
        alt=[]
        for angle in (-.22,.22,-.45,.45):
            alt_uv=photo_uv(source_theta+angle)
            alt.append(sample(a,alt_uv)[...,:3])
        repaired=np.median(alt,axis=0)
        rgb=np.where(bright[...,None],rgb*(1-p.highlight_repair)+repaired*p.highlight_repair,rgb)
    return rgb,rho,target


def artwork(p,xy,cc,iris_radius,layers=False,front_outline=None):
    rgb,rho,target=iris_photo(p,xy,cc,iris_radius,limbal=True)
    q=(np.asarray(xy)-cc)/iris_radius
    theta=np.arctan2(q[...,1],q[...,0])+math.radians(p.iris_rotation)
    sclera=np.broadcast_to(np.array(p.sclera),rgb.shape).copy()
    # Irregular, tapered vessels entering from the periphery, with small branches.
    # Seeded procedural appearance is explicitly labeled, never measured anatomy.
    vein=np.zeros(rho.shape);rng=np.random.default_rng(327)
    for i in range(13 if p.vein_strength>0 else 0):
        angle=(0 if i%2 else np.pi)+rng.uniform(-1.1,1.1)
        tip=rng.uniform(1.16,1.8);phase=rng.uniform(0,6.28);amplitude=rng.uniform(.025,.11)
        center=angle+amplitude*np.sin(rho*rng.uniform(2,5)+phase)+rng.uniform(.012,.025)*np.sin(rho*rng.uniform(8,14)+phase*.7)
        delta=np.arctan2(np.sin(theta-center),np.cos(theta-center))
        taper=np.clip((rho-tip)/1.1,0,1)
        width=max(p.vein_width,.005)*rng.uniform(.45,1.0)*np.maximum(.15,taper)
        line=np.exp(-(delta*np.maximum(rho,1)*iris_radius/width)**2)*taper
        branch=np.zeros(rho.shape)
        for sign in (-1,1):
            junction=rng.uniform(1.9,2.8);length=rng.uniform(.4,.9);travel=np.clip((junction-rho)/length,0,1)
            fork=center+sign*(.04*travel+.14*travel**2)+.012*np.sin(travel*7)*travel
            branch_delta=np.arctan2(np.sin(theta-fork),np.cos(theta-fork))
            fade_branch=np.where((rho<junction)&(rho>junction-length),(1-travel)**.7,0)
            twig=np.exp(-(branch_delta*np.maximum(rho,1)*iris_radius/(width*.45))**2)*fade_branch*.65
            branch=np.maximum(branch,twig)
        vein=np.maximum(vein,np.maximum(line,branch))
    variation=.012*np.sin(q[...,0]*3.7+np.sin(q[...,1]*5.1))*np.sin(q[...,1]*2.3+q[...,0])
    sclera=np.clip(sclera+variation[...,None],0,1)
    blush=np.clip((rho-1.3)/2.7,0,.22)*p.vein_strength
    sclera=sclera*(1-blush[...,None])+np.array([.68,.28,.25])*blush[...,None]
    fade=np.clip((rho-1.08)/.8,0,1)*p.vein_strength
    sclera=sclera*(1-(vein*fade)[...,None])+np.array([.35,.025,.018])*(vein*fade)[...,None]
    photos=getattr(p.id_data,'ocular_photos',None)
    if photos:
        rgb=np.clip(rgb*(2**(0 if photos.preserve_photo else photos.iris_exposure)),0,1)
        if photos.image:
            from .ocular_photos import sclera_rgb
            sclera=sclera_rgb(photos,xy,cc,iris_radius,sclera)
    rear_sclera=sclera.copy() if layers=='SEPARATE_BACK' else None
    if front_outline is not None and photos and photos.image and photos.extend_photo and photos.extension_method=='STRETCH':
        from .ocular_photo_extend import front_stretch
        sclera=front_stretch(photos,xy,cc,iris_radius,front_outline,sclera,p.limbus_width)
    if p.limbus_width>0:
        t=np.clip(((rho-1)*iris_radius+p.limbus_width*.5)/p.limbus_width,0,1)
        edge=(1-t**3*(10-15*t+6*t*t))[...,None]
    else:edge=np.clip((1-rho)/.025,0,1)[...,None]
    from .apparent_pupil import apply
    rgb=apply(p,p,rgb,xy,iris_radius,cc)
    iris_rgb=rgb.copy()
    rgb=sclera*(1-edge)+rgb*edge
    pupil=np.clip(ocular_pupil.distance(p,xy,cc,iris_radius)/(iris_radius*.012),0,1)[...,None]
    result=rgb*pupil+np.array([.003,.003,.003])*(1-pupil)
    if layers:
        iris_rgb=iris_rgb*pupil+np.array([.003,.003,.003])*(1-pupil)
        output=(result,np.concatenate((iris_rgb,edge),axis=-1),np.concatenate((sclera,1-edge),axis=-1))
        return output+(np.concatenate((rear_sclera,1-edge),axis=-1),) if layers=='SEPARATE_BACK' else output
    return result


def gaussian(a,sigma):
    if sigma<.05:return a.copy()
    radius=max(1,int(math.ceil(sigma*3)));x=np.arange(-radius,radius+1)
    kernel=np.exp(-.5*(x/sigma)**2);kernel/=kernel.sum()
    out=a
    for axis in (0,1):
        pad=[(0,0),(0,0)];pad[axis]=(radius,radius)
        out=np.apply_along_axis(lambda row:np.convolve(row,kernel,mode='valid'),axis,np.pad(out,pad,mode='reflect'))
    return out


def relief_values(p,xy,cc,radius):
    """Photo contrast becomes bounded signed relief, not a measured depth map."""
    if p.relief_height<=0:return np.zeros(len(xy))
    step=min(p.relief_spacing*.5,.05)
    count=min(768,max(128,int(math.ceil(2*radius/step))+1));step=2*radius/(count-1)
    axis=np.linspace(-radius,radius,count);X,Y=np.meshgrid(axis,axis)
    grid=np.stack((X,Y),axis=-1)+cc
    rgb,rho,target=iris_photo(p,grid,cc,radius)
    lum=np.log(np.maximum(rgb@np.array([.2126,.7152,.0722]),.005))
    fine=gaussian(lum,p.relief_smooth/step)
    broad=gaussian(lum,max(.35,p.relief_smooth*3)/step)
    detail=fine-broad
    region=(ocular_pupil.distance(p,grid,cc,radius)>.04*radius)&(rho<.94)
    scale=float(np.percentile(np.abs(detail[region]),98)) if region.any() else 0
    if scale<1e-6:return np.zeros(len(xy))
    field=np.clip(detail/scale,-1,1)*p.relief_height*.5
    uv=(np.asarray(xy)-cc+radius)/(2*radius)
    value=sample(field[:,:,None],uv)[...,0]
    radial=np.linalg.norm(np.asarray(xy)-cc,axis=-1)
    t=np.clip(ocular_pupil.distance(p,xy,cc,radius)/.18,0,1)
    outer=np.clip((radius-radial)/.30,0,1)
    fade=t*t*(3-2*t)*outer*outer*(3-2*outer)
    return value*fade*(-1 if p.relief_invert else 1)


def refine_front(coords,front,backs,r,f,cc,radius,spacing):
    """Conforming linear triangle refinement. The outside surface is preserved."""
    points=list(coords);backs=dict(backs);faces=list(front)
    edge_count={}
    for face in faces:
        for a,b in zip(face,face[1:]+face[:1]):
            key=tuple(sorted((a,b)));edge_count[key]=edge_count.get(key,0)+1
    fixed_rim={key for key,count in edge_count.items() if count==1}
    for iteration in range(8):
        mids={}
        for face in faces:
            xy=np.array([[np.dot(points[i],r),np.dot(points[i],f)] for i in face])
            lengths=[np.linalg.norm(xy[j]-xy[(j+1)%3]) for j in range(3)]
            if np.min(np.linalg.norm(xy-cc,axis=1))>radius+max(lengths):continue
            for j,length in enumerate(lengths):
                if length<=spacing*1.05:continue
                a,b=face[j],face[(j+1)%3];key=tuple(sorted((a,b)))
                if key in fixed_rim:continue
                if key not in mids:
                    mid=len(points);mids[key]=mid;points.append((points[a]+points[b])*.5);backs[mid]=(backs[a]+backs[b])*.5
        if not mids:break
        if len(points)>350000:raise RuntimeError('Iris relief is too dense; increase relief surface spacing')
        refined=[]
        for face in faces:
            a,b,c=face;ab=mids.get(tuple(sorted((a,b))));bc=mids.get(tuple(sorted((b,c))));ca=mids.get(tuple(sorted((c,a))))
            n=sum(i is not None for i in (ab,bc,ca))
            if n==0:refined.append(face)
            elif n==3:refined.extend([(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)])
            elif n==1:
                if bc is not None:a,b,c,ab=b,c,a,bc
                elif ca is not None:a,b,c,ab=c,a,b,ca
                refined.extend([(a,ab,c),(ab,b,c)])
            else:
                if ab is None:a,b,c,ab,bc=b,c,a,bc,ca
                elif bc is None:a,b,c,ab,bc=c,a,b,ca,ab
                refined.extend([(ab,b,bc),(a,ab,c),(ab,bc,c)])
        faces=refined
    else:raise RuntimeError('Source mesh is too coarse for this relief spacing; reduce the source surface spacing')
    return np.array(points),faces,backs


def geometry_signature(obj):
    co=np.empty(len(obj.data.vertices)*3,np.float32);obj.data.vertices.foreach_get('co',co)
    faces=np.asarray([i for face in obj.data.polygons for i in face.vertices],dtype=np.int32)
    sizes=np.asarray([len(face.vertices) for face in obj.data.polygons],dtype=np.int32)
    # Keep existing uniform-face signatures; keyed Boolean recesses can contain
    # mixed polygons, whose boundaries must also be included in the signature.
    boundary=sizes.tobytes() if len(set(sizes))>1 else b''
    return hashlib.sha256(co.tobytes()+faces.tobytes()+boundary).hexdigest()


def make_mesh(name,coords,faces,matrix):
    bm=bmesh.new();vs=[bm.verts.new(v) for v in coords]
    for face in faces:bm.faces.new([vs[int(i)] for i in face])
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    bmesh.ops.triangulate(bm,faces=list(bm.faces))
    obj=mu.new_mesh_object(name,bm,mu.get_collection(bpy.context.scene,COL),matrix)
    for f in obj.data.polygons:f.use_smooth=True
    return obj


def material(name,color=None,image=None,clear=False):
    m=bpy.data.materials.new(name);m.use_nodes=True
    bs=m.node_tree.nodes.get('Principled BSDF')
    bs.inputs['Base Color'].default_value=(*color,1) if color is not None else (1,1,1,1)
    bs.inputs['Roughness'].default_value=.12 if clear else .6
    if not clear:bs.inputs['Specular IOR Level'].default_value=0.
    if clear:bs.inputs['Transmission Weight'].default_value=1.;bs.inputs['IOR'].default_value=1.49
    if clear:
        # Match the accepted studio preview: light can reach the pigment under
        # the clear coating. This does not change either manufacturing volume.
        tree=m.node_tree;output=tree.nodes.get('Material Output')
        path=tree.nodes.new('ShaderNodeLightPath');transparent=tree.nodes.new('ShaderNodeBsdfTransparent');mix=tree.nodes.new('ShaderNodeMixShader')
        tree.links.new(path.outputs['Is Shadow Ray'],mix.inputs[0]);tree.links.new(bs.outputs['BSDF'],mix.inputs[1])
        tree.links.new(transparent.outputs[0],mix.inputs[2]);tree.links.new(mix.outputs[0],output.inputs['Surface'])
        m['preview_note']='Transparent shadow approximation for studio visualization; manufacturing still uses the closed clear part.'
    if image:
        tex=m.node_tree.nodes.new('ShaderNodeTexImage');tex.image=image;tex.extension='EXTEND'
        m.node_tree.nodes.active=tex
        m.node_tree.links.new(tex.outputs['Color'],bs.inputs['Base Color'])
    m.diffuse_color=(*color,1) if color is not None else (.85,.85,.8,1)
    return m


def recessed_iris_bed(base, interface, xy, cc, radius, up, right, forward, p, relief):
    """A limbus-aligned plane carries the iris; blend only outside its edge."""
    q=xy-cc;radial=np.linalg.norm(q,axis=1)
    ring=np.abs(radial-radius)<max(.18,p.relief_spacing*2)
    if np.count_nonzero(ring)<12:raise RuntimeError('Too few vertices at the iris edge; refine the ocular surface')
    plane=np.linalg.lstsq(np.c_[np.ones(ring.sum()),q[ring]],base[ring]@up,rcond=None)[0]
    # Preserve the transferred XY footprint exactly. The bed's small relief
    # is added along the eye axis, beneath the unchanged clear dome.
    height=plane[0]+q@plane[1:]-p.clear_thickness-p.iris_depth+relief
    target=xy[:,0,None]*right+xy[:,1,None]*forward+height[:,None]*up
    t=np.clip((radial-radius)/p.iris_blend_width,0,1)
    blend=1-t**3*(10-15*t+6*t*t)
    return interface+(target-interface)*blend[:,None],plane


def recessed_pupil(interface,xy,center,radius,depth,up,distance=None):
    """A black cup below the iris, with a rounded lip inside the pupil edge."""
    radial=np.linalg.norm(xy-center,axis=1) if distance is None else distance+radius
    lip=min(.25,radius*.35)
    t=np.clip((radius-radial)/lip,0,1)
    weight=t**3*(10-15*t+6*t*t)
    return interface-up[None,:]*(depth*weight)[:,None],radial


def rolled_iris_rim(interface,xy,center,radius,height,width,up,blend=0.,distance=None):
    """Raised inner iris with a soft outward falloff; never blur photo detail."""
    distance=np.linalg.norm(xy-center,axis=1)-radius if distance is None else distance
    if blend<=0:
        t=np.clip(distance/width,0,1);weight=np.sin(np.pi*t)**2
    else:
        peak=width*.5
        def smooth(t):
            t=np.clip(t,0,1);return t**3*(10-15*t+6*t*t)
        # Zero slope and curvature at both ends and the crest avoid a visible
        # join where the raised inner band fades into the existing iris relief.
        weight=np.where(distance<peak,smooth(distance/peak),1-smooth((distance-peak)/(width-peak+blend)))
    return interface+up[None,:]*(height*weight)[:,None]


def widen_pupil(core,clear,interface,xy,center,radius,extra,depth,up,right,forward,boundary=None):
    """Subtract a rounded, black chamber under the iris; add its volume to clear.

    The chamber overlaps the existing pupil cup below the iris. Its roof never
    reaches the visible iris, so the mouth diameter stays unchanged.
    """
    from mathutils.bvhtree import BVHTree
    from .mold_auricular import cut
    from .mold_design import has_self_intersections
    if depth<.45:raise RuntimeError('A wider pupil cavity needs at least 0.45 mm Pupil depth')
    radial=np.linalg.norm(xy-center,axis=1);region=radial<radius+extra+.2
    plane=np.linalg.lstsq(np.c_[np.ones(region.sum()),xy[region]-center],interface[region]@up,rcond=None)[0]
    roof=.25;middle=(depth+roof)*.5;half=(depth-roof)*.5
    segments=160;rows=40;points=[];faces=[]
    if boundary is not None:
        boundary_delta=np.asarray(boundary)-center;angles=np.mod(np.arctan2(boundary_delta[:,1],boundary_delta[:,0]),2*np.pi);order=np.argsort(angles)
        edge_radii=np.linalg.norm(boundary_delta,axis=1)[order]
        def edge(angle):return float(np.interp(angle%(2*np.pi),angles[order],edge_radii,period=2*np.pi))
    else:
        def edge(angle):return radius
    def point(rad,angle,height):
        q=rad*np.array([math.cos(angle),math.sin(angle)]);pos=center+q
        return pos[0]*right+pos[1]*forward+(plane[0]+q@plane[1:]-middle+height)*up
    points.append(point(0,0,half))
    for j in range(1,rows):
        phi=math.pi*j/rows
        for k in range(segments):
            angle=2*math.pi*k/segments;points.append(point((edge(angle)+extra)*math.sin(phi),angle,half*math.cos(phi)))
    bottom=len(points);points.append(point(0,0,-half))
    for k in range(segments):
        nxt=(k+1)%segments;faces.append((0,1+k,1+nxt))
        for j in range(rows-2):
            a=1+j*segments;faces.append((a+k,a+segments+k,a+segments+nxt,a+nxt))
        a=1+(rows-2)*segments;faces.append((a+k,bottom,a+nxt))
    tree=BVHTree.FromPolygons([v.co[:] for v in core.data.vertices],[tuple(f.vertices) for f in core.data.polygons])
    heights=np.array([v.co[:] for v in core.data.vertices])@up;low=float(heights.min()-5);high=float(heights.max()+5)
    minimum=float('inf')
    for q in points:
        z=float(q@up);flat=q-up*z
        back=tree.ray_cast(Vector(flat+up*low),Vector(up),high-low)[0]
        if back is None or z-back.dot(Vector(up))<.2:
            raise RuntimeError('The wider pupil cavity reaches too close to the ocular back; reduce its width or depth')
        minimum=min(minimum,z-back.dot(Vector(up)))
        rr=np.linalg.norm(np.array([q@right,q@forward])-center)
        delta=np.array([q@right,q@forward])-center
        if rr>edge(math.atan2(delta[1],delta[0]))+.02:
            front=tree.ray_cast(Vector(flat+up*high),Vector(-up),high-low)[0]
            if front is None or front.dot(Vector(up))-z<.12:
                raise RuntimeError('The wider pupil cavity leaves too little iris above it; reduce its width or increase Pupil depth')
    cutter=make_mesh('Pupil_chamber_work',points,faces,core.matrix_world)
    try:
        for target,operation in ((core,'DIFFERENCE'),(clear,'UNION')):
            cutter.data.materials.clear()
            for mat in target.data.materials:cutter.data.materials.append(mat)
            black=next(i for i,mat in enumerate(target.data.materials) if mat==core.data.materials[2])
            for face in cutter.data.polygons:face.material_index=black
            cut(target,cutter,operation)
            if any(mu.mesh_stats(target)[1:]) or has_self_intersections(target):
                raise RuntimeError('The wider pupil did not form a clean closed material region; reduce its width or depth')
    finally:mu.delete_object(cutter)
    return minimum


def build(context):
    p=context.scene.ocular_production;old_pupil=p.pupil_diameter
    try:return build_parts(context)
    except Exception:
        p.pupil_diameter=old_pupil
        raise


def build_parts(context):
    from .ocular_photos import sync_whole_eye
    sync_whole_eye(context.scene)
    from .mold_inserts import guard_ocular_rebuild
    guard_ocular_rebuild(context.scene)
    p=context.scene.ocular_production;src=context.scene.anaplast.ocular_obj
    if not src or src.type!='MESH':raise RuntimeError('Build the scan-based ocular shell first')
    if src.get('gaze_edge_fitted'):raise RuntimeError('Reset gaze before rebuilding photo layers, then adjust gaze and refit the edge again')
    if 0<p.pupil_depth<.45 and p.pupil_underiris>0:
        raise RuntimeError('A wider pupil cavity needs at least 0.45 mm Pupil depth')
    if src.mode!='OBJECT':raise RuntimeError('Return the ocular shell to Object Mode first')
    if len(src.modifiers):raise RuntimeError('Use an ocular shell without unapplied modifiers')
    if abs(context.scene.unit_settings.scale_length-.001)>1e-8:raise RuntimeError('Use Case setup millimetre units before building the ocular')
    if any(abs(s-1)>1e-5 for s in src.matrix_world.to_scale()):raise RuntimeError('Rebuild the ocular at its intended size; scaled shells are not supported')
    attr=src.data.attributes.get('ocular_front')
    if not attr or not src.data.attributes.get('ocular_back_index') or not src.get('ocular_iris_center'):raise RuntimeError('Rebuild the ocular shell once to record its front surface')
    photo_valid(p)
    workflow=bool(src.get('workflow_sphere_radius_mm'))
    if workflow:
        from .ocular_steps import ball_data,region_hash,cornea_ratio
        marks,scan,ball,wu,wr,wf=ball_data(context);w=context.scene.ocular_steps
        if not src.get('workflow_has_iris'):raise RuntimeError('Transfer the iris and build the cornea before adding photo layers')
        if src['workflow_ball_signature']!=ball['geometry_signature'] or src['workflow_iris_signature']!=region_hash(marks,'IRIS'):raise RuntimeError('Ball or iris changed; build the cornea again')
        desired_iris=w.iris_manual_diameter if w.iris_size_mode=='MANUAL' else w.measured_iris_diameter or w.iris_diameter
        if abs(src['iris_diameter_mm']-desired_iris)>1e-5:raise RuntimeError('Iris size changed; Update ocular to rebuild the iris and cornea')
        if abs(src['workflow_cornea_height_mm']-w.cornea_height)>1e-6 or abs(src['workflow_cornea_radius_mm']-w.iris_diameter*.5*cornea_ratio(w))>1e-6 or abs(src['workflow_shell_thickness_mm']-w.thickness)>1e-6:raise RuntimeError('Geometry settings changed; build the cornea again')
        if abs(src.get('workflow_edge_radius_mm',0)-min(w.edge_radius,w.thickness*.45))>1e-6 or abs(src.get('workflow_inner_flatten',0)-w.inner_flatten)>1e-6 or src.get('workflow_thickness_basis','CENTER')!=w.thickness_basis:raise RuntimeError('Edge or inner-curve settings changed; rebuild the ocular and cornea')
        if w.thickness_basis=='RIM' and abs(src.get('workflow_rim_reference_flatten',0)-w.rim_reference_flatten)>1e-6:raise RuntimeError('Reference fitting curve changed; rebuild the ocular and cornea')
        if w.thickness_basis=='RIM' and abs(src.get('workflow_back_blend_depth_mm',0)-w.back_blend_depth)>1e-6:raise RuntimeError('Back blend depth changed; rebuild the ocular and cornea')
    r=np.array(src['ocular_right']);u=np.array(src['ocular_up']);f=np.cross(u,r)
    cc=np.array(src['ocular_iris_center']);cc2=np.array([cc@r,cc@f])
    radius=context.scene.anaplast.ocular_iris_diameter*.5
    # The built shell stores its iris diameter, so later UI edits cannot silently move the design.
    radius=float(src.get('iris_diameter_mm',2*radius))*.5
    if p.apparent_pupil:
        from .apparent_pupil import validate
        validate(p,radius)
        p.pupil_diameter=p.pupil_min
    elif p.pupil_size_mode=='PHOTO':
        traced=ocular_pupil.contour(p)
        photo_pupil=2*radius*(traced['equivalent'] if traced is not None else p.photo_pupil)
        if not .5<=photo_pupil<=10:raise RuntimeError('Photo-derived pupil must be between 0.5 and 10 mm; check photo alignment or choose Set diameter')
        p.pupil_diameter=photo_pupil
    pupil_shape=ocular_pupil.shape(p,cc2,radius)
    pupil_extent=max(np.linalg.norm(pupil_shape[3]-cc2,axis=1)) if pupil_shape is not None else p.pupil_diameter*.5
    if p.limbus_width>=2*(radius-p.pupil_diameter*.5-.1):
        raise RuntimeError('Limbal blend is too wide for this iris and pupil; reduce Limbal blend width')
    front=[tuple(face.vertices) for face in src.data.polygons if attr.data[face.index].value==1]
    coords=np.array([v.co[:] for v in src.data.vertices])
    pair=src.data.attributes['ocular_back_index']
    original_ids=sorted({i for face in front for i in face})
    back_ids=np.array([pair.data[i].value-1 for i in original_ids])
    if np.any(back_ids<0) or np.any(back_ids>=len(coords)):raise RuntimeError('Shell correspondence was edited; rebuild the ocular shell')
    backs={i:coords[j] for i,j in zip(original_ids,back_ids)}
    if p.relief_height>0 or p.pupil_depth>0:
        spacing=min(p.relief_spacing,p.pupil_diameter*.06) if p.pupil_depth>0 else p.relief_spacing
        coords,front,backs=refine_front(coords,front,backs,r,f,cc2,radius,spacing)
    if p.pupil_rim_height>0:
        rim_extent=p.pupil_rim_width+p.pupil_rim_blend
        if pupil_extent+rim_extent>radius-.2:
            raise RuntimeError('The pupil border and its blend must stay inside the iris; reduce Rim width or Iris blend')
        coords,front,backs=refine_front(coords,front,backs,r,f,cc2,pupil_extent+rim_extent,max(.03,min(p.relief_spacing,p.pupil_rim_width/5)))
    cornea_adjustment=0.
    clear_outer_coords=coords
    if workflow:
        from .cornea_surface import reproject
        clear_outer_coords,cornea_adjustment=reproject(coords,front,len(src.data.vertices),src)
    ids=sorted({i for face in front for i in face});index={v:i for i,v in enumerate(ids)}
    outer=coords[ids];xy=np.column_stack((outer@r,outer@f))
    rho=np.linalg.norm(xy-cc2,axis=1)/radius
    if p.pupil_diameter>=1.7*radius:raise RuntimeError('Pupil must be smaller than the iris')
    relief=relief_values(p,xy,cc2,radius)
    recess=p.clear_thickness+p.iris_depth*np.maximum(0,1-rho**4)**2-relief
    if np.any(recess<.02):raise RuntimeError('Iris relief reaches the outside; reduce relief height or increase clear cover')
    back_positions=np.array([backs[i] for i in ids])
    base=outer
    if workflow:
        from .ocular_smooth import sphere_height
        if src.get('workflow_edge_radius_mm',0)>0 or src.get('workflow_inner_flatten',0)>0:
            from .ocular_steps import cornea_profile
            cr=src['workflow_cornea_radius_mm'];rise=cornea_profile(np.linalg.norm(xy-cc2,axis=1)/cr,cr,src['workflow_cornea_height_mm'])
            base=outer-rise[:,None]*u
        elif src.get('workflow_angular_extension'):
            center=np.array(src['workflow_sphere_center_world']);radial=back_positions-center
            base=center+radial*(src['workflow_sphere_radius_mm']/np.linalg.norm(radial,axis=1))[:,None]
        else:
            from .ocular_fit import height as fitted_height
            height=fitted_height(xy,np.array(src['workflow_sphere_center_local']),src['workflow_sphere_radius_mm'],json.loads(src.get('workflow_surface_model','null')))
            base=xy[:,0,None]*r+xy[:,1,None]*f+height[:,None]*u
    direction=back_positions-base;available=np.linalg.norm(direction,axis=1)
    if np.any(recess>=available-.01):raise RuntimeError('Clear region reaches through the back; increase shell thickness or reduce clear/iris depth')
    full_wrap=workflow and src.get('workflow_back_blend_depth_mm',0)>0
    body_faces=front+[tuple(fa.vertices) for fa in src.data.polygons if attr.data[fa.index].value!=1]
    if full_wrap:
        body_base=coords.copy();body_base[ids]=base
        triangles=np.array(body_faces,dtype=int)
        face_normals=np.cross(body_base[triangles[:,1]]-body_base[triangles[:,0]],body_base[triangles[:,2]]-body_base[triangles[:,0]])
        source_normals=np.column_stack([np.bincount(triangles.ravel(),np.repeat(face_normals[:,k],3),minlength=len(coords)) for k in range(3)])
        source_normals/=np.linalg.norm(source_normals,axis=1)[:,None]
        interface=base-source_normals[ids]*recess[:,None]
    elif workflow:
        coating_direction=np.array(src['workflow_sphere_center_world'])-base
        coating_direction/=np.linalg.norm(coating_direction,axis=1)[:,None]
        interface=base+coating_direction*recess[:,None]
    else:interface=base+direction*(recess/available)[:,None]
    iris_plane=None
    if p.flat_iris:
        interface,iris_plane=recessed_iris_bed(base,interface,xy,cc2,radius,u,r,f,p,relief)
        outward=base-back_positions;outward/=np.linalg.norm(outward,axis=1)[:,None]
        if np.any(np.sum((outer-interface)*outward,axis=1)<.02) or np.any(np.sum((interface-back_positions)*outward,axis=1)<.05):
            raise RuntimeError('The recessed iris leaves too little clear cover or body; reduce iris recess or increase body thickness')
    iris_interface=interface.copy()
    pupil_distance=ocular_pupil.distance(p,xy,cc2,radius)
    if p.pupil_rim_height>0:
        interface=rolled_iris_rim(interface,xy,cc2,p.pupil_diameter*.5,p.pupil_rim_height,p.pupil_rim_width,u,p.pupil_rim_blend,pupil_distance)
        outward=base-back_positions;outward/=np.linalg.norm(outward,axis=1)[:,None]
        if np.any(np.sum((outer-interface)*outward,axis=1)<.02):
            raise RuntimeError('The rolled pupil border reaches the outer cornea; reduce Pupil rim height')
    if p.pupil_depth>0 and p.pupil_underiris>0 and pupil_extent+p.pupil_underiris>radius-.3:
        raise RuntimeError('The wider pupil cavity must stay inside the iris; reduce Under-iris widening or Pupil diameter')
    interface,pupil_radial=recessed_pupil(interface,xy,cc2,p.pupil_diameter*.5,p.pupil_depth,u,pupil_distance)
    outward=base-back_positions;outward/=np.linalg.norm(outward,axis=1)[:,None]
    remaining=np.sum((interface-back_positions)*outward,axis=1)-(p.clear_thickness if full_wrap else 0.)
    pupil_mask=pupil_radial<p.pupil_diameter*.5
    if p.pupil_depth>0 and np.any(remaining[pupil_mask]<.2):
        raise RuntimeError('The pupil recess reaches too close to the ocular back; reduce Pupil depth or increase the ocular thickness')
    pupil_vertices={ids[j] for j in np.flatnonzero(pupil_mask)} if p.pupil_depth>0 else set()
    core_coords=coords.copy()
    if full_wrap:
        core_coords=body_base-source_normals*p.clear_thickness
    core_coords[ids]=interface
    # The rounded wall follows the recessed pigment boundary at its front end,
    # and retains the fitting surface at its rear end.
    wall_parent=src.data.attributes.get('ocular_rim_front_index');wall_weight=src.data.attributes.get('ocular_rim_front_weight')
    if wall_parent and wall_weight and not full_wrap:
        for i,item in enumerate(wall_parent.data):
            parent=item.value-1
            if parent>=0:
                delta=core_coords[parent]-coords[parent];weight=wall_weight.data[i].value
                core_coords[i]+=delta*weight
    edges={}
    for face in front:
        for a,b in zip(face,face[1:]+face[:1]):
            key=tuple(sorted((a,b)));edges.setdefault(key,[]).append((a,b))
    rim=[v[0] for v in edges.values() if len(v)==1]
    n=len(ids);ff=[tuple(index[i] for i in face) for face in front]
    clear_faces=ff+[tuple(n+i for i in reversed(face)) for face in ff]
    clear_faces += [(index[a],n+index[a],n+index[b],index[b]) for a,b in rim]
    created=[];img=None;layer_images=[]
    try:
        core=make_mesh('Ocular_Color_Work',core_coords,body_faces,src.matrix_world);created.append(core)
        if workflow and src.get('workflow_thickness_basis')!='RIM':
            from .ocular_steps import fitting_normals
            ri=src['workflow_sphere_radius_mm']-src['workflow_shell_thickness_mm'];flat=src.get('workflow_inner_flatten',0)
            offset=ri*flat if src.get('workflow_thickness_basis','CENTER')=='CENTER' else 0.
            fitting_normals(core,back_ids,np.array(src['workflow_sphere_center_world']),u,ri,flat,offset)
        if full_wrap:
            shell_count=len(coords)
            clear=make_mesh('Ocular_Clear_Work',np.vstack((clear_outer_coords,core_coords)),body_faces+[tuple(shell_count+i for i in reversed(face)) for face in body_faces],src.matrix_world)
            # A clear enclosing volume has an outward outer skin and an inward
            # cavity boundary. Recalculation alone may orient both outward.
            bm=bmesh.new();bm.from_mesh(clear.data)
            for inner in (False,True):
                faces=[face for face in bm.faces if all(v.index>=shell_count for v in face.verts)==inner]
                volume=sum(face.verts[0].co.dot(face.verts[1].co.cross(face.verts[2].co)) for face in faces)/6
                if (volume>0)==inner:
                    for face in faces:face.normal_flip()
            bm.to_mesh(clear.data);bm.free();clear.data.update()
            clear['continuous_body_coating']=True
        else:clear=make_mesh('Ocular_Clear_Work',np.vstack((clear_outer_coords[ids],interface)),clear_faces,src.matrix_world)
        created.append(clear)
        from .mold_design import has_self_intersections
        for obj in created:
            if any(mu.mesh_stats(obj)[1:]) or has_self_intersections(obj):
                if src.get('axial_offset_mm'):
                    raise RuntimeError('The copied surface folds into a material boundary. Adjust clear depth and back offset together, or refine the marked area; the visible scan surface has been preserved')
                raise RuntimeError(f'{obj.name}: material regions fold into themselves; reduce clear thickness or iris depth, or increase source shell thickness')
        # The rounded rim can project past the front patch. Include the whole
        # body so its UVs do not clamp to the last photo row and form streaks.
        body_xy=np.column_stack((coords@r,coords@f))
        lo=np.minimum(xy.min(0),body_xy.min(0));hi=np.maximum(xy.max(0),body_xy.max(0));size=p.texture_size
        xs=np.linspace(lo[0],hi[0],size);ys=np.linspace(lo[1],hi[1],size)
        # Carry the front photograph over the forward-facing rounded rim to
        # the silhouette. The rear half retains its separate mirrored sclera.
        photo_faces=[]
        for face in core.data.polygons:
            if np.dot(face.normal,u)>1e-6:photo_faces.append(tuple(face.vertices))
        photo_sets={tuple(sorted(face)) for face in photo_faces}|{tuple(sorted(face)) for face in front}
        outline_edges={}
        for face in body_faces:
            if tuple(sorted(face)) not in photo_sets:continue
            for a,b in zip(face,face[1:]+face[:1]):
                key=tuple(sorted((a,b)));outline_edges[key]=outline_edges.get(key,0)+1
        front_outline=np.array([[[coords[i]@r,coords[i]@f] for i in edge] for edge,count in outline_edges.items() if count==1])
        X,Y=np.meshgrid(xs,ys);rgb,iris_layer,sclera_layer,rear_sclera=artwork(p,np.stack((X,Y),axis=-1),cc2,radius,layers='SEPARATE_BACK',front_outline=front_outline)
        img=bpy.data.images.new('Ocular_Color_Texture',width=size,height=size,alpha=True)
        rgba=np.ones((size,size,4),np.float32);rgba[:,:,:3]=rgb
        img.pixels.foreach_set(rgba.ravel());img.update();img.pack()
        for label,layer in (('Ocular_Iris_Map',iris_layer),('Ocular_Sclera_Map',sclera_layer)):
            image=bpy.data.images.new(label,width=size,height=size,alpha=True);layer_images.append(image)
            image.pixels.foreach_set(np.asarray(layer,np.float32).ravel());image.update();image.pack()
        wrap=bpy.data.images.new('Ocular_Sclera_Wrap',width=size,height=size,alpha=True);layer_images.append(wrap)
        from .ocular_photo_extend import back_map
        wrap_pixels=back_map(context.scene.ocular_photos,np.stack((X,Y),axis=-1),cc2,radius,rear_sclera,front_outline,p.limbus_width) if context.scene.ocular_photos.image else np.array(rear_sclera,np.float32)
        wrap_pixels=np.asarray(wrap_pixels,np.float32);wrap_pixels[:,:,3]=1
        wrap.pixels.foreach_set(wrap_pixels.ravel());wrap.update();wrap.pack()
        core.data.materials.append(material('Ocular photo color',image=img))
        core.data.materials.append(material('Ocular continuous sclera',color=(1,1,1),image=wrap))
        pupil_material=material('Ocular black pupil cup',color=(.003,.003,.003))
        pupil_material.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=1.
        core.data.materials.append(pupil_material)
        clear.data.materials.append(material('Ocular printed clear',clear=True))
        # Both coincident interface faces display the pigment. This avoids ray
        # bias making the pigment vanish in Blender; export still assigns the
        # entire clear solid to transparent printer material.
        clear.data.materials.append(core.data.materials[0])
        if full_wrap:clear.data.materials.append(core.data.materials[1])
        pupil_preview_index=len(clear.data.materials);clear.data.materials.append(pupil_material)
        cuv=clear.data.uv_layers.new(name='Ocular_Interface_Preview')
        front_sets=photo_sets
        for face in clear.data.polygons:
            if full_wrap:
                inner=all(i>=shell_count for i in face.vertices)
                face.material_index=(1 if tuple(sorted(i-shell_count for i in face.vertices)) in front_sets else 2) if inner else 0
            else:face.material_index=1 if all(i>=n for i in face.vertices) else 0
            if full_wrap:
                pupil_face=all(i>=shell_count and i-shell_count in pupil_vertices for i in face.vertices)
            else:pupil_face=all(i>=n and ids[i-n] in pupil_vertices for i in face.vertices)
            if pupil_face:face.material_index=pupil_preview_index
            for loop in face.loop_indices:
                i=clear.data.loops[loop].vertex_index
                point=np.array([coords[i%shell_count]@r,coords[i%shell_count]@f]) if full_wrap else xy[i%n]
                cuv.data[loop].uv=((point-lo)/(hi-lo)).tolist()
        if full_wrap:
            roughness=np.full(shell_count,.4,np.float32);roughness[ids]=.12
            if wall_parent:
                for i,item in enumerate(wall_parent.data):
                    if item.value>0:roughness[i]=.4-.28*wall_weight.data[i].value
            rough=clear.data.attributes.new('ocular_finish','FLOAT','POINT');rough.data.foreach_set('value',np.tile(roughness,2))
            tree=clear.data.materials[0].node_tree;node=tree.nodes.new('ShaderNodeAttribute');node.attribute_name='ocular_finish';tree.links.new(node.outputs['Fac'],tree.nodes.get('Principled BSDF').inputs['Roughness'])
        uv=core.data.uv_layers.new(name='Ocular_Artwork')
        for face in core.data.polygons:
            face.material_index=0 if tuple(sorted(face.vertices)) in front_sets else 1
            if all(i in pupil_vertices for i in face.vertices):face.material_index=2
            for loop in face.loop_indices:
                v=Vector(coords[core.data.loops[loop].vertex_index])
                uv.data[loop].uv=((np.array([v.dot(Vector(r)),v.dot(Vector(f))])-lo)/(hi-lo)).tolist()
        pupil_minimum=float(remaining.min())
        if p.pupil_depth>0 and p.pupil_underiris>0:
            cup_center=pupil_shape[2] if pupil_shape is not None else cc2
            boundary=pupil_shape[3] if pupil_shape is not None else None
            cup_radius=max(np.linalg.norm(boundary-cup_center,axis=1)) if boundary is not None else p.pupil_diameter*.5
            pupil_minimum=min(pupil_minimum,widen_pupil(core,clear,iris_interface,xy,cup_center,cup_radius,p.pupil_underiris,p.pupil_depth,u,r,f,boundary))
        if workflow:
            from .cornea_surface import smooth_normals
            clear['smooth_cornea_vertices']=smooth_normals(clear,src)
            clear['cornea_refinement_max_mm']=cornea_adjustment
            clear['cornea_surface_version']=1
        for role,obj in (('COLOR',core),('CLEAR',clear)):
            obj['ocular_production_role']=role;obj['anaplast_part']='OCULAR';obj['source_shell']=src.name
            obj['clear_thickness_mm']=p.clear_thickness;obj['iris_recess_mm']=p.iris_depth
            from .apparent_pupil import record
            record(obj,p)
            from .eye_design import signature
            obj['eye_design_signature']=signature(context.scene)
            obj['pupil_depth_mm']=p.pupil_depth;obj['pupil_diameter_mm']=p.pupil_diameter
            obj['pupil_underiris_mm']=p.pupil_underiris if p.pupil_depth>0 else 0.
            obj['pupil_rim_height_mm']=p.pupil_rim_height;obj['pupil_rim_width_mm']=p.pupil_rim_width
            obj['pupil_rim_blend_mm']=p.pupil_rim_blend
            obj['pupil_rim_color']='Iris photograph';obj['pupil_interior_material']='Clear';obj['pupil_lining_material']='Black'
            obj['pupil_size_source']='APPARENT_MIN' if p.apparent_pupil else p.pupil_size_mode
            obj['pupil_photo_outline']=p.photo_pupil_outline
            obj['pupil_photo_image']=p.photo.name
            obj['pupil_outline_alignment']=json.dumps([list(p.photo_center),list(p.photo_radius),p.iris_rotation,context.scene.ocular_photos.mirror])
            obj['limbus_blend_width_mm']=p.limbus_width;obj['limbus_photo_detail']=p.limbus_photo_detail
            obj['pupil_shape']=('undercut_black_cup' if p.pupil_underiris>0 else 'recessed_black_cup') if p.pupil_depth>0 else 'flat_color'
            if pupil_shape is not None:obj['pupil_shape']='traced_'+obj['pupil_shape']
            obj['thickness_direction']='Fitted sphere radial; cornea above color surface' if workflow else ('Eye axis' if src.get('axial_offset_mm') else 'Shell normal offset')
            obj['iris_relief_height_mm']=p.relief_height
            obj['iris_relief_smoothing_mm']=p.relief_smooth
            obj['iris_relief_spacing_mm']=p.relief_spacing
            obj['iris_relief_actual_range_mm']=float(np.ptp(relief))
            obj['iris_relief_inverted']=p.relief_invert
            obj['iris_bed_shape']='recessed_plane_with_photo_relief' if p.flat_iris else 'curved_shell'
            if iris_plane is not None:obj['iris_bed_plane']=list(iris_plane);obj['iris_bed_blend_mm']=p.iris_blend_width
            obj['minimum_clear_cover_mm']=float(np.linalg.norm(clear_outer_coords[ids]-interface,axis=1).min())
            obj['minimum_color_core_mm']=pupil_minimum
            obj['built_geometry']=geometry_signature(obj)
        photos=context.scene.ocular_photos
        core['photo_appearance']='Separate iris and scleral photo layers' if photos.image else 'Iris photo; sampled sclera; generated veins'
        core['photo_layer_alignment']=json.dumps({'iris_photo_center':list(p.photo_center),'iris_photo_radius':list(p.photo_radius),'iris_rotation':p.iris_rotation,'iris_exposure':photos.iris_exposure,'sclera_photo_center':list(photos.center),'sclera_photo_radius':list(photos.radius),'sclera_rotation':photos.rotation,'sclera_mirror':photos.mirror,'sclera_exposure':photos.exposure,'sclera_boundary':json.loads(photos.border)})
        if workflow:
            core['corneal_projection_mm']=src['workflow_cornea_height_mm'];core['scleral_radius_mm']=src['workflow_sphere_radius_mm'];core['iris_diameter_mm']=src['iris_diameter_mm']
            core['body_thickness_mm']=src['workflow_shell_thickness_mm'];core['thickness_basis']=src.get('workflow_thickness_basis','CENTER');core['inner_flattening']=src.get('workflow_inner_flatten',0);core['edge_rounding_mm']=src.get('workflow_edge_radius_mm',0)
            core['center_body_thickness_mm']=src.get('workflow_center_body_thickness_mm',src['workflow_shell_thickness_mm'] if core['thickness_basis']=='CENTER' else 0)
            core['rim_anchor_error_mm']=src.get('workflow_rim_anchor_error_mm',0)
            core['back_blend_depth_mm']=src.get('workflow_back_blend_depth_mm',0)
            core['edge_construction']=src.get('workflow_edge_construction','legacy')
            core['automatic_back_setback_mm']=src.get('workflow_automatic_back_setback_mm',0)
        core['photo_credit']=p.photo_credit
        from .ocular_photos import record_settings
        record_settings(core,p,photos)
        core['artwork_projection']=json.dumps({'lo':lo.tolist(),'hi':hi.tolist(),'center':cc2.tolist(),'iris_radius':radius,'front_edges':front_outline.tolist()})
        core['upper_lid_repair_degrees']=p.upper_repair
        clear['optical_preview']='IOR 1.49 preview only; assign actual printer material'
        # A direct artwork rebuild keeps an existing fitted-globe rotation handle.
        if src.parent and src.parent.get('ocular_rotation_control'):
            for obj in (core,clear):
                world=obj.matrix_world.copy();obj.parent=src.parent
                obj.matrix_parent_inverse=src.parent.matrix_world.inverted();obj.matrix_world=world
        # Commit only after both new parts and artwork are valid.
        # Keep uninstalled pedestal settings aimed at the replacement ocular.
        # Removing the old mesh first would clear Blender's object pointers.
        replacements={old:new for old,new in ((p.color_obj,core),(p.clear_obj,clear)) if old}
        for item in context.scene.mold_inserts.items:
            if item.ocular in replacements:item.ocular=replacements[item.ocular]
        for old in (p.color_obj,p.clear_obj):
            if old and old not in created:mu.delete_object(old)
        core.name='Ocular_Color';clear.name='Ocular_Clear'
        p.color_obj=core;p.clear_obj=clear;p.texture=img;src.hide_set(True);src.hide_render=True
        photos.iris_map,photos.sclera_map,photos.sclera_wrap=layer_images
        for obj in created:obj.hide_set(False);obj.hide_render=False
        for obj in context.selected_objects:obj.select_set(False)
        core.select_set(True);clear.select_set(True);context.view_layer.objects.active=core
        p.result=f'Color + clear built; iris relief {np.ptp(relief):.3f} mm range.'
        return core,clear
    except Exception:
        for obj in created:mu.delete_object(obj)
        for image in layer_images:
            if image.users==0:bpy.data.images.remove(image)
        if img and img.users==0:bpy.data.images.remove(img)
        raise


class OcularProduction(bpy.types.PropertyGroup):
    surface_method:EnumProperty(name='Ocular front',items=[('COPY','Copy highlighted surface','Extract the original visible scan/sculpt triangles'),('FIT','Fit smooth surface','Previous quadratic approximation'),('SMOOTH','Smooth scan curvature','Measure the marked sclera radius and replace the scanned eye with a smooth spherical ocular')],default='SMOOTH')
    match_scan_iris:BoolProperty(name='Use marked iris size',default=True,description='Keep the iris diameter measured from the highlighted scan boundary')
    copy_thickness:FloatProperty(name='Back offset along eye axis (mm)',default=1.5,min=.2,max=6)
    mark_obj:PointerProperty(type=bpy.types.Object)
    orbital_preview:PointerProperty(type=bpy.types.Object)
    mark_region:EnumProperty(name='Highlight',items=[('SCLERA','Sclera','Paint the visible white of the eye'),('IRIS','Iris','Paint the visible iris, including the pupil')])
    mark_radius:FloatProperty(name='Marking brush (mm)',default=1.2,min=.1,max=10)
    mark_smooth:FloatProperty(name='Outline smoothing (mm)',default=.25,min=0,max=1)
    mark_report:StringProperty()
    photo:PointerProperty(type=bpy.types.Image)
    photo_credit:StringProperty()
    upper_repair:FloatProperty(name='Upper lid repair (degrees)',default=0,min=0,max=160,description='Reconstruct obscured outer iris from the lower sector; this is inferred detail')
    texture:PointerProperty(type=bpy.types.Image)
    color_obj:PointerProperty(type=bpy.types.Object)
    clear_obj:PointerProperty(type=bpy.types.Object)
    photo_center:FloatVectorProperty(name='Iris center',size=2,default=(.5,.5),min=0,max=1)
    photo_radius:FloatVectorProperty(name='Iris half-width / height',size=2,default=(.2,.3),min=.001,max=1)
    photo_pupil:FloatProperty(name='Photo pupil / iris radius',default=.3,min=.03,max=.8)
    photo_pupil_outline:StringProperty(default='[]')
    pupil_outline_image:PointerProperty(type=bpy.types.Image)
    sclera:FloatVectorProperty(name='Sclera color',subtype='COLOR',size=3,default=(.8,.78,.72),min=0,max=1)
    eye_design_shared:BoolProperty(default=False)
    eye_fit_scan:BoolProperty(default=False)
    eye_details:BoolProperty(default=False)
    eye_button_details:BoolProperty(default=False)
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
    pupil_diameter:FloatProperty(name='Pupil diameter (mm)',default=3.,min=.5,max=10.)
    pupil_size_mode:EnumProperty(name='Pupil size',items=[('MANUAL','Set diameter','Choose the pupil diameter in millimetres'),('PHOTO','From photo ratio','Use the aligned iris photograph pupil-to-iris ratio on update')],default='MANUAL')
    limbus_width:FloatProperty(name='Limbal blend width (mm)',default=.5,min=0,soft_max=1.2,max=3,description='Soft transition centred on the marked iris edge. Zero uses the previous narrow fade')
    limbus_photo_detail:FloatProperty(name='Limbal photo detail',default=1,min=0,max=1,subtype='FACTOR',description='Retain the real outer iris and adjacent limbal colours from the aligned photograph. Reduce if the photo edge contains eyelids or glare')
    pupil_depth:FloatProperty(name='Pupil depth (mm)',default=1.,min=0,soft_max=3.,max=8.,description='Black cup recessed behind the iris, beneath the clear cornea. Zero keeps a flat pupil. The back stays closed')
    pupil_underiris:FloatProperty(name='Under-iris widening (mm / side)',default=1.,min=0,soft_max=2.,max=4.,description='Extra chamber width on each side beneath the iris. The visible pupil opening stays the same size; all chamber surfaces are black. Zero disables widening')
    pupil_rim_height:FloatProperty(name='Pupil rim height (mm)',default=.08,min=0,soft_max=.2,max=.5,description='Tiny rounded iris-colored border raised around the pupil opening. Zero disables it')
    pupil_rim_width:FloatProperty(name='Pupil rim width (mm)',default=.25,min=.15,soft_max=.6,max=1.5,description='Width of the rolled iris-colored border outside the pupil. The opening keeps its diameter')
    pupil_rim_blend:FloatProperty(name='Iris blend (mm)',default=.75,min=0,soft_max=1.5,max=4.,description='Additional distance over which the raised pupil rim gently fades across the iris. Keeps iris photo detail. Zero restores the narrow rolled border')
    iris_rotation:FloatProperty(name='Iris rotation (degrees)',default=0,min=-180,max=180)
    clear_thickness:FloatProperty(name='Clear cover (mm)',default=.25,min=.05,max=3.,description='Cover along the shell offset direction: eye axis for copied fronts, normal for fitted fronts; confirm with printer')
    relief_height:FloatProperty(name='Iris 3D relief (mm)',default=.10,min=0,max=1.,description='Peak-to-valley bound for photo-derived mesh relief. Zero disables it; not measured anatomy')
    relief_smooth:FloatProperty(name='Relief smoothing (mm)',default=.05,min=.01,max=.3)
    relief_spacing:FloatProperty(name='Relief surface spacing (mm)',default=.10,min=.04,max=.3)
    relief_invert:bpy.props.BoolProperty(name='Invert raised / recessed detail',default=False)
    iris_depth:FloatProperty(name='Extra iris recess (mm)',default=.25,min=0,max=3.,description='Modeled iris depth below the clear cornea; not measured from the photo')
    flat_iris:bpy.props.BoolProperty(name='Flatter recessed iris',default=True,description='Place the iris on a flatter bed below its rim, preserving its marked location and diameter')
    iris_blend_width:FloatProperty(name='Iris bed transition (mm)',default=.6,min=.2,max=2.,description='Smooth transition outside the iris into the sclera')
    highlight_repair:FloatProperty(name='Repair bright reflections',default=0,min=0,max=1,description='Optional neighboring-angle replacement; review for lost pale iris detail')
    vein_strength:FloatProperty(name='Generated veins',default=.25,min=0,max=1)
    vein_width:FloatProperty(name='Vein width (mm)',default=.07,min=.01,max=.3)
    texture_size:IntProperty(name='Artwork pixels',default=2048,min=256,max=2048)
    point:EnumProperty(name='Photo point',items=[('CENTER','Iris center',''),('HORIZONTAL','Iris left/right edge',''),('VERTICAL','Iris top/bottom edge',''),('PUPIL','Pupil edge',''),('SCLERA','Clean sclera sample','')])
    result:StringProperty()
    export_variations:BoolProperty(name='Export nine color variations',default=False,description='Export V01–V09 with printed back labels and a comparison sheet; geometry is unchanged')
    variation_region:EnumProperty(name='Vary colors of',items=[('BOTH','Iris and sclera',''),('IRIS','Iris only',''),('SCLERA','Sclera only','')],default='BOTH')
    variation_strength:FloatProperty(name='Variation strength',default=1,min=.1,max=3,soft_max=2,description='1 gives subtle changes; V01 always keeps the original colors')


class OCULAR_OT_photo(bpy.types.Operator):
    bl_idname='anaplast.ocular_photo';bl_label='Load other-eye photo';bl_options={'REGISTER','UNDO'}
    filepath:StringProperty(subtype='FILE_PATH')
    filter_glob:StringProperty(default='*.png;*.jpg;*.jpeg;*.tif;*.tiff',options={'HIDDEN'})
    def invoke(self,context,event):context.window_manager.fileselect_add(self);return {'RUNNING_MODAL'}
    def execute(self,context):
        try:
            img=bpy.data.images.load(self.filepath,check_existing=True);pixels(img);img.pack()
            p=context.scene.ocular_production;p.photo=img;p.photo_credit='';p.upper_repair=0;p.photo_center=(.5,.5);p.photo_radius=(.2,.3)
            p.result='Photo loaded locally. Open and mark iris boundaries and sclera.'
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}


class OCULAR_OT_reference(bpy.types.Operator):
    bl_idname='anaplast.ocular_reference';bl_label='Load macro iris reference';bl_options={'REGISTER','UNDO'}
    choice:EnumProperty(items=[('Shillier','Blue / gold',''),('Mcorrens','Amber','')])
    def execute(self,context):
        path=Path(__file__).resolve().parent.parent/'assets/ocular'/f'{self.choice}_iris.jpg'
        result=bpy.ops.anaplast.ocular_photo(filepath=str(path))
        if result!={'FINISHED'}:return result
        p=context.scene.ocular_production
        if self.choice=='Shillier':p.photo_center=(.491,.492);p.photo_radius=(.45,.465);p.photo_pupil=.409;p.upper_repair=90
        else:p.photo_center=(.49,.513);p.photo_radius=(.35,.327);p.photo_pupil=.398;p.upper_repair=75
        p.highlight_repair=1.;p.sclera=(.80,.77,.73)
        p.photo_credit=(Path(__file__).resolve().parent.parent/'assets/ocular/Attribution.txt').read_text(encoding='utf-8')
        p.result='Public macro reference: review iris bounds and repaired areas before use.'
        return {'FINISHED'}


class OCULAR_OT_photo_view(bpy.types.Operator):
    bl_idname='anaplast.ocular_photo_view';bl_label='Open photo and mark iris'
    back:bpy.props.BoolProperty(default=False)
    def execute(self,context):
        if self.back:context.area.type='VIEW_3D';return {'FINISHED'}
        p=context.scene.ocular_production
        if not p.photo:self.report({'ERROR'},'Load an eye photograph');return {'CANCELLED'}
        area=context.area;area.type='IMAGE_EDITOR'
        def show():
            space=next((s for s in area.spaces if s.type=='IMAGE_EDITOR'),None)
            if space:space.image=p.photo;space.show_region_ui=True
        if any(s.type=='IMAGE_EDITOR' for s in area.spaces):show()
        elif not bpy.app.background:bpy.app.timers.register(show,first_interval=.05)
        return {'FINISHED'}


def set_point(p,uv):
    uv=np.array(uv);c=np.array(p.photo_center);r=np.array(p.photo_radius)
    if p.point=='CENTER':p.photo_center=uv
    elif p.point=='HORIZONTAL':p.photo_radius=(max(.001,abs(uv[0]-c[0])),r[1])
    elif p.point=='VERTICAL':p.photo_radius=(r[0],max(.001,abs(uv[1]-c[1])))
    elif p.point=='PUPIL':p.photo_pupil=float(np.linalg.norm((uv-c)/r))
    else:
        a=pixels(p.photo);offset=np.array([(x,y) for x in (-1,0,1) for y in (-1,0,1)])/np.array(p.photo.size)
        p.sclera=np.median(sample(a,uv+offset)[:,:3],axis=0)


class OCULAR_OT_photo_point(bpy.types.Operator):
    bl_idname='anaplast.ocular_photo_point';bl_label='Click selected point';bl_options={'REGISTER','UNDO'}
    @classmethod
    def poll(cls,context):return context.area and context.area.type=='IMAGE_EDITOR' and context.scene.ocular_production.photo is not None
    def invoke(self,context,event):
        if context.space_data.image!=context.scene.ocular_production.photo:return {'CANCELLED'}
        context.window_manager.modal_handler_add(self);context.window.cursor_modal_set('CROSSHAIR')
        context.area.header_text_set('Click the selected ocular photo point; Esc cancels');return {'RUNNING_MODAL'}
    def finish(self,context):context.window.cursor_modal_restore();context.area.header_text_set(None)
    def modal(self,context,event):
        if event.type in {'ESC','RIGHTMOUSE'}:self.finish(context);return {'CANCELLED'}
        if event.type=='LEFTMOUSE' and event.value=='PRESS':
            region=next(r for r in context.area.regions if r.type=='WINDOW')
            x,y=event.mouse_x-region.x,event.mouse_y-region.y
            if not(0<=x<region.width and 0<=y<region.height):return {'RUNNING_MODAL'}
            uv=region.view2d.region_to_view(x,y)
            if not all(0<=v<=1 for v in uv):return {'RUNNING_MODAL'}
            set_point(context.scene.ocular_production,uv);self.finish(context);context.area.tag_redraw();return {'FINISHED'}
        return {'PASS_THROUGH'}


class OCULAR_OT_produce(bpy.types.Operator):
    bl_idname='anaplast.ocular_produce';bl_label='Build color + printed clear';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:build(context)
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}


def write_obj(obj,path,mtl):
    me=obj.data;uv=me.uv_layers.active;lines=['# Anaplast Studio ocular; world coordinates in millimetres',f'mtllib {mtl}',f'o {obj.name}']
    for v in me.vertices:
        co=obj.matrix_world@v.co;lines.append('v '+' '.join(f'{x:.8f}' for x in co))
    if uv:
        for loop in uv.data:lines.append('vt '+' '.join(f'{x:.8f}' for x in loop.uv))
    normal_matrix=obj.matrix_world.to_3x3().inverted().transposed()
    for normal in me.corner_normals:
        n=(normal_matrix@normal.vector).normalized();lines.append('vn '+' '.join(f'{x:.8f}' for x in n))
    lines.append('s 1')
    for face in me.polygons:
        lines.append('usemtl '+('Ocular_Clear' if obj.get('ocular_production_role')=='CLEAR' else ('Ocular_Color' if face.material_index==0 else 'Ocular_Pupil' if face.material_index==2 else 'Ocular_Variant_Label' if face.material_index==3 else 'Ocular_Back')))
        lines.append('f '+' '.join(f'{me.loops[i].vertex_index+1}/{i+1}/{i+1}' if uv else f'{me.loops[i].vertex_index+1}//{i+1}' for i in face.loop_indices))
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def write_materials(path,back_color,back_map=None):
    path.write_text('newmtl Ocular_Color\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd Ocular_Color.png\n\nnewmtl Ocular_Back\nKd '+' '.join(map(str,back_color))+'\nKs 0 0 0\nNs 1\nd 1\nillum 1\n\nnewmtl Ocular_Clear\nKd 1 1 1\nKs 0.04 0.04 0.04\nNs 100\nNi 1.49\nd 0.15\nillum 4\n',encoding='utf-8')
    if back_map:path.write_text(path.read_text(encoding='utf-8').replace('\n\nnewmtl Ocular_Clear','\nmap_Kd '+back_map+'\n\nnewmtl Ocular_Clear'),encoding='utf-8')
    with path.open('a',encoding='utf-8') as file:file.write('\nnewmtl Ocular_Pupil\nKd 0.003 0.003 0.003\nKs 0 0 0\nNs 1\nd 1\nillum 1\n')


def export(context,directory):
    p=context.scene.ocular_production
    if not p.color_obj or not p.clear_obj or not p.texture:raise RuntimeError('Build both material regions first')
    if p.color_obj.get('gaze_edge_pending') or p.clear_obj.get('gaze_edge_pending'):raise RuntimeError('Apply gaze & refit edge before exporting the rotated ocular')
    if abs(context.scene.unit_settings.scale_length-.001)>1e-8:raise RuntimeError('Export requires millimetre scene units')
    if not np.allclose(np.array(p.color_obj.matrix_world),np.array(p.clear_obj.matrix_world),atol=1e-6):raise RuntimeError('The color and clear parts moved apart; restore their shared position before export')
    for obj in (p.color_obj,p.clear_obj):
        if obj.get('built_geometry')!=geometry_signature(obj):
            from . import mold_inserts
            installed=mold_inserts.state(context.scene)
            if obj.name not in installed.get('hosts',{}):raise RuntimeError('A material region was edited; rebuild the pair from the source shell before export')
            mold_inserts.checked_hosts(context.scene,installed)
            if installed.get('settings')!=mold_inserts.settings_signature(context.scene):raise RuntimeError('Build / update inserts before exporting the keyed ocular')
        if obj.mode!='OBJECT' or obj.modifiers:raise RuntimeError('Export requires Object Mode and no unapplied modifiers')
        from .mold_design import has_self_intersections
        if any(mu.mesh_stats(obj)[1:]) or has_self_intersections(obj):raise RuntimeError('A material region is no longer a closed valid surface; rebuild before export')
    # Create a new job directory; never overwrite an earlier production export.
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    job=root/'Ocular_Print';i=1
    while job.exists():job=root/f'Ocular_Print_{i:03d}';i+=1
    job.mkdir()
    copy=p.texture.copy()
    try:copy.filepath_raw=str(job/'Ocular_Color.png');copy.file_format='PNG';copy.save()
    finally:bpy.data.images.remove(copy)
    for obj,name in ((p.color_obj,'Ocular_Color.obj'),(p.clear_obj,'Ocular_Clear.obj')):write_obj(obj,job/name,'Ocular.mtl')
    write_materials(job/'Ocular.mtl',p.color_obj.data.materials[1].diffuse_color[:3],'Ocular_Sclera_Wrap.png' if context.scene.ocular_photos.sclera_wrap else None)
    bpy.data.libraries.write(str(job/'Ocular_Materials.blend'),{p.color_obj,p.clear_obj},path_remap='RELATIVE',fake_user=True)
    data={'units':'mm','color':'Ocular_Color.obj','texture':'Ocular_Color.png','clear':'Ocular_Clear.obj','registration':'Common world coordinates; do not independently auto-center parts','clear_cover_mm':p.clear_obj.get('clear_thickness_mm'),'iris_recess_mm':p.clear_obj.get('iris_recess_mm'),'status':'Material-separated design, not printer-sliced or process-validated','material_assignment':'Assign color texture to core and actual transparent resin to clear volume in printer software. OBJ transparency is only a preview hint.','appearance':'Iris photo; sampled sclera; generated veins. Iris depth is modeled, not recovered from a photograph.'}
    photos=context.scene.ocular_photos
    for image,filename in ((photos.iris_map,'Ocular_Iris_Map.png'),(photos.sclera_map,'Ocular_Sclera_Map.png'),(photos.sclera_wrap,'Ocular_Sclera_Wrap.png')):
        if image:
            copy=image.copy()
            try:copy.filepath_raw=str(job/filename);copy.file_format='PNG';copy.save()
            finally:bpy.data.images.remove(copy)
    data['appearance']=p.color_obj.get('photo_appearance','')
    data['photo_layer_alignment']=json.loads(p.color_obj.get('photo_layer_alignment','{}'))
    data['sclera_extension']=json.loads(p.color_obj.get('sclera_extension','{}'))
    data['editable_color_layers']=['Ocular_Iris_Map.png','Ocular_Sclera_Map.png'] if photos.iris_map and photos.sclera_map else []
    data['corneal_projection_mm']=p.color_obj.get('corneal_projection_mm')
    data['iris_bed']={key:p.color_obj.get(key) for key in ('iris_bed_shape','iris_bed_plane','iris_bed_blend_mm')}
    data['pupil']={key:p.color_obj.get(key) for key in ('pupil_depth_mm','pupil_diameter_mm','pupil_underiris_mm','pupil_shape','pupil_rim_height_mm','pupil_rim_width_mm','pupil_rim_blend_mm','pupil_rim_color','pupil_interior_material','pupil_lining_material')}
    data['pupil']['apparent_band']={key:p.color_obj.get(key) for key in ('apparent_pupil','apparent_pupil_max_mm','apparent_pupil_darkness','apparent_pupil_outer_darkness','apparent_pupil_outer_fade_mm')}
    data['pupil']['size_source']=p.color_obj.get('pupil_size_source','MANUAL')
    data['pupil']['photo_outline']=json.loads(p.color_obj.get('pupil_photo_outline','[]'))
    data['pupil']['size_definition']='Equal-area diameter; traced shape and center retained' if data['pupil']['photo_outline'] else 'Circular diameter'
    data['limbus']={key:p.color_obj.get(key) for key in ('limbus_blend_width_mm','limbus_photo_detail')}
    if data['iris_bed']['iris_bed_plane'] is not None:data['iris_bed']['iris_bed_plane']=list(data['iris_bed']['iris_bed_plane'])
    data['continuous_clear_coating']=bool(p.clear_obj.get('continuous_body_coating'))
    data['scleral_radius_mm']=p.color_obj.get('scleral_radius_mm')
    data['iris_diameter_mm']=p.color_obj.get('iris_diameter_mm')
    data['ocular_body']={key:p.color_obj.get(key) for key in ('body_thickness_mm','center_body_thickness_mm','thickness_basis','inner_flattening','edge_rounding_mm','back_blend_depth_mm','automatic_back_setback_mm','edge_construction')}
    data['photo_derived_mesh_relief']={key:p.color_obj.get(key) for key in ('iris_relief_height_mm','iris_relief_smoothing_mm','iris_relief_spacing_mm','iris_relief_actual_range_mm','iris_relief_inverted','minimum_clear_cover_mm','minimum_color_core_mm')}
    if p.color_obj.get('photo_credit'):
        (job/'Photo_Attribution.txt').write_text(p.color_obj['photo_credit'],encoding='utf-8')
        data['photo_attribution']='Photo_Attribution.txt'
    (job/'Print_Assignment.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    (job/'Read_me.txt').write_text('Import both OBJ files together in millimetres, keeping their shared position. Assign the clear volume explicitly to the printer transparent resin. The MTL transparency does not configure a printer. Use the PNG for the core appearance. The two solids share the same interface; they are printed together. Validate the exact material combination, colors, clear finish and dimensions with a sample before production. This package does not contain the original eye photograph.\n',encoding='utf-8')
    p.result='Exported '+str(job);return job


class OCULAR_OT_package(bpy.types.Operator):
    bl_idname='anaplast.ocular_package';bl_label='Export color + clear package'
    directory:StringProperty(subtype='DIR_PATH')
    def invoke(self,context,event):context.window_manager.fileselect_add(self);return {'RUNNING_MODAL'}
    def execute(self,context):
        try:
            if context.scene.ocular_production.export_variations:
                from .ocular_variations import export_variations
                export_variations(context,self.directory)
            else:export(context,self.directory)
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}


class OCULAR_OT_display(bpy.types.Operator):
    bl_idname='anaplast.ocular_display';bl_label='Ocular view'
    mode:EnumProperty(items=[('COLOR','Photo colors','Inspect the photo texture without lighting, shadows or clear-coat reflections'),('CLEAR','With clear cornea','Show the cornea using scene lighting in Rendered view')])
    def execute(self,context):
        from .overlay import clipping_area
        p=context.scene.ocular_production;area=clipping_area(context)
        if area is None or not p.color_obj or not p.clear_obj:return {'CANCELLED'}
        p.color_obj.hide_viewport=False;p.color_obj.hide_set(False)
        p.clear_obj.hide_viewport=False;p.clear_obj.hide_set(self.mode=='COLOR')
        sh=area.spaces.active.shading
        sh.use_scene_world=True;sh.use_scene_lights=True
        sh.use_scene_world_render=True;sh.use_scene_lights_render=True
        sh.type='RENDERED' if self.mode=='CLEAR' else 'SOLID'
        if self.mode=='COLOR':
            sh.light='FLAT';sh.color_type='TEXTURE';sh.show_shadows=False;sh.show_cavity=False;sh.show_specular_highlight=False
        area.tag_redraw();return {'FINISHED'}


def photo_controls(l,p):
    ocular_pupil.controls(l,p)
    l.prop(p,'point');l.operator('anaplast.ocular_photo_point')
    l.prop(p,'photo_center');l.prop(p,'photo_radius');l.prop(p,'photo_pupil');l.prop(p,'sclera')


def draw_workflow(l,context):
    p=context.scene.ocular_production
    box=l.box();box.label(text='5. Shared eye photograph')
    box.label(text='Choose and mark it in Eye photograph')
    photos=context.scene.ocular_photos;box.prop(photos,'iris_exposure')
    if photos.image:box.label(text=photos.image.name)
    if photos.image:box.prop(photos,'extend_photo')
    box.prop(photos,'exposure');box.prop(photos,'rotation');box.prop(photos,'mirror')
    box.prop(p,'sclera')
    from .apparent_pupil import controls
    controls(box,p)
    if not p.apparent_pupil:box.prop(p,'pupil_diameter')
    box.prop(p,'pupil_depth');box.prop(p,'pupil_underiris');box.prop(p,'iris_rotation')
    row=box.row(align=True);row.prop(p,'pupil_rim_height');row.prop(p,'pupil_rim_width')
    box.prop(p,'pupil_rim_blend')
    if not p.apparent_pupil:box.prop(p,'pupil_size_mode')
    box.prop(p,'limbus_width');box.prop(p,'limbus_photo_detail')
    box.prop(p,'upper_repair');box.prop(p,'highlight_repair');box.prop(p,'vein_strength');box.prop(p,'vein_width')
    box.prop(p,'clear_thickness');box.prop(p,'flat_iris');box.prop(p,'iris_depth')
    if p.flat_iris:box.prop(p,'iris_blend_width')
    box.prop(p,'relief_height')
    if p.relief_height>0:
        box.prop(p,'relief_smooth');box.prop(p,'relief_spacing');box.prop(p,'relief_invert')
    box.prop(p,'texture_size')
    box.label(text='6. Build color layers under the clear cornea')
    box.operator('anaplast.ocular_produce')
    row=box.row(align=True)
    row.operator('anaplast.ocular_display',text='Photo colors').mode='COLOR'
    row.operator('anaplast.ocular_display',text='With clear cornea').mode='CLEAR'
    for key,label in (('color_obj','Colored region'),('clear_obj','Printed clear')):
        obj=getattr(p,key)
        if obj:box.prop(obj,'hide_viewport',text='Hide '+label)
    from .ocular_variations import draw_export
    draw_export(box,p)
    if p.result:box.label(text=p.result)


class OCULAR_PT_photo(bpy.types.Panel):
    bl_label='Eye photograph';bl_space_type='IMAGE_EDITOR';bl_region_type='UI';bl_category='Eye photo'
    def draw(self,context):
        l=self.layout;p=context.scene.ocular_production
        if context.space_data.image==context.scene.ocular_photos.image and context.scene.ocular_photos.image:
            from .ocular_photos import draw_controls
            draw_controls(l,context.scene.ocular_photos);return
        l.label(text='Mark center, two iris edges, pupil, then sclera')
        photo_controls(l,p)
        l.operator('anaplast.ocular_photo_view',text='Return to ocular model').back=True


def draw_photo():
    c=bpy.context
    if not c.area or c.area.type!='IMAGE_EDITOR':return
    from .ocular_photos import draw_overlay
    if draw_overlay(c):
        ocular_pupil.overlay(c);return
    p=c.scene.ocular_production
    if not p.photo or c.space_data.image!=p.photo:return
    import gpu
    from gpu_extras.batch import batch_for_shader
    shader=gpu.shader.from_builtin('UNIFORM_COLOR');lines=[]
    for scale in ((1,) if p.photo_pupil_outline!='[]' else (1,p.photo_pupil)):
        a=np.arange(97)*2*np.pi/96
        points=np.array(p.photo_center)+scale*np.array(p.photo_radius)*np.column_stack((np.cos(a),np.sin(a)))
        pos=[c.region.view2d.view_to_region(*uv,clip=False) for uv in points]
        for x,y in zip(pos,pos[1:]):lines.extend([x,y])
    x,y=c.region.view2d.view_to_region(*p.photo_center,clip=False);lines.extend([(x-5,y),(x+5,y),(x,y-5),(x,y+5)])
    shader.bind();shader.uniform_float('color',(1,.5,.05,1));batch_for_shader(shader,'LINES',{'pos':lines}).draw(shader)
    ocular_pupil.overlay(c)


_classes=(OcularProduction,OCULAR_OT_photo,OCULAR_OT_reference,OCULAR_OT_photo_view,OCULAR_OT_photo_point,OCULAR_OT_produce,OCULAR_OT_package,OCULAR_OT_display,OCULAR_PT_photo)

def register():
    global _handle
    for c in _classes:bpy.utils.register_class(c)
    ocular_pupil.register()
    bpy.types.Scene.ocular_production=PointerProperty(type=OcularProduction)
    if not bpy.app.background:_handle=bpy.types.SpaceImageEditor.draw_handler_add(draw_photo,(),'WINDOW','POST_PIXEL')

def unregister():
    global _handle
    if _handle is not None:bpy.types.SpaceImageEditor.draw_handler_remove(_handle,'WINDOW');_handle=None
    del bpy.types.Scene.ocular_production
    ocular_pupil.unregister()
    for c in reversed(_classes):bpy.utils.unregister_class(c)
