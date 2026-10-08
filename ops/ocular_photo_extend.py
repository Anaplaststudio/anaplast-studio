"""Photo-only sclera extension, preserving the iris and excluding traced lids."""
import json
from pathlib import Path
import bpy
import numpy as np
from . import ocular_production as prod


def radial_border(border,center,radius,count=1024):
    points=(np.asarray(border)-center)/radius
    if len(points)<3:raise RuntimeError('Trace the photo opening before extending the sclera')
    from .ocular import signed_distance
    if signed_distance(np.zeros((1,2)),points)[0]>=0:
        raise RuntimeError('The photo opening must surround the marked iris center')
    theta=np.arange(count)*2*np.pi/count;directions=np.c_[np.cos(theta),np.sin(theta)]
    result=np.full(count,np.inf)
    for a,b in zip(points,np.roll(points,-1,axis=0)):
        v=b-a;den=directions[:,0]*v[1]-directions[:,1]*v[0]
        safe=np.where(abs(den)>1e-12,den,np.nan)
        t=(a[0]*v[1]-a[1]*v[0])/safe
        fraction=(a[0]*directions[:,1]-a[1]*directions[:,0])/safe
        valid=(t>0)&(fraction>=-1e-9)&(fraction<=1+1e-9)&np.isfinite(t)
        result=np.minimum(result,np.where(valid,t,np.inf))
    if not np.isfinite(result).all():raise RuntimeError('Close the photo opening boundary before extending its sclera')
    return directions,result


def extend(w,pixels,mapped,iris_radius,amount_mm=None):
    """Stretch radial sclera strips. Missing upper strips interpolate nearby
    photographed strips, rather than copying eyelids or fabricating vessels.
    """
    center=np.array(w.center);radius=np.array(w.radius)
    border=json.loads(w.border or '[]')
    directions,limits=radial_border(border,center,radius)
    original_limits=limits.copy()
    # Stay inside the user outline, away from eyelid-edge interpolation.
    limits=limits-(0. if w.preserve_photo else max(.025,w.photo_edge_inset_mm/iris_radius));inner=1.01
    available=limits>inner+.04
    if available.sum()<12:raise RuntimeError('The marked photo opening contains too little exposed sclera; enlarge the outline or check the iris size')
    indices=np.arange(len(limits));valid=indices[available]
    period=len(limits);x=np.r_[valid-period,valid,valid+period]
    limits=np.interp(indices,x,np.tile(limits[available],3))
    radial=np.linspace(0,1,160)
    samples=center+directions[:,None,:]*(inner+(limits[:,None]-inner)*radial[None,:])[...,None]*radius
    strips=prod.sample(pixels,samples)[...,:3]
    # Border may cut across the iris under the lids. Do not sample that skin.
    if not available.all():
        for column in range(strips.shape[1]):
            for channel in range(3):
                strips[:,column,channel]=np.interp(indices,x,np.tile(strips[available,column,channel],3))
    rho=np.linalg.norm(mapped,axis=-1)
    angle=np.mod(np.arctan2(mapped[...,1],mapped[...,0]),2*np.pi)/(2*np.pi)*period
    i=np.floor(angle).astype(int)%period;j=(i+1)%period;blend=angle-np.floor(angle)
    length=limits[i]*(1-blend)+limits[j]*blend-inner
    distance=np.maximum(0,rho-inner)
    amount=float(w.photo_extension_mm if amount_mm is None else amount_mm)/iris_radius
    if amount_mm is None and mapped.ndim==3 and min(mapped.shape[:2])>8:
        # Cover the generated texture footprint, instead of clamping the last
        # photo pixel into long radial streaks beyond the requested overlap.
        low=mapped.reshape(-1,2).min(0);high=mapped.reshape(-1,2).max(0)
        if np.all(low<0) and np.all(high>0):
            edge=np.where(directions>0,high,low)
            reach=np.min(edge/np.where(abs(directions)>1e-8,directions,np.where(directions>=0,1e-8,-1e-8)),axis=1)
            additional=np.maximum(amount,reach-limits)
            amount=additional[i]*(1-blend)+additional[j]*blend
    # Invert d = s + amount*(s/length)^2: no scaling at the iris edge.
    travel=2*distance/(1+np.sqrt(1+4*amount*distance/np.maximum(length**2,1e-10)))
    if w.preserve_photo:
        # Start the added strip at the actual boundary color. Reuse nearby
        # photographed sclera outwards without moving any visible source pixel.
        excess=np.maximum(0,distance-length)
        travel=np.where(distance<=length,distance,length/(1+excess/np.maximum(np.maximum(amount,length),1e-8)))
    t=np.clip(travel/length,0,1)*(strips.shape[1]-1)
    a=np.floor(t).astype(int);b=np.minimum(a+1,strips.shape[1]-1);fraction=t-a
    low=strips[i,a]*(1-fraction[...,None])+strips[i,b]*fraction[...,None]
    high=strips[j,a]*(1-fraction[...,None])+strips[j,b]*fraction[...,None]
    result=low*(1-blend[...,None])+high*blend[...,None]
    if w.preserve_photo:
        # Extension belongs outside the photographed opening. Preserve source
        # vessels in place instead of pulling every radial strip outwards.
        limit=original_limits[i]*(1-blend)+original_limits[j]*blend
        keep=(rho<=limit)&(rho>=.85)
        original=prod.sample(pixels,center+mapped*radius)[...,:3]
        result=np.where(keep[...,None],original,result)
    return result


def front_source(w,xy,cc,radius,edges,limbus_width=0.):
    """Fit every photographed scleral ray between the fixed iris and front rim.

    The complete source strip is used even where the photo opening is wider
    than the ocular. This prevents cropping the nasal/temporal vessels away.
    Stretch/compression is radial, never a separate vertical or horizontal pull.
    """
    q=(np.asarray(xy)-cc)/radius
    angle=np.deg2rad(w.rotation);c=np.cos(angle);s=np.sin(angle)
    photo=(np.asarray(json.loads(w.border))-np.array(w.center))/np.array(w.radius)
    if w.mirror:photo[:,0]*=-1
    photo=photo@np.array([[c,-s],[s,c]])
    directions,source=radial_border(photo,np.zeros(2),np.ones(2))
    # The front outline can contain internal pupil edges. The farthest hit
    # selects the outside silhouette rather than that internal boundary.
    target=np.full(len(directions),-np.inf)
    for a,b in (np.asarray(edges)-cc)/radius:
        v=b-a;den=directions[:,0]*v[1]-directions[:,1]*v[0]
        safe=np.where(abs(den)>1e-12,den,np.nan)
        t=(a[0]*v[1]-a[1]*v[0])/safe
        fraction=(a[0]*directions[:,1]-a[1]*directions[:,0])/safe
        hit=(t>0)&(fraction>=-1e-9)&(fraction<=1+1e-9)&np.isfinite(t)
        target=np.maximum(target,np.where(hit,t,-np.inf))
    if not np.isfinite(target).all():raise RuntimeError('The ocular front must surround the iris to fit its photo')
    rho=np.linalg.norm(q,axis=-1)
    theta=np.mod(np.arctan2(q[...,1],q[...,0]),2*np.pi)*len(directions)/(2*np.pi)
    i=np.floor(theta).astype(int)%len(directions);j=(i+1)%len(directions);fraction=theta-np.floor(theta)
    inner=1+limbus_width/(2*radius)
    source_limit=source[i]*(1-fraction)+source[j]*fraction
    target_limit=target[i]*(1-fraction)+target[j]*fraction
    valid=(source_limit>inner+.015)&(target_limit>inner+.015)
    travel=np.clip((rho-inner)/np.maximum(target_limit-inner,1e-8),0,1)
    sample_radius=inner+travel*(source_limit-inner)
    changed=valid&(rho>inner)
    mapped=q*np.where(changed,sample_radius/np.maximum(rho,1e-8),1)[...,None]
    uvq=np.stack((mapped[...,0]*c-mapped[...,1]*s,mapped[...,0]*s+mapped[...,1]*c),axis=-1)
    if w.mirror:uvq[...,0]*=-1
    return np.array(w.center)+uvq*np.array(w.radius),changed,mapped


def front_stretch(w,xy,cc,radius,edges,fallback,limbus_width=0.):
    uv,changed,_=front_source(w,xy,cc,radius,edges,limbus_width)
    rgb=prod.sample(prod.pixels(w.image),uv)[...,:3]
    exposure=0 if w.preserve_photo else (w.iris_exposure if w.link_exposure and w.image==w.id_data.ocular_production.photo else w.exposure)
    return np.where(changed[...,None],np.clip(rgb*2**exposure,0,1),fallback)


def front_edges(core,mapping,materials=None):
    """Recover the built front outline from UVs, independent of current gaze."""
    if 'front_edges' in mapping and materials is None:return np.asarray(mapping['front_edges'])
    edges={};uv=core.data.uv_layers.active
    for face in core.data.polygons:
        if (materials[face.index] if materials is not None else face.material_index)!=0:continue
        loops=list(face.loop_indices)
        for a,b in zip(loops,loops[1:]+loops[:1]):
            key=tuple(sorted((core.data.loops[a].vertex_index,core.data.loops[b].vertex_index)))
            if key in edges:edges[key]=None
            else:edges[key]=(uv.data[a].uv[:],uv.data[b].uv[:])
    result=np.array([edge for edge in edges.values() if edge is not None])
    if not len(result):raise RuntimeError('The ocular front outline is missing; regenerate the ocular once')
    return np.asarray(mapping['lo'])+result*(np.asarray(mapping['hi'])-mapping['lo'])


def front_rim_materials(core,clear):
    """Front-facing rounded rim uses the front photo, up to the silhouette."""
    source=bpy.data.objects.get(core.get('source_shell',''))
    if not source:raise RuntimeError('The original ocular shell is needed to identify its front')
    up=np.asarray(source['ocular_up']);plans=[]
    for obj,old,new,sign in ((core,1,0,1),(clear,2,1,-1)):
        materials=np.array([face.material_index for face in obj.data.polygons],np.int32)
        if obj==clear and not obj.get('continuous_body_coating'):plans.append(materials);continue
        for face in obj.data.polygons:
            if face.material_index==old and np.dot(face.normal,up)*sign>1e-6:materials[face.index]=new
        plans.append(materials)
    return plans


def back_map(w,xy,cc,radius,sclera,outline=None,limbus_width=0.):
    """Reuse the sclera on its own back map; fill the absent iris with a smooth
    continuation of nearby scleral color, never a radial fan or a second iris.
    Photographed vessels outside that central fill are retained exactly.
    """
    from .ocular_photos import sclera_rgb
    xy=np.asarray(xy);q=(xy-cc)/radius;rho=np.linalg.norm(q,axis=-1)
    result=np.array(sclera,copy=True);result[...,3]=1
    theta=np.arange(128)*2*np.pi/128;ring=cc+1.2*radius*np.c_[np.cos(theta),np.sin(theta)]
    radial=outline is not None and w.extend_photo and w.extension_method=='STRETCH'
    photo=w
    if radial:
        class BackPhoto:
            # Both surfaces share front-projected UVs. Looking from behind
            # already reverses X, so an extra image flip would cancel the
            # requested visible mirror. Only counter-flip for an unmirrored
            # rear view; reverse rotation as well to reflect ocular X, not
            # the photograph's possibly rotated horizontal axis.
            mirror=bool(w.mirror)!=bool(not w.mirror_back)
            rotation=w.rotation if w.mirror_back else -w.rotation
            def __getattr__(self,name):return getattr(w,name)
        photo=BackPhoto()
        fallback=sclera_rgb(photo,xy,cc,radius,np.broadcast_to(np.asarray(w.id_data.ocular_production.sclera),result[...,:3].shape))
        result[...,:3]=front_stretch(photo,xy,cc,radius,outline,fallback,limbus_width)
    colors=sclera_rgb(photo,ring,cc,radius,np.tile(np.asarray(w.id_data.ocular_production.sclera),(128,1)))
    if radial:colors=front_stretch(photo,ring,cc,radius,outline,colors,limbus_width)
    coefficients=np.fft.rfft(colors,axis=0)/len(colors)
    for start in range(0,len(q),64):
        part=q[start:start+64];r=np.clip(rho[start:start+64]/1.2,0,1);a=np.arctan2(part[...,1],part[...,0])
        fill=np.broadcast_to(coefficients[0].real,(*r.shape,3)).copy()
        for k in range(1,16):
            fill+=2*r[...,None]**k*(coefficients[k].real*np.cos(k*a)[...,None]-coefficients[k].imag*np.sin(k*a)[...,None])
        weight=np.clip((1.2-rho[start:start+64])/.12,0,1);weight=weight*weight*(3-2*weight)
        result[start:start+64,:,:3]=result[start:start+64,:,:3]*(1-weight[...,None])+np.clip(fill,0,1)*weight[...,None]
    return result[:,::-1].copy() if not w.mirror_back and not radial else result


def ai_extension(w,original_pixels,uv,fallback):
    """Use an externally AI-extended photograph only beyond the original mask.
    The source canvas and registration must match, so iris/pupil never move.
    """
    if not w.extended_image:raise RuntimeError('Load an AI-extended photo, or choose Photo stretch')
    pixels=prod.pixels(w.extended_image)
    if pixels.shape!=original_pixels.shape:raise RuntimeError('The AI result must keep the original image dimensions and alignment')
    from .ocular import signed_distance
    border=np.array(json.loads(w.border or '[]'))
    distance=signed_distance(np.asarray(uv).reshape(-1,2),border).reshape(np.asarray(uv).shape[:-1])
    blend=np.clip(distance/max(w.feather,.002),0,1)
    blend=blend*blend*(3-2*blend)
    blend*=np.linalg.norm((np.asarray(uv)-np.array(w.center))/np.array(w.radius),axis=-1)>1.03
    sample=prod.sample(pixels,uv)
    inside=np.all((uv>=0)&(uv<=1),axis=-1)*sample[...,3]
    return fallback*(1-(blend*inside)[...,None])+sample[...,:3]*(blend*inside)[...,None]


def projection(scene):
    core=scene.ocular_production.color_obj
    stored=core.get('artwork_projection')
    if stored:return json.loads(stored)
    source=bpy.data.objects.get(core.get('source_shell',''))
    if not source:raise RuntimeError('The original ocular shell is needed to update this older color map')
    mesh=source.get('gaze_basis_mesh') or source.data;front=mesh.attributes.get('ocular_front')
    if not front:raise RuntimeError('This older ocular has no recorded front mapping; regenerate it once')
    ids=sorted({v for face in mesh.polygons if front.data[face.index].value==1 for v in face.vertices})
    coords=np.array([mesh.vertices[i].co[:] for i in ids]);r=np.array(source['ocular_right']);u=np.array(source['ocular_up']);f=np.cross(u,r)
    xy=np.c_[coords@r,coords@f];center=np.array(source['ocular_iris_center'])
    return dict(lo=xy.min(0).tolist(),hi=xy.max(0).tolist(),center=[float(center@r),float(center@f)],iris_radius=float(core.get('iris_diameter_mm',source.get('iris_diameter_mm',12)))*.5)


def update_colors(scene):
    from .ocular_photos import sync_whole_eye
    p=scene.ocular_production;photos=scene.ocular_photos
    if not p.color_obj or not p.clear_obj or not p.texture:raise RuntimeError('Generate the ocular first')
    sync_whole_eye(scene);mapping=projection(scene)
    materials=front_rim_materials(p.color_obj,p.clear_obj)
    outline=front_edges(p.color_obj,mapping,materials[0])
    old_lo=np.asarray(mapping['lo']);old_hi=np.asarray(mapping['hi']);span=old_hi-old_lo
    # Repair existing oculars using their stored UVs, not their current gaze
    # pose. An affine rebase preserves every already-mapped photo location.
    uv_updates=[];low=np.zeros(2);high=np.ones(2)
    for obj in (p.color_obj,p.clear_obj):
        layer=obj.data.uv_layers.active
        if layer is None:raise RuntimeError('The ocular has no photo UV map; regenerate it once')
        values=np.empty(len(layer.data)*2,np.float32);layer.data.foreach_get('uv',values);values=values.reshape(-1,2)
        low=np.minimum(low,values.min(0));high=np.maximum(high,values.max(0));uv_updates.append((layer,values))
    rebase=bool(np.any(low < -1e-6) or np.any(high > 1+1e-6))
    if rebase:
        mapping=dict(mapping,lo=(old_lo+low*span).tolist(),hi=(old_lo+high*span).tolist())
    core=p.color_obj
    alignment=json.dumps([list(p.photo_center),list(p.photo_radius),p.iris_rotation,photos.mirror])
    if (core.get('pupil_photo_outline','[]')!=p.photo_pupil_outline or
        (p.photo_pupil_outline!='[]' and (core.get('pupil_photo_image')!=p.photo.name or core.get('pupil_outline_alignment')!=alignment))):
        raise RuntimeError('Pupil outline or alignment changed; rebuild the ocular to update its opening and colors together')
    class FixedGeometry:
        # Color edits must use the built pupil, even if a size slider was edited.
        pupil_diameter=float(p.color_obj.get('pupil_diameter_mm',p.pupil_diameter))
        def __getattr__(self,name):return getattr(p,name)
    width,height=p.texture.size
    X,Y=np.meshgrid(np.linspace(mapping['lo'][0],mapping['hi'][0],width),np.linspace(mapping['lo'][1],mapping['hi'][1],height))
    rgb,iris,sclera,rear_sclera=prod.artwork(FixedGeometry(),np.stack((X,Y),axis=-1),np.array(mapping['center']),mapping['iris_radius'],layers='SEPARATE_BACK',front_outline=outline)
    color=np.ones((height,width,4),np.float32);color[...,:3]=rgb
    wrap=back_map(photos,np.stack((X,Y),axis=-1),np.array(mapping['center']),mapping['iris_radius'],rear_sclera,outline,p.limbus_width) if photos.image else rear_sclera.copy()
    wrap[...,3]=1
    previous=(p.texture,photos.iris_map,photos.sclera_map,photos.sclera_wrap);new=[]
    try:
        for name,array in zip(('Ocular_Color_Texture','Ocular_Iris_Map','Ocular_Sclera_Map','Ocular_Sclera_Wrap'),(color,iris,sclera,wrap)):
            image=bpy.data.images.new(name,width=width,height=height,alpha=True);new.append(image)
            image.pixels.foreach_set(np.asarray(array,np.float32).ravel());image.update();image.pack()
    except Exception:
        for image in new:bpy.data.images.remove(image)
        raise
    replacements=dict(zip(previous,new))
    for obj,indices in zip((p.color_obj,p.clear_obj),materials):obj.data.polygons.foreach_set('material_index',indices)
    if rebase:
        for layer,values in uv_updates:layer.data.foreach_set('uv',np.asarray((values-low)/(high-low),np.float32).ravel())
    for obj in (p.color_obj,p.clear_obj):
        for material in obj.data.materials:
            if material and material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type=='TEX_IMAGE' and node.image in replacements:node.image=replacements[node.image]
    p.texture,photos.iris_map,photos.sclera_map,photos.sclera_wrap=new
    from .apparent_pupil import record
    for obj in (p.color_obj,p.clear_obj):record(obj,p)
    mapping['front_edges']=outline.tolist()
    p.color_obj['artwork_projection']=json.dumps(mapping)
    from .ocular_photos import record_settings
    record_settings(p.color_obj,p,photos)
    p.result='Photo colors updated; ocular geometry and gaze unchanged'
    return new


def save_guide(scene,directory):
    """Local files only. The user chooses the external AI editor and uploads."""
    from .ocular_variations import save_png,staging_directory
    w=scene.ocular_photos
    if not w.image:raise RuntimeError('Load the source eye photo first')
    border=json.loads(w.border or '[]')
    radial_border(border,np.array(w.center),np.array(w.radius))
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    from .ocular import signed_distance
    width,height=w.image.size;mask=np.ones((height,width,4),np.float32)
    for start in range(0,height,128):
        stop=min(height,start+128);X,Y=np.meshgrid(np.linspace(0,1,width),np.arange(start,stop)/(height-1))
        uv=np.stack((X,Y),axis=-1)
        outside=signed_distance(uv.reshape(-1,2),np.array(border)).reshape(uv.shape[:-1])>0
        iris=np.linalg.norm((uv-np.array(w.center))/np.array(w.radius),axis=-1)<=1.03
        mask[start:stop,:,:3]=(outside&~iris)[...,None]
    with staging_directory(root) as temporary:
        job=temporary/'Guide';job.mkdir();copy=w.image.copy()
        try:copy.filepath_raw=str(job/'Source.png');copy.file_format='PNG';copy.save()
        finally:bpy.data.images.remove(copy)
        save_png(mask,job/'Edit_mask.png')
        (job/'Instructions.txt').write_text('Use Source.png as the reference and Edit_mask.png as the edit guide: white may change, black must stay unchanged. Extend the existing sclera throughout the white area, replacing skin, eyelids and background there with continuous scleral texture. Match the source colors and vessel density; no whitening, extra vessels, iris resizing, pupil changes, reflections or new limbal ring. Keep the entire original canvas, exact pixel dimensions and image alignment. Return a PNG. This mask uses white=edit; adapt it if your AI editor uses the opposite convention. Review the generated result, then use Load AI-extended photo in Anaplast Studio. The module protects the original photographed opening and iris; AI supplies only the area outside. These files are saved locally; nothing was uploaded automatically.\n',encoding='utf8')
        destination=root/'Ocular_AI_extension_guide';n=1
        while destination.exists():destination=root/f'Ocular_AI_extension_guide_{n:03d}';n+=1
        job.rename(destination)
    return destination
