"""Nine export-only color trials, with geometry-preserving printed rear IDs."""
import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from . import ocular_production as prod

# Multipliers operate on linear image pixels, before PNG color encoding.
PRESETS = (
    ('V01', 'Original', 0., (1., 1., 1.), 1.),
    ('V02', 'Lighter', .12, (1., 1., 1.), 1.),
    ('V03', 'Darker', -.12, (1., 1., 1.), 1.),
    ('V04', 'Warmer', 0., (1.035, 1., .965), 1.),
    ('V05', 'Cooler', 0., (.965, 1., 1.035), 1.),
    ('V06', 'Redder', 0., (1.05, .985, .985), 1.),
    ('V07', 'Less red', 0., (.95, 1.015, 1.015), 1.),
    ('V08', 'More saturated', 0., (1., 1., 1.), 1.08),
    ('V09', 'Less saturated', 0., (1., 1., 1.), .92),
)
GLYPHS = {
    'V': ['10001','10001','10001','10001','10001','01010','00100'],
    '0': ['01110','10001','10011','10101','11001','10001','01110'],
    '1': ['00100','01100','00100','00100','00100','00100','01110'],
    '2': ['01110','10001','00001','00010','00100','01000','11111'],
    '3': ['11110','00001','00001','01110','00001','00001','11110'],
    '4': ['00010','00110','01010','10010','11111','00010','00010'],
    '5': ['11111','10000','10000','11110','00001','00001','11110'],
    '6': ['01110','10000','10000','11110','10001','10001','01110'],
    '7': ['11111','00001','00010','00100','01000','01000','01000'],
    '8': ['01110','10001','10001','01110','10001','10001','01110'],
    '9': ['01110','10001','10001','01111','00001','00001','01110'],
}


def parameters(preset, strength):
    code, name, exposure, tint, saturation = preset
    return dict(id=code, name=name, exposure_stops=exposure*strength,
                linear_rgb_multipliers=(1+(np.array(tint)-1)*strength).tolist(),
                saturation=1+(saturation-1)*strength)


def transform(rgb, settings):
    value=np.asarray(rgb)*2**settings['exposure_stops']*settings['linear_rgb_multipliers']
    luminance=(value@np.array([.2126,.7152,.0722]))[...,None]
    return np.clip(luminance+(value-luminance)*settings['saturation'],0,1)


def vary(front, back, iris_alpha, settings, region):
    mask = iris_alpha if region=='IRIS' else 1-iris_alpha if region=='SCLERA' else np.ones_like(iris_alpha)
    # Preserve the authored black pupil in the artwork as well as its separate material.
    mask=mask*(np.max(front[...,:3],axis=-1)>.004)
    result=front.copy()
    result[...,:3]+=(transform(front[...,:3],settings)-front[...,:3])*mask[...,None]
    rear=back.copy()
    if region!='IRIS':rear[...,:3]=transform(back[...,:3],settings)
    return result,rear


def bitmap(code):
    return np.concatenate([np.array([[int(v) for v in row] for row in GLYPHS[c]],bool)
                           if i==0 else np.c_[np.zeros(7,bool),np.array([[int(v) for v in row] for row in GLYPHS[c]],bool)]
                          for i,c in enumerate(code)],axis=1)


def rear_label(obj):
    """Find a 3.4 x 1.4 mm rear patch, clear of the center/key and curved rim.

    Ray checks use the final keyed mesh. Only rear material faces are labelled;
    sharing projected UVs with the front cannot put text on the visible sclera.
    """
    src=bpy.data.objects.get(obj.get('source_shell',''))
    if not src or 'ocular_right' not in src or 'ocular_up' not in src:
        raise RuntimeError('Rebuild the ocular once to record its rear orientation before exporting labeled variations')
    right=np.array(src['ocular_right']);up=np.array(src['ocular_up']);forward=np.cross(up,right)
    basis=np.array([right,forward,up]);coords=np.array([v.co[:] for v in obj.data.vertices]);q=coords@basis.T
    uv=obj.data.uv_layers.active
    if not uv:raise RuntimeError('The ocular has no color UV map; Update ocular first')
    tree=BVHTree.FromPolygons(q.tolist(),[list(p.vertices) for p in obj.data.polygons],all_triangles=False)
    lo=q.min(0);hi=q.max(0);center=(lo+hi)*.5;span=hi-lo
    # Use a separate rear-label projection; gaze refitting may have warped the
    # original color UVs. Bake those colors into this projection below.
    fit=np.array([[1/span[0],0],[0,1/span[1]],[-lo[0]/span[0],-lo[1]/span[1]]])
    # Prefer the inferior rear edge, then try the remaining peripheral positions.
    candidates=[center[:2]+np.array([x*span[0],y*span[1]]) for y in (-.30,.30,-.20,.20,0) for x in (0,-.20,.20,-.30,.30)]
    size=np.array([3.4,1.4]);dx,dy=np.meshgrid(np.linspace(-.5,.5,15),np.linspace(-.5,.5,9))
    for candidate in candidates:
        if np.linalg.norm((candidate-center[:2])/span[:2])<.2:continue
        low=candidate-size*.5;high=candidate+size*.5;hits=[];faces=set()
        for a,b in np.c_[dx.ravel(),dy.ravel()]*size+candidate:
            hit,normal,index,distance=tree.ray_cast(Vector((a,b,lo[2]-1)),Vector((0,0,1)),float(span[2]+2))
            if hit is None or normal.z>-.8 or obj.data.polygons[index].material_index!=1:break
            hits.append(hit.z);faces.add(index)
        else:
            if np.ptp(hits)>.7:continue
            corners=np.array([[low[0],low[1]],[high[0],low[1]],[high[0],high[1]],[low[0],high[1]]])
            mapped=np.c_[corners,np.ones(4)]@fit
            if mapped.min()<0 or mapped.max()>1:continue
            # Include every rear triangle whose projection overlaps the checked patch.
            for face in obj.data.polygons:
                if face.material_index!=1 or np.dot(np.array(face.normal),up)>-.8:continue
                pts=q[list(face.vertices),:2]
                if np.all(pts.max(0)>=low) and np.all(pts.min(0)<=high):faces.add(face.index)
            return dict(center=candidate.tolist(),size_mm=size.tolist(),projection=basis.tolist(),uv_fit=fit.tolist(),faces=sorted(faces))
    raise RuntimeError('No clear 3.4 x 1.4 mm back area was found for the label away from the key. The export was not created')


def label_background(obj,back,placement):
    """Bake the selected rear faces' existing colors into the label UV map."""
    h,w=back.shape[:2];result=np.ones_like(back);filled=np.zeros((h,w),bool)
    basis=np.array(placement['projection']);fit=np.array(placement['uv_fit'])
    coords=np.array([v.co[:] for v in obj.data.vertices]);q=coords@basis.T
    mapped=np.c_[q[:,:2],np.ones(len(q))]@fit
    selected=set(placement['faces']);uv=obj.data.uv_layers.active
    obj.data.calc_loop_triangles()
    for triangle in obj.data.loop_triangles:
        if triangle.polygon_index not in selected:continue
        points=mapped[list(triangle.vertices)]*np.array([w-1,h-1])
        matrix=np.c_[points[1]-points[0],points[2]-points[0]]
        if abs(np.linalg.det(matrix))<1e-10:continue
        low=np.maximum(0,np.floor(points.min(0)).astype(int));high=np.minimum([w-1,h-1],np.ceil(points.max(0)).astype(int))
        if np.any(high<low):continue
        X,Y=np.meshgrid(np.arange(low[0],high[0]+1),np.arange(low[1],high[1]+1))
        local=np.stack((X,Y),axis=-1);bary=(local-points[0])@np.linalg.inv(matrix).T
        weights=np.concatenate((1-bary.sum(-1,keepdims=True),bary),axis=-1)
        inside=np.min(weights,axis=-1)>=-1e-5
        source=np.array([uv.data[i].uv[:] for i in triangle.loops])
        sampled=prod.sample(back,weights@source)
        result[Y[inside],X[inside]]=sampled[inside];filled[Y[inside],X[inside]]=True
    if not filled.any():raise RuntimeError('Could not map the rear label onto the ocular')
    # Texture padding avoids seams where selected and unselected faces meet.
    for _ in range(3):
        old=filled.copy()
        for axis,shift in ((0,1),(0,-1),(1,1),(1,-1)):
            neighbor=np.roll(old,shift,axis);take=neighbor&~filled
            result[take]=np.roll(result,shift,axis)[take];filled[take]=True
    return result,mapped


def stamp(back, code, placement):
    result=back.copy();h,w=back.shape[:2]
    fit=np.array(placement['uv_fit']);inverse=np.linalg.inv(fit[:2])
    grid=np.stack(np.meshgrid(np.linspace(0,1,w),np.linspace(0,1,h)),axis=-1)
    xy=(grid-fit[2])@inverse
    # Mirror horizontally when seen from behind; image rows run bottom to top.
    delta=(xy-np.array(placement['center']))/np.array(placement['size_mm'])
    ink=bitmap(code);gy,gx=ink.shape
    tx=(.5-delta[...,0])*gx;ty=(.5-delta[...,1])*gy
    inside=(tx>=0)&(tx<gx)&(ty>=0)&(ty<gy)
    mask=inside&ink[np.clip(ty.astype(int),0,gy-1),np.clip(tx.astype(int),0,gx-1)]
    if mask.sum()<60:raise RuntimeError('Artwork resolution is too low for a readable printed back label')
    result[mask,:3]=.008
    return result


def save_png(array, path):
    h,w=array.shape[:2];image=bpy.data.images.new('Temporary variation export',width=w,height=h,alpha=True,float_buffer=True)
    try:
        image.pixels.foreach_set(np.asarray(array,np.float32).ravel());image.update()
        image.filepath_raw=str(path);image.file_format='PNG';image.save()
    finally:bpy.data.images.remove(image)


def comparison_sheet(thumbnails,path):
    # A portable PNG color sheet; a companion HTML table supplies descriptive names.
    cell=360;header=42;sheet=np.ones((3*(cell+header),3*cell,4),np.float32);sheet[...,:3]=.18
    for i,(code,pixels) in enumerate(thumbnails):
        x=(i%3)*cell;y=(2-i//3)*(cell+header)
        h,w=pixels.shape[:2];xs=np.linspace(0,w-1,cell).astype(int);ys=np.linspace(0,h-1,cell).astype(int)
        sheet[y:y+cell,x:x+cell]=pixels[ys[:,None],xs]
        glyph=np.repeat(np.repeat(bitmap(code)[::-1],4,axis=0),4,axis=1)
        patch=sheet[y+cell+7:y+cell+35,x+12:x+12+glyph.shape[1],:3];patch[glyph]=.95
    save_png(sheet,path)


@contextmanager
def staging_directory(root):
    # Inherit the chosen output directory's permissions on Windows. A private
    # TemporaryDirectory ACL would otherwise follow the completed package.
    root=root.resolve();path=root/('.ocular_variations_'+uuid.uuid4().hex);path.mkdir()
    try:yield path
    finally:
        if path.exists():
            if path.resolve().parent!=root:raise RuntimeError('Unexpected export staging location')
            shutil.rmtree(path)


def export_variations(context,directory):
    p=context.scene.ocular_production;photos=context.scene.ocular_photos
    if not p.color_obj or not p.texture or not photos.iris_map:
        raise RuntimeError('Generate the ocular and its iris color layer before exporting variations')
    front=prod.pixels(p.texture);layer=prod.pixels(photos.iris_map)
    if layer.shape!=front.shape:raise RuntimeError('The color layers have different sizes; Update ocular first')
    placement=rear_label(p.color_obj)
    back=prod.pixels(photos.sclera_wrap) if photos.sclera_wrap else np.broadcast_to(np.r_[p.color_obj.data.materials[1].diffuse_color[:3],1.],front.shape).copy()
    if back.shape!=front.shape:raise RuntimeError('The sclera wrap has a different size; Update ocular first')
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    previous_result=p.result
    try:
        with staging_directory(root) as temporary:
            job=prod.export(context,temporary)  # Includes existing validity and installed-key checks.
            reference=job/'Original_reference';reference.mkdir()
            for path in list(job.iterdir()):
                if path.is_file():path.rename(reference/path.name)
            core=p.color_obj.copy();core.data=p.color_obj.data.copy()
            try:
                label_base,label_uv=label_background(p.color_obj,back,placement)
                # Export-only material assignment, not a geometric cut or raised marking.
                while len(core.data.materials)<4:core.data.materials.append(p.color_obj.data.materials[1])
                for index in placement['faces']:
                    face=core.data.polygons[index];face.material_index=3
                    for loop in face.loop_indices:core.data.uv_layers.active.data[loop].uv=label_uv[core.data.loops[loop].vertex_index]
                color_obj=job/'Color_template.obj';prod.write_obj(core,color_obj,'Ocular.mtl')
            finally:
                mesh=core.data;bpy.data.objects.remove(core);bpy.data.meshes.remove(mesh)
            entries=[];thumbnails=[]
            for preset in PRESETS:
                settings=parameters(preset,p.variation_strength);code=settings['id'];folder=job/code;folder.mkdir()
                colors,rear=vary(front,back,layer[...,3],settings,p.variation_region)
                save_png(colors,folder/'Ocular_Color.png');save_png(rear,folder/'Ocular_Sclera_Wrap.png')
                label_color=label_base.copy()
                if p.variation_region!='IRIS':label_color[...,:3]=transform(label_base[...,:3],settings)
                save_png(stamp(label_color,code,placement),folder/'Ocular_Back_Label.png')
                shutil.copyfile(color_obj,folder/'Ocular_Color.obj');shutil.copyfile(reference/'Ocular_Clear.obj',folder/'Ocular_Clear.obj')
                prod.write_materials(folder/'Ocular.mtl',p.color_obj.data.materials[1].diffuse_color[:3],'Ocular_Sclera_Wrap.png')
                with (folder/'Ocular.mtl').open('a',encoding='utf8') as handle:
                    handle.write('\nnewmtl Ocular_Variant_Label\nKd '+' '.join(map(str,p.color_obj.data.materials[1].diffuse_color[:3]))+'\nKs 0 0 0\nd 1\nillum 1\nmap_Kd Ocular_Back_Label.png\n')
                assignment=json.loads((reference/'Print_Assignment.json').read_text(encoding='utf8'))
                assignment.update(variation=settings,variation_region=p.variation_region,variation_strength=p.variation_strength,
                                  back_label=code,back_label_map='Ocular_Back_Label.png',editable_color_layers=[])
                assignment['label_note']='Printed pigment on the back; no mesh geometry changed. Preserve the separate label material when importing.'
                assignment['label_placement']={k:v for k,v in placement.items() if k!='faces'}
                if (reference/'Photo_Attribution.txt').exists():shutil.copyfile(reference/'Photo_Attribution.txt',folder/'Photo_Attribution.txt')
                (folder/'Print_Assignment.json').write_text(json.dumps(assignment,indent=2),encoding='utf8')
                entries.append(settings);thumbnails.append((code,colors[::max(1,len(colors)//360),::max(1,len(colors)//360)].copy()))
            color_obj.unlink()
            comparison_sheet(thumbnails,job/'Color_comparison.png')
            (job/'Variations.json').write_text(json.dumps(dict(algorithm='linear-rgb-v1',region=p.variation_region,strength=p.variation_strength,variants=entries,label_placement=placement),indent=2),encoding='utf8')
            rows=''.join(f'<tr><td>{e["id"]}</td><td>{e["name"]}</td><td>{e["exposure_stops"]:.3f}</td><td>{e["saturation"]:.3f}</td></tr>' for e in entries)
            (job/'Compare.html').write_text('<!doctype html><meta charset="utf-8"><title>Ocular color trials</title><style>body{font:18px system-ui;max-width:1100px;margin:32px auto;background:#eee;color:#222}img{width:100%}td,th{padding:8px;text-align:left}</style><h1>Ocular color trials</h1><p>Region: '+p.variation_region+'. Strength: '+str(p.variation_strength)+'. V01 uses the original colors. All versions share the same geometry.</p><img src="Color_comparison.png" alt="Nine color-map variations"><table><tr><th>Back ID</th><th>Variation</th><th>Exposure (stops)</th><th>Saturation</th></tr>'+rows+'</table><p>This sheet compares color maps, not rendered eye optics. Printed color depends on the printer, material and finishing. Keep the same printing process for all nine.</p>',encoding='utf8')
            (job/'Read_me.txt').write_text('Print one package from each V01–V09 folder. Each contains the same geometry, color texture, separate clear volume, and a printed ID on the back. Import both OBJ files together in millimetres with their shared coordinates. Preserve all MTL materials and texture maps, including Ocular_Variant_Label. Assign actual transparent resin to the clear volume; MTL transparency is only a preview. Do not print Original_reference: it is an unlabelled archival copy. Compare.html and Color_comparison.png identify the trials. Variations.json records exact adjustments for reproducing a choice. These are color trials, not printer calibration. Geometry, pupil lining and clear-material assignment are unchanged.\n',encoding='utf8')
            destination=root/'Ocular_Color_Variations';count=1
            while destination.exists():destination=root/f'Ocular_Color_Variations_{count:03d}';count+=1
            job.rename(destination)
        p.result='Exported nine labeled variations: '+str(destination)
        return destination
    except Exception:
        p.result=previous_result
        raise


def draw_export(layout,p):
    box=layout.box();box.prop(p,'export_variations')
    if p.export_variations:
        box.prop(p,'variation_region');box.prop(p,'variation_strength',slider=True)
        box.label(text='V01 original; V02–V09 subtle color changes')
        box.label(text='Printed back IDs; geometry stays unchanged')
    box.operator('anaplast.ocular_package',text='Export 9 labeled variations' if p.export_variations else 'Export ocular + colors',icon='EXPORT')
