"""Two-region orbital scan marking. All paint is stored on a separate scan copy."""
import json,hashlib
import numpy as np
import bpy
from mathutils import Vector
from . import ocular as oc,wedge_marking as wm
from ..utils import mesh as mu
from .report import rep


def values_for(obj,region=None):
    region=region or obj.get('ocular_active_region','SCLERA')
    attr=obj.data.attributes.get('ocular_mark_'+region)
    values=np.zeros(len(obj.data.vertices),np.float32)
    if attr:attr.data.foreach_get('value',values)
    return values


def write_values(obj,values,region=None):
    region=region or obj.get('ocular_active_region','SCLERA')
    attr=obj.data.attributes.get('ocular_mark_'+region) or obj.data.attributes.new('ocular_mark_'+region,'FLOAT','POINT')
    attr.data.foreach_set('value',np.asarray(values,np.float32));obj.data.update();show_colors(obj)


def show_colors(obj):
    colors=np.tile([.3,.34,.38,1],(len(obj.data.vertices),1)).astype(np.float32)
    colors[values_for(obj,'SCLERA')>.5]=[.93,.92,.80,1]
    colors[values_for(obj,'IRIS')>.5]=[.04,.45,1.,1]
    attr=obj.data.color_attributes.get('Ocular_Marks') or obj.data.color_attributes.new(name='Ocular_Marks',type='FLOAT_COLOR',domain='POINT')
    attr.data.foreach_set('color',colors.ravel());obj.data.color_attributes.active_color=attr;obj.data.update()


def mark_signature(obj):
    return hashlib.sha256(values_for(obj,'SCLERA').tobytes()+values_for(obj,'IRIS').tobytes()).hexdigest()


def source_for_marks(context,obj=None):
    """Resolve a marking copy's original source, including older saved copies."""
    obj=obj if obj is not None else context.scene.ocular_production.mark_obj
    if obj is None:return None
    src=obj.get('ocular_source_object')
    if not isinstance(src,bpy.types.Object):src=bpy.data.objects.get(obj.get('ocular_source_name',''))
    return src if src is not None and src.name in context.scene.objects else None


def validate_source(context,src):
    if not src or src.type!='MESH':raise RuntimeError('Assign the eye scan first')
    if src.get('ocular_edit_copy'):raise RuntimeError('Select the original Scan, Mirror scan or Sculpt as the eye source')
    if src.modifiers:raise RuntimeError('Use an eye scan with applied modifiers for marking')
    if abs(context.scene.unit_settings.scale_length-.001)>1e-8:raise RuntimeError('Use millimetre case units first')


def marks_match(context,obj,src,signature):
    if obj is None or obj.type!='MESH' or not obj.get('ocular_edit_copy'):return False
    original=source_for_marks(context,obj)
    return (original is None or original==src) and obj.get('ocular_source_signature')==signature and wm.geometry_signature(obj)==signature


def validate_marks(context):
    if context.object and context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    src=oc.reference(context.scene.anaplast)
    obj=context.scene.ocular_production.mark_obj
    if obj is None:raise RuntimeError('Mark the sclera first')
    validate_source(context,src)
    signature=wm.geometry_signature(src)
    if not marks_match(context,obj,src,signature):obj=new_marks(context)
    # Store an object reference after validating legacy name/signature records,
    # so renaming the source will not disconnect future marks.
    obj['ocular_source_object']=src;obj['ocular_source_name']=src.name
    return obj,src,signature


def new_marks(context):
    """Replace old marks only after the selected source's fresh copy is ready."""
    p=context.scene.ocular_production;ap=context.scene.anaplast
    validate_source(context,oc.reference(ap))
    old=p.mark_obj;old_source=ap.ocular_scan
    p.mark_obj=None
    try:obj=ensure_marks(context)
    except Exception:
        new=p.mark_obj
        if new is not None and new!=old:mu.delete_object(new)
        p.mark_obj=old;ap.ocular_scan=old_source
        raise
    if old is not None and old!=oc.reference(ap) and old.get('ocular_edit_copy'):
        mu.delete_object(old)
    obj.name='Ocular_Scan_Marks'
    context.scene.ocular_steps.edit_marks=True
    return obj


def ensure_marks(context):
    if context.object and context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    p=context.scene.ocular_production;src=oc.reference(context.scene.anaplast)
    validate_source(context,src)
    signature=wm.geometry_signature(src);obj=p.mark_obj
    if obj is not None and not marks_match(context,obj,src,signature):return new_marks(context)
    if obj is None:
        obj=mu.duplicate_object(src,'Ocular_Scan_Marks',mu.get_collection(context.scene,oc.COLLECTION));p.mark_obj=obj
        obj['ocular_edit_copy']=True;obj['ocular_source_signature']=signature;obj['ocular_source_name']=src.name
        obj['ocular_source_object']=src
        write_values(obj,np.zeros(len(obj.data.vertices)),'SCLERA');write_values(obj,np.zeros(len(obj.data.vertices)),'IRIS')
    else:
        obj['ocular_source_object']=src;obj['ocular_source_name']=src.name
    # Make the fallback source visible and stable in the source selector.
    context.scene.anaplast.ocular_scan=src
    obj['ocular_active_region']=p.mark_region
    if context.object and context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    if 'ocular_mark_visibility' not in context.scene:
        context.scene['ocular_mark_visibility']=json.dumps({o.name:o.hide_get() for o in context.view_layer.objects})
    for o in context.view_layer.objects:
        o.select_set(False)
        if o.type=='MESH':o.hide_set(o!=obj)
    obj.hide_set(False);obj.hide_render=True;obj.select_set(True);context.view_layer.objects.active=obj;show_colors(obj)
    if context.area and context.area.type=='VIEW_3D':
        sd=context.space_data
        if 'ocular_mark_shading' not in context.scene:context.scene['ocular_mark_shading']=json.dumps({'type':sd.shading.type,'color_type':sd.shading.color_type})
        sd.shading.type='SOLID';sd.shading.color_type='VERTEX'
        obj['ocular_view_front']=list(sd.region_3d.view_rotation@Vector((0,0,1)))
        obj['ocular_view_right']=list(sd.region_3d.view_rotation@Vector((1,0,0)))
    return obj


def restore(context):
    for name,hidden in json.loads(context.scene.pop('ocular_mark_visibility','{}')).items():
        obj=bpy.data.objects.get(name)
        if obj and obj.name in context.view_layer.objects:obj.hide_set(hidden)
    if context.scene.ocular_production.mark_obj:context.scene.ocular_production.mark_obj.hide_set(True)
    previous=json.loads(context.scene.pop('ocular_mark_shading','{}'))
    if context.area and context.area.type=='VIEW_3D':
        context.space_data.shading.type=previous.get('type','SOLID');context.space_data.shading.color_type=previous.get('color_type','MATERIAL')


def contours(obj,mask,co):
    """Closed 0.5 contour loops on source triangles, preserving actual topology."""
    obj.data.calc_loop_triangles();graph={};points={}
    for tri in obj.data.loop_triangles:
        ids=list(tri.vertices);cut=[]
        for a,b in zip(ids,ids[1:]+ids[:1]):
            if mask[a]!=mask[b]:
                key=tuple(sorted((a,b)));points[key]=(co[a]+co[b])*.5;cut.append(key)
        if len(cut)==2:
            a,b=cut;graph.setdefault(a,[]).append(b);graph.setdefault(b,[]).append(a)
    if not graph:raise RuntimeError('Paint a filled region with a border on the scan')
    if any(len(v)!=2 for v in graph.values()):raise RuntimeError('The marked border reaches an open or damaged scan edge; move it inside the scanned surface')
    unseen=set(graph);loops=[]
    while unseen:
        start=min(unseen);cur=start;prev=None;path=[]
        while cur in unseen:
            unseen.remove(cur);path.append(points[cur]);nexts=graph[cur];nxt=nexts[0] if nexts[0]!=prev else nexts[1];prev,cur=cur,nxt
        if cur!=start:raise RuntimeError('The marking border branches; erase stray marks and retry')
        loops.append(np.array(path))
    return loops


def smooth_loop(points,amount,count=192):
    d=np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1);total=d.sum()
    if total<1:raise RuntimeError('The marked region is too small')
    length=np.r_[0,np.cumsum(d)];points=np.vstack((points,points[0]));t=np.arange(count)*total/count
    result=np.column_stack([np.interp(t,length,points[:,i]) for i in range(3)])
    if amount>0:
        freq=np.fft.rfftfreq(count,d=total/count);weight=np.exp(-.5*(2*np.pi*freq*amount)**2)
        result=np.fft.irfft(np.fft.rfft(result,axis=0)*weight[:,None],n=count,axis=0)
    return result


def circle_fit(xy):
    def solve(q):
        A=np.c_[2*q,np.ones(len(q))];v=np.linalg.lstsq(A,(q*q).sum(1),rcond=None)[0]
        return v[:2],np.sqrt(max(0,v[2]+v[:2]@v[:2]))
    rng=np.random.default_rng(746);best=None
    for _ in range(160):
        q=xy[rng.choice(len(xy),3,replace=False)]
        if abs(np.linalg.det(np.c_[q,np.ones(3)]))<1e-5:continue
        center,radius=solve(q)
        if not 1.5<radius<10:continue
        residual=np.abs(np.linalg.norm(xy-center,axis=1)-radius);inliers=residual<max(.2,.035*radius)
        score=(int(inliers.sum()),-float(np.median(residual[inliers])) if inliers.any() else -1e6)
        if best is None or score>best[0]:best=(score,inliers)
    if best is None or best[1].sum()<len(xy)*.45:raise RuntimeError('The iris border does not contain enough circular edge; refine the iris highlight')
    good=best[1];center,radius=solve(xy[good]);residual=np.abs(np.linalg.norm(xy-center,axis=1)-radius)
    return center,radius,float(np.sqrt(np.mean(residual[good]**2))),float(good.mean())


def fit(context):
    p=context.scene.ocular_production;ap=context.scene.anaplast;obj=ensure_marks(context)
    sclera=values_for(obj,'SCLERA')>.5;iris=values_for(obj,'IRIS')>.5
    if sclera.sum()<12 or iris.sum()<12:raise RuntimeError('Highlight the visible sclera and the visible iris including its pupil')
    co=np.array([obj.matrix_world@v.co for v in obj.data.vertices])
    union=sclera|iris;outer=contours(obj,union,co);inner=contours(obj,iris,co)
    if len(outer)!=1:raise RuntimeError('Sclera and iris must form one filled opening. Join gaps, fill holes, or erase marks on the other eye')
    if len(inner)!=1:raise RuntimeError('Paint one filled iris region including the pupil; erase stray islands')
    # Use iris surface orientation, sign and horizontal from the patient frame/view.
    q=co[iris];center3=q.mean(0);_,_,vt=np.linalg.svd(q-center3,full_matrices=False);u=Vector(vt[-1])
    from .orient import build_frame
    frame,_=build_frame()
    forward=frame.col[1].to_3d() if frame else Vector(obj.get('ocular_view_front',(0,0,1)))
    right=-frame.col[0].to_3d() if frame else Vector(obj.get('ocular_view_right',(1,0,0)))
    if u.dot(forward)<0:u=-u
    if abs(u.dot(forward))<.25:raise RuntimeError('The iris orientation is ambiguous; inspect the highlight and use a frontal view')
    r=(right-u*right.dot(u)).normalized();f=u.cross(r).normalized()
    outline=smooth_loop(outer[0],p.mark_smooth)
    poly=np.column_stack((outline@r,outline@f));oc.radial_outline(poly,ap.ocular_extension)
    border=smooth_loop(inner[0],min(.10,p.mark_smooth),192);xy=np.column_stack((border@r,border@f))
    cc,radius,error,coverage=circle_fit(xy)
    if oc.signed_distance(cc[None],poly)[0]>=0:raise RuntimeError('The fitted iris center lies outside the highlighted opening')
    _,extended=oc.radial_outline(poly,ap.ocular_extension)
    a=np.arange(128)*2*np.pi/128;circle=cc+radius*np.column_stack((np.cos(a),np.sin(a)))
    if np.any(oc.signed_distance(circle,extended)>=0):raise RuntimeError('The iris extends beyond the ocular outline; refine the marks or increase the beyond-lids extension')
    trace=oc.save_trace(context,[Vector(v) for v in outline],(u,r))
    trace['ocular_marks_signature']=mark_signature(obj);trace['ocular_marked_source']=obj.name
    ap.ocular_iris_diameter=radius*2;ap.ocular_cornea_diameter=radius*2
    marker=bpy.data.objects.get('Ocular_Cornea_Center')
    if marker is None:marker=bpy.data.objects.new('Ocular_Cornea_Center',None);mu.get_collection(context.scene,oc.COLLECTION).objects.link(marker)
    marker.location=r*cc[0]+f*cc[1]+u*Vector(center3).dot(u);marker.empty_display_type='CIRCLE';marker.empty_display_size=radius;marker.show_in_front=True;marker.hide_render=True
    marker.rotation_euler=u.to_track_quat('Z','Y').to_euler()
    p.mark_report=f'Iris estimate {2*radius:.2f} mm; arc fit {error:.2f} mm RMS. Review center and diameter.'
    trace['iris_fit_diameter_mm']=radius*2
    trace['iris_fit_center_world']=list(marker.matrix_world.translation)
    trace['iris_fit_rms_mm']=error;trace['iris_fit_inlier_fraction']=coverage
    return trace


def preview_scan(context):
    from .ocular_steps import preview_sclera
    return preview_sclera(context)


def generate(context):
    p=context.scene.ocular_production;ap=context.scene.anaplast;obj=p.mark_obj;src=oc.reference(ap)
    if not obj or not ap.ocular_trace:raise RuntimeError('Fit the highlighted borders and iris first')
    if obj.get('ocular_source_signature')!=wm.geometry_signature(src):raise RuntimeError('The scan changed; start a new marking copy')
    if ap.ocular_trace.get('ocular_marks_signature')!=mark_signature(obj):raise RuntimeError('The highlights changed; fit the borders and iris again')
    if wm.geometry_signature(obj)!=obj.get('ocular_source_signature'):raise RuntimeError('The marking copy geometry changed; start a new copy')
    if context.object and context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    previous=ap.ocular_obj;previous_name=previous.name if previous else None
    if previous and previous.name=='Ocular_Shell':previous.name='Ocular_Previous_Shell'
    new=None
    try:
        new=oc.build(context)
        if p.photo:
            from .ocular_production import build
            parts=build(context)
        else:parts=(new,)
    except Exception:
        if new:mu.delete_object(new)
        ap.ocular_obj=previous
        if previous:previous.name=previous_name
        raise
    restore(context)
    preview_scan(context)
    if previous and previous!=new:previous.hide_set(True)
    for part in parts:part.hide_set(False)
    if len(parts)==2:new.hide_set(True)
    p.mark_report='Ocular generated from scan highlights.'+(' Photo color, iris relief and clear included.' if p.photo else ' Load an eye photo, then build color + printed clear.')
    return parts


class OCULAR_OT_marks(bpy.types.Operator):
    bl_idname='anaplast.ocular_marks';bl_label='Ocular scan highlights';bl_options={'REGISTER','UNDO'}
    action:bpy.props.EnumProperty(items=[('START','Show scan highlights',''),('NEW','New marking copy',''),('CLEAR','Clear selected region',''),('FIT','Fit borders and iris',''),('BUILD','Generate ocular from fitted marks',''),('RETURN','Return to case','')])
    def execute(self,context):
        try:
            p=context.scene.ocular_production
            if self.action=='RETURN':restore(context)
            elif self.action=='NEW':new_marks(context)
            elif self.action=='FIT':fit(context)
            elif self.action=='BUILD':generate(context)
            else:
                obj=ensure_marks(context)
                if self.action=='CLEAR':write_values(obj,np.zeros(len(obj.data.vertices)))
        except Exception as e:rep(self,{'ERROR'},f'Ocular marks ({self.action}): {e}');return {'CANCELLED'}
        return {'FINISHED'}


def draw_marking(l,context):
    p=context.scene.ocular_production;b=l.box();b.label(text='Generate from highlighted scan')
    b.operator('anaplast.ocular_marks',text='Show scan for highlighting').action='START'
    b.prop(p,'mark_region',expand=True);b.prop(p,'mark_radius')
    row=b.row(align=True)
    for label,erase,lasso in [('Paint',False,False),('Erase',True,False),('Lasso',False,True)]:
        op=row.operator('anaplast.wedge_mark_tool',text=label);op.target='OCULAR';op.erase=erase;op.lasso=lasso
    b.label(text='Sclera: cream | Iris including pupil: blue')
    b.prop(p,'mark_smooth');b.prop(context.scene.anaplast,'ocular_extension')
    b.operator('anaplast.ocular_marks',text='1. Fit highlighted borders and iris').action='FIT'
    b.prop(p,'surface_method')
    if p.surface_method=='COPY':b.prop(p,'copy_thickness');b.label(text='Visible curvature copied; extension added beyond border')
    if p.surface_method=='SMOOTH':
        b.prop(p,'match_scan_iris');b.prop(context.scene.anaplast,'ocular_thickness')
        b.label(text='Smooth scan curvature; no rough scan detail')
    b.operator('anaplast.ocular_marks',text='2. Generate ocular from fitted marks').action='BUILD'
    row=b.row(align=True);row.operator('anaplast.ocular_marks',text='Clear this region').action='CLEAR';row.operator('anaplast.ocular_marks',text='New copy').action='NEW'
    b.operator('anaplast.ocular_marks',text='Return to case').action='RETURN'
    if p.mark_report:b.label(text=p.mark_report)


def register():bpy.utils.register_class(OCULAR_OT_marks)
def unregister():bpy.utils.unregister_class(OCULAR_OT_marks)
