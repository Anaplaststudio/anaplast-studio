"""Staged sclera -> sphere -> shell -> iris -> cornea workflow, entirely local."""
import hashlib,json,math
import numpy as np
import bpy,bmesh
from mathutils import Vector
from mathutils.geometry import convex_hull_2d,delaunay_2d_cdt
from . import ocular as oc,ocular_marking as mark,ocular_smooth as smooth,ocular_fit as fit
from ..utils import mesh as mu
from .report import rep


def region_hash(obj,region):return hashlib.sha256(mark.values_for(obj,region).tobytes()).hexdigest()


def validated_marks(context):
    return mark.validate_marks(context)


def fit_ball(context):
    obj,src,signature=validated_marks(context);w=context.scene.ocular_steps
    world=np.array([obj.matrix_world@v.co for v in obj.data.vertices]);selected,ignored=fit.selection(obj,world)
    if selected.sum()<30:raise RuntimeError('Mark a broad scleral area on both sides of the iris, excluding lids and iris')
    u,r,f=fit.local_frame(world,selected,mark.values_for(obj,'IRIS')>.5,obj.get('ocular_view_front',(0,0,1)),obj.get('ocular_view_right',(1,0,0)))
    local=np.c_[world@r,world@f,world@u]
    center,radius,model,metrics=fit.fit(local[selected],w.measured_diameter*.5 if w.use_measured_diameter else None)
    rms=metrics['rms_mm']
    wc=Vector(r*center[0]+f*center[1]+u*center[2])
    bm=bmesh.new();bmesh.ops.create_uvsphere(bm,u_segments=48,v_segments=24,radius=radius)
    ball=mu.new_mesh_object('Ocular_Fitted_Ball',bm,mu.get_collection(context.scene,oc.COLLECTION));ball.location=wc
    ball.display_type='WIRE';ball.hide_render=True;ball.show_in_front=True
    ball['fit_center_world']=list(wc);ball['fit_center_local']=list(center);ball['fit_radius_mm']=radius;ball['fit_rms_mm']=rms
    ball['ocular_up']=list(u);ball['ocular_right']=list(r);ball['source_signature']=signature;ball['sclera_signature']=region_hash(obj,'SCLERA')
    ball['use_measured_diameter']=w.use_measured_diameter;ball['measured_diameter_mm']=w.measured_diameter
    ball['fit_surface_model']=json.dumps(model);ball['fit_metrics']=json.dumps(metrics)
    ball['fit_sclera_indices']=np.flatnonzero(selected).tolist();ball['fit_ignored_vertices']=ignored
    ball['fit_iris_signature']=region_hash(obj,'IRIS')
    context.view_layer.update()
    ball['reference_only']=True;ball['geometry_signature']=mark.wm.geometry_signature(ball)
    if w.ball:mu.delete_object(w.ball)
    w.ball=ball;w.iris_fit='';w.report=fit_summary(ball)
    mark.restore(context);src.hide_set(False);ball.hide_set(False)
    return ball


def fit_summary(ball):
    metrics=json.loads(ball.get('fit_metrics','{}'))
    result=f'Estimated globe diameter {2*ball["fit_radius_mm"]:.2f} mm; sclera fit RMS {ball["fit_rms_mm"]:.2f} mm'
    if metrics:result+=f'; 95% within {metrics["p95_mm"]:.2f} mm ({metrics["surface"]})'
    if ball.get('fit_ignored_vertices',0):result+=f'; excluded {ball["fit_ignored_vertices"]} remote painted vertices'
    return result+'.'


def ball_data(context):
    obj,src,signature=validated_marks(context);ball=context.scene.ocular_steps.ball
    if not ball:raise RuntimeError('Fit the scleral ball first')
    w=context.scene.ocular_steps
    if bool(ball.get('use_measured_diameter',False))!=w.use_measured_diameter or (w.use_measured_diameter and abs(ball.get('measured_diameter_mm',0)-w.measured_diameter)>1e-6):raise RuntimeError('Diameter settings changed; fit the ball again')
    if ball['source_signature']!=signature or ball['sclera_signature']!=region_hash(obj,'SCLERA'):raise RuntimeError('Sclera marks changed; fit the ball again')
    if 'fit_iris_signature' in ball and ball['fit_iris_signature']!=region_hash(obj,'IRIS'):raise RuntimeError('Iris marks changed; fit the eye again to check the nearby sclera')
    if ball['geometry_signature']!=mark.wm.geometry_signature(ball):raise RuntimeError('The reference ball was edited; fit it again')
    u=np.array(ball['ocular_up']);r=np.array(ball['ocular_right']);f=np.cross(u,r)
    return obj,src,ball,u,r,f


def cornea_ratio(w):
    ratio=w.cornea_manual_diameter/w.iris_diameter if w.cornea_size_mode=='MANUAL' else w.cornea_width
    if ratio<1:raise RuntimeError('Cornea diameter must cover the iris; increase it or reduce iris diameter')
    if ratio>1.5:raise RuntimeError('Cornea diameter is too large for this iris; keep it within 1.5 times the iris diameter')
    return ratio


def fit_iris(context):
    pending=context.scene.ocular_steps.ball;marks=context.scene.ocular_production.mark_obj
    if pending and marks and 'fit_iris_signature' in pending and pending['fit_iris_signature']!=region_hash(marks,'IRIS'):
        fit_ball(context)
    obj,src,ball,u,r,f=ball_data(context);w=context.scene.ocular_steps;ap=context.scene.anaplast
    selected=mark.values_for(obj,'IRIS')>.5
    if selected.sum()<12:raise RuntimeError('Mark the visible iris including its pupil')
    world=np.array([obj.matrix_world@v.co for v in obj.data.vertices]);loops=mark.contours(obj,selected,world)
    if len(loops)!=1:raise RuntimeError('Mark one filled iris without islands or holes')
    border=mark.smooth_loop(loops[0],.08,192);xy=np.c_[border@r,border@f]
    center,radius,rms,coverage=mark.circle_fit(xy)
    height=fit.fitted_height(center[None],ball)[0]
    diameter=w.iris_manual_diameter if w.iris_size_mode=='MANUAL' else 2*radius
    if w.cornea_size_mode=='MANUAL' and not diameter<=w.cornea_manual_diameter<=diameter*1.5:
        raise RuntimeError('Cornea diameter must be between the iris diameter and 1.5 times that diameter')
    w.measured_iris_diameter=2*radius;w.iris_xy=center;w.iris_diameter=diameter;w.iris_fit=region_hash(obj,'IRIS');w.iris_ball_signature=ball['geometry_signature']
    ap.ocular_iris_diameter=diameter;ap.ocular_cornea_diameter=diameter*cornea_ratio(w)
    marker=bpy.data.objects.get('Ocular_Cornea_Center')
    if not marker:marker=bpy.data.objects.new('Ocular_Cornea_Center',None);mu.get_collection(context.scene,oc.COLLECTION).objects.link(marker)
    marker.location=Vector(r*center[0]+f*center[1]+u*height);marker.empty_display_type='CIRCLE';marker.empty_display_size=diameter*.5
    marker.rotation_euler=Vector(u).to_track_quat('Z','Y').to_euler();marker.show_in_front=True;marker.hide_render=True
    w.report=f'Iris: {diameter:.2f} mm; scan measurement {2*radius:.2f} mm; border RMS {rms:.2f} mm. Add the cornea next.'
    mark.restore(context)
    return marker


def cornea_profile(rho,radius,height):
    if height==0:return np.zeros_like(rho)
    if height>=radius:raise RuntimeError('Cornea height must be less than its footprint radius')
    rc=(radius*radius+height*height)/(2*height)
    cap=np.maximum(0,np.sqrt(np.maximum(0,rc*rc-(np.minimum(rho,1)*radius)**2))-(rc-height))
    t=np.clip((rho-.85)/.15,0,1)
    return cap*(1-t*t*(3-2*t))


def preview_sclera(context):
    """Cut sclera first; cut iris only after its measurement is used by the ocular."""
    p=context.scene.ocular_production;src=oc.reference(context.scene.anaplast);obj=p.mark_obj
    iris=mark.values_for(obj,'IRIS')>.5
    selected=(mark.values_for(obj,'SCLERA')>.5)&~iris
    # A broad imported sculpt mask can include iris pixels. Keep the full scan
    # until a separate iris mark tells us which region must be protected.
    if not iris.any():selected[:]=False
    shell=context.scene.anaplast.ocular_obj;w=context.scene.ocular_steps;transferred=False
    if w.ball and 'fit_sclera_indices' in w.ball and w.ball.get('sclera_signature')==region_hash(obj,'SCLERA') and w.ball.get('fit_iris_signature')==region_hash(obj,'IRIS'):
        selected=fit.accepted_mask(obj,w.ball)&~iris
    if shell and shell.get('workflow_has_iris') and w.ball:
        u=np.array(shell['ocular_up']);r=np.array(shell['ocular_right']);f=np.cross(u,r);cc=np.array(shell['ocular_iris_center'])
        transferred=(shell['workflow_iris_signature']==region_hash(obj,'IRIS') and shell['workflow_ball_signature']==w.ball['geometry_signature'] and np.allclose([cc@r,cc@f],w.iris_xy,atol=1e-5) and abs(shell['iris_diameter_mm']-w.iris_diameter)<1e-5)
    if transferred:selected|=iris
    from .ocular_cut import cut_field,fill_cut_islands
    me=src.data;me.calc_loop_triangles();co=np.array([v.co[:] for v in me.vertices])
    world=np.array([src.matrix_world@v.co for v in me.vertices])
    if len(selected)!=len(co):raise RuntimeError('Scan geometry changed; mark the sclera again')
    if w.ball:
        right=np.array(w.ball['ocular_right']);normal=np.array(w.ball['ocular_up'])
    else:
        right=np.array(obj.get('ocular_view_right',(1,0,0)));normal=np.array(obj.get('ocular_view_front',(0,0,1)))
    filled=0
    if transferred:selected,filled=fill_cut_islands(obj,selected,world)
    field=cut_field(obj,selected,world,right,np.cross(normal,right))
    selected=field<0
    vertices=[];vertex_sources=[];vertex_weights=[];faces=[];corner_sources=[];corner_weights=[];materials=[];smoothing=[];cache={}
    def vertex(a,b):
        key=(min(a,b),max(a,b))
        if key not in cache:
            t=0. if a==b else float(field[a]/(field[a]-field[b]))
            cache[key]=len(vertices);vertices.append(co[a]*(1-t)+co[b]*t);vertex_sources.append((a,b));vertex_weights.append(t)
        return cache[key]
    for tri in me.loop_triangles:
        ids=list(tri.vertices);loops=list(tri.loops);poly=[];source=[]
        for j in range(3):
            k=(j+1)%3;a,b=ids[j],ids[k]
            if not selected[a]:poly.append(vertex(a,a));source.append((loops[j],loops[j],0.))
            if selected[a]!=selected[b]:poly.append(vertex(a,b));source.append((loops[j],loops[k],float(field[a]/(field[a]-field[b]))))
        for j in range(1,len(poly)-1):
            faces.append((poly[0],poly[j],poly[j+1]));samples=(source[0],source[j],source[j+1])
            corner_sources.extend((a,b) for a,b,t in samples);corner_weights.extend(t for a,b,t in samples)
            face=me.polygons[tri.polygon_index];materials.append(face.material_index);smoothing.append(face.use_smooth)
    new_me=bpy.data.meshes.new('Sclera_Cut_Preview');new_me.from_pydata(vertices,[],faces);new_me.update()
    for material in me.materials:new_me.materials.append(material)
    new_me.polygons.foreach_set('material_index',materials);new_me.polygons.foreach_set('use_smooth',smoothing)
    indices={'POINT':np.array(vertex_sources,dtype=np.int32),'CORNER':np.array(corner_sources,dtype=np.int32)}
    weights={'POINT':np.array(vertex_weights)[:,None],'CORNER':np.array(corner_weights)[:,None]}
    def interpolate(data,domain):
        sampled=data[indices[domain]];t=weights[domain]
        return sampled[:,0]*(1-t)+sampled[:,1]*t
    # Preserve texture coordinates and common color/scalar attributes across cut edges.
    for uv in me.uv_layers:
        data=np.empty((len(me.loops),2),np.float32);uv.data.foreach_get('uv',data.ravel())
        new_uv=new_me.uv_layers.new(name=uv.name);new_uv.data.foreach_set('uv',interpolate(data,'CORNER').ravel())
    specs={'FLOAT':('value',1),'FLOAT_VECTOR':('vector',3),'FLOAT_COLOR':('color',4),'BYTE_COLOR':('color',4),'INT':('value',1),'BOOLEAN':('value',1)}
    for attr in me.attributes:
        if attr.name in new_me.attributes or attr.domain not in indices or attr.data_type not in specs:continue
        key,n=specs[attr.data_type];dtype=np.int32 if attr.data_type in {'INT','BOOLEAN'} else np.float32
        data=np.empty((len(attr.data),n),dtype);attr.data.foreach_get(key,data.ravel());sampled=data[indices[attr.domain]]
        values=sampled[np.arange(len(sampled)),(weights[attr.domain][:,0]>.5).astype(int)] if dtype==np.int32 else interpolate(data,attr.domain)
        target=new_me.attributes.new(attr.name,attr.data_type,attr.domain);target.data.foreach_set(key,values.ravel())
    new=bpy.data.objects.new('Orbital_Sclera_Cut_Preview',new_me);mu.get_collection(context.scene,oc.COLLECTION).objects.link(new);new.matrix_world=src.matrix_world
    new['ocular_preview_only']=True;new['source_scan']=src.name;new['cut_sclera_vertex_count']=int(selected.sum());new['scanned_iris_preserved']=not transferred;new['iris_removed_after_transfer']=transferred
    new['purpose']='Smooth contour cut from source copy; iris removed only after verified transfer; eyelids preserved'
    new['cut_border_smoothing_mm']=.25
    new['cut_removed_fragment_vertices']=filled
    old=p.orbital_preview;p.orbital_preview=new
    if old:mu.delete_object(old)
    src.hide_set(True);src.hide_render=True;new.hide_set(False);new.hide_render=False
    return new


def fitting_normals(obj,indices,center,up,inner_radius,flatten,offset):
    """Use the fitted ellipsoid's normals across unequal triangle strips."""
    me=obj.data;me.update()
    normals=np.array([v.normal[:] for v in me.vertices]);indices=np.asarray(indices,dtype=int)
    points=np.array([me.vertices[int(i)].co[:] for i in indices])-center-offset*np.asarray(up)
    axial=points@up
    gradient=points+(axial*((1-flatten)**-2-1))[:,None]*up
    gradient/=np.linalg.norm(gradient,axis=1)[:,None]
    sign=1 if np.median(np.sum(gradient*normals[indices],axis=1))>0 else -1
    normals[indices]=gradient*sign
    me.normals_split_custom_set_from_vertices(normals.tolist())


def rim_anchored_heights(chart,faces,reference,amount):
    """Fill the bowl toward a harmonic sheet while fixing every rim vertex."""
    triangles=np.array([(face[0],face[j],face[j+1]) for face in faces for j in range(1,len(face)-1)],dtype=int)
    aa=[];bb=[];ww=[]
    for k in range(3):
        a=triangles[:,k];b=triangles[:,(k+1)%3];c=triangles[:,(k+2)%3]
        va=chart[a]-chart[c];vb=chart[b]-chart[c]
        area=np.abs(va[:,0]*vb[:,1]-va[:,1]*vb[:,0])
        if np.any(area<1e-12):raise RuntimeError('Degenerate fitting triangle; rebuild the ocular outline')
        aa.append(a);bb.append(b);ww.append(.5*np.sum(va*vb,axis=1)/area)
    a=np.concatenate(aa);b=np.concatenate(bb);weight=np.concatenate(ww);n=len(chart)
    edges=np.sort(np.c_[a,b],axis=1);unique,count=np.unique(edges,axis=0,return_counts=True)
    rim=np.unique(unique[count==1]);free=np.ones(n,bool);free[rim]=False
    diagonal=np.bincount(a,weight,minlength=n)+np.bincount(b,weight,minlength=n)
    def multiply(v):
        return diagonal*v-np.bincount(a,weight*v[b],minlength=n)-np.bincount(b,weight*v[a],minlength=n)
    x=reference.copy();residual=-multiply(x);residual[~free]=0
    z=np.divide(residual,diagonal,out=np.zeros(n),where=free);direction=z.copy();rz=float(residual@z)
    for _ in range(3000):
        if np.max(np.abs(residual[free]))<1e-9:break
        q=multiply(direction);denom=float(direction@q)
        if denom<=0:raise RuntimeError('Could not solve the rim-anchored fitting surface')
        alpha=rz/denom;x+=alpha*direction;residual-=alpha*q;residual[~free]=0
        z=np.divide(residual,diagonal,out=np.zeros(n),where=free);next_rz=float(residual@z)
        direction=z+(next_rz/rz)*direction;rz=next_rz
    else:raise RuntimeError('Fitting surface did not converge; refine the outline')
    x[rim]=reference[rim]
    filled=reference+amount*(x-reference);filled[rim]=reference[rim]
    return filled,rim


def build_shell(context,cornea=False):
    from .mold_inserts import guard_ocular_rebuild
    guard_ocular_rebuild(context.scene)
    obj,src,ball,u,r,f=ball_data(context);w=context.scene.ocular_steps;ap=context.scene.anaplast;p=context.scene.ocular_production
    if cornea and (w.iris_fit!=region_hash(obj,'IRIS') or w.iris_ball_signature!=ball['geometry_signature']):raise RuntimeError('Transfer the marked iris to this ball first')
    if w.thickness>=ball['fit_radius_mm']:raise RuntimeError('Shell thickness must be smaller than the fitted radius')
    world=np.array([obj.matrix_world@v.co for v in obj.data.vertices]);selected=mark.values_for(obj,'SCLERA')>.5
    if 'fit_sclera_indices' in ball:selected=fit.accepted_mask(obj,ball)
    if cornea:selected|=mark.values_for(obj,'IRIS')>.5
    projected=np.c_[world[selected]@r,world[selected]@f]
    center=np.array(ball['fit_center_local']);radius=ball['fit_radius_mm'];angular=bool(ball.get('use_measured_diameter',False)) or 'fit_surface_model' in ball
    if cornea:
        # The scanned lids may hide part of the iris. Preserve its measured
        # circle beneath them instead of shrinking it to the visible paint.
        angles=np.arange(256)*2*np.pi/256
        iris_ring=np.array(w.iris_xy)+(w.iris_diameter*.5*cornea_ratio(w)+.7)*np.c_[np.cos(angles),np.sin(angles)]
        projected=np.vstack((projected,iris_ring))
    hull=projected[convex_hull_2d([Vector(v) for v in projected])]
    guide=mark.smooth_loop(np.c_[hull,np.zeros(len(hull))],p.mark_smooth,192)[:,:2]
    if angular:guide=smooth.sphere_to_domain(guide,center,radius)
    _,boundary=oc.radial_outline(guide,ap.ocular_extension,256)
    edge_radius=min(w.edge_radius,w.thickness*.45)
    cc=np.array(w.iris_xy) if cornea else boundary.mean(0);ir=w.iris_diameter*.5 if cornea else 0
    chart=lambda v:smooth.sphere_to_domain(v,center,radius) if angular else v
    interior=boundary
    if edge_radius>0:
        ring_center=boundary.mean(axis=0);ring_directions=boundary-ring_center
        ring_directions/=np.linalg.norm(ring_directions,axis=1)[:,None]
        _,interior=oc.radial_outline(boundary,-edge_radius*2,256,center=ring_center,directions=ring_directions)
    if edge_radius>0 and cornea:
        angles=np.arange(256)*2*np.pi/256
        protect=chart(cc+(ir*cornea_ratio(w)+.7)*np.c_[np.cos(angles),np.sin(angles)])
        combined=np.vstack((interior,protect))
        midpoint=(combined.min(0)+combined.max(0))*.5
        radii=(combined.max(0)-combined.min(0))*.5
        radii*=np.max(np.linalg.norm((combined-midpoint)/radii,axis=1))
        interior=midpoint+radii*np.c_[np.cos(angles),np.sin(angles)]
    step=.2;lo=interior.min(0);hi=interior.max(0);X,Y=np.meshgrid(np.arange(lo[0],hi[0],step),np.arange(lo[1],hi[1],step))
    grid=np.c_[X.ravel(),Y.ravel()];grid=grid[oc.signed_distance(grid,interior)<-.6*step]
    points=interior;edges=[]
    if cornea:
        cr=ir*cornea_ratio(w);angles=np.arange(160)*2*np.pi/160
        if np.any(oc.signed_distance(chart(cc+cr*np.c_[np.cos(angles),np.sin(angles)]),boundary)>=0):raise RuntimeError('Cornea reaches outside the ocular; increase extension or reduce its width')
        for ring_radius in sorted(set([ir,cr,.85*cr])):
            ring=chart(cc+ring_radius*np.c_[np.cos(angles),np.sin(angles)]);start=len(points)
            if np.any(oc.signed_distance(ring,interior)>=-.02):continue
            edges.extend((start+i,start+(i+1)%len(ring)) for i in range(len(ring)))
            points=np.vstack((points,ring));grid_xy=smooth.sphere_from_domain(grid,center,radius)[0] if angular else grid;grid=grid[abs(np.linalg.norm(grid_xy-cc,axis=1)-ring_radius)>.4*step]
    if cornea:points=np.vstack((points,chart(cc[None])))
    points=np.vstack((points,grid));output=delaunay_2d_cdt([Vector(v) for v in points],edges,[list(range(len(interior)))],1,1e-6,False)
    xy=np.array(output[0]);faces=output[2];center=np.array(ball['fit_center_local']);radius=ball['fit_radius_mm']
    domain=xy.copy()
    if angular:
        xy,z=smooth.sphere_from_domain(xy,center,radius)
        z+=fit.correction(xy,fit.model_for(ball))
    else:z=fit.fitted_height(xy,ball)
    base=xy[:,0,None]*r+xy[:,1,None]*f+z[:,None]*u
    rise=cornea_profile(np.linalg.norm(xy-cc,axis=1)/(ir*cornea_ratio(w)),ir*cornea_ratio(w),w.cornea_height) if cornea else np.zeros(len(xy))
    wc=np.array(ball['fit_center_world']);normal=(base-wc)/radius
    outer=base+rise[:,None]*u
    inner_radius=radius-w.thickness;back_delta=normal*inner_radius
    flatten=w.inner_flatten;axial=back_delta@u
    offset=inner_radius*flatten if w.thickness_basis=='CENTER' else 0.
    back_delta+=((1-flatten)*axial+offset-axial)[:,None]*u
    rim_fixed_error=0.;automatic_setback=0.
    if w.thickness_basis=='RIM':
        # A shallow analytic back avoids inheriting the outline's ripples or
        # folding its projection where the globe extends past its equator.
        ref=w.rim_reference_flatten
        reference=(1-ref)*axial+inner_radius*ref
        height,rim_ids=rim_anchored_heights(domain,faces,reference,flatten)
        height-=w.back_blend_depth
        fitting_chart=domain if angular else domain-center[:2]
        radial=np.linalg.norm(fitting_chart,axis=1)
        shrink=np.divide(inner_radius*np.tanh(radial/radius),radial,out=np.full(len(radial),inner_radius/radius),where=radial>1e-10)
        fitting_xy=fitting_chart*shrink[:,None]
        q=fitting_xy-fitting_xy.mean(0)
        plane=np.c_[np.ones(len(q)),q]
        slope=np.linalg.lstsq(plane,height,rcond=None)[0]
        height=plane@slope-(1-flatten)*np.sum(q*q,axis=1)/(2*inner_radius)
        automatic_setback=max(0.,float(np.max(height-(base-wc)@u))+1.5)
        height-=automatic_setback
        back_delta=fitting_xy[:,0,None]*r+fitting_xy[:,1,None]*f+height[:,None]*u
        rim_fixed_error=float(np.max(np.abs(height[rim_ids]-reference[rim_ids])))
    # RIM uses a shallow concave paraboloid. The explicit CENTER and MINIMUM
    # choices retain their ellipsoid thickness definitions.
    back=wc+back_delta
    bm=bmesh.new();pair=bm.verts.layers.int.new('ocular_back_index');layer=bm.faces.layers.int.new('ocular_front');wall_parent=bm.verts.layers.int.new('ocular_rim_front_index');wall_weight=bm.verts.layers.float.new('ocular_rim_front_weight');vs=[bm.verts.new(v) for v in outer]
    for face in faces:bm.faces.new([vs[i] for i in face])[layer]=1
    rim=[tuple(e.verts) for e in bm.edges if e.is_boundary];rear=[bm.verts.new(v) for v in back];mapping={v:i for i,v in enumerate(vs)}
    for face in list(bm.faces):bm.faces.new([rear[mapping[v]] for v in reversed(face.verts)])
    if edge_radius>0:
        rim_vertices={v for edge in rim for v in edge};controls={};jets={}
        # Choose endpoint speeds for a gradual change in curvature, then
        # smooth those speeds around the loop to avoid lateral ripples.
        ts=np.linspace(.002,.998,65)
        basis=lambda degree:np.array([math.comb(degree,k)*(1-ts)**(degree-k)*ts**k for k in range(degree+1)]).T
        B1=basis(4);B2=basis(3)
        scales=np.linspace(.45,2.,19);sf,sb=np.meshgrid(scales,scales);sf=sf.ravel();sb=sb.ravel()
        def curve_points(F,B,tf,kf,tb,kb,lf,lb):
            df=lf[:,None]*tf;db=lb[:,None]*tb
            return np.stack((np.broadcast_to(F,df.shape),F+df/5,F+2*df/5+kf*lf[:,None]**2/20,B-2*db/5+kb*lb[:,None]**2/20,B-db/5,np.broadcast_to(B,df.shape)),axis=1)
        def best_handles(F,B,tf,kf,tb,kb):
            chord=np.linalg.norm(B-F);cp=curve_points(F,B,tf,kf,tb,kb,chord*sf,chord*sb)
            vel=np.einsum('tj,njk->ntk',B1,5*np.diff(cp,axis=1));acc=np.einsum('tj,njk->ntk',B2,20*np.diff(cp,n=2,axis=1))
            speed=np.linalg.norm(vel,axis=2);curv=np.cross(vel,acc)/np.maximum(speed,1e-9)[:,:,None]**3
            ds=.5*(speed[:,1:]+speed[:,:-1])*(ts[1]-ts[0]);variation=np.linalg.norm(np.diff(curv,axis=1),axis=2)/np.maximum(ds,1e-9)
            curvature=np.linalg.norm(curv,axis=2)
            score=np.mean(variation**2,axis=1)+.15*np.mean(curvature**2,axis=1)+.002*np.mean(speed,axis=1)
            score[np.min(speed,axis=1)<chord*.1]=np.inf
            best=int(np.argmin(score));return np.array([chord*sf[best],chord*sb[best]])
        # One curve runs directly from the globe to the fitting surface. There
        # is no front shoulder, sidewall band, or separate rear fillet. Fit
        # local surface derivatives to continue both tangents and curvatures.
        from mathutils.kdtree import KDTree
        tree=KDTree(len(domain))
        for i,pt in enumerate(domain):tree.insert((pt[0],pt[1],0),i)
        tree.balance()
        def surface_jet(i,surface,direction):
            near=tree.find_n((domain[i,0],domain[i,1],0),40)
            ids=np.array([item[1] for item in near]);q=domain[ids]-domain[i]
            design=np.c_[q[:,0],q[:,1],.5*q[:,0]**2,q[:,0]*q[:,1],.5*q[:,1]**2]
            coef=np.linalg.lstsq(design,surface[ids]-surface[i],rcond=None)[0]
            velocity=direction@coef[:2];speed=np.linalg.norm(velocity);tangent=velocity/speed
            a,b=direction;acceleration=a*a*coef[2]+2*a*b*coef[3]+b*b*coef[4]
            curvature=(acceleration-tangent*np.dot(acceleration,tangent))/(speed*speed)
            return tangent,curvature
        for v in rim_vertices:
            i=mapping[v];direction=domain[i]-interior.mean(0);direction/=np.linalg.norm(direction)
            def front_at(q):
                pq,pz=smooth.sphere_from_domain(q,center,radius) if angular else (q,fit.fitted_height(q,ball))
                if angular:pz+=fit.correction(pq,fit.model_for(ball))
                value=pq[:,0,None]*r+pq[:,1,None]*f+pz[:,None]*u
                if cornea:value+=cornea_profile(np.linalg.norm(pq-cc,axis=1)/(ir*cornea_ratio(w)),ir*cornea_ratio(w),w.cornea_height)[:,None]*u
                return value
            eps=.01;samples=front_at(domain[i]+np.array([-eps,0,eps])[:,None]*direction)
            velocity=(samples[2]-samples[0])/(2*eps);speed=np.linalg.norm(velocity);outward=velocity/speed
            acceleration=(samples[2]-2*samples[1]+samples[0])/eps**2
            front_curvature=(acceleration-outward*np.dot(acceleration,outward))/speed**2
            inward,back_curvature=surface_jet(i,back,-direction)
            jets[v]=(outer[i],back[i],outward,front_curvature,inward,back_curvature)
        ordered=sorted(rim_vertices,key=lambda v:np.arctan2(*(domain[mapping[v]]-interior.mean(0))[::-1]))
        handles=np.array([best_handles(*jets[v]) for v in ordered])
        for _ in range(16):handles=.5*handles+.25*(np.roll(handles,1,axis=0)+np.roll(handles,-1,axis=0))
        handles*=np.clip(edge_radius/1.5,.5,1.25)
        for v,(lf,lb) in zip(ordered,handles):controls[v]=curve_points(*jets[v],np.array([lf]),np.array([lb]))[0]
        row={v:v for v in rim_vertices}
        for k in range(1,49):
            t=k/48.;nxt={}
            for v,(a,b,c,d,e,fv) in controls.items():
                nxt[v]=rear[mapping[v]] if k==48 else bm.verts.new((1-t)**5*a+5*(1-t)**4*t*b+10*(1-t)**3*t*t*c+10*(1-t)**2*t**3*d+5*(1-t)*t**4*e+t**5*fv)
                if k<48:
                    nxt[v][wall_parent]=mapping[v]+1;nxt[v][wall_weight]=(1-t)**3*(1+3*t+6*t*t)
            for a,b in rim:bm.faces.new((row[a],nxt[a],nxt[b],row[b]))
            row=nxt
    else:
        for a,b in rim:bm.faces.new((a,rear[mapping[a]],rear[mapping[b]],b))
    bm.verts.index_update()
    for a,b in zip(vs,rear):a[pair]=b.index+1
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bmesh.ops.triangulate(bm,faces=list(bm.faces))
    shell=mu.new_mesh_object('Ocular_Stage_Work',bm,mu.get_collection(context.scene,oc.COLLECTION))
    try:
        from .mold_design import has_self_intersections
        if any(mu.mesh_stats(shell)[1:]) or has_self_intersections(shell):raise RuntimeError('Ocular shell intersects itself; adjust extension or cornea dimensions')
        shell['ocular_up']=list(u);shell['ocular_right']=list(r);shell['ocular_iris_center']=list(r*cc[0]+f*cc[1]);shell['iris_diameter_mm']=2*ir
        shell['workflow_sphere_center_world']=list(wc);shell['workflow_sphere_center_local']=list(center);shell['workflow_sphere_radius_mm']=radius
        shell['workflow_surface_model']=ball.get('fit_surface_model','null');shell['workflow_fit_metrics']=ball.get('fit_metrics','{}')
        shell['workflow_angular_extension']=angular
        shell['workflow_edge_radius_mm']=edge_radius;shell['workflow_inner_flatten']=flatten;shell['workflow_thickness_basis']=w.thickness_basis
        shell['workflow_edge_construction']='single_front_to_back_curve' if edge_radius>0 else 'straight'
        shell['workflow_rounded_front_outline']=bool(edge_radius>0 and cornea)
        shell['workflow_shell_thickness_mm']=w.thickness;shell['workflow_cornea_height_mm']=w.cornea_height if cornea else 0
        if w.thickness_basis=='RIM':
            shell['workflow_rim_reference_flatten']=w.rim_reference_flatten;shell['workflow_rim_anchor_error_mm']=rim_fixed_error;shell['workflow_back_blend_depth_mm']=w.back_blend_depth
            shell['workflow_automatic_back_setback_mm']=automatic_setback
            pole=int(np.argmax(normal@u));shell['workflow_center_body_thickness_mm']=float(radius-(back[pole]-wc)@u)
        shell['workflow_cornea_radius_mm']=ir*cornea_ratio(w) if cornea else 0;shell['workflow_has_iris']=cornea
        shell['anaplast_part']='OCULAR';shell['aperture_extension_mm']=ap.ocular_extension;shell['source_scan']=src.name
        shell['workflow_ball_signature']=ball['geometry_signature'];shell['workflow_iris_signature']=w.iris_fit if cornea else ''
        previous=ap.ocular_obj
        if previous:previous.hide_set(True);previous.hide_render=True;previous.name='Ocular_Previous_Shell'
        shell.name='Ocular_Shell';ap.ocular_obj=shell
        for face in shell.data.polygons:face.use_smooth=True
        if w.thickness_basis!='RIM':fitting_normals(shell,range(len(outer),2*len(outer)),wc,u,inner_radius,flatten,offset)
        for old in (p.color_obj,p.clear_obj):
            if old:old.hide_set(True);old.hide_render=True
        mark.restore(context);ball.hide_set(True);shell.hide_set(False)
        preview_sclera(context)
        w.report='Cornea added above the transferred iris. Load and align the two photographs.' if cornea else f'Ocular created: outer radius {radius:.2f} mm; body thickness {w.thickness:.2f} mm ({w.thickness_basis.lower()}); inner flattening {flatten:.0%}. Mark the iris next.'
        if w.thickness_basis=='RIM':w.report=f"Gentle fitting-to-side blend. Center body thickness approximately {shell['workflow_center_body_thickness_mm']:.2f} mm."
        return shell
    except Exception:mu.delete_object(shell);raise


def sculpt_mask(context,capture=False):
    p=context.scene.ocular_production
    if capture:
        obj=p.mark_obj
        if not obj or context.object!=obj:raise RuntimeError('Return to the ocular marking copy first')
        if obj.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        attr=obj.data.attributes.get('.sculpt_mask')
        if not attr:raise RuntimeError('Paint a sculpt mask first')
        values=np.zeros(len(obj.data.vertices),np.float32);attr.data.foreach_get('value',values)
        mark.write_values(obj,values,obj.get('ocular_active_region',p.mark_region));return
    obj=mark.ensure_marks(context);attr=obj.data.attributes.get('.sculpt_mask') or obj.data.attributes.new('.sculpt_mask','FLOAT','POINT')
    attr.data.foreach_set('value',mark.values_for(obj));bpy.ops.object.mode_set(mode='SCULPT')
    try:bpy.ops.brush.asset_activate(asset_library_type='ESSENTIALS',relative_asset_identifier='brushes/essentials_brushes-mesh_sculpt.blend/Brush/Mask')
    except RuntimeError:pass  # Native brush can also be chosen in Blender's brush shelf.


def size_mode_changed(w,context):
    if w.iris_size_mode=='MANUAL' and not w.is_property_set('iris_manual_diameter'):
        w.iris_manual_diameter=w.iris_diameter
    if w.cornea_size_mode=='MANUAL' and not w.is_property_set('cornea_manual_diameter'):
        diameter=w.iris_manual_diameter if w.iris_size_mode=='MANUAL' else w.iris_diameter
        w.cornea_manual_diameter=diameter*w.cornea_width


class OcularSteps(bpy.types.PropertyGroup):
    show_advanced:bpy.props.BoolProperty(name='Advanced ocular settings',default=False)
    edit_marks:bpy.props.BoolProperty(default=False)
    show_legacy:bpy.props.BoolProperty(name='Show earlier ocular tools',default=False)
    ball:bpy.props.PointerProperty(type=bpy.types.Object)
    use_measured_diameter:bpy.props.BoolProperty(name='Use measured eyeball diameter',default=False,description='Keep your measured diameter and fit only the sphere position; does not resize the scan')
    measured_diameter:bpy.props.FloatProperty(name='Eyeball diameter (mm)',default=24,min=1,max=100)
    thickness:bpy.props.FloatProperty(name='Shell thickness (mm)',default=4.,min=.3,max=8)
    edge_radius:bpy.props.FloatProperty(name='Edge rounding (mm)',default=1.5,min=0,max=2,description='Width of the single curve joining front to back; limited to 45% of reference shell thickness')
    inner_flatten:bpy.props.FloatProperty(name='Inner curve flattening',default=.90,min=0,max=.95,subtype='FACTOR',description='Higher values make the fitting face much shallower; 90% leaves a gentle concavity. Center thickness is retained when using Center.')
    thickness_basis:bpy.props.EnumProperty(name='Thickness behavior',items=[('RIM','Fill and round back','Fill the concavity and add depth for a broad, gently curved join into the sides'),('CENTER','Keep center thickness','Hold thickness at the center; rim/body can be thinner'),('MINIMUM','Minimum body thickness','Hold minimum away from the rounded rim; center becomes thicker')],default='RIM')
    back_blend_depth:bpy.props.FloatProperty(name='Back blend depth (mm)',default=2.5,min=0,max=8,description='Extra body depth for a broad rounded back-to-side transition; increases center and rear rim thickness together')
    rim_reference_flatten:bpy.props.FloatProperty(default=.6,min=0,max=.95,options={'HIDDEN'},description='Reference fitting curve whose perimeter is retained when filling the concavity')
    iris_xy:bpy.props.FloatVectorProperty(size=2)
    iris_diameter:bpy.props.FloatProperty(name='Transferred iris diameter (mm)',default=10,min=1,max=30)
    measured_iris_diameter:bpy.props.FloatProperty(default=0,options={'HIDDEN'})
    iris_size_mode:bpy.props.EnumProperty(name='Iris size',items=[('MARKED','From scan marks','Use the marked iris diameter'),('MANUAL','Set diameter','Keep the marked center and choose the diameter')],default='MARKED',update=size_mode_changed)
    iris_manual_diameter:bpy.props.FloatProperty(name='Iris diameter (mm)',default=12,min=1,max=30)
    iris_fit:bpy.props.StringProperty()
    iris_ball_signature:bpy.props.StringProperty()
    cornea_height:bpy.props.FloatProperty(name='Corneal projection (mm)',default=1.0,min=0,max=4,description='Additional dome height above the fitted scleral sphere')
    cornea_width:bpy.props.FloatProperty(name='Cornea / iris diameter',default=1.05,min=1,max=1.5)
    cornea_size_mode:bpy.props.EnumProperty(name='Cornea size',items=[('FOLLOW','Follow iris','Scale the cornea with the iris'),('MANUAL','Set diameter','Choose a cornea diameter in millimetres')],default='FOLLOW',update=size_mode_changed)
    cornea_manual_diameter:bpy.props.FloatProperty(name='Cornea diameter (mm)',default=12.6,min=1,max=40)
    report:bpy.props.StringProperty()


class OCULAR_OT_steps(bpy.types.Operator):
    bl_idname='anaplast.ocular_steps';bl_label='Ocular production step';bl_options={'REGISTER','UNDO'}
    action:bpy.props.EnumProperty(items=[(x,x,'') for x in ('SCLERA','BALL','SHELL','IRIS','TRANSFER','CORNEA','MASK','CAPTURE','TOGGLE','DONE')])
    def execute(self,context):
        try:
            if self.action in {'SCLERA','IRIS'}:
                context.scene.ocular_production.mark_region=self.action;mark.ensure_marks(context);context.scene.ocular_steps.edit_marks=True
            elif self.action=='DONE':mark.restore(context);context.scene.ocular_steps.edit_marks=False
            elif self.action=='BALL':fit_ball(context)
            elif self.action=='SHELL':build_shell(context)
            elif self.action=='TRANSFER':fit_iris(context)
            elif self.action=='CORNEA':build_shell(context,True)
            elif self.action=='MASK':sculpt_mask(context)
            elif self.action=='CAPTURE':sculpt_mask(context,True)
            elif self.action=='TOGGLE':
                ball=context.scene.ocular_steps.ball
                if ball:ball.hide_set(not ball.hide_get())
        except Exception as e:rep(self,{'ERROR'},f'Ocular step ({self.action}): {e}');return {'CANCELLED'}
        return {'FINISHED'}


def draw(l,context):
    w=context.scene.ocular_steps;p=context.scene.ocular_production
    box=l.box();box.label(text='1. Sclera and fitted ball');box.operator('anaplast.ocular_steps',text='Mark sclera').action='SCLERA'
    row=box.row(align=True)
    for label,erase,lasso in [('Paint',False,False),('Erase',True,False),('Lasso',False,True)]:
        op=row.operator('anaplast.wedge_mark_tool',text=label);op.target='OCULAR';op.erase=erase;op.lasso=lasso
    box.prop(p,'mark_radius');row=box.row(align=True);row.operator('anaplast.ocular_steps',text='Sculpt mask').action='MASK';row.operator('anaplast.ocular_steps',text='Use sculpt mask').action='CAPTURE'
    box.prop(w,'use_measured_diameter')
    if w.use_measured_diameter:box.prop(w,'measured_diameter');box.label(text=f'Radius: {w.measured_diameter*.5:.2f} mm')
    box.operator('anaplast.ocular_steps',text='Fit ball to marked sclera').action='BALL'
    if w.ball:box.label(text=f"Fitted diameter: {2*w.ball['fit_radius_mm']:.2f} mm; radius: {w.ball['fit_radius_mm']:.2f} mm");box.operator('anaplast.ocular_steps',text='Show / hide fitted ball').action='TOGGLE'
    box=l.box();box.label(text='2. Concave-convex ocular');box.prop(w,'thickness_basis');box.prop(w,'thickness',text='Reference thickness (mm)' if w.thickness_basis=='RIM' else 'Shell thickness (mm)');box.prop(w,'inner_flatten');box.prop(w,'edge_radius');box.prop(context.scene.anaplast,'ocular_extension')
    if w.thickness_basis=='RIM':box.prop(w,'back_blend_depth');box.label(text='Fills the concavity; center becomes thicker.')
    if w.use_measured_diameter:box.label(text='Extension follows the sphere using an arc-length chart')
    box.operator('anaplast.ocular_steps',text='Create ocular from fitted ball').action='SHELL'
    box=l.box();box.label(text='3. Transfer iris');box.operator('anaplast.ocular_steps',text='Mark iris on scan (include pupil)').action='IRIS'
    row=box.row(align=True)
    for label,erase,lasso in [('Paint',False,False),('Erase',True,False),('Lasso',False,True)]:
        op=row.operator('anaplast.wedge_mark_tool',text=label);op.target='OCULAR';op.erase=erase;op.lasso=lasso
    row=box.row(align=True);row.operator('anaplast.ocular_steps',text='Sculpt mask').action='MASK';row.operator('anaplast.ocular_steps',text='Use sculpt mask').action='CAPTURE'
    box.operator('anaplast.ocular_steps',text='Transfer iris center and diameter').action='TRANSFER'
    box.prop(w,'iris_size_mode')
    if w.iris_size_mode=='MANUAL':box.prop(w,'iris_manual_diameter')
    if w.iris_fit:box.label(text=f'Transferred iris: {w.iris_diameter:.2f} mm')
    box=l.box();box.label(text='4. Cornea over iris');box.prop(w,'cornea_height');box.prop(w,'cornea_size_mode')
    if w.cornea_size_mode=='MANUAL':box.prop(w,'cornea_manual_diameter')
    else:box.prop(w,'cornea_width')
    box.operator('anaplast.ocular_steps',text='Build / update cornea').action='CORNEA'
    if w.report:l.label(text=w.report)


from .ocular_simple import OCULAR_OT_generate,OCULAR_OT_example_photos
_classes=(OcularSteps,OCULAR_OT_steps,OCULAR_OT_generate,OCULAR_OT_example_photos)
def register():
    for c in _classes:bpy.utils.register_class(c)
    bpy.types.Scene.ocular_steps=bpy.props.PointerProperty(type=OcularSteps)
def unregister():
    del bpy.types.Scene.ocular_steps
    for c in reversed(_classes):bpy.utils.unregister_class(c)

