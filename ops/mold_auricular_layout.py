"""One feature plan for both interfaces of an auricular insert."""
import math
import numpy as np
from mathutils import Vector
from .mold_design import OutlineSampler,interpolate_grid,pry_depth_to_ring,PRY_RING_GAP_MM
from .mold_auto import dist_to_polygon


def contains(tree,point):
    direction=Vector((.267261,.534522,.801784))
    hit,normal,_,_=tree.ray_cast(point,direction,1000.)
    return hit is not None and normal.dot(direction)>0


def key_relief(X,Y,keys,clearance=0.):
    result=np.zeros(np.broadcast_arrays(X,Y)[0].shape,float)
    for key in keys:
        if key.get('shape')=='BULLET':
            dx=X-key['tip'][0];dy=Y-key['tip'][1]
            axial=dx*key['outward'][0]+dy*key['outward'][1]
            across=dx*key['tangent'][0]+dy*key['tangent'][1]
            nose=key['nose'];t=np.clip((axial-nose)/max(.01,key['body_length']-nose),0.,1.);t=t*t*(3-2*t)
            radius=key['inner_radius']+(key['width_radius']-key['inner_radius'])*t
            radius*=np.sqrt(np.maximum(0.,1-(np.maximum(0.,nose-axial)/nose)**2))
            fade=np.clip(axial/nose,0.,1.);fade=fade*fade*(3-2*fade)
            value=(key['height']+clearance)*np.maximum(0.,1-(across/np.maximum(radius+clearance*fade,1e-8))**2)**2*fade
            result=np.maximum(result,value)
            continue
        delta_x=X-key['xy'][0];delta_y=Y-key['xy'][1]
        tangent=key['tangent'];radial=key['outward']
        a=(delta_x*tangent[0]+delta_y*tangent[1])/(key['length_radius']+clearance)
        b=(delta_x*radial[0]+delta_y*radial[1])/(key['width_radius']+clearance)
        profile=np.maximum(0.,1-a*a-b*b)**2
        result=np.maximum(result,(key['height']+clearance)*profile)
    return result


def add_mating_relief(obj,frame,gx,gy,ground,keys,clearance=0.,sign=1.):
    """Raise the existing mating mesh; no separate overlapping key solids."""
    u,r,f=frame
    if any(k.get('shape')=='BULLET' for k in keys):
        import bmesh
        bm=bmesh.new();bm.from_mesh(obj.data);bm.verts.ensure_lookup_table();bm.verts.index_update()
        co=np.array([obj.matrix_world@v.co for v in bm.verts]);xx=co@np.array(r);yy=co@np.array(f);zz=co@np.array(u)
        expected=interpolate_grid(gx,gy,ground,xx,yy,0.);relief=key_relief(xx,yy,keys,clearance)
        on_surface=abs(zz-expected)<.15;edges=[]
        for edge in bm.edges:
            a,b=[v.index for v in edge.verts]
            if on_surface[a] and on_surface[b] and (relief[a]>0 or relief[b]>0):edges.append(edge)
        if edges:bmesh.ops.subdivide_edges(bm,edges=edges,cuts=2,use_grid_fill=True)
        bm.to_mesh(obj.data);bm.free();obj.data.update()
    world=np.array([obj.matrix_world@v.co for v in obj.data.vertices])
    x=world@np.array(r);y=world@np.array(f);z=world@np.array(u)
    expected=interpolate_grid(gx,gy,ground,x,y,float(np.median(ground)))
    relief=key_relief(x,y,keys,clearance)
    movable=(relief>0)&(abs(z-expected)<.15)
    inverse=obj.matrix_world.inverted()
    for index in np.flatnonzero(movable):
        obj.data.vertices[int(index)].co=inverse@(Vector(world[index])+u*float(sign*relief[index]))
    obj.data.update()
    return int(movable.sum())


def end_roof(roof,X,Y,connections,points,inside,frame,source_tree,blend):
    """C2 roof-to-end transition, extending outward below each connector."""
    u,r,f=frame;result=roof.copy()
    for connection in connections:
        end=np.asarray(connection['inner_xy'],float);outer=np.asarray(connection['outer_xy'],float)
        along=outer-end;along/=np.linalg.norm(along);inward=np.array([-along[1],along[0]])
        if inward@(np.asarray(inside)-end)<0:inward=-inward
        origin=points[np.argmin(np.linalg.norm(points[:,:2]-np.asarray(connection.get('anatomical_inner_xy',end)),axis=1))]
        local=points[np.linalg.norm(points-origin,axis=1)<7.];angles=[];weights=[]
        for q in local:
            n=source_tree.find_nearest(r*float(q[0])+f*float(q[1])+u*float(q[2]))[1]
            if n is None:continue
            angles.append(math.acos(min(1.,abs(n.dot(u)))));weights.append(abs(n.dot(u))**4)
        if not weights or sum(weights)<1e-6:raise RuntimeError('Select more of the under-helix surface at each end')
        angle=float(np.average(angles,weights=weights));angle=min(math.radians(80),max(math.radians(8),angle))
        distance=np.maximum(0.,-((X-end[0])*inward[0]+(Y-end[1])*inward[1]))
        t=np.clip(distance/blend,0.,1.)
        # Integral of smoothstep: zero slope/curvature at the roof, then the
        # measured terminal slope without an abrupt line at the join.
        drop=math.tan(angle)*(blend*(t**3-.5*t**4)+np.maximum(0.,distance-blend))
        result=np.minimum(result,roof-drop)
        connection['end_angle_degrees']=math.degrees(angle);connection['end_blend_mm']=float(blend)
        connection['end_anchor']='ROOF_SHORTEST_CONNECTOR';connection['end_extension']='SMOOTH_OUTWARD_TO_BASE'
    return result


def make_plan(p,hull,outline,frame,gx,gy,ground,roof,footprint_tree,angle,key_region=None):
    u,r,f=frame;sampler=OutlineSampler(hull);wall=p.m_skin+p.m_land
    ring=p.m_skin+p.m_spillway_width*.5+1.5 if p.m_ring else p.m_skin+.5
    ring_outer=ring+p.m_spillway_width*.5 if p.m_ring else p.m_skin+1.
    middle=(ring_outer+wall)*.5
    height=p.m_key_diameter*.65
    width=p.m_key_diameter*.5+height*math.tan(p.m_key_flare)+p.m_key_blend
    length=max(width*1.4,math.hypot((wall-ring_outer-p.m_wedge_ring_gap)*.5,width)+1.) if p.m_key_type=='WEDGE' else width
    if p.wedge_keys and width>(wall-ring_outer)*.5-.5:raise RuntimeError('Increase land width or reduce key diameter/flare to fit keys outside the ring')
    angles=np.linspace(0,2*math.pi,720,endpoint=False)
    xy=np.array([sampler.point(a,middle) for a in angles]);rim=np.array([sampler.point(a,wall) for a in angles])
    tangent=np.roll(rim,-1,axis=0)-np.roll(rim,1,axis=0);tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    outward=np.c_[tangent[:,1],-tangent[:,0]]
    flip=np.einsum('ij,ij->i',outward,xy-sampler.center)<0;outward[flip]*=-1
    def wedge_at(q,minimum_gap=.1):
        low=interpolate_grid(gx,gy,ground,q[:,0],q[:,1],0.)
        high=interpolate_grid(gx,gy,roof,q[:,0],q[:,1],0.)
        answer=np.zeros(len(q),bool)
        for i in np.flatnonzero(high-low>minimum_gap):
            point=r*float(q[i,0])+f*float(q[i,1])+u*float((low[i]+high[i])*.5)
            answer[i]=contains(footprint_tree,point)
            if minimum_gap>1. and answer[i]:
                for z in (low[i]+.1,high[i]+height+p.key_clearance+.1):
                    answer[i]&=contains(footprint_tree,r*float(q[i,0])+f*float(q[i,1])+u*float(z))
        return answer
    roof_dy,roof_dx=np.gradient(roof,gy[1]-gy[0],gx[1]-gx[0])
    def gentle(q):
        sx=interpolate_grid(gx,gy,roof_dx,q[:,0],q[:,1],0.)
        sy=interpolate_grid(gx,gy,roof_dy,q[:,0],q[:,1],0.)
        return np.hypot(sx,sy)<.45
    inside=wedge_at(xy);all_inside=wedge_at(xy,height+1.)&gentle(xy);all_outside=~inside
    for t in np.linspace(0,2*math.pi,12,endpoint=False):
        q=xy+tangent*(length+p.key_clearance+.5)*math.cos(t)+outward*(width+p.key_clearance+.5)*math.sin(t)
        valid=dist_to_polygon(q,outline)<-.1
        if key_region is not None:all_inside&=dist_to_polygon(q,key_region)<-.5
        mask=wedge_at(q);all_inside&=wedge_at(q,height+1.)&valid&gentle(q);all_outside&=~mask&valid
    # Reserve a flat corridor near each section centre before allocating keys.
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def pry_depth(i):
        if not p.m_ring:return p.m_pry_depth
        return pry_depth_to_ring(hull,rim[i],outward[i],tangent[i],p.m_pry_width,ring_outer,wall)
    pries=[]
    ground_dy,ground_dx=np.gradient(ground,gy[1]-gy[0],gx[1]-gx[0])
    delta_dy,delta_dx=np.gradient(roof-ground,gy[1]-gy[0],gx[1]-gx[0])
    def corridor(i):
        return np.array([rim[i]-outward[i]*d+tangent[i]*a for d in np.linspace(.25,max(.25,pry_depth(i)),7) for a in np.linspace(-p.m_pry_width*.5,p.m_pry_width*.5,7)])
    def slope(field_x,field_y,q):
        return np.hypot(interpolate_grid(gx,gy,field_x,q[:,0],q[:,1],0.),interpolate_grid(gx,gy,field_y,q[:,0],q[:,1],0.))
    def reserve_pry(rear):
        candidates=[]
        opposite=rim[inside!=rear]
        for i in np.flatnonzero(inside==rear):
            if any(np.linalg.norm(rim[i]-np.asarray(v['rim']))<p.m_pry_width+2 for v in pries):continue
            distance=float(np.linalg.norm(opposite-rim[i],axis=1).min()) if len(opposite) else 100.
            if distance<p.m_pry_width*.5+3.:continue
            if pry_depth(int(i))<.5:continue
            q=corridor(int(i))
            if np.max(slope(ground_dx,ground_dy,q))>math.tan(math.radians(10)):continue
            if rear:
                if not np.all(wedge_at(q,p.m_pry_height+1.)):continue
                if np.max(slope(delta_dx,delta_dy,q))>math.tan(math.radians(5)):continue
            elif np.any(wedge_at(q)):continue
            incline=float(np.degrees(np.arctan(np.max(slope(delta_dx,delta_dy,q))))) if rear else float(np.degrees(np.arctan(np.max(slope(ground_dx,ground_dy,q)))))
            candidates.append((distance-4.*incline,int(i)))
        if not candidates:raise RuntimeError('No flat central land remains for a pry point at all three interfaces. Widen the wedge/base coverage or reduce pry width; inclined ends are excluded')
        _,index=max(candidates)
        q=corridor(index)
        pries.append({'angle':float(angles[index]),'rim':rim[index].tolist(),'tangent':tangent[index].tolist(),'outward':outward[index].tolist(),'depth_mm':float(pry_depth(index)),'ring_gap_mm':PRY_RING_GAP_MM if p.m_ring else None,'on_wedge':rear,'interfaces':['BASE_WEDGE','WEDGE_CAP'] if rear else ['BASE_CAP'],'max_base_slope_degrees':float(np.degrees(np.arctan(np.max(slope(ground_dx,ground_dy,q))))),'max_relative_incline_degrees':float(np.degrees(np.arctan(np.max(slope(delta_dx,delta_dy,q))))) if rear else 0.})
    if p.wedge_pry:
        reserve_pry(True);reserve_pry(False)
    for _ in range(max(0,p.m_pry-len(pries))):reserve_pry(False)
    def clear_pries(q,radius):
        for item in pries:
            delta=q-np.asarray(item['rim']);a=np.maximum(0.,np.abs(delta@np.asarray(item['tangent']))-p.m_pry_width*.5)
            depth=-delta@np.asarray(item['outward']);b=np.maximum(0.,np.maximum(-depth,depth-item['depth_mm']))
            if np.any(np.hypot(a,b)<radius+1.):return False
        return True
    for i in np.flatnonzero(all_inside|all_outside):
        if not clear_pries(xy[i:i+1],length):all_inside[i]=False;all_outside[i]=False
    keys=[]
    def bullet_details(index):
        # Use the local rim normal, so the open outer face is parallel to the
        # edge. Locate the rounded nose outside the ring along that normal.
        end=rim[index];direction=outward[index];start=ring_outer+p.m_wedge_ring_gap
        lo,hi=0.,wall+10.
        for _ in range(28):
            mid=(lo+hi)*.5;point=end-direction*mid
            if sampler.distance(point)>start:lo=mid
            else:hi=mid
        body=(lo+hi)*.5;tip=end-direction*body
        return {'shape':'BULLET','tip':tip.tolist(),'nose':max(p.m_wedge_inner*.5,1.),'body_length':float(body),'inner_radius':float(p.m_wedge_inner*.5+height*math.tan(p.m_key_flare)+p.m_key_blend)}
    # Check the entire open-ended bullet corridor, including its shoulders
    # close to the rim, rather than only its central landing point.
    if p.m_key_type=='WEDGE':
        for i in np.flatnonzero(all_inside|all_outside):
            detail=bullet_details(int(i));tip=np.asarray(detail['tip']);q=[]
            for t in np.linspace(.1,.98,10):
                center=tip+outward[i]*detail['body_length']*t
                w=detail['inner_radius']+(width-detail['inner_radius'])*t
                for side in (-1,0,1):
                    point=center+tangent[i]*(w+p.key_clearance+.3)*side
                    if sampler.distance(point)<wall-.1:q.append(point)
            q=np.asarray(q)
            all_inside[i]&=bool(np.all(wedge_at(q,height+1.)&gentle(q)))
            all_outside[i]&=bool(np.all(~wedge_at(q)))
    def choose_keys(valid,count,on_wedge):
        candidates=np.flatnonzero(valid).tolist()
        for _ in range(count):
            candidates=[i for i in candidates if all(np.linalg.norm(xy[i]-np.asarray(k['xy']))>2*length+1. for k in keys)]
            if not candidates:raise RuntimeError('Not enough clear land for the requested keys; enlarge the land or reduce key size')
            if not keys:
                index=min(candidates,key=lambda i:abs(math.atan2(math.sin(angles[i]-angle),math.cos(angles[i]-angle))))
            else:index=max(candidates,key=lambda i:min(np.linalg.norm(xy[i]-np.asarray(k['xy'])) for k in keys))
            keys.append({'angle':float(angles[index]),'xy':xy[index].tolist(),'tangent':tangent[index].tolist(),'outward':outward[index].tolist(),'width_radius':float(width),'length_radius':float(length),'height':float(height),'on_wedge':bool(on_wedge)})
            if p.m_key_type=='WEDGE':keys[-1].update(bullet_details(index))
    if p.wedge_keys:
        choose_keys(all_inside,max(3,p.wedge_key_count),True)
        choose_keys(all_outside,max(2,p.m_key_count-p.wedge_key_count),False)
    outlets=[]
    @lru_cache(maxsize=None)
    def outlet_clear(i):
        path=np.array([sampler.point(angles[i],d) for d in np.linspace(ring_outer,wall+1.,12)])
        return min([np.linalg.norm(path-np.asarray(k['xy']),axis=1).min()-k['length_radius'] for k in keys]+[100.])
    def angular_distance(a,b):return abs(math.atan2(math.sin(a-b),math.cos(a-b)))
    def pick_outlet(rear):
        candidates=[i for i in range(len(angles)) if bool(inside[i])==rear and clear_pries(np.asarray([sampler.point(angles[i],d) for d in np.linspace(ring_outer,wall+1.,12)]),p.m_spillway_width*.5) and outlet_clear(i)>p.m_spillway_width*.5+1. and all(np.linalg.norm(rim[i]-np.asarray(v['rim']))>p.m_spillway_width+3. for v in outlets)]
        if not candidates:raise RuntimeError('No spillway route remains clear of the keys; enlarge the land or reduce the key size')
        index=max(candidates,key=lambda i:min([outlet_clear(i)]+[np.linalg.norm(rim[i]-np.asarray(v['rim']))*.5 for v in outlets]))
        outlets.append({'angle':float(angles[index]),'rim':rim[index].tolist(),'on_wedge':rear})
    if p.wedge_spillway:pick_outlet(True)
    for _ in range(max(0,p.m_spillways-len(outlets))):pick_outlet(False)
    return {'keys':keys,'outlets':outlets,'pries':pries,'ring_distance':float(ring),'ring_outer':float(ring_outer),'wall_distance':float(wall),'wedge_keys_per_interface':sum(k['on_wedge'] for k in keys)}


def cover_marked_roof(roof,gx,gy,points,overlap,smoothing=5.,radial=None):
    """Fair local coverage corrections without lifting the entire mating land.

    Only the selected rear surface constrains the divider. Gaussian lifting
    repairs any deficit locally after fairing, so an end point cannot raise
    every section and expose a tall wall above the anatomical contact.
    """
    step=float(gx[1]-gx[0])
    ix=np.clip(((points[:,0]-gx[0])/step).astype(int),0,len(gx)-2)
    iy=np.clip(((points[:,1]-gy[0])/(gy[1]-gy[0])).astype(int),0,len(gy)-2)
    target=np.full_like(roof,-np.inf)
    for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
        np.maximum.at(target,(iy+dy,ix+dx),points[:,2]+overlap)
    # Keep a connected, level continuation of each marked height toward the
    # rear rim. Do not constrain unrelated high extrapolations outside it.
    if radial is not None:
        span=math.hypot(gx[-1]-gx[0],gy[-1]-gy[0])
        for distance in np.arange(step,span+step,step):
            x=points[:,0]+radial[0]*distance;y=points[:,1]+radial[1]*distance
            valid=(x>=gx[0])&(x<gx[-1])&(y>=gy[0])&(y<gy[-1])
            if not valid.any():break
            a=((x[valid]-gx[0])/step).astype(int);b=((y[valid]-gy[0])/(gy[1]-gy[0])).astype(int)
            for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
                np.maximum.at(target,(b+dy,a+dx),points[valid,2]+overlap)
    obstacle=np.maximum(0.,target-roof)
    correction=obstacle.copy();penalty=(step/max(3.,smoothing))**2
    for _ in range(1800):
        a=np.pad(correction,1,mode='edge')
        new=np.maximum(obstacle,(a[:-2,1:-1]+a[2:,1:-1]+a[1:-1,:-2]+a[1:-1,2:])/(4+penalty))
        if np.max(abs(new-correction))<1e-5:correction=new;break
        correction=new
    sigma=max(2.5,smoothing*.6)/step;radius=int(math.ceil(sigma*3))
    kernel=np.exp(-.5*(np.arange(-radius,radius+1)/sigma)**2);kernel/=kernel.sum()
    def blur(values):
        for axis in (0,1):
            values=np.apply_along_axis(lambda row:np.convolve(np.pad(row,(radius,radius),mode='edge'),kernel,mode='valid'),axis,values)
        return values
    result=blur(roof+correction)
    # A broad Gaussian centred at the worst remaining deficit has unit peak.
    # Unlike a global bias it changes only the local junction needing cover.
    while True:
        deficit=target-result;index=np.unravel_index(np.argmax(deficit),deficit.shape)
        amount=float(deficit[index])
        if amount<=1e-5:break
        j,i=index;dy=(np.arange(len(gy))-j)/sigma;dx=(np.arange(len(gx))-i)/sigma
        result+=(amount+1e-5)*np.exp(-.5*dy*dy)[:,None]*np.exp(-.5*dx*dx)[None,:]
    return result,float(np.max(result-roof))
