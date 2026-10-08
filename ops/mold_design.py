"""Continuous parting surfaces and dimensions for the automatic mold."""
import math
import numpy as np
import bpy
import bmesh
from mathutils import Vector
from ..utils import mesh as mu


def has_self_intersections(obj):
    """Closed edges do not imply a valid solid: ignore only adjacent triangles."""
    from mathutils.bvhtree import BVHTree
    bm=bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bmesh.ops.triangulate(bm,faces=bm.faces[:])
        bm.faces.ensure_lookup_table()
        tree=BVHTree.FromBMesh(bm)
        for a,b in tree.overlap(tree):
            if a < b and set(bm.faces[a].verts).isdisjoint(bm.faces[b].verts):
                return True
        return False
    finally:
        bm.free()


def interpolate_grid(gx, gy, heights, x, y, fallback):
    x, y = np.broadcast_arrays(np.asarray(x), np.asarray(y))
    tx = np.clip((x-gx[0])/(gx[1]-gx[0]), 0, len(gx)-1)
    ty = np.clip((y-gy[0])/(gy[1]-gy[0]), 0, len(gy)-1)
    i = np.minimum(tx.astype(int), len(gx)-2)
    j = np.minimum(ty.astype(int), len(gy)-2)
    a, b = tx-i, ty-j
    values = np.stack((heights[j,i], heights[j,i+1], heights[j+1,i], heights[j+1,i+1]))
    weights = np.stack(((1-a)*(1-b), a*(1-b), (1-a)*b, a*b))
    weights = np.where(np.isfinite(values), weights, 0.)
    total = weights.sum(axis=0)
    return np.where(total > 1e-12, (np.nan_to_num(values)*weights).sum(axis=0)/np.maximum(total,1e-12), fallback)


class OutlineSampler:
    def __init__(self, hull):
        self.a = np.asarray(hull, dtype=float)
        self.e = np.roll(self.a,-1,axis=0)-self.a
        self.ee = np.maximum((self.e*self.e).sum(axis=1), 1e-12)
        self.center = self.a.mean(axis=0)
        self.radius = np.linalg.norm(self.a-self.center,axis=1).max()

    def distance(self, p):
        ap = np.asarray(p)-self.a
        t = np.clip((ap*self.e).sum(axis=1)/self.ee,0.,1.)
        d = np.linalg.norm(ap-t[:,None]*self.e,axis=1).min()
        inside = np.all(self.e[:,0]*ap[:,1]-self.e[:,1]*ap[:,0]>=0)
        return -d if inside else d

    def point(self, angle, distance):
        direction = np.array((math.cos(angle),math.sin(angle)))
        lo, hi = 0., self.radius+max(distance,0.)+2.
        for _ in range(30):
            r = (lo+hi)*.5
            if self.distance(self.center+direction*r) < distance:
                lo = r
            else:
                hi = r
        return self.center+direction*((lo+hi)*.5)


PRY_RING_GAP_MM = 1.0


def pry_depth_to_ring(hull, rim, outward, tangent, width, ring_outer, limit):
    """Reach toward the ring while preserving its bridge across the whole slot.

    Measure the complete inner edge against the polygon segments, including
    corners between samples. All dimensions are in the mold's XY millimetres.
    """
    poly = np.asarray(hull, float)
    edges = np.roll(poly, -1, axis=0) - poly
    lengths = np.maximum(np.sum(edges * edges, axis=1), 1e-12)
    rim, outward, tangent = map(np.asarray, (rim, outward, tangent))
    required = ring_outer + PRY_RING_GAP_MM

    def clear(depth):
        a = rim - outward * depth - tangent * width * .5
        b = a + tangent * width
        # Distance from both slot ends to every polygon segment.
        delta = np.array((a, b))[:, None, :] - poly[None, :, :]
        t = np.clip(np.sum(delta * edges, axis=2) / lengths, 0., 1.)
        distance = float(np.linalg.norm(delta - t[:, :, None] * edges, axis=2).min())
        # And from every polygon vertex to the slot edge.
        v = b - a
        t = np.clip((poly - a) @ v / max(v @ v, 1e-12), 0., 1.)
        distance = min(distance, float(np.linalg.norm(poly - a - t[:, None] * v, axis=1).min()))
        cross = lambda x, y: x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
        denominator = cross(v, edges)
        valid = abs(denominator) > 1e-12
        safe = np.where(valid, denominator, 1.)
        s = cross(poly - a, edges) / safe
        t = cross(poly - a, v) / safe
        if np.any(valid & (s >= 0.) & (s <= 1.) & (t >= 0.) & (t <= 1.)):
            return False
        # A slot fully inside the convex anatomy is also outside the land.
        if np.all(cross(edges, (a + b) * .5 - poly) >= 0.):
            return False
        return distance >= required

    if not clear(0.):
        return 0.
    # Find the first obstruction, never a second clear region beyond anatomy.
    lo = 0.
    for hi in np.linspace(0., limit, 33)[1:]:
        if not clear(hi):
            for _ in range(28):
                mid = (lo + hi) * .5
                if clear(mid): lo = mid
                else: hi = mid
            return float(lo)
        lo = hi
    return float(lo)


def contoured_pry_cutter(name, rim, outward, tangent, depth, width, opening,
                         frame, gx, gy, field, fallback, col):
    """A rounded slot that follows the land instead of one tilted plane."""
    from .mold_auto import heightfield_slab, rounded_slot, cut
    up, right, fwd = frame
    rim, outward, tangent = map(np.asarray, (rim, outward, tangent))
    step = min(.3, float(gx[1] - gx[0]), float(gy[1] - gy[0]))
    axial = np.linspace(-depth - .5, 10.5, int(math.ceil((depth + 11.) / step)) + 1)
    lateral = np.linspace(-width * .5 - .5, width * .5 + .5, int(math.ceil((width + 1.) / step)) + 1)
    A, B = np.meshgrid(axial, lateral)
    X = rim[0] + A * outward[0] + B * tangent[0]
    Y = rim[1] + A * outward[1] + B * tangent[1]
    H = interpolate_grid(gx, gy, field, X, Y, fallback)
    rv = right * float(outward[0]) + fwd * float(outward[1])
    tv = right * float(tangent[0]) + fwd * float(tangent[1])
    center = right * float(rim[0]) + fwd * float(rim[1]) + rv * ((10. - depth) * .5)
    guard = rounded_slot(name + '_guard', center + up * float((H.min() + H.max()) * .5),
                         rv, tv, up, 10. + depth, width, float(np.ptp(H)) + opening + 4.,
                         min(1., width * .25, depth * .3), col)
    tool = None
    try:
        tool = heightfield_slab(name, X, Y, H + opening * .5, np.ones_like(H, bool), opening, frame, col)
        cut(tool, guard, 'INTERSECT', solver='EXACT')
        return tool
    except Exception:
        if tool is not None: mu.delete_object(tool)
        raise
    finally:
        mu.delete_object(guard)


def flat_land(gx, gy, heights, distance, mask, skin_outline, skin, transition, follow=0.):
    """Ease outward toward a smoothed skin-height continuation or one plane."""
    near=mask & (np.abs(distance-skin)<=1.5*(gx[1]-gx[0])) & np.isfinite(heights)
    plane=float(np.median(heights[near])) if near.any() else float(np.nanmedian(heights[mask]))
    target=np.full_like(heights,plane,dtype=float)
    if follow>0:
        # Sample the closed skin boundary at uniform arc length, remove local
        # texture, then carry its broad height changes outward along the land.
        poly=np.asarray(skin_outline,float);edges=np.roll(poly,-1,axis=0)-poly
        lengths=np.linalg.norm(edges,axis=1);arc=np.r_[0,np.cumsum(lengths)]
        count=max(32,int(np.ceil(arc[-1]/max(.25,gx[1]-gx[0]))))
        s=np.arange(count)*arc[-1]/count;idx=np.minimum(np.searchsorted(arc,s,side='right')-1,len(poly)-1)
        xy=poly[idx]+edges[idx]*((s-arc[idx])/np.maximum(lengths[idx],1e-9))[:,None]
        values=interpolate_grid(gx,gy,heights,xy[:,0],xy[:,1],plane)
        sigma=4./(arc[-1]/count);span=int(np.ceil(3*sigma))
        weights=np.exp(-.5*(np.arange(-span,span+1)/sigma)**2);weights/=weights.sum()
        values=sum(w*np.roll(values,k) for k,w in zip(range(-span,span+1),weights))
        X,Y=np.meshgrid(gx,gy);best=np.full(X.shape,np.inf);param=np.zeros(X.shape)
        for i,(a,e) in enumerate(zip(poly,edges)):
            t=np.clip(((X-a[0])*e[0]+(Y-a[1])*e[1])/max(lengths[i]**2,1e-12),0,1)
            d2=(X-a[0]-t*e[0])**2+(Y-a[1]-t*e[1])**2
            param=np.where(d2<best,arc[i]+t*lengths[i],param);best=np.minimum(best,d2)
        periodic=np.interp(param.ravel(),np.r_[s,arc[-1]],np.r_[values,values[0]]).reshape(X.shape)
        target=plane+(periodic-plane)*follow
        # Closest-boundary coordinates have derivative breaks where the nearest
        # segment changes at rounded polygon corners. Smooth the continuation
        # in the plane as well, rather than extruding those breaks into ridges.
        sigma=3./(gx[1]-gx[0]);span=int(np.ceil(3*sigma))
        weights=np.exp(-.5*(np.arange(-span,span+1)/sigma)**2);weights/=weights.sum()
        for axis in (0,1):
            pads=[(0,0),(0,0)];pads[axis]=(span,span);pad=np.pad(target,pads,mode='edge')
            if axis==0:target=sum(w*pad[i:i+target.shape[0],:] for i,w in enumerate(weights))
            else:target=sum(w*pad[:,i:i+target.shape[1]] for i,w in enumerate(weights))
    # Harmonic extension lets neighbouring regions share the height change.
    # Copying the raw scan derivative into each radial strip amplified noise
    # and steep local slopes. Only the skin boundary stays pinned to the scan.
    field=np.where(mask&(distance<=skin),np.nan_to_num(heights,nan=plane),target).copy()
    movable=mask & (distance>skin)
    field[distance>=skin+transition]=target[distance>=skin+transition]
    for _ in range(700):
        pad=np.pad(field,1,mode='edge')
        average=(pad[:-2,1:-1]+pad[2:,1:-1]+pad[1:-1,:-2]+pad[1:-1,2:])*.25
        field=np.where(movable,average,field)
        field[distance>=skin+transition]=target[distance>=skin+transition]
    t=np.clip((distance-skin)/transition,0.,1.)
    blend=t*t*t*(10+t*(-15+6*t))
    h=field*(1-blend)+target*blend
    return np.where(distance<=skin,heights,h),plane


def smooth_scan_copy(source, frame, hull, skin, floor, fit, band, col):
    if fit <= 0 and band <= 0:
        return source
    from .mold_auto import dist_to_polygon, smoothstep
    obj=mu.duplicate_object(source,'Mold_SmoothedScan_tmp',col)
    n=len(obj.data.vertices)
    co=np.empty(n*3);obj.data.vertices.foreach_get('co',co);co=co.reshape(-1,3)
    matrix=np.array(obj.matrix_world);world=co@matrix[:3,:3].T+matrix[:3,3]
    up,right,fwd=map(np.asarray,frame)
    d=dist_to_polygon(np.column_stack((world@right,world@fwd)),hull)
    blend=min(.75,max(skin*.25,.1))
    central=1-smoothstep((d+blend)/(2*blend))
    outer=1-smoothstep((d-skin+blend)/blend)
    weight=(fit*central+band*(1-central))*outer
    weight=np.where(world@up>=floor,weight,0.)
    group=obj.vertex_groups.new(name='Mold optional smoothing')
    # Quantized weights let Blender receive batches instead of one call per vertex.
    quant=np.rint(np.clip(weight,0.,1.)*100).astype(int)
    for w in range(1,101):
        ids=np.flatnonzero(quant==w).tolist()
        if ids:group.add(ids,w/100.,'REPLACE')
    def setup(mod):
        mod.factor=1.;mod.iterations=12;mod.vertex_group=group.name
    mu.apply_modifier(obj,'SMOOTH',setup)
    return obj


def key_fields(X,Y,hull,p,schedule,ring_d,spillways,ring):
    """C1 key relief in the shared land field, avoiding Boolean root seams.

    Both profiles have zero height and zero gradient at their footprint edge.
    The socket allowance fades there too, retaining contact on the outer land.
    """
    sampler=OutlineSampler(hull);male=np.zeros_like(X);socket=male.copy()
    h=p.m_key_diameter*.65
    extra=h*math.tan(p.m_key_flare)+p.m_key_blend
    wall=p.m_skin+p.m_land
    for angle in schedule['keys']:
        if p.m_key_type=='CAPSULE':
            cx,cy=sampler.point(angle,(ring_d+wall)*.5)
            radius=p.m_key_diameter*.5+extra
            # Reject footprints which would cut into the skin or ring.
            available=(wall-ring_d)*.5-(p.m_spillway_width*.5 if ring and spillways else 0.)-.3
            if radius>available:raise RuntimeError('Key flare/blend is wider than the available land. Increase land width or reduce key diameter, flare or blend.')
            distance=np.hypot(X-cx,Y-cy)
            f=np.maximum(0.,1-(distance/radius)**2)**2
            fs=np.maximum(0.,1-(distance/(radius+p.key_clearance))**2)**2
            value=h*f;larger=(h+p.key_clearance)*fs
        else:
            start=(ring_d+p.m_spillway_width*.5 if ring and spillways else p.m_skin)+p.m_wedge_ring_gap
            sx,sy=sampler.point(angle,start);ex,ey=sampler.point(angle,wall)
            vx,vy=ex-sx,ey-sy;length=math.hypot(vx,vy);vx/=length;vy/=length
            axial=(X-sx)*vx+(Y-sy)*vy;across=-(X-sx)*vy+(Y-sy)*vx
            nose=max(p.m_wedge_inner*.5,1.)
            t=np.clip((axial-nose)/max(length-nose,.01),0,1);t=t*t*(3-2*t)
            radius=.5*(p.m_wedge_inner+(p.m_key_diameter-p.m_wedge_inner)*t)+extra
            # Elliptical nose, then a gently widening body continuing beyond the rim.
            rounded=np.sqrt(np.maximum(0.,1-(np.maximum(0.,nose-axial)/nose)**2))
            r=radius*rounded
            fade=np.clip(axial/nose,0,1);fade=fade*fade*(3-2*fade)
            f=np.maximum(0.,1-(across/np.maximum(r,1e-8))**2)**2*fade
            fs=np.maximum(0.,1-(across/np.maximum(r+p.key_clearance*fade,1e-8))**2)**2*fade
            value=h*f;larger=(h+p.key_clearance)*fs
        male=np.maximum(male,value);socket=np.maximum(socket,larger)
    return male,socket


def spillway_start(skin,width,ring,ring_distance):
    # The visible mouth is on the ring's outer wall. Overlap only the outer
    # half of the already-empty ring so the passages share an open cross-section
    # instead of merely touching at the ring's outermost tangent point.
    return ring_distance if ring else skin+.5


def bullet_key(name, sampler, angle, start, wall, inner_d, outer_d, up, right, fwd, height, col, clearance=0., socket=False):
    """A rounded nose followed by a widening circular body; no blunt inner cap."""
    radial=(right*math.cos(angle)+fwd*math.sin(angle)).normalized()
    tang=up.cross(radial).normalized()
    tip=start+(0. if socket else clearance)
    end=wall+max(8.,outer_d+2.)
    nose=inner_d*.5
    # Nonzero first ring avoids coincident vertices at a zero-radius pole.
    ds=list(np.linspace(.001,nose,9))+list(np.arange(nose+.4,end-tip,.4))+[end-tip]
    bm=bmesh.new();rings=[]
    for s in sorted(set(ds)):
        if s<=nose:
            radius=inner_d*.5*math.sqrt(max(0.,1-(1-s/nose)**2))
        else:
            t=min(1.,(s-nose)/max(wall-tip-nose,1e-6))
            t=t*t*(3-2*t)
            radius=.5*(inner_d+(outer_d-inner_d)*t)
        radius=float(max(.0001,radius-(0. if socket else clearance)))
        px,py=sampler.point(angle,min(tip+s,wall))
        c=right*px+fwd*py+up*height(px,py)+radial*max(0.,tip+s-wall)
        rings.append([bm.verts.new(c+(tang*math.cos(a)+up*math.sin(a))*radius) for a in np.linspace(0,2*math.pi,24,endpoint=False)])
    for a,b in zip(rings[:-1],rings[1:]):
        for i in range(24):bm.faces.new((a[i],a[(i+1)%24],b[(i+1)%24],b[i]))
    bm.faces.new(list(reversed(rings[0])));bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm,faces=bm.faces)
    bmesh.ops.triangulate(bm,faces=bm.faces)
    return mu.new_mesh_object(name,bm,col)
