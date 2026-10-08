"""A thin, sealed cap with weakness only on its non-cavity side."""
import math
import bmesh
import numpy as np
from .mold_auto import dist_to_polygon
from ..utils import mesh as mu


def clean_backing_topology(part):
    """Remove sub-micron slivers and paired zero-thickness internal triangles."""
    bm=bmesh.new()
    try:
        bm.from_mesh(part.data)
        bmesh.ops.remove_doubles(bm,verts=bm.verts[:],dist=2e-5)
        # Narrow Boolean triangles can be valid parts of a closed wall.
        # Dissolving them opens the rim and can create cavity intersections.
        # Weld coincident endpoints, but retain the connecting faces.
        bm.verts.index_update()
        suspect={f for e in bm.edges if len(e.link_faces)>2 for f in e.link_faces}
        groups={}
        for face in suspect:groups.setdefault(tuple(sorted(v.index for v in face.verts)),[]).append(face)
        doomed=[]
        for faces in groups.values():
            if len(faces)==2 and faces[0].normal.dot(faces[1].normal)<0:doomed.extend(faces)
            elif len(faces)>1:doomed.extend(faces[1:])
        if doomed:bmesh.ops.delete(bm,geom=doomed,context='FACES_ONLY')
        orphan=[v for v in bm.verts if not v.link_faces]
        if orphan:bmesh.ops.delete(bm,geom=orphan,context='VERTS')
        wire=[e for e in bm.edges if not e.link_faces]
        if wire:bmesh.ops.delete(bm,geom=wire,context='EDGES')
        # Dissolving Boolean slivers can leave a triangular opening only a few
        # microns wide. Close those collapsed triangles without moving a vertex
        # or filling any genuine opening in the surface.
        boundary=set(e for e in bm.edges if e.is_boundary)
        while boundary:
            stack=[boundary.pop()];edges=[]
            while stack:
                edge=stack.pop();edges.append(edge)
                for v in edge.verts:
                    for other in v.link_edges:
                        if other in boundary:boundary.remove(other);stack.append(other)
            vertices=list({v for e in edges for v in e.verts})
            if len(edges)==3 and len(vertices)==3:
                longest=max(e.calc_length() for e in edges)
                twice_area=(vertices[1].co-vertices[0].co).cross(vertices[2].co-vertices[0].co).length
                altitude=twice_area/max(longest,1e-12)
                if longest<1. and altitude<.001:
                    # A collinear triangular boundary is a T-junction, not a
                    # hole with area. Split its long edge at the existing
                    # middle vertex and weld the two sides without moving it.
                    edge=max(edges,key=lambda e:e.calc_length())
                    middle=next(v for v in vertices if v not in edge.verts)
                    a,b=edge.verts;axis=b.co-a.co
                    fraction=(middle.co-a.co).dot(axis)/max(axis.length_squared,1e-20)
                    if .0001<fraction<.9999:
                        _,new=bmesh.utils.edge_split(edge,a,fraction)
                        bmesh.ops.pointmerge(bm,verts=[middle,new],merge_co=middle.co.copy())
                elif longest<1. and altitude<.001:
                    bmesh.ops.holes_fill(bm,edges=edges,sides=3)
        bm.to_mesh(part.data);part.data.update()
    finally:bm.free()


def repair_backing(part,reference,up,outline,frame,z_lo,z_hi,remaining):
    """Resolve isolated backing crossings while locking original cavity vertices."""
    from mathutils.bvhtree import BVHTree
    from .mold_auto import _extract_inner_sheet
    front=mu.duplicate_object(reference,'Mold_ThinFront_tmp',part.users_collection[0])
    bm=bmesh.new();fb=bmesh.new();moved=set()
    try:
        _extract_inner_sheet(front,up,outline,frame,z_lo,z_hi,True)
        fb.from_mesh(front.data);fb.transform(front.matrix_world);front_tree=BVHTree.FromBMesh(fb)
        bm.from_mesh(part.data);bm.transform(part.matrix_world);bm.verts.ensure_lookup_table();bm.verts.index_update()
        bmesh.ops.triangulate(bm,faces=bm.faces[:]);bm.faces.ensure_lookup_table()
        original=[v.co.copy() for v in bm.verts]
        for attempt in range(8):
            tree=BVHTree.FromBMesh(bm);bad=set()
            for a,b in tree.overlap(tree):
                if a>=b:continue
                one,two=bm.faces[a],bm.faces[b]
                if not set(one.verts).isdisjoint(two.verts):continue
                if min((v.co-w.co).length for v in one.verts for w in two.verts)<1e-5:continue
                bad.update(one.verts);bad.update(two.verts)
            if not bad:
                inv=part.matrix_world.inverted()
                distance=max(((bm.verts[i].co-original[i]).length for i in moved),default=0.)
                if moved:
                    # Commit the same triangles that were checked. Reusing the
                    # old non-planar quads can choose a different diagonal and
                    # reintroduce crossings after the coordinate update.
                    bm.transform(inv);bm.to_mesh(part.data);part.data.update()
                return len(moved),distance
            if len(bad)>2000:raise RuntimeError('Too many intersecting backing regions for a thin cap; increase the cap wall or simplify the source')
            for _ in range(2):bad.update({e.other_vert(v) for v in list(bad) for e in v.link_edges})
            movable={v for v in bad if front_tree.find_nearest(v.co)[3]>1e-4}
            if not movable:raise RuntimeError('The source cavity contains intersections that prevent a sealed thin cap')
            positions={}
            for v in movable:
                neighbours=[e.other_vert(v).co for e in v.link_edges]
                q=v.co*.5+sum(neighbours,v.co*0.)*(.5/len(neighbours))
                loc,n,_,distance=front_tree.find_nearest(q)
                floor=min(.025,remaining*.5)
                if distance<floor or (q-loc).dot(n)>0:q=loc-n*floor
                positions[v]=q
            for v,q in positions.items():v.co=q;moved.add(v.index)
        raise RuntimeError('The thin-cap backing still intersects after local repair; no printable cap was confirmed')
    finally:
        bm.free();fb.free();mu.delete_object(front)


def thickness_field(world, frame, outline, wall, remaining, pattern, width, count, spacing, rotation):
    up,right,fwd=map(np.asarray,frame)
    xy=np.column_stack((world@right,world@fwd))
    center=np.asarray(outline).mean(axis=0);q=xy-center
    cs,sn=math.cos(rotation),math.sin(rotation)
    q=q@np.array(((cs,-sn),(sn,cs)))
    if pattern=='NONE':return np.full(len(world),wall)
    if pattern=='GRID':
        d=np.minimum(np.abs((q[:,0]+spacing*.5)%spacing-spacing*.5),np.abs((q[:,1]+spacing*.5)%spacing-spacing*.5))
    else:
        d=np.full(len(world),np.inf)
        for a in np.arange(count)*2*math.pi/count:
            along=q[:,0]*math.cos(a)+q[:,1]*math.sin(a)
            across=np.abs(q[:,0]*math.sin(a)-q[:,1]*math.cos(a))
            d=np.minimum(d,np.where(along>=0,across,np.linalg.norm(q,axis=1)))
        if pattern=='STARTERS':
            edge=-dist_to_polygon(xy,outline)
            d=np.where(edge<=5.,d,np.inf)
    t=np.clip(d/(width*.5),0.,1.)
    blend=t*t*(3-2*t)
    return remaining+(wall-remaining)*blend


def build_cap(part, up, outline, frame, z_lo, z_hi, p):
    """Offset the enclosing solid, preserving the original cavity in differences.

    Normal extrusion folds at scan creases, even at 0.1 mm. Distance offsets
    resolve these folds before the thin skin is cut out. Grooves remove only
    backing material; they never perforate or displace the cavity surface.
    """
    from .mold_shell import build_shell
    from .mold_auto import cut,prism
    col=part.users_collection[0];reference=mu.duplicate_object(part,'Mold_ThinReference_tmp',col)
    regular=weak=None;grooves=[]
    # This explicit fine grid is essential: the usual backing grid is coarser
    # than the whole cap. Thickness is nominal at this discretisation.
    grid=min(.07,p.m_cap_wall*.7)
    try:
        print(f'Breakable cap: {grid:.3f} mm distance grid, building regular backing',flush=True)
        regular=build_shell(reference,up,p.m_cap_wall,outline,frame,z_lo,z_hi,True,grid,fixed_grid=grid,return_cutter=True)
        mu.apply_boolean(part,regular,'DIFFERENCE');mu.delete_object(regular);regular=None
        if p.m_break_pattern!='NONE':
            print('Breakable cap: building groove backing',flush=True)
            weak=build_shell(reference,up,p.m_break_remaining,outline,frame,z_lo,z_hi,True,grid,fixed_grid=grid,return_cutter=True)
            poly=np.asarray(outline);center=poly.mean(axis=0);radius=np.linalg.norm(poly-center,axis=1).max()+2.
            lines=[];rotation=p.m_break_rotation
            if p.m_break_pattern=='GRID':
                for shift in np.arange(-math.ceil(radius/p.m_break_spacing),math.ceil(radius/p.m_break_spacing)+1)*p.m_break_spacing:
                    for angle in (rotation,rotation+math.pi*.5):
                        d=np.array((math.cos(angle),math.sin(angle)));side=np.array((-d[1],d[0]))
                        lines.append((center+side*shift-d*radius,center+side*shift+d*radius))
            else:
                from .mold_design import OutlineSampler
                sampler=OutlineSampler(outline)
                for a in rotation+np.arange(p.m_break_count)*2*math.pi/p.m_break_count:
                    direction=np.array((math.cos(a),math.sin(a)))
                    if p.m_break_pattern=='STARTERS':
                        edge=sampler.point(a,0.);lines.append((edge-direction*5.,edge+direction*2.))
                    else:lines.append((center,center+direction*radius))
            pattern=None
            for i,(a,b) in enumerate(lines):
                print(f'Breakable cap: groove {i+1}/{len(lines)}',flush=True)
                direction=(b-a)/np.linalg.norm(b-a);side=np.array((-direction[1],direction[0]))*p.m_break_width*.5
                rectangle=[a-side,b-side,b+side,a+side]
                groove=prism('Mold_Groove_tmp',rectangle,z_lo-2.,z_hi+2.,frame,col);grooves.append(groove)
                if pattern is None:pattern=groove
                else:
                    mu.apply_boolean(pattern,groove,'UNION');mu.delete_object(groove);grooves.pop()
            # Intersect the distance offset once with the combined groove mask.
            # All operands have already been validated as closed solids.
            if pattern is not None:
                mu.apply_boolean(pattern,weak,'INTERSECT')
                mu.delete_object(weak);weak=None
                if len(pattern.data.polygons):mu.apply_boolean(part,pattern,'DIFFERENCE')
        # Boolean slivers can look like locked cavity crossings. Weld only the
        # existing sub-micron tolerance before trying to move backing vertices.
        clean_backing_topology(part)
        count,movement=repair_backing(part,reference,up,outline,frame,z_lo,z_hi,p.m_break_remaining)
        clean_backing_topology(part)
        print(f'Breakable cap: repaired {count} backing vertices, maximum adjustment {movement:.4f} mm',flush=True)
        stats=mu.mesh_stats(part)
        if not stats[0] or any(stats[1:]):raise RuntimeError('Thin cap did not form a closed shell')
        from .mold_design import has_self_intersections
        if has_self_intersections(part):raise RuntimeError('The thin-cap mesh still intersects itself; a printable cap was not confirmed')
        part['breakable_cap']=True;part['cap_wall_mm']=p.m_cap_wall
        part['groove_wall_mm']=p.m_break_remaining;part['break_pattern']=p.m_break_pattern;part['thin_grid_mm']=grid
        part['backing_repair_vertices']=count;part['backing_repair_max_mm']=movement
        return stats
    finally:
        for obj in grooves+[regular,weak,reference]:
            if obj is not None:mu.delete_object(obj)
