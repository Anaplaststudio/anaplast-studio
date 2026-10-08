"""One-click two-part mold from the viewing direction, straight from the finished sculpt (no prototype needed).

Looking at the sculpt from the angle the mold should open, click once:
  Â· CAP  = the negative of the sculpt, then a band of skin (5 mm) around it, then a land (10 mm) that starts at the
           skin's own slope and eases smoothly to a plane parallel to the base;
  Â· BASE = the tissue side: the cast / skin surface under and around the sculpt, the same land, flat bottom;
  Â· rounded registration keys on the land, spillways that follow the curves of the land surface, rounded pry
    notches at the parting line on the edge.
Choose the view so the tissue has no undercuts relative to it."""
import math
import json
import numpy as np
import bpy
import bmesh
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from ..utils import mesh as mu
from .report import rep


def view_axes(context):
    rv3d = getattr(context, "region_data", None)
    if rv3d is None:
        return Vector((0, 1, 0)), Vector((1, 0, 0)), Vector((0, 0, 1))
    q = rv3d.view_rotation
    up = (q @ Vector((0, 0, 1))).normalized()
    right = (q @ Vector((1, 0, 0))).normalized()
    fwd = up.cross(right).normalized()
    return up, right, fwd


def convex_hull_2d(pts):
    pts = sorted(set(map(tuple, pts)))
    if len(pts) <= 2:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for q in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return lower[:-1] + upper[:-1]


def offset_polygon(poly, dist, steps=6):
    out = []
    for x, y in poly:
        for k in range(steps * 4):
            a = 2 * math.pi * k / (steps * 4)
            out.append((x + dist * math.cos(a), y + dist * math.sin(a)))
    return convex_hull_2d(out)


def dist_to_polygon(P, poly):
    """Signed distance to a simple polygon of either winding, including concave outlines."""
    P=np.asarray(P,dtype=np.float64);poly=np.asarray(poly,dtype=np.float64)
    distance=np.full(len(P),np.inf);inside=np.zeros(len(P),dtype=bool)
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        ab=b-a;length2=float(ab@ab)
        if length2<1e-20:continue
        ap=P-a;t=np.clip((ap@ab)/length2,0.,1.)
        distance=np.minimum(distance,np.linalg.norm(P-(a+t[:,None]*ab),axis=1))
        if abs(ab[1])>1e-15:
            crosses=(a[1]>P[:,1])!=(b[1]>P[:,1])
            at_x=a[0]+(P[:,1]-a[1])*ab[0]/ab[1]
            inside^=crosses&(P[:,0]<at_x)
    return np.where(inside|(distance<1e-8),-distance,distance)


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def prism(name, poly2d, z0, z1, frame, col):
    up, right, fwd = frame
    bm = bmesh.new()
    bottom = [bm.verts.new(right * x + fwd * y + up * z0) for x, y in poly2d]
    top = [bm.verts.new(right * x + fwd * y + up * z1) for x, y in poly2d]
    bm.faces.new(list(reversed(bottom))); bm.faces.new(top)
    n = len(poly2d)
    for i in range(n):
        bm.faces.new((bottom[i], bottom[(i + 1) % n], top[(i + 1) % n], top[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    return mu.new_mesh_object(name, bm, col)


def heightfield_slab(name, X, Y, H, mask, depth, frame, col):
    """Closed slab whose top is the height field H over the masked grid, thickness `depth` along -up."""
    up, right, fwd = frame
    ny, nx = H.shape
    bm = bmesh.new()
    verts = {}
    for j in range(ny):
        for i in range(nx):
            if mask[j, i]:
                verts[(j, i)] = bm.verts.new(right * X[j, i] + fwd * Y[j, i] + up * H[j, i])
    for j in range(ny - 1):
        for i in range(nx - 1):
            q = [(j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i)]
            if all(k in verts for k in q):
                bm.faces.new([verts[k] for k in q])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    # make the top face normals point +up
    if bm.faces and sum(f.normal.dot(up) for f in bm.faces) < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    res = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
    new_verts = [g for g in res["geom"] if isinstance(g, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, verts=new_verts, vec=-up * depth)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    return mu.new_mesh_object(name, bm, col)


def capsule(name, centre, axis, radius, length, col, segments=24):
    """Rounded key: cylinder with hemispherical ends."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=segments // 2, radius=radius)
    half = max(length * 0.5 - radius, 0.0)
    for v in bm.verts:
        v.co.z += half if v.co.z > 0 else -half
    bmesh.ops.triangulate(bm, faces=bm.faces)
    o = mu.new_mesh_object(name, bm, col)
    o.matrix_world = Matrix.Translation(centre) @ axis.to_track_quat('Z', 'Y').to_matrix().to_4x4()
    return o


def tube_along(name, points, radius, col, segments=24, closed=False, normal=None):
    """Swept tube along a 3D polyline (capped), or a closed loop (torus-like) â€” for spillways and the ring.
    Frames are propagated along the path so the tube never twists into itself."""
    bm = bmesh.new()
    rings = []
    n = len(points)
    prev_q = None
    for k, pnt in enumerate(points):
        if closed:
            t = (points[(k + 1) % n] - points[(k - 1) % n]).normalized()
        else:
            t = (points[min(k + 1, n - 1)] - points[max(k - 1, 0)]).normalized()
        if normal is None:
            q = t.to_track_quat('Z', 'Y')
            axes = (q @ Vector((1,0,0)), q @ Vector((0,1,0)))
        else:
            vertical = (normal-t*normal.dot(t)).normalized()
            axes = (vertical.cross(t).normalized(),vertical)
        ring = [bm.verts.new(pnt + radius*(axes[0]*math.cos(2*math.pi*s/segments)+axes[1]*math.sin(2*math.pi*s/segments))) for s in range(segments)]
        rings.append(ring)
    pairs = list(zip(rings[:-1], rings[1:])) + ([(rings[-1], rings[0])] if closed else [])
    for a, b in pairs:
        for s in range(segments):
            bm.faces.new((a[s], a[(s + 1) % segments], b[(s + 1) % segments], b[s]))
    if not closed:
        bm.faces.new(list(reversed(rings[0]))); bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    return mu.new_mesh_object(name, bm, col)


def clean_sheet_bmesh(bm):
    """Make an open sheet manifold: merge doubles, drop degenerate faces, remove faces on edges shared by
    more than two faces (fins), split bow-tie vertices, and drop loose bits. A sheet like this thickens into
    a closed solid; one that is not gets 'resealed' by a coarse voxel remesh downstream."""
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-4, edges=bm.edges)
    fins = [f for f in bm.faces if any(len(e.link_faces) > 2 for e in f.edges)]
    if fins:
        bmesh.ops.delete(bm, geom=fins, context='FACES')
    # bow-tie vertices (two fans of faces meeting at one vertex, or a boundary vertex with more than two
    # boundary edges) stop Solidify from closing the rim. Delete the faces around them â€” a few triangles at
    # the edge of a patch are worth far less than a watertight solid.
    for _ in range(4):
        bad = []
        for v in bm.verts:
            nb = sum(1 for e in v.link_edges if e.is_boundary)
            if (v.is_boundary and nb > 2) or (not v.is_boundary and not v.is_manifold):
                bad.extend(v.link_faces)
        bad = list({f.index: f for f in bad}.values()) if bad else []
        if not bad:
            break
        bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=list(set(bad)), context='FACES')
        loose_v = [v for v in bm.verts if not v.link_faces]
        if loose_v:
            bmesh.ops.delete(bm, geom=loose_v, context='VERTS')
    loose_v = [v for v in bm.verts if not v.link_faces]
    if loose_v:
        bmesh.ops.delete(bm, geom=loose_v, context='VERTS')
    # keep the largest island only (stray specks become separate open shells)
    bm.faces.ensure_lookup_table()
    seen = set(); islands = []
    for f in bm.faces:
        if f in seen:
            continue
        comp = []; stack = [f]; seen.add(f)
        while stack:
            g = stack.pop(); comp.append(g)
            for e in g.edges:
                for nf in e.link_faces:
                    if nf not in seen:
                        seen.add(nf); stack.append(nf)
        islands.append(comp)
    if len(islands) > 1:
        islands.sort(key=len, reverse=True)
        bmesh.ops.delete(bm, geom=[f for c in islands[1:] for f in c], context='FACES')
    return bm


def orient_toward(obj, up):
    """Make every face normal of an open sheet consistent and facing `up` (toward the viewer)."""
    bm = bmesh.new(); bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    if bm.faces and sum((f.normal for f in bm.faces), Vector()).dot(up) < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bm.to_mesh(obj.data); bm.free(); obj.data.update()


def thicken_into(sheet, up, thickness, name, col, shell=6.0):
    """Closed tissue solid from an open sheet whose normals face `up` (the viewer), as ONE closed surface
    with no overlapping pieces (a voxel remesh decides inside/outside by parity, so overlapping shells turn
    into voids): the sheet is extruded `shell` mm along its own vertex normals (keeps undercuts, too thin to
    fold in a socket), then that inner surface is extruded again straight down the opening axis by
    `thickness`. Two extrusions of a region keep the mesh closed by construction."""
    bm = bmesh.new(); bm.from_mesh(sheet.data); bm.transform(sheet.matrix_world)
    bm.normal_update()
    normals = {v: v.normal.copy() for v in bm.verts}
    # 1. shell: extrude, move the copies inward along the ORIGINAL vertex normals
    res = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
    new_verts = [g for g in res["geom"] if isinstance(g, bmesh.types.BMVert)]
    new_faces = [g for g in res["geom"] if isinstance(g, bmesh.types.BMFace)]
    # map each new vertex to its source normal via the extrusion's vert_map
    vmap = res.get("vert_map", None)
    if vmap:
        for v_new, v_old in vmap.items():
            v_new.co -= normals[v_old] * shell
    else:
        # fall back: nearest original vertex (the extrusion copies sit exactly on their originals before moving)
        originals = [v for v in bm.verts if v in normals]
        kd = None
        try:
            from mathutils.kdtree import KDTree
            kd = KDTree(len(originals)); [kd.insert(v.co, i) for i, v in enumerate(originals)]; kd.balance()
        except Exception:
            pass
        for v in new_verts:
            if kd is not None:
                co, idx, dist = kd.find(v.co)
                v.co -= normals[originals[idx]] * shell
    # 2. filler: extrude the inner surface straight down the opening axis
    res2 = bmesh.ops.extrude_face_region(bm, geom=new_faces)
    nv2 = [g for g in res2["geom"] if isinstance(g, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, verts=nv2, vec=-up * thickness)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    lo, hi = mu.world_bbox(sheet)
    pad = thickness + shell + 5.0
    lo_c = Vector(lo) - Vector((pad,) * 3); hi_c = Vector(hi) + Vector((pad,) * 3)
    bad = [v for v in bm.verts if not all(lo_c[i] <= v.co[i] <= hi_c[i] for i in range(3))]
    if bad:
        bmesh.ops.delete(bm, geom=bad, context='VERTS')
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    solid = bpy.data.objects.new(name, me); col.objects.link(solid)
    return solid


def boundary_loops(bm):
    """Group boundary edges into closed loops (lists of BMEdge)."""
    edges = [e for e in bm.edges if e.is_boundary]
    seen = set(); loops = []
    for e in edges:
        if e in seen:
            continue
        loop = []; stack = [e]; seen.add(e)
        while stack:
            cur = stack.pop(); loop.append(cur)
            for v in cur.verts:
                for ne in v.link_edges:
                    if ne.is_boundary and ne not in seen:
                        seen.add(ne); stack.append(ne)
        loops.append(loop)
    return loops


def sculpt_solid_toward(name, sculpt, up, thickness, roi, col, margin=8.0, max_faces=250000, preserve_hole=None):
    """Watertight cutter for the cap: the sculpt's outer surface EXACTLY as it is (no offset of any kind),
    closed by sweeping a copy straight down the opening axis by `thickness` and joining the boundary.
    Nothing here can fire a vertex off to infinity â€” the old Solidify-based cutter did exactly that at sharp
    folds, and one such spike inflated the bounding box until the resealing remesh chose a 612 mm voxel.
    Any vertex that still lands outside the block region is clipped away as a last line of defence."""
    bm = mu.scan_region_bmesh(sculpt, roi, margin, False)
    clean_sheet_bmesh(bm)
    if not bm.faces:
        me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
        obj = bpy.data.objects.new(name, me); col.objects.link(obj)
        return obj
    avg = sum((f.normal for f in bm.faces), Vector()) / len(bm.faces)
    if avg.dot(up) < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    # Internal holes in the sculpt (scan dropouts at a canthus, a nostril) would become vertical shafts through
    # the cutter, and each shaft leaves a column of cap standing inside the cavity. Fill every boundary loop
    # except the longest one â€” the outer margin.
    loops = boundary_loops(bm)
    if preserve_hole is not None and len(loops)<2:
        bm.free();raise RuntimeError('The Sculpt has no inner ocular opening; update the ocular cut first')
    if len(loops) > 1:
        loops.sort(key=lambda L: sum(e.calc_length() for e in L), reverse=True)
        keep=None
        if preserve_hole is not None:
            def center(loop):
                verts=set(v for edge in loop for v in edge.verts)
                return sum((v.co for v in verts),Vector())/len(verts)
            keep=min(loops[1:],key=lambda loop:(center(loop)-preserve_hole).length)
            if (center(keep)-preserve_hole).length>2:
                bm.free();raise RuntimeError('The ocular opening could not be located in the Sculpt cutter')
        inner = [e for L in loops[1:] if L is not keep for e in L]
        try:
            bmesh.ops.holes_fill(bm, edges=inner, sides=0)
        except Exception:
            pass
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        if sum((f.normal for f in bm.faces), Vector()).dot(up) < 0:
            bmesh.ops.reverse_faces(bm, faces=bm.faces)
    if len(bm.faces) > max_faces:
        me0 = bpy.data.meshes.new(name + "_dec"); bm.to_mesh(me0); bm.free()
        tmp = bpy.data.objects.new(name + "_dec", me0); col.objects.link(tmp)
        ratio = max_faces / len(me0.polygons)
        def dec(m):
            m.decimate_type = 'COLLAPSE'; m.ratio = ratio; m.use_collapse_triangulate = True
        mu.apply_modifier(tmp, 'DECIMATE', dec)
        bm = bmesh.new(); bm.from_mesh(tmp.data)
        mu.delete_object(tmp)
    res = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
    new_verts = [g for g in res["geom"] if isinstance(g, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, verts=new_verts, vec=-up * thickness)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    lo, hi = roi
    pad = margin + thickness + 2.0
    lo_c = Vector(lo) - Vector((pad,) * 3); hi_c = Vector(hi) + Vector((pad,) * 3)
    bad = [v for v in bm.verts if not all(lo_c[i] <= v.co[i] <= hi_c[i] for i in range(3))]
    if bad:
        bmesh.ops.delete(bm, geom=bad, context='VERTS')
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    obj = bpy.data.objects.new(name, me)
    col.objects.link(obj)
    if not mu.is_closed(obj):
        bm2 = bmesh.new(); bm2.from_mesh(obj.data)
        bmesh.ops.remove_doubles(bm2, verts=bm2.verts, dist=1e-3)
        try:
            bmesh.ops.holes_fill(bm2, edges=[e for e in bm2.edges if e.is_boundary], sides=0)
        except Exception:
            pass
        bmesh.ops.recalc_face_normals(bm2, faces=bm2.faces)
        bm2.to_mesh(obj.data); bm2.free(); obj.data.update()
    return obj


def sane(obj, fallback_voxel=0.3, budget=10_000_000):
    """Guarantee an operand is closed and manifold before it meets the solver. The Manifold solver throws a
    C++ exception on any non-manifold operand and takes Blender with it. Tries geometric repair first, and
    only voxel-remeshes the operand if it still is not closed. Returns True if a remesh was needed."""
    n, nonman, open_e, loose = mu.mesh_stats(obj)
    if not open_e and not nonman:
        return False
    bm = bmesh.new(); bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-4, edges=bm.edges)
    try:
        bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)
    except Exception:
        pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data); bm.free(); obj.data.update()
    n, nonman, open_e, loose = mu.mesh_stats(obj)
    if not open_e and not nonman:
        return False
    used = mu.safe_voxel(obj, fallback_voxel, budget)
    if used > 2.0:
        raise RuntimeError(f"{obj.name} spans an absurd volume (a reseal would need a {used:.0f} mm voxel) â€” its geometry has exploded; not continuing")
    mu.voxel_remesh(obj, fallback_voxel, budget=budget)
    return True


def cut(target, cutter, op, voxel=0.3, solver='AUTO'):
    """Boolean that cannot crash Blender: both operands are made closed and manifold first (Manifold throws a
    C++ exception â€” fatal â€” on anything else), then the Manifold solver is used where available because it
    gives clean closed output; the result is checked and repaired if a boolean left it open."""
    sane(target, voxel)
    sane(cutter, voxel)
    # Exact avoids sub-micron crossing slivers at tangent land/key/ring joins.
    # Retain the established manifold path for very large scan operands.
    mu.apply_boolean(target, cutter, op, solver=solver)
    n, nonman, open_e, loose = mu.mesh_stats(target)
    if open_e or nonman:
        sane(target, voxel)


def negative_present(context, cap, sculpt, up, tol=0.35):
    """True if the cap's cavity coincides with the sculpt's outer surface (the negative really got cut)."""
    from mathutils.bvhtree import BVHTree
    tree = BVHTree.FromObject(cap, context.evaluated_depsgraph_get())
    me = sculpt.data
    n = len(me.vertices)
    step = max(1, n // 400)
    mw = sculpt.matrix_world
    inv = cap.matrix_world.inverted()
    hits = 0; total = 0
    for i in range(0, n, step):
        w = mw @ me.vertices[i].co
        loc, nrm, idx, dist = tree.find_nearest(inv @ w, 5.0)
        if loc is None:
            continue
        total += 1
        if dist <= tol:
            hits += 1
    return total > 0 and hits / total >= 0.6


def ridge_along(name, points, width, height, col, closed=True):
    """Wedge ridge swept along a 3D polyline: flanks at 45Â° (height = width/2), apex rounded, sunk 0.6 mm
    below the path so the union has no sliver. Profile is symmetric about the path."""
    bm = bmesh.new()
    n = len(points)
    hw = width * 0.5
    h = hw                                                       # 45Â° flanks
    r = min(0.5, h * 0.3)                                        # rounded apex
    prof = [(-hw, -0.6), (-hw, 0.0), (-r, h - r), (0.0, h), (r, h - r), (hw, 0.0), (hw, -0.6)]
    rings = []
    for k, pnt in enumerate(points):
        t = (points[(k + 1) % n] - points[(k - 1) % n]).normalized() if closed else (points[min(k + 1, n - 1)] - points[max(k - 1, 0)]).normalized()
        upv = Vector(ridge_along.up)
        side = t.cross(upv).normalized()
        rings.append([bm.verts.new(pnt + side * x + upv * y) for x, y in prof])
    pairs = list(zip(rings[:-1], rings[1:])) + ([(rings[-1], rings[0])] if closed else [])
    m = len(prof)
    for a, b in pairs:
        for i in range(m):
            bm.faces.new((a[i], a[(i + 1) % m], b[(i + 1) % m], b[i]))
    if not closed:
        bm.faces.new(list(reversed(rings[0]))); bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    return mu.new_mesh_object(name, bm, col)


ridge_along.up = (0.0, 0.0, 1.0)


def cylinder(name, centre, axis, radius, length, col, segments=32):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=radius, radius2=radius, depth=length)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    o = mu.new_mesh_object(name, bm, col)
    o.matrix_world = Matrix.Translation(centre) @ axis.to_track_quat('Z', 'Y').to_matrix().to_4x4()
    return o


def smooth_land(part, frame, hull, skin, land, surf_h_fn, iterations=1, factor=0.25, band=2.0):
    """Mild smoothing of the vertices that lie on the land (between the skin band and the wall, within
    `band` mm of the parting surface) â€” softens boolean seams, key and groove edges on both parts."""
    up, right, fwd = frame
    R, F = np.array(right), np.array(fwd)
    bm = bmesh.new(); bm.from_mesh(part.data); bm.transform(part.matrix_world)
    W = np.array([v.co[:] for v in bm.verts])
    if not len(W):
        bm.free(); return 0
    P2 = np.column_stack([W @ R, W @ F])
    d = dist_to_polygon(P2, hull)
    sel = []
    for v, dd, (px, py), w in zip(bm.verts, d, P2, W):
        if skin - 0.5 <= dd <= skin + land + 0.5:
            hz = Vector(w).dot(up)
            if abs(hz - surf_h_fn(px, py)) <= band:
                sel.append(v)
    if sel:
        for _ in range(iterations):
            bmesh.ops.smooth_vert(bm, verts=sel, factor=factor, use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.transform(part.matrix_world.inverted())
    bm.to_mesh(part.data); bm.free(); part.data.update()
    return len(sel)


def _extract_inner_sheet(part, up, outline, frame, z_lo, z_hi, is_cap, *, backing_trim=0.):
    """Extract the inner sheet BEFORE remeshing rounds the block's trimming planes.

    The block's outer faces are identified by their planes, not by face normals.
    Normal-based selection on an already remeshed block leaves bevel fragments
    attached to the sheet; Solidify then grows those fragments into perimeter fins.
    """
    right, fwd = frame[1], frame[2]
    bm = bmesh.new(); bm.from_mesh(part.data); bm.transform(part.matrix_world)
    poly = np.asarray(outline)
    R, F = np.array(right), np.array(fwd)
    # Boolean coordinates are single precision; this tolerance is geometric, not
    # tied to the voxel size. Do not shave a voxel-wide strip off the fitting land.
    tol = 1e-3
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    W = np.array([v.co[:] for v in bm.verts])
    P = np.column_stack([W @ R, W @ F])
    heights = W @ np.array(up)
    outer_z = z_hi if is_cap else z_lo
    on_end = np.abs(heights - outer_z) <= tol
    # Store plane membership per vertex, so corners shared by two planes are
    # handled without accidentally deleting a triangle on the inner land.
    planes = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        e = b - a
        length = np.linalg.norm(e)
        if length > tol:
            planes.append(np.abs((P[:, 0]-a[0])*e[1] - (P[:, 1]-a[1])*e[0]) <= tol*length)
    plane_mask = np.array(planes)
    membership = [set() for _ in range(len(W))]
    for k, members in enumerate(plane_mask):
        for i in np.flatnonzero(members):
            membership[i].add(k)
    trim_membership = [set() for _ in W] if backing_trim else None
    trim_outward = []
    if backing_trim:
        bm.normal_update()
        poly_center = poly.mean(axis=0)
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            edge=b-a; length=np.linalg.norm(edge)
            if length <= tol: continue
            outward=np.array((edge[1],-edge[0]))/length
            if outward@((a+b)*.5-poly_center)<0:outward=-outward
            k=len(trim_outward);trim_outward.append(right*float(outward[0])+fwd*float(outward[1]))
            for i in np.flatnonzero(np.abs((P-a)@outward)<=backing_trim):trim_membership[i].add(k)
    doomed = []
    for f in bm.faces:
        ids = [v.index for v in f.verts]
        if all(on_end[i] for i in ids) or set.intersection(*(membership[i] for i in ids)):
            doomed.append(f)
        elif backing_trim:
            # Boolean repair can round the original block planes. Exact plane
            # tests then perforate those walls instead of removing them. Restrict
            # this broader test to the *backing reference*: outward-facing wall
            # and back-plane bevels, never the actual detailed mold surface.
            normal = f.normal
            back_sign = 1. if is_cap else -1.
            remove = all(abs(heights[i]-outer_z)<=backing_trim for i in ids) and normal.dot(up)*back_sign > .05
            if not remove:
                remove=any(normal.dot(trim_outward[k])>.5 for k in set.intersection(*(trim_membership[i] for i in ids)))
            if remove: doomed.append(f)
    if not doomed:
        bm.free()
        raise RuntimeError("Cannot locate the shell trimming planes; rebuild the mold from its source surface")
    if doomed:
        bmesh.ops.delete(bm, geom=doomed, context='FACES')
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context='VERTS')
    # keep the one connected inner surface; sliver debris from the booleans would otherwise become hundreds
    # of tiny shells
    seen = set(); islands = []
    for f in bm.faces:
        if f in seen:
            continue
        comp = []; stack = [f]; seen.add(f)
        while stack:
            g = stack.pop(); comp.append(g)
            for e in g.edges:
                for nf in e.link_faces:
                    if nf not in seen:
                        seen.add(nf); stack.append(nf)
        islands.append(comp)
    if len(islands) > 1:
        islands.sort(key=len, reverse=True)
        bmesh.ops.delete(bm, geom=[f for c in islands[1:] for f in c], context='FACES')
    # Keep the Boolean's existing connectivity. Very narrow, valid triangles
    # occur where the scan meets the land. Welding nearby vertices or dissolving
    # those triangles can join opposite sides and punch internal holes in an
    # otherwise continuous sheet. The shell builder validates the perimeter;
    # its distance field handles sub-grid slivers without changing topology here.
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context='VERTS')
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    # normals must face the cavity side: the base's inner surface looks toward the cap (+up), the cap's
    # toward the base (-up)
    want = 1.0 if not is_cap else -1.0
    if bm.faces and sum(f.normal.dot(up) * f.calc_area() for f in bm.faces) * want < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    bm.transform(part.matrix_world.inverted())
    bm.to_mesh(part.data); bm.free(); part.data.update()


def extract_shell(part, up, wall, outline, frame, z_lo, z_hi, is_cap, voxel=0.15, *, reference=None):
    from .mold_shell import build_shell
    return build_shell(part, up, wall, outline, frame, z_lo, z_hi, is_cap, voxel, reference=reference)


def schedule_angles(counts):
    """Give each feature family its own angles around the land. Every family is spread EVENLY (two pry slots
    are always opposite each other); each later family is rotated as a whole to the offset that keeps it
    farthest from everything already placed, so no key sits on a spillway or a pry slot."""
    placed = []
    out = {}
    for name, n in counts:
        if n <= 0:
            out[name] = []; continue
        base = np.array([2 * math.pi * k / n for k in range(n)])
        if not placed:
            mine = base
        else:
            taken = np.array(placed)
            best, best_d = 0.0, -1.0
            for off in np.linspace(0, 2 * math.pi / n, 90, endpoint=False):
                cand = base + off
                d = np.abs(((cand[:, None] - taken[None, :]) + math.pi) % (2 * math.pi) - math.pi).min()
                if d > best_d:
                    best, best_d = off, d
            mine = base + best
        out[name] = [float(x % (2 * math.pi)) for x in mine]
        placed += out[name]
    return out


def rounded_slot(name, centre, ax_len, ax_wid, ax_up, length, width, thick, radius, col, arc=6):
    """Prism with a rounded-rectangle cross-section: `length` along ax_len, `width` along ax_wid, `thick` along
    ax_up, corner radius `radius`. Built directly, so it is one clean solid."""
    prof = []
    hl, hw = length * 0.5 - radius, width * 0.5 - radius
    for cx_, cy_, a0 in ((hl, hw, 0.0), (-hl, hw, math.pi / 2), (-hl, -hw, math.pi), (hl, -hw, 3 * math.pi / 2)):
        for i in range(arc + 1):
            a = a0 + (math.pi / 2) * i / arc
            prof.append((cx_ + radius * math.cos(a), cy_ + radius * math.sin(a)))
    bm = bmesh.new()
    bot = [bm.verts.new(centre + ax_len * x + ax_wid * y - ax_up * (thick * 0.5)) for x, y in prof]
    top = [bm.verts.new(centre + ax_len * x + ax_wid * y + ax_up * (thick * 0.5)) for x, y in prof]
    n = len(prof)
    bm.faces.new(list(reversed(bot))); bm.faces.new(top)
    for i in range(n):
        bm.faces.new((bot[i], bot[(i + 1) % n], top[(i + 1) % n], top[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    return mu.new_mesh_object(name, bm, col)


def sphere(name, centre, radius, col):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=radius)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    o = mu.new_mesh_object(name, bm, col)
    o.location = centre
    return o


class ANAPLAST_OT_mold_two_part(bpy.types.Operator):
    """Look at the finished sculpt from the direction the mold should open, then click: CAP (negative of the sculpt + skin band + tapered land) and BASE (tissue side) with rounded keys, surface-following spillways and rounded pry notches"""
    bl_idname = "anaplast.mold_two_part"
    bl_label = "Two-part mold from this view"
    bl_options = {'REGISTER', 'UNDO'}

    which: bpy.props.EnumProperty(items=[("BOTH", "Both", ""), ("CAP", "Cap only", ""), ("BASE", "Base only", "")], default="BOTH")


    @classmethod
    def poll(cls, context):
        from .shell import source_surface, current_prototype
        p = context.scene.anaplast
        return (source_surface(p) or current_prototype(p)) is not None and (p.cast_obj or p.face_scan_obj) is not None

    def _step(self, name, info=""):
        import time
        now = time.time()
        dt = now - getattr(self, "_t_last", now)
        self._t_last = now
        self.steps.append({"step": name, "ok": True, "info": info, "s": round(dt, 1)})
        try:
            self._ctx.scene.anaplast.mold_steps = json.dumps(self.steps)
        except Exception:
            pass

    def _fail(self, name, err):
        self.steps.append({"step": name, "ok": False, "info": str(err)[:200], "s": 0})
        try:
            self._ctx.scene.anaplast.mold_steps = json.dumps(self.steps)
        except Exception:
            pass

    def execute(self, context):
        import time
        self.steps = []; self._ctx = context; self._t_last = time.time()
        context.scene.anaplast.mold_steps = "[]"
        try:
            return self._run(context)
        except Exception as e:
            import traceback
            last = self.steps[-1]["step"] if self.steps else "start"
            self._fail("after: " + last, e)
            rep(self, {'ERROR'}, f"Mold build failed after '{last}': {e}")
            traceback.print_exc()
            return {'CANCELLED'}
        finally:
            for o in list(bpy.data.objects):
                if o.name.startswith("Mold_") and o.name.endswith(("_tmp", "_tmp2")) or ("_tmp" in o.name and o.name.startswith("Mold_")):
                    mu.delete_object(o)

    def _run(self, context):
        if context.scene.get('anaplast_insert_state'):
            raise RuntimeError('Restore the original mold in Interchangeable inserts before rebuilding mold parts')
        from . import mold_volume
        volume_ledger = mold_volume.Ledger()
        volume_role = None
        def cut(target,cutter,operation):
            solver='EXACT' if len(target.data.polygons)+len(cutter.data.polygons)<700000 else 'AUTO'
            measured = volume_role if target.name in ('Mold_Base','Mold_Cap') and operation=='DIFFERENCE' else None
            before = volume_ledger.before(target) if measured else None
            result = globals()['cut'](target,cutter,operation,solver=solver)
            if measured:volume_ledger.removed(target,before,measured)
            return result
        from .shell import source_surface, current_prototype, resolve_trim
        p = context.scene.anaplast
        from .mold_design import flat_land, interpolate_grid, OutlineSampler, smooth_scan_copy, key_fields, spillway_start, has_self_intersections
        breakable = p.m_base_type == 'SHELL' and p.m_breakable_cap
        if p.m_key_type=='CAPSULE':
            ring_distance=p.m_skin+p.m_spillway_width*.5+1.5
            available=(p.m_skin+p.m_land-ring_distance)*.5-(p.m_spillway_width*.5 if p.m_ring and p.m_spillways else 0.)-.3
            root=p.m_key_diameter*.5+p.m_key_diameter*.65*math.tan(p.m_key_flare)+p.m_key_blend
            if root>available:raise RuntimeError('Increase land width or reduce key diameter, flare or root blend: the flared key would reach the ring or rim')
        if breakable and p.m_break_pattern != 'NONE' and p.m_break_remaining >= p.m_cap_wall:
            raise RuntimeError("Wall at groove must be smaller than cap wall")
        if p.m_key_type == 'WEDGE':
            key_start = p.m_skin + (p.m_spillway_width + 1.5 if p.m_ring and p.m_spillways else 0.) + p.m_wedge_ring_gap
            if p.m_wedge_inner > p.m_key_diameter:
                raise RuntimeError("Inner key diameter must not exceed outer diameter")
            if p.m_skin+p.m_land-key_start <= p.m_wedge_inner*.5+1.:
                raise RuntimeError("Not enough land outside the ring for these keys; increase Land or reduce inner diameter / ring clearance")
        if p.m_pry and not breakable and not (p.m_ring and p.m_spillways) and p.m_pry_depth >= p.m_land:
            raise RuntimeError("Pry depth must be smaller than the land width")
        sculpt = source_surface(p) or current_prototype(p)
        ocular_impression=None
        if self.which!='BASE' and p.m_ocular_impression:
            from .ocular_mold import Impression
            ocular_impression=Impression(context,sculpt)
        tissue_src = resolve_trim(self, p)
        if tissue_src is None:
            return {'CANCELLED'}
        scan_reference_name = tissue_src.name
        mold_volume.clear(context.scene)
        col = mu.get_collection(context.scene, mu.MOLD_COLLECTION)
        holes_filled = 0                                           # the sculpt itself is never modified; only the
                                                                   # cutter (a copy) gets its holes closed
        previous_wedge=bpy.data.objects.get("Mold_Wedge")
        if previous_wedge:mu.delete_object(previous_wedge)
        p.wedge_report=""
        keep_other = None
        for nm in ("Mold_Base", "Mold_Cap"):
            old_o = bpy.data.objects.get(nm)
            if old_o is None:
                continue
            if (self.which == 'CAP' and nm == "Mold_Base") or (self.which == 'BASE' and nm == "Mold_Cap"):
                old_o.name = nm + "_keep"                                      # rebuilding one part: the other stays
                keep_other = old_o
                continue
            mu.delete_object(old_o)
        up, right, fwd = view_axes(context)
        frame_note = ""
        if self.which == 'CAP':
            base_o = keep_other
            if base_o is not None and "mold_up" in base_o:
                up = Vector(base_o["mold_up"]).normalized(); right = Vector(base_o["mold_right"]).normalized()
                right = (right - up * right.dot(up)).normalized(); fwd = up.cross(right).normalized()
                frame_note = "cap built along the existing base's opening direction"
        frame = (up, right, fwd)
        if context.scene.get('anaplast_last_mold_attempt'):
            attempt = json.loads(context.scene['anaplast_last_mold_attempt'])
            attempt['opening_frame'] = [list(axis) for axis in frame]
            context.scene['anaplast_last_mold_attempt'] = json.dumps(attempt)
        U, R, F = np.array(up), np.array(right), np.array(fwd)
        self._step("frame", f"opening axis {tuple(round(c, 2) for c in up)}" + (" â€” " + frame_note if frame_note else ""))
        # parameters live in the panel
        self.depth_limit = p.m_depth_limit
        self.skin, self.land, self.base_depth, self.cap_height = p.m_skin, p.m_land, p.m_base_depth, p.m_cap_height
        self.key_count, self.key_diameter, self.spillways, self.spillway_width = p.m_key_count, p.m_key_diameter, p.m_spillways, p.m_spillway_width
        self.ring, self.pry_points, self.grid, self.voxel = p.m_ring, p.m_pry, p.m_grid, p.m_voxel
        self.key_type, self.bolts, self.bolt_diameter = p.m_key_type, p.m_bolts, p.m_bolt_diameter
        if self.key_type!='NONE':self.grid=min(self.grid,.25)
        self.base_type, self.shell_wall = p.m_base_type, p.m_shell_wall
        if breakable:
            self.spillways=0;self.ring=False;self.pry_points=0;self.bolts=0

        # is the user actually looking at the sculpt, or at its edge? An edge-on view makes the footprint a
        # sliver and the tissue heights meaningless.
        try:
            facing = abs(mu.average_normal(sculpt).dot(up))
        except Exception:
            facing = 1.0
        edge_on = facing < 0.35
        # ---- sculpt footprint and heights in the view frame ----
        me = sculpt.data
        n = len(me.vertices)
        co = np.empty(n * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
        M = np.array(sculpt.matrix_world); W = co @ M[:3, :3].T + M[:3, 3]
        h = W @ U; x = W @ R; y = W @ F
        step = max(1, n // 6000)
        hull = convex_hull_2d(list(zip(x[::step], y[::step])))
        outline = offset_polygon(hull, self.skin + self.land)
        self._step("footprint", f"sculpt {x.max() - x.min():.0f} Ã— {y.max() - y.min():.0f} mm, {len(hull)} hull points")
        top_h = float(h.max())
        z_hi = top_h + self.cap_height

        # ---- tissue height field over the footprint (rays along the opening direction) ----
        xs = np.array([q[0] for q in outline]); ys = np.array([q[1] for q in outline])
        gx = np.arange(xs.min() - self.grid, xs.max() + 2 * self.grid, self.grid)
        gy = np.arange(ys.min() - self.grid, ys.max() + 2 * self.grid, self.grid)
        X, Y = np.meshgrid(gx, gy)
        P2 = np.column_stack([X.ravel(), Y.ravel()])
        d_out = dist_to_polygon(P2, outline).reshape(X.shape)
        mask = (d_out <= 1.5 * self.grid)      # one cell past the outline: the block trims the
                                                                  # slab, so base and cap share the block's wall
        # region of the tissue near the block only
        # world-axis box that contains the whole ROTATED block: all eight corners, not two of them â€” two corners
        # of an oblique block miss most of it, and anything outside the box was being sliced off by flat planes
        corners = [right * cx_ + fwd * cy_ + up * cz_
                   for cx_ in (xs.min(), xs.max()) for cy_ in (ys.min(), ys.max()) for cz_ in (top_h - 200.0, z_hi)]
        lo = Vector((min(c[i] for c in corners) for i in range(3)))
        hi = Vector((max(c[i] for c in corners) for i in range(3)))
        # --- the tissue patch: the scan cut down to the sculpt footprint + skin band (+2 mm), holes in the
        #     patch closed. Nothing beyond that ever enters the mold: no far-side hits, no distant tissue. ---
        tissue_src = smooth_scan_copy(tissue_src, frame, hull, self.skin, float(h.min())-self.depth_limit,
                                      p.m_fit_smoothing, p.m_skin_smoothing, col)
        tissue_source_for_solid = tissue_src
        self._step("surface controls", f"fitting smoothing {p.m_fit_smoothing:.0%}, skin band {p.m_skin_smoothing:.0%}; sculpt unsmoothed")
        tbm = mu.scan_region_bmesh(tissue_src, (lo, hi), 8.0, p.scan_flip)
        floor_h = float(h.min()) - self.depth_limit
        def crop_patch(bm_src, keep_r, name):
            """The scan within keep_r of the sculpt footprint, facing the viewer, within the depth limit."""
            bmc = bm_src.copy()
            Wt = np.array([v.co[:] for v in bmc.verts])
            if len(Wt):
                P2t = np.column_stack([Wt @ R, Wt @ F])
                dvt = dist_to_polygon(P2t, hull)
                Hv = Wt @ U
                # crop by FACE (a face goes only when all its vertices are beyond keep_r), so the patch reaches
                # at least keep_r everywhere â€” a vertex crop stopped one face ring short, and after the shell's
                # inward offset the tissue solid ended before the land slab began: a ring with nothing in it
                dv = {v: (d_, hv) for v, d_, hv in zip(bmc.verts, dvt, Hv)}
                far_f = [f for f in bmc.faces if all(dv[v][0] > keep_r for v in f.verts) or any(dv[v][1] < floor_h for v in f.verts)]
                if far_f:
                    bmesh.ops.delete(bmc, geom=far_f, context='FACES')
                loose_v = [v for v in bmc.verts if not v.link_faces]
                if loose_v:
                    bmesh.ops.delete(bmc, geom=loose_v, context='VERTS')
                back = [f for f in bmc.faces if f.normal.dot(up) < -0.1]     # the far side of the head projects
                if back:                                                       # inside the footprint too
                    bmesh.ops.delete(bmc, geom=back, context='FACES')
            clean_sheet_bmesh(bmc)
            me_ = bpy.data.meshes.new(name)
            bmc.to_mesh(me_); bmc.free()
            o_ = bpy.data.objects.new(name, me_)
            col.objects.link(o_)
            mu.fill_internal_holes(o_)                                  # scan dropouts closed â€” on this copy only
            orient_toward(o_, up)
            return o_
        patch_wide = crop_patch(tbm, self.skin + self.land + 2.0, "Mold_TissueWide_tmp")   # footprint + skin + land
        patch = crop_patch(tbm, self.skin + 8.0, "Mold_TissuePatch_tmp")                  # for the tissue solid: well past
                                                                                           # the slab's start; the skin prism trims it
        tbm.free()
        tissue_src = patch
        tree = BVHTree.FromObject(patch_wide, context.evaluated_depsgraph_get())
        self._step("tissue patch", f"wide {len(patch_wide.data.polygons):,} faces, skin-band {len(patch.data.polygons):,} faces, open edges {mu.mesh_stats(patch)[2]}")
        Ht = np.full(X.shape, np.nan)
        for idx in zip(*np.nonzero(mask)):
            origin = right * X[idx] + fwd * Y[idx] + up * z_hi
            loc, nrm, fi, dist = tree.ray_cast(origin, -up, 400.0)
            if loc is not None:
                Ht[idx] = (loc - Vector((0, 0, 0))).dot(up)
        if np.isnan(Ht[mask]).all():
            rep(self, {'ERROR'}, "No tissue under the sculpt from this direction â€” orbit so you look onto the skin")
            return {'CANCELLED'}
        # A ray that misses the ear/nose and flies on to the far side of the head returns a "tissue" height tens
        # of mm too deep; one such hit used to drag the whole block bottom down with it. Anything deeper than
        # `depth_limit` below the sculpt's own lowest point is not the tissue this mold sits on â€” treat it as
        # missing and fill it from its neighbours like any other gap.
        floor = float(h.min()) - self.depth_limit
        far = mask & ~np.isnan(Ht) & (Ht < floor)
        n_far = int(far.sum())
        Ht[far] = np.nan
        # fill gaps (scan holes, deep hits) by smooth inpainting: unknown cells relax toward the average of their
        # neighbours until the surface is continuous â€” no plateaus, no steps at the edge of a hole
        missing = mask & np.isnan(Ht)
        if missing.any():
            known = mask & ~np.isnan(Ht)
            kp = np.argwhere(known); up_ = np.argwhere(missing)
            Hf = Ht.copy()
            # seed with nearest known, then relax
            kxy = np.column_stack([X[tuple(kp.T)], Y[tuple(kp.T)]]); kv = Ht[tuple(kp.T)]
            for u in up_:
                q = np.array([X[tuple(u)], Y[tuple(u)]])
                Hf[tuple(u)] = kv[np.argmin(((kxy - q) ** 2).sum(axis=1))]
            Hf[~mask] = np.nan
            for _ in range(300):
                pad = np.pad(Hf, 1, mode='edge')
                nb = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
                with np.errstate(all='ignore'):
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        avg = np.nanmean(nb, axis=0)
                Hf = np.where(missing & ~np.isnan(avg), avg, Hf)
            Ht = Hf

        # ---- parting surface: skin band at tissue height, then land easing to a plane parallel to the base ----
        d_hull = dist_to_polygon(P2, hull).reshape(X.shape)          # <0 inside the sculpt footprint
        cx, cy = float(np.mean([q[0] for q in hull])), float(np.mean([q[1] for q in hull]))
        transition = min(p.m_land_transition, max(1., self.land - max(1., 2*self.grid)))
        Hp, flat_h = flat_land(gx, gy, Ht, d_hull, mask, offset_polygon(hull,self.skin), self.skin, transition,p.m_land_follow)
        self._step("smooth land", f"skin-height following {p.m_land_follow:.0%}; transition {transition:.2f} mm")
        land_height=Hp.copy()
        ring_d = self.skin + self.spillway_width * .5 + 1.5
        sched = schedule_angles([("keys", self.key_count if self.key_type != 'NONE' else 0), ("spill", self.spillways), ("pry", self.pry_points), ("bolt", self.bolts)])
        # Even a breakable cap without its own channels must match the keys
        # already present on the common base, including bullet-key start points.
        key_male,key_socket=key_fields(X,Y,hull,p,sched,ring_d,p.m_spillways,p.m_ring)
        if self.key_type=='CAPSULE':Hp=Hp+key_male
        elif self.key_type=='WEDGE':Hp=Hp-key_socket
        # Store the counterpart independently so clearance affects only keys.
        cap_height=land_height+(key_socket if self.key_type=='CAPSULE' else -key_male if self.key_type=='WEDGE' else 0.)
        # base thickness is measured from the tissue point CLOSEST to the viewer (the highest point under the
        # footprint), never opening the base under the lowest point
        z_lo = min(float(np.nanmax(Hp[mask])) - self.base_depth, float(np.nanmin(Hp[mask])) - 3.0)

        # ---- tissue solid (real cast surface, undercuts and all) + land slab over the land ring only ----
        # Build a continuous slab first, then trim its inner boundary with the
        # exact band prism below. A cell-selected ring can retreat farther than
        # the tissue overlap at oblique edges, leaving narrow full-depth gaps
        # which become long fins in the complementary cap.
        land_mask = mask.copy()
        # a grid of quads is only manifold if cells never touch corner-to-corner alone: drop lone cells and
        # bridge diagonal-only contacts, otherwise the slab has non-manifold vertices and must be resealed
        for _ in range(3):
            m = land_mask
            up_ = np.zeros_like(m); up_[1:, :] = m[:-1, :]
            dn_ = np.zeros_like(m); dn_[:-1, :] = m[1:, :]
            lf_ = np.zeros_like(m); lf_[:, 1:] = m[:, :-1]
            rt_ = np.zeros_like(m); rt_[:, :-1] = m[:, 1:]
            nbrs = up_.astype(int) + dn_.astype(int) + lf_.astype(int) + rt_.astype(int)
            land_mask = m & (nbrs >= 2)
            # bridge diagonals: (j,i)&(j+1,i+1) set with (j+1,i)&(j,i+1) clear â†’ set (j+1,i)
            a = land_mask[:-1, :-1] & land_mask[1:, 1:] & ~land_mask[1:, :-1] & ~land_mask[:-1, 1:]
            b = land_mask[1:, :-1] & land_mask[:-1, 1:] & ~land_mask[:-1, :-1] & ~land_mask[1:, 1:]
            fill = np.zeros_like(land_mask)
            fill[1:, :-1] |= a; fill[:-1, :-1] |= b
            land_mask = land_mask | fill
        slab = heightfield_slab("Mold_Slab_tmp", X, Y, Hp, land_mask, (z_hi - z_lo) + 20.0, frame, col)
        block = prism("Mold_Block_tmp", outline, z_lo, z_hi, frame, col)
        # thicken the patch INTO the head along its own (now viewer-facing) normals â€” undercuts preserved,
        # direction known, no orientation heuristic involved
        # thick enough to reach below the block bottom from the highest skin point (and always past the socket)
        t_need = float(np.nanmax(Ht[mask])) - z_lo + 8.0
        tissue_swept = False
        source_stats = mu.mesh_stats(tissue_source_for_solid)
        if source_stats[0] and not any(source_stats[1:]):
            tissue = mu.duplicate_object(tissue_source_for_solid, "Mold_Tissue_tmp", col)
            remesh_log = []
            self._step("tissue solid", f"original closed scan retained: {len(tissue.data.polygons):,} faces; no resampling")
        else:
            # A 6 mm normal offset can fold back through narrow eye/nasal
            # recesses before the axial backing is added, producing false fins.
            # Keep a shorter inward step on this temporary open-scan solid;
            # the axial extrusion still supplies its full required depth.
            tissue = thicken_into(patch, up, max(t_need, self.depth_limit + 8.0), "Mold_Tissue_tmp", col, shell=1.5)
            v_tissue = mu.voxel_remesh(tissue, self.voxel)
            remesh_log = [f"open-scan repair @ {v_tissue:.2f}"]
            self._step("tissue solid", f"open scan required a closed working copy @ {v_tissue:.2f} mm; inward step 1.50 mm")
        if not mu.is_closed(tissue):
            # try to seal it geometrically before anything coarser
            bmt = bmesh.new(); bmt.from_mesh(tissue.data)
            try:
                bmesh.ops.holes_fill(bmt, edges=[e for e in bmt.edges if e.is_boundary], sides=0)
            except Exception:
                pass
            bmesh.ops.recalc_face_normals(bmt, faces=bmt.faces)
            bmt.to_mesh(tissue.data); bmt.free(); tissue.data.update()
        tmp_t = True
        remeshed = []
        if sane(slab, 0.3, budget=10_000_000):
            remeshed.append("land slab")
        # The gridded join must not overlay the original skin band. Limit it to
        # the land, then overlap the tissue only just outside the band boundary.
        band_guard = prism("Mold_BandGuard_tmp", offset_polygon(hull,self.skin),
                           z_lo-(z_hi-z_lo)-100., z_hi+5., frame, col)
        cut(slab, band_guard, 'DIFFERENCE')
        mu.delete_object(band_guard)
        self._step("height field + land slab", f"{int(mask.sum()):,} cells, deep hits ignored {int(n_far)}, slab {len(slab.data.polygons):,} faces closed {mu.is_closed(slab)}")
        # the scan is kept only under the sculpt and its skin band; it is trimmed there and the smooth land
        # takes over â€” bumps of tissue no longer show through the land
        skin_outline = offset_polygon(hull, self.skin + 0.05)     # slight overlap into the land
        skin_prism = prism("Mold_SkinPrism_tmp", skin_outline, z_lo - 5.0, z_hi + 5.0, frame, col)
        solid_side = mu.duplicate_object(tissue, "Mold_TissueLand_tmp", col)
        cut(solid_side, skin_prism, 'INTERSECT')
        mu.delete_object(skin_prism)
        cut(solid_side, slab, 'UNION')
        if sane(solid_side, 0.3):
            remeshed.append("tissue+land")
        self._step("tissue âˆª land", f"{len(solid_side.data.polygons):,} faces, closed {mu.is_closed(solid_side)}")
        # CAP first = block âˆ’ (tissue âˆª land) âˆ’ prosthesis volume
        cap = prism("Mold_Cap", outline, z_lo, z_hi, frame, col)          # same footprint as the base: flush walls
        cap_side=solid_side
        if self.key_type!='NONE' and p.key_clearance>0:
            socket_slab=heightfield_slab("Mold_SocketSlab_tmp",X,Y,cap_height,land_mask,(z_hi-z_lo)+20.,frame,col)
            guard=prism("Mold_SocketGuard_tmp",offset_polygon(hull,self.skin),z_lo-100.,z_hi+100.,frame,col)
            cut(socket_slab,guard,'DIFFERENCE');mu.delete_object(guard)
            guard=prism("Mold_SocketGuard_tmp",skin_outline,z_lo-5.,z_hi+5.,frame,col)
            cap_side=mu.duplicate_object(tissue,"Mold_SocketSide_tmp",col)
            cut(cap_side,guard,'INTERSECT');mu.delete_object(guard)
            cut(cap_side,socket_slab,'UNION');mu.delete_object(socket_slab)
        cut(cap, cap_side, 'DIFFERENCE')
        if cap_side!=solid_side:mu.delete_object(cap_side)
        self._step("cap blank = block âˆ’ (tissue âˆª land)", f"{len(cap.data.polygons):,} faces, closed {mu.is_closed(cap)}")
        # BASE = block âˆ© (tissue âˆª land): the skin keeps its undercuts, the land is the smoothed scan
        base = mu.duplicate_object(block, "Mold_Base", col)
        cut(base, solid_side, 'INTERSECT')
        self._step("base = block âˆ© (tissue âˆª land)", f"{len(base.data.polygons):,} faces, closed {mu.is_closed(base)}")
        cap_before = mu.duplicate_object(cap, "Mold_CapPre_tmp", col)     # to retry from if the cut does not take
        cavity_before_volume = volume_ledger.before(cap) if self.which!='BASE' else None
        cap_blank=mu.duplicate_object(cap,"Mold_CapBlank_tmp",col)
        # the cutter only has to reach the tissue under the sculpt: depth_limit + margin, not the whole block
        # the cutter must clear the whole prosthesis volume down to the tissue â€” the height field under a deep
        # socket is inpainted from the rim and underestimates it, so it cannot be used to shorten the cutter
        if self.which!='BASE':
            cutter_depth = min((z_hi - z_lo) + 10.0, self.depth_limit + 10.0)
            sculpt_solid = ocular_impression.cutter("Mold_Sculpt_tmp",up,cutter_depth,(lo,hi),col) if ocular_impression else sculpt_solid_toward("Mold_Sculpt_tmp", sculpt, up, cutter_depth, (lo, hi), col, margin=8.0, max_faces=max(250000,len(sculpt.data.polygons)*2))
            cutter_stats = mu.mesh_stats(sculpt_solid)
            intersecting = not any(cutter_stats[1:]) and has_self_intersections(sculpt_solid)
            if any(cutter_stats[1:]) or intersecting:
                if ocular_impression:raise RuntimeError("The temporary ocular impression cutter could not be joined cleanly; original Sculpt and ocular surfaces are unchanged")
                v_cut = mu.voxel_remesh(sculpt_solid, self.voxel)
                remesh_log.append(f"sculpt repair @ {v_cut:.2f}")
                cutter_note = f"{'self-intersection' if intersecting else 'topology'} repair @ {v_cut:.2f} mm; source unchanged"
            else:
                cutter_note = "original sculpt surface retained, no resampling"
            if ocular_impression and getattr(ocular_impression,'repaired_sculpt',False):
                cutter_note=f"temporary scan sweep repaired @ {self.voxel:.2f} mm; exact ocular surface inserted afterward"
            self._step("sculpt cutter", f"{len(sculpt_solid.data.polygons):,} faces; {cutter_note}")
            sweep_ok = mu.is_closed(sculpt_solid)
            cut(cap, sculpt_solid, 'DIFFERENCE')
            tmp_s = True
            # Did the cut actually take? A scan with a few non-manifold edges gives a cutter the solver quietly
            # refuses, and the cap then keeps the tissue (the socket) where the sculpt should be. Check the
            # sculpt's surface against the cap; if it is not there, rebuild the cutter as a sealed voxel solid
            # (guaranteed closed) and cut again with the float solver.
            negative_ok = negative_present(context, cap, sculpt, up)
            if not negative_ok:
                if ocular_impression:raise RuntimeError("The combined Sculpt and ocular impression was not confirmed; check fit before rebuilding the cap")
                mu.delete_object(sculpt_solid)
                sculpt_solid = sculpt_solid_toward("Mold_Sculpt_tmp", sculpt, up, cutter_depth, (lo, hi), col, margin=8.0)
                used = mu.voxel_remesh(sculpt_solid, self.voxel)   # cutter only
                remeshed.append(f"sculpt cutter @ {used:.2f} mm")
                mu.delete_object(cap)
                cap = mu.duplicate_object(cap_before, "Mold_Cap", col)
                cut(cap, sculpt_solid, 'DIFFERENCE')
                negative_ok = negative_present(context, cap, sculpt, up)
                sweep_ok = mu.is_closed(sculpt_solid)
            if len(cap.data.polygons) == 0:
                mu.delete_object(cap)
                cap = mu.duplicate_object(cap_before, "Mold_Cap", col)      # keep the uncut cap rather than nothing
                negative_ok = False
            volume_ledger.removed(cap,cavity_before_volume,'cavity')
            if ocular_impression:ocular_impression.exclude_displacement(cap_before,volume_ledger,col)
            if not negative_ok:volume_ledger.data['errors'].append('The sculpt cavity cut was not confirmed')
            mu.delete_object(cap_before)
            self._step("cap cavity = cap âˆ’ cutter", f"negative present {negative_ok}, {len(cap.data.polygons):,} faces")
        else:
            tmp_s=False;negative_ok=True;mu.delete_object(cap_before)
        # (the built prototype is deliberately NOT subtracted as well: its outer surface is the sculpt's, and
        #  cutting the same surface twice puts a thin plane on a thin plane)

        # ---- helpers ----
        def surf_h(px, py):
            return float(interpolate_grid(gx,gy,land_height,px,py,flat_h))
        sampler = OutlineSampler(hull)
        def ring_point(a, d_from_hull):
            return sampler.point(a,d_from_hull)
        # Keep cavity/land references before channel and pry cuts. Extending a
        # sheet whose perimeter is interrupted by pry slots can fold the backing
        # reference and leave tall walls. Registration contours already belong
        # to these surfaces; bolt bores are drilled after the shell is formed.
        self._step("bolts", f"{self.bolts}")
        base_plain = mu.duplicate_object(base, "Mold_BasePlain_tmp", col)
        cap_plain = mu.duplicate_object(cap, "Mold_CapPlain_tmp", col)

        ring_d = self.skin + self.spillway_width * 0.5 + 1.5                 # the ring hugs the skin band
        self._step("plain parts ready", f"base {len(base.data.polygons):,} / cap {len(cap.data.polygons):,} faces")
        # angles for everything on the land, allocated so families never coincide
        sched = schedule_angles([("keys", self.key_count if self.key_type != 'NONE' else 0), ("spill", self.spillways),
                                 ("pry", self.pry_points), ("bolt", self.bolts)])
        # spillway ring around the land (mid-land), following the surface â€” the radial spillways cross it
        volume_role = 'channels'
        if self.spillways > 0 and self.ring:
            ring_pts = []
            for a in np.linspace(0, 2 * math.pi, 96, endpoint=False):
                px, py = ring_point(a, ring_d)
                ring_pts.append(right * px + fwd * py + up * surf_h(px, py))
            g = tube_along("Mold_Ring_tmp", ring_pts, self.spillway_width * 0.5, col, closed=True, normal=up)
            g2 = mu.duplicate_object(g, "Mold_Ring_tmp2", col)
            cut(cap, g, 'DIFFERENCE')
            cut(cap_blank, g, 'DIFFERENCE')
            cut(base, g2, 'DIFFERENCE')
            mu.delete_object(g); mu.delete_object(g2)
        # spillways: tubes following the land surface, from the skin band out past the edge (grooves in the cap)
        self._step("keys", f"{self.key_type.lower()} Ã— {self.key_count if self.key_type != 'NONE' else 0}")
        for k in range(self.spillways):
            a = sched["spill"][k]
            pts = []
            for dd in np.arange(spillway_start(self.skin,self.spillway_width,self.ring,ring_d), self.skin + self.land + 3.0, .5):      # land only
                px, py = ring_point(a, dd)
                pts.append(right * px + fwd * py + up * surf_h(px, py))
            if len(pts) >= 2:
                g = tube_along(f"Mold_Spill_tmp{k}", pts, self.spillway_width * 0.5, col, normal=up)
                g2 = mu.duplicate_object(g, f"Mold_Spill_tmp{k}b", col)
                cut(cap, g, 'DIFFERENCE')      # half the groove in the cap
                cut(cap_blank, g, 'DIFFERENCE')
                cut(base, g2, 'DIFFERENCE')    # half in the base
                mu.delete_object(g); mu.delete_object(g2)
        volume_role = None
        # Pry openings follow the complete land curve and stop before the ring.
        self._step("spillways + ring", f"{self.spillways} spillways, ring {self.ring}")
        from .mold_design import pry_depth_to_ring, contoured_pry_cutter, PRY_RING_GAP_MM
        pry_plan=[]
        for k in range(self.pry_points):
            a=sched["pry"][k]
            rim=np.asarray(ring_point(a,self.skin+self.land))
            tangent=np.asarray(ring_point(a+.001,self.skin+self.land))-np.asarray(ring_point(a-.001,self.skin+self.land))
            tangent/=np.linalg.norm(tangent)
            outward=np.array((tangent[1],-tangent[0]))
            if outward@(rim-sampler.center)<0:outward=-outward
            depth=p.m_pry_depth
            if self.ring and self.spillways:
                depth=pry_depth_to_ring(hull,rim,outward,tangent,p.m_pry_width,
                    ring_d+self.spillway_width*.5,self.skin+self.land)
                if depth<.5:raise RuntimeError("Not enough land for a pry opening with a 1 mm ring gap; increase Land or reduce Pry width")
            bx=contoured_pry_cutter(f"Mold_Pry_tmp{k}",rim,outward,tangent,depth,p.m_pry_width,
                p.m_pry_height,(up,right,fwd),gx,gy,land_height,flat_h,col)
            try:
                # Dense, curved mating slabs can give a false unchanged result
                # with Exact. These closed pry solids use the Manifold solver.
                for part in (cap,cap_blank,base):
                    globals()['cut'](part,bx,'DIFFERENCE',solver='MANIFOLD')
            finally:mu.delete_object(bx)
            pry_plan.append({'depth_mm':float(depth),'ring_gap_mm':PRY_RING_GAP_MM if self.ring and self.spillways else None,
                             'rim':rim.tolist(),'outward':outward.tolist(),'tangent':tangent.tolist()})
        for part in (base,cap,cap_blank):part['mold_pry_plan']=json.dumps(pry_plan)
        # Form shells before drilling so bolt walls cannot become deep tubes
        # or blind plugs in the offset reference.
        self._step("pry slots", f"{self.pry_points}")
        from .mold_bolts import add_shell_seats, drill_bores
        bolt_positions=[list(ring_point(a,(ring_d+self.skin+self.land)*.5)) for a in sched["bolt"]]
        bolt_seats={}
        if bolt_positions:drill_bores(cap_blank,bolt_positions,self.bolt_diameter,frame)
        # Build the shell backing from the unrounded trimming planes. Retain the
        # original cavity and perimeter in the final difference; do not remesh it.
        v_parts = []
        wanted = {'BOTH': (base, cap), 'CAP': (cap,), 'BASE': (base,)}[self.which]
        for o_ in (base, cap):
            if o_ not in wanted:
                v_parts.append(float('nan')); continue
            mu.keep_main_body(o_)
            if self.base_type == 'SHELL':
                if breakable and o_ == cap:
                    from .mold_breakable import build_cap
                    build_cap(o_,up,outline,frame,z_lo,z_hi,p)
                    used=float('nan')
                    self._step("breakable cap",f"{p.m_cap_wall:.3f} mm nominal wall, {p.m_break_pattern.lower()} pattern; groove wall {p.m_break_remaining:.3f} mm; source unchanged, fine-grid approximation")
                else:
                    bolt_reference=mu.duplicate_object(o_,"Mold_BoltInner_tmp",col) if bolt_positions else None
                    try:
                        self._step("shell reference", f"{o_.name}: {self.shell_wall:.2f} mm wall")
                        used = extract_shell(o_, up, self.shell_wall, outline, frame, z_lo, z_hi,
                                             is_cap=(o_ == cap), voxel=self.voxel,
                                             reference=cap_plain if o_ == cap else base_plain)
                        if bolt_reference is not None:
                            bolt_seats[o_.name]=add_shell_seats(o_,bolt_reference,bolt_positions,self.bolt_diameter,
                                p.m_bolt_sleeve_wall,p.m_bolt_seat_raise,frame,o_==cap,p.m_bolt_sleeve_length)
                    finally:
                        if bolt_reference is not None:mu.delete_object(bolt_reference)
            else:
                repaired = sane(o_, self.voxel)
                used = self.voxel if repaired else float('nan')
            v_parts.append(used)
            mu.keep_main_body(o_)
        backing = " backing" if self.base_type == 'SHELL' else ""
        remesh_log.append(", ".join(f"{nm}{backing} @ {v:.2f}" for nm, v in zip(("base", "cap"), v_parts) if v == v))
        self._step("shell backing" if backing else "remesh parts", ", ".join(f"{nm} @ {v:.2f} mm" for nm, v in zip(("base", "cap"), v_parts) if v == v))
        # mild smoothing of the land on both parts: softens every boolean seam, key, groove and slot edge
        if self.which=='BASE':
            previous=bpy.data.objects.get('Common_Base_Reference')
            if previous:mu.delete_object(previous)
            base_plain.name='Common_Base_Reference';base_plain['anaplast_construction']=True
            base_plain.hide_set(True);base_plain.hide_render=True
        else:mu.delete_object(base_plain)
        mu.delete_object(cap_plain)
        self._step("shell" if self.base_type == 'SHELL' else "full parts", f"base {len(base.data.polygons):,} closed {mu.is_closed(base)} / cap {len(cap.data.polygons):,} closed {mu.is_closed(cap)}")
        for o in (block, slab, solid_side):
            mu.delete_object(o)
        if tmp_t:
            mu.delete_object(tissue)
        mu.delete_object(patch); mu.delete_object(patch_wide)
        if tmp_s:
            mu.delete_object(sculpt_solid)
        holes_left = []
        for o in (base, cap):
            # a long boolean chain leaves slivers behind: keep the main body only
            try:
                dropped = mu.keep_main_body(o)
            except Exception:
                dropped = 0
            # never voxel-remesh the result: that would reintroduce axis-aligned stair-stepping on every
            # wall. The booleans are exact, so only patch actual holes left by a boolean, geometrically â€”
            # never resample the whole mesh.
            n0, nonman, open0, loose = mu.mesh_stats(o)
            if open0 or nonman:
                mu.cleanup_mesh(o)
                n0, nonman, open0, loose = mu.mesh_stats(o)
                if open0:
                    holes_left.append((o.name, open0))
            # Last geometric operation: drill the complete final thickness and
            # inspect the bore after all shell, union and cleanup operations.
            if bolt_positions:
                probes=drill_bores(o,bolt_positions,self.bolt_diameter,frame)
                o["mold_bolt_report"]=json.dumps({'positions':bolt_positions,'bore_diameter_mm':self.bolt_diameter,
                    'clearance_ray_count':probes,'sleeves':bolt_seats.get(o.name,[])})
            for pg in o.data.polygons:
                pg.use_smooth = True
            try:
                o.data.set_sharp_from_angle(angle=math.radians(35.0))   # keep the block edges crisp
            except Exception:
                pass
            o["anaplast_part"] = "MOLD"
            o["mold_up"] = list(up); o["mold_right"] = list(right)
            o["mold_hull"] = [list(q) for q in hull]
            o["mold_z_lo"]=z_lo;o["mold_z_hi"]=z_hi
            o["mold_skin"] = self.skin; o["mold_land"] = self.land
            o["mold_land_transition"] = transition; o["mold_plane_h"] = flat_h
            o["mold_land_follow"] = p.m_land_follow
            o["mold_floor_h"] = float(h.min())-self.depth_limit
            o["mold_source_scan"] = scan_reference_name; o["mold_source_sculpt"] = sculpt.name
            o["mold_fit_smoothing"] = p.m_fit_smoothing; o["mold_skin_smoothing"] = p.m_skin_smoothing
        if ocular_impression:ocular_impression.verify(cap,up)
        if self.which=='BASE':
            for key in base.keys():base_plain[key]=base[key]
            base_plain['anaplast_construction']=True;base_plain['auricular_plain_layout']=True
        base.color = (0.93, 0.70, 0.25, 1.0); cap.color = (0.35, 0.55, 0.85, 1.0)
        if self.which in ('BOTH','CAP','BASE'):
            previous=bpy.data.objects.get('Auricular_CapBlank_Reference')
            if previous:mu.delete_object(previous)
            cap_blank.name='Auricular_CapBlank_Reference';cap_blank.hide_set(True);cap_blank.hide_render=True
            cap_blank['source_sculpt']=sculpt.name;cap_blank['mold_up']=list(up)
            from .scene_helpers import organize_object
            organize_object(cap_blank,context.scene)
        else:mu.delete_object(cap_blank)
        if self.which != 'BOTH':
            drop = base if self.which == 'CAP' else cap
            mu.delete_object(drop)
            if keep_other is not None:
                keep_other.name = "Mold_Base" if self.which == 'CAP' else "Mold_Cap"
                if self.which == 'CAP':
                    base = keep_other
                else:
                    cap = keep_other
            else:
                if self.which == 'CAP':
                    base = cap                                                  # nothing to show beside it
                else:
                    cap = base
        if self.which == 'BOTH' or (self.which=='CAP' and base is not cap and base.get('workflow_base_settings')):
            mold_volume.publish(context.scene,volume_ledger,(base,cap))
        else:
            mold_volume.clear(context.scene,'Build both parts together to record matching cavity and channel volumes')
        # show the result: everything else hidden, mold opened like a book, viewed from the opening direction
        for o in list(bpy.data.objects):
            if o is None:
                continue
            if o.type in {'MESH', 'EMPTY'} and o.name not in (base.name, cap.name):
                try:
                    o.hide_set(True)
                except RuntimeError:
                    pass   # object not linked in this view layer
        p.exploded = False
        try:
            bpy.ops.anaplast.explode_view()
        except Exception:
            pass
        if not bpy.app.background:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    rv3d = area.spaces.active.region_3d
                    rv3d.view_rotation = up.to_track_quat('Z', 'Y')          # look straight down the opening axis
                    region = next((r for r in area.regions if r.type == 'WINDOW'), None)
                    for o in list(bpy.data.objects):
                        if o is None:
                            continue
                        try:
                            o.select_set(o.name in (base.name, cap.name))
                        except RuntimeError:
                            pass
                    if region:
                        with context.temp_override(area=area, region=region):
                            bpy.ops.view3d.view_selected()
        fw = (xs.max() - xs.min(), ys.max() - ys.min(), z_hi - z_lo)
        extra = ""
        if tissue_swept:
            extra += "; tissue solid swept (Solidify could not close it â€” undercuts in the skin band not carried)"
        if holes_filled:
            extra += f"; {holes_filled} hole(s) filled in the sculpt first"
        remesh_log = [entry for entry in remesh_log if entry]
        if remesh_log:
            extra += "; remeshed " + ", ".join(remesh_log) + " mm"
        if remeshed:
            extra += "; resealed: " + ", ".join(remeshed)
        if n_far:
            extra += f"; {n_far:,} deep tissue hits ignored"
        if edge_on:
            extra += " â€” WARNING: you are looking at the sculpt nearly edge-on; orbit so you face it and rebuild"
        warn = "" if negative_ok else " â€” WARNING: the sculpt could NOT be cut into the cap even after resealing it; the cap still shows the tissue there. Clean the sculpt (Lasso â†’ Remove stray pieces) and rebuild"
        if holes_left:
            warn += " â€” " + "; ".join(f"{n} has {c} open edges left (a boolean could not fully close it â€” check that area)" for n, c in holes_left)
        rep(self, {'INFO' if (negative_ok and not holes_left and not edge_on) else 'WARNING'}, f"Block {fw[0]:.0f} Ã— {fw[1]:.0f} Ã— {fw[2]:.0f} mm (sculpt {x.max() - x.min():.0f} Ã— {y.max() - y.min():.0f}){extra}{warn}. Base {len(base.data.polygons):,} faces ({'closed' if mu.is_closed(base) else 'OPEN'}), "
                            f"Cap {len(cap.data.polygons):,} ({'closed' if mu.is_closed(cap) else 'OPEN'}); skin {self.skin:.0f} + land {self.land:.0f} mm, "
                            f"{(str(self.key_count) + ' wedge notches' if self.key_type == 'WEDGE' else str(self.key_count) + ' keys') if self.key_type != 'NONE' else 'no keys'}, {self.bolts} bolt holes, {self.spillways} spillways{' + ring' if self.ring and self.spillways else ''}, {self.pry_points} pry points")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ANAPLAST_OT_mold_two_part)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_mold_two_part)
