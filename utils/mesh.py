"""Context-free mesh helpers (work in headless bpy and in the UI)."""
import struct
import bpy
import bmesh
from mathutils import Vector, Matrix

MOLD_COLLECTION = "Anaplast_Mold"


def get_collection(scene, name=MOLD_COLLECTION):
    if name=='Anaplast_Prototype':
        from ..ops.scene_helpers import group,GROUPS
        return group(scene,GROUPS[4])
    if name==MOLD_COLLECTION and hasattr(scene,'mold_workflow'):
        from ..ops import mold_workflow
        return mold_workflow.collection_for(scene,mold_workflow.ensure_active(scene))
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
    if col.name not in scene.collection.children and not any(col==c for c in scene.collection.children_recursive):
        scene.collection.children.link(col)
    return col


def link_only(obj, col):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)


def world_bbox(obj):
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def new_mesh_object(name, bm, col, matrix=None):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    if matrix is not None:
        obj.matrix_world = matrix
    col.objects.link(obj)
    return obj


def make_box(name, lo, hi, col):
    bm = bmesh.new()
    size = hi - lo
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=size, verts=bm.verts)
    bmesh.ops.translate(bm, vec=(lo + hi) * 0.5, verts=bm.verts)
    return new_mesh_object(name, bm, col)


def make_cylinder(name, center, radius, height, col, segments=96):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments,
                          radius1=radius, radius2=radius, depth=height)
    bmesh.ops.translate(bm, vec=center, verts=bm.verts)
    return new_mesh_object(name, bm, col)


def make_sphere(name, center, radius, col, subdiv=3):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=radius)
    bmesh.ops.translate(bm, vec=center, verts=bm.verts)
    return new_mesh_object(name, bm, col)


def pick_solver(mod, wanted, total_faces):
    """Map our solver choice onto what this Blender offers. 4.x: FAST / EXACT. 5.x: FLOAT / EXACT / MANIFOLD.
    AUTO = MANIFOLD when available (our inputs are closed solids), else EXACT, dropping to the float solver
    above ~700k faces where EXACT crashes."""
    avail = set(mod.bl_rna.properties['solver'].enum_items.keys())
    fast = 'FAST' if 'FAST' in avail else 'FLOAT'
    if wanted == 'AUTO':
        # Manifold (4.5+/5.x) is the right tool for closed solids; the float solver is the fallback.
        # Exact is deliberately not automatic: on prism-vs-tissue inputs it returned the untouched
        # input with extra faces, while Float and Manifold gave the correct thin body.
        return 'MANIFOLD' if 'MANIFOLD' in avail else fast
    if wanted == 'FAST':
        return fast
    if wanted == 'MANIFOLD' and 'MANIFOLD' not in avail:
        return 'EXACT' if total_faces <= 700000 else fast
    if wanted == 'EXACT' and total_faces > 700000:
        return fast
    return wanted


def apply_boolean(target, cutter, operation, solver='AUTO'):
    """Boolean `target` with `cutter` and bake the result into target's mesh.
    Uses the depsgraph rather than modifier_apply so it works without UI context.
    Inputs above ~700k faces total are pushed to the FAST solver: EXACT crashes Blender there."""
    mod = target.modifiers.new("anaplast_bool", 'BOOLEAN')
    mod.operation = operation
    total = len(target.data.polygons) + len(cutter.data.polygons)
    mod.solver = pick_solver(mod, solver, total)
    mod.object = cutter
    dg = bpy.context.evaluated_depsgraph_get()
    ev = target.evaluated_get(dg)
    new_me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=False, depsgraph=dg)
    target.modifiers.remove(mod)
    old_me = target.data
    target.data = new_me
    if old_me.users == 0:
        bpy.data.meshes.remove(old_me)
    return target


def duplicate_object(obj, name, col):
    new = obj.copy()
    try:
        # Insert seats and other Empty helpers have no data block to copy.
        if obj.data is not None:
            new.data = obj.data.copy()
            new.data.name = name
        new.name = name
        col.objects.link(new)
    except Exception:
        delete_object(new)
        raise
    return new


def delete_object(obj):
    me = obj.data if obj.type == 'MESH' else None
    bpy.data.objects.remove(obj)
    if me is not None and me.users == 0:
        bpy.data.meshes.remove(me)


def mesh_stats(obj):
    """Return (n_faces, n_nonmanifold_edges, n_boundary_edges, n_loose_verts)."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    nonman = sum(1 for e in bm.edges if not e.is_manifold)
    boundary = sum(1 for e in bm.edges if e.is_boundary)
    loose = sum(1 for v in bm.verts if not v.link_edges)
    n = len(bm.faces)
    bm.free()
    return n, nonman, boundary, loose


def point_inside(obj, point_world):
    """Parity ray test in object space."""
    inv = obj.matrix_world.inverted()
    p = inv @ point_world
    direction = Vector((1.0, 0.0, 0.0))
    hits = 0
    origin = p.copy()
    for _ in range(64):
        ok, loc, nrm, idx = obj.ray_cast(origin, direction)
        if not ok:
            break
        hits += 1
        origin = loc + direction * 1e-4
    return hits % 2 == 1


def distance_to_surface(obj, point_world):
    inv = obj.matrix_world.inverted()
    ok, loc, nrm, idx = obj.closest_point_on_mesh(inv @ point_world)
    if not ok:
        return float('inf')
    return ((obj.matrix_world @ loc) - point_world).length


def write_binary_stl(obj, filepath):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    mw = obj.matrix_world
    tris = []
    for lt in me.loop_triangles:
        vs = [mw @ me.vertices[i].co for i in lt.vertices]
        n = (vs[1] - vs[0]).cross(vs[2] - vs[0])
        if n.length > 0:
            n.normalize()
        tris.append((n, vs))
    ev.to_mesh_clear()
    with open(filepath, 'wb') as f:
        f.write(b'Anaplast Studio mold part'.ljust(80, b'\0'))
        f.write(struct.pack('<I', len(tris)))
        for n, vs in tris:
            f.write(struct.pack('<3f', *n))
            for v in vs:
                f.write(struct.pack('<3f', *v))
            f.write(struct.pack('<H', 0))
    return len(tris)


def cleanup_mesh(obj, dist=1e-4):
    """Merge near-coincident vertices and dissolve degenerate slivers left by exact booleans.
    Returns (verts_removed, faces_removed)."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    v0, f0 = len(bm.verts), len(bm.faces)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=dist)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=dist)
    # NOTE: never fill boundaries here — open scan surfaces and cropped patches have a
    # legitimate outer edge; capping it with one giant n-gon silently ruins later steps.
    # NOTE: no recalc_face_normals here — it inverts internal cavity shells
    # (they are disconnected components) and breaks subsequent booleans.
    v1, f1 = len(bm.verts), len(bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return v0 - v1, f0 - f1


PALETTE = [(0.90, 0.20, 0.20), (0.20, 0.55, 0.95), (0.20, 0.75, 0.30), (0.95, 0.75, 0.10),
           (0.75, 0.30, 0.85), (0.10, 0.80, 0.80), (0.95, 0.45, 0.10), (0.55, 0.35, 0.20)]


def get_material(name, rgb):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = (*rgb, 1.0)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    return mat


def make_marker(name, location, rgb, col, shape='SPHERE', size=2.0):
    """Coloured, labelled mesh marker (visible in Solid shading, unlike empties)."""
    old = bpy.data.objects.get(name)
    if old:
        delete_object(old)
    bm = bmesh.new()
    if shape == 'SPHERE':
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=size)
    else:
        bmesh.ops.create_cube(bm, size=size * 1.6)
    obj = new_mesh_object(name, bm, col)
    obj.location = location
    obj.show_name = True
    obj.show_in_front = True
    obj.data.materials.append(get_material(f"Anaplast_{rgb}", rgb))
    obj.color = (*rgb, 1.0)
    obj["anaplast_marker"]=True
    obj.hide_render=True
    if hasattr(bpy.context.scene,'anaplast'):
        p=bpy.context.scene.anaplast
        obj.hide_set(not (p.show_landmark_markers or p.landmarks_placing))
    return obj


def apply_modifier(target, kind, setup):
    """Add a modifier, configure it with setup(mod), bake the evaluated mesh, remove it."""
    mod = target.modifiers.new("anaplast_mod", kind)
    setup(mod)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = target.evaluated_get(dg)
    new_me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=False, depsgraph=dg)
    target.modifiers.remove(mod)
    old_me = target.data
    target.data = new_me
    if old_me.users == 0:
        bpy.data.meshes.remove(old_me)
    return target


def apply_solidify(target, thickness):
    def setup(m):
        m.thickness = thickness
        m.offset = -1.0          # grow inward from the outer surface
        m.use_even_offset = True
        m.use_rim = True
        m.use_quality_normals = True
    return apply_modifier(target, 'SOLIDIFY', setup)


def is_closed(obj):
    n, nonman, boundary, loose = mesh_stats(obj)
    return boundary == 0 and n > 0


def solid_copy(obj, thickness, flip=False, name=None, outward=0.0, roi=None, max_faces=150000, margin=20.0):
    """Closed, watertight temporary copy of an open scan surface, thickened INTO the head, so
    booleans against it behave. Only the part of the scan within `margin` mm of the roi box is used,
    decimated to at most `max_faces` — a 3-million-face scan never reaches the boolean solver.
    `outward` = extra mm grown toward the viewer (used for clearance). Returns obj itself if closed."""
    if is_closed(obj):
        return obj, False
    col = obj.users_collection[0] if obj.users_collection else bpy.context.scene.collection
    bm = scan_region_bmesh(obj, roi, margin, flip)
    me = bpy.data.meshes.new(name or "anaplast_solid_tmp")
    bm.to_mesh(me)
    bm.free()
    tmp = bpy.data.objects.new(name or "anaplast_solid_tmp", me)
    col.objects.link(tmp)
    nf = len(me.polygons)
    if nf > max_faces:
        ratio = max_faces / nf
        def dec(m):
            m.decimate_type = 'COLLAPSE'
            m.ratio = ratio
            m.use_collapse_triangulate = True
        apply_modifier(tmp, 'DECIMATE', dec)
    def setup(m):
        m.thickness = thickness
        m.offset = -1.0 + 2.0 * min(max(outward, 0.0), thickness) / thickness   # -1 = all inward
        m.use_even_offset = False
        m.use_rim = True
        m.use_quality_normals = True
    apply_modifier(tmp, 'SOLIDIFY', setup)
    cleanup_mesh(tmp)
    return tmp, True


def keep_largest_island(bm):
    """Delete every connected component except the one with the most faces."""
    seen = set()
    islands = []
    for f in bm.faces:
        if f in seen:
            continue
        stack, comp = [f], []
        seen.add(f)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for e in cur.edges:
                for nf in e.link_faces:
                    if nf not in seen:
                        seen.add(nf)
                        stack.append(nf)
        islands.append(comp)
    if len(islands) <= 1:
        return 0
    islands.sort(key=len, reverse=True)
    junk = [f for comp in islands[1:] for f in comp]
    bmesh.ops.delete(bm, geom=junk, context='FACES')
    return len(islands) - 1


def fill_small_holes(bm, max_sides=24):
    """Close perforations (short boundary loops) but leave the long outer boundary open."""
    boundary = [e for e in bm.edges if e.is_boundary]
    if not boundary:
        return 0
    before = len(bm.faces)
    bmesh.ops.holes_fill(bm, edges=boundary, sides=max_sides)
    return len(bm.faces) - before


def inside_fraction(obj, solid, samples=400):
    """Fraction of obj's vertices that lie inside the closed mesh `solid`."""
    me = obj.data
    n = len(me.vertices)
    if n == 0:
        return 0.0
    step = max(1, n // samples)
    pts = [obj.matrix_world @ me.vertices[i].co for i in range(0, n, step)]
    return sum(1 for q in pts if point_inside(solid, q)) / len(pts)


def scan_orientation_flip(bm):
    """True if this open scan sheet's normals point toward its own centroid (i.e. into the head)."""
    bm.normal_update()
    centroid = sum((v.co for v in bm.verts), Vector()) / max(1, len(bm.verts))
    outward = sum(f.normal.dot(f.calc_center_median() - centroid) for f in bm.faces)
    return outward < 0


def scan_orientation_flip_mesh(me):
    """Same test straight on a Mesh datablock, vectorised — fine for multi-million-face scans."""
    import numpy as np
    n = len(me.polygons)
    if n == 0:
        return False
    nrm = np.empty(n * 3); me.polygons.foreach_get("normal", nrm); nrm = nrm.reshape(-1, 3)
    cen = np.empty(n * 3); me.polygons.foreach_get("center", cen); cen = cen.reshape(-1, 3)
    return float((nrm * (cen - cen.mean(axis=0))).sum()) < 0


def scan_region_bmesh(scan, roi=None, margin=20.0, flip=False):
    """World-space bmesh of the scan, clipped to a box roi=(lo, hi) grown by `margin` mm,
    with consistent normals pointing OUT of the head. roi=None keeps the whole scan."""
    import numpy as np
    me = scan.data
    stored_inward = scan_orientation_flip_mesh(me)          # decided on the WHOLE scan, not the clipped patch
    mw = scan.matrix_world
    M = np.array(mw)
    n = len(me.vertices)
    co = np.empty(n * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    tri = np.empty(nt * 3, dtype=np.int64); me.loop_triangles.foreach_get("vertices", tri); tri = tri.reshape(-1, 3)
    poly = np.empty(nt, dtype=np.int64); me.loop_triangles.foreach_get("polygon_index", poly)
    pn = np.empty(len(me.polygons) * 3); me.polygons.foreach_get("normal", pn); pn = pn.reshape(-1, 3)
    if roi is not None:
        lo = np.array(Vector(roi[0]) - Vector((margin,) * 3)); hi = np.array(Vector(roi[1]) + Vector((margin,) * 3))
        w = co @ M[:3, :3].T + M[:3, 3]
        inside = ((w >= lo) & (w <= hi)).all(axis=1)
        keep = inside[tri].all(axis=1)
        if keep.sum() >= 3:
            tri, poly = tri[keep], poly[keep]
    used = np.unique(tri)
    remap = -np.ones(n, dtype=np.int64); remap[used] = np.arange(len(used))
    tmp = bpy.data.meshes.new("anaplast_roi_tmp")
    tmp.from_pydata([tuple(x) for x in co[used]], [], [tuple(int(i) for i in remap[t]) for t in tri])
    bm = bmesh.new()
    bm.from_mesh(tmp)
    bpy.data.meshes.remove(tmp)
    bm.transform(mw)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)      # consistent winding; the side it picks is arbitrary
    bm.normal_update()
    # which side did recalc pick? compare with the scan's own stored normals (world space), sampled
    stored_w = pn[poly] @ M[:3, :3].T
    bm.faces.ensure_lookup_table()
    nf = len(bm.faces)
    step = max(1, nf // 20000)
    agree = sum(bm.faces[i].normal.dot(Vector(stored_w[i])) for i in range(0, nf, step))
    recalc_inward = (agree > 0) == stored_inward
    if recalc_inward != bool(flip):                        # want normals out of the head (unless user flips)
        for f in bm.faces:
            f.normal_flip()
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bm.normal_update()
    return bm


def oriented_scan_bvh(scan, flip=False, roi=None, margin=20.0):
    """BVH of the scan (clipped to roi + margin when given) in WORLD space, normals pointing out of the head."""
    from mathutils.bvhtree import BVHTree
    bm = scan_region_bmesh(scan, roi, margin, flip)
    tree = BVHTree.FromBMesh(bm)
    bm.free()
    return tree


def signed_heights(obj, tree):
    """Signed height of each vertex of obj above the scan surface (negative = inside tissue)."""
    mw = obj.matrix_world
    out = []
    for v in obj.data.vertices:
        w = mw @ v.co
        loc, nrm, idx, dist = tree.find_nearest(w)
        out.append((w - loc).dot(nrm) if loc is not None else 0.0)
    return out


def lift_clear_of_tissue(target, scan, flip, clearance):
    """Move every vertex of target that sits below `clearance` above the scan up to exactly that height,
    along the scan normal. Vertices already higher are untouched. Returns number of vertices moved."""
    tree = oriented_scan_bvh(scan, flip, roi=world_bbox(target))
    mw, inv = target.matrix_world, target.matrix_world.inverted()
    moved = 0
    for v in target.data.vertices:
        w = mw @ v.co
        loc, nrm, idx, dist = tree.find_nearest(w)
        if loc is None:
            continue
        h = (w - loc).dot(nrm)
        if h < clearance:
            v.co = inv @ (loc + nrm * clearance)
            moved += 1
    target.data.update()
    return moved


def deep_inside_fraction(obj, scan, flip, depth):
    """Fraction of obj's vertices more than `depth` mm below the scan surface."""
    tree = oriented_scan_bvh(scan, flip, roi=world_bbox(obj))
    h = signed_heights(obj, tree)
    return sum(1 for x in h if x < -depth) / max(1, len(h))


def boundary_loops(bm):
    """Group boundary edges into loops. Returns list of edge lists."""
    seen = set()
    loops = []
    for e in bm.edges:
        if not e.is_boundary or e in seen:
            continue
        loop, stack = [], [e]
        seen.add(e)
        while stack:
            cur = stack.pop()
            loop.append(cur)
            for v in cur.verts:
                for ne in v.link_edges:
                    if ne.is_boundary and ne not in seen:
                        seen.add(ne)
                        stack.append(ne)
        loops.append(loop)
    return loops


def fill_holes_except_longest(bm):
    """Close every boundary loop except the longest one (the patch's real outer edge)."""
    loops = boundary_loops(bm)
    if len(loops) <= 1:
        return 0
    loops.sort(key=lambda L: sum(e.calc_length() for e in L), reverse=True)
    holes = [e for L in loops[1:] for e in L]
    before = len(bm.faces)
    bmesh.ops.holes_fill(bm, edges=holes, sides=0)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    return len(bm.faces) - before


def islands(bm):
    seen, out = set(), []
    for f in bm.faces:
        if f in seen:
            continue
        comp, stack = [], [f]
        seen.add(f)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for e in cur.edges:
                for nf in e.link_faces:
                    if nf not in seen:
                        seen.add(nf)
                        stack.append(nf)
        out.append(comp)
    return out


def repair_nonmanifold(obj, max_passes=3):
    """Remove faces that create 3+-face edges (usually boolean slivers), refill the small holes left behind.
    Returns (nonmanifold_before, nonmanifold_after)."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    before = sum(1 for e in bm.edges if len(e.link_faces) > 2)
    for _ in range(max_passes):
        bad = [e for e in bm.edges if len(e.link_faces) > 2]
        if not bad:
            break
        doomed = set()
        for e in bad:
            faces = sorted(e.link_faces, key=lambda f: f.calc_area())
            doomed.update(faces[:-2])            # keep the two largest faces on the edge
        bmesh.ops.delete(bm, geom=list(doomed), context='FACES')
        holes = [e for e in bm.edges if e.is_boundary]
        if holes:
            bmesh.ops.holes_fill(bm, edges=holes, sides=12)
    after = sum(1 for e in bm.edges if len(e.link_faces) > 2)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return before, after


def heal_small_holes(obj, max_sides=24):
    """Close short boundary loops (solver slivers) on a mesh that should be closed. Returns loops closed."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    loops = [L for L in boundary_loops(bm) if len(L) <= max_sides]
    n = len(loops)
    if loops:
        bmesh.ops.holes_fill(bm, edges=[e for L in loops for e in L], sides=max_sides)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return n


def build_fitted_solid(patch, scan, flip, clearance=0.5, gap=0.1, wall=None, roi_margin=25.0):
    """Boolean-free construction of a closed sculpt body from an OPEN outer surface `patch`:
      outer  = patch lifted so it never dips below the tissue (>= clearance)
      inner  = each outer vertex projected onto the tissue (+gap), or, if `wall` is given, the outer
               vertex pushed inward by `wall` but never below tissue+gap  (thin cap)
      rim    = quads stitching the outer and inner boundary loops
    Nothing intersects anything, so no boolean is needed. Returns (vertices moved, n_faces)."""
    import numpy as np
    from mathutils.bvhtree import BVHTree
    lo, hi = world_bbox(patch)
    bmscan = scan_region_bmesh(scan, (lo, hi), roi_margin, flip)
    tree = BVHTree.FromBMesh(bmscan)
    bmscan.free()
    mw, inv = patch.matrix_world, patch.matrix_world.inverted()
    bm = bmesh.new()
    bm.from_mesh(patch.data)
    bm.verts.ensure_lookup_table()
    bm.normal_update()
    n = len(bm.verts)
    outer, inner, moved = [], [], 0
    for v in bm.verts:
        w = mw @ v.co
        loc, nrm, idx, dist = tree.find_nearest(w)
        if loc is None:
            outer.append(w); inner.append(w - v.normal * (wall or 1.0)); continue
        h = (w - loc).dot(nrm)
        if h < clearance:                                   # lift the outer surface clear of the tissue
            w = loc + nrm * clearance
            moved += 1
        if wall is None:
            q = loc + nrm * gap                              # fitted: inner surface = tissue
        else:
            cand = w - (mw.to_3x3() @ v.normal).normalized() * wall
            loc2, nrm2, idx2, dist2 = tree.find_nearest(cand)
            h2 = (cand - loc2).dot(nrm2) if loc2 is not None else gap
            q = cand if h2 >= gap else loc2 + nrm2 * gap     # cap: 'wall' deep unless that enters tissue
        outer.append(w); inner.append(q)
    faces = [[v.index for v in f.verts] for f in bm.faces]
    smooth = [f.smooth for f in bm.faces]
    any_smooth = any(smooth)
    # orient boundary edges consistently with their single face so rim quads face outward
    rim = []
    for e in bm.edges:
        if not e.is_boundary:
            continue
        f = e.link_faces[0]
        vs = [v.index for v in f.verts]
        i0, i1 = e.verts[0].index, e.verts[1].index
        a = vs.index(i0); b = vs.index(i1)
        if (a + 1) % len(vs) == b:                           # edge runs i0->i1 in face order
            rim.append((i1, i0))                             # rim quad runs opposite to close the volume
        else:
            rim.append((i0, i1))
    bm.free()
    out = bmesh.new()
    vo = [out.verts.new(inv @ w) for w in outer]
    vi = [out.verts.new(inv @ q) for q in inner]
    out.verts.ensure_lookup_table()
    for f, sm in zip(faces, smooth):
        try:
            out.faces.new([vo[i] for i in f]).smooth = sm
        except ValueError:
            pass
    for f, sm in zip(faces, smooth):
        try:
            out.faces.new([vi[i] for i in reversed(f)]).smooth = sm
        except ValueError:
            pass
    for a, b in rim:
        try:
            out.faces.new([vo[a], vo[b], vi[b], vi[a]]).smooth = any_smooth
        except ValueError:
            pass
    bmesh.ops.recalc_face_normals(out, faces=out.faces)
    # the solid must have outward normals: outer surface normals should agree with the patch's
    out.normal_update()
    out.faces.ensure_lookup_table()
    agree = 0.0
    step = max(1, len(faces) // 5000)
    for i in range(0, len(faces), step):
        f = out.faces[i]
        agree += f.normal.dot(f.calc_center_median() - (inv @ Vector(((lo + hi) * 0.5))))
    if agree < 0:
        for f in out.faces:
            f.normal_flip()
    bmesh.ops.remove_doubles(out, verts=out.verts, dist=1e-5)
    out.to_mesh(patch.data)
    nf = len(out.faces)
    out.free()
    patch.data.update()
    return moved, nf


def fidelity(obj):
    """(faces, mean edge length mm, min/max edge) — resolution fingerprint of a mesh."""
    import numpy as np
    me = obj.data
    if len(me.edges) == 0:
        return len(me.polygons), 0.0, 0.0, 0.0
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    ed = np.empty(len(me.edges) * 2, dtype=np.int64); me.edges.foreach_get("vertices", ed); ed = ed.reshape(-1, 2)
    L = np.linalg.norm(co[ed[:, 0]] - co[ed[:, 1]], axis=1)
    sc = obj.matrix_world.to_scale()
    L = L * float(sum(sc) / 3.0)
    return len(me.polygons), float(L.mean()), float(np.percentile(L, 5)), float(np.percentile(L, 95))


def frame_view(context, obj):
    """Centre every 3D viewport on `obj` (no-op headless)."""
    if bpy.app.background or obj is None:
        return
    try:
        for o in context.view_layer.objects:
            o.select_set(False)
        obj.hide_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        areas = [a for a in context.screen.areas if a.type == 'VIEW_3D']
        areas.sort(key=lambda a: a.width * a.height, reverse=True)
        keep_ortho = getattr(context.scene.anaplast, "keep_ortho", True)
        for area in areas:
            region = next((r for r in area.regions if r.type == 'WINDOW'), None)
            if region is None:
                continue
            rv3d = area.spaces.active.region_3d
            was = rv3d.view_perspective
            with context.temp_override(area=area, region=region):
                bpy.ops.view3d.view_selected(use_all_regions=False)
            if keep_ortho:
                rv3d.view_perspective = 'ORTHO' if was != 'CAMERA' else was
    except Exception:
        pass


def boundary_distances(bm):
    """Approximate geodesic distance (mm) from each vertex to the mesh's open boundary (Dijkstra over edges)."""
    import heapq
    INF = float('inf')
    dist = {v.index: INF for v in bm.verts}
    heap = []
    for v in bm.verts:
        if v.is_boundary:
            dist[v.index] = 0.0
            heap.append((0.0, v.index))
    heapq.heapify(heap)
    bm.verts.ensure_lookup_table()
    while heap:
        d, i = heapq.heappop(heap)
        if d > dist[i]:
            continue
        for e in bm.verts[i].link_edges:
            o = e.other_vert(bm.verts[i])
            nd = d + e.calc_length()
            if nd < dist[o.index]:
                dist[o.index] = nd
                heapq.heappush(heap, (nd, o.index))
    return dist


def avg_edge_length(bm):
    n = len(bm.edges)
    if n == 0:
        return 1.0
    step = max(1, n // 4000)
    edges = bm.edges[:] if hasattr(bm.edges, "__getitem__") else list(bm.edges)
    sel = edges[::step]
    return sum(e.calc_length() for e in sel) / max(1, len(sel))


def _smoothstep(t):
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    return t * t * (3 - 2 * t)


def adapt_margin(patch, scan, flip, band, clearance, roi_margin=25.0, edge_gap=0.1):
    """Seat an open margin, refining stretched edges at the original local density.

    Refinement and deformation are transactional; protected vertices stay fixed.
    World-space calculations keep density and the adaptation band consistent under scale.
    """
    from mathutils.bvhtree import BVHTree
    from mathutils.geometry import barycentric_transform
    lo, hi = world_bbox(patch)
    bmscan = scan_region_bmesh(scan, (lo, hi), roi_margin, flip)
    bmscan.verts.index_update();bmscan.faces.ensure_lookup_table()
    bmscan.normal_update()
    scan_points=[v.co.copy() for v in bmscan.verts]
    scan_normals=[v.normal.copy() for v in bmscan.verts]
    scan_triangles=[tuple(v.index for v in f.verts) for f in bmscan.faces]
    tree = BVHTree.FromBMesh(bmscan);bmscan.free()
    def contact(origin):
        loc,nrm,idx,dd=tree.find_nearest(origin)
        if loc is not None:
            ids=scan_triangles[idx]
            if len(ids)==3:
                normal=barycentric_transform(loc,*(scan_points[i] for i in ids),
                                              *(scan_normals[i] for i in ids))
                if normal.length_squared>1e-12:
                    normal.normalize()
                    if normal.dot(nrm)>0:nrm=normal
        return loc,nrm
    bm=bmesh.new();bm.from_mesh(patch.data)
    try:
        bm.transform(patch.matrix_world)
        bm.verts.index_update();bm.verts.ensure_lookup_table()
        dist=boundary_distances(bm)
        rest=bm.verts.layers.float_vector.new('_margin_rest')
        distance=bm.verts.layers.float.new('_margin_distance')
        spacing=bm.verts.layers.float.new('_margin_spacing')
        locked=bm.verts.layers.float.get(PROTECT)
        for v in bm.verts:
            v[rest]=v.co.copy();v[distance]=min(dist[v.index],1e12)
            lengths=sorted(e.calc_length() for e in v.link_edges)
            v[spacing]=lengths[len(lengths)//2] if lengths else 1.
        original_count=len(bm.verts);original_faces=len(bm.faces)
        def protected(v):return locked is not None and v[locked]>.5
        def seat(v):
            if protected(v):return
            origin=v[rest];d=v[distance]
            loc,nrm=contact(origin)
            if loc is None:return
            h=(origin-loc).dot(nrm)
            if band>0 and d<band:
                want=edge_gap+(clearance-edge_gap)*_smoothstep(d/band)
                target=loc+nrm*want
                v.co=target if h<want else origin+(target-origin)*(1.-_smoothstep(d/band))
            elif h<clearance:v.co=loc+nrm*clearance
        for v in bm.verts:seat(v)
        # Split only edges that have become both stretched and too long relative
        # to the original neighborhood. New samples inherit UVs, masks and colors.
        for iteration in range(16):
            edges=[]
            for e in bm.edges:
                a,b=e.verts
                if min(a[distance],b[distance])>=band or protected(a) or protected(b):continue
                length=e.calc_length();old=(a[rest]-b[rest]).length
                target=max(1e-6,(a[spacing]+b[spacing])*.5)
                if length>1.5*target and length>1.15*old:edges.append(e)
            if not edges:break
            if len(bm.verts)+len(edges)>max(2000000,original_count*3):
                raise RuntimeError('Margin refinement exceeds the mesh budget. Check the fitting scan and Sculpt alignment or use a wider adaptation band')
            bmesh.ops.subdivide_edges(bm,edges=edges,cuts=1,use_grid_fill=True,use_single_edge=True)
            # Subdivision interpolates the already seated surface. Projecting
            # each new point independently can jump between scan faces/sheets,
            # recreating the same stretched edge on every pass indefinitely.
            # Keep the continuous deformation chosen at the original vertices;
            # interpolate its UVs, masks, rest positions and spacing together.
            # Keep split polygons during refinement: triangulating an almost
            # collinear quad can recreate the same unsplit edge forever.
            # Blender retains the new edge vertices and interpolated attributes.
        else:raise RuntimeError('Margin refinement did not converge. Check the selected fitting surface and adaptation band')
        # A subdivided boundary chord can sit slightly through a curved scan.
        # Seat only that open edge after refinement, with a bounded correction;
        # never feed these tiny contact corrections back into subdivision.
        for v in bm.verts:
            if not v.is_boundary or protected(v):continue
            for _ in range(3):
                loc,nrm=contact(v.co)
                if loc is None:break
                v.co=loc+nrm*edge_gap
                nearest,normal,_,_=tree.find_nearest(v.co)
                if nearest is not None and abs((v.co-nearest).dot(normal)-edge_gap)<.002:break
        moved=sum((v.co-v[rest]).length>1e-7 for v in bm.verts)
        added=len(bm.verts)-original_count
        for layer in (rest,):bm.verts.layers.float_vector.remove(layer)
        for layer in (distance,spacing):bm.verts.layers.float.remove(layer)
        bm.transform(patch.matrix_world.inverted());bm.normal_update()
        # Never change another object sharing this mesh datablock.
        if patch.data.users>1:patch.data=patch.data.copy()
        bm.to_mesh(patch.data);patch.data.update()
        patch['margin_added_vertices']=added
        patch['margin_added_faces']=len(bm.faces)-original_faces
        patch['margin_refinement_passes']=iteration
        return moved
    finally:bm.free()


def make_solid_twin(obj, thickness, flip, col):
    """Full-resolution closed twin of an open scan: outer surface untouched, thickened INTO the head."""
    name = f"{obj.name}_Solid"
    old = bpy.data.objects.get(name)
    if old:
        delete_object(old)
    twin = duplicate_object(obj, name, col)
    bm = bmesh.new()
    bm.from_mesh(twin.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    if scan_orientation_flip(bm) != bool(flip):
        for f in bm.faces:
            f.normal_flip()
    bm.to_mesh(twin.data)
    bm.free()
    def setup(m):
        m.thickness = thickness
        m.offset = -1.0
        m.use_even_offset = False
        m.use_rim = True
        m.use_quality_normals = True
    apply_modifier(twin, 'SOLIDIFY', setup)
    twin["anaplast_part"] = "SOLID_TWIN"
    twin.hide_set(True)
    return twin


# ------------------------------------------------------------------ margin cleanup + prism solids (boolean pathway)
def clean_margin(bm, smooth_iters=6, smooth_factor=0.6, max_rounds=12):
    """Make an open patch's boundary boolean-friendly: strip dangling triangles/fins, remove
    non-manifold boundary vertices, keep the main island, close interior holes, then relax the
    outer loop. Returns (faces removed, holes closed)."""
    removed = 0
    for _ in range(min(max_rounds, 4)):
        # dangling faces = attached to the rest by a single edge (never erode a normal zigzag edge)
        spikes = [f for f in bm.faces if len(f.edges) - sum(1 for e in f.edges if e.is_boundary) <= 1]
        bad = [v for v in bm.verts if sum(1 for e in v.link_edges if e.is_boundary) > 2]
        doomed = set(spikes)
        for v in bad:
            doomed.update(v.link_faces)
        if not doomed:
            break
        removed += len(doomed)
        bmesh.ops.delete(bm, geom=list(doomed), context='FACES')
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context='VERTS')
    keep_largest_island(bm)
    holes = fill_holes_except_longest(bm)
    # relax the outer loop so the margin is a smooth curve instead of a scan-resolution zigzag
    for _ in range(smooth_iters):
        bverts = [v for v in bm.verts if v.is_boundary]
        new = {}
        for v in bverts:
            nb = [e.other_vert(v) for e in v.link_edges if e.is_boundary]
            if len(nb) == 2:
                new[v] = v.co * (1 - smooth_factor) + (nb[0].co + nb[1].co) * (0.5 * smooth_factor)
        for v, co in new.items():
            v.co = co
    return removed, holes


def average_normal(obj):
    """Area-weighted average face normal of obj in world space (unit)."""
    import numpy as np
    me = obj.data
    n = len(me.polygons)
    nrm = np.empty(n * 3); me.polygons.foreach_get("normal", nrm); nrm = nrm.reshape(-1, 3)
    area = np.empty(n); me.polygons.foreach_get("area", area)
    v = (nrm * area[:, None]).sum(axis=0)
    M = np.array(obj.matrix_world)
    v = M[:3, :3] @ v
    return Vector(v).normalized()


def extrude_to_plane(bm_world, direction, plane_t):
    """Close an open patch (world-space bmesh, single outer boundary loop) into a solid by projecting
    every vertex along `direction` onto the plane {x : x·direction = plane_t}. A prism along one
    direction never self-intersects for a surface seen as a height-field from that direction."""
    d = Vector(direction).normalized()
    verts = list(bm_world.verts)
    bm_world.verts.ensure_lookup_table()
    out = bmesh.new()
    vo = [out.verts.new(v.co) for v in verts]
    vi = [out.verts.new(v.co + d * (plane_t - v.co.dot(d))) for v in verts]
    out.verts.ensure_lookup_table()
    idx = {v: i for i, v in enumerate(verts)}
    for f in bm_world.faces:
        ids = [idx[v] for v in f.verts]
        try:
            out.faces.new([vo[i] for i in ids]).smooth = f.smooth
        except ValueError:
            pass
        try:
            out.faces.new([vi[i] for i in reversed(ids)])
        except ValueError:
            pass
    for e in bm_world.edges:
        if not e.is_boundary:
            continue
        f = e.link_faces[0]
        vs = [idx[v] for v in f.verts]
        i0, i1 = idx[e.verts[0]], idx[e.verts[1]]
        a, b = vs.index(i0), vs.index(i1)
        if (a + 1) % len(vs) == b:
            i0, i1 = i1, i0
        try:
            out.faces.new([vo[i0], vo[i1], vi[i1], vi[i0]])
        except ValueError:
            pass
    bmesh.ops.recalc_face_normals(out, faces=out.faces)
    bmesh.ops.remove_doubles(out, verts=out.verts, dist=1e-5)
    return out


def offset_shell(bm_world, wall, smooth_passes=6, feather=0.0, edge_thickness=0.2):
    """Closed shell: outer = patch (untouched), inner = a SMOOTHED copy of the patch pushed `wall` mm along
    smoothed normals, rim stitched. Smoothing the inner side keeps a noisy scan's offset from
    self-intersecting; the outer skin keeps every pore. Rim quads run across the wall, so they never
    coincide with a prism's rim — safe to boolean against."""
    bm_world.normal_update()
    verts = list(bm_world.verts)
    idx = {v: i for i, v in enumerate(verts)}
    pos = [v.co.copy() for v in verts]
    nrm = [v.normal.copy() for v in verts]
    nbrs = [[idx[e.other_vert(v)] for e in v.link_edges] for v in verts]
    bnd = [v.is_boundary for v in verts]
    for _ in range(smooth_passes):
        pos = [pos[i] if bnd[i] or not nbrs[i] else (pos[i] + sum((pos[j] for j in nbrs[i]), Vector()) / len(nbrs[i])) * 0.5 for i in range(len(verts))]
        nrm = [(nrm[i] + sum((nrm[j] for j in nbrs[i]), Vector())).normalized() if nbrs[i] else nrm[i] for i in range(len(verts))]
    # thickness tapers from `edge_thickness` at the open edge to `wall` at `feather` mm in (crown-margin style)
    if feather > 0:
        bm_world.verts.index_update()
        dist = boundary_distances(bm_world)
        thick = [edge_thickness + (wall - edge_thickness) * _smoothstep(dist.get(v.index, float('inf')) / feather) for v in verts]
    else:
        thick = [wall] * len(verts)
    out = bmesh.new()
    vo = [out.verts.new(v.co) for v in verts]
    vi = [out.verts.new(pos[i] - nrm[i] * thick[i]) for i in range(len(verts))]
    for f in bm_world.faces:
        ids = [idx[v] for v in f.verts]
        try:
            out.faces.new([vo[i] for i in ids]).smooth = f.smooth
        except ValueError:
            pass
        try:
            out.faces.new([vi[i] for i in reversed(ids)]).smooth = f.smooth
        except ValueError:
            pass
    for e in bm_world.edges:
        if not e.is_boundary:
            continue
        f = e.link_faces[0]
        vs = [idx[v] for v in f.verts]
        i0, i1 = idx[e.verts[0]], idx[e.verts[1]]
        a, b = vs.index(i0), vs.index(i1)
        if (a + 1) % len(vs) == b:
            i0, i1 = i1, i0
        try:
            out.faces.new([vo[i0], vo[i1], vi[i1], vi[i0]])
        except ValueError:
            pass
    bmesh.ops.recalc_face_normals(out, faces=out.faces)
    bmesh.ops.remove_doubles(out, verts=out.verts, dist=1e-5)
    return out


def signed_volume(bm):
    """Signed volume of a closed bmesh (negative = normals point inward)."""
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
    vol = 0.0
    for f in bm.faces:
        a, b, c = (v.co for v in f.verts)
        vol += a.dot(b.cross(c))
    return vol / 6.0


def ensure_outward(bm):
    """Closed solids must have outward normals or every boolean operates on their complement."""
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    if signed_volume(bm) < 0:
        for f in bm.faces:
            f.normal_flip()
    bm.normal_update()


def prism_object(name, bm_world, direction, plane_t, col):
    out = extrude_to_plane(bm_world, direction, plane_t)
    ensure_outward(out)
    me = bpy.data.meshes.new(name)
    out.to_mesh(me)
    out.free()
    obj = bpy.data.objects.new(name, me)
    col.objects.link(obj)
    return obj


def tissue_prism(scan, roi, direction, plane_t, flip, max_faces, col, margin=25.0):
    """Solid of everything BEHIND the tissue surface (within roi+margin) down to the plane."""
    bm = scan_region_bmesh(scan, roi, margin, flip)
    clean_margin(bm, smooth_iters=0)
    if len(bm.faces) > max_faces:
        me = bpy.data.meshes.new("anaplast_tissue_tmp"); bm.to_mesh(me); bm.free()
        tmp = bpy.data.objects.new("anaplast_tissue_tmp", me); col.objects.link(tmp)
        ratio = max_faces / len(me.polygons)
        def dec(m):
            m.decimate_type = 'COLLAPSE'; m.ratio = ratio; m.use_collapse_triangulate = True
        apply_modifier(tmp, 'DECIMATE', dec)
        bm = bmesh.new(); bm.from_mesh(tmp.data)
        delete_object(tmp)
        clean_margin(bm, smooth_iters=0)
    obj = prism_object("anaplast_tissue_prism", bm, direction, plane_t, col)
    bm.free()
    return obj


def boolean_fit(patch, scan, flip, direction, wall=None, clearance=0.5, band=0.0, max_faces=150000, solver='EXACT', extra_depth=8.0):
    """Boolean pathway. patch: OPEN outer surface object (modified in place into the closed result).
    fitted (wall=None): prism(patch) − prism(tissue)      → tissue-side surface = tissue
    cap    (wall):      prism(patch) − prism(patch shifted by wall) − prism(tissue)
    Returns dict with counts for the report."""
    col = patch.users_collection[0] if patch.users_collection else bpy.context.scene.collection
    d = Vector(direction).normalized()
    # 1. margin: seat (adapt/lift) then clean
    if band > 0:
        adapt_margin(patch, scan, flip, band, clearance)
    else:
        lift_clear_of_tissue(patch, scan, flip, clearance)
    bm = bmesh.new(); bm.from_mesh(patch.data); bm.transform(patch.matrix_world)
    removed, holes = clean_margin(bm)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    # patch normals must face OUT of the head (against d)
    bm.normal_update()
    if sum(f.normal.dot(d) for f in bm.faces) > 0:
        for f in bm.faces:
            f.normal_flip()
    t_max = max(v.co.dot(d) for v in bm.verts)
    t_min = min(v.co.dot(d) for v in bm.verts)
    plane_t = t_max + extra_depth
    lo, hi = world_bbox(patch)
    # 2. solids
    pros = prism_object("anaplast_pros_prism", bm, d, plane_t, col)
    tissue = tissue_prism(scan, (lo - Vector((extra_depth,) * 3), hi + Vector((extra_depth,) * 3)), d, plane_t + extra_depth, flip, max_faces, col)
    if wall is not None:
        # cap: replace the prism by an offset shell (outer surface + `wall`), then subtract the tissue as usual
        delete_object(pros)
        shell_bm = offset_shell(bm, wall)
        ensure_outward(shell_bm)
        me = bpy.data.meshes.new("anaplast_cap_shell"); shell_bm.to_mesh(me); shell_bm.free()
        pros = bpy.data.objects.new("anaplast_cap_shell", me); col.objects.link(pros)
    bm.free()
    n_in = len(pros.data.polygons)
    apply_boolean(pros, tissue, 'DIFFERENCE', solver=solver)
    delete_object(tissue)
    # 3. pick the result body. Some solver builds return the untouched input as an extra body next to
    #    the real result; the real result is the body whose face count differs from the input.
    b2 = bmesh.new(); b2.from_mesh(pros.data)
    comps = islands(b2)
    frag = 0
    if len(comps) > 1:
        real = [c for c in comps if len(c) != n_in] or comps
        keep = max(real, key=len)
        junk = [f for c in comps if c is not keep for f in c]
        bmesh.ops.delete(b2, geom=junk, context='FACES')
        frag = len(comps) - 1
    b2.to_mesh(pros.data); b2.free(); pros.data.update()
    # 4. heal — only if something is open. A clean result is left untouched: the Manifold solver
    #    emits duplicate vertices along sharp seams on purpose, and welding them breaks it.
    n0, nonman0, boundary0, loose0 = mesh_stats(pros)
    if boundary0 or nonman0 or loose0:
        cleanup_mesh(pros)
        heal_small_holes(pros)
        repair_nonmanifold(pros)
    # 4. hand the geometry back to `patch` (keeps name/slot); result is in world space → identity matrix
    old = patch.data
    patch.data = pros.data
    patch.matrix_world = Matrix.Identity(4)
    bpy.data.objects.remove(pros)
    if old.users == 0:
        bpy.data.meshes.remove(old)
    n, nonman, boundary, loose = mesh_stats(patch)
    return {"removed": removed, "holes": holes, "fragments": frag, "faces": n, "nonmanifold": nonman, "open": boundary}


# ------------------------------------------------------------------ ZBrush-style: extract → dynamesh → boolean
VOXEL_BUDGET = 40_000_000      # ACTIVE voxels: OpenVDB stores a narrow band around the surface, not the volume.
                               # Measured: a 105×102×51 mm block at 0.15 mm = ~11 M active voxels, 0.8 GB, 13 s.


def safe_voxel(obj, voxel, budget=VOXEL_BUDGET):
    """The finest voxel size that keeps the remesh's active voxel count (surface area / voxel², times a
    6-voxel band) under `budget`. Surface area is estimated from the bounding box."""
    lo, hi = world_bbox(obj)
    a, b, c = max(hi.x - lo.x, 1e-3), max(hi.y - lo.y, 1e-3), max(hi.z - lo.z, 1e-3)
    area = 2.0 * (a * b + b * c + c * a)
    need = area / (voxel ** 2) * 6.0
    if need <= budget:
        return voxel
    return (area * 6.0 / budget) ** 0.5


def voxel_remesh(obj, voxel, budget=VOXEL_BUDGET, adaptivity=0.0):
    """Voxel remesh: rebuild obj at `voxel` mm resolution (coarsened if the
    surface would exceed the memory budget — see safe_voxel). `adaptivity` is dimensionless;
    zero disables adaptive simplification. Returns the voxel
    size actually used."""
    voxel = safe_voxel(obj, voxel, budget)
    def setup(m):
        m.mode = 'VOXEL'
        m.voxel_size = voxel
        m.adaptivity = adaptivity
        m.use_smooth_shade = True
        m.use_remove_disconnected = False
    apply_modifier(obj, 'REMESH', setup)
    return voxel


def voxel_fit(patch, scan, flip, wall, clearance=0.5, band=6.0, voxel=0.15, scan_thickness=15.0, solver='AUTO', margin=12.0,
              feather=0.0, edge_thickness=0.2, fit_offset=0.0, apply_despeckle=None):
    """Extract a shell of `wall` mm toward the tissue from the open surface, voxel remesh it, voxel remesh the
    thickened tissue, subtract. wall=None → deep extract (fitted solid). Modifies `patch` in place."""
    col = patch.users_collection[0] if patch.users_collection else bpy.context.scene.collection
    p_ = bpy.context.scene.anaplast
    if (p_.despeckle_enabled if apply_despeckle is None else apply_despeckle) and p_.despeckle_mm > 0:
        despeckle(patch, p_.despeckle_mm)
    if band > 0:
        adapt_margin(patch, scan, flip, band, clearance)
    else:
        lift_clear_of_tissue(patch, scan, flip, clearance)
    bm = bmesh.new(); bm.from_mesh(patch.data); bm.transform(patch.matrix_world)
    removed, holes = clean_margin(bm)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    patch_mesh_copy = bpy.data.meshes.new("anaplast_src_copy"); bm.to_mesh(patch_mesh_copy)
    # normals must face OUT of the head so the extract goes toward the tissue
    tree = oriented_scan_bvh(scan, flip, roi=world_bbox(patch), margin=margin)
    agree = 0.0
    step = max(1, len(bm.faces) // 3000)
    faces = bm.faces[:] if hasattr(bm.faces, "__getitem__") else list(bm.faces)
    bm.faces.ensure_lookup_table()
    for i in range(0, len(bm.faces), step):
        f = bm.faces[i]
        loc, nrm, idx, dist = tree.find_nearest(f.calc_center_median())
        if nrm is not None:
            agree += f.normal.dot(nrm)
    if agree < 0:
        for f in bm.faces:
            f.normal_flip()
    depth = wall if wall is not None else scan_thickness
    # a sheet thinner than ~2.5 voxels cannot survive DynaMesh — it aliases into crumbs. The extract stays
    # above that; the true feather (down to edge_thickness) is cut exactly after the voxel build.
    voxel_edge = max(edge_thickness, 2.5 * voxel)
    shell_bm = offset_shell(bm, depth, feather=feather if wall is not None else 0.0, edge_thickness=voxel_edge)
    bm.free()
    ensure_outward(shell_bm)
    me = bpy.data.meshes.new("anaplast_extract"); shell_bm.to_mesh(me); shell_bm.free()
    shell = bpy.data.objects.new("anaplast_extract", me); col.objects.link(shell)
    voxel_remesh(shell, voxel)                                  # dynamesh the extract
    # tissue: clipped scan region, thickened inward, dynameshed at the same resolution
    lo, hi = world_bbox(shell)
    tbm = scan_region_bmesh(scan, (lo, hi), margin, flip)
    clean_margin(tbm, smooth_iters=0)
    # relief is applied after the build by resnap_fitting — offsetting the tissue here self-intersects
    # wherever the surface is concave, which is what made the old relief unreliable
    tme = bpy.data.meshes.new("anaplast_tissue"); tbm.to_mesh(tme); tbm.free()
    tissue = bpy.data.objects.new("anaplast_tissue", tme); col.objects.link(tissue)
    def sol(m):
        m.thickness = scan_thickness
        m.offset = -1.0
        m.use_even_offset = False
        m.use_rim = True
        m.use_quality_normals = True
    apply_modifier(tissue, 'SOLIDIFY', sol)
    tb = bmesh.new(); tb.from_mesh(tissue.data); ensure_outward(tb); tb.to_mesh(tissue.data); tb.free()
    voxel_remesh(tissue, voxel)
    n_in = len(shell.data.polygons)
    apply_boolean(shell, tissue, 'DIFFERENCE', solver=solver)
    delete_object(tissue)
    b2 = bmesh.new(); b2.from_mesh(shell.data)
    comps = islands(b2)
    frag = 0
    if len(comps) > 1:
        real = [c for c in comps if len(c) != n_in] or comps
        keep = max(real, key=len)
        bmesh.ops.delete(b2, geom=[f for c in comps if c is not keep for f in c], context='FACES')
        frag = len(comps) - 1
    b2.to_mesh(shell.data); b2.free(); shell.data.update()
    # final DynaMesh: the boolean's trimmed wedge becomes a clean rounded edge and any voxel crumbs
    # (thin slivers that survived attached by a neck) fall off; then keep the main body only
    voxel_remesh(shell, voxel)
    b3 = bmesh.new(); b3.from_mesh(shell.data)
    keep_largest_island(b3)
    b3.to_mesh(shell.data); b3.free(); shell.data.update()
    # exact feather: thin the inner surface toward the skin down to edge_thickness at the margin (no voxels involved)
    if wall is not None and feather > 0:
        src_tmp = bpy.data.objects.new("anaplast_src_tmp", patch_mesh_copy)
        col.objects.link(src_tmp)
        feather_edge_exact(shell, src_tmp, feather, edge_thickness, wall)
        bpy.data.objects.remove(src_tmp)
        bpy.data.meshes.remove(patch_mesh_copy)
    else:
        bpy.data.meshes.remove(patch_mesh_copy)
    for pg in shell.data.polygons:
        pg.use_smooth = True
    old = patch.data
    patch.data = shell.data
    patch.matrix_world = Matrix.Identity(4)
    bpy.data.objects.remove(shell)
    if old.users == 0:
        bpy.data.meshes.remove(old)
    n, nonman, boundary, loose = mesh_stats(patch)
    return {"removed": removed, "holes": holes, "fragments": frag, "faces": n, "nonmanifold": nonman, "open": boundary}


def despeckle(obj, threshold=0.3, iterations=5):
    """Remove scan spikes: vertices standing more than `threshold` mm off a lightly smoothed copy of the
    surface (along the normal) are moved onto that smoothed surface. Texture below the threshold is untouched.
    Returns the number of vertices corrected."""
    import numpy as np
    me = obj.data
    n = len(me.vertices)
    if n == 0:
        return 0
    orig = np.empty(n * 3); me.vertices.foreach_get("co", orig); orig = orig.reshape(-1, 3)
    nrm = np.empty(n * 3); me.vertices.foreach_get("normal", nrm); nrm = nrm.reshape(-1, 3)
    tmp = duplicate_object(obj, "anaplast_despeckle_tmp", obj.users_collection[0] if obj.users_collection else bpy.context.scene.collection)
    def setup(m):
        m.factor = 0.5
        m.iterations = iterations
    apply_modifier(tmp, 'SMOOTH', setup)
    sm = np.empty(n * 3); tmp.data.vertices.foreach_get("co", sm); sm = sm.reshape(-1, 3)
    delete_object(tmp)
    dev = ((orig - sm) * nrm).sum(axis=1)
    bad = np.abs(dev) > threshold
    prot = protect_mask(obj)
    if prot is not None and len(prot) == n:
        bad &= ~prot
    if bad.any():
        orig[bad] = sm[bad]
        me.vertices.foreach_set("co", orig.ravel())
        me.update()
    return int(bad.sum())


def thickness_map(obj, thin=0.2, full=1.0, max_range=6.0):
    """Colour a closed sculpt by local wall thickness (ray from each vertex inward to the far side):
    red = at/below `thin`, yellow→green up to `full`, blue beyond. Returns (min, median, max, fraction_below_thin)."""
    import numpy as np
    from mathutils.bvhtree import BVHTree
    me = obj.data
    dg = bpy.context.evaluated_depsgraph_get()
    tree = BVHTree.FromObject(obj, dg)
    n = len(me.vertices)
    th = np.full(n, max_range)
    for i, v in enumerate(me.vertices):
        nrm = v.normal
        o = v.co - nrm * 0.02
        loc, hn, idx, dist = tree.ray_cast(o, -nrm, max_range)
        if loc is not None:
            th[i] = dist + 0.02
    t = np.clip((th - thin) / max(full - thin, 1e-6), 0.0, 2.0)          # 0 = thin, 1 = full wall, 2 = double
    r = np.where(t < 1, 1 - t, 0.0)
    g = np.where(t < 1, t, np.clip(2 - t, 0, 1))
    b = np.clip(t - 1, 0, 1)
    attr = me.color_attributes.get("thickness") or me.color_attributes.new("thickness", 'FLOAT_COLOR', 'POINT')
    attr.data.foreach_set("color", np.stack([r, g, b, np.ones(n)], axis=1).ravel())
    me.color_attributes.active_color = attr
    mat = bpy.data.materials.get("Anaplast_Thickness")
    if mat is None:
        mat = bpy.data.materials.new("Anaplast_Thickness"); mat.use_nodes = True
        nt = mat.node_tree
        a = nt.nodes.new("ShaderNodeAttribute"); a.attribute_name = "thickness"
        nt.links.new(a.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    if not me.materials:
        me.materials.append(mat)
    else:
        me.materials[0] = mat
    return float(th.min()), float(np.median(th)), float(th.max()), float((th < thin).mean())


# ------------------------------------------------------------------ skin micro-relief (final step, after the build)
def skin_microtexture(obj, source, strength=1.0, preset='MEDIUM', subdivide=True, seed=0):
    """Displace the OUTER skin of a built prototype with a multi-scale skin relief:
    polygonal cell network (fine + coarse grooves), scattered pores, regional variation.
    Only vertices within 0.3 mm of `source` (the original outer surface) move; inner surface and rim don't.
    Returns (verts displaced, rms displacement mm)."""
    import numpy as np
    from mathutils import noise
    from mathutils.bvhtree import BVHTree
    P = {
        'FINE':   dict(cell=0.7,  groove=0.030, cell2=2.2, groove2=0.035, pore=0.45, pore_depth=0.045, pore_r=0.09, pore_p=0.25),
        'MEDIUM': dict(cell=1.0,  groove=0.045, cell2=3.0, groove2=0.050, pore=0.55, pore_depth=0.060, pore_r=0.12, pore_p=0.30),
        'COARSE': dict(cell=1.4,  groove=0.060, cell2=4.0, groove2=0.070, pore=0.70, pore_depth=0.080, pore_r=0.15, pore_p=0.35),
    }[preset]
    if subdivide:
        def setup(m):
            m.levels = 1; m.render_levels = 1; m.subdivision_type = 'SIMPLE'
        apply_modifier(obj, 'SUBSURF', setup)
    me = obj.data
    n = len(me.vertices)
    co = np.empty(n * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    nrm = np.empty(n * 3); me.vertices.foreach_get("normal", nrm); nrm = nrm.reshape(-1, 3)
    M = np.array(obj.matrix_world); W = co @ M[:3, :3].T + M[:3, 3]
    dg = bpy.context.evaluated_depsgraph_get()
    tree = BVHTree.FromObject(source, dg)
    s_inv = source.matrix_world.inverted()
    outer = np.zeros(n, dtype=bool)
    for i in range(n):
        loc, nn, idx, dist = tree.find_nearest(s_inv @ Vector(W[i]), 0.6)
        if loc is not None and dist < 0.3:
            outer[i] = True
    off = Vector((seed * 13.7, seed * 7.1, seed * 3.3))
    disp = np.zeros(n)
    def smooth(t):
        t = min(max(t, 0.0), 1.0); return t * t * (3 - 2 * t)
    for i in np.nonzero(outer)[0]:
        q = Vector(W[i]) + off
        # fine cell network: grooves where two Voronoi cells meet
        d = noise.voronoi(q / P['cell'])[0]
        edge = (d[1] - d[0]) * P['cell']
        g1 = -P['groove'] * (1.0 - smooth(edge / (0.12 * P['cell'])))
        # coarse network, same idea, larger cells
        d2 = noise.voronoi(q / P['cell2'] + Vector((31.0, 17.0, 5.0)))[0]
        edge2 = (d2[1] - d2[0]) * P['cell2']
        g2 = -P['groove2'] * (1.0 - smooth(edge2 / (0.10 * P['cell2'])))
        # pores: pits at some cell centres of a small Voronoi
        vd, vp = noise.voronoi(q / P['pore'] + Vector((7.0, 43.0, 11.0)))
        c = vp[0]
        h = (abs(c.x * 12.9898 + c.y * 78.233 + c.z * 37.719) * 43758.5453) % 1.0
        pore = 0.0
        if h < P['pore_p']:
            r = vd[0] * P['pore']
            pore = -P['pore_depth'] * (1.0 - smooth(r / P['pore_r']))
        # regional variation so it never looks uniform
        var = 0.7 + 0.3 * (noise.noise(q * 0.08) + 1.0) * 0.5
        disp[i] = (g1 + g2 + pore) * var * strength
    co += nrm * disp[:, None]
    me.vertices.foreach_set("co", co.ravel())
    me.update()
    for pg in me.polygons:
        pg.use_smooth = True
    return int(outer.sum()), float(np.sqrt((disp[outer] ** 2).mean())) if outer.any() else 0.0


def feather_edge_exact(obj, source, feather, edge_thickness, full):
    """After the voxel build: taper the INNER surface toward the outer skin within `feather` mm of the margin,
    down to `edge_thickness` at the edge — exact geometry, independent of the voxel size.
    The outer skin is never moved. Returns vertices moved."""
    import numpy as np
    from mathutils.bvhtree import BVHTree
    sbm = bmesh.new(); sbm.from_mesh(source.data); sbm.transform(source.matrix_world)
    sbm.verts.index_update(); sbm.faces.ensure_lookup_table()
    bdist = boundary_distances(sbm)
    face_d = [min(bdist.get(v.index, 1e9) for v in f.verts) for f in sbm.faces]
    tree = BVHTree.FromBMesh(sbm)
    sbm.free()
    me = obj.data
    mw, inv = obj.matrix_world, obj.matrix_world.inverted()
    moved = 0
    for v in me.vertices:
        w = mw @ v.co
        loc, nrm, idx, dist = tree.find_nearest(w, feather + full + 2.0)
        if loc is None:
            continue
        depth = (loc - w).dot(nrm)                 # +: vertex lies under the skin (inner side)
        if depth < 0.03:
            continue                              # outer skin or its rim: leave alone
        d = face_d[idx]
        if d >= feather:
            continue
        target = edge_thickness + (full - edge_thickness) * _smoothstep(d / feather)
        if depth > target:
            v.co = inv @ (loc - nrm * target)
            moved += 1
    me.update()
    return moved


def case_tag(p):
    """File-name stem: Last_First_YYYY-MM-DD (falls back to Case ID). Letters, digits, - and _ only."""
    import re, time
    parts = [x.strip() for x in (p.patient_last, p.patient_first, getattr(p, "mrn", "")) if x and x.strip()]
    date = p.case_date.strip() if p.case_date and p.case_date.strip() else time.strftime("%Y-%m-%d")
    stem = "_".join(parts + [date]) if parts else f"Case_{date}"
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", stem)


def export_mesh(context, obj, stem, fmt='STL'):
    """Write obj to stem + extension. STL through our own writer (context-free); OBJ through Blender's exporter."""
    if fmt == 'OBJ':
        for o in context.view_layer.objects:
            o.select_set(False)
        obj.hide_set(False); obj.select_set(True); context.view_layer.objects.active = obj
        bpy.ops.wm.obj_export(filepath=stem + ".obj", export_selected_objects=True, export_materials=False,
                              export_uv=bool(obj.data.uv_layers), export_normals=True, apply_modifiers=True, global_scale=1.0)
        return stem + ".obj"
    write_binary_stl(obj, stem + ".stl")
    return stem + ".stl"


def clear_old(folder, pattern_prefix, exts):
    """Delete earlier files for this case+kind so a re-export never leaves stale copies behind."""
    import glob, os
    n = 0
    for ext in exts:
        for f in glob.glob(os.path.join(folder, pattern_prefix + "*" + ext)):
            try:
                os.remove(f); n += 1
            except OSError:
                pass
    return n


PROTECT = "anaplast_protect"


def protect_mask(obj):
    """Per-vertex 0/1 array of protected vertices (painted with the sculpt mask), or None."""
    import numpy as np
    if obj is None:
        return None
    attr = obj.data.attributes.get(PROTECT)
    if attr is None:
        return None
    vals = np.empty(len(obj.data.vertices))
    attr.data.foreach_get("value", vals)
    return vals > 0.5


def bake_sculpt_mask_to_protect(obj):
    """Copy the current sculpt mask into the persistent protect attribute. Returns count."""
    import numpy as np
    me = obj.data
    src = me.attributes.get(".sculpt_mask")
    if src is None:
        return 0
    vals = np.empty(len(me.vertices)); src.data.foreach_get("value", vals)
    attr = me.attributes.get(PROTECT) or me.attributes.new(PROTECT, 'FLOAT', 'POINT')
    attr.data.foreach_set("value", (vals > 0.5).astype(float))
    me.update()
    return int((vals > 0.5).sum())


def restore_detail(proto, source, max_dist=0.6, protect_inner=True, snap=True):
    """Voxel remeshing rounds off fine relief. Move every OUTER vertex of the prototype onto the original
    scan surface (nearest point, within max_dist). Detail remains limited by the remeshed topology.
    Inner/fitting surface and rim are left alone. Returns (moved, mean shift mm)."""
    import numpy as np
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    tree = BVHTree.FromObject(source, dg)
    s_mw, s_inv = source.matrix_world, source.matrix_world.inverted()
    me = proto.data
    mw, inv = proto.matrix_world, proto.matrix_world.inverted()
    n = len(me.vertices)
    co = np.empty(n * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    nrm = np.empty(n * 3); me.vertices.foreach_get("normal", nrm); nrm = nrm.reshape(-1, 3)
    M = np.array(mw); W = co @ M[:3, :3].T + M[:3, 3]
    moved, shifts = 0, []
    outer = np.zeros(n, dtype=bool)
    # 1. identify the outer skin, 2. subdivide-free detail transfer: ray along the prototype normal onto the
    #    source (falling back to nearest point) so the vertex lands on the true surface, not the nearest blob
    for i in range(n):
        loc, nn, idx, dist = tree.find_nearest(s_inv @ Vector(W[i]), max_dist)
        if loc is None or nn is None:
            continue
        if (s_mw.to_3x3() @ nn).normalized().dot(Vector(nrm[i]).normalized()) < 0.3:
            continue
        d_local = (s_inv.to_3x3() @ Vector(nrm[i])).normalized()
        o_local = s_inv @ Vector(W[i])
        hit = None
        for sign in (1.0, -1.0):
            h, hn, hidx, hdist = tree.ray_cast(o_local - d_local * (sign * max_dist), d_local * sign, 2 * max_dist)
            if h is not None and (h - o_local).length <= max_dist:
                hit = h
                break
        outer[i] = True
        if not snap:
            continue
        w_new = s_mw @ (hit if hit is not None else loc)
        shifts.append((Vector(W[i]) - w_new).length)
        co[i] = np.array(inv @ w_new)
        moved += 1
    if moved:
        me.vertices.foreach_set("co", co.ravel())
        me.update()
    attr = me.attributes.get("anaplast_outer") or me.attributes.new("anaplast_outer", 'FLOAT', 'POINT')
    attr.data.foreach_set("value", outer.astype(float))
    return moved, (float(np.mean(shifts)) if shifts else 0.0)


def mask_inner_surface(proto):
    """Set the sculpt mask on everything that is NOT the outer skin, so brushes cannot move the fitting
    surface. Uses the 'anaplast_outer' attribute written by restore_detail. Returns masked count."""
    import numpy as np
    me = proto.data
    src = me.attributes.get("anaplast_outer")
    if src is None:
        return 0
    n = len(me.vertices)
    outer = np.empty(n); src.data.foreach_get("value", outer)
    m = me.attributes.get(".sculpt_mask") or me.attributes.new(".sculpt_mask", 'FLOAT', 'POINT')
    masked = (outer < 0.5).astype(float)
    m.data.foreach_set("value", masked)
    me.update()
    return int(masked.sum())


def resnap_fitting(proto, scan, flip, fit_offset=0.1, max_dist=3.0, margin=12.0):
    """Put the fitting (inner) surface back exactly on the tissue + relief, e.g. after sculpting.
    Outer skin is untouched. Returns (moved, max correction mm)."""
    import numpy as np
    from mathutils.bvhtree import BVHTree
    lo, hi = world_bbox(proto)
    bms = scan_region_bmesh(scan, (lo, hi), margin, flip)
    tree = BVHTree.FromBMesh(bms)
    bms.free()
    me = proto.data
    outer_attr = me.attributes.get("anaplast_outer")
    n = len(me.vertices)
    outer = np.zeros(n)
    if outer_attr is not None:
        outer_attr.data.foreach_get("value", outer)
    mw, inv = proto.matrix_world, proto.matrix_world.inverted()
    moved, worst = 0, 0.0
    for i, v in enumerate(me.vertices):
        if outer[i] > 0.5:
            continue
        w = mw @ v.co
        loc, nrm, idx, dist = tree.find_nearest(w, max_dist)
        if loc is None or nrm is None:
            continue
        target = loc + nrm * fit_offset
        d = (w - target).length
        if d > 1e-4:
            v.co = inv @ target
            worst = max(worst, d)
            moved += 1
    me.update()
    return moved, worst


def hide_markers(prefixes=("LM_", "AL_S_", "AL_T_", "MG_")):
    n = 0
    for o in bpy.data.objects:
        if o.name.startswith(prefixes) and not o.hide_get():
            o.hide_set(True); n += 1
    return n


def even_out(obj, target=0.5, rounds=3):
    """Give an OPEN surface roughly uniform edge length `target` mm without closing it.
    (Blender's voxel remesh would turn the sheet into a closed shell, which the build needs to stay open,
    so this splits long edges and collapses short ones instead.) Returns (faces before, faces after)."""
    me = obj.data
    before = len(me.polygons)
    bm = bmesh.new()
    bm.from_mesh(me)
    for _ in range(rounds):
        long_edges = [e for e in bm.edges if e.calc_length() > target * 1.35]
        if long_edges:
            bmesh.ops.subdivide_edges(bm, edges=long_edges, cuts=1, use_grid_fill=True)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=target * 0.45)
        bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
        bmesh.ops.beautify_fill(bm, faces=bm.faces, edges=bm.edges)
    bm.to_mesh(me)
    bm.free()
    me.update()
    for pg in me.polygons:
        pg.use_smooth = True
    return before, len(me.polygons)


def crop_object(p):
    """The object the crop tools work on: the explicit Crop target, else the Sculpt slot."""
    from ..ops.scene_helpers import is_construction, is_marker
    obj=p.crop_target or p.prosthesis_obj
    if obj and (is_marker(obj) or (p.crop_hide_helpers and is_construction(obj,p.id_data))):return None
    return obj


def _inside_clip(rv3d, pt):
    """True if a world point is inside the viewport's clipping region (Alt+B), or no clipping is active."""
    if rv3d is None or not getattr(rv3d, "use_clip_planes", False):
        return True
    for pl in rv3d.clip_planes:
        if pl[0] * pt.x + pl[1] * pt.y + pl[2] * pt.z + pl[3] < 0.0:
            return False
    return True


def ray_cast_clipped(context, origin, direction, obj=None, max_hits=24):
    """Ray cast that respects the Alt+B clipping region: surfaces that are clipped away in the view are
    skipped, so a click lands on what you actually see. obj=None casts against the whole scene.
    Returns (world_hit, world_normal, object) or (None, None, None)."""
    rv3d = getattr(context, "region_data", None)
    dg = context.evaluated_depsgraph_get()
    o = Vector(origin); d = Vector(direction).normalized()
    for _ in range(max_hits):
        if obj is None:
            ok, loc, nrm, idx, hit_obj, mat = context.scene.ray_cast(dg, o, d)
            if not ok:
                return None, None, None
            w, wn = loc, nrm
        else:
            inv = obj.matrix_world.inverted()
            ok, loc, nrm, idx = obj.ray_cast(inv @ o, (inv.to_3x3() @ d).normalized(), depsgraph=dg)
            if not ok:
                return None, None, None
            w, wn, hit_obj = obj.matrix_world @ loc, (obj.matrix_world.to_3x3() @ nrm).normalized(), obj
        if _inside_clip(rv3d, w):
            return w, wn, hit_obj
        o = w + d * 0.05                     # step past the clipped-away hit and keep going
    return None, None, None


def fill_internal_holes(obj, max_perimeter=None):
    """Make an open sheet hole-free: every boundary loop except the longest (the outer margin) is filled.
    A scan dropout at a canthus or a nostril, a torn edge from cropping — all become faces. Returns
    (holes filled, edges filled). max_perimeter (mm) limits which loops count as holes; None = all."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    edges = [e for e in bm.edges if e.is_boundary]
    if not edges:
        bm.free()
        return 0, 0
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
    if len(loops) <= 1:
        bm.free()
        return 0, 0
    loops.sort(key=lambda L: sum(e.calc_length() for e in L), reverse=True)
    holes = [L for L in loops[1:] if max_perimeter is None or sum(e.calc_length() for e in L) <= max_perimeter]
    inner = [e for L in holes for e in L]
    filled = 0
    if inner:
        try:
            res = bmesh.ops.holes_fill(bm, edges=inner, sides=0)
            filled = len(res.get("faces", []))
        except Exception:
            pass
        # holes_fill can leave large n-gons: triangulate them so later steps see plain triangles
        big = [f for f in bm.faces if len(f.verts) > 4]
        if big:
            bmesh.ops.triangulate(bm, faces=big)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(obj.data)
        obj.data.update()
    bm.free()
    return (len(holes) if filled else 0), len(inner)


def keep_main_body(obj):
    """Delete every connected piece of obj except the largest (by face count). Returns faces removed."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
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
    if len(islands) <= 1:
        bm.free()
        return 0
    islands.sort(key=len, reverse=True)
    doomed = [f for c in islands[1:] for f in c]
    n = len(doomed)
    bmesh.ops.delete(bm, geom=doomed, context='FACES')
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context='VERTS')
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return n
