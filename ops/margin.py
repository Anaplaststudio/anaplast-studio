"""Margin by points — click where the sculpt edge should run; the tool joins the points with
shortest paths over the surface and keeps everything inside. Exact, view-independent, repeatable."""
import heapq
import numpy as np
import bpy
import bmesh
from mathutils import Vector
from ..utils import mesh as mu
from .report import rep

MG_COLLECTION = "Anaplast_Margin"


def margin_points():
    col = bpy.data.collections.get(MG_COLLECTION)
    if not col:
        return []
    return sorted((o for o in col.objects if o.name.startswith("MG_")), key=lambda o: int(o.name.split("_")[-1]))


class ANAPLAST_OT_add_margin_point(bpy.types.Operator):
    """Add a margin point at the 3D cursor (Shift+Right-click on the surface first). Place them in order around the region to keep"""
    bl_idname = "anaplast.add_margin_point"
    bl_label = "Add Margin Point"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        col = mu.get_collection(context.scene, MG_COLLECTION)
        idx = len(margin_points()) + 1
        rgb = mu.PALETTE[0]
        o = mu.make_marker(f"MG_{idx}", context.scene.cursor.location.copy(), rgb, col, 'SPHERE', 0.8)
        o.show_name = False
        update_preview(context)
        return {'FINISHED'}


class ANAPLAST_OT_clear_margin_points(bpy.types.Operator):
    """Remove all margin points"""
    bl_idname = "anaplast.clear_margin_points"
    bl_label = "Clear Margin Points"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        for o in margin_points():
            mu.delete_object(o)
        old = bpy.data.objects.get("Margin_Preview")
        if old:
            mu.delete_object(old)
        return {'FINISHED'}


def catmull_rom_closed(P, spacing=1.0):
    """Dense samples of a closed Catmull-Rom spline through points P (n,3), roughly `spacing` mm apart."""
    P = np.asarray(P, dtype=float)
    n = len(P)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        seg = np.linalg.norm(p2 - p1)
        k = max(2, int(seg / spacing))
        for t in np.linspace(0.0, 1.0, k, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.array(out)


def update_preview(context):
    """Closed smooth curve through the margin points, projected onto the Sculpt surface."""
    p = context.scene.anaplast
    pts = [o.matrix_world.translation.copy() for o in margin_points()]
    old = bpy.data.objects.get("Margin_Preview")
    if len(pts) < 3 or mu.crop_object(p) is None:
        if old:
            mu.delete_object(old)
        return
    from mathutils.bvhtree import BVHTree
    dense = catmull_rom_closed([[q.x, q.y, q.z] for q in pts], spacing=1.0)
    obj = mu.crop_object(p)
    dg = context.evaluated_depsgraph_get()
    tree = BVHTree.FromObject(obj, dg)
    inv, mw = obj.matrix_world.inverted(), obj.matrix_world
    proj = []
    for q in dense:
        loc, nrm, idx, dist = tree.find_nearest(inv @ Vector(q))
        proj.append(mw @ loc if loc is not None else Vector(q))
    if old:
        mu.delete_object(old)
    cu = bpy.data.curves.new("Margin_Preview", type='CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = 0.25
    sp = cu.splines.new('POLY')
    sp.points.add(len(proj) - 1)
    for i, q in enumerate(proj):
        sp.points[i].co = (q.x, q.y, q.z, 1.0)
    sp.use_cyclic_u = True
    o = bpy.data.objects.new("Margin_Preview", cu)
    o.show_in_front = True
    o.hide_select = True
    cu.materials.append(mu.get_material(f"Anaplast_{mu.PALETTE[0]}", mu.PALETTE[0]))
    mu.get_collection(context.scene, MG_COLLECTION).objects.link(o)


def _dijkstra(adj, src, dst, pos):
    """Shortest path over mesh edges (A* with Euclidean heuristic). adj: dict v -> list of (w, length)."""
    goal = pos[dst]
    dist = {src: 0.0}
    prev = {}
    heap = [(np.linalg.norm(pos[src] - goal), 0.0, src)]
    while heap:
        f, d, u = heapq.heappop(heap)
        if u == dst:
            break
        if d > dist.get(u, float('inf')):
            continue
        for w, L in adj.get(u, ()):
            nd = d + L
            if nd < dist.get(w, float('inf')):
                dist[w] = nd
                prev[w] = u
                heapq.heappush(heap, (nd + np.linalg.norm(pos[w] - goal), nd, w))
    if dst not in dist:
        return None
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return path[::-1]


class ANAPLAST_OT_crop_to_margin(bpy.types.Operator):
    """Join the margin points with shortest paths over the Sculpt surface and keep only what lies inside that loop"""
    bl_idname = "anaplast.crop_to_margin"
    bl_label = "Crop to Margin Points"
    bl_options = {'REGISTER', 'UNDO'}

    flip_side: bpy.props.BoolProperty(name="Keep the other side", default=False,
                                      description="If the wrong side of the curve was kept, undo and tick this")

    @classmethod
    def poll(cls, context):
        return mu.crop_object(context.scene.anaplast) is not None and len(margin_points()) >= 3

    def execute(self, context):
        obj = mu.crop_object(context.scene.anaplast)
        pts = [o.matrix_world.translation.copy() for o in margin_points()]
        me = obj.data
        n = len(me.vertices)
        co = np.empty(n * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
        M = np.array(obj.matrix_world); W = co @ M[:3, :3].T + M[:3, 3]
        ne = len(me.edges)
        ed = np.empty(ne * 2, dtype=np.int64); me.edges.foreach_get("vertices", ed); ed = ed.reshape(-1, 2)
        # work only inside the loop's bounding box (+ margin): the interior of a closed loop lies within it
        P = np.array([[q.x, q.y, q.z] for q in pts])
        lo, hi = P.min(axis=0) - 15.0, P.max(axis=0) + 15.0
        inbox = ((W >= lo) & (W <= hi)).all(axis=1)
        sub = np.nonzero(inbox)[0]
        if len(sub) < 10:
            rep(self, {'ERROR'}, "Margin points are not on the Sculpt surface")
            return {'CANCELLED'}
        emask = inbox[ed[:, 0]] & inbox[ed[:, 1]]
        E = ed[emask]
        L = np.linalg.norm(W[E[:, 0]] - W[E[:, 1]], axis=1)
        adj = {}
        for (a, b), l in zip(E.tolist(), L.tolist()):
            adj.setdefault(a, []).append((b, l))
            adj.setdefault(b, []).append((a, l))
        # smooth margin: a closed spline through the points, sampled every ~1 mm; each sample becomes an anchor
        # so the surface path hugs the curve instead of cutting straight chords between clicks
        dense = catmull_rom_closed(P, spacing=1.0)
        anchors = []
        for q in dense:
            a = int(sub[np.argmin(np.linalg.norm(W[sub] - q, axis=1))])
            if not anchors or a != anchors[-1]:
                anchors.append(a)
        if anchors[0] == anchors[-1]:
            anchors.pop()
        loop_verts = set()
        for i in range(len(anchors)):
            a, b = anchors[i], anchors[(i + 1) % len(anchors)]
            path = _dijkstra(adj, a, b, W)
            if path is None:
                rep(self, {'ERROR'}, f"No surface path between margin points {i + 1} and {(i + 1) % len(anchors) + 1} — is the surface continuous there?")
                return {'CANCELLED'}
            loop_verts.update(path)
        # flood fill from the region centre, never stepping onto the loop
        centre = P.mean(axis=0)
        seed = int(sub[np.argmin(np.linalg.norm(W[sub] - centre, axis=1))])
        if seed in loop_verts:
            rep(self, {'ERROR'}, "Region centre lands on the margin — add more points or spread them out")
            return {'CANCELLED'}
        inside = {seed}
        stack = [seed]
        while stack:
            u = stack.pop()
            for w, _ in adj.get(u, ()):
                if w not in inside and w not in loop_verts:
                    inside.add(w); stack.append(w)
        if len(inside) > 0.9 * len(sub):
            rep(self, {'ERROR'}, "The loop did not close — the flood leaked out. Check the points go around the region in order")
            return {'CANCELLED'}
        keep = np.zeros(n, dtype=bool)
        keep[list(inside)] = True
        keep[list(loop_verts)] = True                  # the curve itself belongs to the kept patch
        before = len(me.polygons)
        # delete in place — replacing obj.data proved unreliable on large scans
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        doomed = [v for v in bm.verts if not keep[v.index]]
        if len(doomed) >= len(bm.verts):
            bm.free()
            self.report({'ERROR'}, "Nothing would be left — the curve did not enclose a region. Check the points go around it in order")
            return {'CANCELLED'}
        bmesh.ops.delete(bm, geom=doomed, context='VERTS')
        removed, holes = mu.clean_margin(bm, smooth_iters=2, smooth_factor=0.4)
        bm.to_mesh(me)
        bm.free()
        me.update()
        after = len(me.polygons)
        if after == before:
            self.report({'ERROR'}, "The mesh did not change — is the Sculpt slot pointing at the object you clicked on?")
            return {'CANCELLED'}
        for o in margin_points():
            o.hide_set(True)
        pv = bpy.data.objects.get("Margin_Preview")
        if pv:
            pv.hide_set(True)
        p_ = context.scene.anaplast
        p_.source_prosthesis = obj
        p_.prosthesis_obj = obj
        if p_.auto_even:
            b0, b1 = mu.even_out(obj, p_.source_edge)
            after = len(obj.data.polygons)
        rep(self, {'INFO'}, f"Cropped {obj.name}: {before:,} → {after:,} faces, smooth curve through {len(pts)} points")
        return {'FINISHED'}


_classes = (ANAPLAST_OT_add_margin_point, ANAPLAST_OT_clear_margin_points, ANAPLAST_OT_crop_to_margin)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)


class ANAPLAST_OT_place_margin_points(bpy.types.Operator):
    """Click directly on the surface to place margin points. Backspace removes the last, Enter or right-click finishes, Esc cancels"""
    bl_idname = "anaplast.place_margin_points"
    bl_label = "Place Margin Points (click on surface)"
    bl_options = {'REGISTER', 'UNDO'}

    _added = 0

    @classmethod
    def poll(cls, context):
        return mu.crop_object(context.scene.anaplast) is not None and context.area is not None and context.area.type == 'VIEW_3D'

    _hidden = None

    def invoke(self, context, event):
        self._added = 0
        p = context.scene.anaplast
        obj = mu.crop_object(p)
        obj.hide_set(False)
        # the face scan sits exactly where the mirror is: hide it so clicks land on the mirror
        self._hidden = None
        if p.face_scan_obj is not None and p.face_scan_obj is not obj and not p.face_scan_obj.hide_get():
            p.face_scan_obj.hide_set(True); self._hidden = p.face_scan_obj
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set(f"Margin on {obj.name}: LEFT-CLICK to add a point · BACKSPACE removes last · ENTER / RIGHT-CLICK finish · ESC cancel")
        context.window.cursor_modal_set('CROSSHAIR')
        return {'RUNNING_MODAL'}

    def _hit(self, context, event):
        from bpy_extras import view3d_utils
        region, rv3d = context.region, context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        obj = mu.crop_object(context.scene.anaplast)
        w, wn, o = mu.ray_cast_clipped(context, origin, direction, obj=obj)
        return w

    def modal(self, context, event):
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'NUMPAD_PERIOD', 'TRACKPADPAN', 'TRACKPADZOOM'} or (event.type == 'LEFTMOUSE' and event.alt):
            return {'PASS_THROUGH'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            hit = self._hit(context, event)
            if hit is None:
                return {'RUNNING_MODAL'}
            context.scene.cursor.location = hit
            bpy.ops.anaplast.add_margin_point()
            self._added += 1
            return {'RUNNING_MODAL'}
        if event.type == 'BACK_SPACE' and event.value == 'PRESS':
            pts = margin_points()
            if pts and self._added > 0:
                mu.delete_object(pts[-1]); self._added -= 1
                update_preview(context)
            return {'RUNNING_MODAL'}
        if event.type in {'RET', 'NUMPAD_ENTER', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self._finish(context)
            n = len(margin_points())
            rep(self, {'INFO'}, f"{n} margin points placed — now Crop along Curve" if n >= 3 else f"{n} points (need at least 3)")
            return {'FINISHED'}
        if event.type == 'ESC':
            for o in margin_points()[len(margin_points()) - self._added:]:
                mu.delete_object(o)
            update_preview(context)
            self._finish(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        context.area.header_text_set(None)
        context.window.cursor_modal_restore()
        if self._hidden is not None:
            self._hidden.hide_set(False)


_prev_reg2, _prev_unreg2 = register, unregister


def register():
    _prev_reg2()
    bpy.utils.register_class(ANAPLAST_OT_place_margin_points)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_place_margin_points)
    _prev_unreg2()
