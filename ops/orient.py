"""Tool 1 — orient the case into an anatomical frame, then mirror across the midline.

Frame convention (right-handed):  +X = patient's right,  +Y = anterior,  +Z = superior.
Horizontal = soft-tissue Frankfort plane (tragion R, tragion L, orbitale) when orbitale is placed;
otherwise the subnasale→nasion line stands in for vertical.
Origin = nasion, shifted along X onto the best-fit midsagittal plane.
"""
import bpy
import bmesh
from mathutils import Vector, Matrix
from ..utils import mesh as mu
from .report import rep

LM_COLLECTION = "Anaplast_Landmarks"
LANDMARKS = [
    ("NASION", "Nasion", "Midline: bridge of the nose"),
    ("SUBNASALE", "Subnasale", "Midline: base of the columella"),
    ("TRAGION_R", "Tragion R", "Patient's RIGHT tragus"),
    ("TRAGION_L", "Tragion L", "Patient's LEFT tragus"),
    ("ORBITALE_R", "Orbitale R", "Patient's RIGHT infraorbital rim, lowest point (for Frankfort)"),
    ("ORBITALE_L", "Orbitale L", "Patient's LEFT infraorbital rim, lowest point (for Frankfort)"),
    ("EXOCANTHION_R", "Exocanthion R", "Patient's RIGHT outer eye corner"),
    ("EXOCANTHION_L", "Exocanthion L", "Patient's LEFT outer eye corner"),
]
LM_COLORS = {"NASION": 0, "SUBNASALE": 0, "TRAGION_R": 1, "TRAGION_L": 2,
             "EXOCANTHION_R": 3, "EXOCANTHION_L": 4, "ORBITALE_R": 5, "ORBITALE_L": 6}
MIDLINE = ("NASION", "SUBNASALE")
BILATERAL = (("TRAGION_R", "TRAGION_L"), ("EXOCANTHION_R", "EXOCANTHION_L"))


def lm_name(key):
    return f"LM_{key}"


def get_landmark(key):
    o = bpy.context.scene.objects.get(lm_name(key))
    return o.matrix_world.translation.copy() if o else None


def case_objects(scene):
    """Everything the frame transform applies to: scan, cast, sculpt, landmarks, mold objects."""
    p = scene.anaplast
    objs = {o for o in (p.prosthesis_obj, p.cast_obj, p.face_scan_obj, p.cbct_obj) if o is not None}
    # Scene organization moves landmarks out of the legacy collection. They
    # must still follow the scan when applying the anatomical frame.
    objs.update(o for o in scene.objects if o.name in {lm_name(key) for key,_,_ in LANDMARKS})
    for name in (LM_COLLECTION, mu.MOLD_COLLECTION):
        col = bpy.data.collections.get(name)
        if col:
            objs.update(col.all_objects)
    return objs


def build_frame():
    """Return (matrix_world_of_frame, message) or (None, error)."""
    n, s = get_landmark("NASION"), get_landmark("SUBNASALE")
    if n is None or s is None:
        return None, "Need Nasion and Subnasale"
    pair = None
    for r_key, l_key in BILATERAL:
        r, l = get_landmark(r_key), get_landmark(l_key)
        if r is not None and l is not None:
            pair = (r, l)
            break
    if pair is None:
        return None, "Need a bilateral pair (Tragion R+L or Exocanthion R+L)"
    r, l = pair

    x = (r - l).normalized()                       # patient's right
    orb = [pt for pt in (get_landmark("ORBITALE_R"), get_landmark("ORBITALE_L")) if pt is not None]
    tr_r, tr_l = get_landmark("TRAGION_R"), get_landmark("TRAGION_L")
    if orb and tr_r is not None and tr_l is not None:
        # Frankfort horizontal: plane through both tragions and orbitale -> its normal is Z
        o = sum(orb, Vector()) / len(orb)
        z = (tr_r - tr_l).cross(o - (tr_r + tr_l) * 0.5).normalized()
        if z.dot(n - (tr_r + tr_l) * 0.5) < 0:
            z = -z
        x = (x - z * x.dot(z)).normalized()
        mode = "Frankfort"
    else:
        z = (n - s)
        z = (z - x * z.dot(x)).normalized()        # superior, orthogonal to X
        mode = "nasion-subnasale vertical"
    y = z.cross(x).normalized()                    # anterior
    origin = n.copy()
    # slide origin along X so the midsagittal plane (X=0) is the best fit through midline points
    mid_x = sum(((pt - origin).dot(x) for pt in (n, s)), 0.0) / 2.0
    origin += x * mid_x
    frame = Matrix((
        (x.x, y.x, z.x, origin.x),
        (x.y, y.y, z.y, origin.y),
        (x.z, y.z, z.z, origin.z),
        (0.0, 0.0, 0.0, 1.0),
    ))
    return frame, mode


def repair_legacy_landmark_frame(scene):
    """Recover pre-orientation markers only when the scan confirms the transform."""
    p=scene.anaplast;scan=p.face_scan_obj
    if not p.oriented or not scan or scan.type!='MESH':return False
    markers=[scene.objects.get(lm_name(key)) for key,_,_ in LANDMARKS]
    markers=[o for o in markers if o]
    if len(markers)<4:return False
    frame,_=build_frame()
    if frame is None:return False
    from mathutils.kdtree import KDTree
    from statistics import median
    tree=KDTree(len(scan.data.vertices))
    for vertex in scan.data.vertices:tree.insert(scan.matrix_world@vertex.co,vertex.index)
    tree.balance()
    before=[tree.find(o.matrix_world.translation)[2] for o in markers]
    if max(before)<2.:return False
    inverse=frame.inverted()
    proposed=[inverse@o.matrix_world for o in markers]
    after=[tree.find(matrix.translation)[2] for matrix in proposed]
    # A close match across every surviving landmark is required. Never snap an
    # individual marker to an arbitrary nearest point on the skin.
    if median(before)<5. or max(after)>2. or not all(b<a for a,b in zip(before,after)):
        raise RuntimeError('Facial landmarks do not match the aligned scan. Re-place Nasion and the facial landmarks before positioning the torch')
    for obj,matrix in zip(markers,proposed):
        obj['landmark_before_frame_repair']=[value for row in obj.matrix_world for value in row]
        obj.matrix_world=matrix;obj['landmark_frame_repaired']=True
    return True


class ANAPLAST_OT_add_landmark(bpy.types.Operator):
    """Place a landmark at the 3D cursor (Shift+Right-click on the scan to set the cursor first)"""
    bl_idname = "anaplast.add_landmark"
    bl_label = "Add Landmark"
    bl_options = {'REGISTER', 'UNDO'}

    key: bpy.props.EnumProperty(name="Landmark", items=LANDMARKS)

    def execute(self, context):
        col = mu.get_collection(context.scene, LM_COLLECTION)
        name = lm_name(self.key)
        rgb = mu.PALETTE[LM_COLORS.get(self.key, 7)]
        mu.make_marker(name, context.scene.cursor.location.copy(), rgb, col, 'SPHERE', 1.5)
        from .scene_helpers import marker_visibility
        marker_visibility(context.scene)
        rep(self, {'INFO'}, f"{name} placed")
        return {'FINISHED'}


class ANAPLAST_OT_place_landmarks(bpy.types.Operator):
    """Click the landmarks straight on the scan, in order: Nasion, Subnasale, Tragion R, Tragion L, Orbitale R, Orbitale L.
Tab skips unavailable landmarks; Backspace redoes the last; Enter finishes"""
    bl_idname = "anaplast.place_landmarks"
    bl_label = "Place Landmarks (click on the scan)"
    bl_options = {'REGISTER', 'UNDO'}

    ORDER = [key for key, _, _ in LANDMARKS]
    _i = 0

    @classmethod
    def poll(cls, context):
        p = context.scene.anaplast
        return (p.face_scan_obj or p.cast_obj) is not None and context.area is not None and context.area.type == 'VIEW_3D'

    def invoke(self, context, event):
        self._i = 0
        self._placed=set(); self._skipped=set()
        context.scene.anaplast.landmarks_placing=True
        from .scene_helpers import marker_visibility
        marker_visibility(context.scene)
        context.window_manager.modal_handler_add(self)
        context.window.cursor_modal_set('CROSSHAIR')
        self._header(context)
        return {'RUNNING_MODAL'}

    def _header(self, context):
        if self._i < len(self.ORDER):
            label = dict((k, n) for k, n, _ in LANDMARKS)[self.ORDER[self._i]]
            context.area.header_text_set(f"Click {label}   ({self._i + 1} of {len(self.ORDER)})   ·   TAB skip · BACKSPACE back · ENTER finish")
        else:
            context.area.header_text_set("Landmark sequence complete — ENTER to finish")

    def modal(self, context, event):
        p = context.scene.anaplast
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM', 'NUMPAD_PERIOD'} or (event.type == 'LEFTMOUSE' and event.alt):
            return {'PASS_THROUGH'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS' and self._i < len(self.ORDER):
            from bpy_extras import view3d_utils
            obj = p.face_scan_obj or p.cast_obj
            region, rv3d = context.region, context.region_data
            coord = (event.mouse_region_x, event.mouse_region_y)
            origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
            direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
            w, wn, o = mu.ray_cast_clipped(context, origin, direction, obj=obj)
            if w is None:
                return {'RUNNING_MODAL'}
            context.scene.cursor.location = w
            bpy.ops.anaplast.add_landmark(key=self.ORDER[self._i])
            self._placed.add(self.ORDER[self._i])
            self._i += 1
            self._header(context)
            return {'RUNNING_MODAL'}
        if event.type == 'TAB' and event.value == 'PRESS' and self._i < len(self.ORDER):
            self._skipped.add(self.ORDER[self._i])
            self._i += 1
            self._header(context)
            return {'RUNNING_MODAL'}
        if event.type == 'BACK_SPACE' and event.value == 'PRESS' and self._i > 0:
            self._i -= 1
            o = bpy.data.objects.get(lm_name(self.ORDER[self._i]))
            key=self.ORDER[self._i]
            if o and key in self._placed:
                mu.delete_object(o)
            self._placed.discard(key); self._skipped.discard(key)
            self._header(context)
            return {'RUNNING_MODAL'}
        if event.type in {'RET', 'NUMPAD_ENTER', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self._finish(context)
            rep(self, {'INFO'}, f"{len(self._placed)} landmarks placed; {len(self._skipped)} skipped — Orient Case when enough landmarks are available")
            return {'FINISHED'}
        if event.type == 'ESC':
            self._finish(context)
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        context.scene.anaplast.landmarks_placing=False
        from .scene_helpers import marker_visibility
        marker_visibility(context.scene)
        context.area.header_text_set(None)
        context.window.cursor_modal_restore()


class ANAPLAST_OT_orient_case(bpy.types.Operator):
    """Move every case object so the landmark frame becomes world XYZ (X right, Y anterior, Z up)"""
    bl_idname = "anaplast.orient_case"
    bl_label = "Orient Case to Frame"
    bl_options = {'REGISTER', 'UNDO'}

    bake: bpy.props.BoolProperty(name="Bake into mesh", default=True,
                                 description="Apply the transform to the mesh data so object transforms read as zero")

    def execute(self, context):
        frame, msg = build_frame()
        if frame is None:
            rep(self, {'ERROR'}, msg)
            return {'CANCELLED'}
        inv = frame.inverted()
        for o in case_objects(context.scene):
            o.matrix_world = inv @ o.matrix_world
            if self.bake and o.type == 'MESH' and o.data.users == 1 and not o.name.startswith("LM_"):
                o.data.transform(o.matrix_world)
                o.matrix_world = Matrix.Identity(4)
        context.scene.anaplast.oriented = True
        mu.hide_markers(("LM_",))
        mu.frame_view(context, context.scene.anaplast.face_scan_obj or context.scene.anaplast.cast_obj)
        rep(self, {'INFO'}, f"Oriented — horizontal: {msg}; X right, Y anterior, Z up, midline X 0")
        return {'FINISHED'}


class ANAPLAST_OT_mirror_contralateral(bpy.types.Operator):
    """Mirror the face scan across the midsagittal plane (X = 0) to give a contralateral source"""
    bl_idname = "anaplast.mirror_contralateral"
    bl_label = "Mirror Across Midline"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        p = context.scene.anaplast
        return p.face_scan_obj is not None and p.oriented

    def execute(self, context):
        p = context.scene.anaplast
        src = p.face_scan_obj
        col = mu.get_collection(context.scene)
        name = f"Mirror_{src.name}"
        old = bpy.data.objects.get(name)
        if old:
            mu.delete_object(old)
        m = mu.duplicate_object(src, name, col)
        m.data.transform(Matrix.Diagonal((-1.0, 1.0, 1.0, 1.0)))
        m.data.flip_normals()
        m.data.update()
        m["anaplast_part"] = "MIRROR"
        p.crop_target = m
        rep(self, {'INFO'}, f"{name} created as a reference — crop a region in Sculpt & Texture to create the Sculpt")
        return {'FINISHED'}


FRAME_COLLECTION = "Anaplast_Frame"


def _text(name, body, location, size, col, rgb):
    old = bpy.data.objects.get(name)
    if old:
        bpy.data.objects.remove(old)
    cu = bpy.data.curves.new(name, type='FONT')
    cu.body = body
    cu.size = size
    cu.align_x = 'CENTER'
    cu.align_y = 'CENTER'
    o = bpy.data.objects.new(name, cu)
    o.location = location
    o.show_in_front = True
    o.color = (*rgb, 1.0)
    cu.materials.append(mu.get_material(f"Anaplast_{rgb}", rgb))
    col.objects.link(o)
    return o


class ANAPLAST_OT_frame_case(bpy.types.Operator):
    """Centre the viewport on the face scan (or cast / sculpt)"""
    bl_idname = "anaplast.frame_case"
    bl_label = "Centre View on Case"

    def execute(self, context):
        p = context.scene.anaplast
        obj = p.face_scan_obj or p.cast_obj or p.prosthesis_obj
        if obj is None:
            rep(self, {'ERROR'}, "Nothing imported yet")
            return {'CANCELLED'}
        from .viewport_tools import fit,case_targets
        try:fit(context,case_targets(context))
        except RuntimeError as e:rep(self,{"ERROR"},str(e));return {"CANCELLED"}
        return {'FINISHED'}


_classes = (ANAPLAST_OT_place_landmarks, ANAPLAST_OT_add_landmark, ANAPLAST_OT_orient_case, ANAPLAST_OT_mirror_contralateral,
            ANAPLAST_OT_frame_case)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)


# ------------------------------------------------------------------ automatic symmetry plane
def _decimated_bvh(scan, max_faces=200000):
    """BVH of a temporary decimated copy of the scan (world space) — enough for a symmetry fit, cheap on memory."""
    from mathutils.bvhtree import BVHTree
    tmp = mu.duplicate_object(scan, "anaplast_sym_tmp", scan.users_collection[0])
    nf = len(tmp.data.polygons)
    if nf > max_faces:
        ratio = max_faces / nf
        def dec(m):
            m.decimate_type = 'COLLAPSE'; m.ratio = ratio; m.use_collapse_triangulate = True
        mu.apply_modifier(tmp, 'DECIMATE', dec)
    bm = bmesh.new(); bm.from_mesh(tmp.data); bm.transform(tmp.matrix_world)
    tree = BVHTree.FromBMesh(bm)
    import numpy as np
    pts = np.array([v.co[:] for v in bm.verts])
    bm.free()
    mu.delete_object(tmp)
    return tree, pts


def refine_symmetry_plane(scan, n0, c0, iterations=6, icp_iters=12, max_dist=4.0, samples=8000):
    """Refine a symmetry plane (normal n0 through point c0) using the scan itself:
    reflect the scan across the plane, rigidly ICP it back onto the original (defect regions drop out
    through the distance gate), and read the plane off the composite reflection. Returns (n, c, rms)."""
    import numpy as np
    from mathutils.bvhtree import BVHTree
    from .align import kabsch
    tree, pts = _decimated_bvh(scan)
    rng = np.random.default_rng(0)
    src = pts[rng.choice(len(pts), min(samples, len(pts)), replace=False)]
    n = np.array(n0, dtype=float); n /= np.linalg.norm(n)
    c = np.array(c0, dtype=float)
    rms = float('nan')
    for _ in range(iterations):
        Mlin = np.eye(3) - 2 * np.outer(n, n)
        refl = (src - c) @ Mlin.T + c                       # reflected samples
        R, t = np.eye(3), np.zeros(3)
        cur = refl.copy()
        for _ in range(icp_iters):
            Q, keep = [], []
            for i, q in enumerate(cur):
                loc, nrm, idx, dist = tree.find_nearest(Vector(q))
                if loc is not None and dist <= max_dist:
                    Q.append(loc[:]); keep.append(i)
            if len(keep) < 100:
                break
            P, Q = cur[keep], np.array(Q)
            Ri, ti = kabsch(P, Q)
            cur = cur @ Ri.T + ti
            R, t = Ri @ R, Ri @ t + ti
            rms = float(np.sqrt(((cur[keep] - Q) ** 2).sum(axis=1).mean()))
        # composite S(p) = R(Mlin(p - c) + c) + t is (nearly) a reflection: normal = eigenvector at -1
        A = R @ Mlin
        w, V = np.linalg.eig(A)
        k = int(np.argmin(np.abs(w + 1)))
        n_new = np.real(V[:, k]); n_new /= np.linalg.norm(n_new)
        if n_new @ n < 0:
            n_new = -n_new
        # a fixed point of S lies on the plane: solve (I - A) p = R(c - Mlin c) + t in least squares
        b = R @ (c - Mlin @ c) + t
        p_fix = np.linalg.lstsq(np.eye(3) - A, b, rcond=None)[0]
        # keep the plane point near the original origin: project c0 onto the new plane
        c_new = c - ((c - p_fix) @ n_new) * n_new
        if abs(n_new @ n) > 0.99999 and np.linalg.norm(c_new - c) < 0.01:
            n, c = n_new, c_new
            break
        n, c = n_new, c_new
    return n, c, rms


class ANAPLAST_OT_auto_symmetry(bpy.types.Operator):
    """Refine the midline automatically from the scan's own left/right symmetry (landmarks are only the starting guess). Re-orients the case and rebuilds the mirror"""
    bl_idname = "anaplast.auto_symmetry"
    bl_label = "Auto-Refine Symmetry Plane"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        p = context.scene.anaplast
        return p.face_scan_obj is not None and p.oriented

    def execute(self, context):
        import numpy as np, math
        p = context.scene.anaplast
        scan = p.face_scan_obj
        # current frame: midsagittal = X 0 through the origin
        n, c, rms = refine_symmetry_plane(scan, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        angle = math.degrees(math.acos(max(-1.0, min(1.0, float(n @ np.array([1.0, 0.0, 0.0]))))))
        offset = float(c @ n)
        # new frame: X = refined normal, Z = old Z made orthogonal, origin = old origin projected onto the plane
        x = Vector(n.tolist()).normalized()
        z = Vector((0, 0, 1)); z = (z - x * z.dot(x)).normalized()
        y = z.cross(x).normalized()
        origin = Vector((0, 0, 0)) - x * float((np.zeros(3) - c) @ n)
        frame = Matrix((
            (x.x, y.x, z.x, origin.x),
            (x.y, y.y, z.y, origin.y),
            (x.z, y.z, z.z, origin.z),
            (0.0, 0.0, 0.0, 1.0),
        ))
        inv = frame.inverted()
        for o in case_objects(context.scene):
            o.matrix_world = inv @ o.matrix_world
            if o.type == 'MESH' and o.data.users == 1 and not o.name.startswith("LM_"):
                o.data.transform(o.matrix_world)
                o.matrix_world = Matrix.Identity(4)
        # rebuild the mirror if there is one
        old = bpy.data.objects.get(f"Mirror_{scan.name}")
        if old is not None:
            mu.delete_object(old)
            bpy.ops.anaplast.mirror_contralateral()
        p.symmetry_rms = rms
        rep(self, {'INFO'}, f"Midline refined: rotated {angle:.2f}°, shifted {offset:.2f} mm from the landmark plane; symmetric skin RMS {rms:.2f} mm. Mirror rebuilt")
        return {'FINISHED'}


_auto = (ANAPLAST_OT_auto_symmetry,)
_prev_reg, _prev_unreg = register, unregister


def register():
    _prev_reg()
    for c in _auto:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_auto):
        bpy.utils.unregister_class(c)
    _prev_unreg()
