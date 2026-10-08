"""Statistical shape model (SSM) — train from a folder of same-topology meshes, fit to a patient.

Synthetic-first: meshes from a parametric generator (MPFB / MakeHuman) share one topology, so
vertex correspondence is free and no non-rigid registration is needed. Real scans registered
onto the same template can be dropped into the same folder later.

Model file (.npz): mean (n,3) mm · components (k,n,3) · variances (k,) · faces (m,3)
                   · region (indices into the template) · template_verts (int)
"""
import os
import glob
import json
import numpy as np
import bpy
import bmesh
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from ..utils import mesh as mu
from .report import rep
from .align import kabsch, to_matrix

SSM_COLLECTION = "Anaplast_SSM"


def mpfb_candidates():
    """MPFB can be installed as a legacy add-on ('mpfb') or as an extension
    ('bl_ext.<repo>.mpfb'), and the repo name varies. Try every module that looks like MPFB."""
    import sys
    names = []
    for mod in list(sys.modules):
        if mod.endswith(".mpfb") or mod == "mpfb":
            names.append(mod)
    for addon in bpy.context.preferences.addons.keys():
        if addon.endswith("mpfb") and addon not in names:
            names.append(addon)
    for repo in ("user_default", "blender_org", "extensions"):
        cand = f"bl_ext.{repo}.mpfb"
        if cand not in names:
            names.append(cand)
    if "mpfb" not in names:
        names.append("mpfb")
    return names


# ---------------------------------------------------------------- mesh file readers (keep vertex order!)
def read_obj(path):
    verts, faces = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                for i in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[i], idx[i + 1]))
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


def read_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        return read_obj(path)
    if ext == ".ply":
        before = set(bpy.data.objects)
        bpy.ops.wm.ply_import(filepath=path, use_scene_unit=False, global_scale=1.0)
        new = [o for o in bpy.data.objects if o not in before and o.type == 'MESH']
        if not new:
            raise ValueError("no mesh in " + path)
        o = new[0]
        me = o.data
        v = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", v)
        me.calc_loop_triangles()
        f = np.array([lt.vertices[:] for lt in me.loop_triangles], dtype=np.int64)
        mu.delete_object(o)
        return v.reshape(-1, 3), f
    raise ValueError("Use OBJ or PLY (STL loses vertex order): " + path)


def largest_island(verts, faces):
    """Keep the biggest connected piece — MakeHuman helper geometry comes through as separate strips."""
    import numpy as np
    f = np.asarray(faces, dtype=np.int64)
    n = len(verts)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for tri in f:
        a, b, c = (find(int(x)) for x in tri)
        if a != b:
            parent[b] = a
        if a != c:
            parent[c] = a
    roots = np.array([find(int(i)) for i in range(n)])
    used = np.unique(f)
    if len(used) == 0:
        return verts, faces
    vals, counts = np.unique(roots[used], return_counts=True)
    keep_root = vals[counts.argmax()]
    keep = roots == keep_root
    fk = keep[f].all(axis=1)
    f = f[fk]
    u = np.unique(f)
    remap = -np.ones(n, dtype=np.int64)
    remap[u] = np.arange(len(u))
    return np.asarray(verts)[u], remap[f]


def crop_head(verts, faces, face_only=False):
    """Keep the head (above the neck), or just the face. MakeHuman meshes share one topology, so the
    same vertex selection applies to every generated head — correspondence is preserved."""
    import numpy as np
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    top = v[:, 2].max()
    span = top - v[:, 2].min()
    neck = top - 0.16 * span                      # head is roughly the top 1/6 of a standing figure
    keep = v[:, 2] >= neck
    if face_only and keep.any():
        head = v[keep]
        cy = float(np.median(head[:, 1]))
        keep = keep & (v[:, 1] <= cy)             # MakeHuman figures face -Y: the face is the LOW-y half
    fk = keep[f].all(axis=1)
    if not fk.any():                              # crop would empty the mesh: keep the head instead
        keep = v[:, 2] >= neck
        fk = keep[f].all(axis=1)
        if not fk.any():
            return v, f
    f = f[fk]
    used = np.unique(f)
    remap = -np.ones(len(v), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return v[used], remap[f]


def write_obj(path, verts, faces):
    with open(path, "w") as f:
        f.write("# Anaplast Studio SSM sample\n")
        for v in verts:
            f.write("v %.5f %.5f %.5f\n" % tuple(v))
        for t in faces:
            f.write("f %d %d %d\n" % (t[0] + 1, t[1] + 1, t[2] + 1))


# ---------------------------------------------------------------- core maths
def procrustes_align(X, ref):
    """Rigidly align X (n,3) onto ref (n,3). No scaling: sculpt size matters."""
    R, t = kabsch(X, ref)
    return X @ R.T + t


def train_pca(shapes, keep_variance=0.98, max_modes=60):
    """shapes: (N, n, 3) already aligned. Returns mean, components (k,n,3), variances (k,)."""
    N, n, _ = shapes.shape
    X = shapes.reshape(N, -1)
    mean = X.mean(axis=0)
    D = X - mean
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    var = (S ** 2) / max(1, N - 1)
    cum = np.cumsum(var) / var.sum()
    k = int(np.searchsorted(cum, keep_variance) + 1)
    k = max(1, min(k, max_modes, len(var)))
    return mean.reshape(n, 3), Vt[:k].reshape(k, n, 3), var[:k]


def instance(mean, comps, b):
    return mean + np.tensordot(b, comps, axes=(0, 0))


def solve_coeffs(mean, comps, var, targets, mask, lam, weights=None):
    """Weighted least squares for b: min Σ w |mean+Σ b φ - target|² + lam Σ b²/var."""
    k = comps.shape[0]
    idx = np.nonzero(mask)[0]
    w = np.ones(len(idx)) if weights is None else weights[idx]
    sw = np.sqrt(w)[:, None]
    A = (comps[:, idx, :] * sw[None]).reshape(k, -1)          # (k, 3m)
    r = ((targets[idx] - mean[idx]) * sw).ravel()             # (3m,)
    G = A @ A.T + lam * np.diag(1.0 / var)
    return np.linalg.solve(G, A @ r)


def weighted_kabsch(P, Q, w):
    w = w / w.sum()
    cP, cQ = (P * w[:, None]).sum(0), (Q * w[:, None]).sum(0)
    H = ((P - cP) * w[:, None]).T @ (Q - cQ)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cQ - R @ cP


def model_to_object(name, verts, faces, col):
    old = bpy.data.objects.get(name)
    if old:
        mu.delete_object(old)
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(int(i) for i in f) for f in faces])
    me.update()
    o = bpy.data.objects.new(name, me)
    col.objects.link(o)
    return o


def boundary_edge_segments(obj):
    """Per-face list of boundary edge segments (world space) for faces that touch an open edge."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    mw = obj.matrix_world
    bmesh.ops.triangulate(bm, faces=bm.faces)          # match loop-triangle indexing used by the BVH
    segs = {}
    for f in bm.faces:
        for e in f.edges:
            if e.is_boundary:
                segs.setdefault(f.index, []).append((mw @ e.verts[0].co, mw @ e.verts[1].co))
    bm.free()
    return segs


def on_boundary(loc_world, tri_index, segs, eps=0.05):
    """True if the nearest point sits on an open edge of the scan (i.e. the model vertex is over a hole)."""
    if tri_index not in segs:
        return False
    for a, b in segs[tri_index]:
        ab = b - a
        t = max(0.0, min(1.0, (loc_world - a).dot(ab) / max(ab.length_squared, 1e-12)))
        if (loc_world - (a + ab * t)).length < eps:
            return True
    return False


def vertex_normals(verts, faces):
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    nrm = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(nrm, 1e-12)


def object_verts_world(obj):
    me = obj.data
    v = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", v); v = v.reshape(-1, 3)
    mw = np.array(obj.matrix_world)
    return v @ mw[:3, :3].T + mw[:3, 3]


# ---------------------------------------------------------------- operators
class ANAPLAST_OT_ssm_set_region(bpy.types.Operator):
    """In Edit Mode on ONE dataset mesh, select the region to model (lasso), then click this to record it"""
    bl_idname = "anaplast.ssm_set_region"
    bl_label = "Set Region from Selection"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and context.edit_object is not None

    def execute(self, context):
        obj = context.edit_object
        bm = bmesh.from_edit_mesh(obj.data)
        sel = sorted(v.index for v in bm.verts if v.select)
        if len(sel) < 10:
            rep(self, {'ERROR'}, "Select the region first (lasso in Edit Mode)")
            return {'CANCELLED'}
        p = context.scene.anaplast
        p.ssm_region = json.dumps({"template_verts": len(bm.verts), "region": sel})
        bpy.ops.object.mode_set(mode='OBJECT')
        rep(self, {'INFO'}, f"Region recorded: {len(sel)} of {len(bm.verts)} vertices")
        return {'FINISHED'}


class ANAPLAST_OT_ssm_train(bpy.types.Operator):
    """Build a shape model from every OBJ/PLY in the dataset folder (all must share one topology)"""
    bl_idname = "anaplast.ssm_train"
    bl_label = "Train Model"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        p = context.scene.anaplast
        return bool(p.ssm_dataset_dir) and bool(p.ssm_region)

    def execute(self, context):
        p = context.scene.anaplast
        folder = bpy.path.abspath(p.ssm_dataset_dir)
        files = sorted(glob.glob(os.path.join(folder, "*.obj")) + glob.glob(os.path.join(folder, "*.ply")))
        if len(files) < 3:
            rep(self, {'ERROR'}, f"Need at least 3 OBJ/PLY meshes in {folder} (found {len(files)})")
            return {'CANCELLED'}
        reg = json.loads(p.ssm_region)
        region = np.array(reg["region"], dtype=np.int64)
        shapes, faces_all, bad = [], None, []
        for fpath in files:
            try:
                v, f = read_mesh(fpath)
            except Exception as e:
                bad.append(f"{os.path.basename(fpath)}: {e}")
                continue
            if len(v) != reg["template_verts"]:
                bad.append(f"{os.path.basename(fpath)}: {len(v)} verts ≠ template {reg['template_verts']}")
                continue
            if faces_all is None:
                faces_all = f
            shapes.append(v[region])
        if len(shapes) < 3:
            rep(self, {'ERROR'}, "Fewer than 3 usable meshes. " + " | ".join(bad[:3]))
            return {'CANCELLED'}
        shapes = np.array(shapes)
        # unit guess: MPFB exports metres unless told otherwise; the model is stored in mm
        extent = np.ptp(shapes[0], axis=0).max()
        scale = 1000.0 if extent < 1.0 else (100.0 if extent < 10.0 else 1.0)
        shapes *= scale
        ref = shapes[0]
        for _ in range(3):                                      # generalised Procrustes, rigid only
            aligned = np.array([procrustes_align(s, ref) for s in shapes])
            ref = aligned.mean(axis=0)
        mean, comps, var = train_pca(aligned, keep_variance=p.ssm_keep_variance / 100.0)
        # region faces, re-indexed
        lookup = -np.ones(reg["template_verts"], dtype=np.int64)
        lookup[region] = np.arange(len(region))
        rf = lookup[faces_all]
        rf = rf[(rf >= 0).all(axis=1)]
        out = bpy.path.abspath(p.ssm_model_path) or os.path.join(folder, "ssm_model.npz")
        np.savez_compressed(out, mean=mean, components=comps, variances=var, faces=rf,
                            region=region, template_verts=reg["template_verts"], scale=scale,
                            n_samples=len(shapes))
        p.ssm_model_path = out
        note = f"; skipped {len(bad)}" if bad else ""
        rep(self, {'INFO'}, f"Model: {len(shapes)} samples → {comps.shape[0]} modes ({p.ssm_keep_variance:.0f}% variance), "
                              f"{len(region)} verts, saved {os.path.basename(out)}{note}")
        return {'FINISHED'}


def load_model(path):
    if not path.lower().endswith(".npz"):
        raise ValueError("the Model field must point to a trained .npz model (from Train Model), not a mesh file")
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


class ANAPLAST_OT_ssm_place_mean(bpy.types.Operator):
    """Create the model's mean shape as SSM_Mean and set it as Moving in the Align panel — align it onto the patient, then Fit"""
    bl_idname = "anaplast.ssm_place_mean"
    bl_label = "Place Mean Shape"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bool(context.scene.anaplast.ssm_model_path)

    def execute(self, context):
        p = context.scene.anaplast
        try:
            m = load_model(bpy.path.abspath(p.ssm_model_path))
        except Exception as e:
            rep(self, {'ERROR'}, f"Cannot load model: {e}")
            return {'CANCELLED'}
        col = mu.get_collection(context.scene, SSM_COLLECTION)
        o = model_to_object("SSM_Mean", m["mean"], m["faces"], col)
        o.location = Vector((0.0, 0.0, 0.0))
        p.align_moving = o
        if p.face_scan_obj and not p.align_target:
            p.align_target = p.face_scan_obj
        rep(self, {'INFO'}, "SSM_Mean placed. Align it onto the patient (points + ICP), then Fit Model")
        return {'FINISHED'}


class ANAPLAST_OT_ssm_fit(bpy.types.Operator):
    """Deform the model to match the patient scan where tissue exists; the defect is filled by the model's statistics"""
    bl_idname = "anaplast.ssm_fit"
    bl_label = "Fit Model to Patient"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        p = context.scene.anaplast
        return bool(p.ssm_model_path) and bpy.data.objects.get("SSM_Mean") is not None and p.align_target is not None

    def execute(self, context):
        p = context.scene.anaplast
        m = load_model(bpy.path.abspath(p.ssm_model_path))
        mean, comps, var, faces = m["mean"], m["components"], m["variances"], m["faces"]
        inst = bpy.data.objects["SSM_Mean"]
        target = p.align_target
        pose = np.array(inst.matrix_world)                     # where the user put the mean
        R0, t0 = pose[:3, :3], pose[:3, 3]
        dg = context.evaluated_depsgraph_get()
        bvh = BVHTree.FromObject(target, dg)
        t_mw, t_inv = target.matrix_world, target.matrix_world.inverted()
        t_rot = t_mw.to_3x3()
        segs = boundary_edge_segments(target)
        n = mean.shape[0]
        tri = faces if faces.shape[1] == 3 else np.array([(f[0], f[i], f[i + 1]) for f in faces for i in range(1, len(f) - 1)])
        # landmark pairs from the Align panel: source points on SSM_Mean ↔ target points on the scan.
        # Each source point is bound to its nearest model vertex and pulls hard every iteration.
        from .align import _points
        S, T = _points("AL_S_"), _points("AL_T_")
        lm_idx, lm_tgt = [], []
        world0 = mean @ R0.T + t0
        for so, to in zip(S, T):
            sp = np.array(so.matrix_world.translation)
            lm_idx.append(int(np.argmin(np.linalg.norm(world0 - sp, axis=1))))
            lm_tgt.append(np.array(to.matrix_world.translation))
        lm_idx = np.array(lm_idx, dtype=np.int64); lm_tgt = np.array(lm_tgt).reshape(-1, 3)
        LM_W = 50.0
        b = np.zeros(comps.shape[0])
        lam = p.ssm_regularisation
        used = 0
        rms = float('nan')
        for it in range(p.ssm_iterations):
            shape = instance(mean, comps, b)                    # model space
            world = shape @ R0.T + t0
            mnorm = vertex_normals(world, tri)
            Q = np.empty_like(world); mask = np.zeros(n, dtype=bool)
            for i, q in enumerate(world):
                loc, nrm, idx, dist = bvh.find_nearest(t_inv @ Vector(q))
                if loc is None:
                    continue
                w = t_mw @ loc
                if (w - Vector(q)).length > p.ssm_max_dist:
                    continue                                    # far from any tissue: defect
                if on_boundary(w, idx, segs):
                    continue                                    # projects onto a hole rim: defect
                sn = (t_rot @ nrm).normalized()
                if abs(sn.dot(Vector(mnorm[i]))) < 0.5:
                    continue                                    # surface facing the wrong way: not the same skin
                Q[i] = (w.x, w.y, w.z); mask[i] = True
            used = int(mask.sum())
            if used < 20:
                rep(self, {'ERROR'}, "Model is too far from the scan — align SSM_Mean first (points + ICP) or raise max distance")
                return {'CANCELLED'}
            weights = np.ones(n)
            if len(lm_idx):
                Q[lm_idx] = lm_tgt; mask[lm_idx] = True; weights[lm_idx] = LM_W
            # pose update on inliers (rigid, landmarks weighted), then shape update in model space
            R, t = weighted_kabsch(world[mask], Q[mask], weights[mask])
            R0, t0 = R @ R0, R @ t0 + t
            Qm = (Q - t0) @ R0                                  # targets back into model space (R0 orthonormal)
            b = solve_coeffs(mean, comps, var, Qm, mask, lam, weights)
            # clamp to ±3 SD so the fit never leaves plausible anatomy
            b = np.clip(b, -3 * np.sqrt(var), 3 * np.sqrt(var))
            world = instance(mean, comps, b) @ R0.T + t0
            rms_new = float(np.sqrt(((world[mask] - Q[mask]) ** 2).sum(axis=1).mean()))
            if abs(rms - rms_new) < 1e-4:
                rms = rms_new
                break
            rms = rms_new
        shape = instance(mean, comps, b)
        col = mu.get_collection(context.scene, SSM_COLLECTION)
        fit = model_to_object("SSM_Fit", shape, faces, col)
        fit.matrix_world = to_matrix(R0, t0)
        # colour: vertices the scan supported vs. those the statistics filled in (defect)
        attr = fit.data.color_attributes.new("support", 'FLOAT_COLOR', 'POINT')
        cols = np.where(mask[:, None], np.array([[0.55, 0.75, 0.55, 1.0]]), np.array([[0.95, 0.65, 0.25, 1.0]]))
        attr.data.foreach_set("color", cols.ravel())
        fit.data.color_attributes.active_color = attr
        fit['anaplast_part']='SCULPT'
        p.prosthesis_obj = fit
        p.crop_target = fit
        p.source_prosthesis = fit
        inst.hide_set(True)
        free = n - used
        lm = f", {len(lm_idx)} landmark pairs" if len(lm_idx) else ""
        rep(self, {'INFO'}, f"Fit: RMS {rms:.2f} mm on {used} supported vertices{lm}; {free} vertices ({free / n:.0%}) filled from statistics. "
                              f"SSM_Fit set as Prosthesis (orange = filled-in region)")
        return {'FINISHED'}


class ANAPLAST_OT_ssm_generate_mpfb(bpy.types.Operator):
    """Generate N random heads with MPFB (MakeHuman plugin) into the dataset folder. Needs the MPFB extension installed"""
    bl_idname = "anaplast.ssm_generate_mpfb"
    bl_label = "Generate Heads with MPFB"
    bl_options = {'REGISTER'}

    count: bpy.props.IntProperty(name="How many", default=100, min=3, max=2000)
    strength: bpy.props.FloatProperty(name="Variation", default=0.7, min=0.1, max=1.0)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        p = context.scene.anaplast
        folder = bpy.path.abspath(p.ssm_dataset_dir)
        if not folder:
            rep(self, {'ERROR'}, "Set the dataset folder first")
            return {'CANCELLED'}
        os.makedirs(folder, exist_ok=True)
        HumanService = TargetService = LocationService = None
        tried = []
        for pkg in mpfb_candidates():
            try:
                import importlib
                HumanService = importlib.import_module(pkg + ".services.humanservice").HumanService
                TargetService = importlib.import_module(pkg + ".services.targetservice").TargetService
                LocationService = importlib.import_module(pkg + ".services.locationservice").LocationService
                break
            except Exception as e:
                tried.append(f"{pkg}: {type(e).__name__}")
                HumanService = None
        if HumanService is None:
            rep(self, {'ERROR'}, "MPFB found but its modules would not load — enable it in Preferences → Add-ons and open its panel once, then retry. Tried: " + "; ".join(tried[:4]))
            return {'CANCELLED'}
        rng = np.random.default_rng()
        for o in list(bpy.data.objects):                       # leftovers from earlier runs
            if o.name.startswith("Human") and o.type == 'MESH':
                try:
                    bpy.data.objects.remove(o, do_unlink=True)
                except Exception:
                    pass
        try:
            tdir = LocationService.get_mpfb_data("targets")
            groups = [g for g in ("nose", "ears", "eyes", "cheek", "chin", "forehead", "head", "mouth", "neck") if os.path.isdir(os.path.join(tdir, g))]
            targets = [fp for g in groups for fp in glob.glob(os.path.join(tdir, g, "*.target*"))]
            if not targets:
                raise RuntimeError(f"no targets found under {tdir}")
            written = 0
            gender_range = {"ANY": (0.0, 1.0), "F": (0.0, 0.15), "M": (0.85, 1.0)}[p.mpfb_gender]
            age_range = {"ANY": (0.25, 0.9), "YOUNG": (0.28, 0.45), "MID": (0.45, 0.68), "OLD": (0.68, 0.95)}[p.mpfb_age]
            for i in range(self.count):
                # MPFB expects a "race" sub-dictionary whose three weights sum to 1
                r = rng.dirichlet([1.0, 1.0, 1.0])
                macro = {"gender": float(rng.uniform(*gender_range)), "age": float(rng.uniform(*age_range)),
                         "proportions": float(rng.uniform(0.2, 0.8)), "weight": float(rng.uniform(0.3, 0.8)),
                         "height": 0.5, "muscle": 0.5, "cupsize": 0.5, "firmness": 0.5,
                         "race": {"african": float(r[0]), "asian": float(r[1]), "caucasian": float(r[2])}}
                human = HumanService.create_human(mask_helpers=True, detailed_helpers=False, extra_vertex_groups=False,
                                                  feet_on_ground=True, scale=100.0, macro_detail_dict=macro)
                before_objs = set(bpy.data.objects)
                for fp in rng.choice(targets, size=min(len(targets), 40), replace=False):
                    w = float(rng.uniform(0.0, self.strength)) if rng.random() < 0.5 else 0.0
                    if w > 0:
                        TargetService.load_target(human, str(fp), weight=w)
                if p.mpfb_subdiv > 0:
                    sub = human.modifiers.new("anaplast_subdiv", 'SUBSURF')
                    sub.levels = p.mpfb_subdiv
                    sub.render_levels = p.mpfb_subdiv
                    sub.subdivision_type = 'CATMULL_CLARK'
                dg = context.evaluated_depsgraph_get()
                ev = human.evaluated_get(dg)
                me = ev.to_mesh()
                v = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", v); v = v.reshape(-1, 3)
                me.calc_loop_triangles()
                f = np.array([lt.vertices[:] for lt in me.loop_triangles], dtype=np.int64)
                ev.to_mesh_clear()
                if p.mpfb_keep != 'BODY':
                    v, f = crop_head(v, f, face_only=(p.mpfb_keep == 'FACE'))
                v, f = largest_island(v, f)          # drop helper strips (hair/clothes helpers) if any survive
                write_obj(os.path.join(folder, f"mpfb_{i:04d}.obj"), v, f)
                written += 1
                for o in list(set(bpy.data.objects) - before_objs) + [human]:
                    try:
                        bpy.data.objects.remove(o, do_unlink=True)
                    except Exception:
                        pass
        except Exception as e:
            rep(self, {'ERROR'}, f"MPFB generation failed after {written if 'written' in dir() else 0} heads: {type(e).__name__}: {e}")
            return {'CANCELLED'}
        what = {"HEAD": "heads", "FACE": "faces", "BODY": "full bodies"}[p.mpfb_keep]
        hint = " (3 is the minimum to train; 50+ makes a useful model)" if written < 20 else ""
        rep(self, {'INFO'}, f"Wrote {written} {what} to {folder}{hint}. Import one, lasso the region, Set Region, then Train")
        return {'FINISHED'}


_classes = (ANAPLAST_OT_ssm_set_region, ANAPLAST_OT_ssm_train, ANAPLAST_OT_ssm_place_mean,
            ANAPLAST_OT_ssm_fit, ANAPLAST_OT_ssm_generate_mpfb)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
