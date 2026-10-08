"""Offline shape/photo fitting. NumPy only; no Blender, network, or executable models.

Photo cameras are scaled orthographic in this first prototype. Coordinates are
image fractions, X right/Y up. Identity coefficients are in model units (FLAME
standard deviations); photograph pixels are observations, not depth measurements.
"""
import io
import json
import pickle
import zipfile
from pathlib import Path
import numpy as np


class _SparseRecord:
    """Discard unused scipy sparse metadata without importing or executing scipy."""
    pass


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy', 'dtype'): np.dtype,
            ('numpy.core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
            ('numpy._core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
            ('numpy.core.multiarray', 'scalar'): np.core.multiarray.scalar,
            ('numpy._core.multiarray', 'scalar'): np.core.multiarray.scalar,
            ('numpy.core.numeric', '_frombuffer'): np.core.numeric._frombuffer,
            ('numpy._core.numeric', '_frombuffer'): np.core.numeric._frombuffer,
            ('scipy.sparse._csc', 'csc_matrix'): _SparseRecord,
            ('scipy.sparse.csc', 'csc_matrix'): _SparseRecord,
            ('scipy.sparse._csr', 'csr_matrix'): _SparseRecord,
            ('scipy.sparse.csr', 'csr_matrix'): _SparseRecord,
        }
        if (module, name) not in allowed:
            raise ValueError('Unsupported model object: ' + module + '.' + name)
        return allowed[module, name]


def validate(mean, basis, faces):
    mean = np.asarray(mean, dtype=float)
    basis = np.asarray(basis, dtype=float)
    faces = np.asarray(faces)
    if mean.ndim != 2 or mean.shape[1] != 3 or not 10 <= len(mean) <= 100000:
        raise ValueError('Expected a head mesh with 10–100,000 vertices')
    if basis.ndim != 3 or basis.shape[:2] != mean.shape or not 1 <= basis.shape[2] <= 300:
        raise ValueError('Invalid identity shape basis')
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError('Expected integer triangle indices')
    if not len(faces) or faces.min() < 0 or faces.max() >= len(mean):
        raise ValueError('Invalid face indices')
    if not np.isfinite(mean).all() or not np.isfinite(basis).all():
        raise ValueError('Model contains non-finite coordinates')
    if not 30 < np.ptp(mean, axis=0).max() < 1000:
        raise ValueError('Model dimensions are not plausible in millimetres; check source units')
    return mean, basis, faces.astype(np.int32)


def read_model(path, units=1000.):
    path = Path(path)
    if path.stat().st_size > 512 * 1024**2:
        raise ValueError('Model exceeds 512 MB prototype limit')
    landmarks, region = {}, []
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as d:
            mean, basis, faces = d['mean'], d['basis'], d['faces']
            landmarks = json.loads(str(d['landmarks'])) if 'landmarks' in d else {}
            region = d['region'].tolist() if 'region' in d else []
    else:
        if path.suffix.lower() == '.zip':
            with zipfile.ZipFile(path) as z:
                candidates = [i for i in z.infolist() if Path(i.filename).name.lower() == 'flame2023_open.pkl']
                if len(candidates) != 1 or candidates[0].file_size > 512 * 1024**2:
                    raise ValueError('ZIP must contain one flame2023_Open.pkl (FLAME 2023 Open)')
                payload = z.read(candidates[0])
        elif path.name.lower() == 'flame2023_open.pkl':
            payload = path.read_bytes()
        else:
            raise ValueError('Choose FLAME2023Open.zip, flame2023_Open.pkl, or a prepared nose library .npz')
        d = ArrayUnpickler(io.BytesIO(payload), encoding='latin1').load()
        mean = np.asarray(d['v_template']) * units
        dirs = np.asarray(d['shapedirs'])
        if dirs.ndim != 3 or dirs.shape[2] < 300:
            raise ValueError('Expected the 300 FLAME identity components')
        basis = dirs[:, :, :300] * units  # expression/pose directions excluded
        faces = d['f']
    mean, basis, faces = validate(mean, basis, faces)
    landmarks = {str(k): int(v) for k, v in landmarks.items()}
    if any(v < 0 or v >= len(mean) for v in landmarks.values()):
        raise ValueError('Library landmark outside mesh')
    region = np.asarray(region, dtype=int)
    if len(region) and (region.min() < 0 or region.max() >= len(mean)):
        raise ValueError('Library nose region outside mesh')
    return mean, basis, faces, landmarks, region


def similarity(source, target):
    source, target = np.asarray(source), np.asarray(target)
    if len(source) < 3 or source.shape != target.shape:
        raise ValueError('At least three corresponding landmarks are required')
    a, b = source-source.mean(0), target-target.mean(0)
    if np.linalg.svd(a, compute_uv=False)[1] < 1e-5 or np.linalg.svd(b, compute_uv=False)[1] < 1e-5:
        raise ValueError('Landmarks are collinear; use points spread across the face')
    u, s, vt = np.linalg.svd(a.T @ b)
    sign = np.ones(3); sign[-1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ np.diag(sign) @ u.T
    scale = float((s*sign).sum()/(a*a).sum())
    if not .4 < scale < 2.5:
        raise ValueError('Unusual head scale; check landmark labels and scan units')
    linear = rotation * scale
    shift = target.mean(0)-source.mean(0) @ linear.T
    return linear, shift


def rotation(v):
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.eye(3)
    x, y, z = v/theta
    k = np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    return np.eye(3)+np.sin(theta)*k+(1-np.cos(theta))*(k@k)


def rotvec(r):
    theta = np.arccos(np.clip((np.trace(r)-1)/2,-1,1))
    if theta < 1e-8:
        return np.zeros(3)
    if np.pi-theta < 1e-5:
        _, vecs = np.linalg.eigh((r+np.eye(3))/2)
        return vecs[:,-1]*theta
    return theta/(2*np.sin(theta))*np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])


def project(points, cam):
    return np.exp(np.clip(cam[3],-8,8))*(points @ rotation(cam[:3]).T)[:,:2]+cam[4:6]


def initial_camera(points, xy):
    # Linear estimate followed by projection onto a proper orthogonal camera.
    a, b = points-points.mean(0), xy-xy.mean(0)
    if np.linalg.matrix_rank(a) < 3 or np.linalg.matrix_rank(b) < 2:
        raise ValueError('Photo points need depth variation and spread over the face')
    p = np.linalg.lstsq(a, b, rcond=None)[0].T
    u,s,vt = np.linalg.svd(p, full_matrices=False)
    rows = u @ vt
    r = np.vstack((rows, np.cross(rows[0],rows[1])))
    scale = max(float(s.mean()),1e-5)
    return np.r_[rotvec(r), np.log(scale), xy.mean(0)-scale*(points.mean(0)@r.T)[:2]]


def fit_photos(mean, basis, photo_data, iterations=60, strength=2.5):
    """Joint identity + per-photo cameras; returns conservative regularized fit.

photo_data: list of (vertex indices, 2D coordinates normalised by max image side).
No scan/defect surface is used here. Manual landmarks are not automatic detection.
"""
    if not photo_data:
        raise ValueError('Add and mark at least one photograph first')
    center = mean.mean(0); extent = float(np.ptp(mean,axis=0).max())
    shape = (mean-center)/extent; dirs=basis/extent; modes=dirs.shape[2]
    cameras=[]
    for ids, xy in photo_data:
        if len(ids)<8 or len(set(ids))!=len(ids) or not np.isfinite(xy).all():
            raise ValueError('Each photo needs at least 8 distinct, visible model landmarks')
        cameras.append(initial_camera(shape[ids],xy))
    params=np.r_[np.zeros(modes),np.concatenate(cameras)]
    def residual(p):
        v=shape+np.einsum('nck,k->nc',dirs,p[:modes])
        errs=[]
        for j,(ids,xy) in enumerate(photo_data):
            errs.append(((project(v[ids],p[modes+6*j:modes+6*j+6])-xy)/.005).ravel())
        return np.r_[np.concatenate(errs),p[:modes]/strength]
    damping=.01; r=residual(params); cost=r@r; history=[float(cost)]
    for _ in range(iterations):
        # Huber IRLS stops a poorly placed point dominating all views.
        weights=np.minimum(1.,3./np.maximum(np.abs(r),1e-12)); weights[-modes:]=1.
        jac=np.empty((len(r),len(params)))
        for k in range(len(params)):
            p=params.copy(); p[k]+=1e-5
            jac[:,k]=(residual(p)-r)/1e-5
        a=jac.T@(weights[:,None]*jac); g=jac.T@(weights*r)
        step=np.linalg.solve(a+damping*np.diag(np.maximum(np.diag(a),1)), -g)
        candidate=params+step; candidate[:modes]=np.clip(candidate[:modes],-3,3)
        nr=residual(candidate)
        def huber(z):
            w=np.abs(z); loss=np.where(w<=3,z*z,6*w-9); loss[-modes:]=z[-modes:]**2
            return float(loss.sum())
        if huber(nr)<huber(r):
            params,r=candidate,nr; damping=max(damping/3,1e-7); history.append(huber(r))
            if np.linalg.norm(step)<1e-6:break
        else:
            damping=min(damping*10,1e9)
            if damping>=1e9:break
    co=mean+np.einsum('nck,k->nc',basis,params[:modes])
    cams=params[modes:].reshape(-1,6)
    rms=[float(np.sqrt(np.mean(np.sum((project((co[ids]-center)/extent,cam)-xy)**2,axis=1)))) for cam,(ids,xy) in zip(cams,photo_data)]
    return co,params[:modes],cams,{'photo_rms_fraction':rms,'accepted_steps':len(history)-1,'camera':'scaled orthographic','views':len(photo_data),'coefficients_at_limit':int((np.abs(params[:modes])>2.999).sum())}
