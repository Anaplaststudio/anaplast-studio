"""Local painted-eye fitting; no patient data leaves Blender."""
import json
import numpy as np
from mathutils.kdtree import KDTree
from . import ocular_marking as mark, ocular_smooth as smooth


def selection(obj,world):
    iris=mark.values_for(obj,'IRIS')>.5
    sclera=(mark.values_for(obj,'SCLERA')>.5)&~iris
    if not iris.any():return sclera,0
    # Keep both scleral patches beside the iris, even when they are disconnected.
    # Remote brush islands do not enlarge the globe or cut holes in the scan.
    tree=KDTree(int(iris.sum()))
    for i,p in enumerate(world[iris]):tree.insert(p,i)
    tree.balance()
    limit=max(2.,float(np.linalg.norm(np.ptp(world[iris],axis=0)))*.4)
    adjacency={int(i):[] for i in np.flatnonzero(sclera)}
    for edge in obj.data.edges:
        a,b=edge.vertices
        if a in adjacency and b in adjacency:
            adjacency[a].append(b);adjacency[b].append(a)
    remaining=set(adjacency);accepted=np.zeros(len(world),bool)
    while remaining:
        pending=[remaining.pop()];group=[]
        while pending:
            a=pending.pop();group.append(a)
            for b in adjacency[a]:
                if b in remaining:remaining.remove(b);pending.append(b)
        if any(tree.find(world[i])[2]<=limit for i in group):accepted[group]=True
    if accepted.sum()<30:raise RuntimeError('Paint the sclera beside the marked iris; no broad nearby sclera was found')
    return accepted,int(sclera.sum()-accepted.sum())


def local_frame(world,selected,iris,view_front,view_right):
    pts=np.asarray(world[selected|iris],float);center=np.median(pts,axis=0)
    _,_,vt=np.linalg.svd(pts-center,full_matrices=False)
    u=vt[-1]
    hint=np.asarray(view_front,float)
    if u@hint<0:u=-u
    r=np.asarray(view_right,float);r-=u*(r@u)
    if np.linalg.norm(r)<.1:r=vt[0]-u*(vt[0]@u)
    r/=np.linalg.norm(r);f=np.cross(u,r)
    return u,r,f


def sphere(points):
    """Multi-start geometric fit constrained to the visible outward hemisphere."""
    pts=np.asarray(points,float);origin=np.mean(pts,axis=0);q=pts-origin
    spread=np.max(np.linalg.norm(q[:,:2],axis=1));candidates=[]
    for rad in (spread*1.1,spread*1.6,spread*2.5,spread*4,12.,20.):
        try:c,rad,_=smooth.sphere_fit_fixed_radius(q,max(rad,spread*1.01))
        except RuntimeError:continue
        for _ in range(100):
            delta=c-q;dist=np.linalg.norm(delta,axis=1);res=dist-rad
            scale=max(.03,1.4826*np.median(abs(res-np.median(res))))
            weight=np.minimum(1,2*scale/np.maximum(abs(res),1e-12))
            J=np.c_[delta/np.maximum(dist[:,None],1e-9),-np.ones(len(q))]
            step=np.linalg.lstsq(J*np.sqrt(weight[:,None]),-res*np.sqrt(weight),rcond=None)[0]
            if np.linalg.norm(step)<1e-7:break
            loss=np.sum(weight*res**2)
            for alpha in (1,.5,.25,.125,.0625,.03125,.015625):
                nc=c+alpha*step[:3];nr=rad+alpha*step[3]
                if nr<=0 or nr>100 or nc[2]>=q[:,2].min():continue
                if np.max(np.linalg.norm(q[:,:2]-nc[:2],axis=1))>=nr*.98:continue
                if np.sum(weight*(np.linalg.norm(q-nc,axis=1)-nr)**2)<loss:
                    c,rad=nc,nr;break
            else:break
        if c[2]<q[:,2].min() and rad<100:
            residual=np.linalg.norm(q-c,axis=1)-rad
            candidates.append((float(np.mean(np.minimum(residual**2,.5**2))),c+origin,float(rad)))
    if not candidates:raise RuntimeError('The marked sclera does not contain enough outward curvature to estimate a globe')
    _,center,radius=min(candidates,key=lambda x:x[0])
    if radius>=99:raise RuntimeError('The painted sclera is too shallow to estimate a reliable diameter; use a measured diameter or broaden the paint')
    return center,radius


def design(xy,mid,scale):
    x,y=((np.asarray(xy)-mid)/scale).T
    return np.c_[np.ones(len(x)),x,y,x*x,x*y,y*y]


def height(xy,center,radius,model=None):
    z=smooth.sphere_height(xy,center,radius)
    return z+correction(xy,model)


def correction(xy,model):
    if model:
        q=(np.asarray(xy)-np.array(model['mid']))/np.array(model['scale'])
        # Retain the fitted correction over the entire painted region, then
        # blend its continuation back to the sphere outside that region.
        t=np.clip((np.max(abs(q),axis=1)-1)/1.5,0,1)
        fade=1-t**3*(10-15*t+6*t*t)
        return design(xy,np.array(model['mid']),np.array(model['scale']))@np.array(model['coef'])*fade
    return np.zeros(len(xy))


def fit(points,fixed_radius=None):
    samples=smooth.balanced_samples(points,spacing=.3)
    if len(samples)<30:raise RuntimeError('Paint a broader scleral surface to measure its curvature')
    if fixed_radius:
        center,radius,_=smooth.sphere_fit_fixed_radius(samples,fixed_radius)
    else:center,radius=sphere(samples)
    base=smooth.sphere_height(samples[:,:2],center,radius)
    residual=samples[:,2]-base;model=None
    if not fixed_radius:
        mid=(samples[:,:2].max(0)+samples[:,:2].min(0))*.5
        scale=np.maximum(np.ptp(samples[:,:2],axis=0)*.5,.1)
        D=design(samples[:,:2],mid,scale);weights=np.ones(len(samples))
        for _ in range(15):
            # Mild regularization on curvature avoids chasing scan texture.
            penalty=np.diag([0,0,0,.12,.12,.12])
            coef=np.linalg.lstsq(np.vstack((D*np.sqrt(weights[:,None]),penalty)),np.r_[residual*np.sqrt(weights),np.zeros(6)],rcond=None)[0]
            err=residual-D@coef;sigma=max(.04,1.4826*np.median(abs(err-np.median(err))))
            weights=np.minimum(1,2.5*sigma/np.maximum(abs(err),1e-9))
        # A single broad correction can be oval but cannot reproduce local ripples.
        proposed={'mid':mid.tolist(),'scale':scale.tolist(),'coef':coef.tolist()}
        before=np.sqrt(np.mean(residual**2));after=np.sqrt(np.mean((residual-D@coef)**2))
        delta=samples[:,:2]-center[:2];depth=base-center[2]
        curvature=-np.eye(2)[None]/depth[:,None,None]-delta[:,:,None]*delta[:,None,:]/depth[:,None,None]**3
        curvature+=np.array([[2*coef[3]/scale[0]**2,coef[4]/np.prod(scale)],[coef[4]/np.prod(scale),2*coef[5]/scale[1]**2]])
        convex=np.max(np.linalg.eigvalsh(curvature))<-.0001
        if convex and after<before*.9 and np.max(abs(D@coef))<min(2.,radius*.15):model=proposed
    error=height(samples[:,:2],center,radius,model)-samples[:,2]
    metrics={'rms_mm':float(np.sqrt(np.mean(error**2))),'p95_mm':float(np.percentile(abs(error),95)),
             'sphere_rms_mm':float(np.sqrt(np.mean(residual**2))),'within_0_2_mm_percent':float(np.mean(abs(error)<=.2)*100),
             'sample_count':len(samples),'surface':'smooth oval adjustment' if model else 'sphere'}
    return center,radius,model,metrics


def model_for(ball):return json.loads(ball.get('fit_surface_model','null'))


def fitted_height(xy,ball):
    return height(xy,np.array(ball['fit_center_local']),ball['fit_radius_mm'],model_for(ball))


def accepted_mask(obj,ball):
    selected=np.zeros(len(obj.data.vertices),bool)
    selected[np.asarray(ball['fit_sclera_indices'],dtype=int)]=True
    return selected
