"""Measure scan curvature and replace the eye with a smooth spherical ocular."""
import numpy as np


def balanced_samples(points,spacing=.4):
    points=np.asarray(points,float)
    cells=np.floor(points[:,:2]/spacing).astype(np.int64)
    _,groups=np.unique(cells,axis=0,return_inverse=True)
    return np.array([np.median(points[groups==i],axis=0) for i in range(groups.max()+1)])


def sphere_fit(points):
    q=np.asarray(points,float);origin=q.mean(axis=0);q=q-origin
    if len(q)<30 or np.linalg.matrix_rank(q-q.mean(0))<3:raise RuntimeError('Too little curved sclera to estimate its radius')
    A=np.c_[2*q,np.ones(len(q))];v=np.linalg.lstsq(A,np.sum(q*q,axis=1),rcond=None)[0]
    c=v[:3];radius=np.sqrt(max(v[3]+c@c,1e-9))
    for _ in range(80):
        delta=c-q;distance=np.linalg.norm(delta,axis=1);residual=distance-radius
        scale=max(.02,1.4826*np.median(np.abs(residual-np.median(residual))))
        weight=np.minimum(1,1.5*scale/np.maximum(abs(residual),1e-12))
        J=np.c_[delta/np.maximum(distance[:,None],1e-12),-np.ones(len(q))]
        step=np.linalg.lstsq(J*np.sqrt(weight[:,None]),-residual*np.sqrt(weight),rcond=None)[0]
        if np.linalg.norm(step)<1e-8:break
        loss=np.sum(weight*residual**2)
        for alpha in (1,.5,.25,.125,.0625,.03125):
            nc=c+alpha*step[:3];nr=radius+alpha*step[3]
            if nr>0 and np.sum(weight*(np.linalg.norm(q-nc,axis=1)-nr)**2)<loss:
                c,radius=nc,nr;break
        else:break
    c=c+origin
    if not np.isfinite(radius) or radius>1000 or c[2]>=np.median(points[:,2]):raise RuntimeError('The sclera marks do not define a reliable outward curve; refine them on the eye surface')
    return c,float(radius),float(np.sqrt(np.mean((np.linalg.norm(points-c,axis=1)-radius)**2)))


def sphere_fit_fixed_radius(points,radius):
    """Fit position only, preserving a measured radius and the front hemisphere."""
    q=np.asarray(points,float)
    if len(q)<30 or not np.isfinite(q).all() or not np.isfinite(radius) or radius<=0:
        raise RuntimeError('Mark at least 30 scleral vertices and enter a positive measured diameter')
    span=float(np.max(np.ptp(q[:,:2],axis=0)))
    if span>=2*radius:
        raise RuntimeError(f'Sclera marks span {span:.2f} mm, exceeding the measured {2*radius:.2f} mm diameter. Review the marks or scan scale; the scan has not been resized.')
    xy=(q[:,:2].min(0)+q[:,:2].max(0))*.5
    # Find a feasible projected center before optimizing depth and position.
    for _ in range(300):
        d=q[:,:2]-xy;dist=np.linalg.norm(d,axis=1);i=int(np.argmax(dist))
        if dist[i]<radius*.999:break
        xy+=d[i]/dist[i]*(dist[i]-radius*.998)
    if np.max(np.linalg.norm(q[:,:2]-xy,axis=1))>=radius*.999:
        raise RuntimeError('The sclera marks extend beyond the measured sphere; review the marks or scan scale')
    c=np.r_[xy,np.median(q[:,2]-np.sqrt(radius**2-np.sum((q[:,:2]-xy)**2,axis=1)))]
    for _ in range(100):
        delta=c-q;distance=np.linalg.norm(delta,axis=1);residual=distance-radius
        scale=max(.02,1.4826*np.median(abs(residual-np.median(residual))))
        weight=np.minimum(1,1.5*scale/np.maximum(abs(residual),1e-12))
        J=delta/np.maximum(distance[:,None],1e-12)
        step=np.linalg.lstsq(J*np.sqrt(weight[:,None]),-residual*np.sqrt(weight),rcond=None)[0]
        if np.linalg.norm(step)<1e-8:break
        loss=np.sum(weight*residual**2)
        for alpha in (1,.5,.25,.125,.0625,.03125,.015625):
            nc=c+alpha*step
            if nc[2]>=q[:,2].min() or np.max(np.linalg.norm(q[:,:2]-nc[:2],axis=1))>=radius*.999:continue
            if np.sum(weight*(np.linalg.norm(q-nc,axis=1)-radius)**2)<loss:c=nc;break
        else:break
    return c,float(radius),float(np.sqrt(np.mean((np.linalg.norm(q-c,axis=1)-radius)**2)))


def sphere_height(xy,center,radius):
    inside=radius**2-np.sum((np.asarray(xy)-center[:2])**2,axis=1)
    if np.any(inside<=0):raise RuntimeError('The extension reaches beyond the fitted sphere; reduce extension or review the sclera marks')
    return center[2]+np.sqrt(inside)


def sphere_to_domain(xy,center,radius):
    """Front-hemisphere coordinates to an angular chart measured in mm of arc."""
    delta=np.asarray(xy)-np.asarray(center)[:2];distance=np.linalg.norm(delta,axis=1)
    if np.any(distance>=radius):raise RuntimeError('The marked outline exceeds the measured sphere; refine the marks or sphere placement')
    return delta*np.divide(radius*np.arcsin(distance/radius),distance,out=np.ones_like(distance),where=distance>1e-12)[:,None]


def sphere_from_domain(domain,center,radius):
    domain=np.asarray(domain);distance=np.linalg.norm(domain,axis=1);theta=distance/radius
    if np.any(theta>=np.pi-.05):raise RuntimeError('The extension wraps too far around the eyeball; reduce it')
    xy=np.asarray(center)[:2]+domain*np.divide(np.sin(theta),theta,out=np.ones_like(theta),where=theta>1e-12)[:,None]
    return xy,np.asarray(center)[2]+radius*np.cos(theta)


def fit_surface(sclera,iris,iris_center,iris_radius):
    samples=balanced_samples(sclera);center,radius,rms=sphere_fit(samples)
    return center,radius,0.,rms


def evaluate(xy,center,radius,amplitude,iris_center,iris_radius):
    rho=np.linalg.norm(np.asarray(xy)-iris_center,axis=1)/iris_radius
    return sphere_height(xy,center,radius)+amplitude*np.maximum(0,1-rho**2)**3


def marked_fit(context,xy,u,r,f,iris_center):
    from . import ocular_marking as mark
    p=context.scene.ocular_production;obj=p.mark_obj
    from . import ocular as oc
    source=oc.reference(context.scene.anaplast)
    if not source or source.modifiers:raise RuntimeError('Assign the original marked scan with applied modifiers')
    if abs(context.scene.unit_settings.scale_length-.001)>1e-8:raise RuntimeError('Use millimetre case units first')
    if not obj:raise RuntimeError('Highlight the sclera and iris first')
    signature=mark.wm.geometry_signature(source)
    if obj.get('ocular_source_signature')!=signature or mark.wm.geometry_signature(obj)!=signature:raise RuntimeError('The scan changed; make a new marking copy')
    if context.scene.anaplast.ocular_trace.get('ocular_marks_signature')!=mark.mark_signature(obj):raise RuntimeError('The highlights changed; fit their borders again')
    world=np.array([obj.matrix_world@v.co for v in obj.data.vertices]);local=np.c_[world@r,world@f,world@u]
    iris=mark.values_for(obj,'IRIS')>.5;sclera=(mark.values_for(obj,'SCLERA')>.5)&~iris
    if sclera.sum()<30 or iris.sum()<30:raise RuntimeError('Highlight both sclera and iris before fitting smooth curvature')
    center,radius,amplitude,rms=fit_surface(local[sclera],local[iris],iris_center,context.scene.anaplast.ocular_iris_diameter*.5)
    z=evaluate(xy,center,radius,amplitude,iris_center,context.scene.anaplast.ocular_iris_diameter*.5)
    source=local[sclera|iris];residual=source[:,2]-evaluate(source[:,:2],center,radius,amplitude,iris_center,context.scene.anaplast.ocular_iris_diameter*.5)
    metrics={'fitted_sclera_radius_mm':radius,'fitted_sphere_center_local':list(center),'fitted_cornea_prominence_mm':amplitude,'sclera_radial_fit_rms_mm':rms,'smooth_axial_fit_rms_mm':float(np.sqrt(np.mean(residual**2))),'smooth_axial_fit_p95_mm':float(np.percentile(abs(residual),95))}
    return z,metrics
