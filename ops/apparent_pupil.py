"""Shared color-only apparent pupil band; no changes to relief or sclera."""
import numpy as np
from . import ocular_pupil

def validate(b,radius):
    if not b.apparent_pupil:return
    if b.pupil_max<=b.pupil_min:raise RuntimeError('Maximum pupil fade diameter must be larger than minimum pupil diameter')
    if b.pupil_max+2*b.pupil_outer_fade>=2*radius*.9:raise RuntimeError('Keep the pupil fade inside the iris; reduce maximum diameter or soft edge')
    if b.pupil_outer_darkness>b.pupil_darkness:raise RuntimeError('Darkening at maximum must not exceed inner iris darkening')

def apply(b, p, rgb, xy, radius, center=None):
    """Color-only, contour-following darkening; never feeds the iris relief map."""
    if not b.apparent_pupil:return rgb
    validate(b,radius)
    if abs(p.pupil_diameter-b.pupil_min)>1e-5:raise RuntimeError("Real pupil size changed; Update ocular to rebuild its opening and colors together")
    width=(b.pupil_max-b.pupil_min)*.5
    distance=ocular_pupil.distance(p,xy,np.zeros(2) if center is None else center,radius)
    t=np.clip(distance/width,0,1)
    fade=t*t*t*(10-15*t+6*t*t)
    dark=b.pupil_darkness+(b.pupil_outer_darkness-b.pupil_darkness)*fade
    if b.pupil_outer_fade>0:
        edge=np.clip((distance-width)/b.pupil_outer_fade,0,1)
        edge=edge*edge*edge*(10-15*edge+6*edge*edge)
    else:edge=(distance>width).astype(float)
    gain=1-dark*(1-edge)
    return rgb*gain[...,None]


def controls(layout,b):
    layout.prop(b,'apparent_pupil')
    if b.apparent_pupil:
        for name in ('pupil_min','pupil_max','pupil_darkness','pupil_outer_darkness','pupil_outer_fade'):layout.prop(b,name)

def record(obj,b):
    obj['apparent_pupil']=b.apparent_pupil
    for key,field in (('apparent_pupil_max_mm','pupil_max'),('apparent_pupil_darkness','pupil_darkness'),('apparent_pupil_outer_darkness','pupil_outer_darkness'),('apparent_pupil_outer_fade_mm','pupil_outer_fade')):
        obj[key]=getattr(b,field) if b.apparent_pupil else 0.
