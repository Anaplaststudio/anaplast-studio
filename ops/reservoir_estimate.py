"""Generous Sculpt-based sizing for wax cups; not a casting-volume report."""
import math
import numpy as np
from . import mold_volume


def estimate(scene):
    from .shell import source_surface
    obj=source_surface(scene.anaplast)
    method='NOMINAL_FALLBACK';amount=10.;source='No Sculpt selected'
    if obj is not None and obj.type=='MESH':
        source=obj.name
        factor=float(scene.unit_settings.scale_length)**3*1e6
        try:
            # Translation / exploded display position does not change volume.
            amount=mold_volume.volume(obj)*factor
            if not math.isfinite(amount) or amount<=0:raise ValueError('Empty Sculpt')
            method='SCULPT_SOLID'
        except (ValueError,RuntimeError):
            # Open cropped surfaces have no reliable enclosed volume. The
            # local bounding box is deliberately conservative and cheap.
            points=np.array([v.co[:] for v in obj.data.vertices])
            amount=float(np.prod(np.ptp(points,axis=0)))*abs(obj.matrix_world.to_3x3().determinant())*factor if len(points) else 0.
            method='SCULPT_BOUNDING_BOX'
            if not math.isfinite(amount) or amount<=0:amount=10.;method='NOMINAL_FALLBACK'
    return {'fill_ml':max(.1,amount*1.20),'source':source,'method':method,'padding_percent':20.}
