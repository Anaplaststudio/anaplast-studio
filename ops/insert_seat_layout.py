"""Plan mounting footprints without moving patient or attachment landmarks."""
import math
import numpy as np
from mathutils import Vector


def rotated(frame, degrees):
    u, r, f = frame
    a = math.radians(degrees)
    return u, r * math.cos(a) + f * math.sin(a), f * math.cos(a) - r * math.sin(a)


def footprint(item, width, slim):
    from . import mold_inserts as mi
    radius = width * .5
    angles = np.linspace(0, 2 * math.pi, 128, endpoint=False)
    if slim:
        offset = radius + max(2.5, item.screw * .5 + item.clearance + .8)
        tab = (item.screw + .4) * .5 + .8
        points = mi.d_profile(radius + .8, .85)
        points += [(tab * math.cos(a), sign * offset + tab * math.sin(a))
                   for sign in (-1, 1) for a in angles]
    else:
        side = radius + (4. if item.kind == 'AIRWAY' else 1.5)
        points = [(max(-side * .65, side * math.cos(a)), (radius + 4.) * math.sin(a)) for a in angles]
    # Include the socket as well as the flange, even with unusual clearance.
    points += mi.d_profile(radius + item.clearance, .85 if slim else .65)
    return np.asarray(mi.aw.convex_hull_2d(points))


def projected(center, frame, polygon, base_frame):
    points = np.asarray(center) + polygon[:, :1] * np.asarray(frame[1]) + polygon[:, 1:] * np.asarray(frame[2])
    return np.column_stack((points @ base_frame[1], points @ base_frame[2]))


def separated(a, b, gap=.5):
    """Conservative convex separation: include 0.5 mm between mounting parts."""
    for poly in (a, b):
        edge = np.roll(poly, -1, axis=0) - poly
        length = np.linalg.norm(edge, axis=1)
        edge = edge[length > 1e-10] / length[length > 1e-10, None]
        axes = np.column_stack((-edge[:, 1], edge[:, 0]))
        pa, pb = a @ axes.T, b @ axes.T
        if np.any(np.maximum(pb.min(0) - pa.max(0), pa.min(0) - pb.max(0)) >= gap):
            return True
    return False


def choose(item, center, initial, width, slim, obstacles, base_frame, automatic):
    polygon = footprint(item, width, slim)
    angles = [0] + [a for step in range(1, 37) for a in (step * 5, -step * 5)] if automatic else [0]
    for angle in angles:
        candidate = rotated(initial, angle)
        outline = projected(center, candidate, polygon, base_frame)
        if all(separated(outline, other) for _, other in obstacles):
            return candidate, outline
    names = ', '.join(name for name, other in obstacles
                      if not separated(projected(center, initial, polygon, base_frame), other))
    raise RuntimeError(f'{item.seat_name or item.name}: mounting parts are too close to {names}. '
                       'Change Seat angle or move an unbuilt seat farther away; existing seats are kept in place.')


def plan_pair(entries, base_frame):
    """Joint search for two marked seats; fixed seats keep their built frames."""
    candidates=[]
    for item,center,frame,width,slim,automatic in entries:
        angles=[0]+[a for step in range(1,37) for a in (step*5,-step*5)] if automatic else [0]
        poly=footprint(item,width,slim)
        candidates.append([(angle,rotated(frame,angle),projected(center,rotated(frame,angle),poly,base_frame)) for angle in angles])
    pairs=sorted(((a,b) for a in candidates[0] for b in candidates[1]),key=lambda ab:abs(ab[0][0])+abs(ab[1][0]))
    for a,b in pairs:
        if separated(a[2],b[2]):return {entries[0][0].uid:a[1],entries[1][0].uid:b[1]}
    raise RuntimeError('These two mounting seats do not fit at the marked positions. Move an unbuilt seat farther away, or adjust a built Seat angle and rebuild. Marks and the installed mold are unchanged.')
