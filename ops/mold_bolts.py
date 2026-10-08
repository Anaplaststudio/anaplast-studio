"""Open clamp bores and local, flat bolt seats on shell backs."""
import math
import bpy
import numpy as np
from mathutils import Vector
from ..utils import mesh as mu


def axial_bounds(part, up):
    co = np.empty(len(part.data.vertices) * 3)
    part.data.vertices.foreach_get('co', co)
    matrix = np.asarray(part.matrix_world)
    world = co.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]
    z = world @ np.asarray(up)
    return float(z.min()), float(z.max())


def bore_samples(xy, radius):
    yield np.asarray(xy, float)
    for fraction in (.5, .9):
        for angle in np.linspace(0., 2. * math.pi, 24, endpoint=False):
            yield np.asarray(xy) + radius * fraction * np.array((math.cos(angle), math.sin(angle)))


def verify_bores(part, positions, diameter, frame):
    from .mold_auricular import world_tree
    up, right, fwd = frame
    lo, hi = axial_bounds(part, up)
    tree = world_tree(part)
    count = 0
    for xy in positions:
        for q in bore_samples(xy, diameter * .5):
            origin = right * float(q[0]) + fwd * float(q[1]) + up * (hi + 2.)
            if tree.ray_cast(origin, -up, hi - lo + 4.)[0] is not None:
                raise RuntimeError(f'{part.name}: a clamp bolt bore remains obstructed; the mold is not ready')
            count += 1
    return count


def drill_bores(part, positions, diameter, frame):
    from .mold_auto import cylinder
    from .mold_auricular import cut
    up, right, fwd = frame
    lo, hi = axial_bounds(part, up)
    for i, xy in enumerate(positions):
        # Centre on the complete finished part, not its local mating height.
        center = right * float(xy[0]) + fwd * float(xy[1]) + up * ((lo + hi) * .5)
        tool = cylinder(f'Mold_Bolt_tmp{i}', center, up, diameter * .5, hi - lo + 10.,
                        part.users_collection[0], segments=64)
        try:
            cut(part, tool, 'DIFFERENCE')
        finally:
            mu.delete_object(tool)
    return verify_bores(part, positions, diameter, frame)


def add_shell_seats(part, inner_reference, positions, diameter, wall, rise, frame, is_cap, length_mode="SHORT"):
    """Clip bosses to the original inner surface, leaving the mating fit intact.

    Each outside seat is a plane perpendicular to the opening/bolt axis. Its
    height clears the outermost backing vertex in the sleeve footprint by rise.
    The bore is drilled separately, after all unions and surface cleanup.
    """
    from .mold_auto import cylinder
    from .mold_auricular import cut, world_tree
    up, right, fwd = frame
    sign = 1. if is_cap else -1.
    radius = diameter * .5 + wall
    col = part.users_collection[0]
    lo, hi = axial_bounds(part, up)
    reference_lo, reference_hi = axial_bounds(inner_reference, up)
    tree = world_tree(part)
    backing_vertices = np.array([part.matrix_world @ v.co for v in part.data.vertices])
    projected = np.c_[backing_vertices @ np.asarray(right), backing_vertices @ np.asarray(fwd)]
    levels = backing_vertices @ np.asarray(up)
    result = []
    for i, xy in enumerate(positions):
        xy = np.asarray(xy, float)
        samples = [xy]
        for fraction in (.25, .5, .75, 1.):
            samples.extend(xy + radius * fraction * np.array((math.cos(a), math.sin(a)))
                           for a in np.linspace(0., 2. * math.pi, 64, endpoint=False))
        heights = []
        start_z = hi + 5. if is_cap else lo - 5.
        for q in samples:
            hit = tree.ray_cast(right * float(q[0]) + fwd * float(q[1]) + up * start_z,
                                -up * sign, hi - lo + 10.)[0]
            if hit is None:
                raise RuntimeError('A bolt sleeve extends beyond the shell or crosses an opening. Reduce sleeve wall thickness or widen the land')
            heights.append(hit.dot(up))
        # Include every mesh vertex in the footprint: catches fine local peaks
        # between the ray samples, rather than letting them stand above the seat.
        local = levels[np.linalg.norm(projected - xy, axis=1) <= radius]
        outward_height = max(sign * h for h in heights)
        if len(local): outward_height = max(outward_height, float((sign * local).max()))
        seat_z = sign * (outward_height + rise)
        if length_mode == "FULL":
            original_end = reference_hi if is_cap else reference_lo
            seat_z = sign * max(sign * original_end, sign * seat_z)
        reference = mu.duplicate_object(inner_reference, 'Mold_BoltReference_tmp', col)
        tool = None
        try:
            # Extend only the old block's back plane if the new raised seat
            # exceeds it. The inner/mating surface and outer footprint stay put.
            end = reference_hi if is_cap else reference_lo
            extended_z = max(reference_hi, seat_z + 2.) if is_cap else min(reference_lo, seat_z - 2.)
            inv = reference.matrix_world.inverted()
            for v in reference.data.vertices:
                q = reference.matrix_world @ v.co
                if abs(q.dot(up) - end) < .001:
                    v.co = inv @ (q + up * (extended_z - q.dot(up)))
            reference.data.update()
            other = reference_lo - 2. if is_cap else reference_hi + 2.
            center = right * float(xy[0]) + fwd * float(xy[1]) + up * ((seat_z + other) * .5)
            tool = cylinder('Mold_BoltSleeve_tmp', center, up, radius, abs(seat_z - other), col, segments=64)
            cut(tool, reference, 'INTERSECT')
            if not len(tool.data.polygons):
                raise RuntimeError('The clamp sleeve does not meet the shell')
            cut(part, tool, 'UNION')
        finally:
            if tool is not None: mu.delete_object(tool)
            mu.delete_object(reference)
        result.append({'xy':xy.tolist(), 'hole_diameter_mm':float(diameter),
                       'length_mode':length_mode, 'sleeve_wall_mm':float(wall), 'sleeve_outer_diameter_mm':float(radius * 2.),
                       'seat_raise_mm':float(rise), 'seat_plane_mm':float(seat_z),
                       'outer_shell_extreme_mm':float(sign * outward_height)})
    return result
