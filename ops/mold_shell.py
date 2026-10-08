"""Distance-based backing for a two-part mold, without offset-face folds."""
import bpy
import bmesh
from mathutils import Vector
from ..utils import mesh as mu


def build_shell(part, up, wall, outline, frame, z_lo, z_hi, is_cap, voxel, *, fixed_grid=None, return_cutter=False, reference=None):
    from .mold_auto import _extract_inner_sheet, cut

    cutter = mu.duplicate_object(reference if reference is not None else part, "Mold_Offset_tmp", part.users_collection[0])
    group = None
    try:
        _extract_inner_sheet(cutter, up, outline, frame, z_lo, z_hi, is_cap,
                             backing_trim=max(.05, voxel*6.) if reference is not None else 0.)
        bm = bmesh.new()
        bm.from_mesh(cutter.data)
        bm.transform(cutter.matrix_world)
        border = [e for e in bm.edges if e.is_boundary]
        if not border:
            bm.free()
            raise RuntimeError("The mold has no open inner-sheet boundary to extend")
        if reference is not None:
            # A perforated remnant of the block is not a valid inner sheet.
            # Extruding all its little loops produces extra rims and fins.
            remaining=set(border);stack=[remaining.pop()]
            while stack:
                edge=stack.pop()
                for vertex in edge.verts:
                    for adjacent in vertex.link_edges:
                        if adjacent in remaining:
                            remaining.remove(adjacent);stack.append(adjacent)
            if remaining or any(sum(e.is_boundary for e in v.link_edges)!=2 for e in border for v in e.verts):
                bm.free()
                raise RuntimeError('Cannot form a clean shell perimeter from this cavity; rebuild the mold from its source surface')
        right, fwd = frame[1:]
        center = Vector((sum(x for x, y in outline) / len(outline),
                         sum(y for x, y in outline) / len(outline)))
        # Extend the reference beyond the original perimeter, and close it far
        # behind the cavity. Eroding this reference must not leave a bottom or
        # side wall in the actual mold. Only the temporary reference is extended.
        reach = wall * 3.0 + (1.0 if fixed_grid else 10.0)
        extended, end = {}, {}
        for v in {v for e in border for v in e.verts}:
            xy = Vector((v.co.dot(right), v.co.dot(fwd)))
            direction = (xy - center).normalized()
            extended[v] = bm.verts.new(v.co + right * direction.x * reach + fwd * direction.y * reach)
            q = extended[v].co
            z = z_hi + reach if is_cap else z_lo - reach
            end[v] = bm.verts.new(q + up * (z - q.dot(up)))
        for edge in border:
            a, b = edge.verts
            bm.faces.new((a, b, extended[b], extended[a]))
            bm.faces.new((extended[a], extended[b], end[b], end[a]))
        bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.transform(cutter.matrix_world.inverted())
        bm.to_mesh(cutter.data)
        bm.free()
        cutter.data.update()
        if not mu.is_closed(cutter):
            raise RuntimeError("Could not close the extended shell reference")

        # This grid controls only the backing. The original detailed cavity and
        # the straight perimeter planes are retained by the final difference.
        used = fixed_grid if fixed_grid else mu.safe_voxel(cutter, min(2.0 * voxel, wall / 6.0))
        group = bpy.data.node_groups.new("Mold shell offset", 'GeometryNodeTree')
        group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
        source = group.nodes.new('NodeGroupInput')
        output = group.nodes.new('NodeGroupOutput')
        grid = group.nodes.new('GeometryNodeMeshToSDFGrid')
        grid.inputs['Voxel Size'].default_value = used
        offset = group.nodes.new('GeometryNodeSDFGridOffset')
        offset.inputs['Distance'].default_value = -wall
        mesh = group.nodes.new('GeometryNodeGridToMesh')
        mesh.inputs['Threshold'].default_value = 0.0
        mesh.inputs['Adaptivity'].default_value = 0.03
        links = group.links
        links.new(source.outputs['Geometry'], grid.inputs['Mesh'])
        links.new(grid.outputs['SDF Grid'], offset.inputs['Grid'])
        links.new(offset.outputs['Grid'], mesh.inputs['Grid'])
        links.new(mesh.outputs['Mesh'], output.inputs['Geometry'])
        mu.apply_modifier(cutter, 'NODES', lambda mod: setattr(mod, 'node_group', group))
        bpy.data.node_groups.remove(group);group=None
        bpy.context.view_layer.update()
        faces, nonman, boundary, loose = mu.mesh_stats(cutter)
        if not faces or nonman or boundary or loose:
            raise RuntimeError("The shell backing offset did not produce a closed solid")
        if return_cutter:
            cutter["shell_grid_mm"]=used
            result=cutter;cutter=None
            return result
        cut(part, cutter, 'DIFFERENCE', voxel=voxel)
        mu.keep_main_body(part)
        faces, nonman, boundary, loose = mu.mesh_stats(part)
        if not faces or nonman or boundary or loose:
            raise RuntimeError("The shell difference did not produce a closed mold half")
        return used
    finally:
        if cutter is not None:mu.delete_object(cutter)
        if group is not None:
            bpy.data.node_groups.remove(group)
