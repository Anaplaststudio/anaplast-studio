"""Refit the rounded peripheral body while retaining rigid iris/cornea gaze."""
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree
from ..utils import mesh as mu


def parts(scene):return (scene.anaplast.ocular_obj,scene.ocular_production.color_obj,scene.ocular_production.clear_obj)


def positions(mesh):
    values=np.empty(len(mesh.vertices)*3,np.float64);mesh.vertices.foreach_get('co',values)
    return values.reshape((-1,3))


def restore_unfitted(scene):
    for obj in parts(scene):
        if obj and obj.get('gaze_basis_mesh') and obj.data!=obj['gaze_basis_mesh']:
            check_cached_geometry(obj)
    for obj in parts(scene):
        if not obj:continue
        basis=obj.get('gaze_basis_mesh')
        if basis:
            obj.data=basis
            if obj.get('gaze_basis_signature'):obj['built_geometry']=obj['gaze_basis_signature']
        obj['gaze_edge_fitted']=False;obj['gaze_edge_pending']=True


def check_cached_geometry(obj):
    from .ocular_production import geometry_signature
    basis=obj.get('gaze_basis_mesh')
    if not basis:return
    expected=obj.get('gaze_basis_signature') if obj.data==basis else obj.get('gaze_edge_signature')
    if expected and geometry_signature(obj)!=expected:
        raise RuntimeError('Ocular geometry was edited after fitting; update the ocular before adjusting gaze')


def refit(context):
    from .ocular_view import prepare_gaze
    from .ocular_steps import ball_data
    from .ocular_production import geometry_signature
    from .mold_design import has_self_intersections
    prepare_gaze(context);ball_data(context)
    s=context.scene;source,core,clear=parts(s);objects=(source,core,clear)
    for obj in objects:check_cached_geometry(obj)
    if not source.get('workflow_sphere_radius_mm'):raise RuntimeError('Generate the scan-based ocular before refitting its border')
    bases={o:o.get('gaze_basis_mesh') or o.data for o in objects}
    src_mesh=bases[source];co=positions(src_mesh)
    front=src_mesh.attributes.get('ocular_front');pair=src_mesh.attributes.get('ocular_back_index')
    if not front or not pair:raise RuntimeError('Rebuild the ocular to record its rounded border')
    backs={item.value-1 for item in pair.data if item.value>0}
    walls=[tuple(face.vertices) for face in src_mesh.polygons if front.data[face.index].value!=1 and not all(i in backs for i in face.vertices)]
    if not walls:raise RuntimeError('The ocular has no recorded rounded border')
    wall_ids=sorted({i for face in walls for i in face})
    tree=BVHTree.FromPolygons(co.tolist(),walls,all_triangles=True)
    u=np.array(source['ocular_up']);r=np.array(source['ocular_right']);f=np.cross(u,r)
    cc=np.array(source['ocular_iris_center']);radius=float(source['workflow_cornea_radius_mm'])+.1
    p=s.ocular_production
    plane=core.get('iris_bed_plane')
    if not plane:raise RuntimeError('Build the recessed iris before refitting the ocular border')
    protected_z=float(plane[0])-p.clear_thickness-p.iris_depth-p.pupil_depth-.5
    staged=[];old_data={o:o.data for o in objects}
    basis_signatures={o:o.get('gaze_basis_signature') or geometry_signature(o) for o in objects}
    try:
        for obj in objects:
            original=positions(bases[obj]);xy=np.c_[(original-cc)@r,(original-cc)@f]
            radial=np.maximum(np.linalg.norm(xy,axis=1)-radius,0.)
            below=np.maximum(protected_z-original@u,0.)
            protected_distance=np.hypot(radial,below)
            wall_distance=np.array([tree.find_nearest(pt)[3] for pt in original])
            if np.any((wall_distance<1e-5)&(protected_distance<1e-5)):
                raise RuntimeError('The cornea is too close to the rounded border to refit independently; increase the extension')
            t=protected_distance/np.maximum(protected_distance+wall_distance,1e-12)
            weight=t**3*(10-15*t+6*t*t)
            # One continuous deformation is shared by both material regions.
            # The rounded rim stays at its original scan-based position, while
            # the complete iris, pupil chamber and cornea remain rigidly rotated.
            matrix=np.array(obj.matrix_world);inverse=np.linalg.inv(matrix)
            rotated=original@matrix[:3,:3].T+matrix[:3,3]
            world=rotated+(original-rotated)*weight[:,None]
            local=world@inverse[:3,:3].T+inverse[:3,3]
            mesh=bases[obj].copy();mesh.name=obj.name+' gaze edge';mesh.vertices.foreach_set('co',local.ravel());mesh.update()
            if mesh.has_custom_normals:mesh.normals_split_custom_set(np.zeros((len(mesh.loops),3)))
            candidate=bpy.data.objects.new('Ocular edge validation',mesh);staged.append((obj,candidate,mesh))
            if obj==clear and clear.get('cornea_surface_version'):
                from .cornea_surface import smooth_normals
                smooth_normals(candidate,source)
            if any(mu.mesh_stats(candidate)[1:]) or has_self_intersections(candidate):
                raise RuntimeError('This gaze leaves too little room for a smooth closed border; reduce the X/Z rotation and apply again')
            stored_world=positions(mesh)@matrix[:3,:3].T+matrix[:3,3]
            if obj==source:
                error=float(np.max(np.linalg.norm(stored_world[wall_ids]-original[wall_ids],axis=1)))
                if error>.001:raise RuntimeError('The original ocular border could not be retained accurately')
            candidate['protected_error']=float(np.max(np.linalg.norm(stored_world[protected_distance<1e-9]-rotated[protected_distance<1e-9],axis=1))) if np.any(protected_distance<1e-9) else 0.
        # Commit only after all three closed meshes validate.
        for obj,candidate,mesh in staged:
            if not obj.get('gaze_basis_mesh'):
                obj['gaze_basis_mesh']=bases[obj]
                obj['gaze_basis_signature']=basis_signatures[obj]
            obj.data=mesh
            obj['gaze_edge_signature']=geometry_signature(obj)
            if obj!=source:obj['built_geometry']=obj['gaze_edge_signature']
            obj['gaze_edge_fitted']=True;obj['gaze_edge_pending']=False
            obj['gaze_edge_border_error_mm']=error
            obj['gaze_edge_protected_error_mm']=candidate['protected_error']
        for obj,previous in old_data.items():
            if previous!=bases[obj] and previous.users==0:bpy.data.meshes.remove(previous)
        s.ocular_view.gaze_message='Border refitted; original extension retained.'
        return error
    finally:
        for obj,candidate,mesh in staged:
            bpy.data.objects.remove(candidate,do_unlink=True)
            if mesh.users==0:bpy.data.meshes.remove(mesh)
