"""Exact ocular-envelope impression using the final gaze and the cut Sculpt."""
import numpy as np
import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from . import ocular as oc,ocular_steps as steps,ocular_marking as mark,ocular_cut
from ..utils import mesh as mu


class Impression:
    def __init__(self,context,sculpt):
        marks,source,ball,u,r,f=steps.ball_data(context)
        p=context.scene.ocular_production;eye=context.scene.anaplast.ocular_obj
        if sculpt!=source:
            raise RuntimeError('Select the original Sculpt used to mark the ocular; its cut preview is used automatically for the impression')
        if not eye or not eye.get('workflow_has_iris'):raise RuntimeError('Generate the ocular and cornea before building their cap impression')
        if eye.get('gaze_edge_pending'):raise RuntimeError('Apply gaze & refit edge before building the ocular impression')
        if eye.modifiers or any(mu.mesh_stats(eye)[1:]):raise RuntimeError('The ocular envelope must be a closed mesh with applied modifiers')
        self.preview=steps.preview_sclera(context)
        if not self.preview['iris_removed_after_transfer']:raise RuntimeError('Update the ocular to transfer the current iris marks before building its impression')
        self.eye=eye;self.r=r;self.f=f
        self.voxel=context.scene.anaplast.m_voxel
        world=np.array([marks.matrix_world@v.co for v in marks.data.vertices])
        selected=(mark.values_for(marks,'SCLERA')>.5)|(mark.values_for(marks,'IRIS')>.5)
        if 'fit_sclera_indices' in ball:
            from .ocular_fit import accepted_mask
            selected=accepted_mask(marks,ball)|(mark.values_for(marks,'IRIS')>.5)
        selected,_=ocular_cut.fill_cut_islands(marks,selected,world)
        loop=max(mark.contours(marks,selected,world),key=len)
        self.center=Vector(loop.mean(axis=0));curve=mark.smooth_loop(loop,.25,512)
        self.polygon=np.c_[curve@r,curve@f]
        self.signature=mark.wm.geometry_signature(eye)

    def cutter(self,name,up,depth,roi,col):
        from .mold_auto import sculpt_solid_toward
        eye=self.eye;normal_matrix=eye.matrix_world.to_3x3().inverted().transposed()
        points=[eye.matrix_world@v.co for v in eye.data.vertices]
        faces=[tuple(p.vertices) for p in eye.data.polygons if (normal_matrix@p.normal).dot(up)>1e-6]
        if not faces:raise RuntimeError('The ocular does not face the cap opening direction')
        mesh=bpy.data.meshes.new('Ocular_front_tmp');mesh.from_pydata(points,[],faces);mesh.update()
        sheet=bpy.data.objects.new('Ocular_front_tmp',mesh);col.objects.link(sheet)
        sculpt=None;ocular=None
        try:
            sculpt=sculpt_solid_toward(name,self.preview,up,depth,roi,col,max_faces=max(250000,len(self.preview.data.polygons)*2),preserve_hole=self.center)
            ocular=sculpt_solid_toward('Ocular_sweep_tmp',sheet,up,depth,roi,col,max_faces=max(250000,len(faces)*2))
            if any(mu.mesh_stats(sculpt)[1:]) or any(mu.mesh_stats(ocular)[1:]):raise RuntimeError('The cut Sculpt and ocular could not form closed impression cutters')
            from .mold_design import has_self_intersections
            # Scan folds can intersect their own downward sweep. Repair that
            # temporary scan cutter before inserting the exact ocular surface,
            # as in the ordinary cap path; never remesh the ocular cutter.
            if has_self_intersections(sculpt):
                mu.voxel_remesh(sculpt,self.voxel)
                self.repaired_sculpt=True
            # The repaired sweep contains non-planar quads. Letting the
            # Boolean choose their diagonals created tiny intersecting faces
            # at the ocular junction. Resolve triangulation consistently
            # before joining; all vertex positions stay unchanged.
            for operand in (sculpt,ocular):
                bm=bmesh.new()
                try:
                    bm.from_mesh(operand.data)
                    bmesh.ops.triangulate(bm,faces=bm.faces[:],quad_method='BEAUTY',ngon_method='EAR_CLIP')
                    bm.to_mesh(operand.data);operand.data.update()
                finally:bm.free()
            # Insert the original corneal surface after any scan repair.
            # No offset or clearance is introduced on the ocular impression.
            mu.apply_boolean(sculpt,ocular,'UNION',solver='MANIFOLD')
            if any(mu.mesh_stats(sculpt)[1:]):raise RuntimeError('The ocular impression did not join cleanly; check its fit against the Sculpt')
            result=sculpt;sculpt=None;return result
        finally:
            for obj in (sheet,ocular,sculpt):
                if obj:mu.delete_object(obj)

    def verify(self,cap,up):
        points=np.array([self.eye.matrix_world@v.co for v in self.eye.data.vertices])
        front=self.eye.data.attributes.get('ocular_front')
        if front is None:raise RuntimeError('Rebuild the ocular to identify its front surface')
        ids=set(i for p,a in zip(self.eye.data.polygons,front.data) if a.value for i in p.vertices)
        ids=np.array(sorted(ids));xy=np.c_[points[ids]@self.r,points[ids]@self.f]
        ids=ids[oc.signed_distance(xy,self.polygon)<-.35]
        ids=ids[::max(1,len(ids)//500)]
        tree=BVHTree.FromPolygons([cap.matrix_world@v.co for v in cap.data.vertices],[tuple(p.vertices) for p in cap.data.polygons])
        # Exclude points concealed behind the retained Sculpt. Those correctly
        # have a Sculpt impression instead of an ocular impression.
        skin=BVHTree.FromPolygons([self.preview.matrix_world@v.co for v in self.preview.data.vertices],[tuple(p.vertices) for p in self.preview.data.polygons])
        errors=[]
        for i in ids:
            point=Vector(points[i]);hit=skin.ray_cast(point+up*.01,up,100)
            if hit[0] is not None:continue
            hit=tree.find_nearest(point)
            if hit[0] is not None:errors.append(hit[3])
        if len(errors)<10:raise RuntimeError('Not enough exposed ocular surface to verify the cap impression')
        if max(errors)>.02:raise RuntimeError(f'Ocular impression is not accurate enough ({max(errors):.3f} mm); check ocular fit and mold direction')
        cap['ocular_impression']=True;cap['ocular_impression_clearance_mm']=0.
        cap['ocular_impression_source']=self.eye.name;cap['ocular_impression_signature']=self.signature
        cap['ocular_impression_max_error_mm']=max(errors)
        cap['ocular_impression_scan_repaired']=bool(getattr(self,'repaired_sculpt',False))

    def exclude_displacement(self,blank,ledger,col):
        from .mold_volume import volume
        occupied=mu.duplicate_object(self.eye,'Ocular_displacement_tmp',col)
        try:
            mu.apply_boolean(occupied,blank,'INTERSECT',solver='MANIFOLD')
            value=volume(occupied)
            if value>ledger.data['cavity']+.01:raise RuntimeError('Ocular displacement exceeds the mold cavity; check ocular fit')
            ledger.data['cavity']=max(0.,ledger.data['cavity']-value)
            ledger.data['ocular_displacement']=value
        finally:mu.delete_object(occupied)
