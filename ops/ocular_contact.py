"""Disposable sclera-versus-Sculpt surface-distance inspection."""
import json
import numpy as np
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from . import ocular as oc,ocular_steps as steps,ocular_marking as mark,ocular_fit as fit
from ..utils import mesh as mu


def finish(context):
    v=context.scene.ocular_view
    for obj in list(context.scene.objects):
        if obj.get('ocular_fit_helper'):mu.delete_object(obj)
    for name,state in json.loads(v.fit_visibility or '{}').items():
        obj=context.scene.objects.get(name)
        if obj:obj.hide_set(state[0]);obj.hide_render=state[1];obj.display_type=state[2]
    if context.area and context.area.type=='VIEW_3D' and v.fit_shading:
        sh=context.space_data.shading
        for key,value in json.loads(v.fit_shading).items():setattr(sh,key,value)
    v.fit_visibility='';v.fit_shading='';v.fit_active=False


def reference_patch(source,selected):
    """Clip original Sculpt triangles at the painted half-weight boundary."""
    source.data.calc_loop_triangles()
    world=[source.matrix_world@v.co for v in source.data.vertices]
    vertices=[];faces=[];cache={}
    def vertex(a,b):
        key=(min(a,b),max(a,b))
        if key not in cache:
            cache[key]=len(vertices);vertices.append((world[a]+world[b])*.5)
        return cache[key]
    for tri in source.data.loop_triangles:
        ids=list(tri.vertices);poly=[]
        for j,a in enumerate(ids):
            b=ids[(j+1)%3]
            if selected[a]:poly.append(vertex(a,a))
            if selected[a]!=selected[b]:poly.append(vertex(a,b))
        for j in range(1,len(poly)-1):faces.append((poly[0],poly[j],poly[j+1]))
    if not faces:raise RuntimeError('Paint the Sculpt sclera before checking its contact')
    return vertices,faces


def area_samples(coords,faces):
    """Three interior samples per triangle; weights sum to surface area."""
    triangles=np.array([(face[0],face[j],face[j+1]) for face in faces for j in range(1,len(face)-1)])
    points=coords[triangles]
    areas=np.linalg.norm(np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0]),axis=1)*.5
    bary=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    return np.einsum('sk,tkd->tsd',bary,points).reshape(-1,3),np.repeat(areas/3,3)


def surface_distances(coords,tree,footprint,u,r,f):
    distance=np.zeros(len(coords));overlap=np.zeros(len(coords),bool)
    for k,co in enumerate(coords):
        hit=tree.find_nearest(Vector(co))
        normal=hit[1] if hit[1].dot(Vector(u))>=0 else -hit[1]
        distance[k]=hit[3] if (Vector(co)-hit[0]).dot(normal)>=0 else -hit[3]
        projected=footprint.find_nearest(Vector((float(co@r),float(co@f),0.)))
        overlap[k]=projected[0] is not None and projected[3]<1e-5
    return distance,overlap


def build(context):
    finish(context)
    s=context.scene;v=s.ocular_view;p=s.ocular_production
    marks,source,ball,u,r,f=steps.ball_data(context);eye=s.anaplast.ocular_obj
    if not eye or 'ocular_front' not in eye.data.attributes:raise RuntimeError('Generate the ocular first')
    # Use the uncut Sculpt and only its accepted painted sclera, never lids
    # or the preview's hole. Iris marks cannot contribute to contact.
    selected=fit.accepted_mask(marks,ball)&~(mark.values_for(marks,'IRIS')>.5)
    vertices,patch_faces=reference_patch(source,selected)
    tree=BVHTree.FromPolygons(vertices,patch_faces,all_triangles=True)
    patch_coords=np.asarray(vertices)
    footprint=BVHTree.FromPolygons(np.c_[patch_coords@r,patch_coords@f,np.zeros(len(vertices))],patch_faces,all_triangles=True)
    coords=np.array([eye.matrix_world@vert.co for vert in eye.data.vertices])
    local=np.array([vert.co[:] for vert in eye.data.vertices]);ru=np.array(eye['ocular_right']);normal=np.array(eye['ocular_up']);fu=np.cross(normal,ru);cc=np.array(eye['ocular_iris_center'])
    rho=np.linalg.norm(np.c_[(local-cc)@ru,(local-cc)@fu],axis=1)
    sclera=rho>float(eye['workflow_cornea_radius_mm'])+.05
    faces=[tuple(face.vertices) for face,a in zip(eye.data.polygons,eye.data.attributes['ocular_front'].data) if a.value and all(sclera[i] for i in face.vertices)]
    ids=np.array(sorted(set(i for face in faces for i in face)))
    if not len(ids):raise RuntimeError('No sclera surface is available to check')
    distance,valid=surface_distances(coords[ids],tree,footprint,u,r,f)
    colors=np.tile((.58,.58,.58,1.),(len(ids),1));t=v.fit_tolerance
    touch=np.abs(distance)<=t;above=valid&(distance>t);below=valid&(distance<-t)
    colors[touch]=(.08,.85,.12,1);colors[above]=(.05,.3,1,1);colors[below]=(1,.08,.03,1)
    samples,weights=area_samples(coords,faces)
    sampled,supported=surface_distances(samples,tree,footprint,u,r,f)
    total=float(weights.sum());contact=float(weights[np.abs(sampled)<=t].sum())
    unmatched=float(weights[~supported&(np.abs(sampled)>t)].sum())
    mesh=bpy.data.meshes.new('Ocular_sclera_fit');index={int(i):k for k,i in enumerate(ids)}
    mesh.from_pydata(coords[ids],[],[tuple(index[i] for i in face) for face in faces]);mesh.update()
    attr=mesh.color_attributes.new(name='Sclera fit',type='FLOAT_COLOR',domain='POINT');attr.data.foreach_set('color',colors.ravel());mesh.color_attributes.active_color=attr
    for face in mesh.polygons:face.use_smooth=True
    obj=bpy.data.objects.new('Ocular sclera fit',mesh);mu.get_collection(s,oc.COLLECTION).objects.link(obj);obj['ocular_fit_helper']=True;obj['anaplast_construction']=True;obj.show_in_front=True
    mat=bpy.data.materials.new('Ocular fit colors');mat.use_nodes=True;node=mat.node_tree.nodes.new('ShaderNodeVertexColor');node.layer_name='Sclera fit';mat.node_tree.links.new(node.outputs['Color'],mat.node_tree.nodes['Principled BSDF'].inputs['Base Color']);mesh.materials.append(mat)
    objects=list({o.as_pointer():o for o in (eye,p.color_obj,p.clear_obj,p.orbital_preview,source,marks) if o}.values())
    v.fit_visibility=json.dumps({o.name:[o.hide_get(),o.hide_render,o.display_type] for o in objects})
    for o in objects:o.hide_set(o!=source)
    source.display_type='WIRE';source.hide_set(False)
    if context.area and context.area.type=='VIEW_3D':
        sh=context.space_data.shading;v.fit_shading=json.dumps({key:getattr(sh,key) for key in ('type','color_type')});sh.type='SOLID';sh.color_type='VERTEX'
    v.fit_report=f'Estimated sclera contact: {contact:.2f} / {total:.2f} mm² ({100*contact/max(total,1e-12):.1f}%) within {t:.2f} mm of the original painted Sculpt sclera. Total includes the scleral extension; iris/cornea excluded. Outside painted coverage and not contacting: {unmatched:.2f} mm².'
    obj['contact_area_mm2']=contact;obj['sclera_area_mm2']=total;obj['contact_percent']=100*contact/max(total,1e-12)
    obj['fit_reference']=source.name;obj['fit_reference_region']='PAINTED_SCLERA';obj['fit_tolerance_mm']=t
    obj['fit_report']=v.fit_report;obj['ocular_signature']=mark.wm.geometry_signature(eye);v.fit_active=True
    return obj,distance,valid


class OCULAR_OT_fit(bpy.types.Operator):
    bl_idname='anaplast.ocular_fit_check';bl_label='Check sclera fit';bl_options={'REGISTER','UNDO'}
    bl_description='Estimate contact area and percentage of ocular sclera against the original painted Sculpt sclera; includes the scleral extension in the total, excludes iris and cornea'
    restore:bpy.props.BoolProperty(default=False,options={'HIDDEN'})
    def execute(self,context):
        try:
            if self.restore:finish(context)
            else:build(context)
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        if not self.restore:self.report({'INFO'},context.scene.ocular_view.fit_report)
        return {'FINISHED'}
