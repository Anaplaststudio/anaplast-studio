"""Three-piece backing: preserve all cavity and mating surfaces, open outside."""
import bpy,bmesh,numpy as np
from mathutils import Vector
from ..utils import mesh as mu


def erode(obj,wall,voxel):
    used=mu.safe_voxel(obj,min(voxel*2.,wall/6.))
    if used>wall/3.:raise RuntimeError('The requested shell is too thin for this case size; increase its wall thickness')
    group=bpy.data.node_groups.new('Auricular shell backing','GeometryNodeTree')
    try:
        group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
        group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
        source=group.nodes.new('NodeGroupInput');output=group.nodes.new('NodeGroupOutput')
        grid=group.nodes.new('GeometryNodeMeshToSDFGrid');grid.inputs['Voxel Size'].default_value=used
        offset=group.nodes.new('GeometryNodeSDFGridOffset');offset.inputs['Distance'].default_value=-wall
        mesh=group.nodes.new('GeometryNodeGridToMesh');mesh.inputs['Adaptivity'].default_value=.03
        links=group.links;links.new(source.outputs['Geometry'],grid.inputs['Mesh']);links.new(grid.outputs['SDF Grid'],offset.inputs['Grid']);links.new(offset.outputs['Grid'],mesh.inputs['Grid']);links.new(mesh.outputs['Mesh'],output.inputs['Geometry'])
        mu.apply_modifier(obj,'NODES',lambda mod:setattr(mod,'node_group',group))
    finally:bpy.data.node_groups.remove(group)
    if not len(obj.data.polygons) or any(mu.mesh_stats(obj)[1:]):raise RuntimeError('Shell backing did not form a closed cutter')
    return used


def wedge_cutter(part,outline,frame,wall,voxel):
    """Continue only the external rim before erosion; keep both contact skins."""
    obj=mu.duplicate_object(part,'Auricular_ShellCutter_work',part.users_collection[0]);bm=bmesh.new()
    try:
        bm.from_mesh(obj.data);bm.transform(obj.matrix_world);bm.verts.ensure_lookup_table();bm.verts.index_update()
        u,r,f=frame;world=np.asarray([v.co[:] for v in bm.verts]);xy=np.c_[world@np.asarray(r),world@np.asarray(f)];poly=np.asarray(outline);members=[set() for v in bm.verts]
        for k,(a,b) in enumerate(zip(poly,np.roll(poly,-1,axis=0))):
            delta=b-a;length=np.linalg.norm(delta)
            if length<1e-8:continue
            for i in np.flatnonzero(np.abs((xy[:,0]-a[0])*delta[1]-(xy[:,1]-a[1])*delta[0])<.002*length):members[int(i)].add(k)
        doomed=[face for face in bm.faces if set.intersection(*(members[v.index] for v in face.verts))]
        if not doomed:raise RuntimeError('Cannot locate the wedge outside edge; rebuild the three-piece mold')
        doomed_set=set(doomed)
        border=[e for e in bm.edges if sum(face in doomed_set for face in e.link_faces)==1]
        if not border:raise RuntimeError('Cannot open the wedge at the mold rim')
        center=poly.mean(axis=0);reach=wall*3.+10.;extended={}
        for v in {v for face in doomed for v in face.verts}:
            q=np.array([v.co.dot(r),v.co.dot(f)]);direction=q-center;direction/=max(np.linalg.norm(direction),1e-8)
            extended[v]=bm.verts.new(v.co+r*float(direction[0]*reach)+f*float(direction[1]*reach))
        for face in doomed:bm.faces.new(tuple(extended[v] for v in face.verts))
        for e in border:
            a,b=e.verts;bm.faces.new((a,b,extended[b],extended[a]))
        bmesh.ops.delete(bm,geom=doomed,context='FACES_ONLY')
        orphan_edges=[e for e in bm.edges if not e.link_faces]
        if orphan_edges:bmesh.ops.delete(bm,geom=orphan_edges,context='EDGES')
        bmesh.ops.recalc_face_normals(bm,faces=bm.faces)
        orphan=[v for v in bm.verts if not v.link_faces]
        if orphan:bmesh.ops.delete(bm,geom=orphan,context='VERTS')
        bm.transform(obj.matrix_world.inverted());bm.to_mesh(obj.data);obj.data.update()
        if any(mu.mesh_stats(obj)[1:]):raise RuntimeError('Cannot close the extended wedge backing')
        used=erode(obj,wall,voxel);return obj,used
    except Exception:mu.delete_object(obj);raise
    finally:bm.free()


def make_parts(parts,walls,voxel):
    from . import mold_auricular as aw
    from .mold_shell import build_shell
    base,wedge,cap=parts;frame=aw.frame_for(base);u,r,f=frame
    outline=aw.offset_polygon([list(v) for v in base['mold_hull']],float(base['mold_skin'])+float(base['mold_land']))
    results=[];report={}
    try:
        for part,wall,label in zip(parts,walls,('base','wedge','cap')):
            obj=mu.duplicate_object(part,'Auricular_Shell_'+label+'_work',part.users_collection[0]);results.append(obj);cutter=None
            if wall<=0:
                obj['auricular_shell']=False;report[label]={'body':'solid'};continue
            try:
                if label=='wedge':cutter,used=wedge_cutter(obj,outline,frame,wall,voxel)
                else:
                    levels=[(obj.matrix_world@v.co).dot(u) for v in obj.data.vertices]
                    cutter=build_shell(obj,u,wall,outline,frame,min(levels),max(levels),label=='cap',voxel,return_cutter=True)
                    used=float(cutter.get('shell_grid_mm',min(voxel*2.,wall/6.)))
                aw.cut(obj,cutter,'DIFFERENCE')
            finally:
                if cutter is not None:mu.delete_object(cutter)
            cleanup=aw.remove_surface_slivers(obj,(part,))
            print('Shell cleanup',label,cleanup,flush=True)
            fragments=aw.separate_islands(obj,obj.users_collection[0])
            if fragments is not None:
                # Occasionally the SDF/Boolean intersection leaves a microscopic
                # inverted cell. Remove only a sub-0.03 mm closed fragment with
                # negligible volume; substantive disconnected material is rejected.
                bm=bmesh.new();bm.from_mesh(fragments.data);tiny_volume=abs(bm.calc_volume(signed=True));bm.free()
                lo,hi=mu.world_bbox(fragments);tiny=(hi-lo).length<.03 and tiny_volume<1e-5
                mu.delete_object(fragments)
                if not tiny:raise RuntimeError(f'The {label} shell has detached material; increase its thickness')
                cleanup['microscopic_fragment_volume_mm3']=tiny_volume
            stats=mu.mesh_stats(obj)
            if not stats[0] or any(stats[1:]):raise RuntimeError(f'The {label} shell is not a closed printable solid')
            bm=bmesh.new();bm.from_mesh(obj.data);volume=bm.calc_volume(signed=True);bm.free()
            if volume<=0:raise RuntimeError(f'The {label} shell has invalid volume')
            obj['auricular_shell']=True;obj['shell_wall_mm']=wall
            report[label]={'nominal_wall_mm':wall,'backing_grid_mm':used,'closed':True,'connected_components':1,'volume_mm3':volume,'numerical_cleanup':cleanup,'opening':'outside rim' if label=='wedge' else 'back and outside rim'}
            print('Auricular shell',label,report[label],flush=True)
        return results,report
    except Exception:
        for obj in results:mu.delete_object(obj)
        raise


LABELS=('Base','Wedge','Cap')

def walls(p):
    w=bpy.context.scene.mold_workflow
    if w.enabled:return tuple(getattr(w,k+'_wall') if getattr(w,k+'_body')=='SHELL' else 0 for k in ('base','wedge','cap'))
    return (p.wedge_base_wall,p.wedge_wall,p.wedge_cap_wall)


def wants_shell(p):
    w=bpy.context.scene.mold_workflow
    return any(walls(p)) if w.enabled else p.wedge_body=='SHELL' 


def solid_source(obj):
    if not obj.get('auricular_shell'):return obj
    saved=bpy.data.objects.get('Auricular_Solid'+obj.name.removeprefix('Mold_')+'_Reference')
    if saved is None:raise RuntimeError('Rebuild Base > Wedge > Cap to recover the solid construction reference')
    return saved


def cache_solids(parts,col):
    staged=[]
    try:
        for part,label in zip(parts,LABELS):
            obj=mu.duplicate_object(part,'Auricular_SolidCache_work',col);obj['auricular_shell']=False;obj.hide_set(True);obj.hide_render=True;staged.append((obj,'Auricular_Solid'+label+'_Reference'))
        for obj,name in staged:
            old=bpy.data.objects.get(name)
            if old:mu.delete_object(old)
            obj.name=name
        return [obj for obj,_ in staged]
    except Exception:
        for obj,_ in staged:
            if obj.name in bpy.data.objects:mu.delete_object(obj)
        raise


def update_bodies(context):
    import json
    from . import mold_auricular as aw
    from .explode import ensure_collapsed
    ensure_collapsed(context.scene);p=context.scene.anaplast
    parts=[bpy.data.objects.get('Mold_'+label) for label in LABELS]
    if any(o is None or not o.get('three_piece') for o in parts):raise RuntimeError('Finish Base > Wedge > Cap first')
    sources=[solid_source(o) for o in parts];result=[]
    info=json.loads(parts[1]['three_piece_report'])
    try:
        if wants_shell(p):
            result,report=make_parts(sources,walls(p),p.m_voxel);info['shells']=report
        else:info.pop('shells',None)
        cache_solids(sources,parts[0].users_collection[0])
        # cache_solids may replace the old cached input objects. Resolve again.
        if not result:result=[mu.duplicate_object(bpy.data.objects['Auricular_Solid'+label+'_Reference'],'Auricular_Restore_work',parts[0].users_collection[0]) for label in LABELS]
        for obj,new in zip(parts,result):
            old=obj.data;obj.data=new.data.copy();obj.matrix_world=new.matrix_world.copy()
            if old.users==0:bpy.data.meshes.remove(old)
            obj['auricular_shell']=bool(new.get('auricular_shell'));obj['three_piece_report']=json.dumps(info);aw.shade_part(obj,aw.frame_for(parts[0])[0])
        p.wedge_report='Three-piece shells updated. Wedge opens at the outside edge.' if p.wedge_body=='SHELL' else 'Three-piece solid bodies restored.'
    finally:
        for obj in result:
            if obj.name in bpy.data.objects:mu.delete_object(obj)
