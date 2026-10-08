"""View-only framing and a view-aligned millimetre grid."""
import math
import bpy
from mathutils import Vector

_grid_handle=None


def surface_object(o):
    return (o.type=='MESH' and not o.name.startswith(('LM_','AL_S_','AL_T_','AL_FAVOUR','MG_'))
            and not o.get('nose_landmark') and not o.get('landmark') and not o.get('anaplast_marker')
            and not any(c.name in {'Anaplast_Landmarks','Anaplast_AlignPoints','Anaplast_Frame'} for c in o.users_collection))


def case_targets(context):
    p=context.scene.anaplast
    from .sculpt_surface import target
    for obj in (p.face_scan_obj,target(context),p.cast_obj):
        if obj and obj.visible_get(view_layer=context.view_layer):return [obj]
    return context.view_layer.objects


def fit(context,objects):
    from .overlay import clipping_area
    area=clipping_area(context)
    if not area:raise RuntimeError('Use Fit in a 3D viewport')
    objects=[o for o in objects if surface_object(o) and o.visible_get(view_layer=context.view_layer)]
    if not objects:raise RuntimeError('No visible meshes to fit')
    import numpy as np
    space=area.spaces.active;rv=space.region_3d
    q=rv.view_rotation.inverted();points=[]
    dg=context.evaluated_depsgraph_get()
    for obj in objects:
        # Actual vertices avoid the empty corners of a rotated bounding box.
        ev=obj.evaluated_get(dg) if obj.modifiers else obj
        me=ev.data
        coords=np.empty((len(me.vertices),3),dtype=np.float32);me.vertices.foreach_get('co',coords.ravel())
        matrix=np.asarray(q.to_matrix().to_4x4()@obj.matrix_world)
        if len(coords):points.append(coords@matrix[:3,:3].T+matrix[:3,3])
    if not points:raise RuntimeError('No mesh vertices to fit')
    points=np.concatenate(points)
    center=(points.min(axis=0)+points.max(axis=0))*.5
    points-=center
    region=next((x for x in area.regions if x.type=='WINDOW'),None)
    width=max(1,region.width) if region else 1000;height=max(1,region.height) if region else 1000
    left,right,bottom,top=visible_rect(area,region)
    # Blender's viewport uses a 36 mm sensor and a 2x view-plane zoom factor.
    px=space.lens/36*max(1,height/width);py=space.lens/36*max(1,width/height)
    xmin,xmax=2*(left+6)/width-1,2*(right-6)/width-1
    ymin,ymax=2*(bottom+6)/height-1,2*(top-6)/height-1
    sx,sy=(xmax-xmin)*.5,(ymax-ymin)*.5
    ox,oy=(xmax+xmin)*.5,(ymax+ymin)*.5
    # Center the model in the unobscured part of the window, including a narrow sidebar.
    if rv.view_perspective=='ORTHO':
        dist=max(float(np.abs(points[:,0]).max())*px/sx,float(np.abs(points[:,1]).max())*py/sy,.001)
    else:
        z=points[:,2]
        dist=max(float(np.max((px*points[:,0]+(sx+ox)*z)/sx)),float(np.max((-px*points[:,0]+(sx-ox)*z)/sx)),float(np.max((py*points[:,1]+(sy+oy)*z)/sy)),float(np.max((-py*points[:,1]+(sy-oy)*z)/sy)),float(z.max())+space.clip_start*1.1,.001)
    center[0]-=ox*dist/px;center[1]-=oy*dist/py
    rv.view_location=rv.view_rotation@Vector(center)
    rv.view_distance=dist
    if rv.view_perspective=='CAMERA':rv.view_perspective='PERSP'
    area.tag_redraw()


def visible_rect(area,region):
    if region is None:return 0,1000,0,1000
    left,right,bottom,top=0,region.width,0,region.height
    for other in area.regions:
        if other.type not in {'UI','TOOLS','HEADER','TOOL_HEADER','ASSET_SHELF','ASSET_SHELF_HEADER'}:continue
        if other.width<2 or other.height<2:continue
        x0=max(0,other.x-region.x);x1=min(region.width,other.x+other.width-region.x)
        y0=max(0,other.y-region.y);y1=min(region.height,other.y+other.height-region.y)
        if x1<=x0 or y1<=y0:continue
        if other.type in {'UI','TOOLS'}:
            if x0==0:left=max(left,x1)
            elif x1==region.width:right=min(right,x0)
        elif y0==0:bottom=max(bottom,y1)
        elif y1==region.height:top=min(top,y0)
    if right-left<30 or top-bottom<30:return 0,region.width,0,region.height
    return left,right,bottom,top


def grid_lines(context):
    p=context.scene.anaplast;rv=context.space_data.region_3d
    unit=context.scene.unit_settings.scale_length*1000
    step=p.grid_spacing/max(unit,1e-9)
    right=rv.view_rotation@Vector((1,0,0));up=rv.view_rotation@Vector((0,1,0))
    center=rv.view_location.copy();extent=max(rv.view_distance*2,step*10)
    # Coarsen by integer multiples when zoomed out instead of drawing millions of lines.
    stride=max(1,math.ceil(extent/step/120));step*=stride
    n=min(120,math.ceil(extent/step));lines=[]
    for i in range(-n,n+1):
        offset=i*step
        lines.extend((center+right*offset-up*extent,center+right*offset+up*extent,
                      center+up*offset-right*extent,center+up*offset+right*extent))
    return lines,stride


def draw_grid():
    import gpu
    from gpu_extras.batch import batch_for_shader
    c=bpy.context
    if not c.area or c.area.type!='VIEW_3D' or not c.scene.anaplast.grid_front:return
    coords,_=grid_lines(c);shader=gpu.shader.from_builtin('UNIFORM_COLOR')
    depth=gpu.state.depth_test_get();blend=gpu.state.blend_get()
    try:
        gpu.state.depth_test_set('NONE');gpu.state.blend_set('ALPHA')
        shader.bind();shader.uniform_float('color',(.48,.62,.72,c.scene.anaplast.grid_opacity))
        batch_for_shader(shader,'LINES',{'pos':coords}).draw(shader)
    finally:gpu.state.depth_test_set(depth);gpu.state.blend_set(blend)


def register_grid():
    global _grid_handle
    if not bpy.app.background and _grid_handle is None:
        _grid_handle=bpy.types.SpaceView3D.draw_handler_add(draw_grid,(),'WINDOW','POST_VIEW')


def unregister_grid():
    global _grid_handle
    if _grid_handle is not None:bpy.types.SpaceView3D.draw_handler_remove(_grid_handle,'WINDOW');_grid_handle=None
