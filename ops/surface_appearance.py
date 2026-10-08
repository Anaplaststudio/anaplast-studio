"""Independent Workbench appearance without changing production shader connections.

Object-linked material copies hold an unconnected active preview image. Original
shader nodes, texture images and UV render selection are left intact.
"""
import bpy,math
import numpy as np

PREVIEW='Anaplast Studio viewport color (unconnected)'
PREVIEW_UV='Anaplast Studio View Colors'
LEGACY_PREVIEW='B4D viewport color (unconnected)'
LEGACY_UV='B4D View Colors'


def preview_node(nodes):
    node=nodes.get(PREVIEW) or nodes.get(LEGACY_PREVIEW)
    if node is None:node=nodes.new('ShaderNodeTexImage')
    node.name=PREVIEW
    return node


def preview_uv(obj):
    uv=obj.data.uv_layers.get(PREVIEW_UV) or obj.data.uv_layers.get(LEGACY_UV)
    if uv and uv.name==LEGACY_UV:
        uv.name=PREVIEW_UV
        # Keep explicit material references working when reusing an old atlas.
        for slot in obj.material_slots:
            mat=slot.material
            if mat and mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type=='UVMAP' and node.uv_map==LEGACY_UV:node.uv_map=uv.name
        if obj.get('b4d_original_active_uv')==LEGACY_UV:obj['b4d_original_active_uv']=uv.name
    return uv

def original_material(mat):
    return mat.get('b4d_original_material',mat) if mat else None


def scan_image(mat):
    mat=original_material(mat)
    if not mat or not mat.use_nodes:return None
    nodes=mat.node_tree.nodes
    # Prefer the actual base-color texture over bump/roughness images.
    bsdf=next((n for n in nodes if n.type=='BSDF_PRINCIPLED'),None)
    def upstream(socket,seen):
        for link in socket.links:
            n=link.from_node
            if n in seen:continue
            seen.add(n)
            if n.type=='TEX_IMAGE' and n.image:return n.image
            for inp in n.inputs:
                image=upstream(inp,seen)
                if image:return image
    if bsdf:
        found=upstream(bsdf.inputs['Base Color'],set())
        if found:return found
    active=nodes.active
    if active and active.type=='TEX_IMAGE' and active.image:return active.image
    return next((n.image for n in nodes if n.type=='TEX_IMAGE' and n.image and n.name not in {PREVIEW,LEGACY_PREVIEW}),None)


def color_source(obj):
    if obj.type!='MESH':return None
    if obj.data.uv_layers and any(scan_image(slot.material) for slot in obj.material_slots):return 'TEXTURE'
    attrs=obj.data.color_attributes
    if attrs and (attrs.active_color or len(attrs)):return 'VERTEX'
    return None


def vertex_image(obj):
    """Rasterize corner colors into a dedicated preview atlas, preserving original UVs.

    Each loop triangle gets a padded four-pixel tile. Colors vary linearly across
    the tile, so bilinear sampling reproduces the triangle's vertex interpolation.
    """
    me=obj.data
    if me.users>1:obj.data=me=me.copy()
    attr=me.color_attributes.active_color or me.color_attributes[0]
    me.calc_loop_triangles();tris=me.loop_triangles
    # Per-polygon corner UVs cannot represent two atlas locations for a shared
    # n-gon corner. Use a per-polygon tile with corner colors sampled below.
    # Triangle scans are exact; quads use four corners, as ordinary UV interpolation.
    count=len(me.polygons);width=max(1,math.ceil(math.sqrt(count)))*4
    height=max(4,math.ceil(count/(width//4))*4)
    rgba=np.ones((height,width,4),np.float32)
    uv=preview_uv(obj)
    oldrender=next((u for u in me.uv_layers if u.active_render),None)
    oldactive=me.uv_layers.active
    if oldactive and oldactive.name!=PREVIEW_UV:obj['b4d_original_active_uv']=oldactive.name
    if uv is None:uv=me.uv_layers.new(name=PREVIEW_UV)
    uvcoords=np.zeros((len(me.loops),2),np.float32)
    # Work in batches to keep high-resolution scans responsive and memory bounded.
    starts=np.empty(count,np.int32);sizes=np.empty(count,np.int32)
    me.polygons.foreach_get('loop_start',starts);me.polygons.foreach_get('loop_total',sizes)
    vertex_ids=np.empty(len(me.loops),np.int32);me.loops.foreach_get('vertex_index',vertex_ids)
    values=np.empty((len(attr.data),4),np.float32);attr.data.foreach_get('color',values.ravel())
    for n in (3,4):
        indices=np.flatnonzero(sizes==n)
        for offset in range(0,len(indices),32768):
            ids=indices[offset:offset+32768]
            loops=starts[ids,None]+np.arange(n)[None,:]
            colors=values[loops if attr.domain=='CORNER' else vertex_ids[loops]]
            xs=(ids%(width//4))*4;ys=(ids//(width//4))*4
            for yy in range(4):
                for xx in range(4):
                    u=xx-1;v=yy-1
                    if n==4:value=(1-u)*(1-v)*colors[:,0]+u*(1-v)*colors[:,1]+u*v*colors[:,2]+(1-u)*v*colors[:,3]
                    else:value=colors[:,0]+u*(colors[:,1]-colors[:,0])+v*(colors[:,2]-colors[:,0])
                    rgba[ys+yy,xs+xx]=value
            corners=[(1,1),(2,1),(2,2),(1,2)] if n==4 else [(1,1),(2,1),(1,2)]
            for j,(cx,cy) in enumerate(corners):
                uvcoords[loops[:,j],0]=(xs+cx+.5)/width
                uvcoords[loops[:,j],1]=(ys+cy+.5)/height
    for i in np.flatnonzero((sizes!=3)&(sizes!=4)):
        # Irregular n-gons get their average color; triangulated scans retain
        # interpolated per-corner detail through the path above.
        x=(i%(width//4))*4;y=(i//(width//4))*4
        loops=np.arange(starts[i],starts[i]+sizes[i])
        rgba[y:y+4,x:x+4]=values[loops if attr.domain=='CORNER' else vertex_ids[loops]].mean(axis=0)
        uvcoords[loops]=((x+2.)/width,(y+2.)/height)
    uv.data.foreach_set('uv',uvcoords.ravel());me.uv_layers.active=uv
    if oldrender:oldrender.active_render=True
    image=bpy.data.images.new(obj.name+' · Scan color preview',width,height,float_buffer=True)
    image.colorspace_settings.name='Non-Color';image.pixels.foreach_set(rgba.ravel());image.pack()
    obj['b4d_vertex_preview_image']=image
    return image


def ensure_materials(obj):
    if not obj.material_slots:
        # A neutral production material is equivalent to an unassigned surface.
        mat=bpy.data.materials.new(obj.name+' · Surface');obj.data.materials.append(mat)
    for slot in obj.material_slots:
        mat=slot.material
        if mat and mat.get('b4d_display_owner')==obj:
            preview_node(mat.node_tree.nodes)
            continue
        original=original_material(mat)
        copy=original.copy() if original else bpy.data.materials.new(obj.name+' · Surface')
        copy.name=obj.name+' · View'
        if original:copy['b4d_original_material']=original
        copy['b4d_display_owner']=obj
        was_nodes=original.use_nodes if original else False
        copy.use_nodes=True
        if original and not was_nodes:
            bsdf=copy.node_tree.nodes.get('Principled BSDF')
            if bsdf:
                bsdf.inputs['Base Color'].default_value=original.diffuse_color
                bsdf.inputs['Roughness'].default_value=original.roughness
                bsdf.inputs['Metallic'].default_value=original.metallic
        preview_node(copy.node_tree.nodes)
        slot.link='OBJECT';slot.material=copy


def update_object(obj):
    ensure_materials(obj)
    use=obj.b4d_scan_color;kind=color_source(obj)
    atlas=None
    if use and kind=='VERTEX':
        atlas=obj.get('b4d_vertex_preview_image')
        if not atlas:atlas=vertex_image(obj)
        uv=preview_uv(obj)
        if uv:obj.data.uv_layers.active=uv
    else:
        uv=obj.data.uv_layers.get(obj.get('b4d_original_active_uv',''))
        if uv:obj.data.uv_layers.active=uv
    for slot in obj.material_slots:
        mat=slot.material;nodes=mat.node_tree.nodes;node=nodes.get(PREVIEW)
        image=(scan_image(mat) if kind=='TEXTURE' else atlas) if use else None
        if image is None:
            image=mat.get('b4d_tint_image')
            if image is None:
                image=bpy.data.images.new(mat.name+' · Tint',1,1,float_buffer=True)
                image.colorspace_settings.name='Non-Color';mat['b4d_tint_image']=image
            image.pixels[:]=[*obj.color[:3],1.];image.update();image.pack()
        node.image=image;nodes.active=node
        mat.diffuse_color=tuple(obj.color)


def update(obj,context):
    if obj.type!='MESH':return
    from .viewport_tools import surface_object
    from .scene_helpers import is_construction
    scene=context.scene if context else bpy.context.scene
    # Workbench uses one global color mode. Give each working object its own
    # active preview image, so an untouched object's tint remains its own.
    for other in scene.objects:
        if surface_object(other) and not is_construction(other,scene):
            if other==obj or not all(slot.material and slot.material.get('b4d_display_owner')==other for slot in other.material_slots) or not other.material_slots:
                update_object(other)
    scene['b4d_object_appearance']=True
    if context and context.screen:
        for area in context.screen.areas:
            if area.type=='VIEW_3D':
                sh=area.spaces.active.shading;sh.type='SOLID';sh.color_type='TEXTURE';sh.show_xray=False
                area.tag_redraw()


def tint_get(obj):return obj.color[:3]
def tint_set(obj,value):obj.color=(*value,obj.color[3])
