"""Keep construction geometry and landmark markers accessible but out of the way."""
import bpy,json


def set_hidden(obj,scene,hidden):
    for layer in scene.view_layers:
        layer.update()
        if obj.name in layer.objects:obj.hide_set(hidden,view_layer=layer)


def is_marker(o):
    return bool(o.get('anaplast_marker') or o.get('nose_landmark') or o.get('landmark') or
                o.name.startswith(('LM_','AL_S_','AL_T_','AL_FAVOUR','MG_')))


def is_construction(o,scene=None):
    if o.get('anaplast_part')=='INSERT' and o.get('insert_uid'):return False
    if o.name in {'Mold_Base','Mold_Wedge','Mold_Cap','Mold_A','Mold_B','Mold_C','Mold_D'}:return False
    if scene and hasattr(scene,'anaplast'):
        p=scene.anaplast
        if any(o==getattr(p,k,None) for k in ('face_scan_obj','cast_obj','cbct_obj','prosthesis_obj','source_prosthesis','ocular_scan')):return False
    name=o.name
    return bool(o.get('reference_only') or o.get('anaplast_part') in {'SOLID_TWIN','WEDGE_MARKS'} or
                o.get('wedge_edit_copy') or o.get('wedge_coverage_preview') or
                name in {'Wedge_Undercut_Suggestion','Wedge_Refined_Border'} or
                (name.startswith('Auricular_') and ('_Reference' in name or '_CapCut_' in name)) or
                name in {'Cast_Solid_Preview','Ocular_Fitted_Ball'} or o.get('anaplast_construction'))


GROUPS=('01  Scans','02  Mirrors','03  Photos','04  Sculpts','05  Prototype','06  Oculars','07  Substructure','08  Molds','99  Helpers')


def group(scene,name):
    col=next((c for c in scene.collection.children if c.get('anaplast_scene_group')==name),None)
    if col is None:
        col=bpy.data.collections.new(name);col['anaplast_scene_group']=name;scene.collection.children.link(col)
    return col


def move(obj,col):
    if col in obj.users_collection and len(obj.users_collection)==1:return
    for old in list(obj.users_collection):old.objects.unlink(obj)
    col.objects.link(obj)


def category(o,scene):
    p=scene.anaplast;role=o.get('anaplast_part','');name=o.name.lower()
    # Finished inserts in older cases can inherit the source reference tags.
    if role=='INSERT' and o.get('insert_uid'):return GROUPS[7]
    if o.get('workflow_transaction'):return GROUPS[-1]
    if is_marker(o) or is_construction(o,scene):return GROUPS[-1]
    if o.type=='EMPTY' and o.empty_display_type=='IMAGE':return GROUPS[2]
    if role in {'SCULPT','SCULPT_REGION'}:return GROUPS[3]
    if role=='MIRROR' or name.startswith('mirror'):return GROUPS[1]
    if any(o==getattr(p,k,None) for k in ('face_scan_obj','cast_obj','cbct_obj')):return GROUPS[0]
    if role in {'MOLD','INSERT'} or o.get('mold_set') or o.name in {'Mold_Base','Mold_Wedge','Mold_Cap','Mold_BreakableShell'}:return GROUPS[7]
    if role in {'OCULAR','IRIS_BUTTON'} or o==p.ocular_obj:return GROUPS[5]
    if role=='SUBSTRUCTURE':return GROUPS[6]
    if role in {'SHELL','HOLLOW','FITTED'} or name.startswith('prototype'):return GROUPS[4]
    if any(o==getattr(p,k,None) for k in ('prosthesis_obj','source_prosthesis','sculpt_surface')):return GROUPS[3]
    # Older imported/cropped Sculpts may predate persistent role tags. Match
    # Blender's duplicate-name suffix, not arbitrary helpers containing "sculpt".
    if o.type=='MESH' and not role and (name=='sculpt' or
            (name.startswith('sculpt.') and name[7:].isdigit()) or o.get('sculpt_crop_complete')):
        return GROUPS[3]
    if role in {'SCAN','CAST','CBCT'} or 'facescan' in name:return GROUPS[0]
    return GROUPS[-1]


def organize_object(o,scene,*,preserve_visibility=False):
    if not hasattr(scene,'anaplast'):return
    previous=set(json.loads(o.get('anaplast_legacy_collections','[]')))
    previous.update(c.name for c in o.users_collection if c.name.startswith('Anaplast_'))
    if previous:o['anaplast_legacy_collections']=json.dumps(sorted(previous))
    cat=category(o,scene)
    if cat==GROUPS[3] and o.type=='MESH' and not o.get('anaplast_part'):
        o['anaplast_part']='SCULPT'
    if cat==GROUPS[7]:
        from . import mold_workflow
        col=mold_workflow.collection_for(scene,o.get('mold_set') or mold_workflow.ensure_active(scene))
    else:col=group(scene,cat)
    move(o,col)
    if is_construction(o,scene):
        o['anaplast_construction']=True
        if not preserve_visibility and o.name in scene.objects:
            set_hidden(o,scene,not scene.anaplast.show_construction_helpers);o.hide_render=True


def marker_visibility(scene):
    if not hasattr(scene,'anaplast'):return
    show=scene.anaplast.show_landmark_markers or scene.anaplast.landmarks_placing
    for o in scene.objects:
        if is_marker(o):
            set_hidden(o,scene,not (show or o.get('alignment_visible',False)))
            o.hide_render=True


def organize(scene,*,preserve_visibility=False):
    # Create groups only for actual objects. During ordinary creation/deletion,
    # keep visibility controlled by the active editing tool (e.g. wedge marks).
    for o in list(scene.objects):organize_object(o,scene,preserve_visibility=preserve_visibility)
    for col in list(scene.collection.children):
        if not col.get('anaplast_scene_group'):
            helpers=group(scene,GROUPS[-1])
            if col.name not in helpers.children:helpers.children.link(col)
            scene.collection.children.unlink(col)
    for col in list(scene.collection.children):
        if col.get('anaplast_scene_group') and not col.all_objects:
            # Only unlink our empty top-level folders; never delete user objects
            # or nested collections, which may be shared with another scene.
            scene.collection.children.unlink(col)
            if col.users==0:bpy.data.collections.remove(col)
    ordered=sorted(scene.collection.children,key=lambda c:c.get('anaplast_scene_group',c.name))
    for col in ordered:scene.collection.children.unlink(col)
    for col in ordered:scene.collection.children.link(col)
    if not preserve_visibility:marker_visibility(scene)
    _known[scene.as_pointer()]=object_ids(scene)


_known={}
_running=False


def object_ids(scene):
    return frozenset(o.as_pointer() for o in scene.objects)


def refresh_collections():
    global _running
    if _running:return None
    _running=True
    try:
        for scene in bpy.data.scenes:
            if hasattr(scene,'anaplast') and _known.get(scene.as_pointer())!=object_ids(scene):
                organize(scene,preserve_visibility=True)
    finally:_running=False
    return None


@bpy.app.handlers.persistent
def on_objects_changed(scene,*_):
    # Defer collection edits until the operator/depsgraph evaluation finishes.
    if _running or not hasattr(scene,'anaplast'):return
    if _known.get(scene.as_pointer())!=object_ids(scene) and not bpy.app.timers.is_registered(refresh_collections):
        bpy.app.timers.register(refresh_collections,first_interval=.3)


@bpy.app.handlers.persistent
def on_load(_):
    _known.clear()
    for scene in bpy.data.scenes:
        if hasattr(scene,'anaplast'):
            scene.anaplast.landmarks_placing=False
            p=scene.anaplast
            if not p.sculpt_surface:
                # Migrate the old category selector once; new selection remains fixed.
                from .sculpt_surface import target
                if scene==bpy.context.scene:p.sculpt_surface=target(bpy.context) or p.source_prosthesis or p.prosthesis_obj
            p.sculpt_target='OBJECT'
            w=scene.mold_workflow
            if not w.enabled:
                base=bpy.data.objects.get('Mold_Base');three=bool(base and base.get('three_piece'))
                w.base_body=w.wedge_body=w.cap_body=p.wedge_body if three else p.m_base_type
                w.base_wall=p.wedge_base_wall if three else p.m_shell_wall
                w.wedge_wall=p.wedge_wall;w.cap_wall=p.wedge_cap_wall if three else p.m_shell_wall
                w.cap_support='WEDGE' if three and bpy.data.objects.get('Mold_Wedge') else 'BASE';w.enabled=True
            from .mold_workflow import adopt
            adopt(scene)
            organize(scene)


def register():
    _known.clear()
    bpy.app.handlers.load_post.append(on_load)
    bpy.app.handlers.depsgraph_update_post.append(on_objects_changed)
    if not bpy.app.timers.is_registered(refresh_collections):bpy.app.timers.register(refresh_collections,first_interval=.3)
def unregister():
    if on_load in bpy.app.handlers.load_post:bpy.app.handlers.load_post.remove(on_load)
    if on_objects_changed in bpy.app.handlers.depsgraph_update_post:bpy.app.handlers.depsgraph_update_post.remove(on_objects_changed)
    if bpy.app.timers.is_registered(refresh_collections):bpy.app.timers.unregister(refresh_collections)
    _known.clear()
