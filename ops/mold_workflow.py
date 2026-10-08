"""Named mold assemblies and a transactional front end to the geometry builders."""
import json, uuid
import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty, PointerProperty
from ..utils import mesh as mu


BODY=[('FULL','Solid',''),('SHELL','Shell','')]
def collection_for(scene,uid):
    from .scene_helpers import group,GROUPS
    parent=group(scene,GROUPS[7])
    col=next((c for c in parent.children if c.get('mold_set')==uid),None)
    if col is None:
        col=bpy.data.collections.new(f'Mold {len(parent.children)+1:02d}');col['mold_set']=uid;parent.children.link(col)
    return col


def ensure_active(scene):
    w=scene.mold_workflow
    if not w.active_id:w.active_id=uuid.uuid4().hex
    return w.active_id


def owned(o):
    return not o.get('workflow_transaction') and (o.get('mold_set') or o.name.startswith(('Mold_','Auricular_','Common_Base','Wedge_','InsertMark_','InsertOriginal_')))


def capture(scene):
    p=scene.anaplast;data={}
    for prop in p.bl_rna.properties:
        key=prop.identifier
        if key.startswith(('m_','wedge_')) or key in ('key_clearance','mold_steps','mold_silicone','mold_silicone_density'):
            value=getattr(p,key)
            if prop.type=='POINTER':data[key]={'object':value.name if value else None}
            elif prop.type in {'BOOLEAN','INT','FLOAT','STRING','ENUM'}:data[key]=list(value) if getattr(prop,'is_array',False) else value
    return data


def apply_settings(scene,data):
    for key,value in data.items():
        if isinstance(value,dict):value=bpy.data.objects.get(value.get('object') or '')
        try:setattr(scene.anaplast,key,value)
        except (TypeError,AttributeError):pass


def adopt(scene):
    uid=ensure_active(scene)
    for obj in scene.objects:
        if owned(obj) and not obj.get('mold_set'):
            obj['mold_set']=uid;obj['mold_role']=obj.name
    return uid


def insert_settings(scene):
    data=[]
    for item in scene.mold_inserts.items:
        row={}
        for prop in item.bl_rna.properties:
            key=prop.identifier
            if key in {'rna_type','seat_piece'}:continue
            value=getattr(item,key)
            if prop.type=='POINTER':row[key]={'object':value.name if value else None}
            elif prop.type in {'STRING','BOOLEAN','ENUM','INT','FLOAT'}:row[key]=list(value) if getattr(prop,'is_array',False) else value
        data.append(row)
    return data


def restore_inserts(scene,data):
    scene.mold_inserts.items.clear();scene.mold_inserts.active=0
    for row in data:
        item=scene.mold_inserts.items.add()
        for key,value in row.items():
            if isinstance(value,dict):value=bpy.data.objects.get(value.get('object') or '')
            setattr(item,key,value)


def archive_active(scene):
    from . import explode,mold_volume
    explode.collapse(scene);uid=adopt(scene);col=collection_for(scene,uid)
    col['settings']=json.dumps(capture(scene));col['volume']=scene.get(mold_volume.KEY,'')
    col['inserts']=json.dumps(insert_settings(scene))
    col['workflow']=json.dumps({k:getattr(scene.mold_workflow,k) for k in ('base_body','wedge_body','cap_body','cap_support','base_wall','wedge_wall','cap_wall')})
    for obj in list(scene.objects):
        if obj.get('mold_set')==uid:
            obj['mold_role']=obj.name;obj['mold_was_hidden']=obj.hide_get();obj['mold_was_render_hidden']=obj.hide_render
            obj.name=col.name+' · '+obj.name;obj.hide_set(True);obj.hide_render=True
    col.hide_render=True


def activate(scene,uid):
    if scene.get('anaplast_insert_state'):raise RuntimeError('Restore the installed inserts before switching molds; their ocular recess is shared with this case')
    archive_active(scene);scene.mold_workflow.active_id=uid;col=collection_for(scene,uid);col.hide_render=False
    for obj in scene.objects:
        if obj.get('mold_set')==uid:
            obj.name=obj.get('mold_role',obj.name);obj.hide_set(obj.get('mold_was_hidden',False));obj.hide_render=obj.get('mold_was_render_hidden',False)
    apply_settings(scene,json.loads(col.get('settings','{}')))
    restore_inserts(scene,json.loads(col.get('inserts','[]')))
    for key,value in json.loads(col.get('workflow','{}')).items():setattr(scene.mold_workflow,key,value)
    from . import mold_volume
    if col.get('volume'):scene[mold_volume.KEY]=col['volume']
    else:mold_volume.clear(scene,'Build a matching cap to calculate material')


_menu=[]
_menu_strings={}
def mold_items(self,context):
    global _menu
    if not context:return []
    parent=next((c for c in context.scene.collection.children if c.get('anaplast_scene_group')=='08  Molds'),None)
    pairs=[(c['mold_set'],c.name) for c in parent.children if c.get('mold_set')] if parent else []
    # Blender retains enum string pointers; keep every returned tuple alive.
    _menu=[_menu_strings.setdefault(pair,(pair[0],pair[1],'')) for pair in pairs]
    return _menu


class WorkflowSettings(bpy.types.PropertyGroup):
    active_id:StringProperty()
    selection:EnumProperty(name='Saved mold',items=mold_items)
    enabled:BoolProperty(default=False)
    base_body:EnumProperty(name='Base body',items=BODY,default='FULL')
    wedge_body:EnumProperty(name='Wedge body',items=BODY,default='FULL')
    cap_body:EnumProperty(name='Cap body',items=BODY,default='FULL')
    cap_support:EnumProperty(name='Fits over',items=[('BASE','Base',''),('WEDGE','Base + Wedge','')])
    base_wall:FloatProperty(name='Base wall (mm)',default=4,min=.5)
    wedge_wall:FloatProperty(name='Wedge wall (mm)',default=4,min=.5)
    cap_wall:FloatProperty(name='Cap wall (mm)',default=4,min=.5)
    show_keys:BoolProperty(name='Registration keys',default=True)
    show_pry:BoolProperty(name='Pry points')
    show_flow:BoolProperty(name='Ring & feed channels')
    show_bolts:BoolProperty(name='Clamp bolts')
    report:StringProperty()


class Transaction:
    """Keep independent mesh backups until a complete build has succeeded."""
    def __init__(self,scene):
        from .scene_helpers import group,GROUPS
        from . import mold_volume
        self.scene=scene;self.uid=adopt(scene);self.settings=capture(scene);self.backups=[];self.inserts=insert_settings(scene)
        self.before={o.as_pointer() for o in scene.objects};self.volume=scene.get(mold_volume.KEY,'')
        self.workflow={p.identifier:getattr(scene.mold_workflow,p.identifier) for p in scene.mold_workflow.bl_rna.properties if p.type in {'STRING','BOOLEAN','ENUM','FLOAT'} and p.identifier not in {'rna_type','selection'}}
        try:
            for obj in list(scene.objects):
                if obj.get('mold_set')==self.uid and not obj.get('workflow_transaction'):
                    saved=mu.duplicate_object(obj,'Workflow_backup',group(scene,GROUPS[-1]))
                    self.backups.append((obj.name,saved,obj.hide_get(),obj.hide_render))
                    saved['workflow_transaction']=True;saved.hide_set(True);saved.hide_render=True
        except Exception:
            # Preparation has not changed the originals. Discard only this
            # attempt's copies if any later backup fails.
            for _,saved,_,_ in self.backups:mu.delete_object(saved)
            raise
    def finish(self,success):
        from . import mold_volume
        if not success:
            for obj in list(self.scene.objects):
                if not obj.get('workflow_transaction') and (obj.get('mold_set')==self.uid or (obj.as_pointer() not in self.before and owned(obj))):mu.delete_object(obj)
            for name,obj,hidden,render in self.backups:
                obj.name=name;obj.pop('workflow_transaction',None);obj.hide_set(hidden);obj.hide_render=render
            for key,value in self.workflow.items():setattr(self.scene.mold_workflow,key,value)
            apply_settings(self.scene,self.settings)
            restore_inserts(self.scene,self.inserts)
            if self.volume:self.scene[mold_volume.KEY]=self.volume
            else:mold_volume.clear(self.scene,'Build a matching cap to calculate material')
        else:
            for _,obj,_,_ in self.backups:mu.delete_object(obj)


def preserve(scene,names):
    held=[]
    for name in names:
        obj=bpy.data.objects.get(name)
        if obj:
            held.append((name,obj,obj.hide_get(),obj.hide_render));obj.name='Workflow_hold_'+uuid.uuid4().hex;obj['workflow_transaction']=True
    return held


def release(held):
    for name,obj,hidden,render in held:
        current=bpy.data.objects.get(name)
        if current and current!=obj:mu.delete_object(current)
        obj.name=name;obj.pop('workflow_transaction',None);obj.hide_set(hidden);obj.hide_render=render


def build_part(context,part,choice='REPLACE'):
    from . import explode,mold_auricular as aw,mold_volume,scene_helpers,mold_auricular_shell as shells
    s=context.scene;p=s.anaplast;w=s.mold_workflow
    if s.get('anaplast_insert_state'):raise RuntimeError('Restore the installed inserts before rebuilding mold parts')
    if context.object and context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    explode.collapse(s)
    if part!='BASE' and not bpy.data.objects.get('Mold_Base'):raise RuntimeError('Build the base first')
    if part=='CAP' and p.m_ocular_impression and w.cap_support=='WEDGE':raise RuntimeError('Ocular impressions currently require Cap over Base')
    if part in {'CAP','BREAK'}:
        recorded=json.loads(bpy.data.objects['Mold_Base'].get('workflow_base_settings','{}'))
        shared=('m_skin','m_land','m_land_transition','m_land_follow','m_fit_smoothing','m_skin_smoothing','m_key_type','m_key_count','m_key_diameter','m_key_flare','m_key_blend','m_wedge_inner','m_wedge_ring_gap','m_ring','m_spillways','m_spillway_width','m_pry','m_pry_width','m_pry_height','m_pry_depth','m_bolts','m_bolt_diameter','key_clearance')
        if any(key in recorded and recorded[key]!=getattr(p,key) for key in shared):raise RuntimeError('Mold features changed since the base was built. Replace the base first so the mating parts match')
    transaction=None;success=False;oldsettings=capture(s);held=[]
    s['anaplast_last_mold_attempt']=json.dumps({'piece':part,'choice':choice,'base_body':w.base_body,'wedge_body':w.wedge_body,'cap_body':w.cap_body,'base_wall_mm':w.base_wall,'wedge_wall_mm':w.wedge_wall,'cap_wall_mm':w.cap_wall,'status':'BUILDING'})
    try:
        transaction=Transaction(s)
        w.enabled=True
        if part=='BASE':
            if choice=='NEW' and bpy.data.objects.get('Mold_Base'):
                archive_active(s);w.active_id=uuid.uuid4().hex;collection_for(s,w.active_id)
                for prop in ('wedge_mark_obj','wedge_trace','wedge_suggestion'):setattr(p,prop,None)
            else:
                for obj in list(s.objects):
                    if obj.get('mold_set')==w.active_id and not obj.get('workflow_transaction'):mu.delete_object(obj)
                for prop in ('wedge_mark_obj','wedge_trace','wedge_suggestion'):setattr(p,prop,None)
            restore_inserts(s,[])
            p.m_base_type=w.base_body;p.m_shell_wall=w.base_wall;p.m_breakable_cap=False
            result=bpy.ops.anaplast.mold_two_part(which='BASE')
            if result!={'FINISHED'}:raise RuntimeError('Base construction failed; the previous mold was restored')
            base=bpy.data.objects['Mold_Base']
            base['workflow_base_settings']=json.dumps(oldsettings)
            seat=mu.duplicate_object(base,'Common_BaseSeat_Reference',mu.get_collection(s));seat['anaplast_construction']=True;seat.hide_set(True);seat.hide_render=True
        elif part=='WEDGE':
            # A cap fitted to the previous wedge is retained only for comparison.
            cap=bpy.data.objects.get('Mold_Cap')
            if cap:
                cap.name='Previous cap — rebuild required';cap['mold_stale']=True;cap.hide_set(True);cap.hide_render=True
            previous=bpy.data.objects.get('Mold_Wedge')
            if previous and choice=='KEEP':
                previous.name='Previous wedge';previous['mold_stale']=True;previous.hide_set(True);previous.hide_render=True
            info=aw.build(context);explode.collapse(s)
            base=bpy.data.objects['Mold_Base'];wedge=bpy.data.objects['Mold_Wedge'];prepared=bpy.data.objects['Auricular_WedgeCapBlank_Reference']
            sources=(base,wedge,prepared);shells.cache_solids(sources,mu.get_collection(s))
            if w.base_body=='SHELL' or w.wedge_body=='SHELL':
                results,_=shells.make_parts(sources,(w.base_wall if w.base_body=='SHELL' else 0,w.wedge_wall if w.wedge_body=='SHELL' else 0,0),p.m_voxel)
                for obj,new,mode in zip(sources[:2],results[:2],(w.base_body,w.wedge_body)):
                    obj.data=new.data.copy();obj['auricular_shell']=mode=='SHELL'
                for obj in results:mu.delete_object(obj)
            w.cap_support='WEDGE'
        elif part=='CAP' and w.cap_support=='WEDGE':
            if not bpy.data.objects.get('Mold_Wedge'):raise RuntimeError('Build a wedge or choose Cap over Base')
            base=bpy.data.objects['Mold_Base'];cached=bpy.data.objects.get('Auricular_SolidBase_Reference')
            if not base.get('three_piece') and cached:
                base.data=cached.data.copy();base['auricular_shell']=False;base['three_piece']=True
            aw.build_cap(context);explode.collapse(s)
            for name in ('Mold_Base','Mold_Wedge','Mold_Cap'):
                obj=bpy.data.objects[name];obj.pop('mold_stale',None);obj.hide_set(False)
        else:
            # Two-piece cap construction must not delete the existing wedge,
            # cap alternative, or the inputs used to rebuild a three-piece cap.
            names=[o.name for o in s.objects if o.name.startswith('Auricular_') or o.name=='Mold_Wedge']
            if part=='BREAK':names+=['Mold_Cap']
            held=preserve(s,names)
            # Use the cap builder's thin-shell path before full-block repair,
            # bores and pry cuts. Thinning the finished solid cap carries bolt
            # walls and rounded block fragments into the offset reference.
            p.m_base_type='SHELL' if part=='BREAK' else w.cap_body;p.m_shell_wall=w.cap_wall;p.m_breakable_cap=part=='BREAK'
            base=bpy.data.objects['Mold_Base'];base_data=base.data;was_three=bool(base.get('three_piece'))
            if was_three:
                seat=bpy.data.objects.get('Common_BaseSeat_Reference')
                if seat is None:raise RuntimeError('Rebuild the common base once before making a cap directly over it')
                base.data=seat.data.copy()
            try:
                result=bpy.ops.anaplast.mold_two_part(which='CAP')
                if result!={'FINISHED'}:raise RuntimeError('Cap construction failed; the previous pieces were restored')
                explode.collapse(s)
                cap=bpy.data.objects['Mold_Cap']
                if part=='BREAK':
                    if not cap.get('breakable_cap'):raise RuntimeError('The cap builder did not confirm a breakable shell; the previous pieces were restored')
                    previous=bpy.data.objects.get('Mold_BreakableShell')
                    if previous:mu.delete_object(previous)
                    cap.name='Mold_BreakableShell';cap['anaplast_part']='MOLD';cap.hide_set(True);cap.hide_render=True
                else:cap['cap_support']='BASE'
            finally:
                if was_three and part=='BREAK':base.data=base_data
            if was_three and part=='CAP':
                base['three_piece']=False
                for _,obj,_,_ in held:
                    if obj.get('mold_role')=='Mold_Wedge' or obj.get('three_piece'):obj['mold_stale']=True
            release(held);held=[]
            if part=='BREAK':
                if transaction.volume:s[mold_volume.KEY]=transaction.volume
                else:mold_volume.clear(s,'Build matching mold pieces to calculate material')
        explode.collapse(s);adopt(s);scene_helpers.organize(s)
        for obj in s.objects:
            if obj.get('mold_stale'):obj.hide_set(True);obj.hide_render=True
        w.report={'BASE':'Base ready. Add a wedge, inserts, or a cap.','WEDGE':'Wedge ready. Check coverage, then build its cap.','CAP':'Cap ready. Check mating surfaces.','BREAK':'Separate breakable shell created and hidden to avoid overlapping the cap. Use Show breakable shell to inspect it.'}[part]
        success=True
    except Exception as exc:
        # Keep the diagnosis with the saved case, independently of the session
        # log and of the transaction that restores the previous mold geometry.
        attempt=json.loads(s.get('anaplast_last_mold_attempt','{}'))
        attempt['error']=str(exc)
        attempt['error_type']=type(exc).__name__
        s['anaplast_last_mold_attempt']=json.dumps(attempt)
        raise
    finally:
        if held:release(held)
        attempt=json.loads(s.get('anaplast_last_mold_attempt','{}'))
        attempt['status']='FINISHED' if success else 'FAILED'
        attempt['steps']=json.loads(p.mold_steps or '[]')
        s['anaplast_last_mold_attempt']=json.dumps(attempt)
        # Temporary builder switches do not replace the user-facing choices.
        for key in ('m_base_type','m_shell_wall','m_breakable_cap'):setattr(p,key,oldsettings[key])
        if transaction is not None:transaction.finish(success)
        if not success:
            # Restoring old geometry must not erase the failed build's diagnosis.
            p.mold_steps=json.dumps(attempt['steps']);scene_helpers.organize(s)


class ANAPLAST_OT_mold_workflow(bpy.types.Operator):
    bl_idname='anaplast.mold_workflow';bl_label='Build mold piece';bl_options={'REGISTER','UNDO'}
    part:EnumProperty(items=[('BASE','Build base',''),('WEDGE','Build wedge',''),('CAP','Build cap',''),('BREAK','Build separate breakable shell',''),('SWITCH','Switch mold',''),('ORGANIZE','Organize scene',''),('SHOW_BREAK','Show breakable shell','')])
    choice:EnumProperty(name='Existing piece',items=[('REPLACE','Replace existing',''),('NEW','Start new mold',''),('KEEP','Keep old wedge hidden and build new','')],default='REPLACE')
    def invoke(self,context,event):
        existing=bpy.data.objects.get('Mold_Base' if self.part=='BASE' else 'Mold_Wedge')
        if self.part=='BASE' or (self.part=='WEDGE' and existing):return context.window_manager.invoke_props_dialog(self,width=430)
        return self.execute(context)
    def draw(self,context):
        self.layout.label(text='Build base: start a new mold or replace the active mold?' if self.part=='BASE' else 'A wedge already exists in this mold.')
        row=self.layout.row();row.prop_enum(self,'choice','REPLACE')
        row.prop_enum(self,'choice','NEW' if self.part=='BASE' else 'KEEP')
    def execute(self,context):
        try:
            if self.part=='SWITCH':activate(context.scene,context.scene.mold_workflow.selection)
            elif self.part=='ORGANIZE':
                from .scene_helpers import organize
                adopt(context.scene);organize(context.scene)
            elif self.part=='SHOW_BREAK':
                obj=bpy.data.objects.get('Mold_BreakableShell')
                if obj is None:raise RuntimeError('Build the separate shell first')
                show=obj.hide_get();obj.hide_set(not show);obj.hide_render=not show
                if show:
                    for name in ('Mold_Cap','Mold_Wedge'):
                        part=bpy.data.objects.get(name)
                        if part:part.hide_set(True)
            else:build_part(context,self.part,self.choice)
        except Exception as exc:
            import traceback
            from .report import rep
            traceback.print_exc();rep(self,{'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


def register():
    bpy.utils.register_class(WorkflowSettings);bpy.types.Scene.mold_workflow=PointerProperty(type=WorkflowSettings)
    bpy.utils.register_class(ANAPLAST_OT_mold_workflow)
def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_mold_workflow);del bpy.types.Scene.mold_workflow;bpy.utils.unregister_class(WorkflowSettings)
