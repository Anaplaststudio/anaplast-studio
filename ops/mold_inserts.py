"""Local, reversible mold cartridges: wax feeds, ocular holders and airway cores.

Build into temporary meshes, validate, then replace host mesh datablocks together.
Reference hosts are never edited. Airway cross sections are nested along the
opening axis, rather than merely angling the visible tip to look releasable.
"""
import json, math, uuid
import bpy, bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy.props import (BoolProperty, StringProperty, EnumProperty, FloatProperty,
                       IntProperty, FloatVectorProperty, PointerProperty, CollectionProperty)
from ..utils import mesh as mu
from . import mold_auricular as aw, mold_volume as amounts, explode

STATE='anaplast_insert_state'
COL='Anaplast_Inserts'
REF='Anaplast_Insert_References'
KINDS=[('WAX','Wax reservoir / injector',''),('OCULAR','Ocular pedestal',''),('AIRWAY','Nasal airway core',''),('HOLDER','Ocular space holder','Removable holder using an existing ocular pedestal'),('SEAT','Mounting seat','Socket and matching blank, ready for an attachment')]
# Preserve saved numeric values for Blank and Combined in older cases.
ADD_KINDS=KINDS[:3]+[('BLANK','Blank','Close the selected existing seat with its matching plug'),('COMBINED','Wax reservoir + ocular pedestal','Add both attachments linked to one mounting seat')]+[KINDS[3]]
HOLDER_FIELDS=('gaze_horizontal','gaze_vertical','gaze_clearance','wax_pillars','wax_pillar_width')
MARKS=[('BASE1','Patient / mounting point',''),('BASE2','Second patient opening',''),
       ('TIP1','First nostril',''),('TIP2','Second nostril',''),('EYE','Ocular key position','')]


def mesh_poll(self,obj): return obj.type=='MESH' and not obj.get('anaplast_construction')


_seat_enum_cache={}
def mounting_seat_choices(item,context):
    scene=item.id_data
    values=(('NONE','Choose seat','',0,0),)+tuple((i.uid,i.seat_name or i.name,'Mounting seat',0,1+(uuid.uuid5(uuid.NAMESPACE_OID,i.uid).int%2147483646)) for i in scene.mold_inserts.items if i.kind!='HOLDER' and i.seat_mode=='OWN')
    return _seat_enum_cache.setdefault(values,list(values))


def second_seat_choices(item,context):
    exclude=item.uid if hasattr(item,'uid') else raw_seat_link(item,'target_seat')
    values=tuple(v for v in mounting_seat_choices(item,context) if v[0]=='NONE' or v[0]!=exclude)
    return _seat_enum_cache.setdefault(values,list(values))


def raw_seat_link(item,key):
    value=item.get(key,0)
    return next((v[0] for v in mounting_seat_choices(item,None) if v[4]==value), 'NONE')


def update_target_seat(item,context):
    if raw_seat_link(item,'second_seat')==raw_seat_link(item,'target_seat'):item.second_seat='NONE'


def repair_seat_links(scene):
    p=scene.mold_inserts;repaired=[]
    for item in p.items:
        for key in ('bridge_seat','flow_seat'):
            value=raw_seat_link(item,key)
            if item.get(key,0) and (value=='NONE' or value==item.uid):
                setattr(item,key,'NONE');repaired.append(item.seat_name or item.name)
    for key in ('target_seat','second_seat'):
        if p.get(key,0) and raw_seat_link(p,key)=='NONE':setattr(p,key,'NONE')
    update_target_seat(p,None)
    if repaired:p.report='Corrected invalid seat connection: '+', '.join(repaired)+'. Marks and geometry retained. Build / update all inserts to continue.'
    return repaired


def validate_seat_links(scene,seat_only_uid=None):
    repair_seat_links(scene)
    roots={i.uid:i for i in scene.mold_inserts.items if i.kind!='HOLDER' and i.seat_mode=='OWN'}
    used=set()
    for item in roots.values():
        other=raw_seat_link(item,'bridge_seat')
        if other=='NONE' or item.mode!='INSERT' or item.uid==seat_only_uid:continue
        if other not in roots:raise RuntimeError(f'{item.name}: choose a valid second mounting seat')
        if item.uid in used or other in used or raw_seat_link(roots[other],'bridge_seat')!='NONE':
            raise RuntimeError(f'{item.seat_name or item.name}: already part of another two-seat insert. Use Disconnect second seat first.')
        used.update((item.uid,other))


def seat_connection(scene,seat):
    if not seat:return None
    return next((i for i in scene.mold_inserts.items if raw_seat_link(i,'bridge_seat')!='NONE'
                 and (i.uid==seat.uid or raw_seat_link(i,'bridge_seat')==seat.uid)),None)


def get_seat_count(item):return int(item.use_two_seats)
def set_seat_count(item,value):item.use_two_seats=bool(value)

def update_seat_mode(item,context):
    if item.seat_mode=='OWN':item.seat_owner='NONE'


def seat_choices(item,context):
    entries=context.scene.mold_inserts.items if context and hasattr(context.scene,'mold_inserts') else []
    values=(('NONE','Choose existing seat','',0,0),)+tuple((i.uid,i.name,'Share this attachment\'s mounting seat',0,1+(uuid.uuid5(uuid.NAMESPACE_OID,i.uid).int%2147483646)) for i in entries if i.uid!=item.uid and i.kind in {'WAX','OCULAR'} and i.kind!=item.kind and i.seat_mode=='OWN')
    # Blender retains references to enum strings; keep them alive between draws.
    return _seat_enum_cache.setdefault(values,list(values))


def pedestal_choices(item,context):
    scene=item.id_data
    values=(('NONE','Choose ocular pedestal','',0,0),)+tuple((i.uid,i.name,'Use this pedestal\'s ocular, key and mounting position',0,1+(uuid.uuid5(uuid.NAMESPACE_OID,i.uid).int%2147483646)) for i in scene.mold_inserts.items if i.kind=='OCULAR')
    return _seat_enum_cache.setdefault(values,list(values))


def update_mark_sizes(item,context):
    for mark,_,_ in MARKS:
        obj=bpy.data.objects.get(f'InsertMark_{item.uid[:6]}_{mark}')
        if obj:
            obj.empty_display_size=(item.nostril_diameter*.5 if mark.startswith('TIP') else item.patient_diameter*.5 if item.kind=='AIRWAY' else item.key_size*.5 if mark=='EYE' else item.dock_diameter*.5)


def seat_members(item):
    owner=item.uid if item.seat_mode=='OWN' else item.seat_owner
    return [i for i in item.id_data.mold_inserts.items if i.uid==owner or (i.seat_mode=='SHARED' and i.seat_owner==owner)]


def get_seat_piece(item):
    enabled={i.kind for i in seat_members(item) if i.mode=='INSERT'}
    return (1 if 'WAX' in enabled else 0)+(2 if 'OCULAR' in enabled else 0)


def set_seat_piece(item,value):
    kinds={1:{'WAX'},2:{'OCULAR'},3:{'WAX','OCULAR'}}.get(value,set())
    for member in seat_members(item):member.mode='INSERT' if member.kind in kinds else 'BLANK'


class InsertItem(bpy.types.PropertyGroup):
    uid: StringProperty()
    name: StringProperty(default='Insert')
    kind: EnumProperty(items=KINDS)
    mode: EnumProperty(name='Installed piece',items=[('INSERT','Use insert','Enable this attachment; two enabled attachments on a shared seat use the combined insert'),('BLANK','Blank','Use the contour-restoring blank instead of this attachment')])
    seat_piece: EnumProperty(name='Installed piece',items=[('BLANK','Blank','Close this shared seat with its contour-restoring plug',0),('WAX','Wax reservoir','Use only the reservoir on this seat',1),('OCULAR','Ocular pedestal','Use only the pedestal on this seat',2),('COMBINED','Wax reservoir + ocular pedestal','Use both attachments together',3)],get=get_seat_piece,set=set_seat_piece,options={'SKIP_SAVE'})
    seat_mode: EnumProperty(name='Mounting seat',items=[('OWN','Separate seat','Create a seat with its own base mark'),('SHARED','Use existing seat','Share a wax / ocular mounting seat')],default='OWN',update=update_seat_mode)
    seat_owner: EnumProperty(name='Existing seat',items=seat_choices)
    base1: FloatVectorProperty(size=3); base2: FloatVectorProperty(size=3)
    tip1: FloatVectorProperty(size=3); tip2: FloatVectorProperty(size=3); eye: FloatVectorProperty(size=3)
    marks: StringProperty(default='[]')
    mark_sources: StringProperty(default='{}')
    ocular: PointerProperty(name='Ocular',type=bpy.types.Object,poll=mesh_poll)
    ocular_use: EnumProperty(name='During wax casting',items=[('REAL','Hold real ocular','Key the pedestal into the real ocular'),('SUBSTITUTE','Removable try-in substitute','Keep the real ocular out of the wax; create rear space for gaze adjustment')],default='REAL')
    pedestal_uid: EnumProperty(name='Ocular pedestal',items=pedestal_choices)
    holder_enabled: BoolProperty(name='Include space holder',default=True)
    gaze_horizontal: FloatProperty(name='Left / right range (± degrees)',default=15,min=0,max=45)
    gaze_vertical: FloatProperty(name='Up / down range (± degrees)',default=15,min=0,max=45)
    gaze_clearance: FloatProperty(name='Rear clearance (mm)',default=.1,min=0,soft_max=.5)
    wax_pillars: BoolProperty(name='Form four corner wax pillars',default=True)
    wax_pillar_width: FloatProperty(name='Wax pillar width (mm)',default=3,min=1,soft_max=6)
    key: EnumProperty(name='Ocular key',items=[('PAW','Dog-paw (David style)','Large pad and four smaller pads, based on the supplied reference'),('D','Tapered D key','Flat side prevents rotation')])
    key_size: FloatProperty(name='Key width (mm)',default=6,min=2,soft_max=12,update=update_mark_sizes)
    key_depth: FloatProperty(name='Key depth (mm)',default=1.5,min=.3,soft_max=3)
    key_clearance: FloatProperty(name='Key clearance (mm)',default=.1,min=0,soft_max=.4)
    stem: FloatProperty(name='Pedestal width (mm)',default=8,min=3,soft_max=20)
    chambers: EnumProperty(name='Airway',items=[('SEPARATE','Two separate chambers','Two passages on one removable cartridge'),('SHARED','Single shared chamber','Two nostrils joining one patient-side opening')])
    airway_shape: EnumProperty(name='Openings',items=[('CIRCULAR','Circular marks','Compatibility with previously marked round openings'),('OUTLINE','Marked shapes','Smooth drawn or painted outlines define each opening')],default='CIRCULAR')
    airway_path: EnumProperty(name='Passage',items=[('DRAIN','Sloped connection (legacy)','Direct connection between the marked openings; undercuts allowed'),('STRAIGHT','Straight withdrawal','Nested sections for removal along the mold axis'),('CURVED','Curved nasal cavity — test','Build the cavity inside the nasal Sculpt and blend curved nostril passages into it; inspect thickness and removal')],default='CURVED')
    airway_wall: FloatProperty(name='Silicone wall target (mm)',default=4.,min=1.,soft_max=8.,description='Target for the main cavity; the sampled report can show thinner nostril transitions')
    airway_reservoir: BoolProperty(name='Include wax reservoir + air return',default=False,description='Separate feed and overflow cups on the same compact airway mounting seat')
    airway_outlines: StringProperty(default='{}')
    airway_smoothing: FloatProperty(name='Outline smoothing (mm)',default=.35,min=.1,soft_max=.8,description='Smooth paint and lasso irregularities while retaining the general opening shape')
    airway_brush: FloatProperty(name='Paint radius (mm)',default=1,min=.2,max=10)
    patient_diameter: FloatProperty(name='Patient opening diameter (mm)',default=10,min=2,soft_max=30,update=update_mark_sizes,description='Design dimension; must be checked for this case')
    nostril_diameter: FloatProperty(name='Nostril diameter (mm)',default=4,min=1,soft_max=15,update=update_mark_sizes,description='Design dimension at each marked nostril')
    draft: FloatProperty(name='Withdrawal draft (degrees)',default=2,min=.5,soft_max=8)
    wax_mode: EnumProperty(name='Wax feed',items=[('GRAVITY','Reservoir + funnel',''),('INJECTION','Injection spout','Smooth tapered nozzle seat; not a pressure-rated coupling')])
    reservoir_layout: EnumProperty(name='Reservoir layout',items=[('STRAIGHT','Original straight feed','Keep the separate straight wax bore'),('SIDE_CUPS','Compact feed + overflow','Two cups above one narrow seat; independent side wax inlet and local air return, with cap on table and base above')],default='STRAIGHT')
    vent_diameter: FloatProperty(name='Air return bore (mm)',default=2.5,min=1,soft_max=6)
    overflow_ml: FloatProperty(name='Overflow capacity (mL)',default=1,min=.2,soft_max=10)
    allowance: FloatProperty(name='Extra wax (%)',default=10,min=1,soft_max=30,description='Working reserve, not a measured material shrinkage coefficient')
    cavity_ml: FloatProperty(name='Fill estimate (mL)',default=0,min=0,soft_max=100,description='0 estimates from the Sculpt with 20% extra; enter a value to override. No mold-volume measurement is needed')
    feed_diameter: FloatProperty(name='Feed bore (mm)',default=4,min=1,soft_max=12)
    cup_height: FloatProperty(name='Reservoir height (mm)',default=15,min=5,soft_max=40)
    nozzle_diameter: FloatProperty(name='Nozzle seat diameter (mm)',default=8,min=2,soft_max=20)
    dock_diameter: FloatProperty(name='Minimum seat width (mm)',default=12,min=6,soft_max=40,update=update_mark_sizes)
    clearance: FloatProperty(name='Seat clearance (mm)',default=.1,min=0,soft_max=.3)
    screw: FloatProperty(name='Retaining screw bore (mm)',default=2,min=1,soft_max=4,description='Pilot bore; select hardware and test its fit')
    pilot_depth: FloatProperty(name='Pilot depth (mm)',default=2.5,min=1,soft_max=6)
    report: StringProperty()
    seat_spec: StringProperty(description='Built mounting seat geometry; retained when an attachment is chosen')
    seat_name: StringProperty()
    bridge_seat: EnumProperty(name='Second seat',items=second_seat_choices)
    flow_seat: EnumProperty(name='Air vent seat',items=second_seat_choices)
    seat_auto_orient: BoolProperty(name='Auto orient new seat',default=True,description='Turn a new seat to clear nearby mounting parts; built seats retain their angle')
    seat_angle: FloatProperty(name='Seat angle (degrees)',default=0,min=-180,max=180,description='Rotate this mounting seat and its screw tabs in the mold plane; rebuilding an existing seat requires confirmation')


class InsertSettings(bpy.types.PropertyGroup):
    items: CollectionProperty(type=InsertItem)
    active: IntProperty(default=0,min=0)
    new_kind: EnumProperty(items=ADD_KINDS)
    report: StringProperty()
    target_seat: EnumProperty(name='Seat',items=mounting_seat_choices,update=update_target_seat)
    second_seat: EnumProperty(name='Second seat',items=second_seat_choices)
    use_two_seats: BoolProperty(name='Use two seats',default=False)
    seat_count: EnumProperty(name='Mounting',items=[('ONE','One seat','One mounting position',0),('TWO','Two seats','Two different mounting positions',1)],get=get_seat_count,set=set_seat_count,options={'SKIP_SAVE'})
    two_seat_layout: EnumProperty(name='Two-seat arrangement',items=[('CONNECTED','One connected insert','One removable piece engaging both seats'),('SEPARATE','Separate inserts','Build one independent attachment per seat')],default='CONNECTED')


def active(scene):
    p=scene.mold_inserts
    return p.items[p.active] if p.active<len(p.items) else None


def signature(obj):
    return dict(amounts.fingerprint(obj),matrix=[list(row) for row in obj.matrix_world])


def mark_matches(item,mark,obj,source,cache=None):
    """Retain marks after remote edits only if they still lie on the current surface."""
    if obj is None:return False
    if cache is None:cache={}
    key=obj.as_pointer()
    current=cache.get(('signature',key))
    if current is None:current=cache[('signature',key)]=signature(obj)
    if current==source['signature']:return True
    if item.kind not in {'AIRWAY','WAX'} or mark not in {'BASE1','BASE2','TIP1','TIP2'}:return False
    matrix=source['signature'].get('matrix')
    if matrix is None or not np.allclose(np.asarray(obj.matrix_world),matrix,atol=1e-6,rtol=0):return False
    raw=json.loads(item.airway_outlines).get(mark) if item.kind=='AIRWAY' else None
    points=raw if raw else [getattr(item,mark.lower())]
    tree=cache.get(('tree',key))
    if tree is None:tree=cache[('tree',key)]=aw.world_tree(obj)
    for point in points:
        hit=tree.find_nearest(Vector(point))
        if hit[0] is None or hit[3]>.02:return False
    return True


def closed(obj,connected=False):
    stats=mu.mesh_stats(obj)
    if not stats[0] or any(stats[1:]): raise RuntimeError(f'{obj.name}: not a closed printable part ({stats})')
    if amounts.volume(obj)<1e-5: raise RuntimeError(f'{obj.name}: empty part')
    if connected:
        bm=bmesh.new();bm.from_mesh(obj.data)
        try:
            remaining=set(bm.verts);stack=[remaining.pop()]
            while stack:
                v=stack.pop()
                for e in v.link_edges:
                    neighbor=e.other_vert(v)
                    if neighbor in remaining:remaining.remove(neighbor);stack.append(neighbor)
            if remaining:raise RuntimeError(f'{obj.name}: disconnected material; move the marks or increase support size')
        finally:bm.free()


def discard_boolean_dust(obj):
    """Remove isolated numerical specks and doubled zero-volume sheets.

    Never weld or smooth the usable surface or edit an original reference.
    Real disconnected parts remain for the connected-solid check to reject.
    """
    bm=bmesh.new();bm.from_mesh(obj.data);remaining=set(bm.verts);groups=[]
    try:
        while remaining:
            stack=[remaining.pop()];group=[]
            while stack:
                v=stack.pop();group.append(v)
                for edge in v.link_edges:
                    other=edge.other_vert(v)
                    if other in remaining:remaining.remove(other);stack.append(other)
            groups.append(group)
        removed=[]
        if len(groups)>1:
            largest=max(groups,key=lambda g:(len(g),float(np.linalg.norm(np.ptp(np.array([v.co[:] for v in g]),axis=0)))))
            for group in groups:
                if group is largest:continue
                co=np.array([(obj.matrix_world@v.co)[:] for v in group])
                span=np.linalg.norm(np.ptp(co,axis=0))
                # Boolean specks can contain more than eight vertices. Bound
                # their physical size first, independently of triangulation.
                if span<=.0001:
                    removed.extend(group);continue
                # Thin Boolean sheets may have more than eight vertices.
                # Extend cleanup only to sub-0.3 mm isolated fragments; the
                # volume and planarity tests below still have to pass.
                if len(group)>64 or (len(group)>8 and span>.3):continue
                faces={face for vertex in group for face in vertex.link_faces}
                centered=co-co.mean(0);thin=np.linalg.svd(centered,compute_uv=False)[-1]<=.00001
                volume=0.
                for face in faces:
                    points=[np.array((obj.matrix_world@v.co)[:])-co.mean(0) for v in face.verts]
                    for j in range(1,len(points)-1):volume+=np.dot(points[0],np.cross(points[j],points[j+1]))/6.
                if abs(volume)<.000001 and (thin or span<.1):removed.extend(group)
        if removed:
            bmesh.ops.delete(bm,geom=removed,context='VERTS');bm.to_mesh(obj.data);obj.data.update()
            obj['boolean_dust_vertices_removed']=len(removed)
    finally:bm.free()


class Work:
    def __init__(self,scene): self.scene=scene;self.objects=[];self.col=mu.get_collection(scene,COL)
    def add(self,obj): self.objects.append(obj);return obj
    def copy(self,obj,name): return self.add(mu.duplicate_object(obj,name,self.col))
    def delete(self,obj):
        if obj in self.objects:self.objects.remove(obj)
        mu.delete_object(obj)
    def finish(self,keep=()):
        for obj in list(self.objects):
            if obj not in keep:self.delete(obj)


def loft(work,name,rings):
    """Closed rings with matching vertex counts; all coordinates are world mm."""
    n=len(rings[0]);verts=[tuple(p) for ring in rings for p in ring]
    faces=[tuple(reversed(range(n))),tuple((len(rings)-1)*n+i for i in range(n))]
    for j in range(len(rings)-1):
        for i in range(n):faces.append((j*n+i,j*n+(i+1)%n,(j+1)*n+(i+1)%n,(j+1)*n+i))
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(verts,[],faces);mesh.update()
    obj=bpy.data.objects.new(name,mesh);work.col.objects.link(obj);work.add(obj)
    bm=bmesh.new();bm.from_mesh(mesh);bmesh.ops.recalc_face_normals(bm,faces=bm.faces)
    bmesh.ops.triangulate(bm,faces=list(bm.faces));bm.to_mesh(mesh);bm.free();return obj


def round_rings(centers,radii,frame,segments=64):
    _,r,f=frame
    return [[c+(r*math.cos(a)+f*math.sin(a))*radius for a in np.linspace(0,2*math.pi,segments,endpoint=False)] for c,radius in zip(centers,radii)]


def cylinder(work,name,a,b,radius,frame): return loft(work,name,round_rings([a,b],[radius,radius],frame))


def d_profile(radius,flat=.65):
    # Circle truncated on one side; deterministic indexing at all depths.
    return [(max(-radius*flat,radius*math.cos(a)),radius*math.sin(a)) for a in np.linspace(0,2*math.pi,64,endpoint=False)]


def compact_seat(center,frame,footprints,minimum_width,slim=False):
    """Fit the actual D polygon around the attachment roots, without moving them.

    Footprints are (world center, radius including support wall). Optimize the
    seat's offset along right and its radius, instead of applying the D flat's
    .65 factor to the entire shared span. Compact two-passage roots already
    include 1.5 mm walls, so use a shallower key flat and a 0.5 mm seat allowance.
    Other attachments keep their existing one-mm allowance and D profile.
    """
    _,r,f=frame;profile=d_profile(1.,.85 if slim else .65);constraints=[]
    for a,b in zip(profile,profile[1:]+profile[:1]):
        dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy)
        if length<1e-10:continue
        nx,ny=dy/length,-dx/length;h=nx*a[0]+ny*a[1]
        for point,radius in footprints:
            delta=point-center
            constraints.append((nx,h,nx*delta.dot(r)+ny*delta.dot(f)+radius+(.5 if slim else 1.)))
    def interval(radius):
        low,high=-math.inf,math.inf
        for nx,h,value in constraints:
            required=value-radius*h
            if abs(nx)<1e-10:
                if required>0:return None
            elif nx>0:low=max(low,required/nx)
            else:high=min(high,required/nx)
        return (low,high) if low<=high else None
    low=minimum_width*.5;high=low;fits_minimum=interval(low) is not None
    while interval(high) is None:high*=2
    for _ in range(48):
        mid=(low+high)*.5
        if interval(mid) is None:low=mid
        else:high=mid
    radius=high if fits_minimum else high+.002
    a,b=interval(radius);shift=max(a,min(b,0.))
    return center+r*shift,2*radius


def profile_solid(work,name,profile,center,z0,z1,frame,top_scale=1.):
    u,r,f=frame;xy=center-u*center.dot(u)
    return loft(work,name,[[xy+u*z+r*x*scale+f*y*scale for x,y in profile] for z,scale in ((z0,1),(z1,top_scale))])


def merge(work,a,b):
    original=a.data.copy()
    try:
        aw.cut(a,b,'UNION')
        try:closed(a,connected=True)
        except RuntimeError:
            first=a.data;a.data=original.copy()
            try:
                mu.apply_boolean(a,b,'UNION',solver='EXACT');closed(a,connected=True)
            except RuntimeError:
                failed=a.data;a.data=first
                if failed.users==0:bpy.data.meshes.remove(failed)
            else:
                if first.users==0:bpy.data.meshes.remove(first)
    finally:
        if original.users==0:bpy.data.meshes.remove(original)
    work.delete(b);closed(a);return a



def intersection_volume(work,a,b,separate_touching=False):
    tmp=work.copy(a,'Insert_overlap_work')
    if separate_touching:
        # The same ten-nanometre separation used by the fallback below,
        # applied first for complementary nasal seats to avoid a stalled
        # coplanar intersection. Only the disposable check copy is altered.
        matrix=tmp.matrix_world;inverse=matrix.inverted()
        normal_matrix=matrix.to_3x3().inverted().transposed()
        normals=[(normal_matrix@v.normal).normalized() for v in tmp.data.vertices]
        for vertex,normal in zip(tmp.data.vertices,normals):
            vertex.co=inverse@(matrix@vertex.co-normal*.00001)
        tmp.data.update()
    try:
        try:aw.cut(tmp,b,'INTERSECT')
        except RuntimeError:
            # Complementary fitting surfaces can make one operand order
            # numerically degenerate. Retry the same intersection reversed;
            # retain the closed-surface and volume checks on the result.
            work.delete(tmp);tmp=work.copy(b,'Insert_overlap_reversed_work')
            try:aw.cut(tmp,a,'INTERSECT')
            except RuntimeError:
                if any(any(mu.mesh_stats(obj)[1:]) for obj in (a,b)):raise
                # Resolve exactly touching complementary triangles on a check
                # copy only: ten nanometres in our millimetre working space.
                # The result must still pass the existing volume check.
                work.delete(tmp);tmp=work.copy(a,'Insert_overlap_contact_work')
                matrix=tmp.matrix_world;inverse=matrix.inverted()
                normal_matrix=matrix.to_3x3().inverted().transposed()
                normals=[(normal_matrix@v.normal).normalized() for v in tmp.data.vertices]
                for vertex,normal in zip(tmp.data.vertices,normals):
                    vertex.co=inverse@(matrix@vertex.co-normal*.00001)
                tmp.data.update();aw.cut(tmp,b,'INTERSECT')
        return amounts.volume(tmp) if len(tmp.data.polygons) else 0.
    finally:work.delete(tmp)


def required_marks(item):
    if item.kind in {'WAX','OCULAR'} and item.seat_mode=='SHARED':return set()
    if item.kind in {'WAX','SEAT'}:return {'BASE1'}
    if item.kind=='OCULAR':return {'BASE1'}
    return {'BASE1','TIP1','TIP2'}|({'BASE2'} if item.chambers=='SEPARATE' else set())


def mark_label(item,mark):
    return {'BASE1':'patient opening on Mold Base' if item.chambers=='SHARED' else 'first patient opening on Mold Base',
            'BASE2':'second patient opening on Mold Base (Two separate chambers)',
            'TIP1':'first nostril on Sculpt','TIP2':'second nostril on Sculpt'}.get(mark,mark)


def missing_marks(item):
    if item.kind=='AIRWAY' and item.airway_shape=='OUTLINE':
        outlines=json.loads(item.airway_outlines)
        present={key for key,points in outlines.items() if len(points)>=3}
    else:present=set(json.loads(item.marks))
    return required_marks(item)-present


def validate_marks(item):
    missing=missing_marks(item)
    if not missing:return
    if item.kind=='AIRWAY':
        action='draw or paint ' if item.airway_shape=='OUTLINE' else 'mark '
        labels=', '.join(mark_label(item,mark) for mark in sorted(missing))
        raise RuntimeError(f'{item.name}: {action}{labels}. The mounting seat does not define the patient airway opening.')
    raise RuntimeError(f'{item.name}: mark '+', '.join(sorted(missing)))


def seat_groups(entries):
    roots={i.uid:i for i in entries if i.seat_mode=='OWN'}
    groups={uid:[item] for uid,item in roots.items()}
    for item in entries:
        if item.seat_mode=='OWN':continue
        owner=roots.get(item.seat_owner)
        if item.kind not in {'WAX','OCULAR'} or owner is None or owner.kind not in {'WAX','OCULAR'}:
            raise RuntimeError(f'{item.name}: choose a separate wax or ocular seat to share')
        groups[owner.uid].append(item)
    for group in groups.values():
        if len(group)>1 and (len(group)!=2 or {i.kind for i in group}!={'WAX','OCULAR'}):
            raise RuntimeError('A shared seat accepts one wax reservoir and one ocular pedestal')
    return list(groups.values())


def release_sections(start,end,root_radius,tip_radius,draft,frame):
    """Nested disks guarantee axial withdrawal toward the base (-up).

    The final 2 mm are coaxial at the nostril, not a slanted cut tip.
    Reject an undersized patient opening rather than silently enlarging anatomy.
    """
    u,_,_=frame;height=(end-start).dot(u)
    if height<3:raise RuntimeError('Nostril marks must be at least 3 mm above the patient opening along the mold opening direction')
    offset=(end-start)-u*height;lead=min(2.,height*.25);slope=math.tan(math.radians(draft))
    needed=tip_radius+offset.length+height*slope
    if root_radius+1e-6<needed:
        raise RuntimeError(f'This path would undercut: patient opening needs at least {2*needed:.2f} mm diameter for the marked offset, or move the marks into alignment')
    bend=end-u*lead
    return [start-u*.4,start,bend,end,end+u*.6], [root_radius+.4*slope,root_radius,tip_radius+lead*slope,tip_radius,max(.2,tip_radius-.6*slope)]


def airway(work,item,frame):
    if item.airway_shape=='OUTLINE':
        if item.airway_path=='CURVED':
            from .airway_cavity import cores
            return cores(work,item,frame)
        from .airway_marking import cores
        return cores(work,item,frame)
    starts=[Vector(item.base1),Vector(item.base2 if item.chambers=='SEPARATE' else item.base1)]
    ends=[Vector(item.tip1),Vector(item.tip2)];rad=item.patient_diameter*.5;tip=item.nostril_diameter*.5
    if item.chambers=='SEPARATE':
        u=frame[0];delta=starts[1]-starts[0];delta-=u*delta.dot(u)
        if delta.length<2*rad+.5:raise RuntimeError('Separate patient openings overlap; reduce their diameter or move the marks apart')
    cores=[]
    for start,end in zip(starts,ends):
        centers,radii=release_sections(start,end,rad,tip,item.draft,frame)
        cores.append(loft(work,'Airway_core_work',round_rings(centers,radii,frame)))
    if item.chambers=='SEPARATE' and intersection_volume(work,*cores)>.001:
        raise RuntimeError('The two airway chambers intersect; reposition marks or use the shared chamber option')
    # The docking cartridge connects separate cores below the fitting surface.
    center=sum(starts,Vector())*.5;u=frame[0]
    width=max(item.dock_diameter,2*(max(((q-center)-u*(q-center).dot(u)).length for q in starts)+rad+1.)/.65)
    return center,width,cores,ends


def bounds(obj,u):
    levels=[(obj.matrix_world@v.co).dot(u) for v in obj.data.vertices]
    return min(levels),max(levels)


def column(tree,point,frame,lo,hi):
    u=frame[0];xy=point-u*point.dot(u)
    front=tree.ray_cast(xy+u*(hi+5),-u,hi-lo+10)[0]
    back=tree.ray_cast(xy+u*(lo-5),u,hi-lo+10)[0]
    if front is None or back is None:raise RuntimeError('Mount extends beyond the base or crosses an existing opening; move its mark inward')
    return back.dot(u),front.dot(u)


def support_backing_islands(work,host,stem,base,center,profile,rear,frame):
    """Support detached rear slivers inside the existing seat footprint.

    A Boolean on a folded shell can leave a real thin island. Keep its surface
    and add a web from the planar backing, strictly behind the fitting face.
    Do not delete real islands or bridge outside the mounting footprint.
    """
    discard_boolean_dust(host)
    bm=bmesh.new();bm.from_mesh(host.data);remaining=set(bm.verts);groups=[]
    try:
        while remaining:
            stack=[remaining.pop()];group=[]
            while stack:
                v=stack.pop();group.append(v)
                for e in v.link_edges:
                    w=e.other_vert(v)
                    if w in remaining:remaining.remove(w);stack.append(w)
            groups.append(group)
        if len(groups)<2:return stem,0
        main=max(groups,key=len)
        islands=[[host.matrix_world@v.co for v in g] for g in groups if g is not main]
    finally:bm.free()
    u,r,f=frame;tree=aw.world_tree(base);lo,hi=bounds(base,u);count=0
    for points in islands:
        if len(points)>2000:raise RuntimeError('A large base section separates at this seat; choose another mounting point')
        xy=np.array([[(p-center).dot(r),(p-center).dot(f)] for p in points])
        if np.linalg.norm(np.ptp(xy,axis=0))>5:
            raise RuntimeError('A large base section separates at this seat; choose another mounting point')
        hull=aw.convex_hull_2d(xy)
        if len(hull)<3:raise RuntimeError('A disconnected base edge cannot be supported at this seat')
        hull=aw.offset_polygon(hull,.15)
        # The complete support stays inside the existing flange, not on an
        # arbitrary detached region elsewhere in the case.
        for x,y in hull:
            for a,b in zip(profile,profile[1:]+profile[:1]):
                if (b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0])< -1e-6:
                    raise RuntimeError('A disconnected base section extends outside the seat; choose another mounting point')
        roof=max(p.dot(u) for p in points)+.05
        samples=list(hull)+[tuple(p) for p in xy]
        minxy=np.min(hull,axis=0);maxxy=np.max(hull,axis=0)
        samples.extend((x,y) for x in np.arange(minxy[0],maxxy[0]+.05,.05) for y in np.arange(minxy[1],maxxy[1]+.05,.05))
        for x,y in samples:
            front=column(tree,center+r*float(x)+f*float(y),frame,lo,hi)[1]
            if roof>front-.02:raise RuntimeError('A detached fitting region cannot be supported without changing its surface; choose another mounting point')
        web=profile_solid(work,'Seat_support_web',hull,center,rear+.01,roof,frame)
        stem_web=work.copy(web,'Seat_stem_support_web')
        host=merge(work,host,web);stem=merge(work,stem,stem_web);count+=1
    discard_boolean_dust(host);closed(host,connected=True)
    return stem,count


def projected_min_height(triangles,profile):
    """Lowest triangle height inside a convex CCW footprint, including crossings.

    Clipping matters: vertices outside the seat (or screw) must not push the
    entire flat contact plane away from the mold.
    """
    profile=np.asarray(profile,dtype=float);triangles=np.asarray(triangles,dtype=float)
    low=profile.min(0);high=profile.max(0)
    take=np.all(triangles[:,:,:2].min(1)<=high,axis=1)&np.all(triangles[:,:,:2].max(1)>=low,axis=1)
    candidates=triangles[take];result=math.inf
    if not len(candidates):return result
    edges=np.roll(profile,-1,axis=0)-profile
    # Fully contained triangles need no clipping.
    inside=np.ones((len(candidates),3),dtype=bool);outside=np.zeros(len(candidates),dtype=bool)
    for a,e in zip(profile,edges):
        rel=candidates[:,:,:2]-a
        positive=e[0]*rel[:,:,1]-e[1]*rel[:,:,0]>=-1e-10
        inside&=positive;outside|=~positive.any(1)
    whole=inside.all(1)
    if np.any(whole):result=float(candidates[whole,:,2].min())
    for tri in candidates[~whole&~outside]:
        poly=list(tri)
        for a,e in zip(profile,edges):
            if not poly:break
            clipped=[];previous=poly[-1]
            d0=e[0]*(previous[1]-a[1])-e[1]*(previous[0]-a[0])
            for point in poly:
                d1=e[0]*(point[1]-a[1])-e[1]*(point[0]-a[0])
                if (d0>=0)!=(d1>=0):clipped.append(previous+(point-previous)*(d0/(d0-d1)))
                if d1>=0:clipped.append(point)
                previous=point;d0=d1
            poly=clipped
        if poly:result=min(result,min(float(p[2]) for p in poly))
    return result


def docking(work,item,base,host,center,width,frame,slim=False):
    """D-keyed cartridge with outside flange, pilot bores and a fitting-contour plug."""
    u,r,f=frame;lo,hi=bounds(base,u);tree=aw.world_tree(base);radius=width*.5
    outer=radius+4.;side=radius+(4. if item.kind=='AIRWAY' else 1.5)
    flat=.85 if slim else .65
    screw_offset=radius+max(2.5,item.screw*.5+item.clearance+.8)
    # The compact root already includes its own passage walls. Give the seat
    # a narrow retaining lip, widening only around the two retaining bores.
    if slim:
        tab_radius=(item.screw+.4)*.5+.8
        outline=d_profile(radius+.8,flat)
        outline.extend((tab_radius*math.cos(a),sign*screw_offset+tab_radius*math.sin(a))
                       for sign in (-1,1) for a in np.linspace(0,2*math.pi,64,endpoint=False))
        hull=np.asarray(aw.convex_hull_2d(outline));edge=np.roll(hull,-1,axis=0)-hull
        normals=np.column_stack((edge[:,1],-edge[:,0]));heights=np.sum(normals*hull,axis=1)
        outer=float(hull[:,1].max());side=float(hull[:,0].max())
    def flange_profile(segments):
        if slim:
            result=[]
            for a in np.linspace(0,2*math.pi,segments,endpoint=False):
                direction=np.array((math.cos(a),math.sin(a)));den=normals@direction;valid=den>1e-12
                distance=float((heights[valid]/den[valid]).min())
                result.append(tuple(direction*distance))
            return result
        return [(max(-side*.65,side*math.cos(a)),outer*math.sin(a)) for a in np.linspace(0,2*math.pi,segments,endpoint=False)]
    left=min(x for x,y in flange_profile(256))
    def in_flange(x,y):
        if slim:return bool(np.all(normals@np.array((x,y))<heights))
        return (x/side)**2+(y/outer)**2<1
    points=[center]+[center+r*x+f*y for x,y in flange_profile(64)]
    # Sample inside the seat as well as its border before adding the underside
    # pad. Its top stays below the fitting surface, preserving that contour.
    for x in np.arange(left,side+.5,1.):
        for y in np.arange(-outer,outer+.5,1.):
            if in_flange(x,y):points.append(center+r*float(x)+f*float(y))
    levels=[column(tree,q,frame,lo,hi) for q in points]
    rear=min(v[0] for v in levels)-.3
    # A flat seat needs to clear the rear contour only within its footprint.
    # Screw engagement is a LOCAL constraint at the two pilot bores, not a
    # thickness allowance beneath every fitting-surface dip in the whole pad.
    base.data.calc_loop_triangles()
    vertices=np.array([base.matrix_world@v.co for v in base.data.vertices])
    triangles=np.array([t.vertices[:] for t in base.data.loop_triangles],dtype=int)
    co=vertices[triangles];px=co@r-center.dot(r);py=co@f-center.dot(f)
    projected=np.stack((px,py,co@u),axis=2)
    rear=min(rear,projected_min_height(projected,flange_profile(256))-.3)
    upward=np.cross(co[:,1]-co[:,0],co[:,2]-co[:,0])@u>1e-10
    original_rear=rear
    pilot_limits=[]
    for sign in (-1,1):
        # Circumscribe the bore plus a small construction tolerance. Include
        # complete triangle crossings, not just the screw-axis ray hit.
        rad=(item.screw*.5+.05)/math.cos(math.pi/64)
        profile=[(rad*math.cos(a),sign*screw_offset+rad*math.sin(a)) for a in np.linspace(0,2*math.pi,64,endpoint=False)]
        front=projected_min_height(projected[upward],profile)
        if not math.isfinite(front):raise RuntimeError('Cannot find fitting surface above a retaining screw; move the mount inward')
        pilot_limits.append(front-.2)
        rear=min(rear,front-.2-item.pilot_depth-1.)
    # Only the added backing is sampled. Its upper face sits inside the
    # existing shell; the original fitting triangles are retained by union.
    # Concentric rings give a smooth D perimeter without staircase edges.
    segments=256;steps=max(2,math.ceil(outer/.35));xy=center-u*center.dot(u)
    profile=flange_profile(segments)
    def inside_height(q):
        back,front=column(tree,q,frame,lo,hi)
        return (back+front)*.5
    verts=[xy+u*inside_height(xy)];faces=[]
    for k in range(1,steps+1):
        for x,y in profile:
            q=xy+r*(x*k/steps)+f*(y*k/steps);verts.append(q+u*inside_height(q))
    for j in range(segments):faces.append((0,1+j,1+(j+1)%segments))
    for k in range(steps-1):
        a=1+k*segments;b=a+segments
        for j in range(segments):
            t=(j+1)%segments;faces.extend([(a+j,b+j,b+t),(a+j,b+t,a+t)])
    bottom=len(verts);verts.extend([xy+r*x+f*y+u*rear for x,y in profile]);top=1+(steps-1)*segments
    faces.append(tuple(reversed(range(bottom,bottom+segments))))
    for j in range(segments):
        t=(j+1)%segments;faces.append((top+j,bottom+j,bottom+t,top+t))
    mesh=bpy.data.meshes.new('Insert_backing_work');mesh.from_pydata(verts,[],faces);mesh.update()
    pad=bpy.data.objects.new('Insert_backing_work',mesh);work.col.objects.link(pad);work.add(pad);closed(pad)
    # Refine only the added support at sharp folds. Inserting a local sample
    # keeps its neighbouring contact points, unlike lowering a whole triangle.
    fitting=[]
    # Interior sampling avoids ambiguous ray hits exactly on scan edges.
    for x in np.arange(left+.037,side,.15):
        for y in np.arange(-outer+.071,outer,.15):
            if not in_flange(x,y):continue
            q=xy+r*float(x)+f*float(y)+u*(hi+5)
            front=tree.ray_cast(q,-u,hi-lo+10)[0]
            if front is not None:fitting.append((q,front.dot(u)))
    for attempt in range(12):
        pad_tree=aw.world_tree(pad);refine={}
        for q,z in fitting:
            hit=pad_tree.ray_cast(q,-u,hi-rear+10)
            if hit[0] is None:continue
            excess=hit[0].dot(u)-z
            if excess>.003 and (hit[2] not in refine or excess>refine[hit[2]][0]):
                refine[hit[2]]=(excess,q,z)
        if not refine:break
        if attempt==11:raise RuntimeError('Seat backing cannot follow this sharp fitting fold; reduce its width or move its mark')
        vertices_pad=[v.co.copy() for v in pad.data.vertices];faces_pad=[]
        for face in pad.data.polygons:
            ids=list(face.vertices)
            if face.index not in refine:faces_pad.append(ids);continue
            _,q,front_z=refine[face.index];z=min(inside_height(q),front_z-.05)
            points=[vertices_pad[i] for i in ids]
            a,b,c=[np.array([v.dot(r),v.dot(f)]) for v in points[:3]]
            try:bary=np.linalg.solve(np.column_stack((b-a,c-a)),np.array([q.dot(r),q.dot(f)])-a)
            except np.linalg.LinAlgError:bary=np.array([0.,0.])
            if len(ids)!=3 or min(bary[0],bary[1],1-bary.sum())<1e-5:
                for i in ids:vertices_pad[i]+=u*min(0.,z-vertices_pad[i].dot(u))
                faces_pad.append(ids)
            else:
                k=len(vertices_pad);vertices_pad.append(q+u*(z-q.dot(u)))
                faces_pad.extend([(ids[0],ids[1],k),(ids[1],ids[2],k),(ids[2],ids[0],k)])
        replacement=bpy.data.meshes.new('Insert_backing_refined');replacement.from_pydata(vertices_pad,[],faces_pad);replacement.update()
        old=pad.data;pad.data=replacement
        if old.users==0:bpy.data.meshes.remove(old)
    closed(pad)
    stem=work.copy(pad,'Insert_stem_work')
    for v in stem.data.vertices:
        if abs(v.co.dot(u)-rear)<.002:v.co-=u*.1
    stem.data.update()
    host=merge(work,host,pad)
    stem,supports=support_backing_islands(work,host,stem,base,center,flange_profile(64),rear,frame)
    socket=profile_solid(work,'Insert_socket_work',d_profile(radius+item.clearance,flat),center,min(lo-5,rear-3),hi+5,frame)
    male=profile_solid(work,'Insert_plug_work',d_profile(radius,flat),center,lo-5,hi+5,frame)
    blank=work.copy(base,'Insert_blank_work');aw.cut(blank,male,'INTERSECT');work.delete(male);closed(blank)
    # Below the fitting surface only: connect the flange to the original skin.
    stem_clip=profile_solid(work,'Insert_stem_clip',d_profile(radius,flat),center,rear-3,hi+5,frame)
    aw.cut(stem,stem_clip,'INTERSECT');work.delete(stem_clip)
    # Retain the original fitting contour on the plug. The perimeter has the
    # selected assembly clearance; Restore original mold removes the seam too.
    blank=merge(work,blank,stem)
    flange=profile_solid(work,'Insert_flange_work',flange_profile(64),center,rear-2,rear,frame)
    blank=merge(work,blank,flange)
    aw.cut(host,socket,'DIFFERENCE');closed(host)
    screws=[]
    for sign,front_limit in zip((-1,1),pilot_limits):
        q=center+f*(sign*screw_offset);back,front=column(tree,q,frame,lo,hi)
        back=rear  # The flange meets the new planar underside with no gap.
        depth=min(item.pilot_depth,front_limit-back-1.)
        if depth<1:raise RuntimeError('Base too thin at a retaining screw; move the mount or increase local base thickness')
        xy=q-u*q.dot(u)
        tool=cylinder(work,'Insert_pilot_work',xy+u*(rear-3),xy+u*(back+depth),item.screw*.5,frame)
        aw.cut(host,tool,'DIFFERENCE');work.delete(tool)
        tool=cylinder(work,'Insert_flange_bore_work',xy+u*(rear-3),xy+u*(rear+.2),(item.screw+.4)*.5,frame)
        aw.cut(blank,tool,'DIFFERENCE');work.delete(tool)
        screws.append({'engagement_mm':depth,'reach_from_flange_mm':back+depth-(rear-2)})
    work.delete(socket);closed(blank)
    return blank,{'center':list(center),'right':list(r),'forward':list(f),'radius':outer,'rear':rear,'screws':screws,'flat_contact':True,'contact_gap_mm':0.,'seat_width_mm':width,'footprint_mm':[side-left,2*outer],'extra_backing_mm':original_rear-rear,'local_backing_supports':supports,'compact_mount':slim,'key_flat':flat,'screw_offset_mm':screw_offset}


def paw_profiles(width):
    # Reference arrangement: one broad central pad and four separate oval toes.
    profiles=[]
    pad=[]
    for a in np.linspace(0,2*math.pi,64,endpoint=False):
        radius=1+.13*math.cos(3*(a-math.pi*.5))
        pad.append((math.cos(a)*.31*width*radius,(-.16+.25*math.sin(a)*radius)*width))
    profiles.append(pad)
    for x,y,rx,ry,angle in [(-.37,.13,.115,.16,-.35),(-.16,.36,.115,.17,-.12),(.16,.36,.115,.17,.12),(.37,.13,.115,.16,.35)]:
        profiles.append([((x+rx*math.cos(a)*math.cos(angle)-ry*math.sin(a)*math.sin(angle))*width,
                          (y+rx*math.cos(a)*math.sin(angle)+ry*math.sin(a)*math.cos(angle))*width) for a in np.linspace(0,2*math.pi,40,endpoint=False)])
    return profiles


def ocular_center_contact(ocular,frame):
    """Project the ocular's outline center onto its back along the mold axis.

    Use actual world-space vertices, not the object origin or the raised iris.
    Build passes an untouched host copy, so existing key recesses cannot shift
    the result when updating an already installed pedestal.
    """
    u,r,f=frame
    points=np.array([ocular.matrix_world@v.co for v in ocular.data.vertices])
    if not len(points):raise RuntimeError('Choose a non-empty ocular for the pedestal')
    x=points@r;y=points@f;z=points@u
    center=r*float((x.min()+x.max())*.5)+f*float((y.min()+y.max())*.5)
    tree=BVHTree.FromPolygons(points.tolist(),[tuple(p.vertices) for p in ocular.data.polygons])
    hit=tree.ray_cast(center+u*float(z.min()-5),u,float(np.ptp(z)+10))
    if hit[0] is None or hit[1].dot(u)>-.15:
        raise RuntimeError('The ocular center does not present a back surface toward the base; check its orientation')
    return hit[0],tree,hit


def pedestal(work,item,blank,ocular,frame,coating=None,start=None,root_radius=None,recess_tools=None):
    u,r,f=frame;start=Vector(item.base1) if start is None else Vector(start);end,tree,hit=ocular_center_contact(ocular,frame);height=(end-start).dot(u)
    if height<2:raise RuntimeError('The ocular key must be above its base mounting point along the mold axis')
    # Check residual thickness throughout every key footprint, not only centre.
    profiles=paw_profiles(item.key_size) if item.key=='PAW' else [d_profile(item.key_size*.5)]
    keys=[];cutters=[];z=end.dot(u)
    for profile in profiles:
        back_levels=[]
        for x,y in profile[::4]+[(0,0)]:
            q=end+r*x+f*y
            inside=tree.ray_cast(q-u*2,u,4)[0]
            if inside is None:raise RuntimeError('The centered key footprint extends beyond the ocular; reduce its width')
            exit_hit=tree.ray_cast(inside+u*.01,u,30)[0]
            if exit_hit is None or exit_hit.dot(u)-(z+item.key_depth+item.key_clearance)<1.:
                raise RuntimeError('Not enough ocular thickness behind the centered key; reduce its depth or width')
            back_levels.append(inside.dot(u))
        cx=sum(x for x,y in profile)/len(profile);cy=sum(y for x,y in profile)/len(profile)
        local=[(x-cx,y-cy) for x,y in profile];key_center=end+r*cx+f*cy
        key=profile_solid(work,'Ocular_key_work',local,key_center,min(z-.8,min(back_levels)-.5),z+item.key_depth,frame,top_scale=.88)
        # Expand each pad around its own centre. Scaling the whole paw layout
        # would move the toes and create interfering, leaning pins.
        expanded=ma_offset(local,item.key_clearance)
        xy=key_center-u*z
        levels=[(min(min(back_levels)-.5,z-1.),1.),(z-.8,1.),(z+item.key_depth,.88),(z+item.key_depth+max(.001,item.key_clearance),.88)]
        cutter=loft(work,'Ocular_recess_work',[[xy+u*level+r*x*scale+f*y*scale for x,y in expanded] for level,scale in levels])
        if ocular.data.materials:
            material=ocular.data.materials[ocular.data.polygons[hit[2]].material_index]
            if material is not None:cutter.data.materials.append(material)
        cutters.append(cutter);keys.append(key)
    radius=max(item.stem,item.key_size+1)*.5
    low,high=bounds(ocular,u);back=[];front=[]
    for radial in np.linspace(0,radius,max(3,math.ceil(radius/.5)+1)):
        for angle in np.linspace(0,2*math.pi,64,endpoint=False):
            q=end+r*(radial*math.cos(angle))+f*(radial*math.sin(angle));xy=q-u*q.dot(u)
            rear_hit=tree.ray_cast(xy+u*(low-1),u,high-low+2)[0]
            if rear_hit is None:raise RuntimeError('Pedestal support extends outside the ocular back; reduce Pedestal width')
            front_hit=tree.ray_cast(rear_hit+u*.001,u,high-low+2)[0]
            if front_hit is None:raise RuntimeError('Cannot locate the ocular thickness above the pedestal support')
            back.append(rear_hit.dot(u));front.append(front_hit.dot(u))
    top_level=max(back)+.15
    if top_level>=min(front)-.05:raise RuntimeError('The ocular is too thin or tilted for this pedestal width; reduce Pedestal width')
    top=end+u*(top_level-z)
    stem=loft(work,'Ocular_pedestal_work',round_rings([start-u*1,top],[root_radius or item.stem*.5,radius],frame))
    # Extend into the rear skin, then trim against the actual ocular. The
    # bearing surface conforms to its back with no deliberate axial gap.
    # Finish this small part before joining the detailed fitting-contour plug.
    aw.cut(stem,ocular,'DIFFERENCE')
    if coating:aw.cut(stem,coating,'DIFFERENCE')
    if recess_tools is not None:recess_tools.append(work.copy(stem,'Matching_ocular_bearing_surface'))
    had_materials=bool(ocular.data.materials)
    for key,cutter in zip(keys,cutters):
        if recess_tools is not None:recess_tools.append(work.copy(cutter,'Matching_ocular_key_recess'))
        stem=merge(work,stem,key);aw.cut(ocular,cutter,'DIFFERENCE')
        if coating:
            cutter.data.materials.clear()
            if coating.data.materials and coating.data.materials[0]:cutter.data.materials.append(coating.data.materials[0])
            aw.cut(coating,cutter,'DIFFERENCE')
        work.delete(cutter)
    if not had_materials:ocular.data.materials.clear()
    blank=merge(work,blank,stem)
    closed(ocular);blank['ocular_key_center_world']=list(end);blank['ocular_bearing_gap_mm']=0.;return blank


def wax(work,item,blank,center,dock,frame,fill_ml):
    u=frame[0];rear=dock['rear']-2;xy=center-u*center.dot(u)
    inner=item.feed_diameter*.5;h=item.cup_height
    reserve=fill_ml*1000*item.allowance/100
    # Frustum capacity includes its conical funnel; never under-size the reserve.
    top=max(inner+1,(-inner+math.sqrt(max(0,12*reserve/(math.pi*h)-3*inner*inner)))*.5)
    wall=2.;mouth=xy+u*(rear-h)
    if item.wax_mode=='INJECTION':
        nozzle=max(inner,item.nozzle_diameter*.5)
        def capacity(rad):return math.pi*(h*.2*(nozzle**2+nozzle*rad+rad**2)+h*.8*(rad**2+rad*inner+inner**2))/3
        lower=max(inner,nozzle);upper=max(top,lower)
        while capacity(upper)<reserve:upper*=1.2
        for _ in range(40):
            mid=(lower+upper)*.5
            if capacity(mid)<reserve:lower=mid
            else:upper=mid
        top=upper;centers=[mouth,xy+u*(rear-h*.8),xy+u*rear];radii=[nozzle,top,inner]
        capacity_ml=capacity(top)/1000
    else:
        centers=[mouth,xy+u*rear];radii=[top,inner]
        capacity_ml=math.pi*h*(inner*inner+inner*top+top*top)/3000
    body=loft(work,'Wax_feed_work',round_rings(centers+[xy+u*(rear+.5)], [v+wall for v in radii]+[inner+wall],frame))
    blank=merge(work,blank,body)
    hole=loft(work,'Wax_funnel_work',round_rings([mouth-u*1]+centers+[xy+u*(rear+.6)],[radii[0]]+radii+[inner],frame))
    aw.cut(blank,hole,'DIFFERENCE');work.delete(hole)
    outlet=bounds(blank,u)[1]+1.
    hole=cylinder(work,'Wax_bore_work',xy+u*(rear-h-1),xy+u*outlet,inner,frame)
    aw.cut(blank,hole,'DIFFERENCE');work.delete(hole);closed(blank)
    if aw.world_tree(blank).ray_cast(xy+u*(rear-h-.5),u,outlet-(rear-h-.5))[0] is not None:
        raise RuntimeError('The wax passage is obstructed; change the seat position or feed diameter')
    return blank,{'fill_volume_ml':fill_ml,'allowance_percent':item.allowance,'reserve_target_ml':reserve/1000,
                  'reservoir_capacity_ml':capacity_ml,'mode':item.wax_mode,'mouth_diameter_mm':2*radii[0]}


def state(scene):return json.loads(scene.get(STATE,'{}'))


def guard_ocular_rebuild(scene):
    # The host ledger records the last successful installation. Editable entries
    # can include an unbuilt pedestal, or change after installation. Only an
    # installed ocular host (including its clear coating) needs protection here.
    # Blanked pedestals still retain ocular hosts for later rebuild / restore.
    ocular_hosts=set(state(scene).get('hosts',{}))-{'Mold_Base','Mold_Cap','Mold_Wedge'}
    if ocular_hosts:
        raise RuntimeError('An ocular pedestal is installed. Restore the original mold in Interchangeable inserts before rebuilding the ocular')


def ma_offset(profile,clearance):
    if not clearance:return profile
    from .mold_auto import offset_polygon
    return offset_polygon(profile,clearance)


def settings_signature(scene):
    result=[]
    for item in scene.mold_inserts.items:
        data={}
        for prop in item.bl_rna.properties:
            key=prop.identifier
            if key in {'rna_type','report','seat_piece','seat_spec'}:continue
            value='NONE' if key=='seat_owner' and item.seat_mode=='OWN' else getattr(item,key)
            if prop.type=='POINTER':value=value.name if value else ''
            elif getattr(prop,'is_array',False):value=list(value)
            data[key]=value
        result.append(data)
    return json.dumps(result,sort_keys=True)


class HolderPedestal:
    """Build-time view: holder owns its controls, pedestal owns its connection."""
    def __init__(self,pedestal,holder):
        object.__setattr__(self,'pedestal',pedestal);object.__setattr__(self,'holder',holder)
    def __getattr__(self,name):
        if name=='ocular_use':return 'SUBSTITUTE' if self.holder.holder_enabled else 'REAL'
        return getattr(self.holder if name in HOLDER_FIELDS else self.pedestal,name)
    def __setattr__(self,name,value):setattr(self.pedestal,name,value)


def build_entries(scene):
    items=list(scene.mold_inserts.items);holders={}
    for item in items:
        if item.kind!='HOLDER':continue
        parent=next((i for i in items if i.kind=='OCULAR' and i.uid==item.pedestal_uid),None)
        if parent is None:raise RuntimeError(f'{item.name}: choose an ocular pedestal first')
        if parent.uid in holders:raise RuntimeError('Use one ocular space holder per pedestal')
        if item.holder_enabled and parent.mode!='INSERT':raise RuntimeError('Enable the ocular pedestal before including its space holder')
        holders[parent.uid]=item
    return [HolderPedestal(i,holders[i.uid]) if i.uid in holders else i for i in items if i.kind!='HOLDER'],holders


def migrate_holders(scene):
    """Expose existing substitute settings as entries without rebuilding meshes."""
    if not hasattr(scene,'mold_inserts'):return
    p=scene.mold_inserts;st=state(scene)
    old=json.loads(st.get('settings','[]'));current=json.loads(settings_signature(scene))
    strip=lambda rows:[{k:v for k,v in row.items() if k not in {'pedestal_uid','holder_enabled'}} for row in rows]
    unchanged=bool(st) and strip(old)==strip(current)
    for parent in list(p.items):
        if parent.kind!='OCULAR' or parent.ocular_use!='SUBSTITUTE':continue
        holder=next((i for i in p.items if i.kind=='HOLDER' and i.pedestal_uid==parent.uid),None)
        if holder is None:
            holder=p.items.add();holder.kind='HOLDER';holder.uid=uuid.uuid4().hex;holder.name='Ocular space holder';holder.pedestal_uid=parent.uid
            for field in HOLDER_FIELDS:setattr(holder,field,getattr(parent,field))
            holder.report=parent.report
        parent.ocular_use='REAL'
        for obj in scene.objects:
            if obj.get('insert_uid')==parent.uid and obj.get('insert_role') in {'FORMER','FORMER_COMBINED'}:
                obj['insert_uid']=holder.uid;obj['insert_kind']='HOLDER'
    # New default properties and the entry conversion do not change built geometry.
    if unchanged:st['settings']=settings_signature(scene);scene[STATE]=json.dumps(st)


@bpy.app.handlers.persistent
def load_holders(_):
    for scene in bpy.data.scenes:
        repair_seat_links(scene);migrate_holders(scene)


def migrate_open_cases():
    load_holders(None)
    return None


def checked_hosts(scene,st):
    for name,info in st.get('hosts',{}).items():
        obj=bpy.data.objects.get(name)
        if obj is None or signature(obj)!=info['installed']:
            raise RuntimeError(f'{name} changed after insert installation. Undo that edit or return to the saved case before rebuilding inserts')
        reference=bpy.data.objects.get(info['reference'])
        if reference is None or ('original' in info and signature(reference)!=info['original']):
            raise RuntimeError('An original insert reference was changed or removed; recover the saved case')


class SeatChangeRequired(RuntimeError):
    """The complete candidate passed checks, but an existing seat would change."""


def reservoir_volume(scene, item, estimate):
    return item.cavity_ml or estimate['fill_ml']


def draw_reservoir_volume(layout,scene,item):
    layout.prop(item,'cavity_ml',text='Fill estimate (mL; 0 = automatic)')
    if item.cavity_ml:return
    layout.label(text='Automatic: Sculpt estimate + 20%')
    layout.label(text='No mold-volume measurement needed')


def selected_seat(scene):
    item=active(scene)
    if item and item.kind=='HOLDER':
        item=next((i for i in scene.mold_inserts.items if i.uid==item.pedestal_uid),None)
    if item and item.seat_mode=='SHARED':
        item=next((i for i in scene.mold_inserts.items if i.uid==item.seat_owner),None)
    return item


def marking_target(scene,target):
    """Draw on the retained original fitting surface after a seat is cut."""
    st=state(scene)
    if st:
        checked_hosts(scene,st)
        info=st.get('hosts',{}).get(target.name) if target else None
        if info:return bpy.data.objects[info['reference']]
    return target


def marking_source(scene,target):
    name=next((n for n,info in state(scene).get('hosts',{}).items() if info['reference']==target.name),target.name)
    return {'name':name,'signature':signature(target)}


def assign_insert(scene):
    """Assign attachments to built seats; never silently create another socket."""
    repair_seat_links(scene)
    p=scene.mold_inserts
    def find(uid):return next((i for i in p.items if i.uid==uid and i.kind!='HOLDER' and i.seat_mode=='OWN'),None)
    first=find(p.target_seat);second=find(p.second_seat) if p.use_two_seats else None
    if first is None and p.target_seat=='NONE' and not p.use_two_seats:
        selected=selected_seat(scene)
        if selected and selected.seat_spec:first=selected
        else:
            built=[i for i in p.items if i.kind!='HOLDER' and i.seat_mode=='OWN' and i.seat_spec]
            if len(built)==1:first=built[0]
        if first:p.target_seat=first.uid
    if first is None or not first.seat_spec:raise RuntimeError('Build and select the first mounting seat before choosing its insert')
    if p.use_two_seats and (second is None or not second.seat_spec or second.uid==first.uid):
        raise RuntimeError('Choose two different built mounting seats')
    if p.new_kind=='BLANK':
        closing={seat.uid for seat in (first,second) if seat}
        for item in p.items:
            if item.bridge_seat in closing:item.bridge_seat='NONE'
            if raw_seat_link(item,'flow_seat') in closing:item.flow_seat='NONE'
        for seat in (first,second):
            if seat:
                for member in seat_members(seat):member.mode='BLANK'
                seat.bridge_seat='NONE'
        return
    assignments=[(first,'WAX' if p.new_kind=='COMBINED' else p.new_kind)]
    if second:assignments.append((second,'OCULAR' if p.new_kind=='COMBINED' else ('SEAT' if p.new_kind=='AIRWAY' or p.two_seat_layout=='CONNECTED' else p.new_kind)))
    for seat,kind in assignments:
        if seat.kind not in {'SEAT',kind}:raise RuntimeError(f'{seat.seat_name or seat.name} already has {seat.name}; choose an unused seat or keep that attachment')
        if len(seat_members(seat))>1:raise RuntimeError('This seat already has a shared attachment; select it in the list to change the installed piece')
    for seat,kind in assignments:
        was_empty=seat.kind=='SEAT';seat.kind=kind;seat.name=(seat.seat_name or 'Seat')+' · '+dict((a,b) for a,b,_ in KINDS)[kind]
        seat.mode='BLANK' if kind=='SEAT' else 'INSERT';seat.bridge_seat='NONE';seat.flow_seat='NONE'
        if was_empty:
            if kind=='WAX':seat.reservoir_layout='SIDE_CUPS'
            elif kind=='AIRWAY':
                seat.airway_shape='OUTLINE';seat.chambers='SHARED'
                # A mounting-point click is not a traced patient opening.
                if not json.loads(seat.airway_outlines).get('BASE1'):
                    seat.marks=json.dumps(sorted(set(json.loads(seat.marks))-{'BASE1'}))
                    sources=json.loads(seat.mark_sources);sources.pop('BASE1',None);seat.mark_sources=json.dumps(sources)
            elif kind=='OCULAR':seat.ocular=scene.ocular_production.color_obj or scene.anaplast.ocular_obj
    if p.new_kind=='COMBINED' and second is None:
        child=p.items.add();child.uid=uuid.uuid4().hex;child.kind='OCULAR';child.name=(first.seat_name or 'Seat')+' · Ocular pedestal'
        child.seat_mode='SHARED';child.seat_owner=first.uid;child.ocular=scene.ocular_production.color_obj or scene.anaplast.ocular_obj
    if second and p.two_seat_layout=='CONNECTED':first.bridge_seat=second.uid
    if second and p.new_kind=='AIRWAY':first.flow_seat=second.uid
    p.active=next(n for n,i in enumerate(p.items) if i.uid==first.uid)


def connect_seat_parts(work,first,second,dock_a,dock_b,item_a,item_b,frame):
    """Rear connector only: keep the patient-side geometry and both seats intact."""
    u,r,f=frame;ca=Vector(dock_a['center']);cb=Vector(dock_b['center'])
    # Contact the flange beside the central feed bore. Run below both flanges.
    axis=cb-ca;axis-=u*axis.dot(u)
    if axis.length<.01:raise RuntimeError('The two seat centers must be different')
    side=u.cross(axis.normalized())
    contacts=[Vector(d['center'])+side*(d['seat_width_mm']*.48) for d in (dock_a,dock_b)]
    bottom=min(dock_a['rear'],dock_b['rear'])-5.
    profiles=[]
    for q in contacts:
        profiles.extend(((q+r*(2*math.cos(a))+f*(2*math.sin(a))).dot(r),(q+r*(2*math.cos(a))+f*(2*math.sin(a))).dot(f)) for a in np.linspace(0,2*math.pi,64,endpoint=False))
    profile=aw.convex_hull_2d(profiles)
    bridge=profile_solid(work,'Two_seat_connector_work',profile,Vector((0,0,0)),bottom,bottom+2.5,frame)
    for q,dock in zip(contacts,(dock_a,dock_b)):
        xy=q-u*q.dot(u)
        bridge=merge(work,bridge,cylinder(work,'Connector_contact_work',xy+u*bottom,xy+u*(dock['rear']-.5),2.,frame))
    envelope=work.copy(bridge,'Two_seat_envelope_work')
    for obj in (first,second):
        exterior=work.copy(obj,'Seat_envelope_work')
        if isinstance(obj.get('cast_preview_solid'),bpy.types.Mesh):exterior.data=obj['cast_preview_solid'].copy()
        envelope=merge(work,envelope,exterior)
    first=work.copy(first,'Two_seat_insert_work');second=work.copy(second,'Second_seat_work')
    if any(obj.get('single_passage_radii') for obj in (first,second)):
        from . import insert_single_cup
        insert_single_cup.clear_pair_passages(work,(first,second),frame)
    passages=[]
    for obj in (first,second):passages.extend(json.loads(obj.get('compact_passages','[]')))
    combined=merge(work,first,bridge)
    combined=merge(work,combined,second)
    # Preserve retaining screw access through the new connector.
    for item,dock in ((item_a,dock_a),(item_b,dock_b)):
        c=Vector(dock['center']);forward=Vector(dock['forward'])
        for sign in (-1,1):
            q=c+forward*(sign*dock['screw_offset_mm']);xy=q-u*q.dot(u)
            cutter=cylinder(work,'Connector_screw_clearance_work',xy+u*(bottom-1),xy+u*(dock['rear']+.2),(item.screw+.4)*.5,frame)
            aw.cut(combined,cutter,'DIFFERENCE');work.delete(cutter)
    if passages:combined['compact_passages']=json.dumps(passages)

    combined['cast_preview_solid']=envelope.data.copy()
    combined['insert_seat_uids']=json.dumps([item_a.uid,item_b.uid])
    return combined


def fixed_seat_width(center,frame,footprints,minimum,flat=.85):
    """Fit roots into the existing D seat without moving or rotating its center."""
    _,r,f=frame;radius=minimum*.5;profile=d_profile(1.,flat)
    for a,b in zip(profile,profile[1:]+profile[:1]):
        dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy)
        nx,ny=dy/length,-dx/length;h=nx*a[0]+ny*a[1]
        for point,size in footprints:
            delta=point-center
            radius=max(radius,(nx*delta.dot(r)+ny*delta.dot(f)+size+.5)/h)
    return radius*2


def build(context,seat_only_uid=None,allow_seat_change=False):
    from . import insert_seat_layout as placement
    from . import mold_insert_reservoir as compact
    from . import mold_ocular_substitute as substitute
    scene=context.scene;validate_seat_links(scene,seat_only_uid);explode.collapse(scene)
    if abs(scene.unit_settings.scale_length-.001)>1e-8:raise RuntimeError('Set up the case in millimetres first')
    migrate_holders(scene);entries,holder_entries=build_entries(scene);groups=seat_groups(entries)
    if not entries:raise RuntimeError('Add an insert first')
    if bpy.data.objects.get('Mold_Base') is None:raise RuntimeError('Build the Mold Base first')
    if any(bpy.data.objects.get(n) and bpy.data.objects[n].get('flipped') for n in ('Mold_Base','Mold_Cap')):
        raise RuntimeError('Return the flipped mold to its assembled position first')
    st=state(scene);checked_hosts(scene,st)
    base_amount=st.get('volume')
    if not base_amount:
        try:base_amount=amounts.read(scene,validate=True)
        except ValueError:base_amount={}
    only_members={i.uid for g in groups if seat_only_uid=='ALL' or g[0].uid==seat_only_uid for i in g}
    feed_items=[i for i in entries if i.uid not in only_members and (i.kind=='WAX' or (i.kind=='AIRWAY' and i.airway_path=='CURVED' and i.airway_reservoir))]
    from . import reservoir_estimate
    estimate=reservoir_estimate.estimate(scene) if any(not i.cavity_ml for i in feed_items) else {}
    fills={i.uid:reservoir_volume(scene,i,estimate) for i in feed_items}
    roots={g[0].uid:g[0] for g in groups};flow_pairs={};vent_owners={}
    for item in feed_items:
        if item.kind!='AIRWAY':continue
        uid=raw_seat_link(item,'flow_seat')
        if uid=='NONE':uid=raw_seat_link(item,'bridge_seat')
        other=roots.get(uid)
        if other and other.kind in {'SEAT','OCULAR'} and other.uid not in only_members:
            if uid in vent_owners:raise RuntimeError('Choose a separate air vent seat for each wax feed')
            flow_pairs[item.uid]=other;vent_owners[uid]=item
    def mount_center(item):
        saved=json.loads(item.seat_spec) if item.seat_spec else json.loads(item.report).get('mount',{}) if item.report else {}
        return Vector(saved.get('center',item.base1))
    work=Work(scene);snapshots=[]
    generated=[];hosts={};reports={};docks=[];seat_specs={};seat_changes=[];built_docks={}
    only_members={i.uid for g in groups if seat_only_uid=='ALL' or g[0].uid==seat_only_uid for i in g}
    try:
        for item in entries:
            if item.uid in only_members:
                if item.seat_mode=='OWN' and not item.seat_spec and 'BASE1' not in json.loads(item.marks):
                    raise RuntimeError(f'{item.name}: mark the mounting seat on Mold Base')
            else:validate_marks(item)
        names={name for name in ('Mold_Base','Mold_Cap','Mold_Wedge') if bpy.data.objects.get(name)}
        if any(i.kind=='AIRWAY' and i.mode=='INSERT' and i.airway_path!='CURVED' and i.uid not in only_members for i in entries):names.add('Mold_Cap')
        names.update(i.ocular.name for i in entries if i.kind=='OCULAR' and i.ocular)
        production=getattr(scene,'ocular_production',None)
        coatings={i.ocular.name:production.clear_obj.name for i in entries
                  if i.kind=='OCULAR' and i.ocular and production and
                  i.ocular==production.color_obj and production.clear_obj}
        names.update(coatings.values())
        names.update(st.get('hosts',{}))
        originals={}
        for name in names:
            obj=bpy.data.objects.get(name)
            if obj is None or obj.type!='MESH':raise RuntimeError(f'Missing {name}')
            if obj.modifiers:raise RuntimeError(f'Apply modifiers on {name} before adding inserts')
            closed(obj)
            info=st.get('hosts',{}).get(name)
            ref=bpy.data.objects.get(info['reference']) if info else work.copy(obj,'InsertOriginal_'+name)
            if ref is None:raise RuntimeError('An insert construction reference is missing; recover the saved case')
            if not info:snapshots.append(ref)
            originals[name]=ref;hosts[name]=work.copy(ref,'InsertHost_'+name+'_work')
        base=originals['Mold_Base'];host=hosts['Mold_Base'];frame=aw.frame_for(base);u=frame[0]
        # Plan both blank seats together before making any mounting geometry.
        planned={}
        if seat_only_uid=='ALL' and len(groups)==2:
            plans=[]
            for group in groups:
                seat=group[0];saved=json.loads(seat.seat_spec) if seat.seat_spec else None
                center=Vector(saved['center'] if saved else seat.base1)
                initial=(u,Vector(saved['right']),Vector(saved['forward'])) if saved else frame
                initial=placement.rotated(initial,seat.seat_angle-(saved.get('angle_setting',0.) if saved else 0.))
                width=max(seat.dock_diameter,saved['width'] if saved else 0.)
                plans.append((seat,center,initial,width,True,not saved and seat.seat_auto_orient))
            planned=placement.plan_pair(plans,frame)
        mark_cache={}
        for item in entries:
            for mark,source in json.loads(item.mark_sources).items():
                if item.uid in only_members and mark!='BASE1':continue
                if item.kind=='OCULAR' and mark=='EYE':continue  # obsolete manual ocular mark
                if item.seat_mode=='SHARED' and mark=='BASE1':continue
                obj=originals.get(source['name'],bpy.data.objects.get(source['name']))
                if not mark_matches(item,mark,obj,source,mark_cache):
                    raise RuntimeError(f'{item.name}: the surface for {mark} changed; restore and place its marks again')
        def attachment(item,insert,center,dock,cores=(),tips=(),apply=False,root_radius=None,compact_ocular=None,has_pedestal=False):
            detail={'mount':dock}
            if item.kind=='AIRWAY':
                if item.airway_shape=='OUTLINE' and item.airway_path=='CURVED':
                    from . import airway_cavity
                    for core in cores:
                        insert=airway_cavity.attach(work,item,insert,core,base,host,hosts.get('Mold_Cap'),center,frame,apply)
                    checks=[json.loads(core['airway_cavity_report']) for core in cores]
                    detail.update(chambers=item.chambers,airway_path='CURVED',airway_checks=checks,removal_validated=False,cap_applied=apply and 'Mold_Cap' in hosts)
                    detail['mount_resampled_mm']=insert.get('airway_mount_resampled_mm',0.)
                    detail['mount_sampled_deviation_mm']=insert.get('airway_mount_sampled_deviation_mm',0.)
                    if item.airway_reservoir:
                        fill=fills[item.uid]
                        if item.uid in flow_pairs:
                            from . import insert_single_cup
                            insert,info=insert_single_cup.make(work,item,insert,center,dock,frame,base,fill,'FEED',center-mount_center(flow_pairs[item.uid]))
                        else:
                            ports=airway_cavity.reservoir_ports(item,insert,center,base,frame)
                            insert,info=compact.make(work,item,insert,center,dock,frame,fill,base,port_data=ports)
                        detail.update(info)
                        detail['fill_estimate']=dict(estimate) if not item.cavity_ml else {'method':'MANUAL','fill_ml':item.cavity_ml}
                        if apply:aw.cut(host,insert,'DIFFERENCE')
                    return insert,detail
                if 'Mold_Cap' not in hosts and apply:raise RuntimeError('Build the cap before the airway')
                for core_index,(core,tip) in enumerate(zip(cores,tips),1):
                    core=work.copy(core,'Airway_attachment_work')
                    if apply:
                        cap=hosts['Mold_Cap'];test=work.copy(core,'Airway_contact_work');aw.cut(test,cap,'INTERSECT')
                        if len(test.data.vertices) and item.airway_shape=='OUTLINE' and item.airway_path=='DRAIN':
                            from .airway_marking import outline
                            from mathutils.geometry import tessellate_polygon
                            rim=[Vector(p) for p in outline(item,f'TIP{core_index}')]
                            end_tree=BVHTree.FromPolygons(rim,[tuple(range(len(rim)))])
                            if any(end_tree.find_nearest(test.matrix_world@v.co)[3]>.85 for v in test.data.vertices):
                                raise RuntimeError('Airway crosses the cap away from the marked nostril; reposition its path or outline')
                        elif len(test.data.vertices):
                            low,_=bounds(test,u)
                            tip_level=tip.dot(u)
                            if item.airway_shape=='OUTLINE':
                                from .airway_marking import outline
                                tip_level=float((outline(item,f'TIP{core_index}')@np.asarray(u)).min())
                            if low<tip_level-.8:raise RuntimeError('Airway crosses the cap before the nostril; reposition the marks')
                        work.delete(test);aw.cut(cap,core,'DIFFERENCE')
                        if item.airway_shape=='OUTLINE' and item.airway_path=='DRAIN':
                            # The marked head may overhang a compact mounting stem.
                            # Seat its short root extension into the base surface.
                            aw.cut(host,core,'DIFFERENCE')
                    insert=merge(work,insert,core)
                detail.update(chambers=item.chambers,withdrawal_direction=list(-u),release=('Downhill in upright Z; undercuts allowed' if item.airway_path=='DRAIN' else 'Smooth marked sections, narrowing toward nostril') if item.airway_shape=='OUTLINE' else 'Nested axial sections; 2 mm coaxial nostril lead')
            elif item.kind=='OCULAR':
                if not item.ocular:raise RuntimeError('Choose the generated ocular')
                if substitute.enabled(item):
                    insert,info=substitute.build(work,item,insert,base,[originals[n] for n in ('Mold_Base','Mold_Cap','Mold_Wedge') if n in originals],center,frame,dock,originals[item.ocular.name],root_radius)
                    detail.update(info)
                    return insert,detail
                eye=hosts[item.ocular.name];coating=hosts.get(coatings.get(item.ocular.name))
                if not apply:
                    eye=work.copy(originals[item.ocular.name],'Spare_eye_work')
                    if coating:coating=work.copy(originals[coatings[item.ocular.name]],'Spare_coating_work')
                insert=pedestal(work,item,insert,eye,frame,coating,start=center,root_radius=root_radius)
                detail.update(key=item.key,ocular_key_center_world=list(insert['ocular_key_center_world']),ocular_key_position='AUTOMATIC_CENTER')
            else:
                fill=fills[item.uid]
                if item.uid in flow_pairs:
                    from . import insert_single_cup
                    insert,details=insert_single_cup.make(work,item,insert,center,dock,frame,base,fill,'FEED',center-mount_center(flow_pairs[item.uid]))
                elif compact.enabled(item):
                    insert,details=compact.make(work,item,insert,center,dock,frame,fill,base,compact_ocular,has_pedestal)
                else:insert,details=wax(work,item,insert,center,dock,frame,fill)
                detail.update(details)
                detail['fill_estimate']=dict(estimate) if not item.cavity_ml else {'method':'MANUAL','fill_ml':item.cavity_ml}
            return insert,detail

        def record(obj,item,role,enabled,seat_uid,kind=None):
            if role in {'FORMER','FORMER_COMBINED'} and item.uid in holder_entries:
                item=holder_entries[item.uid];kind='HOLDER'
            discard_boolean_dust(obj)
            if compact_mode:compact.clean_seams(obj)
            closed(obj,connected=True);obj['insert_uid']=item.uid;obj['insert_role']=role
            obj['insert_kind']=kind or item.kind;obj['anaplast_part']='INSERT';obj['insert_active']=enabled;obj['insert_seat_uid']=seat_uid
            for key in ('anaplast_construction','reference_only','workflow_transaction'):
                if key in obj:del obj[key]
            if base.get('mold_set'):obj['mold_set']=base['mold_set']
            obj.name=f'Mold_Insert_{(kind or item.kind).title()}_{item.uid[:4]}_{role.title()}'
            obj.color=(.25,.6,.7,1) if role!='BLANK' else (.7,.65,.35,1)
            for face in obj.data.polygons:face.use_smooth=item.kind=='AIRWAY' and item.airway_path=='CURVED'
            generated.append(obj)

        for group in groups:
            owner=group[0];center=Vector(owner.base1);width=owner.dock_diameter;cores=[];tips=[]
            seat_only=owner.kind=='SEAT' or owner.uid in only_members
            previous_seat=json.loads(owner.seat_spec) if owner.seat_spec else None
            # Restore removes installed sockets, but the saved mount remains
            # the placement plan. An airway's BASE1 is its patient opening,
            # which may be far from the separately marked mounting seat.
            if previous_seat is None and owner.report:
                old_dock=json.loads(owner.report).get('mount',{})
                if 'seat_width_mm' in old_dock:
                    saved=next((row for row in json.loads(st.get('settings','[]')) if row.get('uid')==owner.uid),{})
                    previous_seat=dict(center=old_dock['center'],anchor=list(owner.base1),right=old_dock['right'],forward=old_dock['forward'],width=old_dock['seat_width_mm'],compact_mount=old_dock.get('compact_mount',False),**{k:saved.get(k,getattr(owner,k)) for k in ('clearance','screw','pilot_depth')})
            if previous_seat:center=Vector(previous_seat['anchor'])
            ocular=next((i for i in group if i.kind=='OCULAR'),None)
            wax_item=next((i for i in group if i.kind=='WAX'),None)
            compact_mode=seat_only or (previous_seat.get('compact_mount',True) if previous_seat else (compact.enabled(wax_item) or (owner.kind=='AIRWAY' and owner.airway_shape=='OUTLINE' and owner.airway_path=='CURVED')))
            root_radius=compact.dimensions(wax_item,ocular)['radius'] if compact.enabled(wax_item) else None
            if wax_item and wax_item.uid in flow_pairs:root_radius=max(wax_item.feed_diameter*.5+1.5,ocular.stem*.5 if ocular else 0.)
            if not seat_only and substitute.enabled(ocular):substitute.plan(work,ocular,base,center,frame)
            if not seat_only and owner.kind=='AIRWAY':center,width,cores,tips=airway(work,owner,frame)
            wax_center=center.copy()
            if not seat_only and len(group)>1 and not compact_mode:
                if not ocular.ocular:raise RuntimeError('Choose the generated ocular')
                eye_end=ocular_center_contact(originals[ocular.ocular.name],frame)[0]
                direction=frame[1] if (eye_end-center).dot(frame[1])<=0 else -frame[1]
                offset=max(ocular.stem,ocular.key_size+1)*.5+wax_item.feed_diameter*.5+2.
                wax_center=center+direction*offset
                # Leave material between the wax bore and pedestal stem.
            seat_center=center.copy();seat_frame=frame
            footprints=[]
            if not seat_only and owner.kind!='AIRWAY':
                if root_radius is not None:footprints.append((center,root_radius))
                else:
                    if ocular:footprints.append((center,max(ocular.stem,ocular.key_size+1)*.5))
                    if wax_item:footprints.append((wax_center,wax_item.feed_diameter*.5+2.))
                if len(footprints)>1:
                    # Put the two roots along the full diameter of the D, with
                    # the screw pads at its ends, instead of wasting room on
                    # the truncated side. Keep attachment positions unchanged.
                    a,ar=footprints[0];b,br=footprints[1];axis=(b-a).normalized()
                    seed=(a+b)*.5+axis*(br-ar)*.5
                    seat_frame=(u,-frame[2],frame[1])
                else:seed=center
                seat_center,width=compact_seat(seed,seat_frame,footprints,width,slim=compact_mode)
            if owner.kind=='AIRWAY' and owner.airway_reservoir and not seat_only:
                root=Vector(previous_seat['center']) if previous_seat else center
                # fixed_seat_width adds 0.5 mm itself; keep 1.5 mm total wall
                # around a single bore without unnecessarily widening a seat.
                footprints=[(root,owner.feed_diameter*.5+1.0 if owner.uid in flow_pairs else compact.dimensions(owner)['radius'])]
            if owner.uid in vent_owners:
                root=Vector(previous_seat['center']) if previous_seat else center
                footprints.append((root,vent_owners[owner.uid].vent_diameter*.5+1.5))
            if previous_seat:
                seat_center=Vector(previous_seat['center'])
                seat_frame=(frame[0],Vector(previous_seat['right']),Vector(previous_seat['forward']))
                angle_change=owner.seat_angle-previous_seat.get('angle_setting',0.)
                seat_frame=placement.rotated(seat_frame,angle_change)
                if abs(angle_change)>1e-6:seat_changes.append(f'{owner.seat_name or owner.name}: rotate mounting seat by {angle_change:.1f} degrees')
                width=fixed_seat_width(seat_center,seat_frame,footprints,max(width,previous_seat['width']),.85 if compact_mode else .65)
                changed=[key for key in ('clearance','screw','pilot_depth') if abs(getattr(owner,key)-previous_seat[key])>1e-6]
                if previous_seat.get('compact_mount',True)!=compact_mode:changed.append('mount_profile')
                if width>previous_seat['width']+.01:
                    seat_changes.append(f'{owner.name}: seat width {previous_seat["width"]:.2f} → {width:.2f} mm')
                if changed:seat_changes.append(owner.name+': mounting fit / screw settings changed')
                if owner.kind=='AIRWAY':center=seat_center.copy()
            else:
                seat_frame=placement.rotated(seat_frame,owner.seat_angle)
                if footprints:width=fixed_seat_width(seat_center,seat_frame,footprints,width,.85 if compact_mode else .65)
            # A restored two-seat setup can need a wider wax-feed root.
            # Plan both unbuilt seats together at the actual required width.
            if not st and len(groups)==2 and not docks and not planned and all(not g[0].seat_spec for g in groups):
                other=groups[1][0]
                if other.kind=='SEAT':
                    saved=json.loads(other.report).get('mount',{}) if other.report else {}
                    other_center=Vector(saved.get('center',other.base1))
                    other_frame=(u,Vector(saved['right']),Vector(saved['forward'])) if saved else frame
                    planned=placement.plan_pair([(owner,seat_center,seat_frame,width,compact_mode,owner.seat_auto_orient),
                        (other,other_center,other_frame,other.dock_diameter,True,other.seat_auto_orient)],frame)
            if owner.uid in planned:seat_frame=planned[owner.uid]
            obstacles=list(docks)
            for other_group in groups:
                other=other_group[0]
                if other.uid==owner.uid or other.uid in built_docks or not other.seat_spec:continue
                saved=json.loads(other.seat_spec)
                other_frame=placement.rotated((u,Vector(saved['right']),Vector(saved['forward'])),other.seat_angle-saved.get('angle_setting',0.))
                other_poly=placement.footprint(other,max(other.dock_diameter,saved['width']),saved.get('compact_mount',True))
                obstacles.append((other.seat_name or other.name,placement.projected(Vector(saved['center']),other_frame,other_poly,frame)))
            # Root geometry stays put. Automatic turning is limited to seats
            # without offset roots; a built or manually turned seat is fixed.
            auto=not previous_seat and owner.seat_auto_orient and owner.uid not in planned and not footprints
            seat_frame,seat_outline=placement.choose(owner,seat_center,seat_frame,width,compact_mode,obstacles,frame,auto)
            docks.append((owner.seat_name or owner.name,seat_outline))
            blank,dock=docking(work,owner,base,host,seat_center,width,seat_frame,slim=compact_mode)
            if compact_mode:
                discard_boolean_dust(blank);compact.clean_seams(blank)
            dock.update(seat_uid=owner.uid,shared=len(group)>1)
            built_docks[owner.uid]=dock
            seat_specs[owner.uid]=json.dumps(dict(center=list(seat_center),anchor=list(center),right=list(seat_frame[1]),forward=list(seat_frame[2]),width=width,compact_mount=compact_mode,clearance=owner.clearance,screw=owner.screw,pilot_depth=owner.pilot_depth,angle_setting=owner.seat_angle))
            if seat_only:
                for item in group:reports[item.uid]=json.dumps({'mount':dock,'seat_only':True,'seat_owner':owner.name})
                record(blank,owner,'BLANK',True,owner.uid)
                continue
            enabled=[i for i in group if i.mode=='INSERT'];both=len(enabled)>1
            for item in group:
                insert=work.copy(blank,'Insert_active_work')
                insert,detail=attachment(item,insert,wax_center if item.kind=='WAX' else center,dock,cores,tips,apply=item in enabled and not both,root_radius=root_radius,compact_ocular=ocular)
                detail['seat_owner']=owner.name
                if len(group)>1:detail['wax_feed_center_world']=list(wax_center)
                if substitute.enabled(item):
                    former=substitute.detach(work,insert,dock,frame,item)
                    record(former,item,'FORMER',item in enabled and not both,owner.uid)
                    detail['detachable_former']=True
                record(insert,item,'INSERT',item in enabled and not both,owner.uid)
                reports[item.uid]=json.dumps(detail)
            if len(group)>1:
                combined=work.copy(blank,'Insert_combined_work')
                combined,_=attachment(ocular,combined,center,dock,apply=both,root_radius=root_radius)
                if compact_mode:
                    discard_boolean_dust(combined);compact.clean_seams(combined)
                combined,combined_detail=attachment(wax_item,combined,wax_center,dock,compact_ocular=ocular,has_pedestal=True)
                if compact_mode:
                    combined_detail['seat_owner']=owner.name
                    combined_detail['fill_estimate']=dict(estimate) if not wax_item.cavity_ml else {'method':'MANUAL','fill_ml':wax_item.cavity_ml}
                    reports[wax_item.uid]=json.dumps(combined_detail)
                if substitute.enabled(ocular):
                    former=substitute.detach(work,combined,dock,frame,ocular)
                    record(former,ocular,'FORMER_COMBINED',both,owner.uid,kind='OCULAR')
                record(combined,owner,'COMBINED',both,owner.uid,kind='WAX_OCULAR')
            record(blank,owner,'BLANK',not enabled,owner.uid)
        # Give the second seat its own vent / overflow cup. Keep its original
        # blank as an alternative, or retain its ocular pedestal when assigned.
        for uid,feed in vent_owners.items():
            from . import insert_single_cup
            owner=roots[uid];dock=built_docks[uid];center=Vector(dock['center'])
            options=[o for o in generated if o.get('insert_active') and o.get('insert_seat_uid')==uid and o.get('insert_role') not in {'FORMER','FORMER_COMBINED'}]
            if len(options)!=1:raise RuntimeError('Choose one active piece on the air vent seat')
            original=options[0];vent=work.copy(original,'Separate_vent_work')
            vent,info=insert_single_cup.make(work,feed,vent,center,dock,frame,base,fills[feed.uid],'VENT',center-Vector(built_docks[feed.uid]['center']))
            aw.cut(host,vent,'DIFFERENCE');original['insert_active']=False
            record(vent,owner,'VENT',True,uid,kind='AIR_VENT')
            report=json.loads(reports[uid]);report.update(info);reports[uid]=json.dumps(report)
            if raw_seat_link(feed,'bridge_seat')!=uid:
                feed_parts=[o for o in generated if o.get('insert_active') and o.get('insert_seat_uid')==feed.uid]
                if len(feed_parts)!=1:raise RuntimeError('Choose one active airway piece on the reservoir seat')
                airway_part=feed_parts[0]
                insert_single_cup.clear_pair_passages(work,(airway_part,vent),frame)
                vent_space=work.copy(vent,'Vent_seating_clearance_work')
                vent_space.data=vent['cast_preview_solid'].copy()
                aw.cut(airway_part,vent_space,'DIFFERENCE');work.delete(vent_space)
                discard_boolean_dust(airway_part);compact.clean_seams(airway_part);closed(airway_part,connected=True)
        # A two-seat attachment joins removable pieces behind the base. The
        # separate seat plugs stay available as alternatives, but are hidden.
        claimed=set();roots={g[0].uid:g[0] for g in groups}
        for owner in roots.values():
            uid=owner.bridge_seat
            if uid=='NONE' or owner.mode!='INSERT' or owner.uid in only_members:continue
            other=roots.get(uid)
            if other is None or other.uid==owner.uid:raise RuntimeError('Choose a different second mounting seat')
            if owner.uid in claimed or uid in claimed or other.bridge_seat!='NONE':raise RuntimeError('A seat can belong to only one two-seat insert')
            claimed.update((owner.uid,uid))
            pieces=[]
            for seat in (owner,other):
                options=[o for o in generated if o.get('insert_active') and o.get('insert_seat_uid')==seat.uid and o.get('insert_role') not in {'FORMER','FORMER_COMBINED'}]
                if len(options)!=1:raise RuntimeError('Build one active removable piece for each selected seat')
                pieces.append(options[0])
            combined=connect_seat_parts(work,*pieces,built_docks[owner.uid],built_docks[uid],owner,other,frame)
            compact_mode=True
            record(combined,owner,'TWO_SEAT',True,owner.uid,kind='TWO_SEAT')
            for obj in pieces:obj['insert_active']=False
            for seat in (owner,other):
                detail=json.loads(reports[seat.uid]);detail['connected_seats']=[owner.seat_name or owner.name,other.seat_name or other.name]
                reports[seat.uid]=json.dumps(detail)
        for name,obj in hosts.items():
            if name=='Mold_Base':discard_boolean_dust(obj)
            closed(obj,connected=name not in coatings.values() and amounts.fingerprint(obj)!=amounts.fingerprint(originals[name]))
        installed=[obj for obj in generated if obj['insert_active']]
        substitute_eyes={i.ocular.name for i in entries if substitute.enabled(i) and i.mode=='INSERT' and i.ocular}
        substitute_eyes.update(coatings[n] for n in list(substitute_eyes) if n in coatings)
        for obj in installed:
            if obj.get('compact_passages'):
                passage_trees=[aw.world_tree(part) for part in (obj,hosts['Mold_Base'])]
                for path in json.loads(obj['compact_passages']):
                    for a,b in zip(path,path[1:]):
                        a,b=Vector(a),Vector(b);direction=b-a
                        for tree in passage_trees:
                            if tree.ray_cast(a,direction.normalized(),direction.length)[0] is not None:
                                raise RuntimeError('The assembled base blocks the compact wax or air passage; move the mounting point or adjust the bores')
        for index,obj in enumerate(installed):
            for other in [o for name,o in hosts.items() if name not in substitute_eyes]+installed[:index]:
                if intersection_volume(work,obj,other,separate_touching=bool(obj.get('single_passage_radii')))>.01:
                    raise RuntimeError(f'{obj.name} intersects {other.name}; reposition its marks or adjust its dimensions')
        # No host mutation occurs until every insert and Boolean has passed.
        if seat_changes and not allow_seat_change:raise SeatChangeRequired('\n'.join(seat_changes))
        from . import mold_cast_preview
        mold_cast_preview.clear(scene)
        refs=mu.get_collection(scene,REF)
        for obj in snapshots:
            for col in list(obj.users_collection):col.objects.unlink(obj)
            refs.objects.link(obj);obj['anaplast_construction']=True;obj.hide_render=True;obj.hide_set(True)
        newstate={'hosts':{},'volume':base_amount,'volume_record':st.get('volume_record',scene.get(amounts.KEY,'')),
                  'settings':settings_signature(scene)}
        for name,tmp in hosts.items():
            actual=bpy.data.objects[name];actual.data=tmp.data.copy()
            visibility=st.get('hosts',{}).get(name,{}).get('visibility',[actual.hide_get(),actual.hide_render])
            newstate['hosts'][name]={'reference':originals[name].name,'installed':signature(actual),'original':signature(originals[name]),'visibility':visibility}
            actual.hide_set(True if name in substitute_eyes else visibility[0]);actual.hide_render=True if name in substitute_eyes else visibility[1]
        for obj in list(bpy.data.objects):
            if obj.get('insert_uid') and obj not in generated:mu.delete_object(obj)
        for obj in generated:
            obj.hide_set(not obj['insert_active']);obj.hide_render=not obj['insert_active']
        scene[STATE]=json.dumps(newstate)
        amounts.clear(scene,'Removable inserts installed: the original volume excludes their displacement. Restore the original mold to use its volume report')
        for item in entries:
            item.report=reports[item.uid]
            if item.uid in seat_specs:item.seat_spec=seat_specs[item.uid]
            if item.kind=='OCULAR':
                old_mark=bpy.data.objects.get(f'InsertMark_{item.uid[:6]}_EYE')
                if old_mark and old_mark.get('anaplast_construction'):mu.delete_object(old_mark)
        for parent_uid,holder in holder_entries.items():holder.report=reports[parent_uid] if holder.holder_enabled else ''
        scene.mold_inserts.report=('Seat and matching blank ready. Choose an insert for this seat.' if seat_only_uid or all(i.kind=='SEAT' for i in entries) else ('Compact cups ready: cap down, base up. Air return vents beside the pedestal only; check remote pockets separately.' if any(compact.enabled(i) for i in entries if i.kind=='WAX') else ('Flat shared seat ready: reservoir, pedestal, combined attachment and blanking plug.' if any(len(g)>1 for g in groups) else 'Flat mounting seats and matching inserts ready.')))+' Review fit and hardware before manufacture.'
        work.finish(keep=generated+snapshots)
        from .scene_helpers import organize
        organize(scene,preserve_visibility=True)
        return generated
    except Exception:
        work.finish();raise


def restore(scene):
    explode.collapse(scene);st=state(scene);checked_hosts(scene,st)
    for name,info in st.get('hosts',{}).items():
        ref=bpy.data.objects.get(info['reference'])
        if ref is None:raise RuntimeError('Original mold reference is missing')
    from . import mold_cast_preview
    mold_cast_preview.clear(scene)
    for name,info in st.get('hosts',{}).items():
        ref=bpy.data.objects[info['reference']];bpy.data.objects[name].data=ref.data.copy();mu.delete_object(ref)
        if 'visibility' in info:
            bpy.data.objects[name].hide_set(info['visibility'][0]);bpy.data.objects[name].hide_render=info['visibility'][1]
    for obj in list(bpy.data.objects):
        if obj.get('insert_uid'):mu.delete_object(obj)
    if st.get('volume_record'):scene[amounts.KEY]=st['volume_record']
    elif amounts.KEY in scene:del scene[amounts.KEY]
    if STATE in scene:del scene[STATE]
    for item in scene.mold_inserts.items:item.seat_spec=''
    scene.mold_inserts.report='Original mold and ocular geometry restored; marks and settings retained.'


class ANAPLAST_OT_insert_edit(bpy.types.Operator):
    bl_idname='anaplast.insert_edit';bl_label='Edit mold inserts';bl_options={'REGISTER','UNDO'}
    action: EnumProperty(items=[('ADD','Add',''),('REMOVE','Remove',''),('BUILD','Build / update inserts',''),('RESTORE','Restore original mold',''),('NEW_SEAT','New seat',''),('BUILD_SEAT','Build seat',''),('UNLINK','Disconnect second seat','Keep both seats and their markings; rebuild inserts to separate the pieces'),('BUILD_SEATS','Build all seats','Plan mounting seats together and build their matching blanks')])
    approval_signature: StringProperty(options={'HIDDEN','SKIP_SAVE'})
    approval_message: StringProperty(options={'HIDDEN','SKIP_SAVE'})
    def invoke(self,context,event):
        self.approval_signature='';self.approval_message=''
        result=self.execute(context)
        if self.approval_message:
            return context.window_manager.invoke_props_dialog(self,width=480,confirm_text='Update these seats')
        return result
    def draw(self,context):
        self.layout.label(text='Change existing mounting seats?',icon='QUESTION')
        for line in self.approval_message.splitlines():self.layout.label(text=line)
        self.layout.label(text='Cancel keeps the installed mold unchanged.')
    def execute(self,context):
        s=context.scene;p=s.mold_inserts
        try:
            repair_seat_links(s)
            migrate_holders(s)
            if self.action=='NEW_SEAT':
                item=p.items.add();item.uid=uuid.uuid4().hex;item.kind='SEAT';item.mode='BLANK'
                used={i.seat_name for i in p.items};number=1
                while f'Seat {number}' in used:number+=1
                item.seat_name=f'Seat {number}';item.name=item.seat_name;p.active=len(p.items)-1;p.target_seat=item.uid
            elif self.action=='ADD':
                def add(kind):
                    item=p.items.add();item.uid=uuid.uuid4().hex;item.kind=kind;item.name=dict((a,b) for a,b,_ in KINDS)[kind]
                    if kind=='OCULAR':item.ocular=s.ocular_production.color_obj or s.anaplast.ocular_obj
                    if kind=='WAX':item.reservoir_layout='SIDE_CUPS'
                    if kind=='AIRWAY':item.airway_shape='OUTLINE';item.chambers='SHARED'
                    return item
                if p.new_kind=='HOLDER':
                    selected=active(s);parents=[i for i in p.items if i.kind=='OCULAR']
                    parent=selected if selected and selected.kind=='OCULAR' else (parents[0] if len(parents)==1 else None)
                    existing=next((i for i in p.items if i.kind=='HOLDER' and parent and i.pedestal_uid==parent.uid),None)
                    if existing:p.active=next(n for n,i in enumerate(p.items) if i.uid==existing.uid)
                    else:
                        holder=add('HOLDER')
                        if parent:holder.pedestal_uid=parent.uid
                        p.active=len(p.items)-1
                else:assign_insert(s)
            elif self.action=='UNLINK':
                connection=seat_connection(s,selected_seat(s))
                if connection:connection.bridge_seat='NONE'
                seat=selected_seat(s)
                for item in p.items:
                    if seat and (item.uid==seat.uid or raw_seat_link(item,'flow_seat')==seat.uid):item.flow_seat='NONE'
                p.use_two_seats=False;p.second_seat='NONE'
                p.report='Seats disconnected in settings. Build / update all inserts to apply; both seats and their marks are retained.'
            elif self.action=='REMOVE':
                if state(s):raise RuntimeError('Restore the original mold before removing an insert; use its blanking plug to keep the seat')
                item=active(s)
                if item:
                    if any(i.seat_mode=='SHARED' and i.seat_owner==item.uid for i in p.items):
                        raise RuntimeError('Choose another seat for the sharing attachment before removing this entry')
                    if any(i.kind=='HOLDER' and i.pedestal_uid==item.uid for i in p.items):
                        raise RuntimeError('Remove or relink its ocular space holder before removing this pedestal')
                    if seat_connection(s,item):raise RuntimeError('Press Disconnect second seat before removing this seat')
                    if any(raw_seat_link(i,'flow_seat')==item.uid for i in p.items):raise RuntimeError('Disconnect the wax / air seat pair before removing this seat')
                    prefix=f'InsertMark_{item.uid[:6]}_'
                    for obj in list(bpy.data.objects):
                        if obj.name.startswith(prefix) or obj.get('airway_preview')==item.uid:mu.delete_object(obj)
                    p.items.remove(p.active);p.active=max(0,p.active-1)
                    repair_seat_links(s)
            elif self.action=='RESTORE':restore(s)
            else:
                seat=selected_seat(s)
                if self.action=='BUILD_SEAT' and seat is None:raise RuntimeError('Add or select a mounting seat first')
                if self.action=='BUILD' and all(i.kind=='SEAT' for i in p.items):
                    raise RuntimeError('Only mounting seats are set up. Choose Nasal airway core (or another insert), then press Set insert for selected seat(s) to show its controls.')
                allow=bool(self.approval_signature) and self.approval_signature==settings_signature(s)
                self.approval_message=''
                build(context,seat_only_uid='ALL' if self.action=='BUILD_SEATS' else seat.uid if self.action=='BUILD_SEAT' else None,allow_seat_change=allow)
                self.approval_signature=''
        except SeatChangeRequired as exc:
            self.approval_message=str(exc);self.approval_signature=settings_signature(s)
            self.report({'WARNING'},'Existing seat needs updating; confirmation is required')
            return {'CANCELLED'}
        except Exception as exc:
            from .report import rep
            rep(self,{'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class ANAPLAST_OT_insert_mark(bpy.types.Operator):
    bl_idname='anaplast.insert_mark';bl_label='Mark insert position';bl_options={'REGISTER','UNDO'}
    mark: EnumProperty(items=MARKS)
    @classmethod
    def poll(cls,context):return context.area and context.area.type=='VIEW_3D' and active(context.scene) is not None
    def invoke(self,context,event):
        item=active(context.scene);explode.collapse(context.scene)
        if self.mark=='BASE1' and item.kind!='AIRWAY':item=selected_seat(context.scene)
        if self.mark=='EYE':
            self.report({'INFO'},'The pedestal key uses the ocular center automatically; mark only the base mounting point')
            return {'FINISHED'}
        if self.mark=='BASE1' and item.seat_mode=='SHARED':
            self.report({'INFO'},'This attachment uses the existing seat\'s base mark');return {'FINISHED'}
        if self.mark=='BASE1' and item.seat_spec and item.kind!='AIRWAY':
            self.report({'ERROR'},'This seat is built. Restore the mold before moving its mounting point');return {'CANCELLED'}
        self.uid=item.uid
        self.window_region=next((region for region in context.area.regions if region.type=='WINDOW'),None)
        self.view=context.space_data.region_3d
        if self.window_region is None or self.view is None:return {'CANCELLED'}
        self.target=(bpy.data.objects.get('Mold_Base') if self.mark.startswith('BASE') else item.ocular if self.mark=='EYE' else aw.reference(context.scene.anaplast))
        if self.target is None:self.report({'ERROR'},'Choose or build the target surface first');return {'CANCELLED'}
        try:self.target=marking_target(context.scene,self.target)
        except RuntimeError as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        self.visibility={o.name:o.hide_get() for o in context.view_layer.objects if o.type=='MESH'}
        for name in self.visibility:bpy.data.objects[name].hide_set(bpy.data.objects[name]!=self.target)
        self.target.hide_set(False);self.tree=aw.world_tree(self.target)
        context.area.header_text_set(f'Click {dict((a,b) for a,b,_ in MARKS)[self.mark]} on {self.target.name}; Esc cancels')
        context.window.cursor_modal_set('CROSSHAIR');context.window_manager.modal_handler_add(self);return {'RUNNING_MODAL'}
    def finish(self,context):
        for name,value in self.visibility.items():
            obj=bpy.data.objects.get(name)
            if obj is not None:obj.hide_set(value)
        context.area.header_text_set(None);context.window.cursor_modal_restore()
    def modal(self,context,event):
        if event.type in {'ESC','RIGHTMOUSE'}:self.finish(context);return {'CANCELLED'}
        if event.type=='LEFTMOUSE' and event.value=='PRESS':
            from bpy_extras import view3d_utils as vu
            region=self.window_region;xy=(event.mouse_x-region.x,event.mouse_y-region.y)
            if not (0<=xy[0]<region.width and 0<=xy[1]<region.height):return {'RUNNING_MODAL'}
            origin=vu.region_2d_to_origin_3d(region,self.view,xy);direction=vu.region_2d_to_vector_3d(region,self.view,xy)
            hit,normal,_,_=self.tree.ray_cast(origin,direction)
            if hit is None:return {'RUNNING_MODAL'}
            item=next((i for i in context.scene.mold_inserts.items if i.uid==self.uid),None)
            if item is None:self.finish(context);return {'CANCELLED'}
            base=bpy.data.objects.get('Mold_Base');up=aw.frame_for(base)[0]
            if self.mark.startswith('BASE') and normal.dot(up)<.15:
                self.report({'WARNING'},'Mark the fitting side of the base, not its back or outside edge');return {'RUNNING_MODAL'}
            setattr(item,self.mark.lower(),hit);marks=set(json.loads(item.marks));marks.add(self.mark);item.marks=json.dumps(sorted(marks))
            if self.mark=='BASE1' and item.kind!='AIRWAY' and not item.seat_spec:
                item.report=''  # A deliberately moved unbuilt seat replaces its old placement plan.
            sources=json.loads(item.mark_sources);sources[self.mark]=marking_source(context.scene,self.target);item.mark_sources=json.dumps(sources)
            name=f'InsertMark_{item.uid[:6]}_{self.mark}';obj=bpy.data.objects.get(name)
            if obj is None:obj=bpy.data.objects.new(name,None);mu.get_collection(context.scene,COL).objects.link(obj)
            obj.location=hit;obj.empty_display_type='CIRCLE';obj.rotation_euler=up.to_track_quat('Z','Y').to_euler()
            obj.empty_display_size=(item.nostril_diameter*.5 if self.mark.startswith('TIP') else item.patient_diameter*.5 if item.kind=='AIRWAY' else item.key_size*.5 if self.mark=='EYE' else item.dock_diameter*.5)
            obj.show_in_front=True;obj.hide_render=True
            obj['anaplast_construction']=True;self.finish(context);return {'FINISHED'}
        return {'PASS_THROUGH'}


class ANAPLAST_UL_inserts(bpy.types.UIList):
    def draw_item(self,context,layout,data,item,icon,active_data,active_propname,index):
        inactive=not item.holder_enabled if item.kind=='HOLDER' else item.mode=='BLANK'
        layout.prop(item,'name',text='',emboss=False);layout.label(text='Seat / blank' if item.kind=='SEAT' else 'Inactive' if inactive else 'Insert')


class ANAPLAST_PT_inserts(bpy.types.Panel):
    bl_label='Interchangeable inserts';bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='Anaplast';bl_parent_id='ANAPLAST_PT_mold';bl_options={'DEFAULT_CLOSED'}
    def draw(self,context):
        p=context.scene.mold_inserts;l=self.layout
        st=state(context.scene)
        if st and st.get('settings')!=settings_signature(context.scene):l.label(text='Settings changed: update inserts',icon='ERROR')
        l.label(text='1 · Build mounting seats')
        l.operator('anaplast.insert_edit',text='Add mounting seat',icon='ADD').action='NEW_SEAT'
        l.template_list('ANAPLAST_UL_inserts','',p,'items',p,'active',rows=3)
        item=active(context.scene)
        seat=selected_seat(context.scene)
        if seat:
            box=l.box();box.prop(seat,'seat_name',text='Seat name')
            if not seat.seat_spec:
                box.operator('anaplast.insert_mark',text='Mark seat on Mold Base').mark='BASE1'
            else:box.label(text='Built seat position retained',icon='CHECKMARK')
            box.prop(seat,'dock_diameter',text='Minimum width (mm)');box.prop(seat,'clearance')
            if not seat.seat_spec:box.prop(seat,'seat_auto_orient')
            box.prop(seat,'seat_angle')
            if seat.seat_spec:box.label(text='Angle changes need a seat rebuild')
            box.prop(seat,'screw');box.prop(seat,'pilot_depth')
            box.operator('anaplast.insert_edit',text='Build seat + matching blank').action='BUILD_SEAT'
        if sum(i.kind!='HOLDER' and i.seat_mode=='OWN' for i in p.items)>1:
            l.label(text='Mark each seat above, then build together')
            l.operator('anaplast.insert_edit',text='Build all seats + blanks',icon='MOD_ARRAY').action='BUILD_SEATS'
        l.label(text='2 · Choose insert and seat(s)')
        l.prop(p,'new_kind',text='Insert')
        if p.new_kind!='HOLDER':
            l.prop(p,'seat_count',expand=True)
            split_flow=p.use_two_seats and p.new_kind=='AIRWAY'
            l.prop(p,'target_seat',text='Reservoir seat' if split_flow else 'Seat')
            if p.use_two_seats:
                if sum(i.kind!='HOLDER' and i.seat_mode=='OWN' for i in p.items)<2:
                    l.label(text='Add and build a second seat first',icon='INFO')
                l.prop(p,'second_seat',text='Ocular seat' if p.new_kind=='COMBINED' else 'Vent seat' if split_flow else 'Second seat')
                if p.new_kind!='BLANK':l.prop(p,'two_seat_layout')
        l.operator('anaplast.insert_edit',text='Use blank' if p.new_kind=='BLANK' else 'Set insert for selected seat(s)' if p.new_kind!='HOLDER' else 'Add ocular space holder').action='ADD'
        if item and item.kind=='SEAT':l.label(text='Set the insert above to show its controls',icon='INFO')
        if item:
            if item.kind in {'WAX','OCULAR'}:
                l.prop(item,'seat_mode',expand=True)
                if item.seat_mode=='SHARED':l.prop(item,'seat_owner')
            sharing=len(seat_members(item))==2 and {i.kind for i in seat_members(item)}=={'WAX','OCULAR'}
            if item.kind=='HOLDER':l.prop(item,'holder_enabled')
            elif sharing:l.prop(item,'seat_piece')
            elif item.kind!='SEAT':l.prop(item,'mode',expand=True)
            if seat and item.kind!='HOLDER':
                connection=seat_connection(context.scene,seat)
                if connection is None:connection=next((i for i in p.items if raw_seat_link(i,'flow_seat')!='NONE' and (i.uid==seat.uid or raw_seat_link(i,'flow_seat')==seat.uid)),None)
                if connection:
                    uid=raw_seat_link(connection,'bridge_seat')
                    if uid=='NONE':uid=raw_seat_link(connection,'flow_seat')
                    other=next((i for i in p.items if i.uid==uid),None)
                    l.label(text='Connected: '+(connection.seat_name or connection.name)+' + '+(other.seat_name or other.name),icon='LINKED')
                    if connection.kind=='AIRWAY' and connection.airway_reservoir:
                        l.label(text='Reservoir: '+(connection.seat_name or connection.name))
                        l.label(text='Air vent: '+(other.seat_name or other.name))
                    l.operator('anaplast.insert_edit',text='Disconnect second seat',icon='UNLINKED').action='UNLINK'
                else:l.label(text='Mounting: '+(seat.seat_name or seat.name)+' only',icon='INFO')
            marks=set(json.loads(item.marks))
            def button(mark,label):
                l.operator('anaplast.insert_mark',text=label,icon='CHECKMARK' if mark in marks else 'RESTRICT_SELECT_OFF').mark=mark
            if item.kind not in {'HOLDER','AIRWAY','SEAT'}:
                if item.seat_mode=='OWN' and not item.seat_spec:button('BASE1','Mark mounting / feed point on base')
                else:l.label(text='Uses the existing seat\'s base mark')
            if item.kind=='HOLDER':
                l.prop(item,'pedestal_uid')
                parent=next((i for i in p.items if i.kind=='OCULAR' and i.uid==item.pedestal_uid),None)
                if parent:
                    l.label(text='Ocular: '+(parent.ocular.name if parent.ocular else 'Choose on pedestal'))
                    l.label(text='Uses the pedestal key and mounting position')
                else:l.label(text='Add or choose an ocular pedestal first',icon='INFO')
                l.prop(item,'gaze_horizontal');l.prop(item,'gaze_vertical');l.prop(item,'gaze_clearance')
                l.prop(item,'wax_pillars')
                if item.wax_pillars:l.prop(item,'wax_pillar_width')
                l.label(text='Visible opening protected; clearance behind')
                l.label(text='Seats directly against pedestal')
            elif item.kind=='AIRWAY':
                l.prop(item,'chambers',expand=True)
                l.label(text='Two nostrils → one patient opening' if item.chambers=='SHARED' else 'Two nostrils → two patient openings')
                if item.seat_spec:l.label(text='Mounting seat is separate from airway outline')
                l.prop(item,'airway_shape',expand=True)
                if item.airway_shape=='OUTLINE':
                    l.prop(item,'airway_path')
                    if item.airway_path=='CURVED':
                        l.prop(item,'airway_wall');l.prop(item,'airway_reservoir')
                        l.label(text='Test build: inspect wall and removal',icon='INFO')
                        if item.airway_reservoir:
                            draw_reservoir_volume(l,context.scene,item);l.prop(item,'allowance')
                            l.prop(item,'feed_diameter');l.prop(item,'vent_diameter')
                            l.prop(item,'cup_height');l.prop(item,'overflow_ml')
                            l.label(text='Cap on table; base above')
                    saved=json.loads(item.airway_outlines)
                    missing=missing_marks(item)
                    l.label(text=f'Opening outlines: {len(required_marks(item))-len(missing)} / {len(required_marks(item))}',icon='INFO' if missing else 'CHECKMARK')
                    for mark,label in [('TIP1','Nostril 1 on Sculpt'),('TIP2','Nostril 2 on Sculpt'),('BASE1','Patient opening on Mold Base' if item.chambers=='SHARED' else 'Patient opening 1 on Mold Base')]+([('BASE2','Patient opening 2 on Mold Base')] if item.chambers=='SEPARATE' else []):
                        box=l.box();box.label(text=label,icon='CHECKMARK' if mark in saved else 'MESH_CIRCLE')
                        row=box.row(align=True)
                        for action,label in [('LASSO','Smooth lasso'),('PAINT','Paint / erase')]:
                            op=row.operator('anaplast.airway_area',text=label);op.action=action;op.mark=mark
                    l.prop(item,'airway_smoothing');l.prop(item,'airway_brush')
                    row=l.row(align=True);row.operator('anaplast.airway_area',text='Preview airway').action='PREVIEW';row.operator('anaplast.airway_area',text='Hide preview').action='HIDE'
                    l.label(text='Enter: accept outline; Shift: erase paint')
                else:
                    button('BASE1','Mark patient opening 1')
                    if item.chambers=='SEPARATE':button('BASE2','Mark patient opening 2')
                    button('TIP1','Mark nostril 1 on Sculpt');button('TIP2','Mark nostril 2 on Sculpt')
                    l.prop(item,'patient_diameter');l.prop(item,'nostril_diameter')
                if item.airway_shape!='OUTLINE' or item.airway_path=='STRAIGHT':
                    l.prop(item,'draft');l.label(text='Withdrawal follows the mold opening axis')
                else:l.label(text='Upright case: Z up; undercuts allowed')
            elif item.kind=='OCULAR':
                l.prop(item,'ocular')
                l.label(text='Key centered automatically on ocular back');l.prop(item,'key',expand=True)
                l.prop(item,'key_size');l.prop(item,'key_depth');l.prop(item,'key_clearance')
                l.prop(item,'stem')
            elif item.kind=='WAX':
                l.prop(item,'wax_mode',expand=True);l.prop(item,'allowance');draw_reservoir_volume(l,context.scene,item)
                l.label(text='Fill volume 0 = recorded mold volume')
                l.prop(item,'feed_diameter');l.prop(item,'cup_height')
                if item.wax_mode=='INJECTION':l.prop(item,'nozzle_diameter')
                else:
                    l.prop(item,'reservoir_layout')
                    if item.reservoir_layout=='SIDE_CUPS':
                        l.prop(item,'vent_diameter');l.prop(item,'overflow_ml')
                        l.label(text='Cap on table; base above')
                        l.label(text='Air return beside pedestal only')
                l.label(text='Allowance is not measured wax shrinkage')
            l.label(text='3 · Build inserts')
            row=l.row();row.enabled=any(i.kind!='SEAT' for i in p.items)
            row.operator('anaplast.insert_edit',text='Build / update all inserts',icon='PLAY').action='BUILD'
            preview=l.box();preview.label(text='Wax / prosthesis preview')
            preview.operator('anaplast.cast_preview',text='Create / update previews').action='BUILD'
            row=preview.row(align=True);row.operator('anaplast.cast_preview',text='Wax').action='WAX';row.operator('anaplast.cast_preview',text='Prosthesis + ocular').action='OCULAR'
            if any(o.get('cast_preview_detached') for o in context.scene.objects):
                preview.label(text='Detached regions: inspect margins',icon='ERROR')
                preview.operator('anaplast.cast_preview',text='Show detached regions too').action='ALL'
            preview.operator('anaplast.cast_preview',text='Return to mold').action='MOLD'
            preview.label(text='From mold cavity; excludes sprues / shrinkage')
            spacing=max((float(o.get('cast_preview_spacing_mm',0)) for o in context.scene.objects if o.get('cast_preview')),default=0.)
            if spacing:preview.label(text=f'Preview sampling: {spacing:.2f} mm')
            row=l.row(align=True);row.operator('anaplast.insert_edit',text='Restore original mold').action='RESTORE';row.operator('anaplast.insert_edit',text='Remove entry').action='REMOVE'
            if item.report:
                info=json.loads(item.report)
                for check in info.get('airway_checks',[]):
                    gap=check['sampled_wall_mm'];target=check['target_wall_mm']
                    l.label(text=f'Sampled wall: {gap:.2f} mm / target {target:.1f}',icon='ERROR' if gap<target else 'INFO')
                    l.label(text='Excludes nostril opening transition')
                    l.label(text='Removal path not validated')
                if 'reservoir_capacity_ml' in info:l.label(text=f"Feed capacity: {info['reservoir_capacity_ml']:.2f} mL")
                if info.get('fill_estimate'):
                    estimate=info['fill_estimate']
                    l.label(text=f"Built fill estimate: {estimate['fill_ml']:.2f} mL")
                    if estimate.get('method')=='SCULPT_BOUNDING_BOX':l.label(text='Open Sculpt: generous box estimate')
                    elif estimate.get('method')=='NOMINAL_FALLBACK':l.label(text='No solid Sculpt: nominal estimate used')
                if 'overflow_capacity_ml' in info:l.label(text=f"Overflow capacity: {info['overflow_capacity_ml']:.2f} mL")
                if info.get('mount_resampled_mm',0):
                    l.label(text=f"Mount cleanup: {info['mount_sampled_deviation_mm']:.3f} mm sampled change")
                if item.kind=='HOLDER' and info.get('ocular_use')=='SUBSTITUTE':
                    access=info['rear_access_mm']
                    l.label(text=f'Rear access: {access[0]:.1f} x {access[1]:.1f} mm')
                    l.label(text='Front opening preserved')
                    l.label(text='Opening may limit the requested gaze range')
                footprint=info.get('mount',{}).get('footprint_mm')
                if item.kind!='HOLDER':
                    if footprint:l.label(text=f"Built footprint: {footprint[0]:.1f} x {footprint[1]:.1f} mm")
                    elif 'seat_width_mm' in info.get('mount',{}):l.label(text=f"Built socket size: {info['mount']['seat_width_mm']:.2f} mm")
                    if info.get('mount',{}).get('extra_backing_mm',0)>.001:l.label(text=f"Extra backing: {info['mount']['extra_backing_mm']:.2f} mm")
        if p.report:
            import textwrap
            for line in textwrap.wrap(p.report,max(25,int(context.region.width/7)-6)):l.label(text=line)


CLASSES=(InsertItem,InsertSettings,ANAPLAST_OT_insert_edit,ANAPLAST_OT_insert_mark,ANAPLAST_UL_inserts,ANAPLAST_PT_inserts)
def register():
    # Panel registration follows its parent in ui.register().
    for cls in CLASSES[:-1]:bpy.utils.register_class(cls)
    bpy.types.Scene.mold_inserts=PointerProperty(type=InsertSettings)
    from . import airway_marking
    airway_marking.register()
    bpy.app.handlers.load_post.append(load_holders)
    # Extension registration runs with restricted bpy.data. Migrate the open
    # case only after registration; saved-file loads use the persistent handler.
    bpy.app.timers.register(migrate_open_cases,first_interval=.1)
def register_panel():bpy.utils.register_class(ANAPLAST_PT_inserts)
def unregister_panel():bpy.utils.unregister_class(ANAPLAST_PT_inserts)
def unregister():
    from . import airway_marking
    airway_marking.unregister()
    if bpy.app.timers.is_registered(migrate_open_cases):bpy.app.timers.unregister(migrate_open_cases)
    if load_holders in bpy.app.handlers.load_post:bpy.app.handlers.load_post.remove(load_holders)
    del bpy.types.Scene.mold_inserts
    for cls in reversed(CLASSES[:-1]):bpy.utils.unregister_class(cls)
