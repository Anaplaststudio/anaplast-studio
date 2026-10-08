"""One feed or overflow passage per seat, for paired mounting inserts."""
import math,json
import numpy as np
from mathutils import Vector


def make(work,item,insert,center,dock,frame,base,fill_ml,role,away):
    from . import mold_inserts as mi,mold_insert_reservoir as compact
    u,r,f=frame;direction=away-u*away.dot(u)
    if direction.length<1e-6:direction=r.copy()
    direction.normalize()
    flow=(u,direction,u.cross(direction))
    bore=(item.feed_diameter if role=='FEED' else item.vent_diameter)*.5
    wall=1.5;root_radius=bore+wall
    xy=center-u*center.dot(u);rear=dock['rear']-2.;neck=3.;height=item.cup_height
    capacity=fill_ml*item.allowance/100 if role=='FEED' else item.overflow_ml
    top=max(bore+1.,(-bore+math.sqrt(max(0.,12*capacity*1000/(math.pi*height)-3*bore*bore)))*.5)
    co=np.array([insert.matrix_world@v.co for v in insert.data.vertices])
    exit_distance=max(root_radius+2.,float(((co-np.array(xy))@np.array(direction)).max())+1.)
    tree=mi.aw.world_tree(base);low,high=mi.bounds(base,u)
    levels=[mi.column(tree,xy+direction*float(t),frame,low,high)[1] for t in np.linspace(0,exit_distance,32)]
    outlet=max(levels)+bore+.3
    boss=mi.cylinder(work,'Single_cup_root_work',xy+u*(dock['rear']+.1),xy+u*(outlet+bore+wall),root_radius,flow)
    insert=mi.merge(work,insert,boss)
    mouth=xy+direction*(top+wall+1)+u*(rear-neck-height)
    throat=xy+u*(rear-neck)
    body=mi.loft(work,'Wax_feed_cup' if role=='FEED' else 'Air_overflow_cup',mi.round_rings([mouth,throat,xy+u*(rear+.5)],[top+wall,bore+wall,bore+wall],flow))
    insert=mi.merge(work,insert,body)
    exterior=insert.data.copy()
    tool=mi.loft(work,'Single_cup_bore_work',mi.round_rings([mouth-u,mouth,throat,xy+u*(rear+.7)],[top,top,bore,bore],flow))
    path=[xy+u*(rear-neck-.2),xy+u*outlet,xy+direction*exit_distance+u*outlet]
    tool=mi.merge(work,tool,compact.tube(work,'Single_cup_outlet_work',path,bore,flow))
    mi.aw.cut(insert,tool,'DIFFERENCE');work.delete(tool)
    mi.discard_boolean_dust(insert);compact.clean_seams(insert);mi.closed(insert,connected=True)
    insert['cast_preview_solid']=exterior
    insert['compact_passages']=json.dumps([[list(p) for p in path]])
    insert['flow_role']=role
    insert['single_passage_radii']=json.dumps([bore])
    info={'layout':'ONE_PASSAGE_PER_SEAT','flow_role':role,'fill_volume_ml':fill_ml,
          'allowance_percent':item.allowance,'reserve_target_ml':capacity if role=='FEED' else 0.,
          'reservoir_capacity_ml' if role=='FEED' else 'overflow_capacity_ml':math.pi*height*(bore*bore+bore*top+top*top)/3000}
    return insert,info


def clear_pair_passages(work,parts,frame):
    """Open a covered vent through the core, into the wax cavity above it."""
    from . import mold_inserts as mi,mold_insert_reservoir as compact
    u=frame[0]
    co=np.array([part.matrix_world@v.co for part in parts for v in part.data.vertices])
    top=float((co@np.array(u)).max())+1.
    for source in parts:
        paths=json.loads(source.get('compact_passages','[]'))
        sizes=json.loads(source.get('single_passage_radii','[]'))
        for n,bore in enumerate(sizes):
            points=[Vector(p) for p in paths[n]];a,b=points[:2];d=b-a
            others=[target for target in parts if target!=source]
            covered=any(mi.aw.world_tree(target).ray_cast(a,d.normalized(),d.length)[0] is not None for target in others)
            if not covered:continue
            # The core covers the side outlet. Vent straight through its local
            # front instead of extending a lateral tunnel into patient skin.
            xy=a-u*a.dot(u);origin=xy+u*top
            hits=[mi.aw.world_tree(target).ray_cast(origin,-u) for target in parts]
            surface=max((hit[0] for hit in hits if hit[0] is not None),key=lambda p:p.dot(u))
            end=surface+u*.25
            for target in parts:
                start=b-u*.2 if target==source else a
                tool=compact.tube(work,'Airway_flow_clearance_work',[start,end],bore+.0001,frame)
                mi.aw.cut(target,tool,'DIFFERENCE');work.delete(tool)
                mi.discard_boolean_dust(target);compact.clean_seams(target);mi.closed(target,connected=True)
            paths[n]=[list(a),list(end)]
        if sizes:source['compact_passages']=json.dumps(paths)
