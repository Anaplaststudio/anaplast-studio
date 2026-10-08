"""Workflow labels without renaming objects referenced by existing saved cases."""
def surface_label(obj,p):
    name=obj.name;role=obj.get('anaplast_part','')
    if name in {'Mold_Base','Mold_Wedge','Mold_Cap'}:label=name.replace('_',' ')
    elif obj==p.face_scan_obj:label='Scan'
    elif role=='MIRROR' or name.startswith('Mirror_'):label='Mirror scan'
    elif name.startswith('Prototype_') or role=='PROTOTYPE':label='Prototype'
    elif obj==p.source_prosthesis or obj==p.prosthesis_obj or name.startswith('Sculpt'):label='Sculpt'
    else:return name
    return label if name==label else f'{label} — {name}'
