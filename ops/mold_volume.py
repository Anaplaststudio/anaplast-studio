"""Measure silicone voids at their construction cuts, before shell hollowing.

Incremental removed volume counts intersecting ring/outlet cuts once and excludes
keys, pry points, bolts and external shell backs. Values describe the built mold,
not the currently selected Sculpt or a bounding box.
"""
import hashlib
import json
import math
import bpy
import bmesh
import numpy as np
from .. import silicones

KEY = 'anaplast_mold_volume'
# Factor II VST-50 SDS (2021-07-23), sections 9: A=1.12, B=1.025 g/mL.
# VST family TDS: A:B=10:1 by mass. Assumes additive uncured volumes;
# this is an inferred mixed density, not a published cured-material density.
VST50_DENSITY = silicones.VST50_DENSITY
VST50_SDS = silicones.VST50_SDS
VST50_TDS = silicones.VST50_TDS


def weight_estimate(scene, data):
    props = scene.anaplast
    result = silicones.material(props)
    density = result['density_g_ml']
    for name in ('cavity', 'channels', 'total'):
        result[name+'_g'] = data[name+'_ml'] * density if density is not None else None
    return result


def volume(obj):
    """Closed, consistently wound mesh volume in world Blender units cubed."""
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        if not bm.faces:
            return 0.
        if any(not e.is_manifold or not e.is_contiguous for e in bm.edges):
            raise ValueError(f'{obj.name}: open or inconsistently joined surface')
        if any(not v.link_faces for v in bm.verts):
            raise ValueError(f'{obj.name}: loose geometry')
        value = abs(bm.calc_volume(signed=True)) * abs(obj.matrix_world.to_3x3().determinant())
        if not math.isfinite(value):
            raise ValueError(f'{obj.name}: invalid volume')
        return value
    finally:
        bm.free()


def fingerprint(obj):
    coords = np.empty(len(obj.data.vertices)*3, np.float32); obj.data.vertices.foreach_get('co', coords)
    loops = np.empty(len(obj.data.loops), np.int32); obj.data.loops.foreach_get('vertex_index', loops)
    return {'mesh': hashlib.sha256(coords.tobytes()+loops.tobytes()).hexdigest(),
            'scale': abs(obj.matrix_world.to_3x3().determinant())}


class Ledger:
    def __init__(self, data=None):
        self.data = json.loads(json.dumps(data)) if data else {'cavity': 0., 'channels': 0., 'errors': [], 'method': 'CONSTRUCTION_REMOVED_VOLUME'}

    def before(self, obj):
        try:
            return volume(obj)
        except Exception as exc:
            self.data['errors'].append(str(exc)); return None

    def removed(self, obj, before, category):
        if before is None:
            return
        after = self.before(obj)
        if after is None:
            return
        delta = before-after
        if delta < -max(1e-5, before*1e-6):
            self.data['errors'].append(f'{obj.name}: a subtraction increased the volume')
        else:
            self.data[category] += max(0., delta)

    def cut(self, obj, cutter, operation, cut_function, category):
        before = self.before(obj)
        result = cut_function(obj, cutter, operation)
        self.removed(obj, before, category)
        return result


def clear(scene, reason='Build the complete mold to calculate its silicone volume'):
    scene[KEY] = json.dumps({'available': False, 'reason': reason})


def publish(scene, ledger, parts):
    data = json.loads(json.dumps(ledger.data))
    if data['errors']:
        clear(scene, 'Volume unavailable: ' + data['errors'][0]); return
    # Final topology must also be valid, even though external backing is excluded.
    for part in parts:
        if ledger.before(part) is None:
            clear(scene, 'Volume unavailable: ' + ledger.data['errors'][-1]); return
    data.update(available=True, parts={o.name: fingerprint(o) for o in parts},
                total=data['cavity']+data['channels'], unit_scale=float(scene.unit_settings.scale_length))
    scene[KEY] = json.dumps(data)


def read(scene, validate=False):
    data = json.loads(scene.get(KEY, '{}'))
    if not data.get('available'):
        raise ValueError(data.get('reason', 'Rebuild this mold once to record separate cavity and channel volumes'))
    if validate:
        current = {name for name in ('Mold_Base','Mold_Wedge','Mold_Cap') if bpy.data.objects.get(name) is not None}
        if current != set(data['parts']):
            raise ValueError('The mold parts changed. Rebuild the complete mold to recalculate volume')
        for name, expected in data['parts'].items():
            obj = bpy.data.objects.get(name)
            if obj is None or obj.modifiers:
                raise ValueError('The mold has changed. Rebuild it to recalculate volume')
            actual = fingerprint(obj)
            if actual['mesh'] != expected['mesh'] or not math.isclose(actual['scale'], expected['scale'], rel_tol=1e-5, abs_tol=1e-8):
                raise ValueError('The mold geometry changed. Rebuild it to recalculate volume')
    # Scene scale converts Blender units to metres; one cubic metre = 1e6 mL.
    factor = float(scene.unit_settings.scale_length)**3 * 1e6
    data['cavity_ml'] = data['cavity']*factor
    data['channels_ml'] = data['channels']*factor
    data['total_ml'] = data['total']*factor
    return data


def draw(layout, scene):
    box = layout.box(); box.label(text='Silicone amount (at build)', icon='DRIVER_DISTANCE')
    box.prop(scene.anaplast, 'mold_silicone')
    if scene.anaplast.mold_silicone == 'CUSTOM':
        box.prop(scene.anaplast, 'mold_silicone_density')
    elif silicones.CATALOGUE[scene.anaplast.mold_silicone]['density'] is None:
        box.label(text='No verified preset density')
        box.prop(scene.anaplast, 'mold_product_density')
    material=silicones.material(scene.anaplast)
    if scene.anaplast.mold_silicone=='A106':box.label(text='Foam: use effective expanded density')
    if material['density_g_ml'] is not None:
        box.label(text=f"Estimate uses {material['density_g_ml']:.3f} g/mL")
    try:
        data = read(scene)
        mass = weight_estimate(scene, data)
        box.label(text='Volume / estimated weight')
        for key,label in [('cavity','Sculpt cavity'),('channels','Ring and channels'),('total','Total')]:
            weight=f" / {mass[key+'_g']:.2f} g" if mass[key+'_g'] is not None else ''
            box.label(text=f"{label}: {data[key+'_ml']:.2f} mL{weight}")
        if mass['total_g'] is None:box.label(text='Enter density to calculate weight')
        box.label(text='Before inserts, additives and mixing waste')
        box.label(text='Excludes shell backs, bolts and pry points')
        box.operator('anaplast.mold_volume', text='Verify saved volume')
    except (ValueError, TypeError) as exc:
        import textwrap
        for line in textwrap.wrap(str(exc),48):box.label(text=line)
        box.label(text='No saved measurement to verify',icon='INFO')


def before_channel_edit(scene, obj):
    try:
        data = read(scene, validate=True)
    except ValueError as exc:
        clear(scene, str(exc)); return None, None
    ledger = Ledger(data)
    return ledger, ledger.before(obj)


def after_channel_edit(scene, obj, ledger, before):
    if ledger is None:
        return
    ledger.removed(obj, before, 'channels')
    publish(scene, ledger, [bpy.data.objects[name] for name in ledger.data['parts']])


class ANAPLAST_OT_mold_volume(bpy.types.Operator):
    """Check the recorded cavity and channel volumes against the current mold geometry"""
    bl_idname = 'anaplast.mold_volume'
    bl_label = 'Verify saved mold volume'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            data = read(context.scene, validate=True)
        except ValueError as exc:
            clear(context.scene, str(exc)); self.report({'WARNING'}, str(exc)); return {'CANCELLED'}
        mass = weight_estimate(context.scene, data)
        weight=f" / estimated {mass['total_g']:.2f} g {mass['material']}" if mass['total_g'] is not None else '; enter density for weight'
        self.report({'INFO'}, f"Sculpt cavity {data['cavity_ml']:.2f} mL; ring and channels {data['channels_ml']:.2f} mL; total {data['total_ml']:.2f} mL"+weight)
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ANAPLAST_OT_mold_volume)


def unregister():
    bpy.utils.unregister_class(ANAPLAST_OT_mold_volume)
