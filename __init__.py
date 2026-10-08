# Anaplast Studio â€” Blender extension
# Spec 01 (Mold Toolkit) v0.2. Numeric defaults are placeholders pending
# bench tests B1â€“B3; none are taken from prior in-house workflows.

from . import props, ui, release_info
from .ops import ocular_view, scene_helpers, mold_volume, mold_inserts, mold_workflow, wedge_curve, mold_cast_preview
from .ops import substructure, ocular, mold_auricular, wedge_marking, component_paint, wedge_coverage, nose, sculpt_surface, ocular_production, ocular_marking, ocular_steps, ocular_photos, iris_button, eye_photo, eye_design
from .ops import (case_setup, import_slot, orient, align, explode, crop, margin, texture, ssm, guide, bar,
                  shell, mold_block, mold_split, mold_keys, export_check, report, snapshots, xyz, xyz_live, combined, view, section, sidebyside, overlay, brushes, measure, multires, sphere, photo, mold_auto, lasso, spout, contact, cap_from)

_modules = (release_info, props, ocular_view, mold_workflow, wedge_curve, scene_helpers, mold_volume, mold_inserts, mold_cast_preview, substructure, ocular, mold_auricular, wedge_marking, component_paint, wedge_coverage, nose, sculpt_surface, ocular_production, ocular_marking, ocular_steps, ocular_photos, case_setup, import_slot, orient, align, explode, crop, margin, texture, ssm, guide, bar,
            shell, mold_block, mold_split, mold_keys, export_check, report, snapshots, xyz, xyz_live, combined, view, section, sidebyside, overlay, brushes, measure, multires, sphere, photo, mold_auto, lasso, spout, contact, cap_from, iris_button, eye_photo, eye_design, ui)


def register():
    for m in _modules:
        m.register()
    report.install_logging()


def unregister():
    for m in reversed(_modules):
        m.unregister()
