import bpy
import textwrap

NOTICE = 'Experimental preview — not clinically validated\n\nAnaplast Studio is an open-source Blender add-on under development for digital workflows in anaplastology and maxillofacial prosthetics.\n\nThis release is provided for education and technical evaluation using synthetic demonstration models. Its safety, accuracy and suitability for patient treatment have not been clinically validated.\n\nQualified clinicians must exercise independent clinical judgment and critically assess outputs. Clinical judgment does not replace the validation or regulatory requirements applicable to patient treatment.\n'

class ANAPLAST_PT_release(bpy.types.Panel):
    bl_label = 'Anaplast Studio v1.0.0'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Anaplast'
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = -100

    def draw(self, context):
        layout = self.layout
        for paragraph in NOTICE.split('\n\n'):
            column = layout.column(align=True)
            for line in textwrap.wrap(paragraph.strip(), width=42):
                column.label(text=line)
        layout.label(text='Source and licenses are included in the ZIP.')

def register():
    bpy.utils.register_class(ANAPLAST_PT_release)

def unregister():
    bpy.utils.unregister_class(ANAPLAST_PT_release)
