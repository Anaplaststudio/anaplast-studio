"""Offline material catalogue. Reviewed 2026-09-25; no network at runtime.

Products without a verified density deliberately have None. Specific gravity
is used numerically as approximate g/mL, not as a guaranteed mixing yield.
Package sizes/cartridges share their parent formulation. Adhesives, pigments,
solvents and silicone fluids are not cavity casting materials.
"""
import math

FACTOR='https://factor2.com/'
CATALOG='https://www.factor2.com/v/vspfiles/downloadables/2023%20Catalog%20new%20.pdf'
TECHNO='https://www.technovent.com/_files/ugd/733960_c5bd41db9c294ae49405e7ed4c4048da.pdf'
VST50_DENSITY=11./(10./1.12+1./1.025)
VST50_SDS=FACTOR+'cdn/shop/files/VST-50_Safety_Data_Sheet.pdf?v=8775496552372887665'
VST50_TDS=FACTOR+'cdn/shop/files/VST-30_Technical_Data_Sheet.pdf?v=7298983724781266204'
CATALOGUE={}

def add(key,label,density=None,source='',basis='Supplier specific gravity; approximate weight',number=None):
    CATALOGUE[key]=dict(label=label,density=density,source=source,basis=basis,
                        number=number if number is not None else len(CATALOGUE))

# Preserve old Blender enum numbers 0 and 1 when opening existing cases.
add('VST50','Factor II - VST-50',VST50_DENSITY,VST50_SDS+' ; '+VST50_TDS,
    'Estimated mixture: A 1.12, B 1.025 g/mL; 10:1 by weight; additive volumes',0)
add('CUSTOM','Custom silicone',number=1)
for code,rho in [('A-101',None),('A-103',1.10),('A-110',None),('A-135',1.10),
                 ('A-150',None),('A-165',None),('A-2000',None),('A-2186',1.10),('A-588-1',None),
                 ('VST-01',None),('VST-05',None),('VST-10',None),('VST-30',None),('VST-50F',None),('VST-50HD',None)]:
    source=(CATALOG if code in {'A-150','A-165'} else
            'https://www.factor2.com/v/vspfiles/Factor2_Catalog_2014.pdf' if code in {'VST-01','VST-05','VST-10'} else
            FACTOR+'v/vspfiles/assets/images/a-2000%20tech.pdf' if code=='A-2000' else FACTOR+'products/'+code.lower())
    add(code.replace('-',''), 'Factor II - '+code,rho,source)
for code,rho in [('3020',1.08),('3040',1.08),('3045',1.08),('4020',None),('4040',None),
                 ('4040-MC',None),('4410',1.10),('4420',1.10),('4510',1.08),('1556',None),('1556-F',None)]:
    add('RTV'+code.replace('-',''),'Factor II / Elkem - RTV-'+code,rho,
        CATALOG if code=='4510' else FACTOR+'products/a-rtv-'+code.lower(),
        'Supplier cured density; approximate weight' if code=='4510' else 'Supplier specific gravity; approximate weight')
for code,lsr,rho in [('221-01','4301',1.07),('221-05','4305',1.07),('221-10','4310',1.09),
                     ('223-25','4325',1.11),('223-30','4330',1.11),('223-40','4340',1.12),
                     ('225-50','4350',1.12),('225-60','4360',1.12),('225-70','4370',1.14)]:
    add('A'+code.replace('-',''),'Factor II / Elkem - A-'+code+' (LSR '+lsr+')',rho,FACTOR+'products/a-'+code)
for code in ('4520','4530','4550','4565'):
    add('HCRA'+code,'Factor II / Elkem - HCRA '+code,None,FACTOR+'collections/hcr-high-consistency-silicones')
for key,label in [('M511','M511'),('S25','Teksil 25 / S25'),('Z004','Z004')]:
    add(key,'Technovent - '+label,1.10,TECHNO,'Supplier family SDS density at 20 C; approximate weight')
for label in ('Teksil 20','Teksil 35','Teksil 65','Alsil 10','Alsil 20','Alsil 35','Alsil 65','Alsil 80'):
    add(label.upper().replace(' ',''),'Technovent - '+label,None,
        FACTOR+'cdn/shop/files/A-110_Technovnet_TDS.pdf?v=4706707166150024439')
for shore in (5,10,20,30,40,55,70):
    add('DERMA'+str(shore),'Spectromatch - derma-sil '+str(shore)+' Shore A',None,
        'https://www.spectromatch.com/products/derma-sil/')
for variant in ('transparent','city','country','beach','soft-form','hard-form'):
    add('MULTISIL_'+variant.upper().replace('-','_'),'Bredent - Multisil-Epithetik '+variant,None,
        'https://bredent.pl/files/bredent_epitezy_1.pdf')
for code,label,source in [('A106','A-106 / 2370 expanding foam',FACTOR+'products/a-106'),
                         ('A340','A-340 firm gel',FACTOR+'products/a-340'),
                         ('A341','A-341 soft gel','https://www.factor2.com/v/vspfiles/sds%20tds%202023/tds%202023/A-341.pdf'),
                         ('A4717','A-4717 high-tack gel','https://www.factor2.com/Silicone_Gels_s/56.htm'),
                         ('M517','M-517 CoForm soft impression',FACTOR+'products/m-517'),
                         ('M518','M-518 CoForm hard impression','https://www.technovent.com/msds')]:
    add(code,'Factor II - '+label,None,source)
add('MDX44210','DuPont Liveo / Silastic - MDX4-4210',None,
    'https://www.dupont.com/content/dam/dupont/amer/us/en/liveo/public/sds/en/liveo-mdx4-4210-US-SDS-000000859731.pdf')

ITEMS=[(key,row['label'],row['basis'] if row['density'] else 'Enter the supplier or measured density for this material',row['number'])
       for key,row in CATALOGUE.items()]

def density_get(props):
    values=props.get('silicone_density_values',{})
    return float(values.get(props.mold_silicone,0.))

def density_set(props,value):
    values=dict(props.get('silicone_density_values',{}));values[props.mold_silicone]=float(value)
    props['silicone_density_values']=values

def material(props):
    key=props.mold_silicone;row=CATALOGUE[key]
    density=row['density'];basis=row['basis']
    if key=='CUSTOM':density=float(props.mold_silicone_density);basis='User-specified mixed density'
    elif density is None:
        density=density_get(props);basis='User-specified density for '+row['label']
    return dict(material=row['label'],density_g_ml=density if math.isfinite(density) and density>0 else None,
                density_basis=basis,density_source=row['source'])
