"""Build source with the separately distributed, checksum-verified reference model."""
from pathlib import Path
import argparse,hashlib,zipfile
p=argparse.ArgumentParser();p.add_argument('--release-zip',type=Path,required=True);args=p.parse_args()
expected='d863b3b213c2bc6ae709cf167f6facfa39ed88fc12f758324b35c0a231653d68'
if hashlib.sha256(args.release_zip.read_bytes()).hexdigest()!=expected:raise SystemExit('Incorrect v1.0.0 reference archive checksum')
root=Path(__file__).resolve().parent;out=root/'build';out.mkdir(exist_ok=True)
model='assets/FLAME2023Open_Nose.npz'
with zipfile.ZipFile(args.release_zip) as source,zipfile.ZipFile(out/'Anaplast_Studio-source-build.zip','w',zipfile.ZIP_DEFLATED) as target:
    for path in sorted(root.rglob('*')):
        rel=path.relative_to(root)
        if not path.is_file() or any(part in {'.git','build','__pycache__'} for part in rel.parts):continue
        if path.suffix in {'.pyc','.blend','.stl','.obj','.dcm','.zip'} or rel.as_posix()==model:continue
        target.write(path,rel.as_posix())
    target.writestr(model,source.read(model))
print(out/'Anaplast_Studio-source-build.zip')
