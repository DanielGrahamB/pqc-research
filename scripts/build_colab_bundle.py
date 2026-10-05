"""Regenerate the self-contained notebook after editing experiment source files."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'PQC_MIMO_Simulations.ipynb'

def main():
    paths = sorted(list((ROOT/'pqc_experiments').glob('*.py'))
                   + list((ROOT/'tests').glob('*.py'))
                   + [ROOT/name for name in ('metrics.py', 'requirements-colab.txt',
                      'requirements-experiments.txt', 'EXPERIMENTS.md', 'run_pilot.py')])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            info=zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            bundle.writestr(info,path.read_bytes())
    data=buffer.getvalue()
    source=(ROOT/'scripts/colab_bootstrap.py').read_text().replace('__BUNDLE_SHA256__',hashlib.sha256(data).hexdigest()).replace('__BUNDLE_BASE64__',base64.b64encode(data).decode())
    notebook=json.loads(NOTEBOOK.read_text())
    cells=[cell for cell in notebook['cells'] if cell.get('id')=='colab-setup']
    if len(cells)!=1:
        raise RuntimeError('Expected one colab-setup cell')
    cells[0]['source']=source.splitlines(True)
    cells[0]['execution_count']=None
    cells[0]['outputs']=[]
    NOTEBOOK.write_text(json.dumps(notebook,indent=1)+'\n')
    print(f'Bundled {len(paths)} files ({len(data):,} compressed bytes) into {NOTEBOOK.name}')

if __name__=='__main__':main()
