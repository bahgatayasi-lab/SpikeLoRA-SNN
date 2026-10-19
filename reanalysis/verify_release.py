"""Verify scientific R3 records and unchanged R2 files; optionally package them.

Standard library only. No training, inference, network access or dataset download.
Two approved tables are materialized locally when absent; see the landing README.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import zipfile
from materialize_approved_tables import materialize
ROOT=Path(__file__).resolve().parents[1]
TAG='v1.1-NCAA-R3'
def digest(data):return hashlib.sha256(data).hexdigest()
def checked(relative):
    p=(ROOT/relative).resolve()
    if not p.is_relative_to(ROOT.resolve()) or not p.is_file():raise ValueError('Missing or unsafe path: '+relative)
    return p
def scientific_summary(data):
    v=json.loads(data);v.pop('environment',None)
    return digest(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--strict-bytes',action='store_true');ap.add_argument('--build-package',action='store_true')
    args=ap.parse_args();materialize()
    manifest=json.loads(checked('R3_MANIFEST.json').read_text())
    for e in manifest['files']:
        data=checked(e['path']).read_bytes()
        if digest(data)!=e['sha256']:
            if args.strict_bytes or e['path']!='reanalysis/results/R3_analysis_summary.json' or scientific_summary(data)!=manifest['summary_scientific_sha256']:
                raise ValueError('R3 hash mismatch: '+e['path'])
    preserved=json.loads(checked('R2_PRESERVATION_MANIFEST.json').read_text())
    for e in preserved['unchanged_existing_files']:
        if digest(checked(e['path']).read_bytes())!=e['sha256']:raise ValueError('R2 hash mismatch: '+e['path'])
    inputs=json.loads(checked('R3_INPUT_MANIFEST.json').read_text())
    for e in inputs['records']:
        if checked(e['path']).read_bytes()!=checked(e['source_path']).read_bytes():raise ValueError('Original-input copy differs: '+e['path'])
    count=sum(e['path'].startswith('reanalysis/results/') and e['path'].endswith('.csv') for e in manifest['files'])
    if count!=17:raise ValueError('Expected 17 derived CSVs')
    print(f'PASS: {len(manifest["files"])} scientific hashes; 17 derived CSVs; {len(preserved["unchanged_existing_files"])} unchanged R2 files; seven exact input copies.')
    print('No model training or inference; this does not independently verify historical model execution.')
    if args.build_package:
        names={e['path'] for e in manifest['files']}|{e['path'] for e in preserved['unchanged_existing_files']}
        names.update(['README.md','CITATION.cff','RELEASE_NOTES_R3.md','ARTIFACT_AVAILABILITY.md','ENVIRONMENT_R3_analysis.json','requirements_R3_analysis.txt',
          'R3_MANIFEST.json','R3_INPUT_MANIFEST.json','R2_PRESERVATION_MANIFEST.json','reanalysis/README.md','reanalysis/verify_release.py',
          'reanalysis/materialize_approved_tables.py','reanalysis/retained_tables/R3_calendar_composition.csv.gz',
          'evidence/R3_CORRECTIONS_AND_SCOPE.md','evidence/R3_MANUSCRIPT_OUTPUT_MAP.md'])
        dist=ROOT/'dist';dist.mkdir(exist_ok=True);archive=dist/f'SpikeLoRA_TS_R3_Reanalysis_{TAG}.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for n in sorted(names):
                info=zipfile.ZipInfo('SpikeLoRA_TS_R3/'+n,(2026,10,19,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
                z.writestr(info,checked(n).read_bytes())
        archive.with_suffix('.zip.sha256').write_text(digest(archive.read_bytes())+'  '+archive.name+'\n')
        print('Built:',archive.name)
if __name__=='__main__':main()
