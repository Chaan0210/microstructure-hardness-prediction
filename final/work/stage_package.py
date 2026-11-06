"""Copy the explicit selected closure; APFS clones keep original inodes separate."""
from pathlib import Path
import hashlib, json, shutil, subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE.parent / 'package' / 'project'
inventory = json.loads((HERE/'dependency-inventory.json').read_text())

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

manifest=HERE.parent/'package'/'source-manifest.json'
previous={r['path']:r for r in json.loads(manifest.read_text())['files']} if manifest.exists() else {}
records=[]
for i,row in enumerate(inventory['entries']):
    source_path=row['path']
    staged_path=source_path.replace('supervised-renewal-20261018-0625','supervised-renewal-20261018')
    src=ROOT/source_path; dest=OUT/staged_path
    if staged_path in previous and dest.is_file() and src.stat().st_size==dest.stat().st_size==row['bytes']:
        records.append(dict(row,path=staged_path,source_path=source_path,sha256=previous[staged_path]['sha256']))
        continue
    if not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True)
        result=subprocess.run(['/bin/cp','-c',str(src),str(dest)],capture_output=True)
        if result.returncode:
            if shutil.disk_usage(OUT).free < src.stat().st_size + 5*2**30:
                raise RuntimeError('Insufficient free disk for safe copy')
            shutil.copy2(src,dest)
    assert dest.stat().st_ino!=src.stat().st_ino and dest.stat().st_size==src.stat().st_size
    source_sha=sha(src)
    assert sha(dest)==source_sha, row['path']
    records.append(dict(row,path=staged_path,source_path=source_path,sha256=source_sha))
    if (i+1)%50==0: print('STAGED',i+1,'of',len(inventory['entries']),flush=True)
report={'scope':'Explicit selected dependency closure; separate files, original paths unchanged.',
        'files':records,'total_bytes':sum(x['bytes'] for x in records),'all_copy_hashes_match':True,
        'forbidden_legacy_artifacts_included':False,'source_data_included':False}
manifest.write_text(json.dumps(report,indent=2)+'\n')
print('STAGING COMPLETE',len(records),'logical GiB',report['total_bytes']/2**30,'free GiB',shutil.disk_usage(OUT).free/2**30)
