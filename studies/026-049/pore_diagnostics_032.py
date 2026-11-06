"""Frozen extractor peak-plateau audit, development images only; no model fit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,csv,hashlib,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, '.venv')).resolve():raise SystemExit('Use project .venv')
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from pore_curvature_features import maps,SCALES

def main():
 start=time.perf_counter();sp=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text());ids=sorted(sp['development_folds']);assert len(ids)==400
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();report=json.loads((_repo_path(ROOT, 'artifacts/run-030/report.json')).read_text());assert sha(_repo_path(ROOT, 'pore_curvature_features.py'))==report['source_sha256']['pore_curvature_features.py']
 stats={str(s):{'peak_pixels':0,'peak_components':0,'images_with_plateau':0,'max_excess_per_image':0} for s in SCALES}
 for ident in ids:
  with Image.open(_repo_path(ROOT, 'data/train', f'{ident}.png')) as im:a=np.asarray(im,dtype=np.uint8).copy()
  _,records=maps(a)
  for r in records:
   n=int(r['peaks'].sum());_,c=ndi.label(r['peaks'],structure=np.ones((3,3)));v=stats[str(r['scale'])];v['peak_pixels']+=n;v['peak_components']+=int(c);v['images_with_plateau']+=int(n>c);v['max_excess_per_image']=max(v['max_excess_per_image'],n-c)
 result={'scope':'Development400 only; no targets/model fitting/holdout/test access','per_scale':stats,'interpretation':'Equality maximum-filter counts local maximum pixels, not guaranteed one object per plateau. Adjacent maxima are components of equal-valued plateaus. Even unique extrema may correspond to phases/junctions/noise and scales may duplicate one physical pore.','seconds':time.perf_counter()-start,'source_sha256':sha(_repo_path(ROOT, 'pore_curvature_features.py'))}
 with (_repo_path(ROOT, 'reports/pore-peak-diagnostic-032.json')).open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
