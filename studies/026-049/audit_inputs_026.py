"""Development PNG inspection only. No model fitting or holdout/test access."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv,hashlib,json,sys
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, '.venv')).resolve():raise SystemExit('Use project .venv')
import numpy as np
from PIL import Image
from scipy.stats import spearmanr

def stats(x):return dict(zip(('min','q10','median','q90','max'),map(float,np.percentile(x,[0,10,50,90,100]))))
def energy(a):return float(np.mean(np.diff(a,axis=0)**2)+np.mean(np.diff(a,axis=1)**2))
def main():
 sp=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text());ids=sorted(sp['development_folds']);assert len(ids)==400
 with (_repo_path(ROOT, 'artifacts/run-022/development_oof.csv')).open() as f:rows=list(csv.DictReader(f))
 rr={r['ID']:r for r in rows};assert len(rows)==len(rr)==400 and set(rr)==set(ids)
 y=np.array([float(rr[i]['hardness']) for i in ids]);pred={m:np.array([float(rr[i][c]) for i in ids]) for m,c in [('candidate003','candidate003'),('candidate004','combined_kernel_multiseed_blend')]}
 metrics=[];formats={};chunks={};metacount=0;constant_edges=0;hashes={}
 for ident in ids:
  p=_repo_path(ROOT, 'data/train', f'{ident}.png');raw=p.read_bytes();hashes[ident]=hashlib.sha256(raw).hexdigest();assert raw[:8]==b'\x89PNG\r\n\x1a\n'
  off=8;cc=[]
  while off<len(raw):
   n=int.from_bytes(raw[off:off+4],'big');cc.append(raw[off+4:off+8].decode('ascii'));off+=12+n
  key=','.join(cc);chunks[key]=chunks.get(key,0)+1
  with Image.open(_repo_io(p)) as im:
   key=f'{im.mode}:{im.size}';formats[key]=formats.get(key,0)+1;metacount+=bool(im.info)
   assert im.mode=='L' and im.size==(256,256)
   a=np.asarray(im,dtype=float)/255;b=np.asarray(im.resize((128,128),Image.Resampling.BOX),dtype=float)/255
  v={'png_bytes':len(raw),'raw_mean':a.mean(),'raw_std':a.std(),'box_mean_shift':b.mean()-a.mean(),'box_std':b.std(),'std_ratio':b.std()/a.std(),'gradient_energy_ratio':energy(b)/energy(a)}
  for w in (4,8):
   mask=np.ones(a.shape,dtype=bool);mask[w:-w,w:-w]=False;v[f'border{w}_mean_minus_interior']=a[mask].mean()-a[~mask].mean()
  constant_edges+=sum(np.ptp(e)==0 for e in (a[0],a[-1],a[:,0],a[:,-1]));metrics.append(v)
 relationships={}
 for name in metrics[0]:
  v=np.array([r[name] for r in metrics]);relationships[name]={'distribution':stats(v),'target_spearman':float(spearmanr(v,y).statistic),**{m:{'signed_residual_spearman':float(spearmanr(v,y-p).statistic),'absolute_residual_spearman':float(spearmanr(v,abs(y-p)).statistic)} for m,p in pred.items()}}
 r={'scope':'Only400 development PNGs and run022 OOF; no fitting or inference','oof_candidate003_column':'candidate003','oof_candidate004_column':'combined_kernel_multiseed_blend','formats':formats,'chunks':chunks,'images_with_metadata':metacount,'constant_outer_edges_of1600':int(constant_edges),'relationships':relationships,'image_sha256':hashes,'limits':['Previous independent input audit compared run020 candidate_mean3_d4 as004; this was an intermediate compact26 candidate, NOT final004. This report corrects all residual correlations using final run022 OOF.','Gradient energies at different pixel spacing are not a literal information-loss percentage.','Correlations are exploratory, noncausal, and use repeatedly inspected development data.','No holdout/test distribution comparison; no claim identifying the cause of Public reversal.']}
 (_repo_path(ROOT, 'reports/input-audit-026.json')).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='image_sha256'},indent=2))
if __name__=='__main__':main()
