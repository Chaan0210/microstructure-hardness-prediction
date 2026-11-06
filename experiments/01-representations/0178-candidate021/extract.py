"""Fresh178 rawtrain500 or lockedtest1000 Lfeatures, no backbone fitting."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse,csv,time
import runtime as f
import numpy as np,torch
R=f.R;P=f.P

def main():
 ap=argparse.ArgumentParser();ap.add_argument('split',choices=['train','test']);split=ap.parse_args().split;plan=f.check()
 if split=='test':
  v=f.read(R/'training-validation.json');assert v['passed'];manifest=f.read(R/'models/manifest.json');assert manifest['validation_sha256']==f.sha(R/'training-validation.json')
  for n,h in manifest['files'].items():assert f.sha(R/'models'/n)==h
  f.inference_guards()
 expected=f.read(P/f'experiments/reproduction-018-176/config/original-{split}-hashes.json');file=P/'data'/('train.csv' if split=='train' else 'sample_submission.csv');assert f.sha(file)==expected['CSV_sha256']
 with file.open(newline='',encoding='utf-8-sig') as ff:r=csv.DictReader(ff);assert r.fieldnames==['ID','hardness'];rows=list(r)
 count=500 if split=='train' else 1000;ids=[row['ID'] for row in rows];assert ids==[f'{split.upper()}_{j:06d}' for j in range(1,count+1)];y=np.array([float(row['hardness']) for row in rows]) if split=='train' else None
 if y is not None:assert np.isfinite(y).all()
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,f.os.PathLike)):
   p=f.Path(f.os.fsdecode(args[0])).absolute()
   if p.is_relative_to(P/'data') and not p.is_relative_to(P/'data'/split):raise RuntimeError('Wrongrawdata scope: '+str(p))
 f.sys.addaudithook(guard);start=time.perf_counter();values=[];hashes={};cpu_spot=None
 with f.Monitor('features-'+split,plan['resources']) as monitor:
  m=f.model();initial=f.statehash(m);m=m.to('mps')
  for j,i in enumerate(ids):
   values.append(f.feature(m,torch.from_numpy(f.image_input(i,split)[None]).to('mps')));hashes[i]=f.sha(P/'data'/split/f'{i}.png')
   if (j+1)%100==0:monitor.sample();f.emit({'stage':split,'rows':j+1,'seconds':time.perf_counter()-start,'resource':monitor.trace[-1]})
  X=np.stack(values).astype(float);assert X.shape==(count,9216) and np.isfinite(X).all();assert hashes==expected['image_sha256'];assert f.statehash(m)==initial;assert all(v.grad is None for v in m.parameters())
  if split=='test':
   m=m.to('cpu');torch.mps.empty_cache();spot=plan['CPU_feature_spotcheck_test_IDs'];cpu=np.stack([f.feature(m,torch.from_numpy(f.image_input(i,split)[None])) for i in spot]);ref=X[[ids.index(i) for i in spot]];error=float(abs(cpu-ref).max());assert error<plan['tolerances']['CPU_MPS_Lfeature_abs'];cpu_spot={'IDs':spot,'max_abs_feature_difference':error}
   with (R/'test-CPU-feature-spotcheck.npz').open('xb') as ff:np.savez_compressed(_repo_io(ff),ids=spot,CPU=cpu,MPS=ref)
  del m;torch.mps.empty_cache()
  with (R/f'{split}-features.npz').open('xb') as ff:np.savez_compressed(_repo_io(ff),ids=ids,X=X,**({'y':y} if y is not None else {}))
  development_exact=None
  if split=='train':
   with np.load(_repo_io(P/'experiments/dinol-177/features.npz'),allow_pickle=False) as old:np.testing.assert_array_equal(X[[ids.index(i) for i in old['ids']]],old['X']);development_exact=True
 rec={'passed':True,'split':split,'rows':count,'feature_sha256':f.sha(R/f'{split}-features.npz'),'raw_CSV_sha256':expected['CSV_sha256'],'image_sha256':hashes,'all_raw_hashes_match_previous':True,'no_backbone_fit':True,'state_unchanged':True,'development400_exact':development_exact,'test_CPU_spot':cpu_spot,'seconds':time.perf_counter()-start,'source_sha256':f.sha(f.Path(__file__))};f.write(R/f'{split}-feature-validation.json',rec);f.emit({k:v for k,v in rec.items() if k!='image_sha256'})
if __name__=='__main__':
 try:main()
 except Exception as e:f.write(R/f'extraction-{f.sys.argv[1]}-failure.json',{'error':repr(e)});raise
