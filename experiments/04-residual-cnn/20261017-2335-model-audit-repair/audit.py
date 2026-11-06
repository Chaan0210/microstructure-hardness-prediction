"""Trace teacher anomalies and first-block lattice sensitivity without new scores."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import csv,json,sys,time
import numpy as np,torch
from threadpoolctl import threadpool_limits
import cnn_repair as cr
def main():
 start=time.monotonic();d,y=cr.load();keys=list(d['keys']);failed=np.nan_to_num(d['X'][:,keys.index('n_obj')],nan=0)<20
 rows=list(csv.DictReader((_repo_path(c.ROOT, 'data/train.csv')).open()));mapping={r['ID']:float(r['hardness']) for r in rows};assert np.array_equal(y,np.array([mapping[str(i)] for i in d['ids']]))
 hashes=[c.ah(x) for x in d['images']];assert len(set(hashes))==500
 teachers=[]
 for p in sorted((c.R4/'teachers').glob('*/predictions.npz')):
  z=np.load(_repo_io(p));tr,q=z['train'],z['query'];assert set(tr).isdisjoint(q);raw=sum(w*z[k] for k,w in {'tab':.384,'ridge':.128,'logridge':.128,'set':.05,'gmm':.15}.items())/.84;pred=raw.copy();pred[failed[q]]=y[tr].mean();assert np.allclose(pred,z['base'],atol=1e-10,rtol=0)
  removal=.05/.84*(z['tab']-z['set']);removal[failed[q]]=0;hi=(d['noise'][q]>=12)&~failed[q];j=int(np.argmax(abs(z['set']-y[q])));teachers.append(dict(name=p.parent.name,train_n=len(tr),query_n=len(q),worst_set_id=str(d['ids'][q[j]]),worst_set_error=float(z['set'][j]-y[q[j]]),worst_is_failed=bool(failed[q[j]]),max_actual_set_replacement_HV_high=float(abs(removal[hi]).max()),rms_actual_set_replacement_HV_high=float(np.sqrt(np.mean(removal[hi]**2)))))
 z=cr.data(d,y,0,'final','073');idx=z['valid'][d['noise'][z['valid']]>=12][:16];qids=d['ids'][idx];records=[]
 for seed in [0,1]:
  path=c.R4/f'aux-seed{seed}/fold0-final';ck=torch.load(_repo_io(path/'model.pt'),map_location='cpu',weights_only=True);model=cr.Base();model.load_state_dict(ck['state_dict']);raw=cr.predict(model,z['images'],z['context'],z['valid'],'aux')*z['meta']['target_std'];stored=np.load(_repo_io(path/'predictions.npz'));err=float(np.max(abs(raw-stored['validation_prediction'])));assert err==0,err
  phase=[]
  for sy,sx in [(0,0),(0,1),(1,0),(1,1)]:
   # Reflect one-pixel border shifts only for a label-blind sensitivity diagnostic.
   im=z['images'][idx];pad=np.pad(im,((0,0),(0,0),(1,1),(1,1)),mode='reflect');im=pad[:,:,1+sy:257+sy,1+sx:257+sx];p=cr.predict(model,im,z['context'][idx],np.arange(len(idx)),'aux')*z['meta']['target_std'];phase.append(p)
  pp=np.array(phase);records.append(dict(seed=seed,baseline_replay_maxdiff=err,phase_rms_difference_HV=float(np.sqrt(np.mean((pp[1:]-pp[0])**2))),max_absolute_phase_difference_HV=float(abs(pp[1:]-pp[0]).max()),mean_across_phase_sd_HV=float(pp.std(0).mean())));c.arrays(f'lattice-seed{seed}.npz',ids=qids,predictions=pp)
 c.save(_repo_io('audit.json'),dict(train_labels_id_aligned=True,unique_original_pixel_hashes=len(set(hashes)),images=500,failed_ids=d['ids'][failed].tolist(),teacher_rows=teachers,lattice=records,lattice_uses_hardness=False,seconds=time.monotonic()-start,limits='Old masks/features were trained/selected on reused images. Shared residuals or declining correlations do not identify irreducible error. Lattice shifts are diagnostics, not proved hardness-preserving training transforms.'))
 print(json.dumps(dict(lattice=records,largest_teacher_cases=sorted(teachers,key=lambda x:abs(x['worst_set_error']),reverse=True)[:5]),indent=2),flush=True)
if __name__=='__main__':
 torch.set_num_threads(2);torch.set_num_interop_threads(1);sys.addaudithook(c.guard)
 with threadpool_limits(limits=2):main()
