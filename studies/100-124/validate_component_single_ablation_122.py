"""Same independent121 validator; replay two new masks with predeclared numerical tolerance."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import component_single_ablation_122 as b
import validate_component_ablation_121 as v
import numpy as np,joblib
a=b.a;e=a.e;n=a.n;r=a.r

def replay():
 a.check();d=a.load('original');path=a.OUT/'replays';path.mkdir(exist_ok=False);V=np.flatnonzero(d['fid']==0);records={};maxerr=0.
 for arm in a.ARMS:
  pert=dict(d);pert['y']=d['y'].copy();pert['y'][V]+=10000;pp,_,rr=a.fold(pert,0,arm,path/(arm+'-outer-label'));oldpath=a.OUT/'original'/f'{arm}-fold-0';oldrec=e.read(oldpath/'report.json');assert rr['outer_calibrator']==oldrec['outer_calibrator'] and rr['anchor_calibrators']==oldrec['anchor_calibrators'] and rr['selected']==oldrec['selected']
  with np.load(_repo_io(oldpath/'predictions.npz')) as z:err=float(abs(pp-z['primary']).max());assert err<1e-10;maxerr=max(maxerr,err)
  alt=dict(d);alt['P']=d['P'].copy();alt['P'][:,~a.MASKS[arm]]=1e6;alt['middleP']={k:P.copy() for k,P in d['middleP'].items()};alt['deepP']={k:P.copy() for k,P in d['deepP'].items()};alt['cells']={}
  for P in [*alt['middleP'].values(),*alt['deepP'].values()]:P[:,~a.MASKS[arm]]=1e6
  for key,(A,B,P,short,old) in d['cells'].items():Q=P.copy();Q[:,~a.MASKS[arm]]=1e6;alt['cells'][key]=(A,B,Q,short,old)
  qq,_,ss=a.fold(alt,0,arm,path/(arm+'-removed-path'));err=float(abs(pp-qq).max());assert err<1e-10;maxerr=max(maxerr,err);assert rr['selected']==ss['selected']
  bp=oldpath/'block-0';br=e.read(bp/'report.json');z=a.readnp(bp/'predictions.npz');res=z['residual'].copy();B=z['block_indices'];it=np.array(br['splits'][0]['train_indices']);iv=np.array(br['splits'][0]['validation_indices']);lookup={int(i):j for j,i in enumerate(B)};res[[lookup[int(i)] for i in iv]]+=10000
  for kind in n.KINDS:
   m=n.fit(d['X'][it],res[[lookup[int(i)] for i in it]],d['blocks'],kind);saved=joblib.load(_repo_io(bp/f'inner-0-{kind}.joblib'))['model'];err=float(abs(n.predict(m,d['X'][V])-n.predict(saved,d['X'][V])).max());assert err<1e-10;maxerr=max(maxerr,err)
  records[arm]={'outer_label_and_removed_path_interventions_passed':True,'inner_label_fixed5fits_passed':True,'calibrators_and_choices_exact':True}
 result={'passed':True,'max_prediction_replay_error':maxerr,'tolerance':1e-10,'arms':records,'upstream112_training_certificate_reused':e.sha(e.OUT/'validation-replay.json'),'source_sha256':e.sha(Path(__file__))};e.write(a.OUT/'replay-validation.json',result);a.check();e.emit(result)
if __name__=='__main__':
 a.torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=[*e.SPLITS,'replay']);x=p.parse_args();replay() if x.mode=='replay' else v.validate(x.mode)
