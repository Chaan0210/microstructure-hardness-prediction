"""Exploratory, post-084 frozen control follow-up. No tuning or joint training."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import dinob_joint_084 as e
import numpy as np,torch
fv=e.fv;OUT=_repo_path(ROOT, 'artifacts/frozen-complement-085');SPLITS=('original','20261031','20261101')

def prepare():
 e.check();v=json.loads((_repo_path(ROOT, 'reports/dinob-joint-084-validation.json')).read_text());assert v['passed'] and not v['eligible_for_candidate'];OUT.mkdir(exist_ok=False)
 p={'study':'085 exploratory frozen-head complement follow-up','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':'User explicitly authorized promoting the original084 frozen control observation to a separate exploratory hypothesis, testing both existing resplits, and candidate011 full500 training/inference only after consistent improvement. No upload.','hypothesis':'A strongly regularized fresh frozen-B head may complement009 at the already fixed10% weight. This is not a claim that joint fine-tuning helps.','selection_bias':'The purpose changed after inspecting084 original results: joint-training superiority was rejected; the frozen control blend was selected post hoc for apparent complementarity. Original13.05399777 is discovery evidence, not fresh confirmation. Both resplits reuse the same400 historically selected rows and are not independent validation. No adjustment for this adaptive selection; conditional intervals omit fitting and selection uncertainty.','config':e.CFG,'formula':'.9*samefold_verified009 + .1*frozen084_head','unchanged':'Exact084 Head6912-32-1, dropout.5,12fixedepochs, AdamW/cosine, SmoothL1beta1, seed20270215+fold, micro2/effective16, foldtrain-only scaler/target stats, no augmentation, no backbone updates. Reuse original frozen fits verbatim. No new calibration; retain nested009 coefficients.','gate':'Run both existing resplits without intervening tuning. Eligible only if fixedblend pooledRMSE strictly improves009 and010 in all3partitions and independent validation passes. Foldcounts/conditionalCI are diagnostics, not tuning criteria. Otherwise reject; no ratios, epochs, seeds, joint training or alternate settings.','full500_rule':'If eligible, lock candidate011 before loading full500 training labels/test. Train identical frozen head once onall500, seed20270215,12epochs with same per-epoch schedule; scaler/target stats fit on500. Keep all009 models/calibrator unchanged. Formula remains.9*009+.1*newhead. Test inference only after model is fixed.','sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['frozen_complement_085.py','validate_frozen_complement_085.py','dinob_joint_084.py']},'evidence':{n:fv.sha(_repo_path(ROOT, n)) for n in ['reports/dinob-joint-084-validation.json','reports/dinob-joint-084-summary.json','artifacts/dinob-joint-084/original/report.json','artifacts/dinob-joint-084/original/development_predictions.npz']},'scope':'Development400 selection only; no additional holdout evaluation/test selection/install/download/cost.009 and010 protected;010 remains unsubmitted.','limits':v['limits']};fv.write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'gate':p['gate']}),flush=True)

def check():
 e.check();p=json.loads((_repo_path(OUT, 'plan.json')).read_text());assert p['config']==e.CFG
 for n,s in {**p['sources'],**p['evidence']}.items():assert fv.sha(_repo_path(ROOT, n))==s,n
 return p

def foldpath(split,k):return (e.OUT/'original'/f'fold-{k}'/'frozen') if split=='original' else _repo_path(OUT, split, f'fold-{k}')

def run(split):
 check();torch.set_num_threads(2);d=e.inputs(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);start=time.perf_counter();p=np.full(400,np.nan);m=None if split=='original' else e.backbone.model();records={}
 for k in range(5):
  path=foldpath(split,k);va=np.flatnonzero(d['fid']==k)
  if split=='original':
   rec=json.loads((path/'report.json').read_text());assert fv.sha(path/'predictions.npz')==rec['predictions_sha256']
   with np.load(_repo_io(path/'predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(z['validation_indices'],va);pp=z['predictions'].copy()
  else:pp,rec=e.fit_arm(d,None,m,k,'frozen',path);torch.mps.empty_cache()
  p[va]=pp;records[str(path.relative_to(ROOT)/'report.json')]=fv.sha(path/'report.json')
 blend=.9*d['reference009']+.1*p;ps={'frozen':p,'primary':blend,'reference009':d['reference009'],'reference010':d['reference010']};assert all(np.isfinite(a).all() for a in ps.values());comp={n:e.base.prior.old.comparison(d['y'],d[n],blend,d['fid']) for n in ('reference009','reference010')}
 with (out/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],**ps)
 r={'scores':{n:{'rmse':fv.rmse(d['y'],a),'fold_rmse':[fv.rmse(d['y'][d['fid']==k],a[d['fid']==k]) for k in range(5)]} for n,a in ps.items()},'comparisons':comp,'advance':all(c['gain']>0 for c in comp.values()),'fold_report_sha256':records,'predictions_sha256':fv.sha(out/'development_predictions.npz'),'original_reused_without_refit':split=='original','seconds':time.perf_counter()-start};fv.write(out/'report.json',r);check();print(json.dumps({'split':split,**r}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare',*SPLITS]);a=ap.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
