"""Explicit post-hoc replication of fixed092; no candidate promotion in this study."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import patch_attention_092 as original
import numpy as np,torch
fv=original.fv;e=original.e;prior=original.prior;CFG=original.CFG;ARMS=original.ARMS;OUT=_repo_path(ROOT, 'artifacts/patch-replication-093');SPLITS=('20261031','20261101')
inputs=original.inputs;get_cache=original.get_cache;normalize=original.normalize;Pool=original.Pool;predict=original.predict;seed_for=original.seed_for;fit_pair=original.fit_pair

def result_path(split):return original.OUT/'original' if split=='original' else _repo_path(OUT, split)

def prepare():
 original.check();v=json.loads((_repo_path(ROOT, 'reports/patch-attention-092-validation.json')).read_text());assert v['passed'] and not v['results']['original']['advance'];r=json.loads((original.OUT/'original/report.json').read_text());top=r['sample_diagnostics']['top5_row_IDs_diagnostic_only'];assert len(set(top))==5;OUT.mkdir(exist_ok=False)
 p={'study':'093 post-hoc fixed092 replication, not candidate selection','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':'User explicitly authorized relaxing the failed092 resplit screen after observing attention improve matchedmean inall5 originalfolds. Purpose is reproducibility and fixedsample tracking, not promotion.','selection_bias':'This extension was selected after observing092 original gains. The092 >=3positivefold gate against011 failed(2/5) andremains recorded;093 deliberately relaxes it posthoc. Same400 historicallyreused rows andoriginal-selectedtop5 are notindependent confirmation or unbiased subgroups. No revised pre-registration claim.','unchanged':'Import exact092 Pool,fit_pair,predict,normalize,schedule,CFG andtoken cache. Bothmean andattention trained onthetwo existingresplits;20epochs,seed20270322+fold,all12913parameters,coordinates,.9*011+.1*arm. No settings/ratio/seed/epoch changes orfallbackrecipe.','config':CFG,'fixed_original_top5_ids':top,'top5_definition':'Five largest original squared-error improvements: (y-011)^2-(y-attentionblend)^2, stable descending order. Keep exactlytheseIDs inbothresplits; do not selectnewtop5 for thisprimarytracking.','comparisons':'Separate: attention versusparameter-matchedmean (standalone andblend); fixedblend versus011; perID gain fororiginaltop5; RMSE/MSE/absolute-error andconditionalbootstrapgain onfixedremaining395; sameIDsignconsistency acrosspartitions. No candidatepromotionthreshold.','candidate_policy':'No candidateCSV,full500training,testinference,submission orhyperparametersearch regardless of results. End with validity assessment ofthisfixedhypothesis only.','resources':'092actual paired5foldabout3minutes,cache alreadyvalidated;two resplits about6minutes plus2firstfoldpairreplays ~75seconds. Existinguv CPU2/MPS,cost0;reusecache,nonewdownloads.','sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['patch_replication_093.py','validate_patch_replication_093.py','summarize_patch_replication_093.py','patch_attention_092.py','validate_patch_attention_092.py']},'evidence':{n:fv.sha(_repo_path(ROOT, n)) for n in ['reports/patch-attention-092-validation.json','reports/patch-attention-092-summary.json','artifacts/patch-attention-092/original/report.json','artifacts/patch-attention-092/original/development_predictions.npz','artifacts/patch-attention-092/cache.json','reports/candidate011-public-submission.json']},'scope':'Development400only,holdout/testunreadforselection,noexternaldata/install/cost/upload,best011preserved.','limits':v['limits']};fv.write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'posthoc_extension':True,'fixed_original_top5':top,'candidate_promotion':False}),flush=True)

def check():
 original.check();prior.diag.check_best();p=json.loads((_repo_path(OUT, 'plan.json')).read_text());assert p['config']==CFG
 for n,h in {**p['sources'],**p['evidence']}.items():assert fv.sha(_repo_path(ROOT, n))==h,n
 return p

def run(split):
 check();assert split in SPLITS;torch.set_num_threads(2);d=inputs(split);tokens,mom=get_cache(d);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);preds={a:np.full(400,np.nan) for a in ARMS};start=time.perf_counter()
 for k in range(5):
  p,r=fit_pair(d,tokens,mom,k,out/f'fold-{k}');va=np.flatnonzero(d['fid']==k)
  for arm in ARMS:preds[arm][va]=p[arm]
 r=original.evaluate(d,out,preds,time.perf_counter()-start);check();print(json.dumps({'split':split,'original092_gate_fields_diagnostic_only':True,'candidate_promotion':False,**r}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare',*SPLITS]);a=ap.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
