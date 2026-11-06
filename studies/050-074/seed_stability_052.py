"""Fixed three-seed regression-MLP mean; auxiliary classification excluded."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time,resource,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import numpy as np,torch
import auxiliary_head_051 as h
import auxiliary_experiment_051 as e
import frozen_validation as fv
OUT=_repo_path(ROOT, 'artifacts/seed-stability-052')
SEEDS=(20261128,20261205,20261219)
SPLITS=('original','20261031','20261101')

def prepare():
 OUT.mkdir(exist_ok=False);assert json.loads((_repo_path(ROOT, 'reports/auxiliary-051-validation.json')).read_text())['valid']
 prior=json.loads((e.OUT/'plan.json').read_text())
 protected=dict(prior['protected']);protected.update({str(p.relative_to(ROOT)):fv.sha(p) for p in e.OUT.rglob('*') if p.is_file()})
 plan={'hypothesis':'Average fixed regressionMLP seeds to reduce initialization variance without selecting favorable seeds. Auxiliary loss excluded.','seed_bases':list(SEEDS),'fold_seed':'seed_base+fold,as051;full500usesseed_base','reuse':'Originalseed20261128 already validated051 checkpoints, same folds. Reuse exactly, train only two newly fixed seeds.','configuration':h.CONFIG,'auxiliary_weight':0.,'seed_aggregation':'Exact unweighted mean of all3 seed regression predictions; no seed removal,weights,checkpoint/epoch choice or hyperparameter search.','final_formula':'.75unchangedcandidate003+.25mean3MLP; no006Ridge mixing or Public adjustment','scope':'Development400 for validation only; afterall3positive+reprocheck, parent authorized fixeddesign thenfull500training/test1000inference. No holdout scoring, testfit/adaptation, newinstall,cost,upload.','diagnostics':'Foldchanges,per-seed scores,meanMLP training-validation gap,residualcorrelation to003 andbetween seeds; remove1/5/10mostbeneficialrows asposthoc influence stress,not changingfittedmodel orchoice.','gate':'Three-seedmean fixedblend must improve006 pooledRMSE on eachoriginal/20261031/20261101 and passindependentcheckpoint/normalization/reproductionchecks. No bootstrap significance or arbitrary gain cutoff.','bias':'Samehistoricallyreused400rows,overlappingrepartitions; prior051controlfindingposthoc,additionalfixedseeds arenotnewindependentdata.','sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['seed_stability_052.py','auxiliary_head_051.py','auxiliary_experiment_051.py']},'inputs':prior['inputs'],'protected':protected}
 fv.write(_repo_path(OUT, 'plan.json'),plan);print(json.dumps({'prepared':True,'seeds':SEEDS,'config':h.CONFIG}),flush=True)

def check():
 p=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,v in {**p['sources'],**p['inputs'],**p['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==v,n
 return p

def restore(a):
 model=h.Head();model.load_state_dict(a['model']);return model.eval()

def run(split):
 check();torch.set_num_threads(2);ids,X,y,groups,fid,ref003,ref006=e.inputs(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);pred={s:np.full(400,np.nan) for s in SEEDS};train_errors=[];logs=[];t=time.perf_counter()
 oldreport=json.loads((e.OUT/split/'report.json').read_text())
 for fold in range(5):
  tr=np.flatnonzero(fid!=fold);va=np.flatnonzero(fid==fold);assert not set(groups[tr])&set(groups[va]);sc,ym,ys,edges,bins=h.preprocess(X[tr],y[tr]);xt=torch.tensor(sc.transform(X[tr]),dtype=torch.float32);yt=torch.tensor((y[tr]-ym)/ys,dtype=torch.float32);bt=torch.tensor(bins,dtype=torch.long);xv=torch.tensor(sc.transform(X[va]),dtype=torch.float32);yv=torch.tensor((y[va]-ym)/ys,dtype=torch.float32);bv=torch.tensor(np.searchsorted(edges,y[va],side='right'),dtype=torch.long);train_predictions=[];rec={'fold':fold,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va],'models':{}};fd=out/f'fold-{fold}';fd.mkdir()
  for base in SEEDS:
   if base==SEEDS[0]:
    path=e.OUT/split/f'fold-{fold}'/'regression_only.pt';assert fv.sha(path)==oldreport['folds'][fold]['models']['regression_only']['checkpoint_sha256'];a=torch.load(_repo_io(path),map_location='cpu',weights_only=True);model=restore(a);r=oldreport['folds'][fold]['models']['regression_only'].copy();r['reused']=True
   else:
    model,r=h.fit(xt,yt,bt,base+fold,0.,validation=(xv,yv,bv));path=fd/f'seed-{base}.pt';a={'model':model.state_dict(),'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'target_mean':ym,'target_std':ys,'bin_edges':torch.tensor(edges),'train_ids':rec['train_ids'],'validation_ids':rec['validation_ids'],'config':h.CONFIG,'seed':base+fold,'seed_base':base,'aux_weight':0.};torch.save(a,_repo_io(path));r['reused']=False
   assert a['train_ids']==rec['train_ids'] and a['validation_ids']==rec['validation_ids'];np.testing.assert_array_equal(a['scaler_mean'].numpy(),sc.mean_);np.testing.assert_array_equal(a['scaler_scale'].numpy(),sc.scale_);assert a['target_mean']==ym and a['target_std']==ys and a['seed']==base+fold and a['aux_weight']==0.
   pred[base][va]=h.predict(model,xv,ym,ys);tp=h.predict(model,xt,ym,ys);train_predictions.append(tp);fresh=restore(torch.load(_repo_io(path),map_location='cpu',weights_only=True));err=float(abs(h.predict(fresh,xv,ym,ys)-pred[base][va]).max());assert err==0
   r.update(checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=fv.sha(path),reload_max_error=err,train_rmse=fv.rmse(y[tr],tp),validation_rmse=fv.rmse(y[va],pred[base][va]));rec['models'][str(base)]=r
  pm=np.mean(train_predictions,axis=0);train_errors.extend((pm-y[tr]).tolist());rec['mean3_train_rmse']=fv.rmse(y[tr],pm);fv.write(fd/'report.json',rec);logs.append(rec);print(json.dumps({'split':split,'fold':fold,'mean3_train_rmse':rec['mean3_train_rmse'],'seconds':time.perf_counter()-t}),flush=True)
 single=np.stack([pred[s] for s in SEEDS]);mean=single.mean(0);new=.75*ref003+.25*mean;seed_blends={str(s):.75*ref003+.25*pred[s] for s in SEEDS};fg=[fv.rmse(y[fid==k],ref006[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];gain=fv.rmse(y,ref006)-fv.rmse(y,new);delta=(y-ref006)**2-(y-new)**2;benefit_order=np.argsort(-delta,kind='stable');stress=[]
 for n in (1,5,10):
  keep=np.ones(400,dtype=bool);keep[benefit_order[:n]]=False;stress.append({'removed_n':n,'removed_ids':[ids[j] for j in benefit_order[:n]],'remaining_rmse_gain':fv.rmse(y[keep],ref006[keep])-fv.rmse(y[keep],new[keep]),'removed_fraction_of_net_squared_error_gain':float(delta[benefit_order[:n]].sum()/delta.sum()) if abs(delta.sum())>1e-12 else None,'interpretation':'Worst-case removalchosenfromobservedbenefit,diagnosticonly;notunbiasednewvalidation'})
 ix=np.random.default_rng(20261220).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-ref006)[ix]**2,1))-np.sqrt(np.mean((y-new)[ix]**2,1));seedstats={str(s):{'standalone_rmse':fv.rmse(y,pred[s]),'blend_rmse':fv.rmse(y,seed_blends[str(s)]),'blend_fold_rmse':[fv.rmse(y[fid==k],seed_blends[str(s)][fid==k]) for k in range(5)]} for s in SEEDS}
 report={'split':split,'seed_scores':seedstats,'reference006_rmse':fv.rmse(y,ref006),'mean3_standalone_rmse':fv.rmse(y,mean),'mean3_blend_rmse':fv.rmse(y,new),'mean3_train_rmse_pooled':float(np.sqrt(np.mean(np.array(train_errors)**2))),'comparison':{'gain':gain,'fold_gains':fg,'positive_folds':sum(v>0 for v in fg),'conditional_ci95':np.percentile(boot,[2.5,97.5]).tolist()},'residual_correlations':{'to003_each_seed':[float(np.corrcoef(y-ref003,y-pred[s])[0,1]) for s in SEEDS],'to003_mean3':float(np.corrcoef(y-ref003,y-mean)[0,1]),'between_seeds':np.corrcoef(y[None]-single).tolist()},'prediction_seed_std_quantiles':np.quantile(single.std(0),[0,.5,.9,1]).tolist(),'influence_stress':stress,'sample_squared_error_improvements':int((delta>0).sum()),'folds':logs,'seconds':time.perf_counter()-t,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'all_seeds_retained':True,'new_candidate':None}
 with (out/'development_oof.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','hardness','fold','reference003','reference006',*[f'seed_{s}' for s in SEEDS],'mean3','candidate007_design']);w.writerows([ident,y[j],fid[j],ref003[j],ref006[j],*[pred[s][j] for s in SEEDS],mean[j],new[j]] for j,ident in enumerate(ids))
 fv.write(out/'report.json',report);check();print(json.dumps({k:v for k,v in report.items() if k!='folds'}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare',*SPLITS]);a=ap.parse_args();prepare() if a.action=='prepare' else run(a.action)
