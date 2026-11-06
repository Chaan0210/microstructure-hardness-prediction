"""Exploratory fixed raw/TV prediction mean after071 standalone replacement failed."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,time,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import dinob_denoised_071 as base
import numpy as np,torch
prior=base.prior;fv=base.fv;h=base.h;SEEDS=base.SEEDS;b=base.b
OUT=_repo_path(ROOT, 'artifacts/denoise-complement-072')
inputs=base.inputs

def check():
 base.check();p=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for path,digest in {**p['sources'],**p['inputs'],**p['protected']}.items():assert fv.sha(_repo_path(ROOT, path))==digest,path
 return p

def prepare():
 base.check();v=json.loads((_repo_path(ROOT, 'reports/dinob-denoised-071-validation.json')).read_text());assert v['passed'] and not v['eligible_for_candidate'];outcome=json.loads((base.OUT/'original/report.json').read_text());pub=json.loads((_repo_path(ROOT, 'reports/candidate008-public-submission.json')).read_text());OUT.mkdir(exist_ok=False)
 p={'hypothesis':'RawB andTVB have close standalone RMSE13.6158/13.6676 but residualcorrelation0.93467. Equalpredictionmean maycancel representation-specific errors withoutfittingextraheads orweights. It could fail in otherpartitions.',
 'posthoc_disclosure':'Newexploratoryfollowup chosenafter071 failed its primaryreplacement screen. Doesnotreverse071 gate,claimpreregistrationbefore071 results,orclaimindependentvalidation. Do not search additionalratios ifthisfails.',
 'single_primary':'.75*r003+.125*Smean+.0625*Brawmean+.0625*BTVmean = .5*008+.5*071primary. This retains totalBweight.125 and usesfixed50:50 likepriorS/Bcomplementarity064,notoptimizedweights.',
 'screen':'OriginalpooledOOF strictlyimproves008 -> train071samefixedheads onbothknownresplits andevaluatefixedmixture. Foldcount/bootstrapdiagnosticonly.',
 'confirmation':'All3pooledOOF strictlyimprove008 plusstrictverification -> lockexploratoryrecipe beforefull500/testcandidatecreation.',
 'seeds':list(SEEDS),'config':h.CONFIG,'effective_aux_weight':0.,'preprocessing':'Onlytrainingfold StandardScaler+targetscale. Frozenimage-localTVfeaturesverified071. Originalreuses15existing071models. Eachresplit fits15sameheads.',
 'sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['denoise_complement_072.py','validate_denoise_complement_072.py','dinob_denoised_071.py','dinob_mlp_head_063.py']},
 'inputs':{n:fv.sha(_repo_path(ROOT, n)) for n in ['reports/dinob-denoised-071-validation.json','artifacts/dinob-denoised-071/original/report.json','artifacts/dinob-denoised-071/original/development_predictions.npz','reports/dinob-complement-064-validation.json']},
 'protected':{**base.check()['protected'],**pub['protected_sha256']},'resource':'ExistingCPU2threads,verified071features,30additionalheadfitsonlyiforiginalimproves,noinstall/download/cost.',
 'scope':'Development400onlyuntilmodellock;noholdout/testselection.','bias':fv.CAVEAT}
 fv.write(_repo_path(OUT, 'plan.json'),p);fv.write(_repo_path(OUT, 'head-benchmark.json'),json.loads((base.OUT/'head-benchmark.json').read_text()));print(json.dumps({'prepared':True,'hypothesis':p['hypothesis'],'posthoc_disclosure':p['posthoc_disclosure']}),flush=True)

def original():
 check();ids,X,y,g,fid,r003,r007,Smean,Bmean,ref=inputs('original');src=base.OUT/'original';out=_repo_path(OUT, 'original');out.mkdir(exist_ok=False)
 with np.load(_repo_io(src/'development_predictions.npz'),allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 assert d['ids'].tolist()==ids;np.testing.assert_array_equal(d['reference008'],ref);np.testing.assert_array_equal(d['fold'],fid);np.testing.assert_array_equal(d['y'],y)
 d['primary']=.75*r003+.125*Smean+.0625*Bmean+.0625*d['Bdenoised'];report=json.loads((src/'report.json').read_text());report['scores']={name:{'rmse':fv.rmse(y,d[name]),'fold_rmse':[fv.rmse(y[fid==k],d[name][fid==k]) for k in range(5)]} for name in report['scores']};report['comparison']=prior.old.comparison(y,ref,d['primary'],fid);report['models_reused_from']=str(src.relative_to(ROOT));report['seconds']=0.
 with (out/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),**d)
 report['predictions_sha256']=fv.sha(out/'development_predictions.npz');fv.write(out/'report.json',report);fv.write(_repo_path(OUT, 'selection.json'),{'advance':bool(report['comparison']['gain']>0),'primary':'primary','comparison':report['comparison'],'diagnostic_only':'foldcounts/bootstrapnotgates'});print(json.dumps(report),flush=True)

def run(split):
 check();assert json.loads((_repo_path(OUT, 'head-benchmark.json')).read_text())['resource_pass']
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 torch.set_num_threads(2);ids,X,y,g,fid,r003,r007,Smean,Bmean,ref=inputs(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);pred=np.full((3,400),np.nan);trainerrors=[];t=time.perf_counter()
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(g[tr])&set(g[va]);sc,ym,ys,xt,yt,bt,edges=prior.old.prep(X[tr],y[tr]);fd=out/f'fold-{k}';fd.mkdir();rec={'fold':k,'models':{}};tp=[];oldlog=json.loads((_repo_path(ROOT, 'artifacts/seed-stability-052', split, f'fold-{k}/report.json')).read_text())
  for si,seed in enumerate(SEEDS):
   m,log=h.fit(xt,yt,bt,seed+k,0.);assert log['batch_order_sha256']==oldlog['models'][str(seed)]['batch_order_sha256'];path=fd/f'seed-{seed}.pt';torch.save({'model':m.state_dict(),'input_dim':6912,'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'scaler_var':torch.tensor(sc.var_),'target_mean':ym,'target_std':ys,'bin_edges_unused':torch.tensor(edges),'seed':seed+k,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va],'config':h.CONFIG,'effective_aux_weight':0.},_repo_io(path));pred[si,va]=h.predict(m,sc.transform(X[va]),ym,ys);tp.append(h.predict(m,xt,ym,ys));log.update(checkpoint_sha256=fv.sha(path),validation_rmse=fv.rmse(y[va],pred[si,va]));rec['models'][str(seed)]=log
  trainerrors.extend((np.mean(tp,axis=0)-y[tr]).tolist());fv.write(fd/'report.json',rec);print(json.dumps({'split':split,'fold':k,'seconds':time.perf_counter()-t}),flush=True)
 mean=pred.mean(0);new=.75*r003+.125*Smean+.0625*Bmean+.0625*mean;ps={'reference008':ref,'reference003':r003,'Smean':Smean,'B504mean':Bmean,'Bdenoised':mean,'primary':new};scores={n:{'rmse':fv.rmse(y,v),'fold_rmse':[fv.rmse(y[fid==k],v[fid==k]) for k in range(5)]} for n,v in ps.items()};comp=prior.old.comparison(y,ref,new,fid)
 with (out/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),y=y,fold=fid,seed_predictions=pred,**ps)
 report={'split':split,'scores':scores,'comparison':comp,'standalone_Bdenoised_B504_residual_correlation':float(np.corrcoef(y-mean,y-Bmean)[0,1]),'mean3_train_rmse_pooled':float(np.sqrt(np.mean(np.array(trainerrors)**2))),'seconds':time.perf_counter()-t,'predictions_sha256':fv.sha(out/'development_predictions.npz'),'new_candidate':None};fv.write(out/'report.json',report)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':bool(comp['gain']>0),'primary':'primary','comparison':comp,'diagnostic_only':'foldcounts and bootstrap; no hard threshold'})
 check();print(json.dumps(report),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','original','20261031','20261101']);a=ap.parse_args();{'prepare':prepare,'original':original}.get(a.mode,lambda:run(a.mode))()
