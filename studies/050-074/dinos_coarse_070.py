"""Fixed coarse spatial heterogeneity with S descriptor, unchanged008 recipe."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,time,resource,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import numpy as np,torch
import dinob_062 as b
import dinob_complement_064 as prior
fv=prior.fv;h=prior.h;SEEDS=prior.SEEDS;OUT=_repo_path(ROOT, 'artifacts/dinos-coarse-070')

def check():
 prior.check();b.check();p=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for name,digest in {**p['sources'],**p['inputs'],**p['protected']}.items():assert fv.sha(_repo_path(ROOT, name))==digest,name
 return p

def coarse(p):
 p=np.asarray(p,dtype=np.float64);assert p.shape==(36,36,384);var=p.var((0,1));parts=[]
 for grid in (2,4):
  size=36//grid;means=p.reshape(grid,size,grid,size,384).mean((1,3));parts.append(means.var((0,1))/(var+1e-4))
 return np.concatenate(parts)

def token_inputs():
 from dinob_062 import streamsha
 ex=json.loads((_repo_path(ROOT, 'artifacts/fisher-067/extraction.json')).read_text());v=json.loads((_repo_path(ROOT, 'reports/fisher-067-validation.json')).read_text());assert v['passed'] and v['tokens_reextracted']==400 and v['token_max_error']==0
 path=_repo_path(ROOT, 'artifacts/fisher-067/development_tokens.npy');assert streamsha(path)==ex['tokens_sha256'];ids,_=b.ds.base.scope();assert ex['ids']==ids
 return ids,np.load(_repo_io(path),mmap_mode='r'),ex

def prepare():
 prior.check();b.check();pub=json.loads((_repo_path(ROOT, 'reports/candidate008-public-submission.json')).read_text());old=prior.check();assert not json.loads((_repo_path(ROOT, 'artifacts/dinob-prenorm-069/selection.json')).read_text())['advance'];OUT.mkdir(exist_ok=False)
 p={'hypothesis':'068absoluteerrorsweaklyassociatewithcoarseautocorrelation andtilebrightnessvariation. Globaltokenmean/std andorderlessFisher discardspatialsegregation. Fixedcoarseregionalvariancefraction mayaddinformationaboutlongrangepatchheterogeneity.',
 'limits_of_evidence':'Correlations~0.13 aredescriptive,multipleinspected,notcausal. Negative069meansunnormalizedamplitudesdidnothelpthisrecipe,notallamplitudeinformationuseless.',
 'single_primary':'Append2x384 descriptors toexistingS3456: variance ofregionalmeans on2x2 and4x4 nonoverlappinggrids,dividedbyglobalpatchvariance+1e-4. FinalnormalizedS504tokens;4224dim input,same3seedMLP. Formula .75r003+.125Scoarsemean+.125Bmean.',
 'duplicate_check':'042lags1/4 pairwisesquaredifferences testlocalcoherence(7/28nativepx); this integrates2Dregions~128/64nativepx. 056covariance and067Fisher orderless. 066changesresolution ratherthan retainingregionalspatialvariation. No learnedcoordinates orabsolutequadrantidentity.',
 'risk':'Adds768inputs, changesinitializationandparametercount; representationexperimentnotcapacitymatched. D4 invariance ofcachedtokenmap descriptors doesnot implyDINOinputrotationinvariance.',
 'seeds':list(SEEDS),'config':h.CONFIG,'effective_aux_weight':0.,'preprocessing':'Descriptorhasno learnedfit;StandardScalerandtargetscale outertrainonly.',
 'screen':'OriginalprimarypooledOOF strictlyimproves008 -> bothknownresplits; otherwise reject. No weight/grid/epsilonsearch.',
 'confirmation':'All3pooledOOF improve008+strictreload/refitvalidation. Foldcount/bootstrapdiagnosticonly.',
 'resource':'Reuseverified796MBmemmaptokens;CPU2threads,15fitsprojected<300s,RSS<2GB,noinstall/download/cost.',
 'sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['dinos_coarse_070.py','validate_dinos_coarse_070.py','dinob_mlp_head_063.py','dinob_mlp_063.py','dinob_complement_064.py']},
 'inputs':{**old['inputs'],**{n:fv.sha(_repo_path(ROOT, n)) for n in ['reports/dinob-complement-064-validation.json','artifacts/error-diagnostics-068/report.json','artifacts/fisher-067/extraction.json','reports/fisher-067-validation.json','artifacts/dinob-prenorm-069/selection.json']}},
 'protected':{**old['protected'],**pub['protected_sha256'],'reports/candidate008-public-submission.json':fv.sha(_repo_path(ROOT, 'reports/candidate008-public-submission.json'))},
 'scope':'Onlydevelopment400;holdout/testnotreadforsel;protectedfinalcacheshashedonly.','bias':fv.CAVEAT}
 fv.write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'hypothesis':p['hypothesis']}),flush=True)

def extract():
 check();ids,tokens,source=token_inputs();start=time.perf_counter();X=np.stack([coarse(t.reshape(36,36,384)) for t in tokens]);assert X.shape==(400,768) and np.isfinite(X).all() and X.min()>=0 and X.max()<=1+1e-8
 rng=np.random.default_rng(20270108);p=rng.normal(size=(36,36,384));v=coarse(p)
 for k in range(4):
  np.testing.assert_allclose(v,coarse(np.rot90(p,k)),atol=1e-14,rtol=1e-12);np.testing.assert_allclose(v,coarse(np.rot90(p,k)[:,::-1]),atol=1e-14,rtol=1e-12)
 np.testing.assert_array_equal(coarse(np.zeros_like(p)),np.zeros(768));step=np.zeros_like(p);step[:18]=1;checker=np.broadcast_to((np.indices((36,36)).sum(0)%2)[:,:,None],p.shape);assert np.max(abs(coarse(step)-coarse(checker)))>.9;np.testing.assert_array_equal(step.mean((0,1)),checker.mean((0,1)));np.testing.assert_array_equal(step.std((0,1)),checker.std((0,1)))
 with (_repo_path(OUT, 'development_Scoarse.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),X=X)
 report={'shape':list(X.shape),'seconds':time.perf_counter()-start,'feature_sha256':fv.sha(_repo_path(OUT, 'development_Scoarse.npz')),'token_sha256':source['tokens_sha256'],'fixtures':{'tokenmapD4':True,'zero':True,'sameglobalstatsdifferentcoarsestructure':True},'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};fv.write(_repo_path(OUT, 'extraction.json'),report);print(json.dumps(report),flush=True)

def inputs(split):
 ids,S,B,y,g,fid,r003,r007,Smean=prior.old.inputs(split);ex=json.loads((_repo_path(OUT, 'extraction.json')).read_text());assert fv.sha(_repo_path(OUT, 'development_Scoarse.npz'))==ex['feature_sha256']
 with np.load(_repo_path(OUT, 'development_Scoarse.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;X=np.concatenate([S,z['X'].astype(float)],axis=1)
 v=json.loads((_repo_path(ROOT, 'reports/dinob-complement-064-validation.json')).read_text());path=prior.OUT/split/'development_predictions.npz';assert fv.sha(path)==v['results'][split]['predictions_sha256']
 with np.load(_repo_io(path),allow_pickle=False) as z:
  assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['y'],y);np.testing.assert_array_equal(z['fold'],fid);np.testing.assert_array_equal(z['reference003'],r003);np.testing.assert_array_equal(z['reference007'],r007);np.testing.assert_array_equal(z['Smean'],Smean);ref=z['mixed50'].copy();Bmean=z['Bmean'].copy()
 np.testing.assert_allclose(ref,.75*r003+.125*Smean+.125*Bmean,atol=1e-12,rtol=0);assert X.shape==(400,4224) and np.isfinite(X).all();return ids,X,y,g,fid,r003,r007,Smean,Bmean,ref

def benchmark_head():
 check();torch.set_num_threads(2);ids,X,y,g,fid,*_=inputs('original');tr=np.flatnonzero(fid!=0);sc,ym,ys,xt,yt,bt,edges=prior.old.prep(X[tr],y[tr]);t=time.perf_counter();m,rec=h.fit(xt,yt,bt,SEEDS[0],0.);seconds=time.perf_counter()-t;old=json.loads((_repo_path(ROOT, 'artifacts/dinob-mlp-063/original/fold-0/report.json')).read_text())['models'][f'baseB-{SEEDS[0]}'];assert rec['batch_order_sha256']==old['batch_order_sha256'];report={'seconds':seconds,'projected15fits_seconds':seconds*15,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'parameters':sum(v.numel() for v in m.parameters()),'order_matched_B504':True};report['resource_pass']=bool(report['projected15fits_seconds']<300 and report['rss_peak_bytes']<2e9);fv.write(_repo_path(OUT, 'head-benchmark.json'),report);print(json.dumps(report),flush=True)

def run(split):
 check();assert json.loads((_repo_path(OUT, 'head-benchmark.json')).read_text())['resource_pass']
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 torch.set_num_threads(2);ids,X,y,g,fid,r003,r007,Smean,Bmean,ref=inputs(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);pred=np.full((3,400),np.nan);trainerrors=[];t=time.perf_counter()
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(g[tr])&set(g[va]);sc,ym,ys,xt,yt,bt,edges=prior.old.prep(X[tr],y[tr]);fd=out/f'fold-{k}';fd.mkdir();rec={'fold':k,'models':{}};tp=[];oldlog=json.loads((_repo_path(ROOT, 'artifacts/seed-stability-052', split, f'fold-{k}/report.json')).read_text())
  for si,seed in enumerate(SEEDS):
   m,log=h.fit(xt,yt,bt,seed+k,0.);assert log['batch_order_sha256']==oldlog['models'][str(seed)]['batch_order_sha256'];path=fd/f'seed-{seed}.pt';torch.save({'model':m.state_dict(),'input_dim':4224,'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'scaler_var':torch.tensor(sc.var_),'target_mean':ym,'target_std':ys,'bin_edges_unused':torch.tensor(edges),'seed':seed+k,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va],'config':h.CONFIG,'effective_aux_weight':0.},_repo_io(path));pred[si,va]=h.predict(m,sc.transform(X[va]),ym,ys);tp.append(h.predict(m,xt,ym,ys));log.update(checkpoint_sha256=fv.sha(path),validation_rmse=fv.rmse(y[va],pred[si,va]));rec['models'][str(seed)]=log
  trainerrors.extend((np.mean(tp,axis=0)-y[tr]).tolist());fv.write(fd/'report.json',rec);print(json.dumps({'split':split,'fold':k,'seconds':time.perf_counter()-t}),flush=True)
 mean=pred.mean(0);new=.75*r003+.125*mean+.125*Bmean;ps={'reference008':ref,'reference003':r003,'Smean':Smean,'B504mean':Bmean,'Scoarse':mean,'primary':new};scores={n:{'rmse':fv.rmse(y,v),'fold_rmse':[fv.rmse(y[fid==k],v[fid==k]) for k in range(5)]} for n,v in ps.items()};comp=prior.old.comparison(y,ref,new,fid)
 with (out/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),y=y,fold=fid,seed_predictions=pred,**ps)
 report={'split':split,'scores':scores,'comparison':comp,'standalone_Scoarse_S504_residual_correlation':float(np.corrcoef(y-mean,y-Smean)[0,1]),'mean3_train_rmse_pooled':float(np.sqrt(np.mean(np.array(trainerrors)**2))),'seconds':time.perf_counter()-t,'predictions_sha256':fv.sha(out/'development_predictions.npz'),'new_candidate':None};fv.write(out/'report.json',report)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':bool(comp['gain']>0),'primary':'primary','comparison':comp,'diagnostic_only':'foldcounts and bootstrap; no hard threshold'})
 check();print(json.dumps(report),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','extract','benchmark_head','original','20261031','20261101']);a=ap.parse_args();{'prepare':prepare,'extract':extract,'benchmark_head':benchmark_head}.get(a.mode,lambda:run(a.mode))()
