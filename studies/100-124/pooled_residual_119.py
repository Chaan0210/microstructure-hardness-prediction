"""Fixed rbf1 block averaging versus one pooled320 crossfit-residual model."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,datetime,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import residual_subsamples_118 as prior
import nonlinear_residual_115 as n
import numpy as np,joblib
e=n.e;r=n.r;OUT=_repo_path(ROOT, 'artifacts/pooled-residual-119')

def prepare():
 prior.check();proposal=e.read(_repo_path(ROOT, 'reports/pooled-residual-119-proposed-study.json'));OUT.mkdir(exist_ok=False);protected=dict(prior.check()['protected'])
 for name in ['reports/residual-subsamples-118-summary.json','reports/residual-subsamples-118-preservation.json','reports/pooled-residual-119-proposed-study.json']:protected[name]=e.sha(_repo_path(ROOT, name))
 anchors=[]
 for split in e.SPLITS:
  d=r.load(split)
  for k in range(5):
   for j in range(3):A,B,pa,trace=r.anchor(d,split,k,j,True);anchors.append({'split':split,'fold':k,'block':j,**trace})
 e.write(_repo_path(OUT, 'anchor-audit.json'),{'passed':True,'anchors':anchors,'all_model_hashes_rechecked':True})
 plan={'study':'119 fixedrbf1 pooled320 residual comparison','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':'Parentexplicitlyapprovedexecutionofthe119proposal andfull500/candidateCSVifprespecifiedimprovement.','proposal':proposal,'recipe':'Noinnerselection. n.fit(kind=rbf1) unchanged. EachouterT320 usesr_i=y_i-f_Aj(i) fori inBj,Aj=TminusBj. Blockarmfits3models respectivelyBj106/107 andaveragescorrections. Pooledarmfitsone onallT320. Bothaddtounchanged011_outer320.','primary':'PooleduntrimmedMSE onall3partitions. Pooledarm muststrictlybeatblockmean,115and011 plusindependentverification forcandidate. No perpartition/fold/CI/influence-exclusiongate.','ensemble_diagnostic':'Saveall3individualblockpredictions. Exactdecomposition: mean_j MSE(p_j)=MSE(mean_jp_j)+mean_ij(p_ij-mean_jp_ij)^2. Diagnosticindividualerrornotacandidate;pooledversusblockmeanremainsprimary. No newweightsoradditionalmodelvariants.','limits':'Poolingalsochangestrainlocalpreprocessing/gamma/kernelcentering;contrastiswholedata-usealgorithm,notpurecausaln. Sharedanchortrainingcorrelatespseudo-targets.213/214anchorversus320base mismatch,333/334versus500production;repeated400selectionbias remains.','verification':'45anchorlineagesandallmodelhashesrechecked;oneOOFtargetpereachTrow,ownlabelandouterexclusion. Independentrawstatisticsandkernel eigensolve;joblibreload;outerlabelperturbation rerunentireresidualfold. Upstream112retrainingcertificate reused.','resource':proposal['resources'],'sources':{x:e.sha(_repo_path(ROOT, x)) for x in ['pooled_residual_119.py','validate_pooled_residual_119.py','nonlinear_residual_115.py']},'protected':protected,'anchor_audit_sha256':e.sha(_repo_path(OUT, 'anchor-audit.json')),'environment':prior.check()['environment'],'scope':'Development400beforedecision;test/holdoutnotdecodedforselection,nonewdata/install/cost/externaltransmission/upload.'};e.write(_repo_path(OUT, 'plan.json'),plan);e.emit({'prepared':True,'anchors_verified':45,'plan_sha256':e.sha(_repo_path(OUT, 'plan.json'))})

def check():
 prior.check();p=e.read(_repo_path(OUT, 'plan.json'))
 for x,h in {**p['sources'],**p['protected']}.items():assert e.sha(_repo_path(ROOT, x))==h,x
 assert e.sha(_repo_path(OUT, 'anchor-audit.json'))==p['anchor_audit_sha256'];return p

def targets(d,split,k):
 T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);pa=np.full(len(T),np.nan);index={int(i):j for j,i in enumerate(T)};seen=[];records=[]
 for j in range(3):
  A,B,p,trace=r.anchor(d,split,k,j);assert set(A)|set(B)==set(T) and not (set(A)|set(B))&set(V);where=[index[int(i)] for i in B];pa[where]=p;seen.extend(B.tolist());records.append({'A':A,'B':B,'p':p,'trace':trace})
 assert sorted(seen)==T.tolist() and np.isfinite(pa).all();return T,V,pa,d['y'][T]-pa,records

def fit_fold(d,split,k,path):
 path.mkdir(exist_ok=False);start=time.perf_counter();T,V,pa,res,records=targets(d,split,k);small=[];models={};metas=[]
 for j,record in enumerate(records):
  B=record['B'];rr=d['y'][B]-record['p'];m=n.fit(d['X'][B],rr,d['blocks'],'rbf1');pp=n.predict(m,d['X'][V]);file=path/f'block-{j}.joblib';joblib.dump({'model':m,'train_indices':B,'validation_indices':V},_repo_io(file));models[file.name]=e.sha(file);small.append(pp);metas.append({'n_train':len(B),'effective_df':m['effective_df'],'gamma':m['gamma'],'alpha':m['alpha'],'residual_scale':m['residual_scale']})
 m=n.fit(d['X'][T],res,d['blocks'],'rbf1');pooled=n.predict(m,d['X'][V]);file=path/'pooled.joblib';joblib.dump({'model':m,'train_indices':T,'validation_indices':V},_repo_io(file));models[file.name]=e.sha(file);small=np.stack(small)
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=T,validation_indices=V,anchor_predictions=pa,residual=res,block_corrections=small,block_mean_correction=small.mean(0),pooled_correction=pooled)
 report={'fold':k,'block_anchor_traces':[x['trace'] for x in records],'block_models':metas,'pooled_model':{'n_train':len(T),'effective_df':m['effective_df'],'gamma':m['gamma'],'alpha':m['alpha'],'residual_scale':m['residual_scale']},'models_sha256':models,'predictions_sha256':e.sha(path/'predictions.npz'),'seconds':time.perf_counter()-start};e.write(path/'report.json',report);return small,pooled,report

def run(split):
 check();d=r.load(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);start=time.perf_counter();small=np.full((3,400),np.nan);pooled=np.full(400,np.nan)
 for k in range(5):
  V=np.flatnonzero(d['fid']==k);small[:,V],pooled[V],rec=fit_fold(d,split,k,out/f'fold-{k}')
  if split=='original' and k==0:
   b={'first_fold_seconds_reused':rec['seconds'],'projected15_seconds':15*rec['seconds'],'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};b['passed']=b['projected15_seconds']<120 and b['RSS_bytes']<2e9;e.write(_repo_path(OUT, 'benchmark.json'),b);assert b['passed']
  e.emit({'split':split,'fold_complete':k,'seconds':time.perf_counter()-start})
 source=e.read(n.OUT/split/'report.json');assert e.sha(n.OUT/split/'predictions.npz')==source['predictions_sha256']
 with np.load(_repo_io(n.OUT/split/'predictions.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==d['ids'];reference115=z['primary'].copy()
 ps={'reference011':d['reference011'],'reference115':reference115,'block_mean':d['reference011']+small.mean(0),'pooled':d['reference011']+pooled};ind=d['reference011'][None,:]+small
 with (out/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],block_individual=ind,block_corrections=small,pooled_correction=pooled,**ps)
 avg=float(np.mean((d['y'][None,:]-ind)**2));ens=float(np.mean((d['y']-ps['block_mean'])**2));div=float(np.mean((ind-ind.mean(0))**2));assert abs(avg-ens-div)<1e-11
 report={'split':split,'scores':{key:{'MSE':float(np.mean((d['y']-p)**2)),'RMSE':r.rmse(d['y'],p)} for key,p in ps.items()},'ensemble_decomposition':{'mean_individual_MSE_diagnostic':avg,'block_mean_MSE':ens,'averaging_MSE_reduction_exact':div,'mean_individual_minus_pooled_MSE':avg-float(np.mean((d['y']-ps['pooled'])**2))},'predictions_sha256':e.sha(out/'predictions.npz'),'fold_reports_sha256':{f'fold-{k}/report.json':e.sha(out/f'fold-{k}/report.json') for k in range(5)},'seconds':time.perf_counter()-start};e.write(out/'report.json',report);check();e.emit({'split':split,'scores':report['scores'],'ensemble':report['ensemble_decomposition']})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare',*e.SPLITS]);x=p.parse_args();prepare() if x.mode=='prepare' else run(x.mode)
