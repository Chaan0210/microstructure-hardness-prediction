"""177 fixed B/L MLP and nested PCA/SVR comparison on development400 only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='2'
import sys,time,json,argparse,hashlib
from pathlib import Path
import numpy as np,torch,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import features as f,mlp_head as h
R=f.R;P=f.P;SPLITS=['original','20261031','20261101'];SEEDS=[20261128,20261205,20261219];PCS=[16,34,64];CS=[1.,10.,100.]
threadpool_limits(2);torch.set_num_threads(2)
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
def load(p):
 with np.load(_repo_io(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def arrays(path,**kwargs):
 with path.open('xb') as ff:np.savez_compressed(_repo_io(ff),**kwargs)
def check():
 plan=f.read(R/'plan.json')
 for name,hh in plan['inputs'].items():assert f.sha(P/name)==hh,name
 for name,hh in f.read(R/'readout-code-lock.json')['sources'].items():assert f.sha(R/name)==hh,name
 e=f.read(R/'extraction.json');assert e['passed'] and f.sha(R/'features.npz')==e['feature_sha256']
 return plan

def data(split):
 plan=check();z=load(P/'artifacts/run-012/development_features.npz');b=load(P/'artifacts/dinob-062/development_features.npz');ll=load(R/'features.npz');base=load(P/f'artifacts/bootstrap-selector-144/{split}/predictions.npz');control=load(P/f'artifacts/dinob-complement-064/{split}/development_predictions.npz');cert=f.read(P/f'artifacts/bootstrap-selector-144/validation-{split}.json');assert cert['passed'];assert z['ids'].tolist()==plan['IDs'];assert z['groups'].tolist()==plan['groups']
 for o in [b,ll,base,control]:np.testing.assert_array_equal(z['ids'],o['ids'])
 for o in [base,control]:np.testing.assert_array_equal(z['y'],o['y'])
 np.testing.assert_array_equal(base['fold'],control['fold']);d={'ids':z['ids'],'y':z['y'],'g':z['groups'],'fold':base['fold'],'reference':base['primary'],'B':b['X'].astype(float),'L':ll['X'].astype(float),'old_B_seeds':control['Bseeds']};assert len(d['y'])==400
 for k in range(5):assert not set(d['g'][d['fold']==k])&set(d['g'][d['fold']!=k])
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   pp=Path(os.fsdecode(args[0])).absolute()
   if pp.is_relative_to(P/'data'):raise RuntimeError('Readout rawdata/holdout/test access forbidden')
 sys.addaudithook(guard);return d

def mlp(d,arm,T,V,k,path):
 X=d[arm];y=d['y'];sc,ym,ys,_,bins=h.preprocess(X[T],y[T]);xt=torch.tensor(sc.transform(X[T]),dtype=torch.float32);yt=torch.tensor((y[T]-ym)/ys,dtype=torch.float32);bt=torch.tensor(bins);pred=[];logs=[]
 for seed in SEEDS:
  m,log=h.fit(xt,yt,bt,seed+k,0.);pp=h.predict(m,sc.transform(X[V]),ym,ys);pred.append(pp);obj={'model':m.state_dict(),'input_dim':X.shape[1],'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'scaler_var':torch.tensor(sc.var_),'target_mean':ym,'target_std':ys,'train':T,'validation':V,'seed':seed+k};file=path/f'{arm}-MLP-{seed}.pt';torch.save(obj,_repo_io(file));log.update(file=file.name,sha256=f.sha(file),train_rmse=rmse(y[T],h.predict(m,sc.transform(X[T]),ym,ys)),validation_rmse=rmse(y[V],pp));logs.append(log);del m
 pred=np.stack(pred)
 if arm=='B':np.testing.assert_allclose(pred,d['old_B_seeds'][:,V],atol=1e-8,rtol=0)
 return pred,logs

def cls(d,arm):
 width=768 if arm=='B' else 1024
 return d[arm][:,6*width:7*width]
def pca_transform(X,y,T,V,pc):
 sc=StandardScaler().fit(X[T]);pca=PCA(n_components=pc,svd_solver='full',whiten=False).fit(sc.transform(X[T]));a=pca.transform(sc.transform(X[T]));b=pca.transform(sc.transform(X[V]));ym=float(y[T].mean());ys=float(y[T].std());assert ys>0;return {'scaler':sc,'PCA':pca,'target_mean':ym,'target_std':ys,'train':T,'validation':V},a,b,(y[T]-ym)/ys

def svr(d,arm,T,V,k,split,path):
 X=cls(d,arm);y=d['y'];seed=202710177+10000*SPLITS.index(split)+k;folds=list(GroupKFold(3,shuffle=True,random_state=seed).split(X[T],y[T],d['g'][T]));ip=np.full((len(PCS)*len(CS),len(T)),np.nan);files=[]
 for j,(I,H) in enumerate(folds):
  U,Q=T[I],T[H];assert not set(d['g'][U])&set(d['g'][Q]);assert not (set(U)|set(Q))&set(V)
  for pi,pc in enumerate(PCS):
   obj,a,b,yy=pca_transform(X,y,U,Q,pc);models={}
   for ci,C in enumerate(CS):
    m=SVR(C=C,epsilon=.1,kernel='rbf',gamma='scale',cache_size=128,tol=.001).fit(a,yy);assert m.fit_status_==0;ip[pi*len(CS)+ci,H]=m.predict(b)*obj['target_std']+obj['target_mean'];models[str(C)]=m
   obj['models']=models;name=f'{arm}-SVR-inner-{j}-pc-{pc}.joblib';joblib.dump(obj,_repo_io(path/name));files.append(name)
 scores=np.sqrt(np.mean((ip-y[T])**2,axis=1));winner=int(np.argmin(scores));pc=PCS[winner//len(CS)];C=CS[winner%len(CS)];obj,a,b,yy=pca_transform(X,y,T,V,pc);m=SVR(C=C,epsilon=.1,kernel='rbf',gamma='scale',cache_size=128,tol=.001).fit(a,yy);assert m.fit_status_==0;pred=m.predict(b)*obj['target_std']+obj['target_mean'];obj.update(model=m,choice={'PCA':pc,'C':C,'epsilon':.1,'gamma':'scale'});name=f'{arm}-SVR-final.joblib';joblib.dump(obj,_repo_io(path/name));files.append(name);arrays(path/f'{arm}-SVR-inner.npz',T=T,V=V,predictions=ip,scores=scores)
 return pred,{'choice':obj['choice'],'inner_RMSE':scores.tolist(),'chosen_index':winner,'seed':seed,'inner_splits':[{'train':T[I].tolist(),'validation':T[H].tolist()} for I,H in folds],'files':{name:f.sha(path/name) for name in files}}

def comparison(y,ref,new,fid,g):
 fg=[rmse(y[fid==k],ref[fid==k])-rmse(y[fid==k],new[fid==k]) for k in range(5)];unique,inv=np.unique(g,return_inverse=True);rng=np.random.default_rng(202710178);boot=[]
 for _ in range(3000):
  sampled=rng.integers(0,len(unique),len(unique));counts=np.bincount(sampled,minlength=len(unique))[inv];boot.append(float(np.sqrt(counts@((y-ref)**2)/counts.sum())-np.sqrt(counts@((y-new)**2)/counts.sum())))
 return {'gain':rmse(y,ref)-rmse(y,new),'fold_gains':fg,'positive_folds':sum(x>0 for x in fg),'conditional_cluster_bootstrap_CI95':np.quantile(boot,[.025,.975]).tolist(),'CI_not_adjusted_for_model_selection_or_refitting':True}

def run(split):
 plan=check()
 if split!='original':assert f.read(R/'original/validation.json')['passed'] and f.read(R/'original/report.json')['eligible_L_readouts']
 d=data(split);out=R/split;out.mkdir(exist_ok=False);pred={a:np.full((3,400),np.nan) for a in ['B_MLP','L_MLP']};pred.update({a:np.full(400,np.nan) for a in ['B_SVR','L_SVR']});start=time.perf_counter();foldreports=[]
 eligible=['MLP','SVR'] if split=='original' else f.read(R/'original/report.json')['eligible_L_readouts']
 with f.Monitor(split,plan['resources']):
  for k in range(5):
   T=np.flatnonzero(d['fold']!=k);V=np.flatnonzero(d['fold']==k);path=out/f'fold-{k}';path.mkdir();r={'fold':k,'T':T.tolist(),'V':V.tolist(),'models':{}}
   for arm in ['B','L']:
    if 'MLP' in eligible:pred[f'{arm}_MLP'][:,V],r['models'][f'{arm}_MLP']=mlp(d,arm,T,V,k,path)
    if 'SVR' in eligible:pred[f'{arm}_SVR'][V],r['models'][f'{arm}_SVR']=svr(d,arm,T,V,k,split,path)
   f.write(path/'report.json',r);foldreports.append(r);f.emit({'split':split,'fold_complete':k,'seconds':time.perf_counter()-start})
  result={'reference018':d['reference']}
  for rd in eligible:
   for arm in ['B','L']:
    pp=pred[f'{arm}_{rd}'];result[f'{arm}_{rd}']=pp.mean(0) if rd=='MLP' else pp;result[f'blend_{arm}_{rd}']=.9*d['reference']+.1*result[f'{arm}_{rd}']
  assert all(np.isfinite(v).all() for v in result.values());scores={a:rmse(d['y'],v) for a,v in result.items()};diag={};advance=[]
  for rd in eligible:
   lp=result[f'L_{rd}'];bp=result[f'B_{rd}'];mix=result[f'blend_L_{rd}'];diag[rd]={**comparison(d['y'],d['reference'],mix,d['fold'],d['g']),'L_B_error_correlation':float(np.corrcoef(d['y']-lp,d['y']-bp)[0,1]),'L_018_error_correlation':float(np.corrcoef(d['y']-lp,d['y']-d['reference'])[0,1]),'gain_vs_matched_B_blend':scores[f'blend_B_{rd}']-scores[f'blend_L_{rd}']}
   if scores[f'blend_L_{rd}']<scores['reference018'] and scores[f'blend_L_{rd}']<scores[f'blend_B_{rd}']:advance.append(rd)
  seedarrays={key+'_seeds':v for key,v in pred.items() if key.endswith('MLP') and key.split('_')[1] in eligible}
  arrays(out/'predictions.npz',ids=d['ids'],y=d['y'],g=d['g'],fold=d['fold'],**result,**seedarrays)
 rec={'split':split,'RMSE':scores,'diagnostics':diag,'eligible_L_readouts':advance,'readouts_executed':eligible,'seconds':time.perf_counter()-start,'B_MLP_exact_historical_control':'MLP' in eligible,'fixed_mixture':[.9,.1],'fold_report_hashes':{f'fold-{k}/report.json':f.sha(out/f'fold-{k}/report.json') for k in range(5)},'predictions_sha256':f.sha(out/'predictions.npz'),'plan_sha256':f.sha(R/'plan.json'),'code_lock_sha256':f.sha(R/'readout-code-lock.json'),'no_holdout_test_full500':True};f.write(out/'report.json',rec);f.emit(rec)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('split',choices=SPLITS);args=ap.parse_args()
 try:run(args.split)
 except Exception as e:f.write(R/f'readout-{args.split}-failure.json',{'error':repr(e),'type':type(e).__name__});raise
