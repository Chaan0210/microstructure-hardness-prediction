"""180 paired fixed-capacity joint SB/SBL PCA64 MLP on development caches only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse,sys,os,time,hashlib
import numpy as np,torch,joblib
from torch import nn
from scipy.special import erf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import runtime as f
R=f.R;P=f.P;SPLITS=['original','20261031','20261101'];ARMS={'SB':['S','B'],'SBL':['S','B','L']};SEEDS=[20261128,20261205,20261219]

def arrays(p,**z):
 with p.open('xb') as ff:np.savez_compressed(_repo_io(ff),**z)
def load(p):
 with np.load(_repo_io(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
def statehash(m):
 h=hashlib.sha256()
 for n,t in sorted(m.state_dict().items()):h.update(n.encode());h.update(t.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def check():return f.check()
def data(split):
 plan=check();path={'S':'artifacts/dino-multilayer-044/development_features.npz','B':'artifacts/dinob-062/development_features.npz','L':'experiments/dinol-177/features.npz'};X={}
 for k,p in path.items():
  z=load(P/p);assert z['ids'].tolist()==plan['IDs'];X[k]=z['X'].astype(float);assert X[k].shape==(400,{'S':3456,'B':6912,'L':9216}[k]) and np.isfinite(X[k]).all()
 old=load(P/f'experiments/dinol-177/{split}/predictions.npz');ref=load(P/f'artifacts/bootstrap-selector-144/{split}/predictions.npz');assert old['ids'].tolist()==ref['ids'].tolist()==plan['IDs'];assert old['g'].tolist()==plan['groups'];np.testing.assert_array_equal(old['y'],ref['y']);np.testing.assert_array_equal(old['fold'],ref['fold']);np.testing.assert_array_equal(old['reference018'],ref['primary']);np.testing.assert_array_equal(old['blend_L_MLP'],.9*ref['primary']+.1*old['L_MLP'])
 for p in [P/f'experiments/dinol-177/{split}/validation.json',P/f'artifacts/bootstrap-selector-144/validation-{split}.json']:assert f.read(p)['passed']
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   p=f.Path(os.fsdecode(args[0])).absolute()
   if p.is_relative_to(P/'data'):raise RuntimeError('Rawdata/holdout/test forbidden')
 sys.addaudithook(guard);return {'X':X,'ids':old['ids'],'y':old['y'],'g':old['g'],'fold':old['fold'],'018':old['reference018'],'021':old['blend_L_MLP'],'globalL':old['L_MLP']}

def bankfit(X,T):
 sc=StandardScaler().fit(X[T]);np.testing.assert_allclose(sc.mean_,X[T].mean(0),atol=1e-12,rtol=0);np.testing.assert_allclose(sc.var_,X[T].var(0),atol=1e-12,rtol=0);sd=X[T].std(0);sd[sd==0]=1.;np.testing.assert_allclose(sc.scale_,sd,atol=1e-12,rtol=1e-12);assert sc.n_samples_seen_==len(T);return sc

def raw(d,arm,idx,scalers):return np.column_stack([scalers[k].transform(d['X'][k][idx])/np.sqrt(d['X'][k].shape[1]) for k in ARMS[arm]])
def transform(d,arm,idx,obj):return obj['score_scaler'].transform(obj['PCA'].transform(raw(d,arm,idx,obj['bank_scalers'])))
def preprocess(d,arm,T,V):
 scalers={k:bankfit(d['X'][k],T) for k in ARMS[arm]};Z=raw(d,arm,T,scalers);Q=raw(d,arm,V,scalers);pca=PCA(n_components=64,svd_solver='full',whiten=False).fit(Z);A=pca.transform(Z);B=pca.transform(Q);sc=StandardScaler().fit(A);obj={'arm':arm,'train_indices':T,'validation_indices':V,'bank_scalers':scalers,'PCA':pca,'score_scaler':sc}
 np.testing.assert_allclose(pca.mean_,Z.mean(0),atol=1e-12,rtol=0);np.testing.assert_allclose(pca.components_@pca.components_.T,np.eye(64),atol=1e-12,rtol=0);C=Z-Z.mean(0);ev_error=float(abs(C.T@(C@pca.components_.T)/(len(T)-1)-pca.components_.T*pca.explained_variance_).max());assert ev_error<1e-9;assert pca.n_samples_==len(T) and sc.n_samples_seen_==len(T);np.testing.assert_allclose(sc.mean_,A.mean(0),atol=1e-12,rtol=0);np.testing.assert_allclose(sc.scale_,A.std(0),atol=1e-12,rtol=0);manual=((Q-Z.mean(0))@pca.components_.T-A.mean(0))/A.std(0);actual=sc.transform(B);err=float(abs(actual-manual).max());assert err<1e-10;assert np.isfinite(actual).all();assert pca.explained_variance_[-1]>1e-12
 return obj,sc.transform(A),actual,{'PCA_eigen_max_error':ev_error,'manual_transform_max_error':err,'retained_variance':float(pca.explained_variance_ratio_.sum()),'bank_count':len(scalers),'joint_input_width':Z.shape[1],'output_width':64,'n_fit_rows':len(T)}

class Head(nn.Module):
 def __init__(self):
  super().__init__();self.hidden=nn.Linear(64,32);self.drop=nn.Dropout(.25);self.out=nn.Linear(32,1)
 def forward(self,x):return self.out(self.drop(torch.nn.functional.gelu(self.hidden(x)))).squeeze(-1)

def predict(m,Z,ym,ys,device='cpu'):
 m=m.to(device).eval()
 with torch.inference_mode():p=m(torch.tensor(Z,dtype=torch.float32,device=device)).cpu().numpy().astype(float)*ys+ym
 m=m.cpu();torch.mps.empty_cache();return p

def manual_prediction(obj,Z,ym,ys):
 z=Z.astype(np.float32).astype(float)@obj['hidden.weight'].numpy().astype(float).T+obj['hidden.bias'].numpy();h=.5*z*(1+erf(z/np.sqrt(2)));return (h@obj['out.weight'].numpy().astype(float).T+obj['out.bias'].numpy()).ravel()*ys+ym

def fit(Z,y,seed):
 ym=float(y.mean());ys=float(y.std());assert ys>0;xt=torch.tensor(Z,dtype=torch.float32);yt=torch.tensor((y-ym)/ys,dtype=torch.float32);torch.manual_seed(seed);m=Head();assert sum(p.numel() for p in m.parameters())==2113;initial=statehash(m);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100,eta_min=.00001);order=np.random.default_rng(seed+1).permuted(np.tile(np.arange(len(y)),(100,1)),axis=1);torch.manual_seed(seed+1000);curves=[]
 for epoch in range(100):
  m.train()
  for st in range(0,len(y),64):
   ix=order[epoch,st:st+64];opt.zero_grad(set_to_none=True);p=m(xt[ix]);loss=torch.nn.functional.smooth_l1_loss(p,yt[ix],beta=1.);assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
  scheduler.step()
  if (epoch+1)%20==0:curves.append({'epoch':epoch+1,'train_RMSE':rmse(y,predict(m,Z,ym,ys))})
 return m.eval(),ym,ys,{'initial_hash':initial,'order_hash':hashlib.sha256(order.tobytes()).hexdigest(),'final_hash':statehash(m),'parameters':2113,'all_parameter_gradients_active':True,'curves':curves}

def comparison(d,ps):
 y=d['y'];fid=d['fold'];scores={k:rmse(y,p) for k,p in ps.items()};refs=['021','SB_blend'];cmp={}
 for ref in refs:
  fg=[rmse(y[fid==k],ps[ref][fid==k])-rmse(y[fid==k],ps['SBL_blend'][fid==k]) for k in range(5)];cmp[ref]={'gain':scores[ref]-scores['SBL_blend'],'fold_gains':fg,'positive_folds':sum(a>0 for a in fg)}
 err={k:y-p for k,p in ps.items()};delta=ps['SBL_blend']-ps['021'];corr=lambda a,b:float(np.corrcoef(a,b)[0,1]);cor={'joint_SB_SBL_error':corr(err['SB'],err['SBL']),'jointSBL_021_error':corr(err['SBL'],err['021']),'blendSBL_021_error':corr(err['SBL_blend'],err['021']),'jointSBL_globalL_error':corr(err['SBL'],err['globalL']),'blend_update_021_residual':corr(delta,err['021']),'blend_update_RMS':float(np.sqrt(np.mean(delta**2)))}
 rng=np.random.default_rng(202710180);unique,inv=np.unique(d['g'],return_inverse=True);boot={r:[] for r in refs}
 for _ in range(3000):
  c=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))[inv]
  for r in refs:boot[r].append(float(np.sqrt(c@(err[r]**2)/c.sum())-np.sqrt(c@(err['SBL_blend']**2)/c.sum())))
 plan=f.read(R/'plan.json');large=np.isin(d['ids'],plan['failure33']['IDs']);assert large.sum()==33;groups={}
 for label,mask in [('persistent_large33',large),('other367',~large)]:
  groups[label]={'n':int(mask.sum()),'RMSE':{k:rmse(y[mask],p[mask]) for k,p in ps.items()},'bias_y_minus_prediction':{k:float((y[mask]-p[mask]).mean()) for k,p in ps.items()},'SBLblend_better_than021_count':int((abs(err['SBL_blend'][mask])<abs(err['021'][mask])).sum())}
 return scores,cmp,cor,{k:np.quantile(v,[.025,.975]).tolist() for k,v in boot.items()},groups

def run(split):
 plan=check()
 if split!='original':assert f.read(R/'original/report.json')['advance']
 d=data(split);out=R/split;out.mkdir(exist_ok=False);start=time.perf_counter();pred={a:np.full((3,400),np.nan) for a in ARMS};records=[];replay=[];max_device=0.;max_manual=0.;fit_count=0
 with f.Monitor('readout-'+split,plan['resource']):
  for k in range(5):
   T=np.flatnonzero(d['fold']!=k);V=np.flatnonzero(d['fold']==k);assert not set(d['g'][T])&set(d['g'][V]);path=out/f'fold-{k}';path.mkdir();paired={};bankref=None
   for arm in ARMS:
    obj,A,B,report=preprocess(d,arm,T,V);pre=path/f'{arm}-preprocessing.joblib';joblib.dump(obj,_repo_io(pre));loaded=joblib.load(_repo_io(pre));np.testing.assert_array_equal(transform(d,arm,V,loaded),B);np.testing.assert_array_equal(transform(d,arm,T,loaded),A)
    if arm=='SB':bankref=obj['bank_scalers']
    else:
     for bank in ['S','B']:
      for attr in ['mean_','scale_','var_']:np.testing.assert_array_equal(getattr(obj['bank_scalers'][bank],attr),getattr(bankref[bank],attr))
    headlogs=[]
    for si,seed0 in enumerate(SEEDS):
     seed=seed0+k;m,ym,ys,log=fit(A,d['y'][T],seed);fit_count+=1;assert ym==float(d['y'][T].mean()) and ys==float(d['y'][T].std());pp=predict(m,B,ym,ys);checkpoint={'model':m.state_dict(),'ym':ym,'ys':ys,'seed':seed,'T':torch.tensor(T),'V':torch.tensor(V),'preprocessing_sha256':f.sha(pre),'log':log};file=path/f'{arm}-{seed0}.pt';torch.save(checkpoint,_repo_io(file));ck=torch.load(_repo_io(file),map_location='cpu',weights_only=True);mm=Head();mm.load_state_dict(ck['model']);np.testing.assert_array_equal(predict(mm,B,ym,ys),pp);manual=manual_prediction(ck['model'],B,ym,ys);me=float(abs(manual-pp).max());de=float(abs(predict(mm,B,ym,ys,'mps')-pp).max());assert me<.001 and de<.001;max_manual=max(max_manual,me);max_device=max(max_device,de);pred[arm][si,V]=pp
     if arm=='SB':paired[seed]=(log['initial_hash'],log['order_hash'])
     else:assert paired[seed]==(log['initial_hash'],log['order_hash'])
     log.update(checkpoint_sha256=f.sha(file),reload_exact=True,CPU_MPS_error=de,manual_forward_error=me);headlogs.append(log)
    if k==0:
     # Refit preprocessing after all outsideT features/labels are perturbed; train inputs must stayexact.
     dd={**d,'y':d['y'].copy(),'X':{bank:v.copy() for bank,v in d['X'].items()}};dd['y'][V]=100000+np.arange(len(V))
     for bank in ARMS[arm]:dd['X'][bank][V]+=54321.
     other,AA,_,_=preprocess(dd,arm,T,V);np.testing.assert_array_equal(AA,A);np.testing.assert_array_equal(other['PCA'].components_,obj['PCA'].components_);np.testing.assert_array_equal(other['PCA'].mean_,obj['PCA'].mean_)
     for si,seed0 in enumerate(SEEDS):
      mm,ym2,ys2,log2=fit(AA,dd['y'][T],seed0);fit_count+=1;ck=torch.load(_repo_io(path/f'{arm}-{seed0}.pt'),map_location='cpu',weights_only=True);assert all(torch.equal(v,ck['model'][n]) for n,v in mm.state_dict().items());np.testing.assert_array_equal(predict(mm,B,ym2,ys2),pred[arm][si,V]);replay.append({'arm':arm,'seed':seed0,'PCA_and_states_exact':True,'outer_labels_and_features_perturbed':True})
     del dd,other
    report.update(arm=arm,fold=k,T=T.tolist(),V=V.tolist(),preprocessing_sha256=f.sha(pre),heads=headlogs);f.write(path/f'{arm}-report.json',report);records.append(report)
   f.emit({'split':split,'fold_complete':k,'seconds':time.perf_counter()-start})
  assert all(np.isfinite(p).all() for p in pred.values());ps={k:d[k] for k in ['018','021','globalL']}
  for arm in ARMS:ps[arm]=pred[arm].mean(0);ps[arm+'_blend']=.9*d['018']+.1*ps[arm]
  scores,cmp,cor,ci,subgroup=comparison(d,ps);advance=scores['SBL_blend']<min(scores['021'],scores['SB_blend']) and cmp['021']['positive_folds']>=3
  arrays(out/'predictions.npz',ids=d['ids'],y=d['y'],g=d['g'],fold=d['fold'],SB_seeds=pred['SB'],SBL_seeds=pred['SBL'],**ps)
 rec={'split':split,'passed_validation':True,'RMSE':scores,'comparisons':cmp,'correlations':cor,'conditional_group_bootstrap_gain_CI95':ci,'subgroups':subgroup,'subgroup_posthoc_historical_definition_not_gate':True,'advance':advance,'main_head_fits':30,'main_PCA_fits':10,'bank_scaler_fit_calls':25,'fresh_validation_head_replays':6,'validation_extra_PCA_fits':2,'total_head_fits':fit_count,'fold0_poison_replays':replay,'same_capacity_parameters':2113,'paired_initialization_and_orders':True,'shared_SB_scalers_exact':True,'all_reload_exact':True,'CPU_MPS_max_error':max_device,'manual_forward_max_error':max_manual,'PCA_eigen_max_error':max(v['PCA_eigen_max_error'] for v in records),'manual_transform_max_error':max(v['manual_transform_max_error'] for v in records),'all_training_scopes_checked':True,'PCA_retained_variance':{arm:[v['retained_variance'] for v in records if v['arm']==arm] for arm in ARMS},'fold_report_hashes':{str(p.relative_to(out)):f.sha(p) for p in out.glob('fold-*/*-report.json')},'predictions_sha256':f.sha(out/'predictions.npz'),'plan_sha256':f.sha(R/'plan.json'),'code_lock_sha256':f.sha(R/'code-lock.json'),'seconds':time.perf_counter()-start,'holdout_test_full500_access':False,'Public_tuning':False};f.write(out/'report.json',rec);check();f.emit({k:v for k,v in rec.items() if k not in ['fold_report_hashes','fold0_poison_replays']})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('split',choices=SPLITS);args=ap.parse_args()
 try:run(args.split)
 except Exception as e:f.write(R/f'failure-{args.split}.json',{'error':repr(e)});raise
