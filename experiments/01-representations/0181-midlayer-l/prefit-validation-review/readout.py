from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse,time,sys,os
import numpy as np,torch,joblib
from scipy.special import erf
import runtime as f,mlp_head as h
R=f.R;P=f.P;ARMS=['MID','LAST'];SEEDS=[20261128,20261205,20261219];SPLITS=['original','20261031','20261101']

def load(p):
 with np.load(_repo_io(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def arrays(p,**z):
 with p.open('xb') as fp:np.savez_compressed(_repo_io(fp),**z)
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
def check():
 p=f.check()
 for n,hh in f.read(R/'readout-code-lock.json')['sources'].items():assert f.sha(R/n)==hh
 v=f.read(R/'feature-validation.json');assert v['passed'] and f.sha(R/'features.npz')==v['feature_sha256'];return p

def data(split):
 p=check();z=load(R/'features.npz');old=load(P/f'experiments/dinol-177/{split}/predictions.npz');assert z['ids'].tolist()==old['ids'].tolist()==p['IDs'];assert old['g'].tolist()==p['groups'];assert f.read(P/f'experiments/dinol-177/{split}/validation.json')['passed'];np.testing.assert_array_equal(old['blend_L_MLP'],.9*old['reference018']+.1*old['L_MLP'])
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   pp=f.Path(os.fsdecode(args[0])).absolute()
   if pp.is_relative_to(P/'data'):raise RuntimeError('Rawdata forbidden duringreadout')
 sys.addaudithook(guard);return {'ids':old['ids'],'y':old['y'],'g':old['g'],'fold':old['fold'],'018':old['reference018'],'021':old['blend_L_MLP'],'Lmulti':old['L_MLP'],'X':{a:z[a] for a in ARMS}}

def predict(obj,X,device='cpu'):
 Z=(X-obj['scaler_mean'].numpy())/obj['scaler_scale'].numpy();m=h.Head(3072);m.load_state_dict(obj['model']);m.eval()
 if device=='cpu':return h.predict(m,Z,obj['ym'],obj['ys'])
 m=m.to(device)
 with torch.inference_mode():p=m(torch.tensor(Z,dtype=torch.float32,device=device))[0].cpu().numpy().astype(float)*obj['ys']+obj['ym']
 del m;torch.mps.empty_cache();return p

def manual(obj,X):
 Z=((X-obj['scaler_mean'].numpy())/obj['scaler_scale'].numpy()).astype(np.float32).astype(float);s=obj['model'];z=Z@s['shared.0.weight'].numpy().astype(float).T+s['shared.0.bias'].numpy();q=.5*z*(1+erf(z/np.sqrt(2)));return (q@s['reg.weight'].numpy().astype(float).T+s['reg.bias'].numpy()).ravel()*obj['ys']+obj['ym']

def fit_arm(X,y,T,V,seed,path):
 sc,ym,ys,edges,bins=h.preprocess(X[T],y[T]);np.testing.assert_allclose(sc.mean_,X[T].mean(0),atol=1e-12,rtol=0);np.testing.assert_allclose(sc.var_,X[T].var(0),atol=1e-12,rtol=0);assert sc.n_samples_seen_==len(T) and ym==float(y[T].mean()) and ys==float(y[T].std());assert len(set(T)&set(V))==0
 xt=torch.tensor(sc.transform(X[T]),dtype=torch.float32);yt=torch.tensor((y[T]-ym)/ys,dtype=torch.float32);bt=torch.tensor(bins);m,log=h.fit(xt,yt,bt,seed,0.);assert sum(v.numel() for n,v in m.named_parameters() if not n.startswith('cls.'))==196737;pp=h.predict(m,sc.transform(X[V]),ym,ys)
 obj={'model':m.state_dict(),'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'scaler_var':torch.tensor(sc.var_),'ym':ym,'ys':ys,'seed':seed,'T':torch.tensor(T),'V':torch.tensor(V),'input_dim':3072,'config':h.CONFIG,'effective_aux_weight':0.};torch.save(obj,_repo_io(path));loaded=torch.load(_repo_io(path),map_location='cpu',weights_only=True);np.testing.assert_array_equal(pp,predict(loaded,X[V]));err=float(abs(pp-manual(loaded,X[V])).max());dev=float(abs(pp-predict(loaded,X[V],'mps')).max());assert err<.001 and dev<.001
 log.update(checkpoint_sha256=f.sha(path),reload_exact=True,manual_forward_error=err,CPU_MPS_error=dev,train_RMSE=rmse(y[T],h.predict(m,sc.transform(X[T]),ym,ys)),same_effective_parameters=196737)
 return pp,log

def describe(d,ps):
 y=d['y'];fid=d['fold'];scores={k:rmse(y,p) for k,p in ps.items()};cmp={}
 for ref in ['021','LAST_blend']:
  fg=[rmse(y[fid==k],ps[ref][fid==k])-rmse(y[fid==k],ps['MID_blend'][fid==k]) for k in range(5)];cmp[ref]={'gain':scores[ref]-scores['MID_blend'],'fold_gains':fg,'positive_folds':sum(x>0 for x in fg)}
 e={k:y-p for k,p in ps.items()};cor=lambda a,b:float(np.corrcoef(a,b)[0,1]);delta=ps['MID_blend']-ps['021'];corr={'MID_LAST_residual':cor(e['MID'],e['LAST']),'MID_021_residual':cor(e['MID'],e['021']),'LAST_021_residual':cor(e['LAST'],e['021']),'MID_Lmulti_residual':cor(e['MID'],e['Lmulti']),'MIDblend_021_residual':cor(e['MID_blend'],e['021']),'MID_update_021_residual':cor(delta,e['021']),'MID_update_RMS':float(np.sqrt(np.mean(delta**2)))};rng=np.random.default_rng(202710181);unique,inv=np.unique(d['g'],return_inverse=True);boot={k:[] for k in cmp}
 for _ in range(3000):
  c=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))[inv]
  for k in boot:boot[k].append(float(np.sqrt(c@(e[k]**2)/c.sum())-np.sqrt(c@(e['MID_blend']**2)/c.sum())))
 mask=np.isin(d['ids'],f.read(R/'plan.json')['failure33_IDs']);assert mask.sum()==33;groups={}
 for name,m in [('persistent_large33',mask),('other367',~mask)]:groups[name]={'n':int(m.sum()),'RMSE':{k:rmse(y[m],p[m]) for k,p in ps.items()},'bias_y_minus_pred':{k:float(e[k][m].mean()) for k in ps}}
 return scores,cmp,corr,{k:np.quantile(v,[.025,.975]).tolist() for k,v in boot.items()},groups

def run(split):
 plan=check()
 if split!='original':assert f.read(R/'original/report.json')['advance']
 d=data(split);out=R/split;out.mkdir(exist_ok=False);start=time.perf_counter();pred={a:np.full((3,400),np.nan) for a in ARMS};records=[];replays=[]
 with f.Monitor('readout-'+split,plan['resources']):
  for k in range(5):
   T=np.flatnonzero(d['fold']!=k);V=np.flatnonzero(d['fold']==k);assert not set(d['g'][T])&set(d['g'][V]);path=out/f'fold-{k}';path.mkdir();paired={}
   for arm in ARMS:
    logs=[]
    for si,seed in enumerate(SEEDS):
     pp,log=fit_arm(d['X'][arm],d['y'],T,V,seed+k,path/f'{arm}-{seed}.pt');pred[arm][si,V]=pp;pair=(log['initial_state_sha256'],log['batch_order_sha256'])
     if arm=='MID':paired[seed]=pair
     else:assert pair==paired[seed]
     if k==0:
      XX=d['X'][arm].copy();yy=d['y'].copy();XX[V]+=54321.;yy[V]=100000+np.arange(len(V));q=out/f'{arm}-poison-{seed}.pt';_,replog=fit_arm(XX,yy,T,V,seed,q);obj=torch.load(_repo_io(path/f'{arm}-{seed}.pt'),map_location='cpu',weights_only=True);other=torch.load(_repo_io(q),map_location='cpu',weights_only=True);assert all(torch.equal(v,other['model'][n]) for n,v in obj['model'].items());assert all(torch.equal(obj[n],other[n]) for n in ['scaler_mean','scaler_scale','scaler_var']);np.testing.assert_array_equal(predict(other,d['X'][arm][V]),pp);replays.append({'arm':arm,'seed':seed,'exact_states_stats_and_original_predictions':True,'outer_targets_and_features_perturbed':True,'checkpoint_sha256':f.sha(q)})
     logs.append(log)
    rec={'arm':arm,'fold':k,'T':T.tolist(),'V':V.tolist(),'heads':logs};f.write(path/f'{arm}-report.json',rec);records.append(rec)
   f.emit({'split':split,'fold_complete':k,'seconds':time.perf_counter()-start})
  assert all(np.isfinite(v).all() for v in pred.values());ps={k:d[k] for k in ['018','021','Lmulti']}
  for a in ARMS:ps[a]=pred[a].mean(0);ps[a+'_blend']=.9*d['018']+.1*ps[a]
  scores,cmp,corr,ci,groups=describe(d,ps);advance=all(v['gain']>.01 and v['positive_folds']>=3 for v in cmp.values());arrays(out/'predictions.npz',ids=d['ids'],y=d['y'],g=d['g'],fold=d['fold'],MID_seeds=pred['MID'],LAST_seeds=pred['LAST'],**ps)
 allheads=[h for r in records for h in r['heads']];rec={'split':split,'passed_validation':True,'RMSE':scores,'comparisons':cmp,'correlations':corr,'conditional_group_bootstrap_gain_CI95':ci,'historical_subgroups_diagnostic_only':groups,'advance':advance,'main_head_fits':30,'wholefold0_poison_refits':6,'poison_replays':replays,'equal_predictive_parameters_perhead':196737,'paired_initialization_order_exact':True,'same_device_reload_exact':True,'manual_forward_max_error':max(h['manual_forward_error'] for h in allheads),'CPU_MPS_max_error':max(h['CPU_MPS_error'] for h in allheads),'all_train_scaler_target_scopes_verified':True,'train_RMSE_byarm':{a:[h['train_RMSE'] for r in records if r['arm']==a for h in r['heads']] for a in ARMS},'predictions_sha256':f.sha(out/'predictions.npz'),'fold_report_hashes':{str(p.relative_to(out)):f.sha(p) for p in out.glob('fold-*/*-report.json')},'plan_sha256':f.sha(R/'plan.json'),'readout_code_lock_sha256':f.sha(R/'readout-code-lock.json'),'seconds':time.perf_counter()-start,'noholdout_test_full500':True,'close_L_variants_if_original_gate_fails':not advance};f.write(out/'report.json',rec);check();f.emit({k:v for k,v in rec.items() if k not in ['train_RMSE_byarm','fold_report_hashes','poison_replays']})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('split',choices=SPLITS);arg=ap.parse_args()
 try:run(arg.split)
 except Exception as e:f.write(R/f'readout-{arg.split}-failure.json',{'error':repr(e)});raise
