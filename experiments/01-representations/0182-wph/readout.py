"""Fixed WPH versus same-wavelet amplitude covariates in unchanged honest residual protocol."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse,time,sys,os
import numpy as np,joblib
from scipy.spatial.distance import cdist
from sklearn.model_selection import GroupKFold
import runtime as f,residual_models as n
R=f.R;P=f.P;SPLITS=['original','20261031','20261101'];KINDS=n.KINDS

def load(p):
 with np.load(_repo_io(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def arrays(p,**z):
 with p.open('xb') as fp:np.savez_compressed(_repo_io(fp),**z)
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
def check():
 plan=f.check()
 for name,hh in f.read(R/'readout-code-lock.json')['sources'].items():assert f.sha(R/name)==hh
 v=f.read(R/'feature-validation.json');assert v['passed'] and f.sha(R/'features.npz')==v['feature_sha256'];return plan

def data(split):
 plan=check();phys=load(P/'artifacts/run-022/development_features.npz');b=load(P/'artifacts/spatial-joint-140/features.npz');v=load(R/'features.npz');g=load(P/f'experiments/dinol-177/{split}/predictions.npz');base=load(P/f'artifacts/bootstrap-selector-144/{split}/predictions.npz')
 for z in [phys,b,v,g,base]:assert z['ids'].tolist()==plan['IDs']
 np.testing.assert_array_equal(g['g'],phys['groups']);np.testing.assert_array_equal(g['reference018'],base['primary']);np.testing.assert_array_equal(g['y'],base['y']);np.testing.assert_array_equal(g['fold'],base['fold']);np.testing.assert_array_equal(g['blend_L_MLP'],.9*base['primary']+.1*g['L_MLP'])
 for file in [f'artifacts/bootstrap-selector-144/validation-{split}.json',f'artifacts/spatial-joint-140/validation-{split}.json',f'experiments/dinol-177/{split}/validation.json']:assert f.read(P/file)['passed']
 X=np.column_stack([phys['X'].astype(float),b['adjacent']]);blocks=f.read(P/'artifacts/run-022/report.json')['blocks']+['joint_token']*528
 d={'ids':g['ids'],'y':g['y'],'g':g['g'],'fid':g['fold'],'018':g['reference018'],'021':g['blend_L_MLP'],'globalL':g['L_MLP'],'X':{'B':X,'AMP':np.column_stack([X,v['AMP']]),'WPH':np.column_stack([X,v['WPH']])},'blocks':{'B':blocks,'AMP':blocks+['new_wavelet_covariance']*15,'WPH':blocks+['new_wavelet_covariance']*15}}
 assert X.shape==(400,585) and d['X']['AMP'].shape==d['X']['WPH'].shape==(400,600)
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   pp=f.Path(os.fsdecode(args[0])).absolute()
   if pp.is_relative_to(P/'data'):raise RuntimeError('All rawdata forbidden during residual fits')
 sys.addaudithook(guard);return d

def manual(m,X,y,V,blocks):
 if m['kind']=='none':return np.zeros(len(V))
 lo=X.min(0);span=X.max(0)-lo;span[span<10*np.finfo(float).eps]=1.;U=np.clip((X-lo)/span,0,1);UV=np.clip((V-lo)/span,0,1);mu=U.mean(0);var=U.var(0);sd=np.sqrt(var);sd[sd==0]=1.;w=np.array([1/np.sqrt(blocks.count(b)) for b in blocks]);Z=(U-mu)/sd*w;W=(UV-mu)/sd*w;rs=float(np.sqrt(np.mean(y*y)));rs=rs if rs>1e-12 else 1.;alpha=({'ridge1':1.,'ridge01':.1,'rbf1':1.,'rbf01':.1}[m['kind']])*len(y)
 np.testing.assert_array_equal(m['bound'].data_min_,lo);np.testing.assert_allclose(m['bound'].data_max_,X.max(0),atol=0,rtol=0);np.testing.assert_allclose(m['scale'].mean_,mu,atol=1e-12,rtol=0);np.testing.assert_allclose(m['scale'].var_,var,atol=1e-12,rtol=0);np.testing.assert_array_equal(m['weights'],w);assert m['residual_scale']==rs and m['alpha']==alpha and m['n_train']==len(y)
 if m['kind'].startswith('ridge'):
  beta=Z.T@np.linalg.solve(Z@Z.T+alpha*np.eye(len(y)),y/rs);np.testing.assert_allclose(beta,m['regressor'].coef_,atol=2e-10,rtol=1e-10);return W@beta*rs
 D=np.sum((Z[:,None,:]-Z[None,:,:])**2,2);tri=D[np.triu_indices(len(y),1)];pos=tri[tri>1e-12];gamma=1/float(np.median(pos)) if len(pos) else 1.;assert abs(gamma-m['gamma'])<1e-12;K=np.exp(-gamma*D);H=np.eye(len(y))-np.ones((len(y),len(y)))/len(y);C=H@K@H;scale=float(np.trace(C)/len(y));scale=1. if scale<=1e-12 else scale;assert abs(scale-m['kernel_scale'])<1e-12;C/=scale;ev,Q=np.linalg.eigh(C);coef=Q@((Q.T@(y/rs))/(ev+alpha));np.testing.assert_allclose(coef,m['coef'],atol=2e-11,rtol=1e-10);VK=np.exp(-gamma*np.sum((W[:,None,:]-Z[None,:,:])**2,2));return (VK-VK.mean(1)[:,None]-K.mean(0)[None,:]+K.mean())/scale@coef*rs

def weights(loss,groups,seed):
 unique,inv=np.unique(groups,return_inverse=True);draw=np.random.default_rng(seed).integers(0,len(unique),(512,len(unique)));counts=np.stack([np.bincount(row,minlength=len(unique)) for row in draw]);totals=counts[:,inv]@loss.T;wins=np.argmin(totals,axis=1);w=np.bincount(wins,minlength=5)/512
 # Independent group sums + explicit drawn group lookups.
 group_losses=np.array([loss[:,groups==g].sum(1) for g in unique]);other=np.argmin(group_losses[draw].sum(1),axis=1);np.testing.assert_array_equal(wins,other);np.testing.assert_array_equal(w,np.array([(other==i).mean() for i in range(5)]));return w,draw,wins

def block(d,split,k,j,arm,out):
 out.mkdir(exist_ok=False);src=P/f'artifacts/spatial-joint-140/{split}/adjacent-fold-{k}/block-{j}';rec=f.read(src/'report.json');z=load(src/'predictions.npz');assert f.sha(src/'predictions.npz')==rec['predictions_sha256'];B=z['block_indices'];V=z['outer_indices'];T=np.flatnonzero(d['fid']!=k);A=np.setdiff1d(T,B);np.testing.assert_array_equal(V,np.flatnonzero(d['fid']==k));assert set(A)|set(B)==set(T) and not set(B)&set(V);assert not set(d['g'][A])&set(d['g'][B]);assert not set(d['g'][T])&set(d['g'][V]);res=d['y'][B]-z['anchor_predictions'];np.testing.assert_array_equal(res,z['residual']);X=d['X'][arm];idx={int(ii):jj for jj,ii in enumerate(B)};ip=np.full((5,len(B)),np.nan);pp=[];models={};errors=[];inner_poison=0
 expected=list(GroupKFold(3,shuffle=True,random_state=20271013+100*k+j).split(X[B],res,d['g'][B]))
 def fit_save(I,H,target,name):
  m=n.fit(X[I],target,d['blocks'][arm],kind);p=n.predict(m,X[H]);file=out/(name+'.joblib');joblib.dump({'model':m,'train_indices':I,'validation_indices':H},_repo_io(file));obj=joblib.load(_repo_io(file));np.testing.assert_array_equal(p,n.predict(obj['model'],X[H]));q=manual(obj['model'],X[I],target,X[H],d['blocks'][arm]);err=float(abs(p-q).max());assert err<2e-9;errors.append(err);models[file.name]=f.sha(file);return p
 for t,(it,iv) in enumerate(expected):
  I=B[it];H=B[iv];assert rec['splits'][t]=={'train_indices':I.tolist(),'validation_indices':H.tolist()};assert not set(I)&set(H) and not set(d['g'][I])&set(d['g'][H]);assert not (set(I)|set(H))&set(V)
  for ci,kind in enumerate(KINDS):
   ip[ci,iv]=fit_save(I,H,res[it],f'inner-{t}-{kind}')
   if arm in ['AMP','WPH'] and k==0 and t==0:
    pert=res.copy();pert[iv]+=54321.;m=n.fit(X[I],pert[it],d['blocks'][arm],kind);np.testing.assert_array_equal(n.predict(m,X[H]),ip[ci,iv]);inner_poison+=1
 seed=20271204+10000*SPLITS.index(split)+100*k+j;w,draw,wins=weights((ip-res)**2,d['g'][B],seed)
 for kind in KINDS:pp.append(fit_save(B,V,res,'full-'+kind))
 pp=np.stack(pp);pred=w@pp
 if arm=='B':
  np.testing.assert_array_equal(ip,z['inner_predictions']);ref=load(P/f'artifacts/bootstrap-selector-144/{split}/fold-{k}-block-{j}/predictions.npz');np.testing.assert_array_equal(pp,ref['full_candidate_predictions']);np.testing.assert_array_equal(w,ref['weights']);np.testing.assert_array_equal(draw,ref['draws']);np.testing.assert_array_equal(wins,ref['winners']);np.testing.assert_array_equal(pred,ref['primary'])
 arrays(out/'predictions.npz',B=B,V=V,A=A,anchor=z['anchor_predictions'],residual=res,inner=ip,full=pp,weights=w,draws=draw,winners=wins,primary=pred)
 r={'arm':arm,'fold':k,'block':j,'models_sha256':models,'splits':rec['splits'],'source_anchor_report_sha256':f.sha(src/'report.json'),'predictions_sha256':f.sha(out/'predictions.npz'),'independent_max_error':max(errors),'reload_exact':True,'train_scopes_groups_verified':True,'independent_bootstrap_exact':True,'inner_heldout_poison_fits':inner_poison,'weights':w.tolist(),'B_control_exact':arm=='B','seed':seed};f.write(out/'report.json',r);return pred,r

def diagnostics(d,p):
 y=d['y'];fold=d['fid'];scores={k:rmse(y,v) for k,v in p.items()};gain={}
 for ref in ['021','AMP','018']:
  fg=[rmse(y[fold==k],p[ref][fold==k])-rmse(y[fold==k],p['WPH'][fold==k]) for k in range(5)];gain[ref]={'gain':scores[ref]-scores['WPH'],'fold_gains':fg,'positive_folds':sum(a>0 for a in fg)}
 err={k:y-v for k,v in p.items()};delta=p['WPH']-p['021'];corr=lambda a,b:float(np.corrcoef(a,b)[0,1]);cor={'WPH_vs_AMP_residual':corr(err['WPH'],err['AMP']),'WPH_vs021_residual':corr(err['WPH'],err['021']),'WPH_update_vs021_residual':corr(delta,err['021']),'WPH_update_vsAMP_update':corr(delta,p['AMP']-p['021']),'WPH_update_RMS':float(np.sqrt(np.mean(delta**2)))}
 rng=np.random.default_rng(202710182);unique,inv=np.unique(d['g'],return_inverse=True);boot={k:[] for k in ['021','AMP']}
 for _ in range(3000):
  counts=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))[inv]
  for k in boot:boot[k].append(float(np.sqrt(counts@(err[k]**2)/counts.sum())-np.sqrt(counts@(err['WPH']**2)/counts.sum())))
 ids=f.read(R/'plan.json')['failure33_IDs'];idx=np.array([str(i) in ids for i in d['ids']]);assert idx.sum()==33
 return scores,gain,cor,{k:np.quantile(v,[.025,.975]).tolist() for k,v in boot.items()},{k:rmse(y[idx],p[k][idx]) for k in ['021','AMP','WPH']}

def run(split):
 plan=check()
 if split!='original':assert f.read(R/'original/report.json')['advance'] and f.read(R/'original/validation.json')['passed']
 d=data(split);out=R/split;out.mkdir(exist_ok=False);start=time.perf_counter();p={k:d[k] for k in ['018','021','globalL']};p.update({k:np.full(400,np.nan) for k in ['B_replay','AMP','WPH']});reports=[];foldcor={a:np.full((3,400),np.nan) for a in ['B','AMP','WPH']}
 with f.Monitor('readout-'+split,plan['resources']):
  for k in range(5):
   root=P/f'artifacts/spatial-joint-140/{split}/adjacent-fold-{k}';fr=f.read(root/'report.json');z=load(root/'predictions.npz');assert f.sha(root/'predictions.npz')==fr['predictions_sha256'];V=z['outer_indices']
   for arm in ['B','AMP','WPH']:
    seen=[]
    for j in range(3):
     pp,rec=block(d,split,k,j,arm,out/f'{arm}-fold-{k}-block-{j}');foldcor[arm][j,V]=pp;reports.append(rec);seen.extend(load(out/f'{arm}-fold-{k}-block-{j}/predictions.npz')['B'].tolist())
    assert sorted(seen)==np.flatnonzero(d['fid']!=k).tolist()
    pred=z['base']+foldcor[arm][:,V].mean(0)
    if arm=='B':p['B_replay'][V]=pred
    else:p[arm][V]=.9*pred+.1*p['globalL'][V]
   f.emit({'split':split,'fold_complete':k,'seconds':time.perf_counter()-start})
  np.testing.assert_array_equal(p['B_replay'],p['018']);np.testing.assert_array_equal(.9*p['B_replay']+.1*p['globalL'],p['021']);assert all(np.isfinite(v).all() for v in p.values())
  pert={**d,'y':d['y'].copy()};pert['y'][d['fid']==0]=100000+np.arange(sum(d['fid']==0));poison=[]
  for arm in ['AMP','WPH']:
   for j in range(3):
    pp,rr=block(pert,split,0,j,arm,R/f'poison-{split}-{arm}-block-{j}');np.testing.assert_array_equal(pp,foldcor[arm][j,d['fid']==0]);poison.append(rr)
  scores,gain,cor,ci,failure33=diagnostics(d,p);advance=all(gain[ref]['gain']>0 and gain[ref]['positive_folds']>=3 for ref in ['021','AMP'])
  arrays(out/'predictions.npz',ids=d['ids'],y=d['y'],g=d['g'],fold=d['fid'],B_corrections=foldcor['B'],AMP_corrections=foldcor['AMP'],WPH_corrections=foldcor['WPH'],**p)
 rec={'split':split,'passed_validation':True,'RMSE':scores,'comparison':gain,'correlations':cor,'conditional_group_bootstrap_gain_CI95':ci,'failure33_fixed_diagnostic_RMSE':failure33,'bootstrap_limit':'Conditional fixedfits, repeated400 and adaptive history; not independent confirmation.','advance':advance,'gate':plan['gate'],'main_model_objects':len(reports)*20,'main_nonnull_fits':len(reports)*16,'outer0_poison_model_objects':len(poison)*20,'inner_heldout_poison_refits':sum(x['inner_heldout_poison_fits'] for x in reports+poison),'independent_max_error':max(x['independent_max_error'] for x in reports+poison),'all_B_inner_full_weights_predictions_exact':True,'018_and021_control_exact':True,'all_reload_exact':True,'all_train_local_scaling_and_group_scopes_verified':True,'outer_fold0_botharms_all3blocks_poison_exact':True,'upstream_neural_anchors_reused_with_exclusion_certificates':True,'new_neural_fits':0,'block_report_hashes':{str(pth.relative_to(out)):f.sha(pth) for pth in sorted(out.glob('*/report.json'))},'predictions_sha256':f.sha(out/'predictions.npz'),'plan_sha256':f.sha(R/'plan.json'),'readout_code_sha256':f.sha(R/'readout-code-lock.json'),'seconds':time.perf_counter()-start,'holdout_test_full500_access':False};f.write(out/'report.json',rec);check();f.emit({k:v for k,v in rec.items() if k!='block_report_hashes'})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('split',choices=SPLITS);args=ap.parse_args()
 try:run(args.split)
 except Exception as e:f.write(R/f'readout-{args.split}-failure.json',{'error':repr(e)});raise
