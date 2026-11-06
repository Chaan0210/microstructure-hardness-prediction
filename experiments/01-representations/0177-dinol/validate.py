"""177 independent train-local statistics/PCA algebra/SVR duals and label-poison checks."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse,time
from pathlib import Path
import numpy as np,torch,joblib
from scipy.spatial.distance import cdist
from sklearn.model_selection import GroupKFold
import readout as q
f=q.f;R=q.R;P=q.P

def manual(obj,X,y,pc,C):
 T=np.asarray(obj['train']);V=np.asarray(obj['validation']);sc=obj['scaler'];pca=obj['PCA'];mu=X[T].mean(0);var=X[T].var(0);sd=np.sqrt(var);eps=np.finfo(float).eps;sd[var<=len(T)*eps*var+(len(T)*mu*eps)**2]=1.
 np.testing.assert_allclose(sc.mean_,mu,atol=1e-12,rtol=0);np.testing.assert_allclose(sc.var_,var,atol=1e-11,rtol=0);np.testing.assert_allclose(sc.scale_,sd,atol=1e-12,rtol=0);assert int(sc.n_samples_seen_)==len(T);assert obj['target_mean']==float(y[T].mean()) and obj['target_std']==float(y[T].std());assert pca.n_components_==pc and pca.n_samples_==len(T) and not pca.whiten
 Z=(X[T]-mu)/sd;VZ=(X[V]-mu)/sd;np.testing.assert_allclose(pca.mean_,Z.mean(0),atol=1e-12,rtol=0);Z-=pca.mean_;VZ-=pca.mean_;A=Z@pca.components_.T;B=VZ@pca.components_.T;np.testing.assert_allclose(pca.components_@pca.components_.T,np.eye(pc),atol=1e-10,rtol=0);eigerr=float(abs(Z.T@A/(len(T)-1)-pca.components_.T*pca.explained_variance_).max());assert eigerr<1e-7,eigerr
 m=obj['model'];assert m.C==C and m.epsilon==.1 and m.kernel=='rbf' and m.gamma=='scale' and m.fit_status_==0;np.testing.assert_allclose(m.support_vectors_,A[m.support_],atol=1e-10,rtol=0);np.testing.assert_allclose(m._gamma,1/(pc*A.var()),atol=1e-12,rtol=1e-12);pred=(np.exp(-m._gamma*cdist(B,m.support_vectors_,'sqeuclidean'))@m.dual_coef_[0]+m.intercept_[0])*obj['target_std']+obj['target_mean'];direct=m.predict(pca.transform(sc.transform(X[V])))*obj['target_std']+obj['target_mean'];error=float(abs(pred-direct).max());assert error<1e-8,error;return direct,error,eigerr

def validate(split):
 start=time.perf_counter();q.check();d=q.data(split);out=R/split;report=f.read(out/'report.json');assert f.sha(out/'predictions.npz')==report['predictions_sha256'];saved=q.load(out/'predictions.npz');errors={'SVR_manual':0.,'PCA_eigen':0.,'MLP_reload':0.,'B_state_max':0.};nmlp=0;nsvr=0;refs={};eligible=report['readouts_executed']
 for k in range(5):
  path=out/f'fold-{k}';assert f.sha(path/'report.json')==report['fold_report_hashes'][f'fold-{k}/report.json'];r=f.read(path/'report.json');T=np.flatnonzero(d['fold']!=k);V=np.flatnonzero(d['fold']==k);np.testing.assert_array_equal(r['T'],T);np.testing.assert_array_equal(r['V'],V);assert not set(d['g'][T])&set(d['g'][V])
  for arm in ['B','L']:
   if 'MLP' in eligible:
    X=d[arm];pred=[]
    for ii,seed in enumerate(q.SEEDS):
     file=path/f'{arm}-MLP-{seed}.pt';log=r['models'][f'{arm}_MLP'][ii];assert f.sha(file)==log['sha256'];obj=torch.load(_repo_io(file),map_location='cpu',weights_only=False);np.testing.assert_array_equal(obj['train'],T);np.testing.assert_array_equal(obj['validation'],V);assert obj['seed']==seed+k;mu=X[T].mean(0);sd=X[T].std(0);np.testing.assert_allclose(obj['scaler_mean'].numpy(),mu,atol=1e-12,rtol=0);np.testing.assert_allclose(obj['scaler_scale'].numpy(),sd,atol=1e-12,rtol=0);assert obj['target_mean']==float(d['y'][T].mean()) and obj['target_std']==float(d['y'][T].std());m=q.h.Head(X.shape[1]);m.load_state_dict(obj['model']);pp=q.h.predict(m,(X[V]-mu)/sd,obj['target_mean'],obj['target_std']);pred.append(pp);errors['MLP_reload']=max(errors['MLP_reload'],float(abs(pp-saved[f'{arm}_MLP_seeds'][ii,V]).max()));assert q.h.state_hash(obj['model'])==log['final_state_sha256'];nmlp+=1
     if arm=='B':
      oldroot=P/f'artifacts/{"dinob-mlp-063" if split=="original" else "dinob-complement-064"}/{split}/fold-{k}';oldfile=oldroot/f'baseB-seed-{seed}.pt';old=torch.load(_repo_io(oldfile),map_location='cpu',weights_only=True);assert set(old['model'])==set(obj['model']);assert all(torch.equal(old['model'][name],value) for name,value in obj['model'].items());refs[str(oldfile.relative_to(P))]=f.sha(oldfile)
    np.testing.assert_allclose(np.mean(pred,0),saved[f'{arm}_MLP'][V],atol=1e-8,rtol=0)
   if 'SVR' in eligible:
    X=q.cls(d,arm);sr=r['models'][f'{arm}_SVR'];inner=list(GroupKFold(3,shuffle=True,random_state=202710177+10000*q.SPLITS.index(split)+k).split(X[T],d['y'][T],d['g'][T]));ip=np.empty((9,len(T)))
    for name,hh in sr['files'].items():assert f.sha(path/name)==hh
    for j,(I,H) in enumerate(inner):
     np.testing.assert_array_equal(sr['inner_splits'][j]['train'],T[I]);np.testing.assert_array_equal(sr['inner_splits'][j]['validation'],T[H]);assert not set(d['g'][T[I]])&set(d['g'][T[H]])
     for pi,pc in enumerate(q.PCS):
      obj=joblib.load(_repo_io(path/f'{arm}-SVR-inner-{j}-pc-{pc}.joblib'));np.testing.assert_array_equal(obj['train'],T[I]);np.testing.assert_array_equal(obj['validation'],T[H])
      for ci,C in enumerate(q.CS):
       rec={**obj,'model':obj['models'][str(C)]};pp,err,eig=manual(rec,X,d['y'],pc,C);errors['SVR_manual']=max(errors['SVR_manual'],err);errors['PCA_eigen']=max(errors['PCA_eigen'],eig);ip[3*pi+ci,H]=pp;nsvr+=1
    zz=q.load(path/f'{arm}-SVR-inner.npz');np.testing.assert_array_equal(ip,zz['predictions']);scores=np.sqrt(np.mean((ip-d['y'][T])**2,axis=1));np.testing.assert_array_equal(scores,np.array(sr['inner_RMSE']));winner=int(np.argmin(scores));assert winner==sr['chosen_index'];pc=q.PCS[winner//3];C=q.CS[winner%3];obj=joblib.load(_repo_io(path/f'{arm}-SVR-final.joblib'));np.testing.assert_array_equal(obj['train'],T);np.testing.assert_array_equal(obj['validation'],V);pp,err,eig=manual(obj,X,d['y'],pc,C);errors['SVR_manual']=max(errors['SVR_manual'],err);errors['PCA_eigen']=max(errors['PCA_eigen'],eig);np.testing.assert_array_equal(pp,saved[f'{arm}_SVR'][V]);nsvr+=1
 for rd in eligible:
  for arm in ['B','L']:np.testing.assert_array_equal(.9*d['reference']+.1*saved[f'{arm}_{rd}'],saved[f'blend_{arm}_{rd}'])
 for key,value in report['RMSE'].items():assert abs(q.rmse(d['y'],saved[key])-value)<1e-12
 assert errors['MLP_reload']<1e-8
 # One full outerfold poison replay perpartition, bothbackbones, all readouts actuallyscreened.
 poison=dict(d);poison['y']=d['y'].copy();V=np.flatnonzero(d['fold']==0);T=np.flatnonzero(d['fold']!=0);poison['y'][V]+=100000.;path=out/'outer-label-poison';path.mkdir(exist_ok=False);poison_errors={}
 with f.Monitor('validation-'+split,f.read(R/'plan.json')['resources']):
  for arm in ['B','L']:
   if 'MLP' in eligible:
    pp,_=q.mlp(poison,arm,T,V,0,path);np.testing.assert_array_equal(pp,saved[f'{arm}_MLP_seeds'][:,V]);poison_errors[arm+'_MLP']=0.
    for seed in q.SEEDS:
     a=torch.load(_repo_io(path/f'{arm}-MLP-{seed}.pt'),map_location='cpu',weights_only=False)['model'];b=torch.load(_repo_io(out/'fold-0'/f'{arm}-MLP-{seed}.pt'),map_location='cpu',weights_only=False)['model'];assert all(torch.equal(a[key],b[key]) for key in a)
   if 'SVR' in eligible:
    pp,log=q.svr(poison,arm,T,V,0,split,path);np.testing.assert_array_equal(pp,saved[f'{arm}_SVR'][V]);assert log['choice']==f.read(out/'fold-0/report.json')['models'][arm+'_SVR']['choice'];poison_errors[arm+'_SVR']=0.
 rec={'passed':True,'split':split,'MLP_reloads':nmlp,'SVR_dual_reloads':nsvr,'errors':errors,'all_inner_and_outer_scaler_PCA_target_scopes_verified':True,'all_PCA_eigen_equations_checked':True,'all_inner_selection_scores_replayed':True,'all_B_MLP_states_exact_historical':True if 'MLP' in eligible else None,'outer_fold0_target_poison_exact':poison_errors,'poison_scope':'One completeouterfold perpartition,notall5folds','seconds':time.perf_counter()-start,'report_sha256':f.sha(out/'report.json'),'source_sha256':f.sha(Path(__file__)),'references':refs,'no_holdout_test_access':True};f.write(out/'validation.json',rec);f.emit({key:value for key,value in rec.items() if key!='references'})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('split',choices=q.SPLITS);args=ap.parse_args()
 try:validate(args.split)
 except Exception as e:f.write(R/f'validation-{args.split}-failure.json',{'error':repr(e),'type':type(e).__name__});raise
