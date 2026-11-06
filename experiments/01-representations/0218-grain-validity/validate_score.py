from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,time,math
from pathlib import Path
import numpy as np,joblib
from scipy.spatial.distance import pdist,cdist,squareform
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import run as s
P=s.P;s.protect();s.guard();start=time.monotonic();z,xx,blocks,valid,hp=s.features();y=z['y'];g=z['groups'];models=0;poisonfits=0;maxerr=0.;data={};checks=[]
for split in s.SPLITS:
 dest=_repo_path(R, split);done=s.read(dest/'completion.json');assert s.sha(dest/'OOF.npz')==done['OOF_sha256'];o=s.load(dest/'OOF.npz');ref,old,coef=s.source(split)
 for key in ['ids','y','fold']:np.testing.assert_array_equal(o[key],ref[key])
 data[split]=o
 for k in range(5):
  T=np.flatnonzero(o['fold']!=k);V=np.flatnonzero(o['fold']==k)
  for arm in s.ARMS:
   d=dest/f'fold-{k}'/arm;report=s.read(d/'report.json');assert s.sha(d/'model.joblib')==report['model_sha256'] and s.sha(d/'predictions.npz')==report['prediction_sha256'];m=joblib.load(_repo_io(d/'model.joblib'));pv=m.predict(xx[arm][V]);np.testing.assert_allclose(pv,o[arm][V],rtol=0,atol=0);np.testing.assert_array_equal(T,report['train_indices']);np.testing.assert_array_equal(V,report['validation_indices']);assert int(np.argmin(report['inner_scores']))==report['chosen_index']
   U=xx[arm][T].copy();Q=xx[arm][V].copy();good=np.isfinite(U[:,0]);med=float(np.median(U[good,0])) if good.any() else 0.;assert m.impute_==med;U[~good,0]=med;Q[~np.isfinite(Q[:,0]),0]=med;minimum=U.min(0);span=U.max(0)-minimum;span[span==0]=1;Us=np.clip((U-minimum)/span,0,1);Qs=np.clip((Q-minimum)/span,0,1);mean=Us.mean(0);std=Us.std(0);std[std==0]=1;block=np.array(blocks);weights=np.array([1/np.sqrt(np.sum(block==b)) for b in block]);Ut=(Us-mean)/std*weights;Qt=(Qs-mean)/std*weights
   np.testing.assert_allclose(Ut,m.model_.X_,rtol=0,atol=1e-12);np.testing.assert_array_equal(minimum,m.model_.bound_.data_min_);assert m.model_.ym_==float(y[T].mean()) and m.model_.ys_==float(y[T].std())
   dist=pdist(Ut,'sqeuclidean');gamma=1/(np.median(dist[dist>1e-12])*m.bandwidth**2);assert abs(gamma-m.model_.gamma_)<1e-12;K=np.exp(-gamma*squareform(dist));K.flat[::len(T)+1]+=m.alpha;coeff=np.linalg.solve(K,(y[T]-y[T].mean())/y[T].std());ind=np.exp(-gamma*cdist(Qt,Ut,'sqeuclidean'))@coeff*y[T].std()+y[T].mean();err=float(abs(ind-pv).max());assert err<1e-8;maxerr=max(maxerr,err);models+=1
   for q in report['inner_splits']:
    I,H=map(np.array,[q['train'],q['validation']]);assert set(I)|set(H)==set(T) and not set(I)&set(H) and not set(g[I])&set(g[H]);assert not set(V)&set(I)
   if k==0:
    yy=y.copy();yy[V]=np.nan;p2,m2,r2,ip2=s.fit(xx[arm],yy,g,T,V,blocks,k);poisonfits+=28;np.testing.assert_array_equal(p2,pv);np.testing.assert_array_equal(ip2,s.load(d/'predictions.npz')['inner']);assert r2['chosen_index']==report['chosen_index'];np.testing.assert_array_equal(m2.model_.coef_,m.model_.coef_)
  # Independent frozen affine counterfactual; no retraining of calibrators/residuals.
  cal=s.read(P/f'artifacts/spatial-joint-140/{split}/adjacent-fold-{k}/report.json')['outer_calibrator'];raw=s.load(P/f'artifacts/cnn-seed-mean-133/{split}/predictions.npz')['raw'][V];short=s.load(P/f'artifacts/component-ablation-121/{split}/predictions.npz')['short'][V];base=.9*(cal['target_center']+cal['slope']*(raw-cal['prediction_center']))+.1*short;correction=ref['reference018'][V]-base
  for arm in s.ARMS:
   newraw=raw+.3*(o[arm][V]-old[V]);newbase=.9*(cal['target_center']+cal['slope']*(newraw-cal['prediction_center']))+.1*short;full=.9*(newbase+correction)+.1*ref['L_MLP'][V];np.testing.assert_allclose(full,o['ensemble_'+arm][V],rtol=0,atol=1e-10)
 checks.append({'split':split,'OOF_sha256':s.sha(dest/'OOF.npz'),'control_maxerr':done['control_KRR_maxerr'],'control021_maxerr':done['control021_maxerr']})
s.protect();s.write(_repo_path(R, 'validation.json'),{'passed':True,'saved_models_reloaded':models,'independent_kernel_formula_maxerror':maxerr,'train_only_preprocessing_verified':45,'outerfold0_label_NaN_poison_paths':9,'poison_KRR_fits':poisonfits,'inner_folds_partition_and_groups_verified':True,'frozen_ensemble_independent_affine_algebra':True,'evidence':checks,'seconds':time.monotonic()-start,'limits':'Other neural branches/calibrators/residual models not retrained; their fixed counterfactual is not fullprocedure validation.'})
# Scores computed after validation. Fixed bootstrap draws shared by allarms/comparisons.
rng=np.random.default_rng(202610218);draws=rng.integers(0,400,(10000,400));counts=np.stack([np.bincount(a,minlength=400) for a in draws]);missing=~valid.all(1);rest=np.flatnonzero(~missing);dr=rng.integers(0,len(rest),(10000,len(rest)));cr=np.stack([np.bincount(a,minlength=len(rest)) for a in dr]);results={};boots={};perid={};comparisons=[('B-A','A_current','B_validity'),('C-A','A_current','C_valid_mean'),('C-B','B_validity','C_valid_mean')]
def compare(A,B,tag,fold=None):
 if A.ndim==1:A=A[None];B=B[None]
 a=A.mean(0);b=B.mean(0);ga=float(np.sqrt(a.mean())-np.sqrt(b.mean()));boot=np.sqrt(counts@a/400)-np.sqrt(counts@b/400);rb=np.sqrt(cr@a[rest]/len(rest))-np.sqrt(cr@b[rest]/len(rest));boots[tag]=boot
 rows={}
 for name,mask in [('all400',np.ones(400,bool)),('empty3',missing),('other397',~missing)]:
  rows[name]={'n':int(mask.sum()),'baseline_RMSE':float(np.sqrt(a[mask].mean())),'candidate_RMSE':float(np.sqrt(b[mask].mean())),'RMSE_gain':float(np.sqrt(a[mask].mean())-np.sqrt(b[mask].mean())),'total_SSE_gain':float((a[mask]-b[mask]).sum()),'improved_IDs':int(np.sum(a[mask]>b[mask])),'worsened_IDs':int(np.sum(a[mask]<b[mask])),'unchanged_IDs':int(np.sum(a[mask]==b[mask]))}
 rows['all400']['conditional_CI95']=np.quantile(boot,[.025,.975]).tolist();rows['other397']['conditional_CI95']=np.quantile(rb,[.025,.975]).tolist();rows['empty3']['uncertainty']='n3 only; reportall3individualrows, no inferential subgroupCI'
 if fold is not None:rows['fold_gains']=[float(np.sqrt(a[fold==k].mean())-np.sqrt(b[fold==k].mean())) for k in range(5)]
 for i,draw in enumerate(draws[:23]):assert abs((np.sqrt(A[:,draw].mean())-np.sqrt(B[:,draw].mean()))-boot[i])<1e-11
 perid[tag]=a-b;return rows
for split,o in data.items():
 results[split]={}
 for mode,prefix in [('KRR',''),('frozen_ensemble','ensemble_')]:
  results[split][mode]={}
  for label,a,b in comparisons:results[split][mode][label]=compare((y-o[prefix+a])**2,(y-o[prefix+b])**2,split+'_'+mode+'_'+label,o['fold'])
results['pooled_ID_meanSE']={}
for mode,prefix in [('KRR',''),('frozen_ensemble','ensemble_')]:
 results['pooled_ID_meanSE'][mode]={}
 for label,a,b in comparisons:
  A=np.stack([(y-o[prefix+a])**2 for o in data.values()]);B=np.stack([(y-o[prefix+b])**2 for o in data.values()]);results['pooled_ID_meanSE'][mode][label]=compare(A,B,'pooled_'+mode+'_'+label)
emptyrows=[]
for i in np.flatnonzero(missing):
 emptyrows.append({'ID':str(z['ids'][i]),'hardness':float(y[i]),'old_HP':float(xx['A_current'][i,0]),'valid_HP':float(xx['C_valid_mean'][i,0]),'partitions':{split:{'fold':int(o['fold'][i]),'KRR':{a:float(o[a][i]) for a in s.ARMS},'ensemble':{a:float(o['ensemble_'+a][i]) for a in s.ARMS}} for split,o in data.items()}})
change={}
for split in s.SPLITS:
 change[split]={}
 for arm in ['B_validity','C_valid_mean']:
  changed=[]
  for k in range(5):
   a=s.read(_repo_path(R, split, f'fold-{k}/A_current/report.json'));b=s.read(_repo_path(R, split, f'fold-{k}', arm, 'report.json'))
   if a['chosen_index']!=b['chosen_index']:changed.append(k)
  change[split][arm]=changed
classes={}
for mode in ['KRR','frozen_ensemble']:
 q=results['pooled_ID_meanSE'][mode]['C-A'];gg=[results[split][mode]['C-A']['all400']['RMSE_gain'] for split in s.SPLITS];good=q['all400']['RMSE_gain']>0 and sum(v>0 for v in gg)>=2 and any(v>0 for v in gg[1:]) and q['other397']['RMSE_gain']>=0;classes[mode]={'consistent_proxy_signal':bool(good),'classification':'consistent_but_conditional' if good else 'weak_mixed_or_harmful','automaticpromotion':False}
np.savez_compressed(_repo_path(R, 'paired-results.npz'),ids=z['ids'],empty=missing,draws=draws,other397_draws=dr,**perid,**{k+'_boot':v for k,v in boots.items()});s.write(_repo_path(R, 'results.json'),{'results':results,'empty3_individuals':emptyrows,'hyperparameter_changed_folds':change,'prespecified_classification':classes,'statistics':'Paired fixed-fit conditionalID bootstrap10000; pooled eachID3splitmeanSE. Doesnot include fitting/seed/adaptiveselectionvariance. No scores fromholdout/test/Public.','primary':'C-A; B-A/C-Bmechanismcontrols','model_scope':'Directcompact26KRR branch only. Frozenensembleproxy holds residualphysics57/calibrators/anchors fixed. No fullnesteddownstream replacement claim.','training':s.read(_repo_path(R, 'training-complete.json')),'validation_KRR_fits':poisonfits,'unit_fixture_KRR_fits':1});s.protect();s.emit({'pooled_primary':{mode:results['pooled_ID_meanSE'][mode]['C-A'] for mode in ['KRR','frozen_ensemble']},'classes':classes,'seconds':time.monotonic()-start})
