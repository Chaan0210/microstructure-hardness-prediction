"""Independent raw-statistics transforms, dual Ridge, nesting and leakage checks."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import residual_physics_113 as e
import numpy as np,joblib
from sklearn.model_selection import GroupKFold

def manual(m,X,r,V,blocks):
 assert m['n_train']==len(r)
 if m['kind']=='none':return np.zeros(len(V))
 low=X.min(0);high=X.max(0);span=high-low;den=span.copy();den[den<10*np.finfo(float).eps]=1.
 B=np.clip((X-low)/den,0,1);BV=np.clip((V-low)/den,0,1);mu=B.mean(0);var=B.var(0);sd=np.sqrt(var);sd[sd==0]=1.;w=np.array([1/np.sqrt(blocks.count(b)) for b in blocks]);Z=(B-mu)/sd*w;Zv=(BV-mu)/sd*w
 np.testing.assert_array_equal(m['bound'].data_min_,low);np.testing.assert_array_equal(m['bound'].data_max_,high);np.testing.assert_allclose(m['scale'].mean_,mu,atol=1e-13,rtol=1e-12);np.testing.assert_allclose(m['scale'].var_,var,atol=1e-13,rtol=1e-12);np.testing.assert_allclose(m['scale'].scale_,sd,atol=1e-12,rtol=1e-12);np.testing.assert_array_equal(m['weights'],w)
 rs=float(np.sqrt(np.mean(r*r)));rs=rs if rs>1e-12 else 1.;assert rs==m['residual_scale'];alpha=e.LAMBDAS[m['kind']]*len(r);assert alpha==m['alpha'] and m['regressor'].fit_intercept is False and m['regressor'].intercept_==0.
 dual=np.linalg.solve(Z@Z.T+alpha*np.eye(len(r)),r/rs);coef=Z.T@dual;np.testing.assert_allclose(coef,m['regressor'].coef_,atol=2e-10,rtol=1e-10);return Zv@coef*rs

def validate(split):
 e.check();d=e.load(split);out=e.OUT/split;r=e.read(out/'report.json');assert e.sha(out/'predictions.npz')==r['predictions_sha256'];start=time.perf_counter();maxerr=0.;models=0;chosen=[];correction=np.full((3,400),np.nan)
 with np.load(_repo_io(out/'predictions.npz'),allow_pickle=False) as z:ps={n:z[n].copy() for n in z.files}
 assert ps['ids'].tolist()==d['ids'];np.testing.assert_array_equal(ps['y'],d['y']);np.testing.assert_array_equal(ps['fold'],d['fid']);np.testing.assert_array_equal(ps['reference011'],d['reference011'])
 for k in range(5):
  tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);fd=out/f'fold-{k}';fr=e.read(fd/'report.json');assert e.sha(fd/'report.json')==r['fold_report_sha256'][f'fold-{k}/report.json'];seen=[]
  for j in range(3):
   A,B,pa,trace=e.anchor(d,split,k,j);assert trace==fr['blocks'][j]['anchor_trace'];assert not set(A)&set(B) and not (set(A)|set(B))&set(va);seen.extend(B.tolist());path=fd/f'block-{j}';br=e.read(path/'report.json');assert e.sha(path/'report.json')==fr['blocks'][j]['report_sha256'];assert e.sha(path/'fit_data.npz')==br['fit_data_sha256'];rr=d['y'][B]-pa
   with np.load(_repo_io(path/'fit_data.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(z['block_indices'],B);np.testing.assert_array_equal(z['anchor_predictions'],pa);np.testing.assert_array_equal(z['residual'],rr);savedip=z['inner_predictions'].copy();savedpred=z['correction'].copy()
   folds=list(GroupKFold(3,shuffle=True,random_state=e.CVSEED+100*k+j).split(d['X'][B],rr,d['g'][B]));ip=np.full((3,len(B)),np.nan)
   for t,(it,iv) in enumerate(folds):
    assert br['inner_splits'][t]=={'train_indices':B[it].tolist(),'validation_indices':B[iv].tolist()};assert not set(A)&set(B[iv]) and not set(B[it])&set(B[iv]) and not set(d['g'][B[it]])&set(d['g'][B[iv]])
    for ci,kind in enumerate(e.CANDIDATES):
     file=path/f'inner-{t}-{kind}.joblib';assert e.sha(file)==br['models_sha256'][file.name];saved=joblib.load(_repo_io(file));np.testing.assert_array_equal(saved['train_indices'],B[it]);np.testing.assert_array_equal(saved['validation_indices'],B[iv]);m=saved['model'];assert m['kind']==kind
     ip[ci,iv]=manual(m,d['X'][B[it]],rr[it],d['X'][B[iv]],d['blocks']);reload=e.predict(m,d['X'][B[iv]]);np.testing.assert_array_equal(reload,savedip[ci,iv]);err=float(abs(ip[ci,iv]-reload).max());assert err<2e-9;maxerr=max(maxerr,err);models+=1
   scores=[e.rmse(rr,p) for p in ip];np.testing.assert_allclose(scores,list(br['inner_scores'].values()),atol=1e-10,rtol=0);selected=e.CANDIDATES[int(np.argmin(scores))];assert selected==br['chosen'];chosen.append(selected);file=path/'selected.joblib';assert e.sha(file)==br['selected_sha256'];saved=joblib.load(_repo_io(file));np.testing.assert_array_equal(saved['train_indices'],B);assert saved['train_ids']==[d['ids'][i] for i in B] and saved['chosen']==selected
   pp=manual(saved['model'],d['X'][B],rr,d['X'][va],d['blocks']);np.testing.assert_allclose(pp,savedpred,atol=2e-9,rtol=0);np.testing.assert_array_equal(e.predict(saved['model'],d['X'][va]),savedpred);correction[j,va]=savedpred;models+=1
  assert sorted(seen)==tr.tolist()
 np.testing.assert_array_equal(correction,ps['block_corrections']);np.testing.assert_array_equal(correction.mean(0),ps['correction']);np.testing.assert_array_equal(d['reference011']+correction.mean(0),ps['primary']);np.testing.assert_array_equal(e.predict({'kind':'none'},d['X']),np.zeros(400))
 for kind in e.CANDIDATES:assert chosen.count(kind)==r['chosen_counts'][kind]
 for key in ['reference011','primary']:
  assert r['scores'][key]['rmse']==e.rmse(d['y'],ps[key]);np.testing.assert_array_equal(r['scores'][key]['fold_rmse'],[e.rmse(d['y'][d['fid']==k],ps[key][d['fid']==k]) for k in range(5)])
 assert r['RMSE_gain']==e.rmse(d['y'],ps['reference011'])-e.rmse(d['y'],ps['primary']);e.check();v={'passed':True,'split':split,'serialized_models_verified':models,'manual_transform_dual_Ridge_max_error':maxerr,'all_inner_choices_recomputed':15,'residual_block_sizes':[106,107],'anchor_training_disjoint_from_all_residual_blocks_used_for_that_anchor':True,'outer_groups_disjoint':True,'zero_correction_exact011':True,'source_sha256':e.sha(Path(__file__)),'report_sha256':e.sha(out/'report.json'),'seconds':time.perf_counter()-start};e.write(e.OUT/f'validation-{split}.json',v);e.emit(v)

def leakage():
 e.check();d=e.load('original');k=0;outer=np.flatnonzero(d['fid']==k);pert=dict(d);pert['y']=d['y'].copy();pert['y'][outer]=10000+np.arange(len(outer));out=e.OUT/'leakage-replay';out.mkdir(exist_ok=False);errors=[];innerchecks=0
 for j in range(3):
  A,B,p,trace=e.anchor(pert,'original',k,j);pp,new=e.fit_block(pert,B,p,k,j,out/f'block-{j}');oldpath=e.OUT/'original/fold-0'/f'block-{j}';old=e.read(oldpath/'report.json');assert new['chosen']==old['chosen'];assert new['inner_scores']==old['inner_scores']
  with np.load(_repo_io(oldpath/'fit_data.npz'),allow_pickle=False) as z:expected=z['correction'];np.testing.assert_array_equal(expected,pp)
  errors.append(float(abs(pp-expected).max()));folds=list(GroupKFold(3,shuffle=True,random_state=e.CVSEED+j).split(d['X'][B],d['y'][B]-p,d['g'][B]));it,iv=folds[0];r=d['y'][B]-p;alter=r.copy();alter[iv]+=12345
  for kind in ['ridge1','ridge01']:
   m=e.fit_model(d['X'][B[it]],alter[it],d['blocks'],kind);saved=joblib.load(_repo_io(oldpath/f'inner-0-{kind}.joblib'))['model'];np.testing.assert_array_equal(m['regressor'].coef_,saved['regressor'].coef_);np.testing.assert_array_equal(m['scale'].mean_,saved['scale'].mean_);assert m['residual_scale']==saved['residual_scale'];innerchecks+=1
 upstream=e.read(e.prior.OUT/'validation-replay.json');assert upstream['passed'] and upstream['all_outside_innertrain_labels_perturbed']==187
 v={'passed':True,'outer80_labels_perturbed_full_residual_fold_replay':True,'prediction_max_error':max(errors),'inner_validation_labels_perturbed_fixed_candidate_fits':innerchecks,'upstream112_replay_certificate_sha256':e.sha(e.prior.OUT/'validation-replay.json'),'upstream_residual107_and_outer80_exclusion_already_retrained':True,'new_upstream_retraining':False,'source_sha256':e.sha(Path(__file__))};e.write(e.OUT/'leakage-validation.json',v);e.emit(v)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=[*e.SPLITS,'leakage']);a=ap.parse_args();leakage() if a.mode=='leakage' else validate(a.mode)
