"""Audit two fixed representation studies without reselecting any recipe."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import adapted_mlp_059 as a
import pca_mlp_060 as p
import numpy as np,torch,joblib
from sklearn.preprocessing import StandardScaler
fv=a.fv

def run(number):
 e=a if number=='059' else p;plan=e.check();torch.set_num_threads(2);t=time.perf_counter();results={};count=0;feature_errs=[];replay_error=None;anchors={}
 if number=='059':
  saved=json.loads((_repo_path(ROOT, 'artifacts/dino-multilayer-044/plan.json')).read_text());anchors={n:h for n,h in saved['protected'].items() if n.startswith('artifacts/dino-lastblock-043/')};assert len(anchors)==37
  for n,h in anchors.items():assert fv.sha(_repo_path(ROOT, n))==h,n
  oldv=json.loads((_repo_path(ROOT, 'reports/dino-043-validation.json')).read_text());assert oldv['models_reloaded']==15 and oldv['all_frozen_norm_head_tensors_unchanged'] and oldv['fold_scaler_target_group_ids_checkpoint_and_head_hashes_verified'];backbone=a.adapt.ds.base.model().to('mps')
 for split in ('original','20261031','20261101'):
  out=e.OUT/split
  if not out.exists():continue
  ids,X,y,g,fid,r003,r007,oldmean=e.base.inputs(split);report=json.loads((out/'report.json').read_text());assert fv.sha(out/'development_predictions.npz')==report['predictions_sha256']
  with np.load(_repo_io(out/'development_predictions.npz')) as z:assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['fold'],fid);np.testing.assert_array_equal(z['y'],y);sp=z['seed_predictions'].copy();new=z['primary'].copy()
  if number=='059':tokens=a.cache(ids)
  errors=[];trainerrors=[];ext=[]
  for k in range(5):
   tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(g[tr])&set(g[va]);fd=out/f'fold-{k}';log=json.loads((fd/'report.json').read_text());old=json.loads((_repo_path(ROOT, 'artifacts/seed-stability-052', split, f'fold-{k}', 'report.json')).read_text())
   if number=='059':
    assert fv.sha(fd/'features.npz')==log['feature_sha256']
    with np.load(_repo_io(fd/'features.npz')) as z:assert z['ids'].tolist()==ids;A=z['X'].copy()
    np.testing.assert_array_equal(A[:,:2304],X[:,:2304]);tail,r,rec=a.tail_for(backbone,split,k,ids,X,y,fid);assert rec['checkpoint_sha256']==log['adapted_backbone_sha256'];ix=np.array([tr[0],va[0]]);fresh=a.features(tail,tokens,ix);err=float(abs(fresh-A[ix,-1152:]).max());np.testing.assert_allclose(fresh,A[ix,-1152:],rtol=0,atol=2e-4);feature_errs.append(err);np.testing.assert_allclose(r.predict(A[va,-1152:]),rec['predictions'],rtol=0,atol=2e-3);del tail
    sc,ym,ys,xt,yt=e.base.prep(A[tr],y[tr]);xv=sc.transform(A[va]);bins=e.head.preprocess(A[tr],y[tr])[-1]
   else:
    tf=fd/'transform.joblib';assert fv.sha(tf)==log['transform_sha256'];obj=joblib.load(_repo_io(tf));sc=obj['scaler'];pc=obj['pca'];assert obj['train_ids']==[ids[i] for i in tr] and obj['validation_ids']==[ids[i] for i in va];refsc=StandardScaler().fit(X[tr]);np.testing.assert_array_equal(sc.mean_,refsc.mean_);np.testing.assert_array_equal(sc.scale_,refsc.scale_);assert pc.n_samples_==320 and pc.n_components_==64 and not pc.whiten;np.testing.assert_allclose(pc.mean_,sc.transform(X[tr]).mean(0),rtol=0,atol=1e-14);np.testing.assert_allclose(pc.components_@pc.components_.T,np.eye(64),rtol=0,atol=1e-10);Z=e.projected(sc,pc,X[tr]);np.testing.assert_allclose(pc.inverse_transform(pc.transform(Z)),Z,rtol=0,atol=1e-10);scores=pc.transform(sc.transform(X[tr]));np.testing.assert_allclose(scores.var(0,ddof=1),pc.explained_variance_,rtol=1e-8,atol=1e-10);ym=float(y[tr].mean());ys=float(y[tr].std());xt=torch.tensor(Z,dtype=torch.float32);yt=torch.tensor((y[tr]-ym)/ys,dtype=torch.float32);xv=e.projected(sc,pc,X[va]);bins=e.h.preprocess(X[tr],y[tr])[-1]
    if split=='original' and k==0:
     _,pc2,_,_,xt2,_=e.prepare_fold(X,y,tr);np.testing.assert_allclose(xt2.numpy(),xt.numpy(),rtol=0,atol=1e-6);feature_errs.append(float(abs(xt2.numpy()-xt.numpy()).max()))
   head=e.head if number=='059' else e.h;tp=[]
   for si,seed in enumerate(e.SEEDS):
    cp=fd/f'seed-{seed}.pt';record=log['models'][str(seed)];ref=old['models'][str(seed)];assert fv.sha(cp)==record['checkpoint_sha256'];ck=torch.load(_repo_io(cp),weights_only=True,map_location='cpu');assert ck['train_ids']==[ids[i] for i in tr] and ck['validation_ids']==[ids[i] for i in va];assert ck['seed']==seed+k and ck['target_mean']==ym and ck['target_std']==ys;assert record['initial_state_sha256']==ref['initial_state_sha256'] and record['batch_order_sha256']==ref['batch_order_sha256'];assert head.state_hash(ck['model'])==record['final_state_sha256']
    if number=='059':np.testing.assert_array_equal(ck['scaler_mean'].numpy(),sc.mean_);np.testing.assert_array_equal(ck['scaler_scale'].numpy(),sc.scale_);assert ck['backbone_sha256']==log['adapted_backbone_sha256']
    else:assert ck['transform_sha256']==log['transform_sha256']
    model=head.Head();model.load_state_dict(ck['model']);pv=head.predict(model,xv,ym,ys);np.testing.assert_array_equal(pv,sp[si,va]);errors.append(float(abs(pv-sp[si,va]).max()));tp.append(head.predict(model,xt,ym,ys));count+=1
    if split=='original' and k==0 and si==0:
     replay,rr=head.fit(xt,yt,torch.tensor(bins),seed,0.);assert head.state_hash(replay.state_dict())==head.state_hash(ck['model']);rp=head.predict(replay,xv,ym,ys);np.testing.assert_array_equal(rp,pv);replay_error=float(abs(rp-pv).max())
   trainerrors.extend((np.mean(tp,axis=0)-y[tr]).tolist());mean=sp[:,va].mean(0);ext.append(int(((mean<y[tr].min())|(mean>y[tr].max())).sum()))
  mean=sp.mean(0);np.testing.assert_array_equal(new,.75*r003+.25*mean);assert fv.rmse(y,new)==report['scores']['primary']['rmse'];assert float(np.sqrt(np.mean(np.array(trainerrors)**2)))==report['train_rmse_pooled'];c=report['comparison'];assert c['gain']==fv.rmse(y,r007)-fv.rmse(y,new);fg=[fv.rmse(y[fid==k],r007[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];assert fg==c['fold_gains'];seed=20261230 if number=='059' else 20261231;ix=np.random.default_rng(seed).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-r007)[ix]**2,1))-np.sqrt(np.mean((y-new)[ix]**2,1));assert float(boot.std())==c['bootstrap_sd'];np.testing.assert_array_equal(np.percentile(boot,[2.5,97.5]),c['conditional_ci95']);results[split]={'scores':report['scores'],'comparison':c,'train_rmse':report['train_rmse_pooled'],'model_reload_max_error':max(errors),'standalone_labelrange_exceedances':sum(ext),'seconds':report['seconds']}
 sel=json.loads((e.OUT/'selection.json').read_text());c=results['original']['comparison'];assert sel['advance']==bool(c['gain']>c['bootstrap_sd'] and c['positive_folds']>=3)
 if not sel['advance']:assert set(results)=={'original'}
 e.check();v={'passed':True,'experiment':number,'results':results,'models_verified':count,'feature_reproduction_max_error':max(feature_errs),'independent100epoch_replay_error':replay_error,'protected_files':len(plan['protected']),'043_historical_anchor_files_verified':len(anchors),'seconds':time.perf_counter()-t,'scope':'Development400only,noholdout/test/candidateCSV/upload/install/cost; no alternative hyperparameters selected.','bias':fv.CAVEAT};fv.write(_repo_path(ROOT, f'reports/representation-{number}-validation.json'),v);print(json.dumps(v),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('number',choices=['059','060']);run(ap.parse_args().number)
