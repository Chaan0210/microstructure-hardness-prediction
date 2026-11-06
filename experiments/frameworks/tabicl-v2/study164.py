"""One fixed nested-OOF compact-kernel residual TabICL probe, development only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import study162 as e
from kernel164 import BalancedKernel
from sklearn.model_selection import GroupKFold
import numpy as np,json,time,sys,resource,gc
BASE=e.P;R=e.R;P=BASE/'study164'
PAR={**e.PARAMS,'n_estimators':8,'norm_methods':['none','power'],'feat_shuffle_method':'latin','outlier_threshold':4.0}
ARM='compact26_dinob_pca64';GRID=[(w,a) for w in [.25,1.,4.] for a in [.01,.1,1.]]
BLOCKS=json.loads((R/'artifacts/run-012/report.json').read_text())['blocks']

def data(split):
 ids,C,B,y,f,ref,sources=e.inputs(split)
 d=np.load(_repo_io(R/'artifacts/run-012/development_features.npz'),allow_pickle=False);ix={s:i for i,s in enumerate(d['ids'])};g=d['groups'][[ix[s] for s in ids]]
 return dict(ids=ids,C=C,B=B,y=y,fold=f,ref=ref,g=g,sources=sources)

def splits(d,T,k):
 return [(T[a],T[b]) for a,b in GroupKFold(3,shuffle=True,random_state=20261005+k).split(d['C'][T],groups=d['g'][T])]

def anchor(d,A,H,V,k,j,out):
 out.mkdir(exist_ok=False,parents=True);X,y,g=d['C'],d['y'],d['g'];inners=[(A[a],A[b]) for a,b in GroupKFold(3,shuffle=True,random_state=20261005+100*k+j).split(X[A],groups=g[A])]
 assert not set(A)&(set(H)|set(V));scores=[];preds=[];records=[];hashes={}
 for ci,(w,a) in enumerate(GRID):
  p=np.full(len(A),np.nan);look={int(i):z for z,i in enumerate(A)}
  for ell,(tr,va) in enumerate(inners):
   assert not set(g[tr])&set(g[va]);m=BalancedKernel(BLOCKS,w,a).fit(X[tr],y[tr]);p[[look[int(i)] for i in va]]=m.predict(X[va])
   file=out/f'candidate-{ci}-inner-{ell}.joblib';e.joblib.dump({'model':m,'train':tr,'validation':va},_repo_io(file));hashes[file.name]=e.sha(file)
  preds.append(p);scores.append(float(np.mean((y[A]-p)**2)))
 chosen=int(np.argmin(scores));w,a=GRID[chosen];m=BalancedKernel(BLOCKS,w,a).fit(X[A],y[A]);ph=m.predict(X[H]);pv=m.predict(X[V])
 e.joblib.dump({'model':m,'train':A,'validation':H,'outer':V},_repo_io(out/'selected.joblib'));hashes['selected.joblib']=e.sha(out/'selected.joblib')
 np.savez_compressed(_repo_io(out/'predictions.npz'),A=A,H=H,V=V,inner_predictions=np.stack(preds),anchor_H=ph,anchor_V=pv)
 report={'A_ids':d['ids'][A].tolist(),'H_ids':d['ids'][H].tolist(),'V_ids':d['ids'][V].tolist(),'inner_splits':[{'train':a.tolist(),'validation':b.tolist()} for a,b in inners],'grid':GRID,'inner_MSE':scores,'chosen':chosen,'selected_parameters':{'bandwidth':w,'alpha':a},'hashes':hashes,'predictions_sha256':e.sha(out/'predictions.npz'),'own_H_labels_excluded_from_selection_and_fit':True}
 e.write(out/'report.json',report);return ph,pv

def fold(d,split,k,out):
 out.mkdir(exist_ok=False,parents=True);start=time.perf_counter();cpu=time.process_time();T=np.where(d['fold']!=k)[0];V=np.where(d['fold']==k)[0];look={int(i):z for z,i in enumerate(T)};oof=np.full(len(T),np.nan);query=[];seen=[]
 for j,(A,H) in enumerate(splits(d,T,k)):
  ph,pv=anchor(d,A,H,V,k,j,out/f'anchor-{j}');oof[[look[int(i)] for i in H]]=ph;query.append(pv);seen.extend(H.tolist())
 assert sorted(seen)==T.tolist() and np.isfinite(oof).all();base=np.mean(query,0);res=d['y'][T]-oof;anchor_seconds=time.perf_counter()-start
 xt,xv,transform=e.features(ARM,d['C'],d['B'],T,V)
 # All transforms and residual labels belong only to outer train. No residual-model inner selection.
 model=e.TabICLRegressor(**PAR);a=time.perf_counter();model.fit(xt,res);fit_seconds=time.perf_counter()-a
 a=time.perf_counter();cor=model.predict(xv);predict_seconds=time.perf_counter()-a;pred=base+cor;blend=.8*d['ref'][V]+.2*pred;zero_blend=.8*d['ref'][V]+.2*base
 assert np.isfinite(pred).all();assert {k:len(v) for k,v in model.ensemble_generator_.ensemble_configs_.items()}=={'none':4,'power':4}
 model.save(_repo_io(out/'residual.pkl'),save_model_weights=False,save_training_data=True,save_kv_cache=False);e.joblib.dump(transform,_repo_io(out/'transform.joblib'))
 np.savez_compressed(_repo_io(out/'predictions.npz'),train=T,outer=V,ids=d['ids'][V],anchor_oof=oof,residual=res,anchor_query=np.stack(query),base=base,correction=cor,primary=pred,fixed20blend=blend,zero_blend=zero_blend)
 r={'split':split,'fold':k,'train_ids':d['ids'][T].tolist(),'validation_ids':d['ids'][V].tolist(),'n_tabicl_residual_train':len(T),'residual_mean':float(res.mean()),'residual_std':float(res.std()),'anchor_seconds':anchor_seconds,'tabicl_fit_seconds':fit_seconds,'tabicl_predict_seconds':predict_seconds,'wall_seconds':time.perf_counter()-start,'cpu_process_seconds':time.process_time()-cpu,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'RMSE':{'kernel_mean3':e.rmse(d['y'][V],base),'residual_primary':e.rmse(d['y'][V],pred),'reference018':e.rmse(d['y'][V],d['ref'][V]),'fixed20blend':e.rmse(d['y'][V],blend),'zero_blend':e.rmse(d['y'][V],zero_blend)},'files':{n:e.sha(out/n) for n in ['predictions.npz','residual.pkl','transform.joblib']},'anchor_reports':{f'anchor-{j}/report.json':e.sha(out/f'anchor-{j}/report.json') for j in range(3)}}
 e.write(out/'report.json',r);print('FOLD_DONE',split,k,json.dumps({q:v for q,v in r.items() if q not in ['train_ids','validation_ids','files','anchor_reports']}),flush=True);del model;gc.collect();return r

def summarize(split):
 d=data(split);f=d['fold'];y=d['y'];preds={x:np.full(400,np.nan) for x in ['base','primary','fixed20blend','zero_blend']}
 for k in range(5):
  z=np.load(_repo_io(P/split/f'fold-{k}/predictions.npz'),allow_pickle=False);V=np.where(f==k)[0];assert np.array_equal(z['ids'],d['ids'][V])
  for x in preds:preds[x][V]=z[x]
 assert all(np.isfinite(v).all() for v in preds.values());scores={x:e.rmse(y,v) for x,v in preds.items()};scores['reference018']=e.rmse(y,d['ref']);gains=[e.rmse(y[f==k],d['ref'][f==k])-e.rmse(y[f==k],preds['fixed20blend'][f==k]) for k in range(5)]
 gate=scores['reference018']-scores['fixed20blend']>=.01 and sum(v>0 for v in gains)>=3 and scores['primary']<scores['base'] and scores['fixed20blend']<scores['zero_blend']
 r={'split':split,'RMSE':scores,'fold_blend_gains':gains,'resplit_gate_passed':gate,'correction_improves_matched_base':scores['primary']<scores['base']}
 if split=='original':r['direct_target_study163']=json.loads((BASE/'study163/results/original'/ARM/'summary.json').read_text())
 np.savez_compressed(_repo_io(P/split/'oof.npz'),ids=d['ids'],y=y,fold=f,reference018=d['ref'],**preds);r['oof_sha256']=e.sha(P/split/'oof.npz');e.write(P/split/'summary.json',r);print('SUMMARY',json.dumps(r),flush=True);return r

def prepare():
 P.mkdir(exist_ok=False);d=data('original');protected=[R/'artifacts/submission-candidate-018.csv',R/'artifacts/submission-candidate-019.csv',R/'artifacts/submission-candidate-020.csv',R/'deliverables/reproduction-018-v155.zip',R/'deliverables/reproduction-011-014-015-v136.zip',BASE/'requirements.lock',e.CK]
 protected+=list((BASE/'results').rglob('*json'))+list((BASE/'study163').rglob('*json'))
 keep={str(p.relative_to(R)):e.sha(p) for p in protected};keep.update(json.loads((BASE/'preservation-before.json').read_text()))
 plan={'study':164,'hypothesis':'Fixed8 TabICL can model image/physics information remaining in out-of-fold compact26 nonlinear KRR residuals. One90-column residual arm only.','novelty':'119/158 pooled residual learning already exists with fixedrbf1 on calibrated multimodel residuals;164 changes nuisance to compact26 nestedKRR and learner to TabICL. No prior TabICL residual fit found;162/163 direct-target only.','small_sample_assessment':'TabICL gets320 residual rows per outer fold, not106/107. Cached official package description reports pretraining from300 samples. KRR anchors train213/214 and their own innerCV trains142/143, matching existing nestedKRR practice. Overlapping pseudo-targets are dependent; no independent sample-size claim.','nesting':'OuterT320/V80. Three GroupKFold splits A213/214,H106/107 withinT. Each KRR bandwidth/alpha selected only with3-foldCV wholly insideA; refitA predictsH andV. Residual for everyi inH=y_i-KRR_A(x_i), thus ownlabel and allouterV excluded throughout nuisance selection. Pool320 residuals. Fixed TabICL fitT with no residual innerCV, no weight/normalization/seed selection.','query_base':'Mean of same3 A-trained KRR predictions atV + unit TabICL correction; avoids full320-versus213 anchor-training-size mismatch, but mean3 query variance differs from one-anchor OOF residual.','features':'compact26 + existingDINO-B StandardScaler/PCA64. PCA fits outerT only; used only in TabICL that has no innerCV.','parameters':PAR,'kernel_grid':GRID,'kernel_blocks':BLOCKS,'crossfit_seed':'20261005+outerfold','kernel_selection_seed':'20261005+100*outerfold+anchor_index','primary':'0.8 best018 OOF +0.2 (mean3 compactKRR + predicted residual). Fixed before164 outcomes.','controls':'Mean3KRR withoutcorrection and its20% blend; existing163 directtarget90 predictions. No control fallback selection.','resplit_gate':'primaryblend improves018 RMSE>=.01 and>=3/5folds, AND residualstandalone improves matchedmean3KRR AND blend improves matchedzero-correctionblend; same frozen arm only then run existing2resplits.','final_gate':'sameconditionsonall3splits and independentlineage/poison/reload audit; otherwise no500/test/CSV.','resource_gate':'firstwholeouterfold<=60wallseconds,RSS<4e9;samecheckeveryfold; CPU2. No new installs/downloads/APIs/cost.','scope':'400development cached features only; holdout/test not read. 018 and all reproduction material preserved.','inputs':d['sources'],'source_sha256':{str(p.relative_to(R)):e.sha(p) for p in [BASE/'study164.py',BASE/'kernel164.py',BASE/'study162.py',R/'compact_regression.py',R/'frozen_validation.py']},'protected':keep,'limitations':['Many adaptive trials on same400 IDs','Pooledcrossfit residuals cannot be naively inner-CV tuned; no such selector here','This is distinct learner/anchor, not a claim of new pooled residual principle']}
 e.write(P/'plan.json',plan);print('PLAN_LOCKED',e.sha(P/'plan.json'),flush=True)

def main():
 mode=sys.argv[1]
 if mode=='prepare':prepare();return
 plan=json.loads((P/'plan.json').read_text())
 for name,h in plan['source_sha256'].items():assert e.sha(R/name)==h
 split='original' if mode=='benchmark' else mode
 if split not in ['original','20261031','20261101']:raise ValueError(mode)
 if split!='original':assert json.loads((P/'original/summary.json').read_text())['resplit_gate_passed']
 d=data(split)
 for k in ([0] if mode=='benchmark' else range(5)):
  out=P/split/f'fold-{k}'
  if (out/'report.json').exists():r=json.loads((out/'report.json').read_text())
  else:print('START',split,k,flush=True);r=fold(d,split,k,out)
  assert r['wall_seconds']<=60 and r['peak_rss_bytes']<4e9,'resource stop'
 if mode!='benchmark':summarize(split)
if __name__=='__main__':main()
