"""Strict development-only verification of069 pre-LayerNorm representation."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import dinos_coarse_070 as e
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
fv=e.fv

def verify_features():
 ids,tokens,ex=e.token_inputs();r=json.loads((e.OUT/'extraction.json').read_text());assert fv.sha(e.OUT/'development_Scoarse.npz')==r['feature_sha256']
 with np.load(_repo_io(e.OUT/'development_Scoarse.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;X=z['X'].copy()
 maxerr=0.
 for i,t in enumerate(tokens):
  p=t.astype(float).reshape(36,36,384);parts=[]
  for grid in (2,4):
   n=36//grid;regions=np.stack([p[a*n:(a+1)*n,b*n:(b+1)*n].mean((0,1)) for a in range(grid) for b in range(grid)]);parts.append(regions.var(0)/(p.var((0,1))+1e-4))
  val=np.concatenate(parts);np.testing.assert_allclose(val,X[i],atol=1e-12,rtol=1e-10);maxerr=max(maxerr,float(abs(val-X[i]).max()));np.testing.assert_array_equal(e.coarse(p),X[i])
 return {'rows_independently_recomputed':400,'loop_vs_reshape_max_error':maxerr,'token_cache_verified_sha256':ex['tokens_sha256'],'upstream_tokens_reextracted400_exact':'reports/fisher-067-validation.json'}

def main():
 plan=e.check();torch.set_num_threads(2);start=time.perf_counter();feature=verify_features();results={};count=0;replays=[]
 for split in ('original','20261031','20261101'):
  out=e.OUT/split
  if not out.exists():continue
  ids,X,y,g,fid,r003,r007,Smean,Bmean,ref=e.inputs(split);r=json.loads((out/'report.json').read_text());assert fv.sha(out/'development_predictions.npz')==r['predictions_sha256']
  with np.load(_repo_io(out/'development_predictions.npz'),allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
  assert d['ids'].tolist()==ids;np.testing.assert_array_equal(d['y'],y);np.testing.assert_array_equal(d['fold'],fid);np.testing.assert_array_equal(d['reference008'],ref);np.testing.assert_array_equal(d['reference003'],r003);np.testing.assert_array_equal(d['Smean'],Smean);np.testing.assert_array_equal(d['B504mean'],Bmean);errors=[];trainerrors=[]
  for k in range(5):
   tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(g[tr])&set(g[va]);sc=StandardScaler().fit(X[tr]);ym=float(y[tr].mean());ys=float(y[tr].std());edges=np.quantile(y[tr],np.arange(1,7)/7);fd=out/f'fold-{k}';log=json.loads((fd/'report.json').read_text());oldlog=json.loads((_repo_path(ROOT, 'artifacts/seed-stability-052', split, f'fold-{k}/report.json')).read_text());tp=[]
   for si,seed in enumerate(e.SEEDS):
    path=fd/f'seed-{seed}.pt';rec=log['models'][str(seed)];assert fv.sha(path)==rec['checkpoint_sha256'];a=torch.load(_repo_io(path),map_location='cpu',weights_only=True);assert a['train_ids']==[ids[j] for j in tr] and a['validation_ids']==[ids[j] for j in va] and a['seed']==seed+k and a['input_dim']==4224 and a['effective_aux_weight']==0. and a['config']==e.h.CONFIG
    for key,actual in [('scaler_mean',sc.mean_),('scaler_scale',sc.scale_),('scaler_var',sc.var_),('bin_edges_unused',edges)]:np.testing.assert_array_equal(a[key].numpy(),actual)
    assert a['target_mean']==ym and a['target_std']==ys and e.h.state_hash(a['model'])==rec['final_state_sha256'];assert rec['batch_order_sha256']==oldlog['models'][str(seed)]['batch_order_sha256'] and rec['classifier_max_parameter_change']==0.;m=e.h.Head(4224);m.load_state_dict(a['model']);pv=e.h.predict(m,sc.transform(X[va]),ym,ys);np.testing.assert_array_equal(pv,d['seed_predictions'][si,va]);assert fv.rmse(y[va],pv)==rec['validation_rmse'];errors.append(float(abs(pv-d['seed_predictions'][si,va]).max()));tp.append(e.h.predict(m,sc.transform(X[tr]),ym,ys));count+=1
    if k==0 and si==0:
     _,_,_,xt,yt,bt,_=e.prior.old.prep(X[tr],y[tr]);re,rr=e.h.fit(xt,yt,bt,seed+k,0.)
     for key in ('initial_state_sha256','batch_order_sha256','final_state_sha256','curves'):assert rr[key]==rec[key]
     np.testing.assert_array_equal(e.h.predict(re,sc.transform(X[va]),ym,ys),pv);replays.append({'split':split,'seed':seed,'full100epoch_state_exact':True,'prediction_error':0.})
   trainerrors.extend((np.mean(tp,axis=0)-y[tr]).tolist())
  np.testing.assert_array_equal(d['Scoarse'],d['seed_predictions'].mean(0));np.testing.assert_array_equal(d['primary'],.75*r003+.125*d['Scoarse']+.125*Bmean);assert float(np.sqrt(np.mean(np.array(trainerrors)**2)))==r['mean3_train_rmse_pooled']
  for key,score in r['scores'].items():assert score=={'rmse':fv.rmse(y,d[key]),'fold_rmse':[fv.rmse(y[fid==k],d[key][fid==k]) for k in range(5)]}
  new=d['primary'];fg=[fv.rmse(y[fid==k],ref[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];ix=np.random.default_rng(20270104).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-ref)[ix]**2,1))-np.sqrt(np.mean((y-new)[ix]**2,1));c={'gain':fv.rmse(y,ref)-fv.rmse(y,new),'fold_gains':fg,'positive_folds':sum(a>0 for a in fg),'bootstrap_sd':float(boot.std()),'conditional_ci95':np.percentile(boot,[2.5,97.5]).tolist()};assert c==r['comparison'];assert float(np.corrcoef(y-d['Scoarse'],y-Smean)[0,1])==r['standalone_Scoarse_S504_residual_correlation'];results[split]={'scores':r['scores'],'comparison':c,'max_reload_error':max(errors),'train_rmse':r['mean3_train_rmse_pooled'],'head_residual_correlation':r['standalone_Scoarse_S504_residual_correlation'],'report_sha256':fv.sha(out/'report.json'),'predictions_sha256':r['predictions_sha256']}
 sel=json.loads((e.OUT/'selection.json').read_text());assert sel['advance']==(results['original']['comparison']['gain']>0)
 if not sel['advance']:assert set(results)=={'original'}
 eligible=len(results)==3 and all(r['comparison']['gain']>0 for r in results.values());e.check();v={'passed':True,'feature_verification':feature,'results':results,'models_verified':count,'full_replays':replays,'eligible_for_candidate':eligible,'gate':'All3 pooledOOF strictly improve008, exact reload/replay; no foldcount/bootstrap hard gate.','protected_files_verified':len(plan['protected']),'best008_sha256':fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-008.csv')),'validation_source_sha256':fv.sha(Path(__file__)),'seconds':time.perf_counter()-start,'holdout_test_access':False,'candidate_created':False,'cost':0,'new_packages':0,'limits':plan['bias']};name='dinos-coarse-070-validation.json' if not sel['advance'] or len(results)==3 else 'dinos-coarse-070-screen-validation.json';fv.write(_repo_path(ROOT, 'reports', name),v);print(json.dumps(v),flush=True)
if __name__=='__main__':main()
