"""Independent nested leakage, estimator, scaler, prediction and OLS checks."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import nested_calibration_074 as e
import numpy as np,torch,joblib
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
fv=e.fv

def main():
 plan=e.check();torch.set_num_threads(2);start=time.perf_counter();results={};cells=0;heads=0;maxcnn=0.;maxbase=0.;replays=[]
 for split in ('original','20261031','20261101'):
  out=e.OUT/split
  if not (out/'report.json').exists():continue
  d=e.inputs(split);ids=d['ids'];y=d['y'];fid=d['fid'];ref=d['reference008'];r=json.loads((out/'report.json').read_text());assert fv.sha(out/'development_predictions.npz')==r['predictions_sha256']
  with np.load(_repo_io(out/'development_predictions.npz'),allow_pickle=False) as z:pred={k:z[k].copy() for k in z.files}
  assert pred['ids'].tolist()==ids;np.testing.assert_array_equal(pred['y'],y);np.testing.assert_array_equal(pred['fold'],fid);np.testing.assert_array_equal(pred['reference008'],ref);corrected=np.full(400,np.nan);coefs=[]
  for k in range(5):
   tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);fd=out/f'fold-{k}';fr=json.loads((fd/'report.json').read_text());assert fv.sha(fd/'report.json')==r['fold_report_sha256'][f'fold-{k}/report.json'];assert fr['train_ids']==[ids[i] for i in tr] and fr['validation_ids']==[ids[i] for i in va];inner=list(GroupKFold(3,shuffle=True,random_state=e.INNER_SEED+k).split(d['S'][tr],y[tr],d['g'][tr]));ip=np.full(len(tr),np.nan)
   for j,(it,iv) in enumerate(inner):
    ti=tr[it];vi=tr[iv];cell=fd/f'inner-{j}';cr=json.loads((cell/'report.json').read_text());assert fv.sha(cell/'report.json')==fr['inner_report_sha256'][f'inner-{j}/report.json'];assert cr['train_indices']==ti.tolist() and cr['validation_indices']==vi.tolist() and cr['outer_validation_indices']==va.tolist();assert cr['train_ids']==[ids[i] for i in ti] and cr['validation_ids']==[ids[i] for i in vi];assert not set(ti)&set(va) and not set(vi)&set(va) and not set(d['g'][ti])&set(d['g'][vi]) and not set(d['g'][tr])&set(d['g'][va])
    for file,sha in cr['files_sha256'].items():assert fv.sha(cell/file)==sha
    with np.load(_repo_io(cell/'predictions.npz'),allow_pickle=False) as z:p={n:z[n].copy() for n in z.files}
    np.testing.assert_array_equal(p['train_indices'],ti);np.testing.assert_array_equal(p['validation_indices'],vi)
    a=joblib.load(_repo_io(cell/'baseline.joblib'));assert a['train_ids']==[ids[i] for i in ti] and a['weights']==d['weights'];base=sum(a['weights'][n]*m.predict(d['Xbase'][vi]) for n,m in a['models'].items());maxbase=max(maxbase,float(abs(base-p['baseline']).max()));np.testing.assert_allclose(base,p['baseline'],atol=1e-10,rtol=0)
    fresh={n:clone(m).fit(d['Xbase'][ti],y[ti]) for n,m in d['recipes'].items()};bp=sum(d['weights'][n]*m.predict(d['Xbase'][vi]) for n,m in fresh.items());np.testing.assert_allclose(bp,p['baseline'],atol=1e-10,rtol=0);del fresh
    a=joblib.load(_repo_io(cell/'kernel.joblib'));assert a['train_ids']==[ids[i] for i in ti];np.testing.assert_array_equal(a['model'].predict(d['K'][vi]),p['kernel']);kp,km,kr=fv.kernel_fold(d['K'],y,d['g'],d['blocks'],ti,vi,k+100*(j+1));assert kr==cr['logs']['kernel'];np.testing.assert_array_equal(kp,p['kernel']);del a,km
    a=torch.load(_repo_io(cell/'cnn.pt'),weights_only=True,map_location='cpu');norm=a['normalization'];assert a['train_ids']==[ids[i] for i in ti] and a['validation_ids']==[ids[i] for i in vi] and a['seed']==fv.CNN_SEED+k+100*(j+1);assert norm=={'x_mean':float(d['images'][ti].mean()),'x_std':float(d['images'][ti].std()),'y_mean':float(y[ti].mean()),'y_std':float(y[ti].std())};m=fv.engine.TinyCNN().eval();m.load_state_dict(a['state_dict']);cp=[]
    with torch.inference_mode():
     for first in range(0,len(vi),32):
      xx=torch.tensor((d['images'][vi[first:first+32]]-norm['x_mean'])/max(norm['x_std'],1e-6));cp.extend((m(xx).numpy()*norm['y_std']+norm['y_mean']).tolist())
    ce=float(abs(np.array(cp)-p['cnn']).max());assert ce<1e-3;maxcnn=max(maxcnn,ce)
    for arm in ('S','B'):
     X=d[arm];sc=StandardScaler().fit(X[ti]);ym=float(y[ti].mean());ys=float(y[ti].std());edges=np.quantile(y[ti],np.arange(1,7)/7)
     for seed in e.SEEDS:
      path=cell/f'{arm}-{seed}.pt';a=torch.load(_repo_io(path),weights_only=True,map_location='cpu');rec=cr['logs'][arm][str(seed)];assert fv.sha(path)==rec['checkpoint_sha256'];assert a['train_ids']==[ids[i] for i in ti] and a['validation_ids']==[ids[i] for i in vi] and a['seed']==seed+k+100*(j+1) and a['target_mean']==ym and a['target_std']==ys and a['config']==e.h.CONFIG and a['effective_aux_weight']==0 and a['input_dim']==X.shape[1]
      for key,actual in [('scaler_mean',sc.mean_),('scaler_scale',sc.scale_),('scaler_var',sc.var_),('bin_edges_unused',edges)]:np.testing.assert_array_equal(a[key].numpy(),actual)
      assert e.h.state_hash(a['model'])==rec['final_state_sha256'];m=e.h.Head(X.shape[1]);m.load_state_dict(a['model']);pp=e.h.predict(m,sc.transform(X[vi]),ym,ys);np.testing.assert_array_equal(pp,p[f'{arm}_{seed}']);heads+=1
     np.testing.assert_array_equal(p[arm+'mean'],np.mean([p[f'{arm}_{seed}'] for seed in e.SEEDS],0))
    ip[iv]=.375*p['baseline']+.1875*p['kernel']+.1875*p['cnn']+.125*p['Smean']+.125*p['Bmean'];np.testing.assert_array_equal(ip[iv],p['ensemble008']);cells+=1
    if k==0 and j==0:
     perturbed=dict(d);perturbed['y']=y.copy();perturbed['y'][va]=10000+np.arange(len(va));rp,rr=e.fit_cell(perturbed,ti,vi,va,k,j,e.OUT/f'validation-replay-{split}');err=float(abs(rp-p['ensemble008']).max());assert err<2e-4
     for arm in ('S','B'):
      for seed in e.SEEDS:
       assert rr['logs'][arm][str(seed)]['final_state_sha256']==cr['logs'][arm][str(seed)]['final_state_sha256']
     replays.append({'split':split,'outer_validation_targets_perturbed_only_in_memory':True,'calibration_fit_data_unchanged':True,'entire_inner_recipe_refit_max_prediction_error':err,'all6MLPstates_exact':True})
   path=fd/'calibration.npz';assert fv.sha(path)==fr['calibration_sha256']
   with np.load(_repo_io(path),allow_pickle=False) as z:c={n:z[n].copy() for n in z.files}
   np.testing.assert_array_equal(c['train_indices'],tr);np.testing.assert_array_equal(c['validation_indices'],va);np.testing.assert_array_equal(c['inner_oof'],ip)
   # Independent QR/SVD least-squares solution on320train-onlycalibrationpairs.
   center=float(ip.mean());coef=np.linalg.lstsq(np.column_stack([np.ones(len(tr)),ip-center]),y[tr],rcond=None)[0];np.testing.assert_allclose(coef,[fr['target_center'],fr['slope']],atol=1e-10,rtol=0);assert c['prediction_center']==center and c['target_center']==y[tr].mean();assert c['slope']==fr['slope'];pp=coef[0]+coef[1]*(ref[va]-center);np.testing.assert_allclose(pp,pred['primary'][va],atol=1e-10,rtol=0);np.testing.assert_array_equal(c['outer_prediction'],pred['primary'][va]);corrected[va]=c['outer_prediction'];coefs.append(fr['slope'])
  np.testing.assert_array_equal(corrected,pred['primary']);assert coefs==r['calibration_slopes'];comp=e.prior.old.comparison(y,ref,corrected,fid);assert comp==r['comparison']
  for n,ss in r['scores'].items():assert ss=={'rmse':fv.rmse(y,pred[n]),'fold_rmse':[fv.rmse(y[fid==k],pred[n][fid==k]) for k in range(5)]}
  results[split]={'scores':r['scores'],'comparison':comp,'slopes':coefs,'predictions_sha256':r['predictions_sha256'],'report_sha256':fv.sha(out/'report.json')}
 selection=json.loads((e.OUT/'selection.json').read_text());assert selection['advance']==(results['original']['comparison']['gain']>0);eligible=len(results)==3 and all(r['comparison']['gain']>0 for r in results.values());e.check();report={'passed':True,'results':results,'inner_models_verified':{'CNN':cells,'MLP':heads,'baseline_pipelines':3*cells,'kernel':cells},'all_baseline_and_nested_kernel_refits':cells,'max_CNN_CPU_MPS_reload_error':maxcnn,'max_baseline_reload_error':maxbase,'baseline_reload_tolerance':1e-10,'validator_revision':'v2: allow tiny floating-point tree-reduction error; all models/results unchanged','whole_recipe_refit_leakage_stress_checks':replays,'OLS_independently_recomputed':True,'eligible_for_candidate':eligible,'protected008_sha256':fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-008.csv')),'holdout_test_access':False,'cost':0,'seconds':time.perf_counter()-start,'validation_source_sha256':fv.sha(Path(__file__)),'limits':plan['bias']}
 name='nested-calibration-074-validation.json' if not selection['advance'] or len(results)==3 else 'nested-calibration-074-screen-validation.json';fv.write(_repo_path(ROOT, 'reports', name),report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
