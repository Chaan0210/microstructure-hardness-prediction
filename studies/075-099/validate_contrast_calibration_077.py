"""Audit all upstream training scopes and independently solve extended calibration."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import contrast_calibration_077 as e
import numpy as np,torch,joblib
fv=e.fv

def main():
 plan=e.check();torch.set_num_threads(2);start=time.perf_counter();prior=json.loads((_repo_path(ROOT, 'reports/nested-calibration-074-validation.json')).read_text());assert prior['passed'];results={};models=0;coefficients=0;maxerr=0.;stress=None
 for split in ('original','20261031','20261101'):
  out=e.OUT/split
  if not (out/'report.json').exists():continue
  d=e.diag.load_outer(split);y=d['y'];ids=d['ids'];fid=d['fid'];r=json.loads((out/'report.json').read_text());assert fv.sha(out/'development_predictions.npz')==r['predictions_sha256']
  with np.load(_repo_io(out/'development_predictions.npz'),allow_pickle=False) as z:pred={k:z[k].copy() for k in z.files}
  assert pred['ids'].tolist()==ids;np.testing.assert_array_equal(pred['y'],y);np.testing.assert_array_equal(pred['fold'],fid);np.testing.assert_array_equal(pred['reference009'],d['reference009']);actual=np.full(400,np.nan)
  for k in range(5):
   tr,va,A,C=e.diag.inner_matrix(d,split,k);fd=out/f'fold-{k}';fr=json.loads((fd/'report.json').read_text());assert fr==r['folds'][k];assert fv.sha(fd/'model.json')==fr['model_sha256'];assert fv.sha(fd/'fit_data.npz')==fr['fit_data_sha256'];m=json.loads((fd/'model.json').read_text());assert m['train_ids']==[ids[i] for i in tr] and m['validation_ids']==[ids[i] for i in va] and m['n_train']==320
   with np.load(_repo_io(fd/'fit_data.npz'),allow_pickle=False) as z:q={n:z[n].copy() for n in z.files}
   np.testing.assert_array_equal(q['train_indices'],tr);np.testing.assert_array_equal(q['validation_indices'],va);np.testing.assert_array_equal(q['inner_A'],A);outer=np.column_stack([d['reference008'][va],d['contrast'][va]]);np.testing.assert_array_equal(q['outer_A'],outer)
   mean=A.mean(0);scale=A.std(0);Z=(A-mean)/scale;ym=float(y[tr].mean());coef=np.linalg.solve(Z.T@Z,Z.T@(y[tr]-ym));np.testing.assert_array_equal(m['mean'],mean);np.testing.assert_array_equal(m['scale'],scale);assert m['target_mean']==ym;np.testing.assert_allclose(coef,m['standardized_coefficients'],atol=1e-10,rtol=0);independent=ym+(outer-mean)/scale@coef;pp=e.predict_meta(m,outer);err=float(abs(pp-independent).max());maxerr=max(maxerr,err);assert err<1e-10;np.testing.assert_array_equal(pp,q['predictions']);np.testing.assert_array_equal(pp,pred['primary'][va]);actual[va]=pp;coefficients+=1
   perturbed=y.copy();perturbed[va]=10000+np.arange(len(va));mm=e.fit_meta(A,perturbed[tr]);assert mm=={key:m[key] for key in mm};np.testing.assert_array_equal(e.predict_meta(mm,outer),pp)
   pc=A[:,0]-A[:,0].mean();slope=float(pc@(y[tr]-ym)/(pc@pc));control=ym+slope*(d['reference008'][va]-A[:,0].mean());np.testing.assert_allclose(control,d['reference009'][va],atol=1e-10,rtol=0);np.testing.assert_array_equal(control,q['control009']);np.testing.assert_array_equal(control,pred['control009'][va])
   for j in range(3):
    cell=e.base.OUT/split/f'fold-{k}/inner-{j}';cr=json.loads((cell/'report.json').read_text());ti=np.array(cr['train_indices']);vi=np.array(cr['validation_indices']);assert not set(ti)&set(va) and not set(vi)&set(va);assert set(ti)|set(vi)==set(tr)
    for name,digest in cr['files_sha256'].items():assert fv.sha(cell/name)==digest
    for name in ('baseline.joblib','kernel.joblib'):
     a=joblib.load(_repo_io(cell/name));assert a['train_ids']==[ids[i] for i in ti];models+=3 if name=='baseline.joblib' else 1
    for path in cell.glob('*.pt'):
     a=torch.load(_repo_io(path),map_location='cpu',weights_only=True);assert a['train_ids']==[ids[i] for i in ti] and a['validation_ids']==[ids[i] for i in vi];models+=1
    if split=='original' and k==0 and j==0:
     dd=dict(d);dd['y']=perturbed;_,rr=e.base.fit_cell(dd,ti,vi,va,k,j,e.OUT/'upstream-leakage-replay')
     with np.load(_repo_io(e.OUT/'upstream-leakage-replay/predictions.npz'),allow_pickle=False) as z:rep=np.column_stack([z['ensemble008'],.5*(z['Smean']+z['Bmean'])-(.5*z['baseline']+.25*z['kernel']+.25*z['cnn'])])
     pos=np.array([list(tr).index(i) for i in vi]);delta=float(abs(rep-A[pos]).max());assert delta<1e-4
     for arm in ('S','B'):
      for seed in e.base.SEEDS:assert rr['logs'][arm][str(seed)]['final_state_sha256']==cr['logs'][arm][str(seed)]['final_state_sha256']
     stress={'outer_validation_labels_perturbed_in_memory':True,'full_upstream_refit_aggregate_and_contrast_max_error':delta,'all6MLP_states_exact':True,'new_meta_parameters_unchanged_for_all_checked_folds':True}
  np.testing.assert_array_equal(actual,pred['primary']);comp=e.base.prior.old.comparison(y,d['reference009'],actual,fid);assert comp==r['comparison']
  for n,s in r['scores'].items():assert s=={'rmse':fv.rmse(y,pred[n]),'fold_rmse':[fv.rmse(y[fid==k],pred[n][fid==k]) for k in range(5)]}
  results[split]={'scores':r['scores'],'comparison':comp,'control009_max_error':float(abs(pred['control009']-d['reference009']).max()),'predictions_sha256':r['predictions_sha256'],'report_sha256':fv.sha(out/'report.json')}
 selection=json.loads((e.OUT/'selection.json').read_text());assert selection['advance']==(results['original']['comparison']['gain']>0);eligible=len(results)==3 and all(r['comparison']['gain']>0 for r in results.values());e.check();v={'passed':True,'results':results,'upstream_estimators_hash_and_scope_checked':models,'upstream_numeric_verification_source':'reports/nested-calibration-074-validation.json','independent_meta_solutions':coefficients,'max_independent_solve_prediction_error':maxerr,'leakage_stress':stress,'eligible_for_candidate':eligible,'best009_sha256':fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-009.csv')),'new_packages':0,'cost':0,'holdout_test_access':False,'seconds':time.perf_counter()-start,'source_sha256':fv.sha(Path(__file__)),'limits':plan['bias']};name='contrast-calibration-077-validation.json' if not selection['advance'] or len(results)==3 else 'contrast-calibration-077-screen-validation.json';fv.write(_repo_path(ROOT, 'reports', name),v);print(json.dumps(v),flush=True)
if __name__=='__main__':main()
