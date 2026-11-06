"""Recover serialization only; verify saved development results without refitting."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,csv,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import lineal_experiment_054 as e
import numpy as np,joblib
from sklearn.model_selection import GroupKFold
fv=e.fv;out=e.OUT

def main():
 start=time.perf_counter();p=e.check();report=json.loads((out/'original/report.json').read_text());c=report['comparison']
 # Preserve failed files and original source/plan before serialization-only patch.
 source=_repo_path(ROOT, 'lineal_experiment_054.py');before=source.read_text();old="'advance':gain>boot.std() and sum(v>0 for v in fg)>=3";new="'advance':bool(gain>boot.std() and sum(v>0 for v in fg)>=3)"
 assert before.count(old)==1
 for path,dest in [(source,out/'experiment.before-serialization-fix.py'),(out/'plan.json',out/'plan.before-serialization-fix.json'),(out/'selection.json',out/'selection.failed-json.txt')]:
  with dest.open('xb') as f:f.write(path.read_bytes())
 source.write_text(before.replace(old,new));p['serialization_fix']={'reason':'numpy.bool_ could not be JSON serialized when gate false','scope':'bool cast only; no feature/model/prediction/selection-rule change','recovery':'selection reconstructed from already saved original/report.json; no refit','prior_source_sha256':p['sources']['lineal_experiment_054.py']};p['sources']['lineal_experiment_054.py']=fv.sha(source)
 (out/'plan.json').write_text(json.dumps(p,indent=2)+'\n')
 selection={'advance':bool(c['gain']>c['bootstrap_sd'] and c['positive_folds']>=3),'gain':c['gain'],'bootstrap_sd':c['bootstrap_sd'],'positive_folds':c['positive_folds'],'primary':'primary007','diagnostic_ineligible':True}
 (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n');e.check()
 fixture=e.tests.run();assert fixture['passed']
 with np.load(_repo_io(out/'development_features.npz'),allow_pickle=False) as z:ids=z['ids'].tolist();X=z['X'].copy();names=z['names'].tolist()
 assert ids==e.ids_only() and X.shape==(400,120) and len(set(names))==120 and np.isfinite(X).all()
 ex=json.loads((out/'extraction.json').read_text());assert fv.sha(out/'development_features.npz')==ex['feature_sha256']
 for j in (0,199,399):
  np.testing.assert_array_equal(e.f.extract(e.image(ids[j]))[0],X[j]);assert fv.sha(_repo_path(ROOT, 'data/train', f'{ids[j]}.png'))==ex['image_sha256'][ids[j]]
 with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;y=z['y'].copy();groups=z['groups'].copy()
 path=out/'original/development_predictions.npz';assert fv.sha(path)==report['predictions_sha256']
 with np.load(_repo_io(path),allow_pickle=False) as z:
  assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['y'],y);fid=z['fold'].copy();pred={n:z[n].copy() for n in report['scores']}
 with (_repo_path(ROOT, 'artifacts/seed-stability-052/original/development_oof.csv')).open() as f:rows=list(csv.DictReader(f))
 assert [r['ID'] for r in rows]==ids
 for name,col in [('reference007','candidate007_design'),('reference006','reference006')]:np.testing.assert_array_equal(pred[name],[float(r[col]) for r in rows])
 np.testing.assert_array_equal(pred['primary007'],.9*pred['reference007']+.1*pred['lineal']);np.testing.assert_array_equal(pred['diagnostic006'],.9*pred['reference006']+.1*pred['lineal'])
 errors=[];chosen=[]
 for k,log in enumerate(report['folds']):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(groups[tr])&set(groups[va])
  assert log['train_ids']==[ids[j] for j in tr] and log['validation_ids']==[ids[j] for j in va]
  for (a,b),(it,iv) in zip(log['inner_splits'],GroupKFold(n_splits=3,shuffle=True,random_state=20261221+k).split(X[tr],y[tr],groups[tr])):
   np.testing.assert_array_equal(a,tr[it]);np.testing.assert_array_equal(b,tr[iv]);assert not set(groups[a])&set(groups[b]);assert set(a)|set(b)==set(tr)
  assert log['chosen']==min(log['inner_scores'],key=lambda q:q['rmse'])
  mp=out/'original'/f'fold-{k}.joblib';assert fv.sha(mp)==log['model_sha256'];saved=joblib.load(_repo_io(mp));m=saved['model'];assert saved['train_ids']==log['train_ids'];assert saved['validation_ids']==log['validation_ids'];assert m.alpha==log['chosen']['alpha'];chosen.append(m.alpha)
  np.testing.assert_allclose(m.scaler.mean_,X[tr].mean(0),rtol=0,atol=1e-15);np.testing.assert_allclose(m.scaler.var_,X[tr].var(0),rtol=0,atol=1e-15);assert m.scaler.n_samples_seen_==320;assert m.ymean==float(y[tr].mean()) and m.ystd==float(y[tr].std())
  replay=m.predict(X[va]);np.testing.assert_array_equal(replay,pred['lineal'][va]);errors.append(float(abs(replay-pred['lineal'][va]).max()))
 for n,values in pred.items():
  assert fv.rmse(y,values)==report['scores'][n]['rmse'];assert [fv.rmse(y[fid==k],values[fid==k]) for k in range(5)]==report['scores'][n]['fold_rmse']
 ix=np.random.default_rng(20261222).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-pred['reference007'])[ix]**2,1))-np.sqrt(np.mean((y-pred['primary007'])[ix]**2,1));assert float(boot.std())==c['bootstrap_sd'];np.testing.assert_array_equal(np.percentile(boot,[2.5,97.5]),c['conditional_ci95'])
 assert not selection['advance'];assert not (out/'20261031').exists() and not (out/'20261101').exists();e.check()
 result={'passed':True,'fixtures':fixture,'development_rows':400,'features':120,'raw_feature_replay_rows':3,'model_reload_max_absolute_errors':errors,'chosen_alpha_by_fold':chosen,'fold_train_preprocessing_verified':True,'inner_group_splits_verified':True,'metrics_and_bootstrap_recomputed':True,'protected_artifacts_verified':len(p['protected']),'selection':selection,'scores':report['scores'],'seconds':time.perf_counter()-start,'scope':'No refit; no holdout/test image or label access; no candidate CSV or upload. Repeat partitions skipped per failed preregistered screen. Historical repeated-development selection bias remains.','independent_review':'audit_pipeline source review: serialization issue found; no other blocking issue found'}
 fv.write(_repo_path(ROOT, 'reports/lineal-054-validation.json'),result)
 summary={'outcome':'Negative fixed-panel experiment; retain best007','original007_rmse':report['scores']['reference007']['rmse'],'lineal_alone_rmse':report['scores']['lineal']['rmse'],'fixed_90_10_blend_rmse':report['scores']['primary007']['rmse'],'comparison':c,'extraction_seconds':ex['seconds'],'regression_seconds':report['seconds'],'validation_report':'reports/lineal-054-validation.json','repartition_run':False,'new_candidate':None,'limitation':'Rejects this raw-quantile 120-feature Ridge/10% blend, not all lineal-path or connectivity methods. No topology experiment performed.'}
 fv.write(_repo_path(ROOT, 'reports/lineal-054-summary.json'),summary)
 status_path=_repo_path(ROOT, 'reports/status.json');status=json.loads(status_path.read_text());status['status']='best007_preserved_lineal054_negative_verified';status['running_work']=[];status['latest_research']=summary;status_path.write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({'passed':True,'selection':selection,'reload_error':max(errors),'chosen_alpha':chosen,'seconds':result['seconds']}))
if __name__=='__main__':main()
