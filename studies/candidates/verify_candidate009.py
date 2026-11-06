"""Full500 nested calibration replay and byte-exact final1000 CSV verification."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,csv,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import candidate009_pipeline as e
import numpy as np,torch,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
fv=e.fv

def main():
 decision=e.check();torch.set_num_threads(2);start=time.perf_counter();report=json.loads((e.OUT/'report.json').read_text());d=e.load_training();ids=d['ids'];y=d['y'];assert len(ids)==500 and len(set(ids))==500;assert fv.sha(_repo_path(ROOT, 'data/train.csv'))==report['train_csv_sha256'];freshids,freshy=e.cp.read_csv(_repo_path(ROOT, 'data/train.csv'),'TRAIN',500,True);assert freshids==ids;np.testing.assert_array_equal(freshy,y)
 for ident,digest in report['training_image_sha256'].items():assert fv.sha(_repo_path(ROOT, 'data/train', f'{ident}.png'))==digest
 assert fv.sha(e.OUT/'calibrator.json')==report['calibrator_sha256'];c=json.loads((e.OUT/'calibrator.json').read_text());assert c['train_ids']==ids and c['n_fit']==500 and c['decision_sha256']==fv.sha(e.OUT/'decision.json');assert fv.sha(e.OUT/'calibration_training.npz')==c['training_predictions_sha256']
 with np.load(_repo_io(e.OUT/'calibration_training.npz'),allow_pickle=False) as z:cached={n:z[n].copy() for n in z.files}
 assert cached['ids'].tolist()==ids;folds=list(GroupKFold(3,shuffle=True,random_state=e.study.INNER_SEED).split(d['S'],y,d['g']));ip=np.full(500,np.nan);cnnerror=0.;headcount=0
 for j,(tr,va) in enumerate(folds):
  cell=e.OUT/f'inner-{j}';r=json.loads((cell/'report.json').read_text());assert fv.sha(cell/'report.json')==report['inner_report_sha256'][f'inner-{j}/report.json'];assert r['train_indices']==tr.tolist() and r['validation_indices']==va.tolist() and r['outer_validation_indices']==[];assert not set(d['g'][tr])&set(d['g'][va]);assert np.all(cached['fold'][va]==j)
  for name,digest in r['files_sha256'].items():assert fv.sha(cell/name)==digest
  with np.load(_repo_io(cell/'predictions.npz'),allow_pickle=False) as z:p={n:z[n].copy() for n in z.files}
  np.testing.assert_array_equal(p['train_indices'],tr);np.testing.assert_array_equal(p['validation_indices'],va)
  a=joblib.load(_repo_io(cell/'baseline.joblib'));assert a['train_ids']==[ids[i] for i in tr];base=sum(a['weights'][n]*m.predict(d['Xbase'][va]) for n,m in a['models'].items());np.testing.assert_allclose(base,p['baseline'],atol=1e-10,rtol=0)
  a=joblib.load(_repo_io(cell/'kernel.joblib'));assert a['train_ids']==[ids[i] for i in tr];np.testing.assert_array_equal(a['model'].predict(d['K'][va]),p['kernel']);kp,km,kr=fv.kernel_fold(d['K'],y,d['g'],d['blocks'],tr,va,100*(j+1));assert kr==r['logs']['kernel'];np.testing.assert_array_equal(kp,p['kernel']);del km
  a=torch.load(_repo_io(cell/'cnn.pt'),weights_only=True,map_location='cpu');norm=a['normalization'];assert a['train_ids']==[ids[i] for i in tr] and a['validation_ids']==[ids[i] for i in va] and a['seed']==fv.CNN_SEED+100*(j+1);assert norm=={'x_mean':float(d['images'][tr].mean()),'x_std':float(d['images'][tr].std()),'y_mean':float(y[tr].mean()),'y_std':float(y[tr].std())};m=fv.engine.TinyCNN().eval();m.load_state_dict(a['state_dict']);cp=[]
  with torch.inference_mode():
   for first in range(0,len(va),32):
    x=torch.tensor((d['images'][va[first:first+32]]-norm['x_mean'])/max(norm['x_std'],1e-6));cp.extend((m(x).numpy()*norm['y_std']+norm['y_mean']).tolist())
  err=float(abs(np.array(cp)-p['cnn']).max());assert err<1e-3;cnnerror=max(cnnerror,err)
  for arm in ('S','B'):
   X=d[arm];sc=StandardScaler().fit(X[tr]);ym=float(y[tr].mean());ys=float(y[tr].std());edges=np.quantile(y[tr],np.arange(1,7)/7)
   for seed in e.study.SEEDS:
    file=cell/f'{arm}-{seed}.pt';a=torch.load(_repo_io(file),weights_only=True,map_location='cpu');rec=r['logs'][arm][str(seed)];assert fv.sha(file)==rec['checkpoint_sha256'];assert a['train_ids']==[ids[i] for i in tr] and a['validation_ids']==[ids[i] for i in va] and a['seed']==seed+100*(j+1) and a['target_mean']==ym and a['target_std']==ys and a['config']==e.study.h.CONFIG and a['effective_aux_weight']==0
    for key,val in [('scaler_mean',sc.mean_),('scaler_scale',sc.scale_),('scaler_var',sc.var_),('bin_edges_unused',edges)]:np.testing.assert_array_equal(a[key].numpy(),val)
    assert e.study.h.state_hash(a['model'])==rec['final_state_sha256'];m=e.study.h.Head(X.shape[1]);m.load_state_dict(a['model']);pp=e.study.h.predict(m,sc.transform(X[va]),ym,ys);np.testing.assert_array_equal(pp,p[f'{arm}_{seed}']);headcount+=1
   np.testing.assert_array_equal(p[arm+'mean'],np.mean([p[f'{arm}_{seed}'] for seed in e.study.SEEDS],0))
  ip[va]=.375*p['baseline']+.1875*p['kernel']+.1875*p['cnn']+.125*p['Smean']+.125*p['Bmean'];np.testing.assert_array_equal(ip[va],p['ensemble008'])
 np.testing.assert_array_equal(ip,cached['inner_oof']);center=float(ip.mean());coef=np.linalg.lstsq(np.column_stack([np.ones(500),ip-center]),y,rcond=None)[0];np.testing.assert_allclose(coef,[c['target_center'],c['slope']],atol=1e-10,rtol=0);assert c['prediction_center']==center and c['target_center']==y.mean()
 # Refitonewholeinnerrecipe afterperturbingitsheldouttargets; no datafileismodified.
 tr,va=folds[0];dd=dict(d);dd['y']=y.copy();dd['y'][va]=10000+np.arange(len(va));rp,rr=e.study.fit_cell(dd,tr,va,np.array([],dtype=int),0,0,e.OUT/'validation-replay');replayerr=float(abs(rp-ip[va]).max());assert replayerr<2e-4
 original=json.loads((e.OUT/'inner-0/report.json').read_text())
 for arm in ('S','B'):
  for seed in e.study.SEEDS:assert original['logs'][arm][str(seed)]['final_state_sha256']==rr['logs'][arm][str(seed)]['final_state_sha256']
 # Finaltestinferenceonly fromimmutableverified008basepredictions.
 ids_test,_=e.cp.read_csv(_repo_path(ROOT, 'data/sample_submission.csv'),'TEST',1000,False);manifest=json.loads(e.CSV.with_suffix('.manifest.json').read_text());assert fv.sha(e.CSV)==manifest['submission_sha256'];assert fv.sha(e.CSV.with_suffix('.components.csv'))==manifest['components_sha256'];assert fv.sha(_repo_path(ROOT, 'data/sample_submission.csv'))==manifest['sample_submission_sha256'];assert fv.sha(e.OUT/'report.json')==manifest['model_report_sha256'];assert fv.sha(e.OUT/'calibrator.json')==manifest['calibrator_sha256'];assert fv.sha(e.OUT/'decision.json')==manifest['decision_sha256']
 with (_repo_path(ROOT, 'artifacts/submission-candidate-008.csv')).open(newline='') as f:old=list(csv.DictReader(f))
 with e.CSV.open(newline='') as f:reader=csv.DictReader(f);assert reader.fieldnames==['ID','hardness'];rows=list(reader)
 with e.CSV.with_suffix('.components.csv').open(newline='') as f:parts=list(csv.DictReader(f))
 assert [a['ID'] for a in old]==[a['ID'] for a in rows]==[a['ID'] for a in parts]==ids_test and len(set(ids_test))==1000;base=np.array([float(r['hardness']) for r in old]);assert fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-008.csv'))==decision['best008_sha256'];pred=c['target_center']+c['slope']*(base-c['prediction_center']);assert np.isfinite(pred).all();np.testing.assert_array_equal(pred,[float(r['hardness']) for r in rows]);np.testing.assert_array_equal(base,[float(r['candidate008']) for r in parts]);np.testing.assert_array_equal(pred,[float(r['calibrated009']) for r in parts])
 reproduced=e.CSV.with_suffix('.reproduced.csv')
 with reproduced.open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids_test,pred))
 assert fv.sha(reproduced)==manifest['submission_sha256'];oldmanifest=json.loads((_repo_path(ROOT, 'artifacts/submission-candidate-008.manifest.json')).read_text())
 for ident,digest in oldmanifest['test_image_sha256'].items():assert fv.sha(_repo_path(ROOT, 'data/test', f'{ident}.png'))==digest
 e.check();result={'valid':True,'candidate':'009','submission_sha256':fv.sha(e.CSV),'rows':1000,'columns':['ID','hardness'],'exact_sample_order':True,'finite':True,'duplicates':0,'CSV_byte_exact_reproduction':True,'full500calibration_training_predictions_exact':True,'independent_OLS_solution_verified':True,'inner_models_reloaded':{'baseline_pipelines':9,'kernel':3,'CNN':3,'MLP':headcount},'CNN_CPU_MPS_max_error':cnnerror,'whole_inner_recipe_refit_withheldoutlabelperturbation_max_error':replayerr,'all6refitMLPstates_exact':True,'all1500_image_hashes_verified':True,'slope':c['slope'],'prediction_range':[float(pred.min()),float(pred.max())],'best008_sha256':decision['best008_sha256'],'uploaded':False,'cost':0,'source_sha256':fv.sha(Path(__file__)),'seconds':time.perf_counter()-start};fv.write(_repo_path(ROOT, 'reports/candidate009-verification.json'),result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
