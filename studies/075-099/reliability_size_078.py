"""Same heldout rows: diagnose component reliability at213 versus320 training rows."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import contrast_calibration_077 as study
import numpy as np,torch,joblib
fv=study.fv;OUT=_repo_path(ROOT, 'artifacts/reliability-size-078')

def metrics(y,L,D,res009):
 return {'legacy_rmse':fv.rmse(y,L),'DINOmean_rmse':fv.rmse(y,D),'legacy_minus_DINO_MSE':float(np.mean((y-L)**2-(y-D)**2)),'contrast_std':float(np.std(D-L)),'contrast_residual009_correlation':float(np.corrcoef(D-L,res009)[0,1])}

def main():
 plan=study.check();v=json.loads((_repo_path(ROOT, 'reports/contrast-calibration-077-validation.json')).read_text());assert v['passed'] and not v['eligible_for_candidate'];OUT.mkdir(exist_ok=False);torch.set_num_threads(2);start=time.perf_counter();results={};foldrecords=[];hashes={};allarrays={}
 for split in ('original','20261031','20261101'):
  d=study.diag.load_outer(split);y=d['y'];ids=d['ids'];fid=d['fid'];L=np.full((3,400),np.nan);D=np.full((3,400),np.nan);folds=[]
  for k in range(5):
   outer_tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k)
   for j in range(3):
    cell=study.base.OUT/split/f'fold-{k}/inner-{j}';r=json.loads((cell/'report.json').read_text());tr=np.array(r['train_indices']);assert set(tr)<=set(outer_tr) and not set(tr)&set(va) and not set(d['g'][tr])&set(d['g'][va])
    for name,digest in r['files_sha256'].items():assert fv.sha(cell/name)==digest
    b=joblib.load(_repo_io(cell/'baseline.joblib'));assert b['train_ids']==[ids[i] for i in tr];baseline=sum(b['weights'][n]*m.predict(d['Xbase'][va]) for n,m in b['models'].items());b=joblib.load(_repo_io(cell/'kernel.joblib'));assert b['train_ids']==[ids[i] for i in tr];kernel=b['model'].predict(d['K'][va]);a=torch.load(_repo_io(cell/'cnn.pt'),weights_only=True,map_location='cpu');assert a['train_ids']==[ids[i] for i in tr];n=a['normalization'];m=fv.engine.TinyCNN().eval();m.load_state_dict(a['state_dict']);cnn=[]
    with torch.inference_mode():
     for first in range(0,len(va),32):cnn.extend((m(torch.tensor((d['images'][va[first:first+32]]-n['x_mean'])/max(n['x_std'],1e-6))).numpy()*n['y_std']+n['y_mean']).tolist())
    L[j,va]=.5*baseline+.25*kernel+.25*np.array(cnn);heads=[]
    for arm in ('S','B'):
     for seed in study.base.SEEDS:
      a=torch.load(_repo_io(cell/f'{arm}-{seed}.pt'),weights_only=True,map_location='cpu');assert a['train_ids']==[ids[i] for i in tr];m=study.base.h.Head(a['input_dim']);m.load_state_dict(a['model']);heads.append(study.base.h.predict(m,(d[arm][va]-a['scaler_mean'].numpy())/a['scaler_scale'].numpy(),a['target_mean'],a['target_std']))
    D[j,va]=np.mean(heads,0)
   full=metrics(y[va],d['legacy'][va],d['Dmean'][va],y[va]-d['reference009'][va]);small=metrics(np.tile(y[va],3),L[:,va].reshape(-1),D[:,va].reshape(-1),np.tile(y[va]-d['reference009'][va],3));folds.append({'fold':k,'size213_214_individual_models_pooled_repeated_rows':small,'size320':full,'warning':'The same80validation rows appear three times for213models. This is a descriptivepairedcomparison,not240independentrows. No averaging ofthreeinnerpredictions intoanewcandidate.'})
   print(json.dumps({'split':split,'completed_folds':k+1,'seconds':time.perf_counter()-start}),flush=True)
  res=y-d['reference009'];small=metrics(np.tile(y,3),L.reshape(-1),D.reshape(-1),np.tile(res,3));large=metrics(y,d['legacy'],d['Dmean'],res);results[split]={'size213_214':small,'size320':large,'MSE_advantage_change_320_minus213':large['legacy_minus_DINO_MSE']-small['legacy_minus_DINO_MSE'],'folds':folds};allarrays[f'{split}_legacy_small']=L;allarrays[f'{split}_DINO_small']=D
 with (_repo_path(OUT, 'predictions.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),**allarrays)
 study.check();report={'purpose':'Diagnose a specified training-size reliability mismatch after077 failure. No new calibration form/ratio/weight/regularization choice and no newsubmissionrecipe evaluated.','method':'For eachouterfold, predict its sameheldout80rows with eachofthreeexisting213/214-trainedinner models,comparecomponenterrorstoexisting320-trainedoutermodels. AllupstreamtrainIDs excludeheldout80; fold hasheschecked. CPUCNN reload error previouslyverified<=3.06e-5.',
 'results':results,'interpretation_limits':'Differences also reflect training subsets and fittedmodel variability. They do not establish that samplesize caused077failure, and do not authorizechoosinganewcrossfitfoldcount or coefficientfromouterresults. Existing050Ridgelearningcurves provideonlygeneraldata-sizecontext,notcurrentMLPcalibrationevidence. Same400repeatedlyused.',
 'predictions_sha256':fv.sha(_repo_path(OUT, 'predictions.npz')),'best009_sha256':fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-009.csv')),'source_sha256':fv.sha(Path(__file__)),'input_validation_sha256':fv.sha(_repo_path(ROOT, 'reports/contrast-calibration-077-validation.json')),'seconds':time.perf_counter()-start,'new_fits':0,'new_candidate':False,'holdout_test_access':False,'cost':0};fv.write(_repo_path(OUT, 'report.json'),report);print(json.dumps({s:{k:r[k] for k in ['size213_214','size320','MSE_advantage_change_320_minus213']} for s,r in results.items()}),flush=True)
if __name__=='__main__':main()
