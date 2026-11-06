"""Recover only interrupted080 fold execution; immutable study and recipe remain unchanged."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,datetime,shutil,subprocess
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import dinob_adapter_080 as e
import numpy as np,torch
fv=e.fv

def main():
 e.check();torch.set_num_threads(2);split='20261101';out=e.OUT/split;assert not (out/'report.json').exists();assert json.loads((e.OUT/'selection.json').read_text())['advance'];d=e.inputs(split);tokens=e.get_cache(d);m=e.backbone.model();completed={}
 for k in range(3):
  fd=out/f'fold-{k}';r=json.loads((fd/'report.json').read_text());tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);assert r['config']==e.CFG and r['train_ids']==[d['ids'][i] for i in tr] and r['validation_ids']==[d['ids'][i] for i in va];assert fv.sha(fd/'model.pt')==r['checkpoint_sha256'] and fv.sha(fd/'predictions.npz')==r['predictions_sha256'];completed[f'fold-{k}/report.json']=fv.sha(fd/'report.json')
 partial=out/'fold-3';assert partial.is_dir() and not list(partial.iterdir());archived=out/'fold-3-interrupted-before-checkpoint';assert not archived.exists();partial.rename(archived);oldlog=_repo_path(ROOT, 'reports/dinob-adapter-080-20261101.log');snapshot=_repo_path(ROOT, 'reports/dinob-adapter-080-20261101-interrupted.log');assert not snapshot.exists();shutil.copyfile(_repo_io(oldlog),_repo_io(snapshot))
 recovery={'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'error':'write_stdin: exec-server transport disconnected; failed to resume session: recovery timed out after25s. PID72332 initially alive, later PID72332 and shell72069 absent; all080 process scan then empty.','cause':'Unknown; no Python traceback or exitcode was delivered. Do not label it OOM or a model failure.','completed_folds_verified':completed,'last_logged_progress':'fold3 epoch1, before a checkpoint; preserved emptypartialfolder and log snapshot.','recovery':'Reuse completedfolds0,1,2 unchanged; refit unsavedfold3 fromsameinitialseed forfixed8epochs; runfold4. No recipe,source,selection criterion orcompletedresult changes.','partial_folder':str(archived.relative_to(ROOT)),'log_sha256':fv.sha(snapshot),'source_sha256':fv.sha(Path(__file__))};fv.write(_repo_path(ROOT, 'reports/dinob-adapter-080-connection-recovery.json'),recovery);start=time.perf_counter()
 for k in (3,4):e.fit_fold(d,tokens,m,k,out/f'fold-{k}');torch.mps.empty_cache()
 adapted=np.full(400,np.nan);new=np.full(400,np.nan);slopes=[]
 for k in range(5):
  fd=out/f'fold-{k}';r=json.loads((fd/'report.json').read_text());assert fv.sha(fd/'predictions.npz')==r['predictions_sha256'];va=np.flatnonzero(d['fid']==k)
  with np.load(_repo_io(fd/'predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(z['validation_indices'],va);p=z['adapted'].copy()
  adapted[va]=p;fr=json.loads((e.previous.base.OUT/split/f'fold-{k}/report.json').read_text());s=fr['slope'];slopes.append(s);new[va]=d['reference009'][va]+s*.125*(p-d['Bmean'][va])
 ps={'reference009':d['reference009'],'frozenB':d['Bmean'],'adaptedB':adapted,'primary':new};comp=e.prior.old.comparison(d['y'],d['reference009'],new,d['fid'])
 with (out/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],**ps)
 r={'split':split,'scores':{n:{'rmse':fv.rmse(d['y'],p),'fold_rmse':[fv.rmse(d['y'][d['fid']==k],p[d['fid']==k]) for k in range(5)]} for n,p in ps.items()},'comparison':comp,'unchanged009_slopes':slopes,'fold_report_sha256':{f'fold-{k}/report.json':fv.sha(out/f'fold-{k}/report.json') for k in range(5)},'predictions_sha256':fv.sha(out/'development_predictions.npz'),'seconds_excluding_reused_benchmark':time.perf_counter()-start,'timing_scope':'Recovery execution only; originalcompletedfold durations remain ineachfoldreport. Interruptedepoch compute not included.','recovery_report_sha256':fv.sha(_repo_path(ROOT, 'reports/dinob-adapter-080-connection-recovery.json')),'new_candidate':False};fv.write(out/'report.json',r);e.check();print(json.dumps(r),flush=True)
if __name__=='__main__':main()
