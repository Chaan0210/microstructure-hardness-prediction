"""Sequential095 runner; fails closed and never chooses an alternate arm."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import sys,subprocess,json,datetime,time,traceback,os
ROOT=_RepoPath(__file__).resolve().parents[1];PY=_repo_path(ROOT, '.venv/bin/python');OUT=_repo_path(ROOT, 'artifacts/early-adapter-095')
def stage(script,arg):
 path=_repo_path(ROOT, f'reports/early-adapter-095-{arg}')+Path('') if False else _repo_path(ROOT, f'reports/early-adapter-095-{arg}-{Path(script).stem}.log')
 print(json.dumps({'starting':script,'arg':arg,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'log':str(path)}),flush=True)
 with path.open('xb') as f:r=subprocess.run([str(PY),'-I','-B',str(_repo_path(_repo_io(ROOT), script)),arg],cwd=_repo_io(ROOT),stdout=f,stderr=subprocess.STDOUT)
 print(json.dumps({'finished':script,'arg':arg,'exit':r.returncode}),flush=True)
 if r.returncode:raise RuntimeError(f'{script} {arg} exit{r.returncode}')
def main():
 start=time.time();stamp=datetime.datetime.now(datetime.timezone.utc);b=json.loads((_repo_path(OUT, 'benchmark.json')).read_text());record={'pid':os.getpid(),'started_utc':stamp.isoformat(),'estimated_original_done_utc':(stamp+datetime.timedelta(seconds=b['estimated_all3arms_5fold_seconds'])).isoformat(),'conservative_original_done_utc':(stamp+datetime.timedelta(seconds=b['resource_gate_estimate_seconds_with25pct_margin'])).isoformat(),'stage':'original_training'}
 with (_repo_path(OUT, 'run-start.json')).open('x') as f:json.dump(record,f,indent=2)
 stage('train_early_adapter_095.py','original');stage('validate_early_adapter_095.py','original');r=json.loads((_repo_path(OUT, 'original/report.json')).read_text());done=['original']
 if r['advance']:
  for split in ['20261031','20261101']:stage('train_early_adapter_095.py',split);stage('validate_early_adapter_095.py',split);done.append(split)
 reports={sp:json.loads((_repo_path(OUT, sp, 'report.json')).read_text()) for sp in done};eligible=len(done)==3 and all(q['advance'] and q['fixed395_gain_vs011']>0 for q in reports.values());summary={'completed_partitions':done,'eligible_research_result':eligible,'decision':'Reportfixedearlyhypothesis result;preserve011.No candidateorupload.' if eligible else 'Rejectthisfixedearlyadapterdirection;no rank/layer/epoch/seed/ratiosearch,no latefallback.','results':reports,'validation':{sp:json.loads((_repo_path(ROOT, f'reports/early-adapter-095-{sp}-validation.json')).read_text()) for sp in done},'elapsed_seconds':time.time()-start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cost':0,'holdout_test_access':False,'candidate_created':False,'uploaded':False}
 with (_repo_path(ROOT, 'reports/early-adapter-095-summary.json')).open('x') as f:json.dump(summary,f,indent=2)
 print(json.dumps({'DONE':True,'eligible':eligible,'partitions':done,'seconds':time.time()-start}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  traceback.print_exc();raise
