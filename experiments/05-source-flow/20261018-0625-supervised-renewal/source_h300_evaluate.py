"""Source-excluded new-representation check; explicitly a paired component test."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
for k in ['TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME']:os.environ[k]=str(_repo_path(HERE, 'tmp'))
os.environ['HF_HOME']=str(_repo_path(HERE, 'tmp/huggingface'));os.environ['HF_HUB_OFFLINE']='1';os.environ['TABPFN_DISABLE_TELEMETRY']='1'
import argparse,json,gc
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

def main(a):
 torch.set_num_threads(2);torch.set_num_interop_threads(1);out=_repo_path(HERE, 'source-h300', a.tag);r=json.loads((out/'recipe.json').read_text());tr=np.array(r['train']);q=np.array(r['query']);assert not set(r['source_indices'])&set(q);d=np.load(_repo_path(HERE, 'data.npz'));z=np.load(_repo_path(ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz'));pos={str(v):i for i,v in enumerate(z['ids'])};P=z['X'][[pos[str(v)] for v in d['ids']]].astype(float);s=np.load(_repo_io(out/'flow-features.npz'));g=np.load(_repo_path(HERE, 'domain-semantics/H300-features.npz'));assert np.array_equal(s['ids'],d['ids']) and np.array_equal(g['ids'],d['ids']);A={'control':P,'global_flow':np.c_[P,g['flow']],'source_flow':np.c_[P,s['flow']]};pred={};sys.path.insert(0,str(_repo_path(ROOT, 'experiments/push-300')));import harness
 for name,X in A.items():
  ps=[]
  for seed in [2026101811,2026101823]:
   cache=_repo_path(HERE, f'flow-component-confirmation/{"control" if name=="control" else "flow"}-r0-f0-s{seed}.npz')
   if name!='source_flow':
    c=np.load(_repo_io(cache));assert r['split_seed']==2026101801 and r['fold']==0 and np.array_equal(c['train'],tr) and np.array_equal(c['query'],q);v=c['prediction']
   else:
    m=harness.tabpfn_new('v3.5',seed+r['fold'],'cpu',n_est=8);m.fit(X[tr],d['y'][tr]);v=m.predict(X[q]);del m;gc.collect();np.savez(_repo_io(out/f'head-seed{seed}.npz'),train=tr,query=q,prediction=v)
   ps.append(v)
  tab=np.mean(ps,0);lin=[]
  for kind in ['raw','log']:
   m=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=100));m.fit(X[tr],d['y'][tr] if kind=='raw' else np.log(d['y'][tr]));v=m.predict(X[q]);lin.append(v if kind=='raw' else np.exp(v))
  pred[name+'_tab']=tab;pred[name+'_block']=.6*tab+.2*lin[0]+.2*lin[1]
 y=d['y'][q];rm=lambda v:float(np.mean((v-y)**2)**.5);scores={k:rm(v) for k,v in pred.items()};old=(pred['control_block']-y)**2;new=(pred['source_flow_block']-y)**2;draw=np.random.default_rng(2026101997).integers(len(q),size=(10000,len(q)));boot=np.sqrt(old[draw].mean(1))-np.sqrt(new[draw].mean(1));keep=np.ones(len(q),bool);keep[np.argsort(old-new)[-10:]]=False
 report=dict(scores=scores,new_source_flow_block_gain=rm(pred['control_block'])-rm(pred['source_flow_block']),global_flow_block_gain=rm(pred['control_block'])-rm(pred['global_flow_block']),ci95=np.percentile(boot,[2.5,97.5]).tolist(),top10_positive_removed_gain=float(old[keep].mean()**.5-new[keep].mean()**.5),train_rows=len(tr),query_rows=len(q),auxiliary_source_rows=len(r['source_indices']),auxiliary_query_overlap=0,complete_incumbent_compared=False,full_system_protocol_passed=False,scope='NewH300-styleauxiliaryencodertrainedonlyonouterT. P104stillcontainshistoricalglobalrepresentations. Thisisapairedcomponentcheck,notacomplete083end-to-endcomparison;reusedlabels,notuntoucheddata.')
 np.savez(_repo_io(out/'component-predictions.npz'),ids=d['ids'][q],y=y,train=tr,query=q,**pred);(out/'component-results.json').write_text(json.dumps(report,indent=2));print('SOURCE_CHECK',json.dumps(report),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tag',default='repeat0-fold0-seed1811')
 with threadpool_limits(2):main(p.parse_args())
