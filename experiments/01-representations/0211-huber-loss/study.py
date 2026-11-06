from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c,engine as e
from common import R,P,np,torch
c.check();assert c.read(R/'preflight.json')['passed'];d=c.data();pred={a:np.full(400,np.nan) for a in ['control','huber']};refs=c.load(R/'frozen-reference.npz');records=[]
with c.Monitor('training',1200):
 for k in range(5):
  T,V=c.indices(k);pair=[]
  for arm in ['control','huber']:
   folder=R/f'fold-{k}-{arm}';p,r=e.fit(d,k,arm,folder);pred[arm][V]=p;pair.append(r);records.append(r)
   if arm=='control':
    original=torch.load(_repo_io(P/f'artifacts/run-009/fold-{k}.pt'),map_location='cpu',weights_only=True);new=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);e.equal_tree(original['state_dict'],new['state_dict']);np.testing.assert_array_equal(p,refs['old_first_CNN'][V]);old=c.read(P/f'experiments/contrast-augmentation-206/fold-{k}-control/fit.json')
    for key in ['initial_state_SHA256','rng_start','rng_end','epoch_state_SHA256','averaged_state_SHA256']:assert r[key]==old[key],key
    a=c.load(folder/'streams.npz');b=c.load(P/f'experiments/contrast-augmentation-206/fold-{k}-control/streams.npz')
    for key in ['order','rot','flip','lr','rng_CPU','rng_MPS']:np.testing.assert_array_equal(a[key],b[key])
    np.testing.assert_array_equal(a['values'][:,:2],b['values'])
    oldopt=torch.load(_repo_io(P/f'experiments/update-ema-208/fold-{k}/optimizer-verification.pt'),map_location='cpu',weights_only=True);newopt=torch.load(_repo_io(folder/'final-live-and-optimizer.pt'),map_location='cpu',weights_only=True);e.equal_tree(newopt['optimizer'],oldopt['optimizer']);e.equal_tree(newopt['scheduler'],oldopt['scheduler'])
    c.write(folder/'control-reproduction.json',{'passed':True,'state_and_epoch76_to80_exact':True,'OOF_FP32_exact':True,'all800_order_D4_LR_loss_gradientnorm_RNG_exact':True,'final_optimizer_exact_to208':True})
  for key in ['initial_state_SHA256','rng_start','rng_end']:assert pair[0][key]==pair[1][key],key
  a=c.load(R/f'fold-{k}-control/streams.npz');b=c.load(R/f'fold-{k}-huber/streams.npz')
  for key in ['order','rot','flip','lr','rng_CPU','rng_MPS','input_SHA256']:np.testing.assert_array_equal(a[key],b[key])
  c.write(R/f'paired-streams-fold-{k}.json',{'passed':True,'all800_order_D4_LR_CPU_MPS_RNG_input_exact':True,'same_initial_state':True});c.emit({'paired_fold_complete':k})
 assert all(np.isfinite(p).all() for p in pred.values());c.arrays(R/'OOF.npz',ids=d['ids'],y=d['y'],fold=d['fold'],groups=d['g'],**pred);c.write(R/'training-complete.json',{'fits':10,'records':records,'OOF_SHA256':c.sha(R/'OOF.npz'),'performance_scores_computed':False,'total_fit_seconds':sum(r['seconds'] for r in records)})
c.check()
