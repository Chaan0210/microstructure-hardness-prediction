from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c,engine as e
from common import R,P,np,torch
c.check();assert c.read(R/'preflight.json')['passed'];records=[];hashes={}
with c.Monitor('training',2400):
 for split in c.SPLITS:
  d=c.data(split);pred={a:np.full(400,np.nan) for a in ['control','huber']}
  for k in range(5):
   T,V=c.indices(d,k);pair=[]
   for arm in ['control','huber']:
    folder=R/split/f'fold-{k}-{arm}';p,r=e.fit(d,k,arm,folder);pred[arm][V]=p;pair.append(r);records.append(r)
    if arm=='control':
     oldfolder=P/f'experiments/ema-repartition-209/{split}/fold-{k}';original=torch.load(_repo_io(oldfolder/'last5_uniform.pt'),map_location='cpu',weights_only=True);new=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);e.equal_tree(original['state_dict'],new['state_dict']);np.testing.assert_array_equal(p,d['old_CNN'][V])
     init=torch.load(_repo_io(oldfolder/'ema-update-0.pt'),map_location='cpu',weights_only=True);assert r['initial_state_SHA256']==c.statehash(init)
     prior=c.read(P/f'experiments/auxiliary-repartition-201/{split}/fold-{k}-detach/report.json')
     for key in ['rng_start','rng_end']:assert r[key]=={n.upper():v for n,v in prior[key].items()}
     for epoch in range(76,81):
      oldstate=torch.load(_repo_io(oldfolder/f'epoch-{epoch}.pt'),map_location='cpu',weights_only=True);assert r['epoch_state_SHA256'][str(epoch)]==c.statehash(oldstate)
     a=c.load(folder/'streams.npz');b=c.load(oldfolder/'streams.npz')
     for key in ['order','rot','flip','lr','rng_CPU','rng_MPS']:np.testing.assert_array_equal(a[key],b[key])
     np.testing.assert_array_equal(a['values'][:,:2],b['values']);oldopt=torch.load(_repo_io(oldfolder/'optimizer-verification.pt'),map_location='cpu',weights_only=True);newopt=torch.load(_repo_io(folder/'final-live-and-optimizer.pt'),map_location='cpu',weights_only=True);e.equal_tree(newopt['optimizer'],oldopt['optimizer']);e.equal_tree(newopt['scheduler'],oldopt['scheduler'])
     c.write(folder/'control-reproduction.json',{'passed':True,'source209_state_epoch76_to80_OOF_exact':True,'source209_all800_order_D4_LR_loss_gradientnorm_RNG_exact':True,'final_optimizer_scheduler_exact209':True})
   for key in ['initial_state_SHA256','rng_start','rng_end']:assert pair[0][key]==pair[1][key]
   a=c.load(R/split/f'fold-{k}-control/streams.npz');b=c.load(R/split/f'fold-{k}-huber/streams.npz')
   for key in ['order','rot','flip','lr','rng_CPU','rng_MPS','input_SHA256']:np.testing.assert_array_equal(a[key],b[key])
   c.write(R/split/f'paired-streams-fold-{k}.json',{'passed':True,'all800_order_D4_LR_CPU_MPS_RNG_input_exact':True,'same_initial_state':True});c.emit({'split':split,'paired_fold_complete':k})
  assert all(np.isfinite(p).all() for p in pred.values());c.arrays(R/split/'OOF.npz',ids=d['ids'],y=d['y'],fold=d['fold'],groups=d['g'],**pred);hashes[split]=c.sha(R/split/'OOF.npz')
 c.write(R/'training-complete.json',{'fits':20,'gradient_updates':16000,'records':records,'OOF_SHA256':hashes,'original211_retrained':False,'performance_scores_computed':False,'total_fit_seconds':sum(r['seconds'] for r in records)})
c.check()
