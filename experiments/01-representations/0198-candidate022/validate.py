from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import math
import common as c
from common import R,P,np,torch
c.check();assert c.read(R/'training-complete.json')['passed'];v=c.read(R/'input-validation.json');assert c.sha(R/'train-input.npz')==v['input_SHA256'];d=c.load(R/'train-input.npz');norm=c.normalization(d);records=[]
with c.Monitor('validation',180):
 old=torch.load(_repo_io(c.OLD/'models/CNN-20261003.pt'),map_location='cpu',weights_only=True);assert old['train_ids']==d['ids'].tolist() and old['normalization']==norm;reports={};objs={}
 for name in ['detach','true','true-repeat']:
  folder=R/name;r=c.read(folder/'report.json');assert r['updates']==1280 and c.sha(folder/'model.pt')==r['model_SHA256'];obj=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);objs[name]=obj;reports[name]=r;assert obj['train_ids']==d['ids'].tolist() and obj['normalization']==norm and obj['seed']==20261003 and obj['aux_seed']==202710193
  edges=np.quantile(d['y'],[.25,.5,.75],method='linear');np.testing.assert_array_equal(obj['bin_edges'],edges);np.testing.assert_array_equal(obj['train_bins'],np.searchsorted(edges,d['y'],side='right'))
  avg=None
  for ep in range(76,81):
   s=torch.load(_repo_io(folder/f'epoch-{ep}.pt'),map_location='cpu',weights_only=True);assert c.statehash(s)==r['epoch_state_hashes'][str(ep)]
   if avg is None:avg={k:v/5 for k,v in s.items()}
   else:
    for k in avg:avg[k]+=s[k]/5
  c.equal(avg,obj['state_dict']);st=c.load(folder/'streams.npz');assert st['order'].shape==(1280,32);rng=np.random.default_rng(20261003);ixstep=0
  for ep in range(80):
   order=rng.permutation(500);lr=.0001+.5*(.002-.0001)*(1+math.cos(math.pi*ep/80))
   for off in range(0,500,32):
    ix=order[off:off+32];row=st['order'][ixstep];np.testing.assert_array_equal(row[:len(ix)],ix);assert np.all(row[len(ix):]==-1) and st['length'][ixstep]==len(ix);assert st['rot'][ixstep]==int(rng.integers(4)) and st['flip'][ixstep]==bool(rng.random()<.5);assert abs(st['lr'][ixstep]-lr)<1e-16;ixstep+=1
  assert np.isfinite(st['values']).all();final=torch.load(_repo_io(folder/'final-update-state.pt'),map_location='cpu',weights_only=True)
  for key in ['main_optimizer','aux_optimizer']:
   assert all(int(q['step'])==1280 for q in final[key]['state'].values())
  for q in final['scheduler']:assert q['last_epoch']==80
  saved=c.load(folder/'predictions.npz');np.testing.assert_array_equal(saved['ids'],d['ids']);pm=c.predict(obj,d['images'],'mps');pm2=c.predict(torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True),d['images'],'mps');pc=c.predict(obj,d['images'],'cpu');np.testing.assert_array_equal(pm,saved['prediction']);np.testing.assert_array_equal(pm,pm2);err=float(abs(pm-pc).max());assert err<.001;records.append({'fit':name,'MPS_reload_max_abs':0.,'CPU_MPS_max_abs':err,'average_last5_order_LR_optimizer_updates_verified':True});c.emit(records[-1])
 # Full500 exact null replay verifies batch20 tail, preprocessing and original CNN lineage.
 c.equal({k:v for k,v in objs['detach']['state_dict'].items() if not k.startswith('aux.')},old['state_dict']);np.testing.assert_array_equal(c.predict(old,d['images'],'mps'),c.load(R/'detach/predictions.npz')['prediction'])
 for name in ['model.pt','final-update-state.pt']:
  c.equal(torch.load(_repo_io(R/'true'/name),map_location='cpu',weights_only=True),torch.load(_repo_io(R/'true-repeat'/name),map_location='cpu',weights_only=True))
 c.equal(c.load(R/'true/streams.npz'),c.load(R/'true-repeat/streams.npz'));c.equal(c.load(R/'true/predictions.npz'),c.load(R/'true-repeat/predictions.npz'))
 for ep in range(76,81):c.equal(torch.load(_repo_io(R/'true'/f'epoch-{ep}.pt'),map_location='cpu',weights_only=True),torch.load(_repo_io(R/'true-repeat'/f'epoch-{ep}.pt'),map_location='cpu',weights_only=True))
 for key in ['initial_state_SHA256','initial_base_SHA256','rng_start','input_hashes']:
  assert reports['detach'][key]==reports['true'][key]==reports['true-repeat'][key]
 for key in ['order','length','rot','flip','lr']:
  np.testing.assert_array_equal(c.load(R/'detach/streams.npz')[key],c.load(R/'true/streams.npz')[key])
 cert={'passed':True,'rows':500,'full_fits':3,'updates_per_fit':1280,'detach_original500_weights_predictions_exact':True,'true_repeat_weights_optimizer_RNG_traces_predictions_exact':True,'train_only_all500_normalization_and_quartiles_verified':True,'all500_used_no_holdout_subgroup_scores':True,'test_read':False,'no_selection_changed':True,'records':records,'max_CPU_MPS_difference':max(q['CPU_MPS_max_abs'] for q in records),'input_SHA256':v['input_SHA256'],'primary_model_SHA256':c.sha(R/'true/model.pt'),'original_CNN_SHA256':c.sha(c.OLD/'models/CNN-20261003.pt'),'target_range_for_fixed_format_guard':[float(d['y'].min()),float(d['y'].max())],'inference_precision':'batch32; CPU numpy float32 de-normalization before float64 concatenation, same as legacy','code_lock_SHA256':c.sha(R/'code-lock.json')}
c.check();c.write(R/'training-validation.json',cert);c.emit(cert)
