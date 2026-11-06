from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c,engine as e
from common import R,P,np,torch
c.check();assert torch.backends.mps.is_available()
with c.Monitor('training',600):
 d=c.training_data();c.arrays(R/'train-input.npz',**d);norm=c.normalization(d);old=torch.load(_repo_io(c.OLD/'models/CNN-20261003.pt'),map_location='cpu',weights_only=True);assert old['seed']==20261003 and old['train_ids']==d['ids'].tolist() and old['normalization']==norm
 # Fixed weights, no optimizer steps: verify auxiliary CE routes and RNG isolation.
 probes=[]
 for device in ['cpu','mps']:
  plain=e.initialize(0,device,True);plain_state=e.base_state(plain);plain_rng=e.rngstate();m=e.initialize(0,device);c.equal(e.base_state(m),plain_state);c.equal(e.rngstate(),plain_rng);m.eval();x=torch.from_numpy(np.ascontiguousarray((d['images'][:32]-norm['x_mean'])/norm['x_std'])).to(device);edges=np.quantile(d['y'],[.25,.5,.75]);labels=torch.tensor(np.searchsorted(edges,d['y'][:32],side='right'),dtype=torch.long,device=device)
  for arm in ['detach','true']:
   pred,logits=m(x,arm);ce=e.F.cross_entropy(logits,labels);manual=(torch.logsumexp(logits,dim=1)-logits[torch.arange(32,device=device),labels]).mean();assert abs(float(ce.detach())-float(manual.detach()))<1e-6;grads=torch.autograd.grad(.3*ce,list(m.parameters()),allow_unused=True);g=dict(zip([n for n,_ in m.named_parameters()],grads));trunk=[v for n,v in g.items() if n.startswith('features.')];assert all(v is None for n,v in g.items() if n.startswith('head.'));assert all(v is None for v in trunk) if arm=='detach' else any(v is not None and float(v.abs().sum())>0 for v in trunk);probes.append({'device':device,'arm':arm,'CE_gradient_route_verified':True})
  del m,plain;torch.mps.empty_cache()
 c.write(R/'input-validation.json',{'passed':True,'rows':500,'IDs_exact':True,'raw500_image_BOX_pixels_and_labels_exact_cache':True,'normalization_exact_original_full500':True,'raw_image_hashes_verified':500,'input_SHA256':c.sha(R/'train-input.npz'),'gradient_probes':probes,'test_read':False})
 detach,dp=e.fit(d,'detach',R/'detach');c.equal({k:v for k,v in detach['state_dict'].items() if not k.startswith('aux.')},old['state_dict']);oldp=c.predict(old,d['images'],'mps');np.testing.assert_array_equal(dp,oldp);c.emit({'detach_original_full500_weight_and_prediction_exact':True})
 e.fit(d,'true',R/'true');e.fit(d,'true',R/'true-repeat')
c.check();c.write(R/'training-complete.json',{'passed':True,'full_fits':3,'updates_per_fit':1280,'test_read':False});c.emit({'training_complete':True})
