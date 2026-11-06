from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c,engine as e
from common import R,P,np,torch
c.check();d=c.data();records=[]
with c.Monitor('preflight',300):
 assert c.plan['loss_policy']['delta']==1.0 and c.plan['loss_policy']['multiplier']==2.0
 for dev in ['cpu']+(['mps'] if c.device=='mps' else []):
  r=torch.tensor([-3.,-1.,-.5,0.,.5,1.,3.],device=dev,requires_grad=True);zero=torch.zeros_like(r)
  h=2*e.F.huber_loss(r,zero,delta=1.,reduction='none');expected=torch.where(abs(r)<=1,r*r,2*abs(r)-1);torch.testing.assert_close(h,expected,rtol=0,atol=0)
  grad=torch.autograd.grad(h.sum(),r)[0];torch.testing.assert_close(grad,2*r.detach().clamp(-1,1),rtol=0,atol=0)
  a=torch.tensor([-.5,0.,.5],device=dev,requires_grad=True);m=e.F.mse_loss(a,torch.zeros_like(a));b=2*e.F.huber_loss(a,torch.zeros_like(a),delta=1.);gm=torch.autograd.grad(m,a)[0];gb=torch.autograd.grad(b,a)[0];torch.testing.assert_close(m,b,rtol=0,atol=0);torch.testing.assert_close(gm,gb,rtol=0,atol=0)
  records.append({'device':dev,'piecewise_loss_exact':True,'gradient_exact_including_boundary_and_tails':True,'central_MSE_value_and_gradient_exact':True,'central_curvature':2.0})
 for k in range(5):
  T,V=c.indices(k);norm=c.norm(d,T);poison=dict(d);poison['y']=d['y'].copy();poison['y'][V]=np.nan;assert c.norm(poison,T)==norm
  old=torch.load(_repo_io(P/f'experiments/contrast-augmentation-206/fold-{k}-control/model.pt'),map_location='cpu',weights_only=True);assert old['normalization']==norm and old['train_ids']==d['ids'][T].tolist() and old['validation_ids']==d['ids'][V].tolist()
 ref=c.load(P/'experiments/contrast-augmentation-206/frozen-reference.npz');assert len(ref['best021'])==400
 c.arrays(R/'frozen-reference.npz',**ref)
 c.write(R/'preflight.json',{'passed':True,'device':c.device,'torch':torch.__version__,'numpy':np.__version__,'loss_fixtures':records,'all5_train_normalization_and_splits_match206':True,'all5_validation_label_poison_normalization_exact':True,'model_training_runs':0,'plan_SHA256':c.sha(R/'plan.json'),'code_lock_SHA256':c.sha(R/'code-lock.json')});c.emit({'preflight_passed':True,'device':c.device,'model_training_runs':0})
c.check()
