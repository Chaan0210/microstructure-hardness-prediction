import copy
import common as c,engine as e
from common import R,P,np,torch
c.check();assert torch.backends.mps.is_available();d=c.data();T,V=c.indices(d,0);norm,edges,bins,perm,shuffled=c.preprocessing(d,T,0)
def equal(a,b):
 if torch.is_tensor(a):assert torch.equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b
X=(d['images'][T]-norm['x_mean'])/max(norm['x_std'],1e-6);y=(d['y'][T]-norm['y_mean'])/norm['y_std'];rng=np.random.default_rng(20261003);order=rng.permutation(320);batches=[]
for j in range(8):
 ix=order[j*32:(j+1)*32];rot=int(rng.integers(4));flip=bool(rng.random()<.5);x=np.rot90(X[ix],rot,axes=(2,3));x=x[:,:,:,::-1] if flip else x;batches.append((np.ascontiguousarray(x),y[ix],bins[ix]))
records=[]
with c.Monitor('preflight',180):
 for device in ['cpu','mps']:
  runs=[]
  for plain in [True,False]:
   m=e.initialize(0,device,plain);start=e.rnghash();initial=e.base_state(m);main,aux=e.optimizers(m,plain);trace=[]
   for x,yb,bb in batches:
    q=e.update(m,main,aux,torch.from_numpy(x).to(device),torch.tensor(yb,dtype=torch.float32,device=device),torch.tensor(bb,dtype=torch.long,device=device),'detach',collect=True)
    trace.append({'prediction':q['prediction'],'gradients':q['gradients'],'loss':q['values'][0],'base_gradient_norm':q['values'][2],'state':e.base_state(m),'optimizer':copy.deepcopy(main.state_dict()),'rng':e.rngstate()})
   runs.append({'initial':initial,'rng_start':start,'trace':trace});del m,main,aux
  equal(runs[0],runs[1]);records.append({'device':device,'steps':8,'prediction_loss_gradients_state_optimizer_RNG_exact':True})
  c.emit({'update_equivalence_passed':device,'steps':8})
 for arm in c.plan['arms']:
  m=e.initialize(0,'mps');m.eval();x=torch.from_numpy(batches[0][0]).to('mps');p,logit=m(x,arm);bb=torch.tensor(batches[0][2],dtype=torch.long,device='mps');ce=e.F.cross_entropy(logit,bb);manual=(torch.logsumexp(logit,dim=1)-logit[torch.arange(32,device='mps'),bb]).mean();assert abs(float(ce)-float(manual))<1e-6
  gs=torch.autograd.grad(.3*ce,list(m.parameters()),allow_unused=True);byname=dict(zip([n for n,_ in m.named_parameters()],gs));trunk=[g for n,g in byname.items() if n.startswith('features.')];head=[g for n,g in byname.items() if n.startswith('head.')];aux=[g for n,g in byname.items() if n.startswith('aux.')]
  assert all(g is None for g in head);assert all(g is None for g in trunk) if arm=='detach' else any(g is not None and float(g.abs().sum())>0 for g in trunk);assert all(g is not None for g in aux)
  records.append({'arm':arm,'CE_reaches_scalar_head':False,'CE_reaches_trunk':arm!='detach','CE_reaches_aux_head':True,'independent_CE_difference':abs(float(ce)-float(manual))})
 c.write(R/'preflight.json',{'passed':True,'updates':records,'plan_SHA256':c.sha(R/'plan.json'),'torch':torch.__version__,'numpy':np.__version__,'cost':0})
c.check();c.emit({'preflight_passed':True})
