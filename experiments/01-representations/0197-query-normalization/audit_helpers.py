from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c,engine as e
from common import np,torch

def equal(a,b):
 if torch.is_tensor(a):assert torch.equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b

def verify_reload(d,mom,k,arm,folder):
 ck=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);T=np.array(ck['train_indices']);V=np.array(ck['validation_indices']);norm=c.normalize(d,T,mom);saved=c.load(folder/'predictions.npz');stats=c.load(folder/'normalization.npz');rec=c.read(folder/'report.json');assert c.sha(folder/'model.pt')==rec['model_SHA256'];assert ck['arm']==arm and ck['fold']==k and ck['seed']==e.cfg['seed']+k;assert ck['train_ids']==d['ids'][T].tolist() and ck['validation_ids']==d['ids'][V].tolist();np.testing.assert_array_equal(saved['indices'],V);np.testing.assert_array_equal(saved['train_indices'],T);np.testing.assert_array_equal(saved['ids'],d['ids'][V]);assert c.statehash(ck['state'])==rec['final_state_SHA256']
 for key,val in norm.items():np.testing.assert_array_equal(stats[key],val)
 for key in ['mean','scale']:np.testing.assert_array_equal(ck['state'][key].numpy(),norm[key].astype(np.float32))
 assert ck['target_mean']==norm['target_mean'] and ck['target_std']==norm['target_std'];initial=e.initialize(norm,k,'cpu');assert c.statehash(initial.state_dict())==rec['initial_state_SHA256'];np.testing.assert_array_equal(initial.coords.numpy(),ck['state']['coords'].numpy());del initial
 xx=mom.copy();heldout=np.setdiff1d(np.arange(400),T);xx[heldout]=1e8-1e5*xx[heldout];poison=dict(d);poison['y']=d['y'].copy();poison['y'][heldout]=1e8-1000*poison['y'][heldout];normpoison=c.normalize(poison,T,xx)
 for key in norm:np.testing.assert_array_equal(norm[key],normpoison[key])
 errors={}
 with c.TokenReader() as reader:
  for device in ['mps','cpu']:
   m=e.initialize(norm,k,device);m.load_state_dict(ck['state'],strict=True);vp=e.predict(m,reader,V,norm,arm,device);tp=e.predict(m,reader,T,norm,arm,device);err=max(float(abs(vp-saved['prediction']).max()),float(abs(tp-saved['train_prediction']).max()));errors[device]=err
   if device=='mps':np.testing.assert_array_equal(vp,saved['prediction']);np.testing.assert_array_equal(tp,saved['train_prediction'])
   else:assert err<=.002
   del m
 torch.mps.empty_cache();st=c.load(folder/'streams.npz');expected=e.streams(k);flat=np.stack([T[expected[ep,j:j+16]] for ep in range(20) for j in range(0,320,16)])[:ck['updates']];np.testing.assert_array_equal(st['order'],flat);seeds=np.array([e.seed(k,ep,j+offset) for ep in range(20) for j in range(0,320,16) for offset in range(0,16,2)],dtype=np.int64)[:ck['updates']*8];np.testing.assert_array_equal(st['dropout_seed'],seeds);lrs=np.repeat([e.lr(ep) for ep in range(20)],20)[:ck['updates']];np.testing.assert_array_equal(st['trace'][:,2],lrs);assert np.isfinite(st['trace']).all() and len(st['trace'])==ck['updates'];assert all(x>0 and np.isfinite(x) for x in rec['first_parameter_gradient_norms'].values())
 return {'fold':k,'arm':arm,'updates':ck['updates'],'MPS_reload_maxabs':errors['mps'],'CPU_reload_maxabs':errors['cpu'],'ID_order_seed_LR_gradients_verified':True,'heldout_moments_y_poison_train_normalization_exact':True}

def compare_replay(original,replay):
 for name in ['model.pt','optimizer.pt']:
  a=torch.load(_repo_io(original/name),map_location='cpu',weights_only=True);b=torch.load(_repo_io(replay/name),map_location='cpu',weights_only=True);equal(a,b)
 for name in ['predictions.npz','normalization.npz']:
  a=c.load(original/name);b=c.load(replay/name);assert a.keys()==b.keys()
  for key in a:np.testing.assert_array_equal(a[key],b[key])
 a=c.load(original/'streams.npz');b=c.load(replay/'streams.npz')
 for key in ['order','dropout_seed','trace']:np.testing.assert_array_equal(a[key],b[key])
 return True
