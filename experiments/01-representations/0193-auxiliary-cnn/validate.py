from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import copy,gc,math
import common as c,engine as e
from common import R,P,np,torch,time
c.check();d=c.data();o=c.load(R/'OOF.npz');assert c.sha(R/'OOF.npz')==c.read(R/'report.json')['OOF_SHA256'];checks=[];maxcpu=0.;maxmps=0.;full=[]
with c.Monitor('validation',1200):
 for k in range(5):
  T,V=c.indices(d,k);norm,edges,bins,perm,shuffled=c.preprocessing(d,T,k);poison={**d,'y':d['y'].copy(),'images':d['images'].copy()};poison['y'][V]+=10000;poison['images'][V]+=7;pn,pe,pb,pp,ps=c.preprocessing(poison,T,k);assert pn==norm
  for a,b in [(edges,pe),(bins,pb),(perm,pp),(shuffled,ps)]:np.testing.assert_array_equal(a,b)
  sorted_y=np.sort(d['y'][T]);manual=[]
  for q in [.25,.5,.75]:
   pos=319*q;lo=int(pos);manual.append(sorted_y[lo]+(pos-lo)*(sorted_y[lo+1]-sorted_y[lo]))
  np.testing.assert_allclose(manual,edges,rtol=0,atol=1e-12);np.testing.assert_array_equal(bins,(d['y'][T,None]>=edges).sum(1));assert abs(norm['y_mean']-math.fsum(d['y'][T])/320)<1e-12
  paired=[]
  for arm in c.plan['arms']:
   folder=R/'original'/f'fold-{k}-{arm}';r=c.read(folder/'report.json');assert c.sha(folder/'model.pt')==r['model_SHA256'] and c.sha(folder/'predictions.npz')==r['predictions_SHA256'] and c.sha(folder/'streams.npz')==r['streams_SHA256'];p=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);z=c.load(folder/'predictions.npz');s=c.load(folder/'streams.npz');assert p['train_indices']==T.tolist() and p['validation_indices']==V.tolist() and p['train_ids']==d['ids'][T].tolist() and p['validation_ids']==d['ids'][V].tolist();assert p['normalization']==norm;np.testing.assert_array_equal(p['bin_edges'],edges);np.testing.assert_array_equal(p['train_bins'],bins);np.testing.assert_array_equal(p['permutation'],perm);np.testing.assert_array_equal(p['used_bins'],shuffled if arm=='permuted' else bins);np.testing.assert_array_equal(z['validation_predictions'],o[arm][V]);assert r['updates']==800
   avg=None
   for epoch in range(76,81):
    state=torch.load(_repo_io(folder/f'epoch-{epoch}.pt'),map_location='cpu',weights_only=True);assert c.statehash(state)==r['epoch_state_hashes'][str(epoch)]
    if avg is None:avg={n:v/5 for n,v in state.items()}
    else:
     for n,v in state.items():avg[n]+=v/5
   assert c.statehash(avg)==c.statehash(p['state_dict'])==r['averaged_state_SHA256']
   rng=np.random.default_rng(20261003+k);step=0;X=(d['images'][T]-norm['x_mean'])/max(norm['x_std'],1e-6)
   for epoch in range(80):
    order=rng.permutation(320)
    for first in range(0,320,32):
     ix=order[first:first+32];rot=int(rng.integers(4));flip=bool(rng.random()<.5);np.testing.assert_array_equal(s['order'][step],ix);assert s['rot'][step]==rot and bool(s['flip'][step])==flip;expected=.0001+(.002-.0001)*(1+math.cos(math.pi*epoch/80))/2;assert abs(s['lr'][step]-expected)<1e-15
     if step in [0,99,399,799]:
      xb=np.rot90(X[ix],rot,axes=(2,3));xb=xb[:,:,:,::-1] if flip else xb;assert c.ah(np.ascontiguousarray(xb))==r['input_hashes'][str(step)]
     step+=1
   for device in ['cpu','mps']:
    model=e.initialize(k,device);model.load_state_dict(p['state_dict']);pv=e.predict(model,d['images'],V,norm,device);pt=e.predict(model,d['images'],T,norm,device);err=max(float(abs(pv-z['validation_predictions']).max()),float(abs(pt-z['train_predictions']).max()));assert err<(1e-3 if device=='cpu' else 1e-10),(k,arm,device,err)
    if device=='cpu':maxcpu=max(maxcpu,err)
    else:maxmps=max(maxmps,err)
    del model
   paired.append(s);checks.append({'fold':k,'arm':arm,'reload_train_validation_passed':True,'normalization_bins_permutation_trainonly':True,'eight_hundred_update_streams_LR_exact':True,'five_epoch_state_average_exact':True});gc.collect();torch.mps.empty_cache()
  for field in ['order','rot','flip','lr']:
   np.testing.assert_array_equal(paired[0][field],paired[1][field]);np.testing.assert_array_equal(paired[0][field],paired[2][field])
  c.emit({'model_audits_complete_fold':k})
 T,V=c.indices(d,0);poison={**d,'y':d['y'].copy()};poison['y'][V]+=10000
 for arm in c.plan['arms']:
  folder=R/f'poison-fold0-{arm}';vp,rec=e.fit(poison,0,arm,folder);src=R/'original'/f'fold-0-{arm}';rr=c.read(src/'report.json');assert rec['averaged_state_SHA256']==rr['averaged_state_SHA256'] and rec['epoch_state_hashes']==rr['epoch_state_hashes'];a=c.load(src/'streams.npz');b=c.load(folder/'streams.npz')
  for key in a:np.testing.assert_array_equal(a[key],b[key])
  q=c.load(src/'predictions.npz');z=c.load(folder/'predictions.npz')
  for key in q:np.testing.assert_array_equal(q[key],z[key])
  full.append({'arm':arm,'epochs':80,'all_weights_predictions_losses_gradient_norms_streams_exact':True})
 r={'passed':True,'models_checked':15,'checks':checks,'max_CPU_prediction_error':maxcpu,'max_MPS_prediction_error':maxmps,'poison_all5_normalization_bins_permutation_exact':True,'full80epoch_validation_label_poison':full,'historical_CNN_equivalence':[c.read(R/'original'/f'fold-{k}-detach/legacy-equivalence.json') for k in range(5)],'preflight':c.read(R/'preflight.json'),'OOF_SHA256':c.sha(R/'OOF.npz')};c.write(R/'validation.json',r);c.emit({'validation_passed':True,'models':15,'max_CPU_error':maxcpu,'max_MPS_error':maxmps,'full_poison_replays':3})
c.check()
