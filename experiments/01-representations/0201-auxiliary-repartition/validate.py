from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import math,gc
import common as c,engine as e
from common import R,P,np,torch
c.check();assert c.read(R/'study-complete.json')['passed'];checks=[];maxcpu=0.;maxconvert=0.;maxformula=0.
with c.Monitor('validation',600):
 for split in c.SPLITS:
  d=c.data(split);root=R/split;o=c.load(root/'OOF.npz');f=c.lineage(d)
  for key,value in [('ids',d['ids']),('y',d['y']),('fold',d['fold']),('groups',d['g']),('best021',d['best021'])]:np.testing.assert_array_equal(o[key],value)
  np.testing.assert_array_equal(o['coefficient'],f['coefficient']);np.testing.assert_array_equal(o['detach'],d['best021'])
  for k in range(5):
   T,V=c.indices(d,k);norm,edges,bins,perm,shuffled=c.preprocessing(d,T,k);poison={**d,'y':d['y'].copy(),'images':d['images'].copy()};poison['y'][V]+=10000;poison['images'][V]+=7;pn,pe,pb,pp,ps=c.preprocessing(poison,T,k);assert pn==norm
   for x,z in [(edges,pe),(bins,pb),(perm,pp),(shuffled,ps)]:np.testing.assert_array_equal(x,z)
   del poison;sort=np.sort(d['y'][T]);manual=[]
   for q in [.25,.5,.75]:pos=319*q;lo=int(pos);manual.append(sort[lo]+(pos-lo)*(sort[lo+1]-sort[lo]))
   np.testing.assert_allclose(manual,edges,rtol=0,atol=1e-12);np.testing.assert_array_equal(bins,(d['y'][T,None]>=edges).sum(1));assert abs(norm['y_mean']-math.fsum(d['y'][T])/320)<1e-12;paired=[]
   for arm in c.ARMS:
    folder=root/f'fold-{k}-{arm}';rr=c.read(folder/'report.json');obj=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);z=c.load(folder/'legacy-predictions.npz');st=c.load(folder/'streams.npz');assert rr['updates']==800 and c.sha(folder/'model.pt')==rr['model_SHA256'];assert obj['train_indices']==T.tolist() and obj['validation_indices']==V.tolist() and obj['train_ids']==d['ids'][T].tolist() and obj['validation_ids']==d['ids'][V].tolist() and obj['normalization']==norm
    for key,val in [('bin_edges',edges),('train_bins',bins),('permutation',perm),('used_bins',shuffled if arm=='permuted' else bins)]:np.testing.assert_array_equal(obj[key],val)
    avg=None
    for epoch in range(76,81):
     state=torch.load(_repo_io(folder/f'epoch-{epoch}.pt'),map_location='cpu',weights_only=True);assert c.statehash(state)==rr['epoch_state_hashes'][str(epoch)]
     if avg is None:avg={n:v/5 for n,v in state.items()}
     else:
      for n,v in state.items():avg[n]+=v/5
    assert c.statehash(avg)==c.statehash(obj['state_dict'])==rr['averaged_state_SHA256'];opt=torch.load(_repo_io(folder/'optimizer-final.pt'),map_location='cpu',weights_only=True)
    for which in ['main','aux']:assert all(int(v['step'])==800 for v in opt[which]['state'].values())
    assert all(v['last_epoch']==80 for v in opt['schedulers']);rng=np.random.default_rng(20261003+k);step=0;X=(d['images'][T]-norm['x_mean'])/max(norm['x_std'],1e-6)
    for epoch in range(80):
     order=rng.permutation(320)
     for first in range(0,320,32):
      ix=order[first:first+32];rot=int(rng.integers(4));flip=bool(rng.random()<.5);np.testing.assert_array_equal(st['order'][step],ix);assert st['rot'][step]==rot and bool(st['flip'][step])==flip;expected=.0001+(.002-.0001)*(1+math.cos(math.pi*epoch/80))/2;assert abs(st['lr'][step]-expected)<1e-15
      if step in [0,99,399,799]:
       xb=np.rot90(X[ix],rot,axes=(2,3));xb=xb[:,:,:,::-1] if flip else xb;assert c.ah(np.ascontiguousarray(xb))==rr['input_hashes'][str(step)]
      step+=1
    for device in ['mps','cpu']:
     pv=c.legacy_predict(obj,d['images'],V,device);pt=c.legacy_predict(obj,d['images'],T,device);err=max(float(abs(pv-z['validation_prediction']).max()),float(abs(pt-z['train_prediction']).max()))
     if device=='mps':assert err==0
     else:assert err<.001;maxcpu=max(maxcpu,err)
    np.testing.assert_array_equal(o['raw_'+arm][V],z['validation_prediction']);diff=c.read(folder/'precision.json')['batch64_converted_vs_batch32_max_error'];maxconvert=max(maxconvert,diff);paired.append(st)
    # Independently rebuild the entire frozen021 lineage for each arm/fold.
    comp=f['components'][V].copy();comp[:,2]=(o['raw_'+arm][V]+d['old_CNN_seeds'][1,V]+d['old_CNN_seeds'][2,V])/3;weights=np.array([0.,.1875,.1875,.125,.125]);raw=comp@(weights/weights.sum());cal=c.plan['calibrators'][split][str(k)];base=.9*(cal['target_center']+cal['slope']*(raw-cal['prediction_center']))+.1*f['short'][V];manual=.9*(base+f['correction'][V])+.1*f['L_MLP'][V];err=float(abs(manual-o[arm][V]).max());assert err<1e-10;maxformula=max(maxformula,err);checks.append({'split':split,'fold':k,'arm':arm,'updates800_last5_average_IDs_norm_bins_optimizer_streams_reload':True})
   for field in ['order','rot','flip','lr']:
    np.testing.assert_array_equal(paired[0][field],paired[1][field]);np.testing.assert_array_equal(paired[0][field],paired[2][field])
   c.emit({'validated':split,'fold':k,'models_checked':len(checks)});gc.collect();torch.mps.empty_cache()
assert len(checks)==30;c.check();rec={'passed':True,'models':30,'checks':checks,'max_CPU_MPS_prediction_error':maxcpu,'MPS_reload_exact':True,'max_batch64_converted_vs_batch32_error':maxconvert,'max_independent_fullensemble_error':maxformula,'all10_detach_exact_priorweights_and_OOF':True,'all10_train_only_preprocessing_heldout_x_y_poison_exact':True,'core_Model_initialize_optimizers_update_AST_exact193':True,'full_poison_retraining_in201':0,'reproduction_scope':'10nullfull80replays matchhistoricalweights/OOF exactly;30CPU/MPSreloads,trainonlypoisonpreprocess,800update/LR/D4/seeds/stateaverageaudits. Original193 full3arm80epochlabelpoison replay evidence retained. No newtrue/permutedfullreplay beyond authorized30fits.','new_fit_count':30,'extra_validation_fits':0,'holdout_test_full500_read':False};c.write(R/'validation.json',rec);c.emit(rec)
