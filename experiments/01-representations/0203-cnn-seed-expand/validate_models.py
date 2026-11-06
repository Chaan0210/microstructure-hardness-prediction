from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import gc,math,time
import common as c
from common import np,torch,R,O,PLAN,SPLITS,fv
c.check();done=c.J(O/'study-complete.json');assert done['passed'] and done['new_fits']==45
z,images=c.data();new=c.Z(O/'new-seed-OOF.npz');assert c.H(O/'new-seed-OOF.npz')==done['OOF_sha256']
for key in ['ids','y','fold','groups']:np.testing.assert_array_equal(new[key],z[key])
oldmean=z['CNN_seeds'].mean(1);allseeds=np.concatenate([z['CNN_seeds'],new['predictions']],axis=1);mean6=allseeds.mean(1);coef=.243*z['slopes'];primary=z['R']+coef*(mean6-oldmean)
checks=[];maxcpu=0.;maxformula=0.
with c.Monitor('validation',600):
 for si,split in enumerate(SPLITS):
  original=c.Z(R/f'artifacts/component-ablation-121/{split}/predictions.npz')
  for k in range(5):
   T,V=c.indices(z,si,k);norm=c.normalization(images,z['y'],T);poison_x=images.copy();poison_y=z['y'].copy();poison_x[V]+=7.;poison_y[V]+=10000
   assert c.normalization(poison_x,poison_y,T)==norm;del poison_x,poison_y
   cal=c.J(R/f'artifacts/cnn-seed-mean-133/{split}/pipeline-fold-{k}/report.json')['outer_calibrator']
   comp=original['components'][V].copy();comp[:,2]=mean6[si,V];weights=np.array([0.,.1875,.1875,.125,.125]);raw=comp@(weights/weights.sum())
   adjusted=cal['target_center']+cal['slope']*(raw-cal['prediction_center'])
   full=.9*(.9*adjusted+.1*z['short'][si,V]+z['correction'][si,V])+.1*z['L_seeds'][si,:,V].mean(1)
   # NumPy advanced index places V first in L_seeds[si,:,V].
   err=float(abs(full-primary[si,V]).max());assert err<1e-9;maxformula=max(maxformula,err)
   for j,seed in enumerate(PLAN['new_seeds']):
    folder=O/split/f'seed-{seed}-fold-{k}';rr=c.J(folder/'report.json');obj=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);p=c.Z(folder/'predictions.npz');trace=c.Z(folder/'trace.npz')
    assert obj['normalization']==norm and obj['train_ids']==z['ids'][T].tolist() and obj['validation_ids']==z['ids'][V].tolist()
    assert obj['seed']==seed+k and obj['fold']==k and obj['epochs']==80 and obj['average_last_epochs']==5
    assert rr['updates']==rr['training_forward_calls']==800 and rr['validation_labels_masked_before_trainer']
    for f,h in [('model.pt','model_sha256'),('predictions.npz','predictions_sha256'),('trace.npz','trace_sha256'),('optimizer-final.pt','optimizer_sha256')]:assert c.H(folder/f)==rr[h]
    torch.manual_seed(seed+k);initial=fv.engine.TinyCNN();assert sum(x.numel() for x in initial.parameters())==91905 and c.sh(initial.state_dict())==rr['initial_state_sha256'];del initial
    avg=None
    for epoch in range(76,81):
     state=torch.load(_repo_io(folder/f'epoch-{epoch}.pt'),map_location='cpu',weights_only=True);assert c.sh(state)==rr['epoch_state_hashes'][str(epoch)]
     if avg is None:avg={n:v/5 for n,v in state.items()}
     else:
      for n in avg:avg[n]+=state[n]/5
    assert c.sh(avg)==c.sh(obj['state_dict'])==rr['averaged_state_sha256']
    optimizer=torch.load(_repo_io(folder/'optimizer-final.pt'),map_location='cpu',weights_only=True)
    assert all(int(st['step'])==800 for st in optimizer['state'].values())
    assert all(pg['weight_decay']==.01 and pg['lr']==.0001 and pg['betas']==(.9,.999) and pg['eps']==1e-8 for pg in optimizer['param_groups'])
    assert all(torch.isfinite(t).all() for st in optimizer['state'].values() for t in st.values() if torch.is_tensor(t))
    rng=np.random.default_rng(seed+k);step=0;X=(images[T]-norm['x_mean'])/max(norm['x_std'],1e-6)
    for epoch in range(80):
     order=rng.permutation(320)
     for start in range(0,320,32):
      ix=order[start:start+32];rot=int(rng.integers(4));flip=rng.random()<.5
      assert abs(trace['LR'][step]-(.0001+(.002-.0001)*(1+math.cos(math.pi*epoch/80))/2))<1e-15
      if str(step) in rr['input_hashes']:
       x=np.rot90(X[ix],rot,axes=(2,3));x=x[:,:,:,::-1] if flip else x;assert c.ah(np.ascontiguousarray(x))==rr['input_hashes'][str(step)]
      step+=1
    np.testing.assert_array_equal(p['validation_indices'],V);np.testing.assert_array_equal(p['train_indices'],T);np.testing.assert_array_equal(p['validation_predictions'],new['predictions'][si,j,V])
    errors={}
    for device in ['mps','cpu']:
     pv=c.predict(obj,images,V,device);pt=c.predict(obj,images,T,device);err=max(float(abs(pv-p['validation_predictions']).max()),float(abs(pt-p['train_predictions']).max()));errors[device]=err
     if device=='mps':assert err==0
     else:assert err<=.001;maxcpu=max(maxcpu,err)
    checks.append({'split':split,'fold':k,'seed':seed,'reload':errors,'trainstats_IDs_seed_updates_LR_inputhash_last5_initialstate_verified':True});del obj,optimizer,avg,state,X
   c.emit({'validation_fold_complete':split,'fold':k,'new_models_verified':len(checks)})
   gc.collect();torch.mps.empty_cache()
assert len(checks)==45 and np.isfinite(primary).all();c.check()
c.arrays(O/'ensemble-OOF.npz',ids=z['ids'],y=z['y'],fold=z['fold'],groups=z['groups'],reference=z['R'],primary=primary,old_seeds=z['CNN_seeds'],new_seeds=new['predictions'],all_seeds=allseeds,oldmean=oldmean,mean6=mean6,coefficient=coef)
c.W(O/'validation.json',{'passed':True,'new_models':45,'old_models_reloaded':45,'new_training_replays':0,'new_MPS_reload_exact':True,'max_CPU_MPS_error':maxcpu,'max_independent_fullensemble_error':maxformula,'all15heldout_x_y_poison_trainstats_exact':True,'all45trainer_calls_masked_validation_labels':True,'unchanged_originaltrainer_and_observation_hooks_only':True,'checks':checks,'ensemble_sha256':c.H(O/'ensemble-OOF.npz'),'study_complete_sha256':c.H(O/'study-complete.json'),'no_model_performance_selection':True})
c.access_report('validation-access-audit.json');c.emit({'validation_passed':True,'new_models':45,'max_CPU_MPS_error':maxcpu,'fullensemble_error':maxformula})
