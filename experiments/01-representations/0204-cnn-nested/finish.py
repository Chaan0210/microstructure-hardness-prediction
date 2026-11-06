from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,gc,time,math,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as c
import pipeline as p
import train as trainer
from common import np,torch,R,O,PLAN,SPLITS,fv

def validate_one(z,images,node,seed):
 folder=trainer.path_for(node,seed);rr=c.J(folder/'report.json');obj=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);saved=c.Z(folder/'predictions.npz');trace=c.Z(folder/'trace.npz');T=np.array(node['train']);V=np.array(node['validation']);norm=c.normalization(images,z['y'],T);nb=math.ceil(len(T)/32);steps=80*nb
 assert obj['normalization']==norm==rr['normalization'];assert obj['train_ids']==z['ids'][T].tolist() and obj['validation_ids']==z['ids'][V].tolist();assert obj['seed']==seed+node['offset'] and obj['fold']==node['offset'] and obj['epochs']==80 and obj['average_last_epochs']==5;assert rr['updates']==rr['training_forward_calls']==steps and rr['label_mode']=='masked'
 for f,key in [('model.pt','model_sha256'),('predictions.npz','predictions_sha256'),('trace.npz','trace_sha256'),('optimizer-final.pt','optimizer_sha256')]:assert c.H(folder/f)==rr[key]
 outside=np.setdiff1d(np.arange(400),T);yp=z['y'].copy();yp[outside]=np.nan;xp=images.copy();xp[outside]=np.nan;assert c.normalization(xp,yp,T)==norm;del xp,yp
 torch.manual_seed(seed+node['offset']);initial=fv.engine.TinyCNN();assert c.sh(initial.state_dict())==rr['initial_state_sha256'];del initial
 avg=None
 for epoch in range(76,81):
  state=torch.load(_repo_io(folder/f'epoch-{epoch}.pt'),map_location='cpu',weights_only=True);assert c.sh(state)==rr['epoch_state_hashes'][str(epoch)]
  if avg is None:avg={n:v/5 for n,v in state.items()}
  else:
   for n in avg:avg[n]+=state[n]/5
 assert c.sh(avg)==c.sh(obj['state_dict'])==rr['averaged_state_sha256'];optimizer=torch.load(_repo_io(folder/'optimizer-final.pt'),map_location='cpu',weights_only=True);assert all(int(v['step'])==steps for v in optimizer['state'].values());assert all(q['weight_decay']==.01 and q['lr']==.0001 and q['betas']==(.9,.999) and q['eps']==1e-8 for q in optimizer['param_groups']);assert all(torch.isfinite(t).all() for v in optimizer['state'].values() for t in v.values() if torch.is_tensor(t))
 rng=np.random.default_rng(seed+node['offset']);X=(images[T]-norm['x_mean'])/max(norm['x_std'],1e-6);step=0
 for epoch in range(80):
  order=rng.permutation(len(T))
  for start in range(0,len(T),32):
   ix=order[start:start+32];rot=int(rng.integers(4));flip=rng.random()<.5;assert abs(trace['LR'][step]-(.0001+(.002-.0001)*(1+math.cos(math.pi*epoch/80))/2))<1e-15
   if str(step) in rr['input_hashes']:
    xx=np.rot90(X[ix],rot,axes=(2,3));xx=xx[:,:,:,::-1] if flip else xx;assert c.ah(np.ascontiguousarray(xx))==rr['input_hashes'][str(step)]
   step+=1
 assert step==steps;np.testing.assert_array_equal(saved['train'],T);np.testing.assert_array_equal(saved['validation'],V);errors={}
 for device in ['mps','cpu']:
  pv=c.predict(obj,images,V,device);pt=c.predict(obj,images,T,device);err=max(float(abs(pv-saved['validation_predictions']).max()),float(abs(pt-saved['train_predictions']).max()));errors[device]=err;assert err==0 if device=='mps' else err<=.001
 result={'node':node['name'],'seed':seed,'updates':steps,'reload_errors':errors,'normalization_initialization_LR_batchhash_optimizer_last5_verified':True,'all_nontrain_feature_label_poison_stats_exact':True};del obj,avg,state,optimizer,X;return saved['validation_predictions'],result

def main():
 lock=c.J(O/'analysis-code-lock.json')
 for name,h in lock['sources'].items():assert c.H(O/name)==h
 c.check();done=c.J(O/'training-complete.json');assert done['passed'] and done['main_fits']==540 and done['validation_fits']==48;assert c.J(O/'determinism-precondition.json')['MPS_bitwise_exact'];z,images=c.data();start=time.perf_counter();lookup={};checks=[]
 with c.Monitor('model-validation',2400):
  for node in PLAN['nodes']:
   arr=[]
   for seed in PLAN['new_seeds']:
    pred,rec=validate_one(z,images,node,seed);arr.append(pred);checks.append(rec)
   lookup[node['name']]=np.stack(arr)
   if len(checks)%15==0:c.emit({'stage':'CNN_reload_audit','complete':len(checks),'total':540,'seconds':time.perf_counter()-start})
   gc.collect();torch.mps.empty_cache()
 assert len(checks)==540
 c.arrays(O/'new-nested-predictions.npz',**lookup);c.W(O/'model-validation.json',{'passed':True,'models':540,'MPS_exact_all':True,'max_CPU_error':max(x['reload_errors']['cpu'] for x in checks),'checks':checks,'new_predictions_sha256':c.H(O/'new-nested-predictions.npz'),'training_report_sha256':c.H(O/'training-complete.json'),'seconds':time.perf_counter()-start})
 result=np.full((3,400),np.nan);bases=result.copy();correction=result.copy();slopes=result.copy();fdchecks=[];allanchors=[]
 with c.Monitor('N-downstream',2400):
  for si,split in enumerate(SPLITS):
   for k in range(5):
    folder=O/'N'/split/f'fold-{k}';pred,rec=p.fit_fold(z,si,k,'N',folder,lookup);val=p.validate_fold(folder,z,si,k,'N',lookup);T,V=c.indices(z,si,k);labels=z['y'].copy();labels[V]=np.nan;other=O/'N-outer-poison'/split/f'fold-{k}';p.fit_fold(z,si,k,'N',other,lookup,labels);assert p.compare_poison(folder,other);fz=c.Z(folder/'predictions.npz');result[si,V]=pred;bases[si,V]=fz['base'];correction[si,V]=fz['correction'];slopes[si,V]=rec['outer_calibrator']['slope']
    rz=c.Z(O/'R'/split/f'fold-{k}/predictions.npz');cnn_delta=z['newouter'][si][:,V].mean(0)/2-z['oldouter'][si][:,V].mean(0)/2;np.testing.assert_allclose(fz['raw_outer']-rz['raw_outer'],.3*cnn_delta,atol=1e-10,rtol=0)
    for j in range(3):
     a=c.Z(folder/f'block-{j}/predictions.npz');b=c.Z(O/'R'/split/f'fold-{k}/block-{j}/predictions.npz');np.testing.assert_array_equal(a['draws'],b['draws']);allanchors.append({'split':split,'fold':k,'block':j,'anchor_delta_RMS':float(np.sqrt(np.mean((a['anchor']-b['anchor'])**2))),'residual_delta_RMS':float(np.sqrt(np.mean((a['residual']-b['residual'])**2))),'bootstrap_L1_change':float(abs(a['weights']-b['weights']).sum()),'weight_R':b['weights'].tolist(),'weight_N':a['weights'].tolist(),'anchor_R_sha256':c.ah(b['anchor']),'anchor_N_sha256':c.ah(a['anchor'])})
    fdchecks.append({'split':split,'fold':k,'independent':val,'outer_nan_poison_exact':True,'all_anchors_B_V_poison_exact':True,'same_paired_bootstrap_draws':True});c.emit({'stage':'N_downstream','folds_complete':len(fdchecks),'total':15,'max_independent_error':val['max_independent_error']})
 np.testing.assert_array_equal(c.Z(O/'R-rebuilt.npz')['predictions'],z['R']);delta_cnn=z['newouter'].mean(1)/2-z['oldouter'].mean(1)/2;F=z['R']+.243*z['oldslopes']*delta_cnn;np.testing.assert_allclose(F,z['F'],atol=1e-10,rtol=0)
 # Fixed outer6seed rows allow exact prediction decomposition into affine-base and residual-adaptation effects.
 oldbase=(z['R']-.1*z['L'])/.9-z['oldcorrection'];Fbase=oldbase+.9*z['oldslopes']*.3*delta_cnn;affine_delta=.9*(bases-Fbase);residual_delta=.9*(correction-z['oldcorrection']);np.testing.assert_allclose(result-z['F'],affine_delta+residual_delta,atol=1e-10,rtol=0)
 c.arrays(O/'ensemble-OOF.npz',ids=z['ids'],y=z['y'],groups=z['groups'],fold=z['fold'],R=z['R'],F=z['F'],N=result,Nbase=bases,Ncorrection=correction,Nslopes=slopes,Rslopes=z['oldslopes'],affine_adaptation_delta=affine_delta,residual_adaptation_delta=residual_delta,old_CNN=z['oldouter'],new_CNN=z['newouter'])
 c.W(O/'downstream-validation.json',{'passed':True,'R_preflight':c.J(O/'preflight.json')['passed'],'N_folds':15,'checks':fdchecks,'anchor_changes':allanchors,'alllevel_raw_difference_identity_checked':True,'F_reconstruction_max_error':float(abs(F-z['F']).max()),'N_minus_F_decomposition_exact_within1e10':True,'max_independent_error':max(r['independent']['max_independent_error'] for r in fdchecks),'ensemble_sha256':c.H(O/'ensemble-OOF.npz'),'new_nested_predictions_sha256':c.H(O/'new-nested-predictions.npz'),'no_comparative_scores_yet':True})
 c.check();c.access_report('finish-access.json');c.emit({'all204_model_and_downstream_checks_passed':True,'ensemble_sha256':c.H(O/'ensemble-OOF.npz')})
if __name__=='__main__':
 try:main()
 except BaseException as exc:c.W(O/'finish-failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()});raise
