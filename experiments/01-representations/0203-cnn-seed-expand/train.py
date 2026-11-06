from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import contextlib,gc,io,math,inspect,time
import common as c
from common import np,torch,R,O,PLAN,SPLITS,fv

def baseline(z,images):
 checks=[]
 for si,split in enumerate(SPLITS):
  for k in range(5):
   T,V=c.indices(z,si,k);norm=c.normalization(images,z['y'],T)
   for j,seed in enumerate(PLAN['old_seeds']):
    path=R/PLAN['checkpoints'][split][str(k)][str(seed)];obj=torch.load(_repo_io(path),map_location='cpu',weights_only=True)
    assert obj['train_ids']==z['ids'][T].tolist() and obj['normalization']==norm
    if 'validation_ids' in obj:assert obj['validation_ids']==z['ids'][V].tolist()
    else:assert [i for i in z['ids'].tolist() if i not in set(obj['train_ids'])]==z['ids'][V].tolist()
    assert obj.get('seed',seed+k)==seed+k;assert sum(v.numel() for v in obj['state_dict'].values())==91905
    errors={d:float(abs(c.predict(obj,images,V,d)-z['CNN_seeds'][si,j,V]).max()) for d in ['mps','cpu']}
    assert max(errors.values())<=.001
    checks.append({'split':split,'fold':k,'base_seed':seed,'model_sha256':c.H(path),'normalization_and_trainIDs_exact':True,'validation_ID_metadata_present':'validation_ids' in obj,'validation_scope':'stored IDs when available,otherwise exact development-complement plus cachedfold/prediction verification','stored_seed_verified': 'seed' in obj,'reload_errors':errors});del obj
   c.emit({'stage':'old3_reload','split':split,'fold':k,'models_verified':len(checks)})
  gc.collect();torch.mps.empty_cache()
 c.W(O/'baseline-validation.json',{'passed':True,'models':45,'new_training':0,'checks':checks,'MPS_exact_count':sum(r['reload_errors']['mps']==0 for r in checks),'max_MPS_error':max(r['reload_errors']['mps'] for r in checks),'max_CPU_error':max(r['reload_errors']['cpu'] for r in checks),'cached_reference_unchanged':True})

class Progress(io.TextIOBase):
 def __init__(self,split,seed,k):self.buffer='';self.split=split;self.seed=seed;self.k=k
 def write(self,s):
  self.buffer+=s
  while '\n' in self.buffer:
   line,self.buffer=self.buffer.split('\n',1)
   if line.strip():
    r=c.json.loads(line)
    if r.get('epoch')==80:c.emit({'split':self.split,'seed':self.seed,'fold':self.k,'epoch':80,'stage':'training_endpoint'})
  return len(s)

def fit(z,images,si,k,seed):
 split=SPLITS[si];folder=O/split/f'seed-{seed}-fold-{k}';folder.mkdir(parents=True,exist_ok=False);T,V=c.indices(z,si,k);start=time.perf_counter();norm=c.normalization(images,z['y'],T)
 masked=z['y'].copy();masked[V]=0.;assert np.array_equal(masked[T],z['y'][T])
 c.emit({'fit_start':str(folder.relative_to(O)),'main_fits_total':45})
 capture={'forwards':0,'updates':0,'LR':[],'input_hashes':{},'epoch_state_hashes':{}}
 old_ctor=torch.optim.AdamW;oldseed=fv.CNN_SEED
 def forward_hook(module,args):
  if isinstance(module,fv.engine.TinyCNN) and module.training:
   step=capture['forwards']
   if step==0:capture['model']=module;capture['initial_hash']=c.sh(module.state_dict());assert sum(p.numel() for p in module.parameters())==91905
   if step in [0,99,399,799]:capture['input_hashes'][str(step)]=c.ah(args[0].detach().cpu().numpy())
   capture['forwards']+=1
 def after_step(opt,args,kwargs):
  capture['updates']+=1;capture['LR'].append(float(opt.param_groups[0]['lr']));step=capture['updates']
  if step%10==0 and step>=760:
   epoch=step//10;state=c.cpu(capture['model'].state_dict());torch.save(state,_repo_io(folder/f'epoch-{epoch}.pt'));capture['epoch_state_hashes'][str(epoch)]=c.sh(state)
  if step%10==0 and time.perf_counter()-start>180:raise RuntimeError('Per-fit180second budget exceeded')
 def ctor(*args,**kwargs):
  opt=old_ctor(*args,**kwargs);capture['optimizer']=opt;capture['optimizer_hook']=opt.register_step_post_hook(after_step);return opt
 handle=torch.nn.modules.module.register_module_forward_pre_hook(forward_hook)
 try:
  torch.optim.AdamW=ctor;fv.CNN_SEED=seed
  with contextlib.redirect_stdout(Progress(split,seed,k)):
   prediction,summary=fv.cnn_fold(images,masked,T,V,z['ids'].tolist(),k,'mps',folder/'model.pt')
 finally:
  torch.optim.AdamW=old_ctor;fv.CNN_SEED=oldseed;handle.remove()
  if 'optimizer_hook' in capture:capture['optimizer_hook'].remove()
 assert capture['updates']==capture['forwards']==800
 obj=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);assert obj['normalization']==norm
 avg=None
 for epoch in range(76,81):
  state=torch.load(_repo_io(folder/f'epoch-{epoch}.pt'),map_location='cpu',weights_only=True)
  if avg is None:avg={n:v/5 for n,v in state.items()}
  else:
   for n in avg:avg[n]+=state[n]/5
 assert c.sh(avg)==c.sh(obj['state_dict']);np.testing.assert_array_equal(prediction,c.predict(obj,images,V,'mps'))
 trainpred=c.predict(obj,images,T,'mps');assert np.isfinite(prediction).all() and np.isfinite(trainpred).all()
 torch.save(c.cpu(capture['optimizer'].state_dict()),_repo_io(folder/'optimizer-final.pt'))
 c.arrays(folder/'predictions.npz',train_indices=T,validation_indices=V,validation_predictions=prediction,train_predictions=trainpred)
 c.arrays(folder/'trace.npz',LR=np.array(capture['LR']))
 report={'split':split,'fold':k,'base_seed':seed,'effective_seed':seed+k,'train_labels_only':True,'validation_labels_masked_before_trainer':True,'true_validation_RMSE_not_computed':True,'training_curve':summary['curve'],'train_RMSE':summary['train_rmse'],'updates':capture['updates'],'training_forward_calls':capture['forwards'],'initial_state_sha256':capture['initial_hash'],'input_hashes':capture['input_hashes'],'epoch_state_hashes':capture['epoch_state_hashes'],'averaged_state_sha256':c.sh(avg),'model_sha256':c.H(folder/'model.pt'),'predictions_sha256':c.H(folder/'predictions.npz'),'trace_sha256':c.H(folder/'trace.npz'),'optimizer_sha256':c.H(folder/'optimizer-final.pt'),'trainer_source_sha256':c.H(R/'frozen_validation.py'),'seconds':time.perf_counter()-start}
 c.W(folder/'report.json',report);c.emit({'fit_done':str(folder.relative_to(O)),'seconds':report['seconds'],'updates':800,'scores_deferred':True})
 del capture,obj,avg;gc.collect();torch.mps.empty_cache();return prediction,report

def main():
 c.check();z,images=c.data();assert not (O/'study-complete.json').exists();start=time.perf_counter();reports=[];new=np.full((3,3,400),np.nan)
 with c.Monitor('study',3600):
  baseline(z,images)
  for si,split in enumerate(SPLITS):
   for j,seed in enumerate(PLAN['new_seeds']):
    for k in range(5):
     T,V=c.indices(z,si,k);p,r=fit(z,images,si,k,seed);new[si,j,V]=p;reports.append(r)
     if len(reports)==1:
      eta=44*r['seconds'];c.W(O/'first-fit-benchmark.json',{'seconds':r['seconds'],'remaining_main_fits':44,'projected_remaining_seconds':eta,'within_total_budget':eta+time.perf_counter()-start<3600});assert eta+time.perf_counter()-start<3600;c.emit({'first_fit_ETA_remaining_minutes':eta/60,'planned_remaining_fits':44})
     c.emit({'completed_new_fits':len(reports),'total':45,'elapsed_seconds':time.perf_counter()-start})
 assert len(reports)==45 and np.isfinite(new).all();c.arrays(O/'new-seed-OOF.npz',ids=z['ids'],y=z['y'],fold=z['fold'],groups=z['groups'],seeds=np.array(PLAN['new_seeds']),predictions=new)
 c.check();c.W(O/'study-complete.json',{'passed':True,'new_fits':45,'old_replay_training':0,'comparative_scores_computed':False,'all3seed_equal_retention_required':True,'seconds':time.perf_counter()-start,'sum_fit_seconds':sum(r['seconds'] for r in reports),'reports':reports,'OOF_sha256':c.H(O/'new-seed-OOF.npz'),'plan_sha256':c.H(O/'plan.json')});c.access_report('training-access-audit.json');c.emit({'all45fits_complete':True,'comparative_scoring_deferred_until_validation':True})
if __name__=='__main__':
 try:main()
 except Exception as exc:c.W(O/('training-failure-'+str(int(time.time()))+'.json'),{'error':repr(exc),'type':type(exc).__name__,'blocked':c.BLOCKED});raise
