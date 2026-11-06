from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import contextlib,io,gc,math,time,traceback
import common as c
from common import np,torch,R,O,PLAN,SPLITS,fv

class QuietProgress(io.TextIOBase):
 def write(self,s):return len(s)

def path_for(node,seed,phase='main'):return O/'CNN'/phase/node['name']/f'seed-{seed}'

def fit(z,images,node,seed,phase='main',label_mode='masked'):
 folder=path_for(node,seed,phase);folder.mkdir(parents=True,exist_ok=False);d=c.node_data(node);T=d['train'];V=d['validation'];steps_per_epoch=math.ceil(len(T)/32);steps=80*steps_per_epoch;start=time.perf_counter();norm=c.normalization(images,z['y'],T)
 outside=np.setdiff1d(np.arange(400),T);labels=z['y'].copy()
 if label_mode=='masked':labels[outside]=0.
 elif label_mode=='finite':labels[outside]=12345.+np.arange(len(outside))
 elif label_mode=='nan':labels[outside]=np.nan
 else:raise ValueError(label_mode)
 np.testing.assert_array_equal(labels[T],z['y'][T]);effective=seed+node['offset'];capture={'forwards':0,'updates':0,'LR':[],'input_hashes':{},'epoch_state_hashes':{}};inspect_steps={0,steps_per_epoch-1,steps//2-1,steps-1};oldctor=torch.optim.AdamW;oldseed=fv.CNN_SEED
 def forward_hook(module,args):
  if isinstance(module,fv.engine.TinyCNN) and module.training:
   step=capture['forwards']
   if step==0:capture['model']=module;capture['initial_hash']=c.sh(module.state_dict());assert sum(p.numel() for p in module.parameters())==91905
   if step in inspect_steps:capture['input_hashes'][str(step)]=c.ah(args[0].detach().cpu().numpy())
   capture['forwards']+=1
 def after_step(opt,args,kwargs):
  capture['updates']+=1;capture['LR'].append(float(opt.param_groups[0]['lr']));step=capture['updates']
  if step%steps_per_epoch==0 and step>=76*steps_per_epoch:
   epoch=step//steps_per_epoch;state=c.cpu(capture['model'].state_dict());torch.save(state,_repo_io(folder/f'epoch-{epoch}.pt'));capture['epoch_state_hashes'][str(epoch)]=c.sh(state)
  if step%steps_per_epoch==0 and time.perf_counter()-start>180:raise RuntimeError('Per-fit180seconds exceeded')
 def ctor(*args,**kwargs):
  opt=oldctor(*args,**kwargs);capture['optimizer']=opt;capture['optimizer_hook']=opt.register_step_post_hook(after_step);return opt
 handle=torch.nn.modules.module.register_module_forward_pre_hook(forward_hook)
 try:
  torch.optim.AdamW=ctor;fv.CNN_SEED=seed
  with contextlib.redirect_stdout(QuietProgress()):prediction,summary=fv.cnn_fold(images,labels,T,V,z['ids'].tolist(),node['offset'],'mps',folder/'model.pt')
 finally:
  torch.optim.AdamW=oldctor;fv.CNN_SEED=oldseed;handle.remove()
  if 'optimizer_hook' in capture:capture['optimizer_hook'].remove()
 assert capture['forwards']==capture['updates']==steps;obj=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);assert obj['normalization']==norm
 avg=None
 for epoch in range(76,81):
  state=torch.load(_repo_io(folder/f'epoch-{epoch}.pt'),map_location='cpu',weights_only=True)
  if avg is None:avg={n:v/5 for n,v in state.items()}
  else:
   for n in avg:avg[n]+=state[n]/5
 assert c.sh(avg)==c.sh(obj['state_dict']);np.testing.assert_array_equal(prediction,c.predict(obj,images,V,'mps'));trainpred=c.predict(obj,images,T,'mps');assert np.isfinite(prediction).all()
 torch.save(c.cpu(capture['optimizer'].state_dict()),_repo_io(folder/'optimizer-final.pt'));c.arrays(folder/'predictions.npz',train=T,validation=V,validation_predictions=prediction,train_predictions=trainpred);c.arrays(folder/'trace.npz',LR=np.array(capture['LR']))
 rr={'phase':phase,'node':node['name'],'split':node['split'],'outer':node['outer'],'type':node['type'],'base_seed':seed,'offset':node['offset'],'effective_seed':effective,'label_mode':label_mode,'outside_train_labels_never_real':True,'train_count':len(T),'validation_count':len(V),'updates':steps,'training_forward_calls':capture['forwards'],'initial_state_sha256':capture['initial_hash'],'input_hashes':capture['input_hashes'],'epoch_state_hashes':capture['epoch_state_hashes'],'averaged_state_sha256':c.sh(avg),'curve':summary['curve'],'normalization':norm,'train_RMSE':summary['train_rmse'],'validation_metric_discarded':True,'model_sha256':c.H(folder/'model.pt'),'predictions_sha256':c.H(folder/'predictions.npz'),'trace_sha256':c.H(folder/'trace.npz'),'optimizer_sha256':c.H(folder/'optimizer-final.pt'),'seconds':time.perf_counter()-start};c.W(folder/'report.json',rr)
 del capture,obj,avg,state;gc.collect();torch.mps.empty_cache();return rr

def same(node,seed,phase):
 a=path_for(node,seed);b=path_for(node,seed,phase);ar=c.J(a/'report.json');br=c.J(b/'report.json')
 for key in ['initial_state_sha256','input_hashes','epoch_state_hashes','averaged_state_sha256','curve','normalization','train_RMSE','updates']:assert ar[key]==br[key],(node['name'],seed,phase,key)
 for fn in ['predictions.npz','trace.npz']:
  x=c.Z(a/fn);y=c.Z(b/fn)
  for key in x:np.testing.assert_array_equal(x[key],y[key])
 return {'node':node['name'],'seed':seed,'phase':phase,'all_weights_curves_stats_inputs_predictions_bitwise_exact':True}

def main():
 c.check();assert c.J(O/'preflight.json')['passed'];assert c.J(O/'review-disposition.json')['allow_preflight'];z,images=c.data();start=time.perf_counter();reports=[];tests=[];maincount=0;validationcount=0;initial=[PLAN['nodes'][0],PLAN['nodes'][1]];assert [n['type'] for n in initial]==['middle','deep'];done=set()
 with c.Monitor('study',9000):
  for node in initial:
   for seed in PLAN['new_seeds']:
    r=fit(z,images,node,seed);reports.append(r);maincount+=1;done.add((node['name'],seed));c.emit({'stage':'initial_main','main_complete':maincount,'validation_complete':validationcount,'total_main':540})
  for phase,mode in [('clean','masked'),('nan','nan')]:
   for node in initial:
    for seed in PLAN['new_seeds']:
     reports.append(fit(z,images,node,seed,phase,mode));tests.append(same(node,seed,phase));validationcount+=1;c.emit({'stage':phase+'_replay','main_complete':maincount,'validation_complete':validationcount,'total_validation':48})
  c.W(O/'determinism-precondition.json',{'passed':True,'main_fits':6,'clean_replays':6,'NaN_complement_replays':6,'MPS_bitwise_exact':True,'no_tolerance_change':True,'tests':tests.copy()});c.emit({'determinism_precondition_passed':True,'main_complete':maincount,'validation_complete':validationcount,'remaining_main':534})
  for node in PLAN['nodes']:
   for seed in PLAN['new_seeds']:
    if (node['name'],seed) in done:continue
    r=fit(z,images,node,seed);reports.append(r);maincount+=1;done.add((node['name'],seed))
    if maincount%3==0:
     m=[r for r in reports if r['phase']=='main'];avg={t:np.mean([r['seconds'] for r in m if r['type']==t]) for t in ['middle','deep']};remaining=sum(avg[n['type']] for n in PLAN['nodes'] for s in PLAN['new_seeds'] if (n['name'],s) not in done);c.emit({'stage':'nested_CNN','main_complete':maincount,'total_main':540,'validation_complete':validationcount,'last_node':node['name'],'split':node['split'],'outer':node['outer'],'elapsed_seconds':time.perf_counter()-start,'main_remaining_minutes':float(remaining/60),'last_fit_seconds':r['seconds']})
  assert maincount==540
  for node in PLAN['nodes']:
   if node['si']==0 and node['outer']==0:
    for seed in PLAN['new_seeds']:
     reports.append(fit(z,images,node,seed,'finite','finite'));tests.append(same(node,seed,'finite'));validationcount+=1
    c.emit({'stage':'finite_complement_replay','main_complete':maincount,'validation_complete':validationcount,'total_validation':48})
  assert validationcount==48;assert sum(r['updates'] for r in reports if r['phase']=='main')==237600
 c.check();c.W(O/'training-complete.json',{'passed':True,'main_fits':maincount,'validation_fits':validationcount,'main_updates':237600,'all_updates':sum(r['updates'] for r in reports),'sum_fit_seconds':sum(r['seconds'] for r in reports),'elapsed_seconds':time.perf_counter()-start,'reports':reports,'replay_checks':tests,'comparative_scoring_performed':False,'plan_sha256':c.H(O/'plan.json'),'code_lock_sha256':c.H(O/'code-lock.json')});c.access_report('training-access.json');c.emit({'training_complete':True,'main_fits':maincount,'validation_fits':validationcount,'elapsed_seconds':time.perf_counter()-start})
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  c.W(O/'training-failure.json',{'error_type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'no_automatic_rerun':True});raise
