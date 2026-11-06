"""Development-only adapter activation, gradient and objective diagnostics; no new model fit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,math
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import dinob_adapter_080 as e
import numpy as np,torch
from torch.nn import functional as F
fv=e.fv;OUT=_repo_path(ROOT, 'artifacts/adapter-diagnostics-082')

def prepare():
 e.check();OUT.mkdir(exist_ok=False);paths=[*ROOT.glob('artifacts/submission-candidate-010*'),*(_repo_path(ROOT, 'artifacts/final-model-081')).rglob('*'),_repo_path(ROOT, 'reports/candidate010-verification.json'),_repo_path(ROOT, 'reports/final-candidate010-summary.json')];protected={str(p.relative_to(ROOT)):e.backbone.streamsha(p) for p in paths if p.is_file()};p={'scope':'Development400 only, all15 alreadyfit080folds. No new training/hyperparameter choice, no500/testcache decoded. Existing010 and009 preserved; no upload.',
 'measurements':'All320training+80validationrows perouterfold: qkv/proj adapter branch RMS divided by original branch RMS; final normalized2304descriptor relativeL2 and cosine; B predictions and fixed009contribution changes. Training-only first16IDorderedrows: initial/final gradients for regression,anchor,total; all4 adaptertensor gradients, frozen gradient absence. Compare exactfulltrain objective against analytic freely-adjustable prediction optimum.',
 'analytic_control':'For standardized residualr=y-p0 withabs(r)<1, loss=.5(p-y)^2+(p-p0)^2 has p*=p0+r/3, minimum mean(r^2)/3. This is an unconstrainedperrow lowerbound, not a trainable candidate. Evaluate it only on training pairs. AdamW parameter decay not represented in this lossbound. No change chosen to force larger prediction shifts.',
 'limitations':'Endpoint gradient is a diagnostic on fixed16trainrows, not proof of optimizer convergence. Last onlineepoch objectives mixparameter states. Frozenheads already nearly interpolatetrainlabels; lowresidual signal andanchor can imply smalloptimal prediction movement even with livegradients. Validation associations descriptive, repeated400rows.',
 'source_sha256':fv.sha(Path(__file__)),'input_validation_sha256':fv.sha(_repo_path(ROOT, 'reports/dinob-adapter-080-validation.json')),'protected010':protected};fv.write(_repo_path(OUT, 'plan.json'),p)

def check():
 e.check();p=json.loads((_repo_path(OUT, 'plan.json')).read_text());assert fv.sha(Path(__file__))==p['source_sha256'] and fv.sha(_repo_path(ROOT, 'reports/dinob-adapter-080-validation.json'))==p['input_validation_sha256']
 for n,s in p['protected010'].items():assert e.backbone.streamsha(_repo_path(ROOT, n))==s,n
 return p

def grad_diagnostic(t,d,tokens,ix,old,ym,ys):
 names=[n for n,p in t.named_parameters() if p.requires_grad];params=[p for p in t.parameters() if p.requires_grad];reg=[torch.zeros_like(p) for p in params];anchor=[torch.zeros_like(p) for p in params];losses=[]
 for first in range(0,len(ix),2):
  ii=ix[first:first+2];p=t(e.tensor(tokens[ii]),e.tensor(d['B'][ii,:4608]));lr=F.smooth_l1_loss(p,e.tensor((d['y'][ii]-ym)/ys),beta=1.);la=F.mse_loss(p,e.tensor((old[ii]-ym)/ys));gr=torch.autograd.grad(lr,params,retain_graph=True);ga=torch.autograd.grad(la,params);factor=len(ii)/len(ix)
  for j in range(len(params)):reg[j]+=gr[j]*factor;anchor[j]+=ga[j]*factor
  losses.append([float(lr.detach())*factor,float(la.detach())*factor])
 r=torch.cat([p.reshape(-1) for p in reg]);a=torch.cat([p.reshape(-1) for p in anchor]);both=r+a;rn=float(r.norm());an=float(a.norm());bn=float(both.norm());assert torch.isfinite(both).all();record={'rows':len(ix),'ids':[d['ids'][i] for i in ix],'regression_loss':sum(z[0] for z in losses),'anchor_loss':sum(z[1] for z in losses),'regression_grad_L2':rn,'anchor_grad_L2':an,'combined_grad_L2':bn,'regression_anchor_gradient_cosine':float((r@a)/(r.norm()*a.norm())) if rn>0 and an>0 else None,'would_clip_at1':bn>1,'per_parameter':{n:{'regression_grad_L2':float(g.norm()),'anchor_grad_L2':float(ag.norm()),'total_grad_L2':float((g+ag).norm())} for n,g,ag in zip(names,reg,anchor)},'frozen_parameters_have_no_grad':all(p.grad is None for p in t.parameters() if not p.requires_grad)};assert record['frozen_parameters_have_no_grad'] and rn>0;return record

def measure(t,d,tokens):
 rows={n:{'original_rms':[],'adapter_rms':[]} for n in ('qkv','proj')};late=[];handles=[]
 def makehook(name):
  def hook(module,args,out):
   with torch.no_grad():
    delta=module.up(module.down(args[0]))*(e.CFG['alpha']/e.CFG['rank']);base=out-delta;rows[name]['original_rms'].extend(base.square().mean((1,2)).sqrt().cpu().numpy().astype(float).tolist());rows[name]['adapter_rms'].extend(delta.square().mean((1,2)).sqrt().cpu().numpy().astype(float).tolist())
  return hook
 def normhook(module,args,z):
  with torch.no_grad():
   p=z[:,1:];late.extend(torch.cat((z[:,0],p.mean(1),p.std(1,correction=0)),1).cpu().numpy())
 for name in rows:handles.append(getattr(t.block.attn,name).register_forward_hook(makehook(name)))
 handles.append(t.norm.register_forward_hook(normhook));pred=e.predict(t,d,tokens,np.arange(400))
 for h in handles:h.remove()
 return pred,np.array(late),{n:{k:np.array(v) for k,v in x.items()} for n,x in rows.items()}

def main():
 plan=check();torch.set_num_threads(2);v=json.loads((_repo_path(ROOT, 'reports/dinob-adapter-080-validation.json')).read_text());m=e.backbone.model();results={};start=time.perf_counter();maxreload=0.;maxlossidentity=0.;bounds=[];train_gainfractions=[]
 for split in ('original','20261031','20261101'):
  d=e.inputs(split);tokens=e.get_cache(d);splitout=_repo_path(OUT, split);splitout.mkdir(exist_ok=False);folds=[]
  for k in range(5):
   tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);src=e.OUT/split/f'fold-{k}';r=json.loads((src/'report.json').read_text());rr=json.loads((e.OUT/split/'report.json').read_text());assert fv.sha(e.OUT/split/'report.json')==v['results'][split]['report_sha256'];assert fv.sha(src/'report.json')==rr['fold_report_sha256'][f'fold-{k}/report.json'];assert fv.sha(src/'model.pt')==r['checkpoint_sha256'];states,old=e.head_states(d,k);ym=states[0]['target_mean'];ys=states[0]['target_std'];t=e.Tail(m,states,e.CFG['seed']+k).to('mps');probe=tr[:16];g0=grad_diagnostic(t,d,tokens,probe,old,ym,ys);a=torch.load(_repo_io(src/'model.pt'),weights_only=True,map_location='cpu');t.load_state_dict(a['state']);before=e.h.state_hash(t.state_dict());g1=grad_diagnostic(t,d,tokens,probe,old,ym,ys);p,late,branches=measure(t,d,tokens);p=p*ys+ym;assert before==e.h.state_hash(t.state_dict());delta=p-old
   with np.load(_repo_io(src/'predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(p[va],z['adapted']);np.testing.assert_array_equal(p[tr],z['train_predictions']);maxreload=max(maxreload,float(abs(p[va]-z['adapted']).max()))
   f0=d['B'][:,-2304:];diff=late-f0;relative=np.linalg.norm(diff,axis=1)/np.maximum(np.linalg.norm(f0,axis=1),1e-15);cos=np.sum(late*f0,axis=1)/(np.linalg.norm(late,axis=1)*np.linalg.norm(f0,axis=1));res=(d['y'][tr]-old[tr])/ys;after=(p[tr]-d['y'][tr])/ys;shift=delta[tr]/ys;assert max(abs(res))<1 and max(abs(after))<1;initial=.5*float(np.mean(res**2));reg=.5*float(np.mean(after**2));anch=float(np.mean(shift**2));total=reg+anch;bound=float(np.mean(res**2)/3);theoretical=old[tr]+(d['y'][tr]-old[tr])/3;fraction=(initial-total)/(initial-bound);bounds.append(bound);train_gainfractions.append(fraction);assert total+1e-9>=bound and np.isfinite(fraction)
   scale=json.loads((e.previous.base.OUT/split/f'fold-{k}/report.json').read_text())['slope']*.125;metrics={}
   for name,ix in [('train',tr),('validation',va)]:
    metrics[name]={'n':len(ix),'B_prediction_change_RMS':float(np.sqrt(np.mean(delta[ix]**2))),'B_prediction_change_abs_max':float(abs(delta[ix]).max()),'fixed009_contribution_change_RMS':float(np.sqrt(np.mean((scale*delta[ix])**2))),'normalized_last_descriptor_relative_L2_median':float(np.median(relative[ix])),'normalized_last_descriptor_relative_L2_max':float(relative[ix].max()),'normalized_last_descriptor_cosine_median':float(np.median(cos[ix])),'branches':{n:{'original_output_RMS':float(np.sqrt(np.mean(z['original_rms'][ix]**2))),'adapter_output_RMS':float(np.sqrt(np.mean(z['adapter_rms'][ix]**2))),'adapter_to_original_L2_ratio':float(np.sqrt(np.mean(z['adapter_rms'][ix]**2)/np.mean(z['original_rms'][ix]**2)))} for n,z in branches.items()}}
   rec={'fold':k,'metrics':metrics,'gradients_initial':g0,'gradients_final':g1,'exact_endpoint_train_objective':{'initial_regression':initial,'final_regression':reg,'final_anchor':anch,'final_total':total,'freely_adjustable_prediction_lower_bound':bound,'fraction_of_available_objective_reduction_achieved':fraction,'B_train_RMSE_before':fv.rmse(d['y'][tr],old[tr]),'B_train_RMSE_after':fv.rmse(d['y'][tr],p[tr]),'unconstrained_anchor_optimum_RMSE':fv.rmse(d['y'][tr],theoretical),'max_initial_standardized_residual':float(abs(res).max())},'online_epoch_objectives':[c['train_objective'] for c in r['curve']],'checkpoint_sha256':r['checkpoint_sha256'],'measurement_state_unchanged':True};fv.write(splitout/f'fold-{k}.json',rec)
   with (splitout/f'fold-{k}-measurements.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),train_indices=tr,validation_indices=va,frozen_prediction=old,adapted_prediction=p,descriptor_relative_L2=relative,descriptor_cosine=cos,**{n+'_'+key:val for n,z in branches.items() for key,val in z.items()})
   folds.append(rec);print(json.dumps({'split':split,'completed_fold':k,'objective_fraction':fraction,'train_B_shift_RMS':metrics['train']['B_prediction_change_RMS'],'validation_B_shift_RMS':metrics['validation']['B_prediction_change_RMS'],'seconds':time.perf_counter()-start}),flush=True);del t;torch.mps.empty_cache()
  results[split]={'folds':folds,'mean_fraction_of_available_objective_reduction':float(np.mean([f['exact_endpoint_train_objective']['fraction_of_available_objective_reduction_achieved'] for f in folds]))}
 allfolds=[f for s in results.values() for f in s['folds']];summary={'training_objective_reduction_fraction_range':[min(train_gainfractions),max(train_gainfractions)],'training_objective_reduction_fraction_mean':float(np.mean(train_gainfractions)),'all_initial_up_gradients_positive':all(all(g['regression_grad_L2']>0 for n,g in f['gradients_initial']['per_parameter'].items() if n.endswith('up.weight')) for f in allfolds),'all_final_adapter_gradients_positive':all(all(g['regression_grad_L2']>0 for g in f['gradients_final']['per_parameter'].values()) for f in allfolds),'initial_down_zero_expected':all(all(g['regression_grad_L2']==0 for n,g in f['gradients_initial']['per_parameter'].items() if n.endswith('down.weight')) for f in allfolds),'max_prediction_reload_error':maxreload,'validation_prediction_shift_RMS_range':[min(f['metrics']['validation']['B_prediction_change_RMS'] for f in allfolds),max(f['metrics']['validation']['B_prediction_change_RMS'] for f in allfolds)],'final_gradients_would_clip_count':sum(f['gradients_final']['would_clip_at1'] for f in allfolds)};assert summary['all_initial_up_gradients_positive'] and summary['all_final_adapter_gradients_positive'] and summary['initial_down_zero_expected'];check();report={'passed':True,'results':results,'summary':summary,'bug_found':False,'interpretation':'Gradients flow and predictions/features change; zero initialdown gradients are expected with zero up matrices. Fixed nearly-interpolatingheads create smalltrainingresiduals; under quadraticSmoothL1+anchor1 the optimal perrow change is only one third ofthat residual. Availablelossreductionfraction quantifies progress, but endpointgradients do not proveglobalconvergence or isolate LR from lowrankcapacity/anchoring. A fullblock control must retain the same functionalanchor rather than forcing a larger outputshift.','test_holdout_access':False,'new_training_runs':0,'protected010_files':len(plan['protected010']),'seconds':time.perf_counter()-start,'source_sha256':fv.sha(Path(__file__))};fv.write(_repo_path(OUT, 'report.json'),report);print(json.dumps({'passed':True,'summary':summary,'seconds':report['seconds']}),flush=True)
if __name__=='__main__':
 prepare() if len(sys.argv)>1 and sys.argv[1]=='prepare' else main()
