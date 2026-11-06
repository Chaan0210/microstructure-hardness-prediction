from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import gc
import common as c,engine as e,audit_helpers as ah,mechanism as mech
from legacy import engine as old
from common import np,torch,R,P
c.check();assert torch.backends.mps.is_available();d=c.data();mom=c.moments(d);oldroot=P/'experiments/multihead-decoder-196';(R/'legacy-reproduction').mkdir();(R/'smoke').mkdir();records=[];initial=[];gaps=[]
with c.Monitor('preflight',300) as monitor:
 with c.TokenReader() as reader:
  for k in range(5):
   T,V=c.indices(d,k);norm=c.normalize(d,T,mom);a=old.initialize(norm,k,'cpu');b=e.initialize(norm,k,'cpu');assert c.statehash(a.state_dict())==c.statehash(b.state_dict())==c.read(oldroot/f'fold-{k}/mha/report.json')['initial_state_SHA256'];assert list(a.state_dict())==list(b.state_dict());assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())==38209;q=b.queries.detach().double();manual=(q-q.mean(-1,keepdim=True))/torch.sqrt((q-q.mean(-1,keepdim=True)).square().mean(-1,keepdim=True)+1e-5);actual=e.F.layer_norm(b.queries,(32,),weight=None,bias=None,eps=1e-5);np.testing.assert_allclose(actual.detach().numpy(),manual.numpy(),atol=1e-6,rtol=1e-6)
   r0=mech.probe(a,reader,T[:16],norm,d['y']);r1=mech.probe(b,reader,T[:16],norm,d['y']);initial.append({'fold':k,'raw196':r0,'qnorm197':r1});c.emit({'initial_mechanism_fold':k,'raw_logit_std':r0['median']['cross_logit_std'],'qnorm_logit_std':r1['median']['cross_logit_std'],'raw_ESS':r0['median']['cross_ESS'],'qnorm_ESS':r1['median']['cross_ESS'],'raw_score_value_gradient_ratio':r0['cross_gradient']['score_over_value_norm'],'qnorm_score_value_gradient_ratio':r1['cross_gradient']['score_over_value_norm']});del a,b
 c.write(R/'mechanism-initial.json',{'probe':'first16outertrain,eval,nooptimizerupdate','folds':initial})
 T,V=c.indices(d,0);norm=c.normalize(d,T,mom)
 with c.TokenReader() as reader:raw=reader.read(T[:2])
 for device in ['cpu','mps']:
  runs=[]
  for module in [old,e]:
   m=module.initialize(norm,0,device);m.train();torch.manual_seed(e.seed(0,0,0));x=torch.from_numpy(raw).to(device);target=torch.tensor((d['y'][T[:2]]-norm['target_mean'])/norm['target_std'],dtype=torch.float32,device=device);p=m(x,'mean');loss=e.F.smooth_l1_loss(p,target,beta=1);loss.backward();runs.append({'prediction':p.detach().cpu(),'loss':loss.detach().cpu(),'gradients':{n:z.grad.detach().cpu().clone() for n,z in m.named_parameters()},'state':e.snapshot(m)});del m
  ah.equal(runs[0],runs[1]);records.append({'device':device,'mean_raw_vs_Qnorm_prediction_loss_gradients_initialstate_exact':True});c.emit(records[-1])
 gc.collect();torch.mps.empty_cache();legacy_reports={}
 for arm in c.ARMS:
  _,r=old.fit(d,mom,0,arm,R/'legacy-reproduction'/arm);assert ah.compare_replay(oldroot/'fold-0'/arm,R/'legacy-reproduction'/arm);legacy_reports[arm]=r;records.append({'arm':arm,'original196_full400update_reproduction_exact':True});c.emit(records[-1]);monitor.sample()
 smoke_reports={}
 for arm in c.ARMS:
  _,r=e.fit(d,mom,0,arm,R/'smoke'/arm,max_updates=8);smoke_reports[arm]=r;records.append(ah.verify_reload(d,mom,0,arm,R/'smoke'/arm))
  if arm=='mean':assert ah.compare_replay(oldroot/'smoke/mean',R/'smoke/mean');records.append({'normalizedmean8update_exact196smoke':True})
  else:
   poison=dict(d);poison['y']=d['y'].copy();poison['y'][V]=1e6-1000*poison['y'][V];e.fit(poison,mom,0,arm,R/'smoke/mha-poison',max_updates=8);assert ah.compare_replay(R/'smoke/mha',R/'smoke/mha-poison');records.append({'normalizedMHA8update_heldoutlabelpoison_exact':True})
  for version,module,path in [('196',old,oldroot/'smoke'/arm),('197',e,R/'smoke'/arm)]:
   ck=torch.load(_repo_io(path/'model.pt'),map_location='cpu',weights_only=True);m=module.initialize(norm,0,'cpu');m.load_state_dict(ck['state']);
   with c.TokenReader() as reader:vp=module.predict(m,reader,V,norm,arm,'cpu');tp=module.predict(m,reader,T,norm,arm,'cpu')
   gaps.append({'stage':'8updates','version':version,'arm':arm,'train_RMSE':c.rmse(d['y'][T],tp),'validation_RMSE':c.rmse(d['y'][V],vp),'validation_minus_train_RMSE':c.rmse(d['y'][V],vp)-c.rmse(d['y'][T],tp),'selection_use':False});del m
 for key in ['initial_state_SHA256','parameters','updates','first_input_hashes','order_SHA256','dropseed_SHA256','LR_SHA256']:assert smoke_reports['mean'][key]==smoke_reports['mha'][key]
 projected=sum(q['update_seconds_median_after2'] for q in smoke_reports.values())*400*5*1.5;assert projected<=1200;c.write(R/'preflight-train-validation-gap.json',{'scope':'fold0devtrain320/validation80 diagnosticonly beforemain,noendpoint/configchoice','smoke8update':gaps,'reproduced196_full400update':{arm:{key:legacy_reports[arm][key] for key in ['train_RMSE','validation_RMSE']} for arm in c.ARMS}});result={'passed':True,'checks':records,'parameters_each':38209,'parameters_added':0,'source_model_diff_SHA256':c.sha(R/'model-change.diff'),'initial_mechanism_SHA256':c.sha(R/'mechanism-initial.json'),'train_validation_gap_SHA256':c.sha(R/'preflight-train-validation-gap.json'),'projected_main_seconds_with50pct_overhead':projected,'legacy196_full400update_reproductions':2,'smoke8update_fits':3,'no_score_based_configuration_or_budget_change':True,'plan_SHA256':c.sha(R/'plan.json'),'code_lock_SHA256':c.sha(R/'code-lock.json')};c.write(R/'preflight.json',result);c.emit(result)
c.check()
