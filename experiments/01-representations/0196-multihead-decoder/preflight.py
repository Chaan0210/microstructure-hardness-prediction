import common as c,engine as e,audit_helpers as ah
from common import np,torch,R
import math,gc
c.check();assert torch.backends.mps.is_available();d=c.data();mom=c.moments(d);T,V=c.indices(d,0);norm=c.normalize(d,T,mom);(R/'smoke').mkdir();records=[];counts={};states={}
with c.Monitor('preflight',180):
 with c.TokenReader() as reader:raw=reader.read(T[:2]);np.testing.assert_array_equal(raw[0],reader.read(T[:1])[0])
 for arm in c.ARMS:
  m=e.initialize(norm,0,'cpu');m.eval();counts[arm]=sum(p.numel() for p in m.parameters());states[arm]=c.statehash(m.state_dict());x=torch.from_numpy(raw);p=m(x,arm);target=torch.tensor((d['y'][T[:2]]-norm['target_mean'])/norm['target_std'],dtype=torch.float32);loss=e.F.smooth_l1_loss(p,target,beta=1);loss.backward();grads={n:float(a.grad.norm()) if a.grad is not None else None for n,a in m.named_parameters()};assert all(v is not None and np.isfinite(v) and v>0 for v in grads.values());m.eval()
  with torch.no_grad():
   H=m.token_norm(e.F.gelu(m.project(torch.cat(((x-m.mean)/m.scale,m.coords.expand(2,-1,-1)),2))));q=m.queries.expand(2,-1,-1);mix,w,Q,KV=m.cross.mix(q,H,arm);qn=Q.numpy().astype(float);kv=KV.numpy().astype(float)
   if arm=='mha':
    score=np.einsum('bhqd,bhnd->bhqn',qn,kv)/math.sqrt(8);score-=score.max(-1,keepdims=True);ww=np.exp(score);ww/=ww.sum(-1,keepdims=True);manual=np.einsum('bhqn,bhnd->bhqd',ww,kv);assert float(abs(w-1/1296).max())>1e-8
   else:manual=np.broadcast_to(kv.mean(2,keepdims=True),(2,4,4,8));np.testing.assert_allclose(w.numpy(),1/1296,rtol=0,atol=1e-9)
   expected=manual.transpose(0,2,1,3).reshape(2,4,32);err=float(abs(expected-mix.numpy()).max());assert err<1e-6;np.testing.assert_allclose(w.sum(-1).numpy(),1,atol=1e-6,rtol=0);order=np.random.default_rng(196).permutation(1296);p1=m(x[:,order],arm,coords=m.coords[order]);invariance=float(abs(p1-p).max());assert invariance<1e-5
  records.append({'arm':arm,'parameters':counts[arm],'CPU_all_parameter_tensor_gradients_nonzero':grads,'CPU_numpy_cross_attention_maxabs':err,'token_coordinate_pair_permutation_prediction_maxabs':invariance});del m;gc.collect()
 assert counts['mean']==counts['mha']==38209 and states['mean']==states['mha'];c.emit({'parameter_matching_and_manual_attention_passed':True,'parameters_each':38209})
 reports={}
 for arm in c.ARMS:
  _,report=e.fit(d,mom,0,arm,R/'smoke'/arm,max_updates=8);reports[arm]=report;records.append(ah.verify_reload(d,mom,0,arm,R/'smoke'/arm));poison=dict(d);poison['y']=d['y'].copy();poison['y'][V]=1e6-1000*poison['y'][V];e.fit(poison,mom,0,arm,R/'smoke'/(arm+'-poison'),max_updates=8);assert ah.compare_replay(R/'smoke'/arm,R/'smoke'/(arm+'-poison'));records.append({'arm':arm,'eight_update_heldout_label_poison_model_optimizer_prediction_trace_exact':True});c.emit({'smoke_arm_passed':arm})
 for key in ['initial_state_SHA256','order_SHA256','dropseed_SHA256','LR_SHA256','parameters','updates','first_input_hashes']:assert reports['mean'][key]==reports['mha'][key],key
 projected=sum(q['update_seconds_median_after2'] for q in reports.values())*400*5*1.5;resource_pass=projected<=1200;result={'passed':bool(resource_pass),'checks':records,'parameter_count_each':38209,'projected_main_seconds_with50pct_overhead':projected,'resource_projection_pass':bool(resource_pass),'smoke_updates_each':8,'validation_scores_used_for_selection':False,'MPS_version':torch.__version__,'plan_SHA256':c.sha(R/'plan.json'),'code_lock_SHA256':c.sha(R/'code-lock.json')};c.write(R/'preflight.json',result);c.emit({key:value for key,value in result.items() if key!='checks'});assert resource_pass
c.check()
