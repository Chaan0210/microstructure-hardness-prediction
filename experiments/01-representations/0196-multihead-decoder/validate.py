import math
import common as c,engine as e,audit_helpers as ah
from common import np,R
c.check();d=c.data();mom=c.moments(d);report=c.read(R/'report.json');OOF=c.load(R/'OOF.npz');boot=c.load(R/'bootstrap.npz');assert c.sha(R/'OOF.npz')==report['OOF_SHA256'] and c.sha(R/'bootstrap.npz')==report['bootstrap_SHA256'];records=[]
for key in ['ids','y','fold','groups','best021']:np.testing.assert_array_equal(d[key],OOF[key])
def rmse(y,p):return math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(y,p))/len(y))
with c.Monitor('validation',600) as monitor:
 for k in range(5):
  pair={}
  for arm in c.ARMS:
   folder=R/f'fold-{k}'/arm;q=ah.verify_reload(d,mom,k,arm,folder);assert q['updates']==400;records.append(q);c.emit(q);saved=c.load(folder/'predictions.npz');V=saved['indices'];np.testing.assert_array_equal(OOF[arm][V],saved['prediction']);pair[arm]=c.read(folder/'report.json');assert abs(rmse(d['y'][V],saved['prediction'])-pair[arm]['validation_RMSE'])<1e-12;T=saved['train_indices'];assert abs(rmse(d['y'][T],saved['train_prediction'])-pair[arm]['train_RMSE'])<1e-12;monitor.sample()
  for key in ['initial_state_SHA256','parameters','updates','first_input_hashes','order_SHA256','dropseed_SHA256','LR_SHA256']:assert pair['mean'][key]==pair['mha'][key]
 audit=R/'poison-replays';audit.mkdir();T,V=c.indices(d,0);poison=dict(d);poison['y']=d['y'].copy();poison['y'][V]=1e6-1000*poison['y'][V]
 for arm in c.ARMS:
  e.fit(poison,mom,0,arm,audit/arm);assert ah.compare_replay(R/'fold-0'/arm,audit/arm);c.emit({'full20epoch_poison_replay_exact':arm,'updates':400});monitor.sample()
 ps={'best021':d['best021'],**{a:OOF[a] for a in c.ARMS}};ps.update({a+'_blend':.9*d['best021']+.1*OOF[a] for a in c.ARMS});unique,inv=np.unique(d['groups'],return_inverse=True);np.testing.assert_array_equal(boot['draws'],np.random.default_rng(202710196).integers(len(unique),size=(3000,len(unique))));assert len(unique)==400;ix=np.argsort(inv)[boot['draws']];max_boot=0.
 for name,q in report['scores'].items():
  assert abs(rmse(d['y'],ps[name])-q['RMSE'])<1e-12
  for k in range(5):
   v=d['fold']==k;assert abs(rmse(d['y'][v],ps[name][v])-q['fold_RMSE'][k])<1e-12
 for name,q in report['comparisons'].items():
  y=d['y'];a=ps[q['reference']];b=ps[q['candidate']];assert abs(rmse(y,a)-rmse(y,b)-q['gain'])<1e-12;bb=np.sqrt(np.mean((y[ix]-a[ix])**2,1))-np.sqrt(np.mean((y[ix]-b[ix])**2,1));error=float(abs(bb-boot[name]).max());assert error<1e-12;max_boot=max(max_boot,error);np.testing.assert_allclose(np.quantile(bb,[.025,.975]),q['conditional_paired_groupbootstrap_CI95'],atol=1e-12,rtol=0);benefit=[(float(y[i])-float(a[i]))**2-(float(y[i])-float(b[i]))**2 for i in range(400)];order=sorted(range(400),key=lambda i:(-benefit[i],str(d['ids'][i])));assert d['ids'][order[:5]].tolist()==q['top5_removed_IDs'];keep=np.array(order[5:]);assert abs(rmse(y[keep],a[keep])-rmse(y[keep],b[keep])-q['top5_removed_gain'])<1e-12;fg=[]
  for k in range(5):
   v=d['fold']==k;fg.append(rmse(y[v],a[v])-rmse(y[v],b[v]))
  np.testing.assert_allclose(fg,q['fold_gains'],atol=1e-12,rtol=0);assert sum(z>0 for z in fg)==q['positive_folds']
 a=report['comparisons']['mha_standalone_vs_mean'];b=report['comparisons']['mha_blend_vs_mean'];q=report['comparisons']['mha_blend_vs_best021'];gate=a['gain']>0 and b['gain']>0 and b['positive_folds']>=3 and b['top5_removed_gain']>0 and q['gain']>=.02 and q['positive_folds']>=3 and q['conditional_paired_groupbootstrap_CI95'][0]>0 and q['top5_removed_gain']>0;assert bool(gate)==report['numeric_gate_passed'];assert not c.network_attempts
result={'passed':True,'all10_models_reload_verified':len(records)==10,'checks':records,'max_MPS_reload_difference':max(q['MPS_reload_maxabs'] for q in records),'max_CPU_MPS_difference':max(q['CPU_reload_maxabs'] for q in records),'allfold_train_only_normalization_and_initialstate_order_seed_LR_updates_verified':True,'both_fold0_full20epoch_heldoutlabel_poison_replay_exact':True,'poison_replay_note':'Artificial validation labels used only for leakage audit; any replay RMSE is not a scientific score or selection input.','all_scalar_metrics_folds_top5_gate_verified':True,'max_independent_bootstrap_difference':max_boot,'network_attempts':len(c.network_attempts),'raw_image_holdout_test_access':False};c.check();c.write(R/'validation.json',result);c.emit(result)
