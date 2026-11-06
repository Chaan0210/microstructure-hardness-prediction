import common as c,engine as e
from common import R,P,np,torch
c.check();assert c.read(R/'baseline-validation.json')['passed'] and c.read(R/'baseline-exit.json')['exit_code']==0;records=[]
with c.Monitor('study',1500):
 for split in c.SPLITS:
  d=c.data(split);root=R/split;pred={'detach':c.load(root/'detach-OOF.npz')['prediction'],'true':np.full(400,np.nan),'permuted':np.full(400,np.nan)}
  for k in range(5):
   T,V=c.indices(d,k);reports=[c.read(root/f'fold-{k}-detach/report.json')]
   for arm in ['true','permuted']:
    folder=root/f'fold-{k}-{arm}';_,rr=e.fit(d,k,arm,folder);pred[arm][V]=c.canonical_predictions(d,k,folder);reports.append(rr);records.append({'split':split,'fold':k,'arm':arm,'seconds':rr['seconds']})
   for key in ['initial_state_SHA256','initial_base_SHA256','rng_start','rng_end','input_hashes']:assert reports[0][key]==reports[1][key]==reports[2][key],(split,k,key)
   c.emit({'paired_fold_complete':split,'fold':k,'true_permuted_fits_completed':len(records),'comparative_scores_deferred':True})
  assert all(np.isfinite(v).all() for v in pred.values());f=c.load(root/'frozen-components.npz');replacement={arm:d['best021']+f['coefficient']*(p-d['old_CNN']) for arm,p in pred.items()};np.testing.assert_array_equal(replacement['detach'],d['best021']);c.arrays(root/'OOF.npz',ids=d['ids'],y=d['y'],fold=d['fold'],groups=d['g'],best021=d['best021'],coefficient=f['coefficient'],**{'raw_'+a:p for a,p in pred.items()},**replacement)
assert len(records)==20;c.check();c.write(R/'study-complete.json',{'passed':True,'main_fits_including_detach':30,'new_true_permuted_fits':20,'records':records,'comparative_scores_not_yet_computed':True});c.emit({'all30_fits_complete':True,'validation_next':True})
