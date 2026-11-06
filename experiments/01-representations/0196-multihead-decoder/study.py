import common as c,engine as e
from common import R,np
from scoring import evaluate
c.check();assert c.read(R/'preflight.json')['passed'] and c.read(R/'preflight-exit.json')['exit_code']==0;d=c.data();mom=c.moments(d);ps={a:np.full(400,np.nan) for a in c.ARMS};records=[]
with c.Monitor('study',1200) as monitor:
 for k in range(5):
  folder=R/f'fold-{k}';folder.mkdir();pair={};T,V=c.indices(d,k)
  for arm in c.ARMS:
   p,q=e.fit(d,mom,k,arm,folder/arm);assert q['updates']==400;ps[arm][V]=p;records.append(q);pair[arm]=q;monitor.sample()
  for key in ['initial_state_SHA256','parameters','updates','first_input_hashes','order_SHA256','dropseed_SHA256','LR_SHA256']:assert pair['mean'][key]==pair['mha'][key],key
 assert all(np.isfinite(p).all() for p in ps.values());report,draws,boot=evaluate(d,ps);c.arrays(R/'OOF.npz',ids=d['ids'],y=d['y'],fold=d['fold'],groups=d['groups'],best021=d['best021'],**ps);c.arrays(R/'bootstrap.npz',draws=draws,**boot);report.update({'fold_records':records,'estimand':c.plan['evaluation'],'OOF_SHA256':c.sha(R/'OOF.npz'),'bootstrap_SHA256':c.sha(R/'bootstrap.npz')});c.write(R/'report.json',report);c.emit({n:report[n] for n in ['scores','comparisons','failed_criteria','numeric_gate_passed']})
c.check()
