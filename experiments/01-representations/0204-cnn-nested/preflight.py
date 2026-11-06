import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as c
import pipeline as p
from common import np,R,O,PLAN,SPLITS
c.check();z,images=c.data();start=time.perf_counter();checks=[];output=np.full((3,400),np.nan)
assert (O/'review-disposition.json').exists() and c.J(O/'review-disposition.json')['allow_preflight']
old_effective={seed+n['offset'] for n in PLAN['nodes'] for seed in PLAN['old_seeds']};new_effective={seed+n['offset'] for n in PLAN['nodes'] for seed in PLAN['new_seeds']};assert old_effective.isdisjoint(new_effective)
with c.Monitor('preflight',1800):
 for si,split in enumerate(SPLITS):
  for k in range(5):
   folder=O/'R'/split/f'fold-{k}';pred,rr=p.fit_fold(z,si,k,'R',folder);reference_error=p.reference_check(folder,z,si,k);val=p.validate_fold(folder,z,si,k,'R');T,V=c.indices(z,si,k);labels=z['y'].copy();labels[V]=np.nan;other=O/'R-outer-poison'/split/f'fold-{k}';p.fit_fold(z,si,k,'R',other,labels=labels);assert p.compare_poison(folder,other);output[si,V]=pred;checks.append({'split':split,'fold':k,'reference_max_error':reference_error,'independent_validation':val,'outer_nan_poison_exact':True,'all3anchor_B_V_finite_and_nan_poison_exact':True});c.emit({'stage':'R_preflight','completed_folds':len(checks),'total':15,'reference_max_error':reference_error,'independent_error':val['max_independent_error']})
np.testing.assert_allclose(output,z['R'],atol=1e-9,rtol=0);c.arrays(O/'R-rebuilt.npz',ids=z['ids'],predictions=output);c.check();c.W(O/'preflight.json',{'passed':True,'checks':checks,'old_and_new_effective_seed_disjoint':True,'k_range':[0,4],'R_max_error':float(abs(output-z['R']).max()),'cached_old540_inner_predictions_reused_not_retrained':True,'source_provenance':c.J(O/'preparation.json'),'R_downstream_refitted':True,'outer_poison_folds':15,'neural_fits':0,'nonnull_residual_fits':1440,'affine_fits':120,'seconds':time.perf_counter()-start,'code_lock_sha256':c.H(O/'code-lock.json')});c.access_report('preflight-access.json');c.emit({'R_preflight_passed':True,'neural_fits':0,'seconds':time.perf_counter()-start})
