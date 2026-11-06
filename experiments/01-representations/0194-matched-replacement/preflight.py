import common as c
from common import R,np
c.check()
with c.Monitor('preflight'):
 d=c.data();p,checks=c.legacy_precision(d);f=c.frozen(d);q=f['best021']+f['coef']*(p['detach']-d['cnn']['seeds'][0]);np.testing.assert_array_equal(q,f['best021']);c.arrays(R/'legacy-precision.npz',ids=d['a']['ids'],fold=d['a']['fold'],**p);c.arrays(R/'frozen-components.npz',ids=d['a']['ids'],fold=d['a']['fold'],**f)
 r={'passed':True,'rows':400,'original_fold_count':5,'first_seed_identity':'20261003+fold','detach_originalCNN_legacy_float32_exact':True,'detach_replacement_best021_exact':True,'existing_021_fullmixture_reconstruction_exact':True,'all15_residual_blocks_weights_and_outputs_verified':True,'precision_checks':checks,'calibrators':c.plan['calibrators'],'fixed_coefficient_by_fold':c.plan['coefficient_by_fold'],'new_arm_performance_scored':False,'fits_or_model_inference':0,'legacy_precision_SHA256':c.sha(R/'legacy-precision.npz'),'frozen_components_SHA256':c.sha(R/'frozen-components.npz'),'plan_SHA256':c.sha(R/'plan.json')};c.write(R/'preflight.json',r);c.emit({'preflight_passed':True,'detach_exact400':True,'old021_reconstruction_exact':True,'coefficient_by_fold':r['fixed_coefficient_by_fold'],'new_scores_computed':False})
c.check()
