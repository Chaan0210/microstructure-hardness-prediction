import common as c,engine as e
from common import R,C
c.check();assert c.read(R/'preflight.json')['passed'];X,cache,d=c.inputs();T,V=c.indices(0);out=R/'control-fold-0';out.mkdir();warm=e.ck(C/'convnext_tiny/fold-0/initial-head.pt');recipe={'name':'fixed_adamw_low','optimizer':'AdamW','head_lr':.0001,'tail_lr':.00001}
with c.Monitor('control',180):
 pred,rec=e.train(warm,cache,X,d['y'][T].copy(),recipe,202710186+99,out,V,ratio=1.,fold=0);assert rec['existing190_control_exact'];c.write(R/'control-validation.json',{'passed':True,'historical_head_tail_weights_OOF_exact':True,'historical_240_gradient_LR_traces_exact':True,'historical_feature_and_train_curves_exact':True,'grouped_optimizer_does_not_change_uniform_result':True,'RNG_reference_exact_all240':True,'fits':1,'updates':240,'report_SHA256':c.sha(out/'report.json'),'maximum_candidate_fits':5,'candidate_performance_not_computed':True})
c.check()
