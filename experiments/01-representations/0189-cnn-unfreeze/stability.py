import common as c
from common import np,torch
import math
class Guard:
 def __init__(self,o,X,T,tail,out,mu,scale):
  z=(X[T]-o['feature_mean'].numpy())/o['feature_scale'].numpy();self.ref_rms=float(np.sqrt(np.mean(z*z)));self.ref_max=float(abs(z).max());self.mu=mu;self.scale=scale;self.out=out;self.phase='initial';self.features=[];self.gradients=[];self.first_grad=None;self.initial={str(i):c.statehash(tail[i]) for i in [0,1]}
 def stop(self,reason,record):
  def safe(x):
   if isinstance(x,float) and not math.isfinite(x):return str(x)
   if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
   if isinstance(x,list):return [safe(v) for v in x]
   return x
  c.write(self.out/'stability-STOP.json',safe({'reason':reason,'record':record,'feature_train_reference':[self.ref_rms,self.ref_max],'first_gradient_reference':self.first_grad,'feature_trace':self.features,'gradient_trace':self.gradients}));raise RuntimeError('189stabilitySTOP:'+reason)
 def feature(self,a):
  a=a.detach();z=(a-self.mu)/self.scale
  if not bool(torch.isfinite(a).all() and torch.isfinite(z).all()):self.stop('nonfinite_feature',{'phase':self.phase})
  rms=float(z.square().mean().sqrt());mx=float(z.abs().max());rec={'phase':self.phase,'rows':len(a),'raw_min':float(a.min()),'raw_max':float(a.max()),'Z_RMS':rms,'Z_maxabs':mx,'RMS_ratio':rms/max(1.,self.ref_rms),'maxabs_ratio':mx/max(1.,self.ref_max)};self.features.append(rec)
  if not math.isfinite(rms) or rec['RMS_ratio']>100 or rec['maxabs_ratio']>100:self.stop('feature_distribution_explosion',rec)
 def gradient(self,head_tail_norms,tail,epoch,batch):
  norms=[]
  for stage in [tail[0],tail[1]]:norms.append(float(torch.stack([p.grad.detach().norm()**2 for p in stage.parameters() if p.grad is not None]).sum().sqrt()))
  total=math.hypot(*head_tail_norms);rec={'epoch':epoch,'batch':batch,'head':head_tail_norms[0],'layer3':norms[0],'layer4':norms[1],'total':total}
  if not all(math.isfinite(x) and x>0 for x in [*head_tail_norms,*norms,total]):self.stop('nonfinite_or_zero_stage_gradient',rec)
  if self.first_grad is None:self.first_grad=total
  rec['ratio_to_first_train_update']=total/max(1.,self.first_grad);self.gradients.append(rec)
  if total>1e6 or rec['ratio_to_first_train_update']>100:self.stop('gradient_explosion',rec)
 def save(self,tail):
  final={str(i):c.statehash(tail[i]) for i in [0,1]};assert all(final[k]!=self.initial[k] for k in final)
  obj={'passed':True,'stage0_is_layer3_stage1_is_layer4':True,'both_stages_changed':True,'stage_initial':self.initial,'stage_final':final,'train_feature_reference_RMS':self.ref_rms,'train_feature_reference_maxabs':self.ref_max,'first_train_gradient_reference':self.first_grad,'max_gradient_ratio':max(x['ratio_to_first_train_update'] for x in self.gradients),'max_feature_RMS_ratio':max(x['RMS_ratio'] for x in self.features),'max_feature_maxabs_ratio':max(x['maxabs_ratio'] for x in self.features),'gradient_trace':self.gradients,'feature_trace':self.features,'limits':c.plan['stability']};c.write(self.out/'stability.json',obj)
