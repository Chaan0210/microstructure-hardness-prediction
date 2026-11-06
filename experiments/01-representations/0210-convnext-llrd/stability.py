import common as c
from common import np,torch
import math
class Guard:
 def __init__(self,o,X,T,tail,out,mu,scale,train_tail):
  z=(X[T]-o['feature_mean'].numpy())/o['feature_scale'].numpy();self.ref_rms=float(np.sqrt(np.mean(z*z)));self.ref_max=float(abs(z).max());self.mu=mu;self.scale=scale;self.out=out;self.train_tail=train_tail;self.phase='initial';self.features=[];self.gradients=[];self.first_grad=None;self.initial=c.statehash(tail)
 def stop(self,reason,record):
  def safe(x):
   if isinstance(x,float) and not math.isfinite(x):return str(x)
   if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
   if isinstance(x,list):return [safe(v) for v in x]
   return x
  c.write(self.out/'stability-STOP.json',safe({'reason':reason,'record':record,'feature_reference':[self.ref_rms,self.ref_max],'first_gradient_reference':self.first_grad,'feature_trace':self.features,'gradient_trace':self.gradients}));raise RuntimeError('210stabilitySTOP:'+reason)
 def feature(self,a):
  a=a.detach();z=(a-self.mu)/self.scale
  if not bool(torch.isfinite(a).all() and torch.isfinite(z).all()):self.stop('nonfinite_feature',{'phase':self.phase})
  rms=float(z.square().mean().sqrt());mx=float(z.abs().max());q={'phase':self.phase,'rows':len(a),'raw_min':float(a.min()),'raw_max':float(a.max()),'Z_RMS':rms,'Z_maxabs':mx,'RMS_ratio':rms/max(1.,self.ref_rms),'maxabs_ratio':mx/max(1.,self.ref_max)};self.features.append(q)
  if not math.isfinite(rms) or q['RMS_ratio']>100 or q['maxabs_ratio']>100:self.stop('feature_distribution_explosion',q)
 def gradient(self,norms,epoch,batch):
  total=math.hypot(*norms);q={'epoch':epoch,'batch':batch,'head':norms[0],'tail':norms[1],'total':total}
  if not all(math.isfinite(x) for x in [*norms,total]) or norms[0]<=0 or (self.train_tail and norms[1]<=0) or (not self.train_tail and norms[1]!=0):self.stop('invalid_gradient',q)
  if self.first_grad is None:self.first_grad=total
  q['ratio_to_first_train_update']=total/max(1.,self.first_grad);self.gradients.append(q)
  if total>1e6 or q['ratio_to_first_train_update']>100:self.stop('gradient_explosion',q)
 def save(self,tail):
  final=c.statehash(tail);assert (final!=self.initial)==self.train_tail
  c.write(self.out/'stability.json',{'passed':True,'tail_trainable':self.train_tail,'tail_changed':final!=self.initial,'tail_initial':self.initial,'tail_final':final,'train_feature_reference_RMS':self.ref_rms,'train_feature_reference_maxabs':self.ref_max,'first_train_gradient_reference':self.first_grad,'max_gradient_ratio':max(x['ratio_to_first_train_update'] for x in self.gradients),'max_feature_RMS_ratio':max(x['RMS_ratio'] for x in self.features),'max_feature_maxabs_ratio':max(x['maxabs_ratio'] for x in self.features),'gradient_trace':self.gradients,'feature_trace':self.features,'limits':c.plan['stability']})
