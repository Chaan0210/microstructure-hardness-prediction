from common import np
def evaluate(d,pred):
 y=d['y'];ref=d['blend_L_MLP'];fold=d['fold'];rmse=lambda y,p:float(np.sqrt(np.mean((y-p)**2)));blend=.9*ref+.1*pred;fg=[rmse(y[fold==k],ref[fold==k])-rmse(y[fold==k],blend[fold==k]) for k in range(5)];unique,inv=np.unique(d['g'],return_inverse=True);rng=np.random.default_rng(202710185);boot=[]
 for _ in range(3000):
  counts=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[inv];boot.append(float(np.sqrt(counts@((y-ref)**2)/counts.sum())-np.sqrt(counts@((y-blend)**2)/counts.sum())))
 ci=np.quantile(boot,[.025,.975]).tolist();keep=np.argsort((y-ref)**2-(y-blend)**2)[:-5];gain=rmse(y,ref)-rmse(y,blend);dropgain=rmse(y[keep],ref[keep])-rmse(y[keep],blend[keep]);gate=gain>=.02 and sum(v>0 for v in fg)>=3 and ci[0]>0 and dropgain>0
 return {'RMSE':{'best021':rmse(y,ref),'standalone':rmse(y,pred),'blend10':rmse(y,blend)},'blend_gain':gain,'fold_blend_gains':fg,'positive_folds':sum(v>0 for v in fg),'conditional_group_bootstrap_CI95':ci,'gain_after_removing_top5':dropgain,'residual_correlation_vs021':float(np.corrcoef(y-pred,y-ref)[0,1]),'advance_gate':bool(gate),'uncertainty_note':'Conditional reused400; notselectionadjusted orindependenttest.'}
