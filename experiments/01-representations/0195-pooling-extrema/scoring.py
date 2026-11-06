import common as c
from common import np

def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
def evaluate(d,pred):
 y=d['y'];fold=d['fold'];ps={'best021':d['best021'],**pred};ps.update({arm+'_blend':.9*d['best021']+.1*pred[arm] for arm in c.ARMS});unique,inv=np.unique(d['groups'],return_inverse=True);draws=np.random.default_rng(202710195).integers(len(unique),size=(3000,len(unique)))
 counts=np.stack([np.bincount(ix,minlength=len(unique))[inv] for ix in draws]);den=counts.sum(1);scores={name:{'RMSE':rmse(y,p),'fold_RMSE':[rmse(y[fold==k],p[fold==k]) for k in range(5)]} for name,p in ps.items()};pairs={'baseline_blend_vs_best021':('best021','baseline_blend')}
 for arm in ['plus_max','plus_min']:
  pairs[arm+'_standalone_vs_baseline']=('baseline',arm);pairs[arm+'_blend_vs_baseline']=('baseline_blend',arm+'_blend');pairs[arm+'_blend_vs_best021']=('best021',arm+'_blend')
 reports={};boot={}
 for name,(reference,candidate) in pairs.items():
  p,q=ps[reference],ps[candidate];a=(y-p)**2;b=(y-q)**2;bb=np.sqrt((counts@a)/den)-np.sqrt((counts@b)/den);boot[name]=bb;benefit=a-b;order=np.lexsort((d['ids'],-benefit));drop=order[:5];keep=order[5:];fg=[scores[reference]['fold_RMSE'][k]-scores[candidate]['fold_RMSE'][k] for k in range(5)]
  reports[name]={'reference':reference,'candidate':candidate,'gain':scores[reference]['RMSE']-scores[candidate]['RMSE'],'fold_gains':fg,'positive_folds':sum(x>0 for x in fg),'conditional_paired_groupbootstrap_CI95':np.quantile(bb,[.025,.975]).tolist(),'conditional_CI97_5_sensitivity':np.quantile(bb,[.0125,.9875]).tolist(),'top5_removed_gain':rmse(y[keep],p[keep])-rmse(y[keep],q[keep]),'top5_removed_IDs':d['ids'][drop].tolist(),'remaining395_reference_RMSE':rmse(y[keep],p[keep]),'remaining395_candidate_RMSE':rmse(y[keep],q[keep]),'residual_correlation':float(np.corrcoef(y-p,y-q)[0,1])}
 gates={}
 for arm in ['plus_max','plus_min']:
  a=reports[arm+'_standalone_vs_baseline'];b=reports[arm+'_blend_vs_baseline'];q=reports[arm+'_blend_vs_best021'];checks={'paired_standalone_gain_positive':a['gain']>0,'paired_blend_gain_positive':b['gain']>0,'paired_blend_at_least3folds':b['positive_folds']>=3,'paired_blend_top5removed_positive':b['top5_removed_gain']>0,'reference_gain_at_least_0_02':q['gain']>=.02,'reference_at_least3folds':q['positive_folds']>=3,'reference_CI95_lower_positive':q['conditional_paired_groupbootstrap_CI95'][0]>0,'reference_top5removed_positive':q['top5_removed_gain']>0};gates[arm]={'criteria':checks,'failed_criteria':[name for name,ok in checks.items() if not ok],'numeric_gate_passed':all(checks.values())}
 return {'scores':scores,'comparisons':reports,'candidate_gates':gates,'numeric_gate_passed':any(q['numeric_gate_passed'] for q in gates.values())},draws,boot
