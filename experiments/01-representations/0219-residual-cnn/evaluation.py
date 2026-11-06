"""Post-fit evaluation only: all five frozen packages must precede scoring."""
import numpy as np
from contracts import require

def evaluate_all(y,R0,R1,Rperm,incumbent,fold,completed_folds):
 fold=np.asarray(fold) if fold is not None else None
 require(sorted(completed_folds)==list(range(5)),'All five folds must be completed before scoring')
 arrays=[np.asarray(a,dtype=float) for a in (y,R0,R1,Rperm,incumbent)];require(all(a.shape==(400,) and np.isfinite(a).all() for a in arrays),'Complete development400 predictions required');y,R0,R1,Rperm,incumbent=arrays;require(fold.shape==(400,) and set(fold)==set(range(5)),'Wrong fold coverage')
 arms={'paired_anchor':R0,'true_residual':R1,'permuted_residual':Rperm,'outer_safe_best021':incumbent};rmse=lambda z:float(np.mean(z*z)**.5);scores={k:rmse(y-v) for k,v in arms.items()};rng=np.random.default_rng(20264100);draw=rng.integers(0,400,(10000,400));comparisons={}
 for name,p in [('R0',R0),('Rperm',Rperm),('P021',incumbent)]:
  gain=np.sqrt(np.mean((y-p)[draw]**2,axis=1))-np.sqrt(np.mean((y-R1)[draw]**2,axis=1));comparisons[name]={'RMSE_gain':rmse(y-p)-rmse(y-R1),'conditional_paired_ID_bootstrap95':np.quantile(gain,[.025,.975]).tolist(),'fold_gains':[rmse((y-p)[fold==k])-rmse((y-R1)[fold==k]) for k in range(5)]}
 return {'scores':scores,'comparisons':comparisons,'mechanism_point_gate':scores['true_residual']<scores['paired_anchor'] and scores['true_residual']<scores['permuted_residual'],'incumbent_point_gate':scores['true_residual']<scores['outer_safe_best021'],'automatic_promotion':False,'limitations':'400 labels already reused adaptively. New partitions/seeds are not independent confirmation; conditional bootstrap does not remove model-selection bias. A-only anchor has fewer training samples than incumbent. Never calibrate residual mixture from these outcomes.'}
