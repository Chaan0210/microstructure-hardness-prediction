"""Frozen paired source metrics reused from the preceding completed study."""
import numpy as np
def rmse(p,y):return float(np.sqrt(np.mean((p-y)**2)))

def coefficient(delta,y,base,high):
    d=delta[high];r=(y-base)[high];return float(np.clip(d@r/(d@d+len(d)),0,.5))

def metrics(y,b,p,fold,seed=2026101781):
    a=(b-y)**2;c=(p-y)**2;draw=np.random.default_rng(seed).integers(0,len(y),(10000,len(y)));gg=np.sqrt(a[draw].mean(1))-np.sqrt(c[draw].mean(1));order=np.argsort(-(a-c));D=np.c_[np.ones(len(y)),b-b.mean()];cf=np.linalg.lstsq(D,p-b,rcond=None)[0];orth=p-b-D@cf
    return dict(n=len(y),reference_rmse=rmse(b,y),candidate_rmse=rmse(p,y),gain=rmse(b,y)-rmse(p,y),conditional_CI95=np.quantile(gg,[.025,.975]).tolist(),conditional_CI9833=np.quantile(gg,[.008333,.991667]).tolist(),fold_gains=[rmse(b[fold==k],y[fold==k])-rmse(p[fold==k],y[fold==k]) for k in range(5)],improved_folds=int(sum(rmse(p[fold==k],y[fold==k])<rmse(b[fold==k],y[fold==k])-1e-12 for k in range(5))),
        removed_largest_gains={str(n):dict(gain=rmse(b[order[n:]],y[order[n:]])-rmse(p[order[n:]],y[order[n:]]),share_net_SSE_gain=float((a-c)[order[:n]].sum()/(a-c).sum()) if abs((a-c).sum())>1e-12 else None) for n in [5,10]},
        affine_diagnostic=dict(intercept=float(cf[0]),slope_delta=float(cf[1]),affine_only_gain=rmse(b,y)-rmse(b+D@cf,y),orthogonal_only_gain=rmse(b,y)-rmse(b+orth,y),posthoc_not_a_new_candidate=True),n_helped=int((c<a-1e-10).sum()),n_harmed=int((c>a+1e-10).sum()))
