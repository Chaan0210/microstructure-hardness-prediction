"""Label-free noise-shift correction of grain features: the microstructure is independent of image noise, so any trend of a feature's
location/scale with the noise level is measurement bias. Fit per-feature smooth trends on TRAIN features only and map every row to the
clean-reference level (noise = train 10th percentile).
mode 'loc'  : f - m(noise) + m(ref)
mode 'locscale': (f - m(noise)) / s(noise) * s(ref) + m(ref)
m, s = quadratic fits in noise (robust: fit on rows with finite values; s from |residual|)."""
import numpy as np

def fit(Xtr, noise_tr, skip):
    ref = np.nanquantile(noise_tr, 0.1); P = {}
    for j in range(Xtr.shape[1]):
        if j in skip: continue
        ok = np.isfinite(Xtr[:, j]) & np.isfinite(noise_tr)
        if ok.sum() < 50 or np.nanstd(Xtr[ok, j]) == 0: continue
        cm = np.polyfit(noise_tr[ok], Xtr[ok, j], 2)
        res = np.abs(Xtr[ok, j] - np.polyval(cm, noise_tr[ok]))
        cs = np.polyfit(noise_tr[ok], res, 2)
        P[j] = (cm, cs)
    return ref, P

def apply(X, noise, model, mode='loc'):
    ref, P = model; X = X.copy(); noise = np.where(np.isfinite(noise), noise, ref)
    for j, (cm, cs) in P.items():
        m, m0 = np.polyval(cm, noise), np.polyval(cm, ref)
        if mode == 'loc':
            X[:, j] = X[:, j] - m + m0
        else:
            s, s0 = np.maximum(np.polyval(cs, noise), 1e-9), max(np.polyval(cs, ref), 1e-9)
            X[:, j] = (X[:, j] - m) / s * s0 + m0
    return X
