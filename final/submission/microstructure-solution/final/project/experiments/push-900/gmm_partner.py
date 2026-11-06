import sys, numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
H = Path(__file__).resolve().parent; E = H.parent; S9 = E / 'probabilistic-grains-900'
sys.path.insert(0, str(S9 / 'probabilistic/contrast-adaptive'))
import grain_measurement as gm
ALPHAS = gm.CONFIG['alpha_ridge']
def images(bank, ids):
    bid = [str(i) for i in bank['ids']]; obj = np.asarray(bank['objects'], float); off = np.asarray(bank['offsets'], int); gX = np.asarray(bank['global_X'], float)
    ok_ = {str(k): j for j, k in enumerate(bank['object_keys'])}; gk = {str(k): j for j, k in enumerate(bank['global_keys'])}; req = {'area', 'log_area', 'intensity_se2', 'shade_darkness', 'log_ar', 'cos2', 'sin2', 'pore_flag'}
    out = []
    for i in ids:
        row = bid.index(i); z = obj[off[row]:off[row + 1]]; g = gX[row]; get = lambda k: z[:, ok_[k]]; span = max(float(g[gk['contrast_span']]), 1.)
        ok = (get('area') > 0) & (get('pore_flag') < 0.5)
        for k in req: ok &= np.isfinite(get(k))
        z = z[ok]; empty = len(z) == 0 or ('whole_image_fallback' in ok_ and len(z) > 0 and np.any(z[:, ok_['whole_image_fallback']] > .5))
        if empty:
            d, v, a, basis, phys, sv = np.array([0.]), np.array([0.01]), np.array([1.]), np.array([[1., 0., 0., 0., 0.]]), np.zeros(10), 0.01
        else:
            get = lambda k: z[:, ok_[k]]; area = get('area'); a = area / area.sum(); la, ar = get('log_area'), get('log_ar'); c, s = get('cos2'), get('sin2')
            mla, mar = np.dot(a, la), np.dot(a, ar); basis = np.column_stack([np.ones(len(z)), la - mla, ar - mar, c, s])
            d = np.clip(get('shade_darkness'), -1.5, 2.5); v = np.clip(get('intensity_se2') / span ** 2, 1e-8, 1.)
            sv = float(np.clip(np.dot(a, np.clip(get('shade_se2'), 0, np.inf)) / span ** 2, 0., 0.09)) if 'shade_se2' in ok_ else 0.
            noise = max(float(g[gk['noise_gray']]), 0.) / span; cov = np.clip(float(g[gk['coverage']]), 0., 1.)
            phys = np.array([mla, np.sqrt(np.dot(a, (la - mla) ** 2)), np.dot(a, area ** -0.25), mar, np.dot(a, c), np.dot(a, s), np.log(1. / np.dot(a, a)), cov, noise, np.log(span)])
        out.append(gm.GrainImage(i, d, v, a, basis, phys, sv, empty))
    return out
def gmm_feats(bank, ids):
    m, v, _ = gm.transform(images(bank, ids), 'image_gmm'); return gm.physical_readout(m, v)
def ridge(a): return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=a))
def fit_predict(Xtr, ytr, Xte, seed):
    sc = []
    for a in ALPHAS:
        err = []
        for itr, iva in KFold(3, shuffle=True, random_state=seed).split(Xtr): err.extend((ridge(a).fit(Xtr[itr], ytr[itr]).predict(Xtr[iva]) - ytr[iva]) ** 2)
        sc.append(np.mean(err))
    a = ALPHAS[int(np.argmin(sc))]; return ridge(a).fit(Xtr, ytr).predict(Xte), a
