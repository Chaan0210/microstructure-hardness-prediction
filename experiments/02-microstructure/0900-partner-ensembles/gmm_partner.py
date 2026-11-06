"""Finalize Study900's image_gmm__x98_augmented partner (Ridge on the 98 production columns + per-image two-component GMM phase measurements from the
zero-shot grain objects; label-free per image). mode dev: re-run the nested Study900 protocol and compare with its stored OOF; mode final: fit on the
500 train rows (alpha by 3-fold inner CV on the same grid) and predict the 1000 test rows -> gmm-partner-test.npz.
usage: gmm_partner.py dev|final"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; S9 = _repo_path(E, 'probabilistic-grains-900')
sys.path.insert(0, str(_repo_path(S9, 'probabilistic/contrast-adaptive')))
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
mode = sys.argv[1]; Btr = np.load(_repo_path(S9, 'object-bank-train.npz'), allow_pickle=True)
if mode == 'dev':
    base = np.load(_repo_path(S9, 'base.npz'), allow_pickle=True); ids = [str(i) for i in base['ids']]; y = base['y']; X = np.c_[base['X98'], gmm_feats(Btr, ids)]
    ref = np.load(_repo_path(S9, 'probabilistic/contrast-adaptive/stage2/predictions.npz'), allow_pickle=True); rid = [str(i) for i in ref['ids']]; R = ref['image_gmm__x98_augmented'][:, [rid.index(i) for i in ids]]
    P = np.zeros((3, 400))
    for s in range(3):
        for f in range(5):
            tr, va = base['fold'][s] != f, base['fold'][s] == f; P[s, va], _ = fit_predict(X[tr], y[tr], X[va], 19001 + 101 * s + f)
    print('our OOF RMSE', np.sqrt(((P - y) ** 2).mean()).round(4), ' Study900 OOF RMSE', np.sqrt(((R - y) ** 2).mean()).round(4), ' max |diff|', np.abs(P - R).max().round(4), ' corr', np.corrcoef(P.ravel(), R.ravel())[0, 1].round(5))
    np.savez(_repo_path(H, 'gmm-partner-dev.npz'), ids=np.array(ids), oof=P)
else:
    import csv
    P0 = _RepoPath(__file__).resolve().parents[3]; lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io(P0 / 'data/train.csv')))}
    Ftr = np.load(_repo_path(E, 'push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); Fte = np.load(_repo_path(E, 'push-800/feat-u1dmavgH2Z-anghcor-test.npz'), allow_pickle=True)
    tid = [str(i) for i in Ftr['ids']]; eid = [str(i) for i in Fte['ids']]; Bte = np.load(_repo_path(S9, 'object-bank-test.npz'), allow_pickle=True)
    base = np.load(_repo_path(S9, 'base.npz'), allow_pickle=True); bpos = [tid.index(str(i)) for i in base['ids']]; assert np.allclose(Ftr['X'][bpos].astype(float), base['X98'], equal_nan=True)
    Xtr = np.c_[Ftr['X'].astype(float), gmm_feats(Btr, tid)]; Xte = np.c_[Fte['X'].astype(float), gmm_feats(Bte, eid)]; ytr = np.array([lab[i] for i in tid])
    pred, a = fit_predict(Xtr, ytr, Xte, 19001); print('alpha', a, 'test pred range', pred.min().round(1), pred.max().round(1))
    np.savez(_repo_path(H, 'gmm-partner-test.npz'), ids=np.array(eid), pred=pred, alpha=a)
