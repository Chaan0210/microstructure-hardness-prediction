"""Does the dev gain of a change depend on the training-set size? (dev folds train on 320 rows, the submitted models on 500.)
For FRAC in {0.5, 0.75, 1.0}: in every outer fold keep a random FRAC of the 320 training rows (5 draws), fit RidgeCV/logRidge partners without
and with degraded copies (copies only from sources inside the kept rows), insert them on noise >= 12 rows into the 053 blend, and record
the blend dMSE (aug - plain). usage: lc_partner.py SETS NOISEMIN"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, os, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
from pathlib import Path
SETS = sys.argv[1].split(','); NMIN = float(sys.argv[2])
P = Path(_repo_io('../..')); lab = {r.split(',')[0]: float(r.split(',')[1]) for r in open(_repo_path(P, 'data/train.csv')).read().split()[1:]}
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
B = load(3); sid = B['ids']; y0 = B['y']; nz = np.nan_to_num(np.load(_repo_io('../probabilistic-grains-900/base.npz'), allow_pickle=True)['noise'], nan=0); m12 = (nz >= 12)[None] & ~B['fail']
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
def feats(v):
    F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}
    SX, SY, SID = [], [], []
    for s in SETS:
        S = np.load(_repo_io(f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); X = S['X'].astype(float); k = X[:, keys.index('noise')] * 255 >= NMIN
        SX.append(X[k]); SY += [lab[str(i)] for i in np.array(S['ids'])[k]]; SID += [str(i) for i in np.array(S['ids'])[k]]
    return F['X'].astype(float), pos, np.vstack(SX), np.array(SY), np.array(SID)
FD = {v: feats(v) for v in V}
def fitpred(v, model, Xt, yt, Xv):
    if model == 'ridge': return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000])).fit(Xt, yt).predict(Xv)
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=2.0)).fit(Xt, np.log(yt)).predict(Xv)
def blend(r, l): c = 0.16 * B['p021'] + 0.384 * B['g'] + 0.128 * r + 0.128 * l + 0.05 * B['set'] + 0.15 * B['gmm']; return np.where(B['fail'], B['p021'], c)
gb = np.load(_repo_io('dev065.npz')); G = np.mean([gb[f'b{s}'] for s in ('38', '39', '44')], 0); B['g'] = G
for frac in (0.5, 0.75, 1.0):
    out = []
    for draw in range(5 if frac < 1 else 1):
        rng = np.random.default_rng(100 * draw + int(frac * 100)); RP = {k: np.zeros((3, 400)) for k in ('rp', 'ra', 'lp', 'la')}
        for p_, (name, path) in enumerate(splits.items()):
            d = np.load(_repo_io(path)); ids = [str(i) for i in d['ids']]; y = d['y']; fold = d['fold']; pp = {i: j for j, i in enumerate(ids)}; jj = [pp[i] for i in sid]
            o = {k: np.zeros(400) for k in ('rp', 'ra', 'lp', 'la')}; ol = {k: [] for k in ('lp', 'la')}
            for k in range(5):
                tr = np.flatnonzero(fold != k); va = fold == k; keep = rng.choice(tr, int(round(frac * len(tr))), replace=False); kid = set(np.array(ids)[keep])
                logs = {'lp': [], 'la': []}
                for v in V:
                    XA, pos, SX, SY, SID = FD[v]; X = XA[[pos[i] for i in ids]]; s = np.array([q in kid for q in SID])
                    Xt, yt = X[keep], y[keep]; Xa, ya = np.vstack([Xt, SX[s]]), np.r_[yt, SY[s]]
                    if v == 'u1dmavgH2Z': o['rp'][va] = fitpred(v, 'ridge', Xt, yt, X[va]); o['ra'][va] = fitpred(v, 'ridge', Xa, ya, X[va])
                    logs['lp'].append(fitpred(v, 'log', Xt, yt, X[va])); logs['la'].append(fitpred(v, 'log', Xa, ya, X[va]))
                o['lp'][va] = np.exp(np.mean(logs['lp'], 0)); o['la'][va] = np.exp(np.mean(logs['la'], 0))
            for kk in o: RP[kk][p_] = o[kk][jj]
        cp = blend(np.where(m12, RP['rp'], RP['rp']), RP['lp']); ca = blend(np.where(m12, RP['ra'], RP['rp']), np.where(m12, RP['la'], RP['lp']))
        out.append((((ca - y0) ** 2 - (cp - y0) ** 2).mean(0)))
    r = np.mean(out, 0); print(f'frac {frac:.2f} (train rows {int(round(frac * 320))}): blend dMSE aug-plain {r.mean():+.3f} (SE {r.std() / 20:.3f}) over {len(out)} draws', flush=True)
