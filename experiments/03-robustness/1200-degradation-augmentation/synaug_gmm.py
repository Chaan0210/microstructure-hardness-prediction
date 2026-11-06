"""gmm partner (Study900 image_gmm__x98_augmented: Ridge on X98 + per-image GMM phase measurements of zero-shot grain objects, alpha by inner
3-fold CV) with label-preserving degraded copies of clean train images added to each outer-training fold (sources in the val fold excluded).
mode dev -> sg-<tag>.npz (oof [3, 400]); mode final -> sgf-<tag>.npz (test predictions, 500 rows + sims). usage: synaug_gmm.py dev|final SETS(comma|none) TAG"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; S9 = _repo_path(E, 'probabilistic-grains-900')
sys.path.insert(0, str(_repo_path(E, 'push-900')))
import importlib.util
spec = importlib.util.spec_from_file_location('gp', _repo_path(E, 'push-900/gmm_partner.py'))
src = open(_repo_path(E, 'push-900/gmm_partner.py')).read(); src = src[:src.index("mode = sys.argv[1]")]
gp = {'__file__': str(_repo_path(E, 'push-900/gmm_partner.py')), '__name__': 'gp'}; exec(compile(src, 'gmm_partner_defs', 'exec'), gp)
mode, sets, tag = sys.argv[1], ([] if sys.argv[2] == 'none' else sys.argv[2].split(',')), sys.argv[3]
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(E.parent, 'data/train.csv')))}
SX, SY, SID = [], [], []
for s in sets:
    bank = np.load(_repo_path(H, f'objbank-{s}.npz'), allow_pickle=True); ids = [str(i) for i in bank['ids']]
    R = np.load(_repo_path(H, f'synthrows-{s}-u1dmavgH2Z-anghcor.npz'), allow_pickle=True); assert [str(i) for i in R['ids']] == ids
    SX.append(np.c_[R['X'].astype(float), gp['gmm_feats'](bank, ids)]); SY += [lab[i] for i in ids]; SID += ids
SX = np.vstack(SX) if SX else None; SY = np.array(SY); SID = np.array(SID)
NMIN = float(__import__('os').environ.get('NOISEMIN', '-1'))
if NMIN > 0 and SX is not None:
    _k = [str(x) for x in np.load(_repo_path(H, f'synthrows-{sets[0]}-u1dmavgH2Z-anghcor.npz'), allow_pickle=True)['keys']]; _m = SX[:, _k.index('noise')] * 255 >= NMIN
    SX, SY, SID = SX[_m], SY[_m], SID[_m]; print('kept sim rows', int(_m.sum()))
Btr = np.load(_repo_path(S9, 'object-bank-train.npz'), allow_pickle=True)
if mode == 'dev':
    base = np.load(_repo_path(S9, 'base.npz'), allow_pickle=True); ids = [str(i) for i in base['ids']]; y = base['y']; X = np.c_[base['X98'], gp['gmm_feats'](Btr, ids)]
    P = np.zeros((3, 400))
    for s in range(3):
        for f in range(5):
            tr, va = base['fold'][s] != f, base['fold'][s] == f; Xt, yt = X[tr], y[tr]
            if SX is not None:
                vids = set(np.array(ids)[va]); keep = np.array([i not in vids for i in SID]); Xt = np.vstack([Xt, SX[keep]]); yt = np.r_[yt, SY[keep]]
            P[s, va], _ = gp['fit_predict'](Xt, yt, X[va], 19001 + 101 * s + f)
    np.savez(_repo_path(H, f'sg-{tag}.npz'), ids=np.array(ids), oof=P); print('dev saved', tag, 'rmse', np.sqrt(((P - y) ** 2).mean()).round(4))
else:
    Ftr = np.load(_repo_path(E, 'push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); Fte = np.load(_repo_path(E, 'push-800/feat-u1dmavgH2Z-anghcor-test.npz'), allow_pickle=True)
    tid = [str(i) for i in Ftr['ids']]; eid = [str(i) for i in Fte['ids']]; Bte = np.load(_repo_path(S9, 'object-bank-test.npz'), allow_pickle=True)
    Xt = np.c_[Ftr['X'].astype(float), gp['gmm_feats'](Btr, tid)]; yt = np.array([lab[i] for i in tid]); Xe = np.c_[Fte['X'].astype(float), gp['gmm_feats'](Bte, eid)]
    if SX is not None: Xt = np.vstack([Xt, SX]); yt = np.r_[yt, SY]
    pred, a = gp['fit_predict'](Xt, yt, Xe, 19001); np.savez(_repo_path(H, f'sgf-{tag}.npz'), ids=np.array(eid), pred=pred, alpha=a); print('final saved', tag, 'alpha', a)
