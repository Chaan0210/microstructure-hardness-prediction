"""H9 columns in the gated-row DTA partners (076 = arm B of srcw.py: alpha by source-grouped inner CV on real noise>=12 rows). Real rows get their cumulants
(cumul-train.npz), each copy gets the cumulants measured on the copy image itself (cumul-<set>.npz). Arms: B (= 076 partners, check) and B+h9 (and B+rand).
Study900 dev folds; deltas applied to the 076 dev blend on gated rows. Also holdout-100 (fit on dev-400 + copies of dev sources)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, sys, numpy as np, warnings
warnings.filterwarnings('ignore')
src = open(_repo_io('srcw.py')).read(); exec(compile(src[:src.index('D = {v: data(v)')], 'srcw', 'exec'))
import nsgain as N
from robust_eval import report
CT = np.load(_repo_io('cumul-train.npz')); ct = {str(i): j for j, i in enumerate(CT['ids'])}; CC = {s: np.load(_repo_io(f'cumul-{s}.npz')) for s in ('deg', 'deg2', 'deg3', 'deg4')}
RND = np.random.default_rng(2026101509).normal(size=(500, 2)); MODE = sys.argv[1] if len(sys.argv) > 1 else 'h9'
def data_h9(v):
    F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; ni = keys.index('noise'); ids = [str(i) for i in F['ids']]
    XA = F['X'].astype(float); ex = CT['X'][[ct[i] for i in ids]][:, [1, 4]] if MODE == 'h9' else RND[[ct[i] for i in ids]]
    SX, SID = [], []
    for s in ['deg', 'deg2', 'deg3', 'deg4']:
        S = np.load(_repo_io(H12 / f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); m = S['X'][:, ni].astype(float) * 255 >= 12; sids = [str(i) for i in np.array(S['ids'])]
        cc = {str(i): j for j, i in enumerate(CC[s]['ids'])}
        cex = np.array([CC[s]['X'][cc[i]][[1, 4]] for i in sids]) if MODE == 'h9' else np.array([RND[ct[i]] for i in sids])
        SX.append(np.c_[S['X'].astype(float), cex][m]); SID += [i for i, mm in zip(sids, m) if mm]
    return {i: k for k, i in enumerate(ids)}, np.c_[XA, ex], np.nan_to_num(XA[:, ni] * 255, nan=0), np.vstack(SX), np.array(SID)
D = {v: data_h9(v) for v in V}; out = {}; ho = {}
for kind, vs in (('ridge', ['u1dmavgH2Z']), ('log', V)):
    O = []; H = []
    for v in vs:
        pos, XA, NZ, SX, SID = D[v]; o = np.zeros((3, 400))
        for p_, path in enumerate(splits):
            d = np.load(_repo_io(path)); ids, fold = [str(i) for i in d['ids']], d['fold']
            for k in range(5):
                tr = [i for i, f in zip(ids, fold) if f != k]; va = [i for i, f in zip(ids, fold) if f == k]; vset = set(va); kp = np.array([s not in vset for s in SID])
                jt = np.array([pos[i] for i in tr]); m, a = fit_arm('B', kind, XA[jt], np.array([lab[i] for i in tr]), np.array(tr), NZ[jt], SX[kp], np.array([lab[s] for s in SID[kp]]), SID[kp])
                o[p_, [sid.index(i) for i in va]] = itf(kind, m.predict(XA[[pos[i] for i in va]]))
        O.append(o)
        ids_all = list(pos); trd = [i for i in ids_all if i in dev]; hd = [i for i in ids_all if i not in dev]; kp = np.array([s in dev for s in SID]); jt = np.array([pos[i] for i in trd])
        m, a = fit_arm('B', kind, XA[jt], np.array([lab[i] for i in trd]), np.array(trd), NZ[jt], SX[kp], np.array([lab[s] for s in SID[kp]]), SID[kp]); H.append(itf(kind, m.predict(XA[[pos[i] for i in hd]]))); ho['hid'] = np.array(hd)
    key = 'ridge' if kind == 'ridge' else 'logridge'; out[key] = O[0] if kind == 'ridge' else np.exp(np.mean(np.log(O), 0)); ho[key] = H[0] if kind == 'ridge' else np.exp(np.mean(np.log(H), 0)); print(key, flush=True)
np.savez(_repo_io(f'h9dta-{MODE}.npz'), **out); np.savez(_repo_io(f'h9dta-{MODE}-ho.npz'), **ho)
Z = np.load(_repo_io('srcw.npz')); Zh = np.load(_repo_io('srcw-ho.npz')); y, g = N.y, N.g; base = np.load(_repo_io('dev-combo.npy'))
rb = lambda a: np.sqrt(((a - y) ** 2)[:, g].mean())
print(f'gated partner RMSE: ridge {rb(Z["B_ridge"]):.3f} -> {rb(out["ridge"]):.3f}, logridge {rb(Z["B_logridge"]):.3f} -> {rb(out["logridge"]):.3f}')
report(f'076 + {MODE} cols in gated DTA partners', np.where(g, base + 0.128 * (out['ridge'] - Z['B_ridge']) + 0.128 * (out['logridge'] - Z['B_logridge']), base), base=base)
lab2 = lab; hid = [str(i) for i in ho['hid']]; yh = np.array([lab2[i] for i in hid]); assert hid == [str(i) for i in Zh['hid']]
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); k = [str(x) for x in F['keys']]; pp = {str(i): j for j, i in enumerate(F['ids'])}
gh = np.nan_to_num(F['X'][[pp[i] for i in hid], k.index('noise')].astype(float) * 255, nan=0) >= 12
pr, bs = (ho['ridge'] + ho['logridge']) / 2, (Zh['B_ridge'] + Zh['B_logridge']) / 2; dd = (pr - yh)[gh] ** 2 - (bs - yh)[gh] ** 2
print(f'holdout gated (n={gh.sum()}): partner-mean dMSE {dd.mean():+.2f} ± {dd.std() / np.sqrt(gh.sum()):.2f}')
