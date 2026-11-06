"""Can a local learner (TabPFN, 044 slot = anghcor 98 cols) use degraded copies once the copy->real-noisy feature gap is reduced? (train only)
Per outer fold: real training rows + copies (sets, measured noise >= 12, source not in the validation fold), copy handling OPTION:
 none (no copies) | raw | top50 (copies most 'real-looking' by a domain classifier fitted on the training fold's real noisy rows vs copies)
 | mm (per-feature mean/sd of copies matched to the training fold's real noisy rows; noise column untouched).
Outputs dev g for all rows (Study900 3 partitions); usage: tabsim_dw.py VARIANT OPTION SETS DEVICE -> tabsim-<VARIANT>-<OPTION>.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, warnings
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, _repo_io('../push-300'))
import harness
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
P = Path(_repo_io('../..')); v, opt, sets, device = sys.argv[1], sys.argv[2], sys.argv[3].split(','), sys.argv[4]
F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; pos = {str(i): k for k, i in enumerate(F['ids'])}; XA = F['X'].astype(float); ni = keys.index('noise')
lab = {r.split(',')[0]: float(r.split(',')[1]) for r in open(_repo_path(P, 'data/train.csv')).read().split()[1:]}
SX, SY, SID = [], [], []
for s in sets:
    S = np.load(_repo_io(f'synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); X = S['X'].astype(float); k = X[:, ni] * 255 >= 12
    SX.append(X[k]); SY += [lab[str(i)] for i in np.array(S['ids'])[k]]; SID += [str(i) for i in np.array(S['ids'])[k]]
SX = np.vstack(SX); SY = np.array(SY); SID = np.array(SID)
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
save = {}
for name, path in splits.items():
    d = np.load(_repo_io(path)); ids, y, fold = [str(i) for i in d['ids']], d['y'], d['fold']; X = XA[[pos[i] for i in ids]]; g = np.zeros(len(y)); nz = np.nan_to_num(X[:, ni] * 255, nan=0)
    for k in range(5):
        tr, va = fold != k, fold == k; vids = set(np.array(ids)[va]); keep = np.array([s not in vids for s in SID]); C, CY = SX[keep], SY[keep]
        Rn = X[tr & (nz >= 12)]
        if opt in ('none',) or opt.startswith('noisyonly'): C, CY = C[:0], CY[:0]
        elif opt.startswith('dose'):   # dose<r>: random copies, at most r x (number of real training rows), one copy per source first
            rr = np.random.default_rng(31 * k + len(name)); nmax = int(float(opt[4:]) * tr.sum()); o_ = rr.permutation(len(C)); seen, pick = set(), []
            for j in o_:
                s_ = SID[keep][j]
                if s_ not in seen: seen.add(s_); pick.append(j)
            pick = (pick + [j for j in o_ if j not in set(pick)])[:nmax]; C, CY = C[pick], CY[pick]
        elif opt == 'top50':
            dc = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight='balanced'))
            dc.fit(np.vstack([Rn, C]), np.r_[np.ones(len(Rn)), np.zeros(len(C))]); pr = dc.predict_proba(C)[:, 1]; sel = pr >= np.median(pr); C, CY = C[sel], CY[sel]
        elif opt == 'mm':
            mu_c, sd_c, mu_r, sd_r = np.nanmean(C, 0), np.nanstd(C, 0), np.nanmean(Rn, 0), np.nanstd(Rn, 0)
            ok = np.isfinite(mu_c) & np.isfinite(mu_r) & (sd_c > 0) & (sd_r > 0); ok[ni] = False
            C = C.copy(); C[:, ok] = (C[:, ok] - mu_c[ok]) / sd_c[ok] * sd_r[ok] + mu_r[ok]
        if opt.startswith('noisyonly'):   # separate model for noisy validation rows: context = real noisy training rows (+ copies unless '-none'); other rows: plain model
            m = harness.tabpfn_new('v3.5', k, device); m.fit(X[tr], y[tr]); g[va] = m.predict(X[va]); vn = va & (nz >= 12); trn = tr & (nz >= 12)
            Cn, CYn = (SX[keep], SY[keep]) if opt == 'noisyonly' else (SX[:0], SY[:0])
            m2 = harness.tabpfn_new('v3.5', k, device); m2.fit(np.vstack([X[trn], Cn]), np.r_[y[trn], CYn]); g[vn] = m2.predict(X[vn]); continue
        m = harness.tabpfn_new('v3.5', k, device); m.fit(np.vstack([X[tr], C]), np.r_[y[tr], CY]); g[va] = m.predict(X[va])
    noisy = nz >= 12
    print(name, opt, 'all %.4f noisy %.4f clean %.4f' % (np.sqrt(((g - y) ** 2).mean()), np.sqrt(((g - y)[noisy] ** 2).mean()), np.sqrt(((g - y)[nz < 9.5] ** 2).mean())), flush=True)
    save.update({f'{name}_ids': np.array(ids), f'{name}_y': y, f'{name}_g': g})
np.savez(_repo_io(f'tabsim-{v}-{opt}.npz'), **save)
