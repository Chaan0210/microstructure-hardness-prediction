"""Group-wise TabPFN sub-models + stacker (nested within outer folds). Groups: phase, size, het, shape/angle, pore/quality (each + noise, n_obj as context).
Arms (1 seed each): direct TabPFN on all columns (reference), stack_ridge (RidgeCV on 5 group OOF preds), stack_pfn (TabPFN on group preds + noise).
usage: group_stack.py PARTITION"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, _repo_io('../push-300'))
import harness
from sklearn.linear_model import RidgeCV
part = sys.argv[1]
D = np.load(_repo_io('../push-600/dev-u1dmavgH2Z-angh.npz'), allow_pickle=True)
ids = [str(i) for i in D[part + '_ids']]; y = D[part + '_y']; fold = D[part + '_fold']; p021 = D[part + '_p021']; failed = D[part + '_failed']; g3 = D[part + '_g']
F = np.load(_repo_io('feat-u1dmavgH2Z-angh-train.npz'), allow_pickle=True); fp = {str(i): k for k, i in enumerate(F['ids'])}; keys = [str(k) for k in F['keys']]; X = F['X'][[fp[i] for i in ids]].astype(float)
def grp(k):
    if k.startswith(('pore', 'px_pore')) or k in ('noise', 'img_std', 'coverage'): return 'poreq'
    if k.startswith(('u_ph', 'm_ph', 'z_f', 'sp_', 'tile_spfrac', 'dark_span', 'obj_I', 'grain_I_std')): return 'phase'
    if k.startswith(('ar_', 'boundary_coh', 'orient', 'solidity', 'stcoh', 'm_el', 'vstd_ar', 'vstd_orient', 'vstd_solid', 'ang_')): return 'shape'
    if k.endswith('_x_m_el_pix') or k.startswith(('coarse_area', 'fine_area', 'grain_logA_std', 'logA_', 'sizefield', 'tile8_logA', 'tile_logA', 'm_la_std', 'm_la_f', 'm_la_tile', 'het4', 'vstd_sizefield', 'vstd_tile8', 'vstd_grain_logA')) or k == 'grain_eqd_q90_q10': return 'het'
    return 'size'
G = {}
for j, k in enumerate(keys): G.setdefault(grp(k), []).append(j)
ctx = [keys.index('noise'), keys.index('n_obj')]
cols = {g: sorted(set(v) | set(ctx)) for g, v in G.items()}
print({g: len(v) for g, v in cols.items()}, flush=True)
def pfn(seed): return harness.tabpfn_new('v3.5', seed, 'mps')
out = {a: np.zeros(len(y)) for a in ['direct1', 'stack_ridge', 'stack_pfn']}
for k in range(5):
    tr = np.where(fold != k)[0]; va = np.where(fold == k)[0]
    m = pfn(k); m.fit(X[tr], y[tr]); out['direct1'][va] = m.predict(X[va])
    fi = np.arange(len(tr)) % 5; np.random.RandomState(1000 + k).shuffle(fi)
    Ztr = np.zeros((len(tr), len(cols))); Zva = np.zeros((len(va), len(cols)))
    for gi, (gname, cc) in enumerate(cols.items()):
        for jj in range(5):
            m = pfn(k); m.fit(X[tr][fi != jj][:, cc], y[tr][fi != jj]); Ztr[fi == jj, gi] = m.predict(X[tr][fi == jj][:, cc])
        m = pfn(k); m.fit(X[tr][:, cc], y[tr]); Zva[:, gi] = m.predict(X[va][:, cc])
    rr = RidgeCV(alphas=np.logspace(-3, 3, 30)).fit(Ztr, y[tr]); out['stack_ridge'][va] = rr.predict(Zva)
    S_tr = np.c_[Ztr, X[tr][:, ctx]]; S_va = np.c_[Zva, X[va][:, ctx]]
    m = pfn(k); m.fit(S_tr, y[tr]); out['stack_pfn'][va] = m.predict(S_va)
    print('fold', k, ' '.join('%s %.3f' % (a, np.sqrt(np.mean((out[a][va] - y[va]) ** 2))) for a in out), 'ridge coefs', np.round(rr.coef_, 2), flush=True)
bl = lambda g: np.sqrt(np.mean((0.25 * p021 + 0.75 * np.where(failed, p021, g) - y) ** 2))
print('base(3 seeds, existing) grain %.4f W.75 %.4f' % (np.sqrt(np.mean((g3 - y) ** 2)), bl(g3)))
for a in out:
    print('%-12s grain %.4f W.75 %.4f | errcorr with base %.3f | mix 50/50 with base W.75 %.4f' % (a, np.sqrt(np.mean((out[a] - y) ** 2)), bl(out[a]), np.corrcoef(out[a] - y, g3 - y)[0, 1], bl(0.5 * out[a] + 0.5 * g3)))
np.savez(_repo_io(f'groupstack-{part}.npz'), ids=np.array(ids), **out)
