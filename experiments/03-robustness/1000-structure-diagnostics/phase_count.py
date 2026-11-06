"""Generator hypothesis H1: the second phase is assigned per grain with probability p and the label uses p -> the number fraction of phase
grains estimates p better than the area fraction. Clean train rows, zero-shot object bank: grains classified dark/bright by a 2-component
GMM on grain mean intensity (area-weighted), pores excluded. Compares corr with y and with the 054 OOF residual (clean dev rows)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, sys, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, _repo_io('../push-900'))
from sklearn.mixture import GaussianMixture
from blend_dev import load
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}
P = np.load(_repo_io('/private/tmp/claude-501/-Users-dongchan-Desktop-PythonWorkspace-microstructure-2026/f62e347b-cb1a-4c17-8d54-2bf505be6706/scratchpad/presets.npz')); nzm = dict(zip(P['tr'], P['ntr']))
Bk = np.load(_repo_io('../probabilistic-grains-900/object-bank-train.npz'), allow_pickle=True); ok_ = [str(k) for k in Bk['object_keys']]; bid = [str(i) for i in Bk['ids']]; off = Bk['offsets']; O = Bk['objects']
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; fpos = {str(i): k for k, i in enumerate(F['ids'])}
rows = {}
for i in bid:
    if nzm[i] >= 9.5: continue
    r = bid.index(i); z = O[off[r]:off[r + 1]]; a = z[:, ok_.index('area')]; I = z[:, ok_.index('intensity_mean')]; bd = z[:, ok_.index('border')]
    m = (a >= 12) & (z[:, ok_.index('pore_flag')] < .5) & np.isfinite(I)
    rep = np.repeat(I[m], np.clip((a[m] / 20).astype(int), 1, 50))[:, None]
    g = GaussianMixture(2, random_state=0, n_init=3).fit(rep); dk = int(np.argmin(g.means_[:, 0])); post = g.predict_proba(I[m][:, None])[:, dk]; hard = post > 0.5
    inner = bd[m] < 0.5
    rows[i] = dict(area_frac=float((a[m] * post).sum() / a[m].sum()), count_frac=float(post.mean()), count_frac_hard=float(hard.mean()),
                   count_frac_inner=float(post[inner].mean()) if inner.sum() > 5 else np.nan, n=int(m.sum()))
ids = list(rows); y = np.array([lab[i] for i in ids]); V = {k: np.array([rows[i][k] for i in ids]) for k in rows[ids[0]]}
for k in ('u_ph', 'z_f25', 'sh_cor15'): V[k] = np.array([F['X'][fpos[i], keys.index(k)] for i in ids]).astype(float)
print('clean train rows', len(ids))
for k, v in V.items():
    if k == 'n': continue
    ok = np.isfinite(v); print(f'{k:16s} corr(y) {np.corrcoef(v[ok], y[ok])[0, 1]:+.3f}')
# residual test on clean dev rows (054 OOF)
B = load(9); res = (B['c054'] - B['y']).mean(0); dpos = {i: k for k, i in enumerate(B['ids'])}
dv = [i for i in ids if i in dpos]; rr = np.array([res[dpos[i]] for i in dv])
for k in ('area_frac', 'count_frac', 'count_frac_hard', 'count_frac_inner'):
    v = np.array([rows[i][k] for i in dv]); ok = np.isfinite(v); print(f'054 residual corr with {k:16s} {np.corrcoef(v[ok], rr[ok])[0, 1]:+.3f}   (n={ok.sum()})')
d = np.array([rows[i]['count_frac'] - rows[i]['area_frac'] for i in dv]); print('054 residual corr with (count - area) fraction', round(np.corrcoef(d, rr)[0, 1], 3))
np.save(_repo_io('phase_count_rows.npy'), rows, allow_pickle=True)
