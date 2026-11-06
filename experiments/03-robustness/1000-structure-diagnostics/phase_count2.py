"""H1 with the production-like grain classification: grain shade_darkness (zero-shot object bank, shading-corrected) > t -> phase grain,
pores excluded. For t in (0.10, 0.15, 0.20): area fraction vs number fraction (all grains; interior grains only). Clean train rows.
Also partial correlation with y given sh_cor15, and the 054 residual correlation."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv, sys, numpy as np
sys.path.insert(0, _repo_io('../push-900'))
from blend_dev import load
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}
P = np.load(_repo_io('/private/tmp/claude-501/-Users-dongchan-Desktop-PythonWorkspace-microstructure-2026/f62e347b-cb1a-4c17-8d54-2bf505be6706/scratchpad/presets.npz')); nzm = dict(zip(P['tr'], P['ntr']))
Bk = np.load(_repo_io('../probabilistic-grains-900/object-bank-train.npz'), allow_pickle=True); ok_ = [str(k) for k in Bk['object_keys']]; bid = [str(i) for i in Bk['ids']]; off = Bk['offsets']; O = Bk['objects']
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]; fpos = {str(i): k for k, i in enumerate(F['ids'])}
ids = [i for i in bid if nzm[i] < 9.5]; y = np.array([lab[i] for i in ids]); sh = np.array([F['X'][fpos[i], keys.index('sh_cor15')] for i in ids]).astype(float)
B = load(9); res = (B['c054'] - B['y']).mean(0); dpos = {i: k for k, i in enumerate(B['ids'])}; dv = np.array([i in dpos for i in ids]); rr = np.array([res[dpos[i]] if i in dpos else np.nan for i in ids])
def pc(x, y, z):   # partial corr of x and y given z
    rx = x - np.polyval(np.polyfit(z, x, 1), z); ry = y - np.polyval(np.polyfit(z, y, 1), z); return np.corrcoef(rx, ry)[0, 1]
for t in (0.10, 0.15, 0.20, 0.25):
    A, N, NI = [], [], []
    for i in ids:
        r = bid.index(i); z = O[off[r]:off[r + 1]]; a = z[:, ok_.index('area')]; d = z[:, ok_.index('shade_darkness')]; bd = z[:, ok_.index('border')]
        m = (a >= 12) & (z[:, ok_.index('pore_flag')] < .5) & np.isfinite(d); ph = d[m] > t; inn = bd[m] < 0.5
        A.append((a[m] * ph).sum() / a[m].sum()); N.append(ph.mean()); NI.append(ph[inn].mean() if inn.sum() > 5 else ph.mean())
    A, N, NI = map(np.array, (A, N, NI))
    out = []
    for nm, v in (('area', A), ('count', N), ('count_inner', NI)):
        out.append(f'{nm} corr(y) {np.corrcoef(v, y)[0, 1]:+.3f} pcorr|sh {pc(v, y, sh):+.3f} res {np.corrcoef(v[dv], rr[dv])[0, 1]:+.3f}')
    print(f't={t:.2f}: ' + ' | '.join(out))
print('sh_cor15 corr(y)', round(np.corrcoef(sh, y)[0, 1], 3))
