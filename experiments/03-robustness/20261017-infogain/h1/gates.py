"""H1 label-free gates G1/G2 (no labels read)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr
HERE = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[3]
inp = np.load(_repo_path(E, 'rmse-research-20261015/inputs.npz'), allow_pickle=True)
ids = [str(i) for i in inp['ids']]; nz = inp['noise']; X = inp['X']; k = list(inp['keys'])
tr = np.load(_repo_path(HERE, 'phi1-train.npz')); dg = np.load(_repo_path(HERE, 'phi1-deg4.npz'))
assert list(tr['ids']) == ids
pos = {i: j for j, i in enumerate(ids)}
clean = (nz < 9.5) & (nz > 0)
out = {}
for j, name in enumerate(['phi1_global', 'phi1_block']):
    out['G1_' + name] = float(spearmanr(tr['X'][clean, j], X[clean, k.index('u_ph')])[0])
# G2: copy vs source stability on deg4 pairs
src = np.array([pos[i] for i in dg['ids']])
z = np.load(_repo_path(E, 'push-1200/synthrows-deg4-u1dmavgH2Z-anghcor.npz'), allow_pickle=True)
zk = list(z['keys']); zpos = {str(i): j for j, i in enumerate(z['ids'])}
assert all(i in zpos for i in dg['ids'])
zi = np.array([zpos[i] for i in dg['ids']])
uph_copy = z['X'][zi, zk.index('u_ph')].astype(float); uph_src = X[src, k.index('u_ph')]
c = np.load(_repo_path(E, 'push-1300/cumul-deg4.npz'), allow_pickle=True); ct = np.load(_repo_path(E, 'push-1300/cumul-train.npz'), allow_pickle=True)
cpos = {str(i): j for j, i in enumerate(c['ids'])}; tpos = {str(i): j for j, i in enumerate(ct['ids'])}
p34_copy = c['X'][[cpos[i] for i in dg['ids']], 1]; p34_src = ct['X'][[tpos[i] for i in dg['ids']], 1]
out['G2_pearson'] = dict(
    phi1_global=float(pearsonr(dg['X'][:, 0], tr['X'][src, 0])[0]),
    phi1_block=float(pearsonr(dg['X'][:, 1], tr['X'][src, 1])[0]),
    u_ph=float(pearsonr(uph_copy, uph_src)[0]),
    h9_p34=float(pearsonr(p34_copy, p34_src)[0]))
# extra diagnostics: does phi1 track u_ph on real noisy rows?
noisy = nz >= 12
out['diag_noisy_spearman_phi1_uph'] = [float(spearmanr(tr['X'][noisy, j], X[noisy, k.index('u_ph')])[0]) for j in (0, 1)]
out['G1_pass'] = bool(max(out['G1_phi1_global'], out['G1_phi1_block']) >= 0.70)
g2 = out['G2_pearson']
out['G2_pass'] = bool(max(g2['phi1_global'], g2['phi1_block']) > max(g2['u_ph'], g2['h9_p34']))
json.dump(out, open(_repo_path(HERE, 'gates.json'), 'w'), indent=1); print(json.dumps(out, indent=1))
