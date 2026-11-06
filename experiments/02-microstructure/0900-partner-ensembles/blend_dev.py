"""Dev OOF ([3, 400], Study900 ids/partitions) of the submitted blend structures, for partner screening on top of 054.
c051 = 0.2*021 + 0.48*grain045 + 0.16*ridge + 0.16*logRidge;  c054 = 0.77*c051 + 0.08*set + 0.15*gmm (failed rows -> 021).
set = mean of the grain-set seeds found (Study900 set_model + seed dirs + setZ/seed*), or the first `nset` of them.
usage: from blend_dev import load; B = load(); B['c054'], B['y'], ...;  python blend_dev.py  (prints RMSEs)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; S9 = _repo_path(E, 'probabilistic-grains-900')
def set_dirs():
    d = [_repo_path(S9, 'set_model'), _repo_path(S9, 'set_model/seed731509'), _repo_path(S9, 'set_model/seed731527')]
    return d + sorted((_repo_path(H, 'setZ')).glob('seed*'), key=lambda p: int(p.name[4:]))
def load(nset=None):
    D = np.load(_repo_path(S9, 'ridge/predictions.npz'), allow_pickle=True); ids = [str(i) for i in D['ids']]
    y = D['y'][0]; fail = D['failed'].astype(bool); p = D['p021']
    lr = np.load(_repo_path(H, 'diverse-ridge_a2_logy_var4.npz'))['oof']; gm = np.load(_repo_path(H, 'gmm-partner-dev.npz'))['oof']
    S = []
    for d in set_dirs():
        f = d / 'oof.npz'
        if not f.exists(): continue
        o = np.load(_repo_io(f)); assert [str(i) for i in o['ids']] == ids; S.append(o['prediction'])
    S = np.array(S if nset is None else S[:nset])
    c51 = 0.2 * p + 0.48 * D['grain045'] + 0.16 * D['ridge'] + 0.16 * lr
    st = S.mean(0); c54 = 0.77 * c51 + 0.08 * st + 0.15 * gm
    W = lambda a: np.where(fail, p, a)
    return dict(ids=ids, y=y, fold=D['fold'], fail=fail, p021=p, g=D['grain045'], ridge=D['ridge'], logridge=lr, gmm=gm, sets=S, set=st,
                c051=W(c51), c054=W(c54))
def rmse(a, y): return float(np.sqrt(((a - y) ** 2).mean()))
def parts(a, y): return np.sqrt(((a - y) ** 2).mean(1))
if __name__ == '__main__':
    B = load(); y = B['y']
    print('n set seeds', len(B['sets']), ' c051', round(rmse(B['c051'], y), 4), ' c054', round(rmse(B['c054'], y), 4), ' set alone', round(rmse(B['set'], y), 4))
    B9 = load(9); print('c054 with 9 seeds', round(rmse(B9['c054'], y), 4))
def partner(B, m, label, base='c054', shares=(0.05, 0.1, 0.15, 0.2)):
    """Report a partner OOF m [3, 400] on top of B[base]: standalone, error corr, optimal share, (1-a)*base + a*m gains (pooled and per partition)."""
    y = B['y']; c = B[base]; m = np.where(B['fail'], B['p021'], m)
    corr = np.corrcoef((m - y).ravel(), (c - y).ravel())[0, 1]; d = m - c; aopt = -np.mean(d * (c - y)) / np.mean(d * d)
    line = [f'{label}: alone {rmse(m, y):.3f} corr {corr:.3f} opt a {aopt:.3f}']
    for a in shares:
        b = c + a * d; line.append(f'a{a}: {rmse(b, y):.4f} ({rmse(c, y) - rmse(b, y):+.4f}; parts ' + '/'.join(f'{v:+.3f}' for v in parts(c, y) - parts(b, y)) + ')')
    print('\n  '.join(line), flush=True)
def align_dev(npz_path, key='g'):
    """dev-<tag>.npz (push-300 format, per-partition ids) -> [3, 400] in Study900 id order."""
    d = np.load(_repo_io(npz_path), allow_pickle=True); ids = [str(i) for i in np.load(_repo_path(S9, 'base.npz'), allow_pickle=True)['ids']]; out = np.zeros((3, 400))
    for k, n in enumerate(['original', '20261031', '20261101']):
        ii = [str(i) for i in d[f'{n}_ids']]; pos = {i: j for j, i in enumerate(ii)}; out[k] = d[f'{n}_{key}'][[pos[i] for i in ids]]
    return out
