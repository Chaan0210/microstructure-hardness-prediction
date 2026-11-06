from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, sys
F = np.load(_repo_io('../push-500/feat-u1dmavgH2Z-train.npz'), allow_pickle=True); keys = list(map(str, F['keys'])); pos = {str(i): k for k, i in enumerate(F['ids'])}
def run(files, W=0.75, label=''):
    D = [np.load(_repo_io(f), allow_pickle=True) for f in files]; tot = {}
    for p in ['original', '20261031', '20261101']:
        y = D[0][p + '_y']; q = D[0][p + '_p021']; g = np.mean([d[p + '_g'] for d in D], 0); f_ = D[0][p + '_failed']; b = (1 - W) * q + W * np.where(f_, q, g); fold = D[0][p + '_fold']
        ids = list(map(str, D[0][p + '_ids'])); nz = np.array([np.nan_to_num(F['X'][pos[i], keys.index('noise')], nan=0.1) * 255 for i in ids]); lq = nz >= 11.9
        outs = {'none': b.copy(), 'global': np.zeros_like(b), 'byq': np.zeros_like(b), 'lqonly': b.copy()}
        for k in range(5):
            tr, va = fold != k, fold == k
            c = np.polyfit(b[tr], y[tr], 1); outs['global'][va] = np.polyval(c, b[va])
            for m_ in (lq, ~lq):
                c = np.polyfit(b[tr & m_], y[tr & m_], 1); outs['byq'][va & m_] = np.polyval(c, b[va & m_])
            c = np.polyfit(b[tr & lq], y[tr & lq], 1); outs['lqonly'][va & lq] = np.polyval(c, b[va & lq])
        for k_, v in outs.items(): tot.setdefault(k_, []).append((v - y) ** 2)
    print(label, ' '.join('%s %.4f' % (k_, np.sqrt(np.mean(v))) for k_, v in tot.items()))
run(['../push-600/dev-%s-angh.npz' % v for v in ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']], label='039')
run(['../push-300/dev-%s-v35.npz' % v for v in ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']], label='038')
