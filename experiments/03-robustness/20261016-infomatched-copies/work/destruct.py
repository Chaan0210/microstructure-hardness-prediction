"""Diagnostics (not decision criteria).
(1) Destructiveness, label-free: for each rendering setting in work/calib/<name>.npz, correlation across sources between a structure statistic of
    the clean source image (work/ps/all.npz) and of its rendered copy. 'noiseonly' (white noise to the same target, nothing else) is the
    reference: what the real noise level alone leaves. A setting that falls far below it is erasing source structure.
(2) Population check: hardness of the copy sources (noise < train median) vs real noisy rows (noise >= 12): mean, SD, KS, Levene.
    Reads train labels for this comparison only; nothing is fitted.
usage: destruct.py -> work/destruct.json"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json, csv, numpy as np
from pathlib import Path
from scipy import stats as st
W = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[3]; P = _RepoPath(__file__).resolve().parents[4]
A = np.load(_repo_path(W, 'ps/all.npz')); keys = [str(k) for k in A['keys']]; pos = {str(i): k for k, i in enumerate(A['ids'])}
STRUCT = ['z3_f12', 'z3_q95', 'z1.5_q99', 'z1.5_q998', 'aniso_low', 'aniso_mid', 'rp1', 'rp2', 'rp3', 'bp4_16_sd']
order = [l.split()[0] for l in open(_repo_path(W, 'calib/sweep2.log'))]
auc = {l.split()[0]: (float(l.split()[2]), float(l.split()[4])) for l in open(_repo_path(W, 'calib/sweep2.log'))}
res = {'destruct': {}}
print(f"{'setting':15s} {'logit':>6s} {'gbm':>6s} " + ' '.join(f'{k[:9]:>9s}' for k in STRUCT))
for n in order:
    C = np.load(_repo_path(W, f'calib/{n}.npz')); ids = [str(i) for i in C['ids']]; Xs = A['X'][[pos[i] for i in ids]]; Xc = C['X']
    r = {k: float(np.corrcoef(Xs[:, keys.index(k)], Xc[:, keys.index(k)])[0, 1]) for k in STRUCT}
    res['destruct'][n] = dict(pix_auc_logit=auc[n][0], pix_auc_gbm=auc[n][1], source_copy_r=r)
    print(f'{n:15s} {auc[n][0]:6.3f} {auc[n][1]:6.3f} ' + ' '.join(f'{r[k]:9.3f}' for k in STRUCT))
ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))]; y = np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))])
nz = np.load(_repo_path(E, 'push-300/noise_train.npy')) * 255; src = nz < np.median(nz); real = nz >= 12
a, b = y[src], y[real]
res['population'] = dict(n_src=int(src.sum()), n_real=int(real.sum()), mean_src=float(a.mean()), mean_real=float(b.mean()), sd_src=float(a.std(ddof=1)), sd_real=float(b.std(ddof=1)),
                         welch_p=float(st.ttest_ind(a, b, equal_var=False).pvalue), ks_p=float(st.ks_2samp(a, b).pvalue), levene_p=float(st.levene(a, b).pvalue))
print('population', {k: round(v, 3) for k, v in res['population'].items()})
json.dump(res, open(_repo_path(W, 'destruct.json'), 'w'), indent=1)
