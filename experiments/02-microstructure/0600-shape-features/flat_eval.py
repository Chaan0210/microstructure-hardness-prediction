"""usage: flat_eval.py MODE[,MODE..] -> partial corr (phase_q.py controls) by noise half + paired bootstrap vs first mode"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np, csv
sys.path.insert(0, _repo_io('../push-500'))
P = '../..'
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io(f'{P}/data/train.csv')))}
F = np.load(_repo_io('../push-400/feat-u1dmavg-train.npz'), allow_pickle=True); k = [str(x) for x in F['keys']]; ids = [str(i) for i in F['ids']]
y = np.array([lab[i] for i in ids]); n = np.nan_to_num(F['X'][:, k.index('noise')], nan=1); h = n >= np.median(n)
ctrl = ['m_la_mean', 'm_la_f32', 'm_la_tile8', 'm_el_glob', 'm_el_pix', 'stcoh_s3_w8_locmed', 'stcoh_s1.5_w32_glob', 'pore_frac', 'm_la_std']
Xc = F['X'][:, [k.index(c) for c in ctrl]].astype(float); Xc = np.where(np.isnan(Xc), np.nanmedian(Xc, 0), Xc); C = np.c_[np.ones(len(y)), Xc]
def pcorr(v, sel):
    A = C[sel]; ry = y[sel] - A @ np.linalg.lstsq(A, y[sel], rcond=None)[0]; rv = v[sel] - A @ np.linalg.lstsq(A, v[sel], rcond=None)[0]
    return np.corrcoef(rv, ry)[0, 1]
modes = sys.argv[1].split(','); R = {m: np.load(_repo_io(f'flat-{m}.npz')) for m in modes}
for m in modes: assert [str(i) for i in R[m]['ids']] == ids
hi = np.where(h)[0]; rng = np.random.RandomState(0); boots = [rng.randint(0, len(hi), len(hi)) for _ in range(300)]
for m in modes:
    print(m, 'bg_sd clean/noisy', round(float(np.median(R[m]['bg_sd'][~h])), 2), round(float(np.median(R[m]['bg_sd'][h])), 2))
    for a in ['soft', 'argmax', 'z_f25', 'z_f30', 'z_f50', 'z_mean']:
        line = f'  {a:7s} clean {pcorr(R[m][a], ~h):+.3f} noisy {pcorr(R[m][a], h):+.3f}'
        if m != modes[0]:
            d = [pcorr(R[m][a], hi[b]) - pcorr(R[modes[0]][a], hi[b]) for b in boots]; line += f'  d_noisy {np.mean(d):+.3f} (90% {np.percentile(d,5):+.3f},{np.percentile(d,95):+.3f})'
        print(line)
