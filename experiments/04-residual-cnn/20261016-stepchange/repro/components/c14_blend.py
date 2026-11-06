"""Assemble candidate-076-structure predictions from components, for dev (Study900 400 x 3) and test:
  where(failed, p021, 0.16*p021 + 0.384*TAB + 0.128*RIDGE + 0.128*LOGR + 0.05*SET + 0.15*GMM)
  gated = noise >= 12 & ~failed: TAB = mean of 12 'h9-full' members, RIDGE/LOGR = DTA arm B, GMM = full;
  non-gated: TAB = mean of 12 'h9-nb' members, RIDGE/LOGR/GMM = nb.
usage: c14_blend.py EXPROOT DEVDIR DEVTAG FINTAB PARTNERS OUTPREFIX
  DEVDIR/dev-DEVTAG-{nb|full}-{slot}-{v}.npz  (push-300 dev_eval format; DEVTAG may contain {fs},{s},{v} placeholders instead)
  FINTAB npz with nb_{slot}, full_{slot} test arrays; PARTNERS = c14_partners.py output"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[5]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np
from pathlib import Path
E = Path(sys.argv[1]).resolve(); DD = Path(sys.argv[2]); tagpat = sys.argv[3]; FT = np.load(_repo_io(sys.argv[4])); PT = np.load(_repo_io(sys.argv[5])); OUT = sys.argv[6]
S9 = E / 'probabilistic-grains-900'; P = E.parent
base = np.load(_repo_io(S9 / 'base.npz'), allow_pickle=True); sid = [str(i) for i in base['ids']]; nzd = np.nan_to_num(base['noise'], nan=0)
D = np.load(_repo_io(S9 / 'ridge/predictions.npz'), allow_pickle=True); assert [str(i) for i in D['ids']] == sid; y = D['y'][0]; fail = D['failed'].astype(bool); p021 = D['p021']
sets = [np.load(_repo_io(d / 'oof.npz'))['prediction'] for d in (S9 / 'set_model', S9 / 'set_model/seed731509', S9 / 'set_model/seed731527')]; setd = np.mean(sets, 0)
def align_dev(path, key='g'):
    d = np.load(_repo_io(path), allow_pickle=True); out = np.zeros((3, 400))
    for k, n in enumerate(['original', '20261031', '20261101']):
        ii = [str(i) for i in d[f'{n}_ids']]; pos = {i: j for j, i in enumerate(ii)}; out[k] = d[f'{n}_{key}'][[pos[i] for i in sid]]
    return out
S, V = ['038', '039', '044'], ['u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4']
devname = (lambda fs, s, v: f'dev-h9phys-{fs}-044-{v}.npz' if s == '044' else f'dev-h9-{fs}-{s}-{v}.npz') if tagpat == 'ORIG076' else (lambda fs, s, v: 'dev-' + tagpat.format(fs=fs, s=s, v=v) + '.npz')  # ORIG076: production 076 file names (push-1300/eval_combo.py)
tab = {fs: np.mean([align_dev(DD / devname(fs, s, v)) for s in S for v in V], 0) for fs in ('nb', 'full')}
W = dict(cnn=0.16, tab=0.384, ridge=0.128, logr=0.128, set=0.05, gmm=0.15)
def blend(gate, ng, failed, p, C):
    return np.where(failed, p, sum(W[k] * C[k] for k in W))
g = (nzd >= 12) & ~fail[0]; ng = ~g & ~fail[0]
Cd = dict(cnn=p021, set=setd, tab=np.where(ng, tab['nb'], tab['full']), ridge=np.where(g, PT['dta_ridge_dev'], PT['ridge_nb_dev']), logr=np.where(g, PT['dta_logridge_dev'], PT['logridge_nb_dev']),
          gmm=np.where(ng, PT['gmm_nb_dev'], PT['gmm_full_dev']))
dv = blend(g, ng, fail, p021, Cd); rm = lambda a, m=slice(None): float(np.sqrt(((a - y)[:, m] ** 2).mean()))
print('DEV RMSE %.4f parts %s | clean<9.5 %.3f mid %.3f noisy %.3f' % (rm(dv), np.round(np.sqrt(((dv - y) ** 2).mean(1)), 4), rm(dv, nzd < 9.5), rm(dv, (nzd >= 9.5) & (nzd < 12)), rm(dv, nzd >= 12)))
print('   components dev RMSE (non-failed):', {k: round(rm(np.where(fail, p021, v)), 3) for k, v in Cd.items()})
order = [r['ID'] for r in csv.DictReader(open(_repo_io(P / 'data/sample_submission.csv')))]; assert [str(i) for i in PT['order']] == order
pt = np.array([float(r['hardness']) for r in csv.DictReader(open(_repo_io(P / 'artifacts/submission-candidate-021.csv')))])
R = np.load(_repo_io(S9 / 'ridge/final500/predictions.npz')); T = np.load(_repo_io(S9 / 'set_model/full500/predictions.npz'), allow_pickle=True); failed = R['failed'] | T['failed']
nzt = PT['test_noise']; gt = (nzt >= 12) & ~failed; ngt = ~gt & ~failed
Ct = dict(cnn=pt, set=T['set_mean_raw'], tab=np.where(ngt, np.mean([FT[f'nb_{s}'] for s in S], 0), np.mean([FT[f'full_{s}'] for s in S], 0)),
          ridge=np.where(gt, PT['dta_ridge_test'], PT['ridge_nb_test']), logr=np.where(gt, PT['dta_logridge_test'], PT['logridge_nb_test']), gmm=np.where(ngt, PT['gmm_nb_test'], PT['gmm_full_test']))
te = blend(gt, ngt, failed, pt, Ct)
np.savez(_repo_io(OUT + '.npz'), dev=dv, test=te, y=y, nz_dev=nzd, fail_dev=fail, failed_test=failed, nz_test=nzt, order=np.array(order), **{f'dev_{k}': v for k, v in Cd.items()}, **{f'test_{k}': v for k, v in Ct.items()})
print('saved', OUT + '.npz')
