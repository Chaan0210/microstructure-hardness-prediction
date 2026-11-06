"""Final 500-row models on the aligned (qm) features -> test predictions (test rows only predicted, never fitted).
  tab SLOT   TabPFN-3.5 grain slot, 4 variants x 3 seeds (100*sd), as push-400/pipeline_final.py predict:
             SLOT 38 -> feat-<v>-qm, 39 -> feat-<v>-anghqm, 44 -> feat-<v>-anghcorqm, 44orig -> ../push-800/feat-<v>-anghcor (reproduction check)
             -> qm-final-tab<SLOT>.npz (ids, g, per-variant preds)
  ridge      RidgeCV (0.1..1000) on the average variant + log-target Ridge alpha 2 geometric mean over the 4 variants -> qm-final-ridge.npz
  gmm        Study900 image_gmm__x98_augmented Ridge (alpha by 3-fold inner CV, seed 19001) on aligned X98 + GMM readout -> qm-final-gmm.npz
usage: final_qm.py tab 38|39|44|44orig  |  final_qm.py ridge  |  final_qm.py gmm"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; P0 = _RepoPath(__file__).resolve().parents[3]
V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P0, 'data/train.csv')))}
order = [r['ID'] for r in csv.DictReader(open(_repo_path(P0, 'data/sample_submission.csv')))]
def pair(path_tr, path_te):
    Tr = np.load(_repo_io(path_tr), allow_pickle=True); Te = np.load(_repo_io(path_te), allow_pickle=True); assert [str(k) for k in Tr['keys']] == [str(k) for k in Te['keys']]
    tpos = {str(i): k for k, i in enumerate(Te['ids'])}; Xte = Te['X'][[tpos[i] for i in order]].astype(float)
    return Tr['X'].astype(float), np.array([lab[str(i)] for i in Tr['ids']]), Xte, [str(i) for i in Tr['ids']]
SRC = {'38': lambda v: (_repo_path(H, f'feat-{v}-qm-train.npz'), _repo_path(H, f'feat-{v}-qm-test.npz')), '39': lambda v: (_repo_path(H, f'feat-{v}-anghqm-train.npz'), _repo_path(H, f'feat-{v}-anghqm-test.npz')),
       '44': lambda v: (_repo_path(H, f'feat-{v}-anghcorqm-train.npz'), _repo_path(H, f'feat-{v}-anghcorqm-test.npz')),
       '44orig': lambda v: (_repo_path(E, f'push-800/feat-{v}-anghcor-train.npz'), _repo_path(E, f'push-800/feat-{v}-anghcor-test.npz'))}
mode = sys.argv[1]
if mode == 'tab':
    sys.path.insert(0, str(_repo_path(E, 'push-300'))); import harness
    slot = sys.argv[2]; per = []
    for v in V:
        Xtr, ytr, Xte, _ = pair(*SRC[slot](v)); ps = []
        for sd in (0, 1, 2):
            m = harness.tabpfn_new('v3.5', 100 * sd, 'mps'); m.fit(Xtr, ytr); ps.append(m.predict(Xte))
        per.append(np.mean(ps, 0)); print(slot, v, 'done', flush=True)
    np.savez(_repo_path(H, f'qm-final-tab{slot}.npz'), ids=np.array(order), g=np.mean(per, 0), per=np.array(per))
elif mode == 'ridge':
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV, Ridge
    Xtr, ytr, Xte, _ = pair(*SRC['44']('u1dmavgH2Z'))
    m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000])).fit(Xtr, ytr); r = m.predict(Xte)
    logs = []
    for v in V:
        Xtr, ytr, Xte, _ = pair(*SRC['44'](v)); logs.append(make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=2.0)).fit(Xtr, np.log(ytr)).predict(Xte))
    np.savez(_repo_path(H, 'qm-final-ridge.npz'), ids=np.array(order), ridge=r, logridge=np.exp(np.mean(logs, 0)), ridge_alpha=m[-1].alpha_)
    print('ridge alpha', m[-1].alpha_)
elif mode == 'gmm':
    src = open(_repo_path(H, 'gmm_partner.py')).read().split("mode = sys.argv[1]")[0]; ns = {'__file__': str(_repo_path(H, 'gmm_partner.py'))}; exec(src, ns)
    S9 = _repo_path(E, 'probabilistic-grains-900'); Btr = np.load(_repo_io(S9 / 'object-bank-train.npz'), allow_pickle=True); Bte = np.load(_repo_io(S9 / 'object-bank-test.npz'), allow_pickle=True)
    Xtr, ytr, Xte, tid = pair(*SRC['44']('u1dmavgH2Z'))
    Xtr = np.c_[Xtr, ns['gmm_feats'](Btr, tid)]; Xte = np.c_[Xte, ns['gmm_feats'](Bte, order)]
    pred, a = ns['fit_predict'](Xtr, ytr, Xte, 19001); np.savez(_repo_path(H, 'qm-final-gmm.npz'), ids=np.array(order), pred=pred, alpha=a); print('gmm alpha', a)
