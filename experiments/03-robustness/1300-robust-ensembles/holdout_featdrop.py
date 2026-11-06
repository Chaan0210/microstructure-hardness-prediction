"""Confirmatory holdout-100 test (rows never used for the feature-group screen): TabPFN v3.5 block members (3 slots x 4 U-Net variants, seeds 0,1,2)
fitted on the 400 dev rows, predicting the 100 holdout rows, with all columns vs without the 'intensity' group, without 'het', without both.
usage: holdout_featdrop.py -> holdout-featdrop.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, json, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, _repo_io('../push-300')); import harness
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io('../../data/train.csv')))}
dev = set(str(i) for i in np.load(_repo_io('../probabilistic-grains-900/base.npz'), allow_pickle=True)['ids'])
G = json.load(open(_repo_io('abl/groups.json'))); V = ['u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4']
S = {'038': '../push-500/feat-{v}H2Z-train.npz', '039': '../push-700/feat-{v}H2Z-angh-train.npz', '044': '../push-800/feat-{v}H2Z-anghcor-train.npz'}
CF = {'full': [], 'nointensity': G['intensity'].split(), 'nohet': G['het'].split(), 'noboth': G['intensity'].split() + G['het'].split()}
out = {}
for s, pat in S.items():
    for v in V:
        F = np.load(_repo_io(pat.format(v=v)), allow_pickle=True); keys = [str(k) for k in F['keys']]; ids = [str(i) for i in F['ids']]; X = F['X'].astype(float)
        tr = np.array([i in dev for i in ids]); y = np.array([lab[i] for i in ids])
        if 'hid' not in out: out['hid'] = np.array(ids)[~tr]; out['y'] = y[~tr]; out['noise'] = np.nan_to_num(X[~tr][:, keys.index('noise')] * 255, nan=0)
        assert list(out['hid']) == list(np.array(ids)[~tr])
        for c, drop in CF.items():
            cols = [j for j, k in enumerate(keys) if k not in drop]; ps = []
            for sd in (0, 1, 2):
                m = harness.tabpfn_new('v3.5', 100 * sd + 7, 'mps'); m.fit(X[tr][:, cols], y[tr]); ps.append(m.predict(X[~tr][:, cols]))
            out[f'{s}_{v}_{c}'] = np.mean(ps, 0)
        print(s, v, 'done', {c: len([k for k in keys if k not in d]) for c, d in CF.items()}, flush=True)
np.savez(_repo_io('holdout-featdrop.npz'), **out)
