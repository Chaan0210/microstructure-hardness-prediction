"""Reproduction driver for the push-400 candidates (031-035): pixel U-Net features -> feature assembly -> TabPFN ensemble -> CSV.
Everything is written to a NEW directory (--out); nothing existing is overwritten.

Inputs
  --base-dir DIR   directory holding the candidate-030 88-column features feat-{train,test}.npz
                   (output of push-300/pipeline_grain.py 'assemble', or push-300/feat-a1s250n4vstd-stcoh-{split}.npz via --base-pattern)
  U-Net weights    unet-u1.pt, unet2-{m1,m2,h1,m3,m4}.pt in this folder (trained label-free by phase_unet.py / phase_unet2.py; MPS training is
                   not bit-deterministic, so reproduction uses the shipped weights)
Stages
  unet SPLIT                 U-Net D4-TTA features: unet-u1-SPLIT.npz, unet2-<net>-SPLIT.npz
  assemble SPLIT             feat-u1phdrop / feat-u1d<net>sz / feat-u1dmavg / feat-u1dmavg4 / feat-u1dmavg5 / feat-unetonly
  predict CID --components   equal mean over components (model@featureset), each = mean of 3 TabPFN seeds fit on all 500 train rows;
                             final = 0.25*best021 + 0.75*grain (best021 where n_obj < 20)  -> submission.csv
Candidate 035 = predict with components v3.5@u1dm1sz,v3.5@u1dm2sz,v3.5@u1dmavg,v3.5@u1dmavg4 (device mps).
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[6]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse, csv, json, os, subprocess, sys, hashlib
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P = HERE.parents[1]
sys.path.insert(0, str(_repo_path(HERE.parent, 'push-300')))
NETS = {'m1': None, 'm2': (5.6343302726745605, 0.9692867994308472), 'h1': None, 'm3': (5.6343302726745605, 0.9692867994308472)}   # (LA_MU, LA_SD) of the training set; None = default (219-image set)
U1COLS = ['u_ph', 'u_ph_hard', 'u_ph_f32']
DROP = ['f_dark_20', 'f_dark_35', 'f_dark_50', 'sp_frac_grains', 'sp_count_frac', 'z_q50', 'z_q75', 'z_q90', 'vstd_sp_frac_grains', 'vstd_sp_count_frac',
        'vstd_f_dark_20', 'vstd_f_dark_35']
MCOLS = ['m_el_glob', 'm_el_loc16', 'm_el_pix', 'm_el_coh', 'm_la_std', 'm_la_f16', 'm_la_f32', 'm_la_tile8', 'm_la_mean', 'm_ph_hard']
AVGS = {'mavg': ['m1', 'm2', 'h1'], 'mavg4': ['m1', 'm2', 'h1', 'm3']}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ld(p): return np.load(_repo_io(p), allow_pickle=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('arg', nargs='?')
    ap.add_argument('--out', required=True); ap.add_argument('--unet-dir', default=None, help='read U-Net feature npz from here instead of --out (assemble)')
    ap.add_argument('--base-pattern', default=str(_repo_path(HERE.parent, 'push-300/feat-a1s250n4vstd-stcoh-{split}.npz')))
    ap.add_argument('--components', default='v3.5@u1dm1sz,v3.5@u1dm2sz,v3.5@u1dmavg,v3.5@u1dmavg4')
    ap.add_argument('--best021', default=str(_repo_path(P, 'artifacts/submission-candidate-021.csv'))); ap.add_argument('--device', default='mps'); ap.add_argument('--W', type=float, default=0.75)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    py = str(_repo_path(P, '.venv/bin/python'))
    if a.stage == 'unet':
        env = dict(os.environ, OUTDIR=str(out), NOMAPS='1')
        subprocess.run([py, str(_repo_path(HERE, 'unet_feats.py')), 'u1', a.arg], env=env, check=True)
        for n, la in NETS.items():
            e = dict(env)
            if la: e.update(LA_MU=repr(la[0]), LA_SD=repr(la[1]))
            subprocess.run([py, str(_repo_path(HERE, 'unet2_feats.py')), n, a.arg, n], env=e, check=True)
    elif a.stage == 'assemble':
        s = a.arg; ud = Path(a.unet_dir) if a.unet_dir else out
        F = ld(a.base_pattern.format(split=s)); keys = [str(k) for k in F['keys']]; ids = F['ids']
        U = ld(ud / f'unet-u1-{s}.npz'); uk = [str(k) for k in U['keys']]; assert (U['ids'] == ids).all()
        X = np.hstack([F['X'].astype(float), U['X'][:, [uk.index(c) for c in U1COLS]].astype(float)]); k1 = keys + U1COLS   # = feat-u1ph
        keep = [i for i, c in enumerate(k1) if c not in DROP]; XD = X[:, keep]; kd = [k1[i] for i in keep]                  # = feat-u1phdrop
        np.savez(_repo_io(out / f'feat-u1phdrop-{s}.npz'), X=XD, keys=np.array(kd), ids=ids)
        M = {}
        for n in NETS:
            D = ld(ud / f'unet2-{n}-{s}.npz'); dk = [str(k) for k in D['keys']]; assert (D['ids'] == ids).all(); M[n] = (D, dk)
            np.savez(_repo_io(out / f'feat-u1d{n}sz-{s}.npz'), X=np.hstack([XD, D['X'][:, [dk.index(c) for c in MCOLS]].astype(float)]), keys=np.array(kd + MCOLS), ids=ids)
        for tag, nets in AVGS.items():
            A = np.mean([M[n][0]['X'][:, [M[n][1].index(c) for c in MCOLS]] for n in nets], 0)
            np.savez(_repo_io(out / f'feat-u1d{tag}-{s}.npz'), X=np.hstack([XD, A]), keys=np.array(kd + MCOLS), ids=ids)
        print(json.dumps(dict(stage='assemble', split=s, n_cols_u1phdrop=len(kd))))
    elif a.stage == 'predict':
        import harness
        lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
        preds = []; prim = None
        for c in a.components.split(','):
            model, fs = c.split('@')
            Tr = ld(out / f'feat-{fs}-train.npz'); Te = ld(out / f'feat-{fs}-test.npz'); keys = [str(k) for k in Tr['keys']]; assert keys == [str(k) for k in Te['keys']]
            Xtr, ytr, Xte = Tr['X'].astype(float), np.array([lab[str(i)] for i in Tr['ids']]), Te['X'].astype(float); ps = []
            for sd in (0, 1, 2):
                est = harness.tabpfn(32, 100 * sd, a.device) if model == 'v2' else harness.tabpfn_new(model, 100 * sd, a.device)
                est.fit(Xtr, ytr); ps.append(est.predict(Xte))
            preds.append(np.mean(ps, 0))
            if prim is None: prim = (Te, keys)
        g = np.mean(preds, 0); Te, keys = prim; tids = [str(i) for i in Te['ids']]; pos = {i: k for k, i in enumerate(tids)}
        p021 = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io(a.best021)))}
        order = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/sample_submission.csv')))]
        failed = np.nan_to_num(Te['X'][:, keys.index('n_obj')], nan=0) < 20
        dst = out / f'submission-{a.arg}.csv'; assert not dst.exists(), 'refuse overwrite'
        with dst.open('x', newline='') as f:
            w = csv.writer(f); w.writerow(['ID', 'hardness'])
            for i in order:
                gi = p021[i] if failed[pos[i]] else g[pos[i]]; w.writerow([i, repr(float((1 - a.W) * p021[i] + a.W * gi))])
        print(json.dumps(dict(csv=str(dst), sha256=sha(dst))))
    else:
        raise SystemExit(f'unknown stage {a.stage}')

if __name__ == '__main__':
    main()
