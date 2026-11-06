"""End-to-end grain pipeline for candidate 030 (reproduction driver). All outputs go to a NEW directory (--out); nothing existing is overwritten.

Stages (run in order; each reads the previous stage's outputs from --out):
  zs_segment  SPLIT   zero-shot Cellpose-SAM masks with the masks-v3 rule (cp -2 / flow 0.8; if coverage < 0.95 rerun cp -3 / flow 1.0 and
                      keep it when coverage increases)                                   -> masks-v3-SPLIT.npz
  zs_features SPLIT   v3 grain features (features_v3.extract_v3)                          -> features-v3-SPLIT.npz
  ft_segment  SPLIT   noisy half (noise > train median) with the fine-tuned segmenter, D4 views id/T/UD/R90 -> ftseg-SPLIT.npz
  assemble    SPLIT   v3 features, noisy rows <- 4-view mean, + 16 across-view std columns, + 8 structure-tensor coherence columns
                                                                                         -> feat-SPLIT.npz
  predict             TabPFN-v2 (n_est 32, seeds 0/100/200) and TabPFN-3.5 (default, seeds 0/100/200) fit on all 500 train rows,
                      grain = mean of the two; final = 0.25*best021 + 0.75*grain (best021 where n_obj < 20)   -> submission.csv
Fine-tuning (ft_cellpose.py a1 800 1 2e-5 4 0 -> ft-a1s250.pt) is a separate script; pass its checkpoint with --ckpt.
Data: --data DIR with train.csv, sample_submission.csv, train/*.png, test/*.png. Labels are read only in 'predict' (train.csv).
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse, csv, json, sys, time, hashlib, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(_repo_path(HERE.parent, 'grain-cellpose-220')))

def ids_of(data, split):
    return [r['ID'] for r in csv.DictReader(open(_repo_path(Path(data), 'train.csv' if split == 'train' else 'sample_submission.csv')))]

def load_img(data, split, i):
    from PIL import Image
    return np.array(Image.open(_repo_path(Path(data), split, f'{i}.png')))

def model(ckpt):
    from cellpose import models
    return models.CellposeModel(gpu=True, pretrained_model=str(ckpt))

def seg_rule(M, im):
    m, _, _ = M.eval(im.copy(), flow_threshold=0.8, cellprob_threshold=-2.0)
    if (m > 0).mean() < 0.95:
        m2, _, _ = M.eval(im.copy(), flow_threshold=1.0, cellprob_threshold=-3.0)
        if (m2 > 0).mean() > (m > 0).mean(): m = m2
    return m.astype(np.uint16)

VIEWS = [lambda a: a, lambda a: np.ascontiguousarray(a.T), lambda a: np.ascontiguousarray(a[::-1]), lambda a: np.ascontiguousarray(np.rot90(a))]
VSTD = ['sp_frac_grains', 'sp_count_frac', 'f_dark_20', 'f_dark_35', 'sizefield_range_s32', 'sizefield_std_s32', 'sizefield_std_s16', 'tile8_logA_std',
        'grain_logA_std', 'grain_eqd_mean', 'aw_eqd_mean', 'n_obj', 'solidity_wmean', 'neighbors_std', 'ar_wmean', 'orient_coherence']

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('split', nargs='?')
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--zs-weights', default=str(_repo_path(HERE.parent, 'grain-cellpose-220/weights/cpsam'))); ap.add_argument('--ckpt', default=str(_repo_path(HERE, 'ft-a1s250.pt')))
    ap.add_argument('--best021', default=str(_repo_path(_RepoPath(__file__).resolve().parents[3], 'artifacts/submission-candidate-021.csv'))); ap.add_argument('--device', default='cpu')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    if a.stage == 'zs_segment':
        dst = out / f'masks-v3-{a.split}.npz'; assert not dst.exists(), 'refuse overwrite'
        M = model(a.zs_weights); ids = ids_of(a.data, a.split)
        masks = np.stack([seg_rule(M, load_img(a.data, a.split, i)) for i in ids])
        np.savez_compressed(_repo_io(dst), ids=np.array(ids), masks=masks)
    elif a.stage == 'zs_features':
        from features_v3 import feature_matrix_v3
        z = np.load(_repo_io(out / f'masks-v3-{a.split}.npz')); ids = [str(i) for i in z['ids']]
        X, keys = feature_matrix_v3([load_img(a.data, a.split, i) for i in ids], z['masks'])
        np.savez(_repo_io(out / f'features-v3-{a.split}.npz'), ids=np.array(ids), X=X, keys=np.array(keys))
    elif a.stage == 'ft_segment':
        from features_v3 import extract_v3
        F = np.load(_repo_io(out / f'features-v3-{a.split}.npz'), allow_pickle=True); Ftr = np.load(_repo_io(out / 'features-v3-train.npz'), allow_pickle=True)
        keys = [str(k) for k in F['keys']]; ids = [str(i) for i in F['ids']]
        thr = float(np.nanmedian(Ftr['X'][:, list(Ftr['keys']).index('noise')]))           # train-only threshold
        sel = np.where(F['X'][:, keys.index('noise')] > thr)[0]; M = model(a.ckpt)
        X = np.full((len(sel), len(VIEWS), len(keys)), np.nan)
        for n, k in enumerate(sel):
            im = load_img(a.data, a.split, ids[k])
            for v, f in enumerate(VIEWS):
                x = f(im); r = extract_v3(x, seg_rule(M, x)); X[n, v] = [r.get(q, np.nan) if r else np.nan for q in keys]
        np.savez(_repo_io(out / f'ftseg-{a.split}.npz'), ids=np.array([ids[k] for k in sel]), X=X, keys=np.array(keys), thr=thr)
    elif a.stage == 'assemble':
        import stcoh
        F = np.load(_repo_io(out / f'features-v3-{a.split}.npz'), allow_pickle=True); V = np.load(_repo_io(out / f'ftseg-{a.split}.npz'), allow_pickle=True)
        X = F['X'].astype(float).copy(); keys = [str(k) for k in F['keys']]; ids = [str(i) for i in F['ids']]; pos = {i: k for k, i in enumerate(ids)}
        rows = np.array([pos[str(i)] for i in V['ids']]); X[rows] = np.nanmean(V['X'], 1)
        J = [keys.index(k) for k in VSTD]; E = np.full((len(ids), len(J)), np.nan); E[rows] = np.nanstd(V['X'][:, :, J], 1)
        S = np.array([stcoh.features(load_img(a.data, a.split, i).astype(np.float64)) for i in ids])
        np.savez(_repo_io(out / f'feat-{a.split}.npz'), X=np.hstack([X, E, S]), keys=np.array(keys + [f'vstd_{k}' for k in VSTD] + stcoh.KEYS), ids=np.array(ids))
    elif a.stage == 'predict':
        import harness
        Tr = np.load(_repo_io(out / 'feat-train.npz'), allow_pickle=True); Te = np.load(_repo_io(out / 'feat-test.npz'), allow_pickle=True); keys = [str(k) for k in Tr['keys']]
        assert keys == [str(k) for k in Te['keys']]
        lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(Path(a.data), 'train.csv')))}; y = np.array([lab[str(i)] for i in Tr['ids']])
        per = {}
        for mname in ('v2', 'v3.5'):
            ps = []
            for sd in (0, 1, 2):
                est = harness.tabpfn(32, 100 * sd, a.device) if mname == 'v2' else harness.tabpfn_new(mname, 100 * sd, a.device)
                est.fit(Tr['X'], y); ps.append(est.predict(Te['X']))
            per[mname] = np.mean(ps, 0)
        g = (per['v2'] + per['v3.5']) / 2; tids = [str(i) for i in Te['ids']]; pos = {i: k for k, i in enumerate(tids)}
        p021 = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_io(a.best021)))}
        order = [r['ID'] for r in csv.DictReader(open(_repo_path(Path(a.data), 'sample_submission.csv')))]
        failed = np.nan_to_num(Te['X'][:, keys.index('n_obj')], nan=0) < 20
        rows = []
        for i in order:
            gi = p021[i] if failed[pos[i]] else g[pos[i]]; rows.append((i, 0.25 * p021[i] + 0.75 * gi))
        dst = out / 'submission.csv'; assert not dst.exists(), 'refuse overwrite'
        with dst.open('x', newline='') as f:
            w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in rows]
        print(json.dumps(dict(csv=str(dst), sha256=hashlib.sha256(dst.read_bytes()).hexdigest())))
    else:
        raise SystemExit(f'unknown stage {a.stage}')
    print(json.dumps(dict(stage=a.stage, split=a.split, sec=round(time.time() - t0))), flush=True)

if __name__ == '__main__':
    main()
