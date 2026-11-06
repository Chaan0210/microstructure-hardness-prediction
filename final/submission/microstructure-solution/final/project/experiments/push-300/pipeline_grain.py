import argparse, csv, json, sys, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'grain-cellpose-220'))

def ids_of(data, split):
    return [r['ID'] for r in csv.DictReader(open(Path(data) / ('train.csv' if split == 'train' else 'sample_submission.csv')))]

def load_img(data, split, i):
    from PIL import Image
    return np.array(Image.open(Path(data) / split / f'{i}.png'))

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
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['zs_segment', 'zs_features', 'ft_segment', 'assemble']); ap.add_argument('split', nargs='?')
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--zs-weights', default=str(HERE.parent / 'grain-cellpose-220/weights/cpsam')); ap.add_argument('--ckpt', default=str(HERE / 'ft-a1s250.pt'))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    if a.stage == 'zs_segment':
        dst = out / f'masks-v3-{a.split}.npz'; assert not dst.exists(), 'refuse overwrite'
        M = model(a.zs_weights); ids = ids_of(a.data, a.split)
        masks = np.stack([seg_rule(M, load_img(a.data, a.split, i)) for i in ids])
        np.savez_compressed(dst, ids=np.array(ids), masks=masks)
    elif a.stage == 'zs_features':
        from features_v3 import feature_matrix_v3
        z = np.load(out / f'masks-v3-{a.split}.npz'); ids = [str(i) for i in z['ids']]
        X, keys = feature_matrix_v3([load_img(a.data, a.split, i) for i in ids], z['masks'])
        np.savez(out / f'features-v3-{a.split}.npz', ids=np.array(ids), X=X, keys=np.array(keys))
    elif a.stage == 'ft_segment':
        from features_v3 import extract_v3
        F = np.load(out / f'features-v3-{a.split}.npz', allow_pickle=True); Ftr = np.load(out / 'features-v3-train.npz', allow_pickle=True)
        keys = [str(k) for k in F['keys']]; ids = [str(i) for i in F['ids']]
        thr = float(np.nanmedian(Ftr['X'][:, list(Ftr['keys']).index('noise')]))           # train-only threshold
        sel = np.where(F['X'][:, keys.index('noise')] > thr)[0]; M = model(a.ckpt)
        X = np.full((len(sel), len(VIEWS), len(keys)), np.nan)
        for n, k in enumerate(sel):
            im = load_img(a.data, a.split, ids[k])
            for v, f in enumerate(VIEWS):
                x = f(im); r = extract_v3(x, seg_rule(M, x)); X[n, v] = [r.get(q, np.nan) if r else np.nan for q in keys]
        np.savez(out / f'ftseg-{a.split}.npz', ids=np.array([ids[k] for k in sel]), X=X, keys=np.array(keys), thr=thr)
    elif a.stage == 'assemble':
        import stcoh
        F = np.load(out / f'features-v3-{a.split}.npz', allow_pickle=True); V = np.load(out / f'ftseg-{a.split}.npz', allow_pickle=True)
        X = F['X'].astype(float).copy(); keys = [str(k) for k in F['keys']]; ids = [str(i) for i in F['ids']]; pos = {i: k for k, i in enumerate(ids)}
        rows = np.array([pos[str(i)] for i in V['ids']]); X[rows] = np.nanmean(V['X'], 1)
        J = [keys.index(k) for k in VSTD]; E = np.full((len(ids), len(J)), np.nan); E[rows] = np.nanstd(V['X'][:, :, J], 1)
        S = np.array([stcoh.features(load_img(a.data, a.split, i).astype(np.float64)) for i in ids])
        np.savez(out / f'feat-{a.split}.npz', X=np.hstack([X, E, S]), keys=np.array(keys + [f'vstd_{k}' for k in VSTD] + stcoh.KEYS), ids=np.array(ids))
    else:
        raise SystemExit(f'unknown stage {a.stage}')
    print(json.dumps(dict(stage=a.stage, split=a.split, sec=round(time.time() - t0))), flush=True)

if __name__ == '__main__':
    main()
