"""Label-free calibration of render.py on pixel statistics of REAL noisy train images (noise >= 12). For a parameter set, render one copy of each
clean source (noise < train median, 249 images; noise target drawn per source from the real noisy pool with a fixed per-source seed, so all
settings are paired), compute pixstats, and measure copy-vs-real discrimination (grouped 5-fold logistic / GBM AUC) plus standardized mean
differences. No labels are read. usage: imported (evaluate) or `calib.py NAME 'json-params'` -> work/calib/NAME.json"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, csv, numpy as np, multiprocessing as mp
from pathlib import Path
from PIL import Image
W = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[3]; P = _RepoPath(__file__).resolve().parents[4]
sys.path.insert(0, str(W))
import pixstats, render
from pix_gap import gap, keys as PKEYS, Xr as PXR
_M = None; _ids = None
def _init():
    global _M, _ids
    _M = np.load(_repo_path(E, 'grain-cellpose-220/masks-v3-train.npz'))['masks']; _ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))]
def sources():
    nz = np.load(_repo_path(E, 'push-300/noise_train.npy')) * 255; clean = np.where(nz < np.median(nz))[0]; pool = np.sort(nz[nz >= 12.0]); return clean, pool
def _one(args):
    k, prm, tgt, seed, save = args
    a = np.array(Image.open(_repo_path(P, f'data/train/{_ids[k]}.png'))); r = np.random.RandomState(seed)
    g = render.render(a, _M[k], r, prm, tgt).astype(np.uint8)
    if save: Image.fromarray(g).save(_repo_io(save))
    o = pixstats.stats(g); return _ids[k], [o[q] for q in PKEYS], float(pixstats.noise_hh(g.astype(float)))
def run(prm, tag=0, save_dir=None, procs=7):
    clean, pool = sources(); jobs = []
    for k in clean:
        rt = np.random.RandomState(500_000 + k); tgt = float(pool[rt.randint(len(pool))])   # paired across settings
        jobs.append((int(k), prm, tgt, 1_000_000 * (tag + 1) + int(k), None if save_dir is None else str(_repo_path(Path(save_dir), 'X.png'))))
    if save_dir is not None:
        ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))]; jobs = [(k, p, t, s, str(_repo_path(Path(save_dir), f'{ids[k]}.png'))) for k, p, t, s, _ in jobs]
    with mp.get_context('spawn').Pool(procs, initializer=_init) as pl: out = pl.map(_one, jobs, chunksize=8)
    ids = np.array([o[0] for o in out]); X = np.array([o[1] for o in out]); meas = np.array([o[2] for o in out]); tg = np.array([j[2] for j in jobs])
    return ids, X, meas, tg
def evaluate(prm, tag=0, procs=7):
    ids, X, meas, tg = run(prm, tag, procs=procs); ni = PKEYS.index('noise'); m = X[:, ni] >= 12
    res = gap(X[m], ids[m]); res['n_ge12'] = int(m.sum()); res['noise_err'] = float(np.median(np.abs(meas - tg))); return res, (ids, X)
if __name__ == '__main__':
    name, prm = sys.argv[1], json.loads(sys.argv[2]); (_repo_path(W, 'calib')).mkdir(exist_ok=True)
    res, (ids, X) = evaluate(prm); res['params'] = prm
    json.dump(res, open(_repo_path(W, f'calib/{name}.json'), 'w'), indent=1); np.savez(_repo_path(W, f'calib/{name}.npz'), ids=ids, X=X, keys=np.array(PKEYS))
    print(f"{name}: logit {res['logit']:.3f} gbm {res['gbm']:.3f} n {res['n_ge12']} | top SMD", list(res['smd'].items())[:8], flush=True)
