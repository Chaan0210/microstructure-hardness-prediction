"""Render one copy set (PLAN.md): every clean source (noise < train median, 249 images) -> synth/train_<set>/<ID>.png + synth/train_<set>_params.json.
Variant V1 = render.BASE; V2 = BASE + KEEP_MEAN 1, BW 2, P_LO = P_HI = 1. Noise target per source drawn from the real noise >= 12 train pool with a
seed fixed by (set, source), so the two copies of a variant differ. No labels are read.
usage: render_set.py SET VARIANT COPY   e.g. render_set.py v1a V1 0"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, csv, numpy as np, multiprocessing as mp
from pathlib import Path
from PIL import Image
W = Path(__file__).resolve().parents[1]; E = _RepoPath(__file__).resolve().parents[3]; P = _RepoPath(__file__).resolve().parents[4]
sys.path.insert(0, str(_repo_path(W, 'work'))); import render
VAR = {'V1': {}, 'V2': {'KEEP_MEAN': 1, 'BW': 2, 'P_LO': 1, 'P_HI': 1}}
_M = None; _ids = None
def _init():
    global _M, _ids
    _M = np.load(_repo_path(E, 'grain-cellpose-220/masks-v3-train.npz'))['masks']; _ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))]
def _one(a):
    k, prm, tgt, seed, out = a
    img = np.array(Image.open(_repo_path(P, f'data/train/{_ids[k]}.png'))); r = np.random.RandomState(seed)
    g = render.render(img, _M[k], r, prm, tgt).astype(np.uint8); Image.fromarray(g).save(_repo_io(out))
    return _ids[k], dict(target_noise=tgt, measured_noise=render.hh(g.astype(float)))
if __name__ == '__main__':
    st, var, cp = sys.argv[1], sys.argv[2], int(sys.argv[3]); out = _repo_path(W, f'synth/train_{st}'); out.mkdir(parents=True, exist_ok=False)
    ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))]; nz = np.load(_repo_path(E, 'push-300/noise_train.npy')) * 255
    clean = np.where(nz < np.median(nz))[0]; pool = np.sort(nz[nz >= 12.0])
    jobs = []
    for k in clean:
        rt = np.random.RandomState(700_000 + 1000 * cp + (0 if var == 'V1' else 500) + int(k)); tgt = float(pool[rt.randint(len(pool))])
        jobs.append((int(k), VAR[var], tgt, 9_000_000 + 100_000 * cp + (0 if var == 'V1' else 50_000) + int(k), str(out / f'{ids[k]}.png')))
    with mp.get_context('spawn').Pool(6, initializer=_init) as pl: res = pl.map(_one, jobs, chunksize=8)
    json.dump(dict(res), open(_repo_path(W, f'synth/train_{st}_params.json'), 'x'), indent=0)
    print(st, var, cp, len(res), 'median |measured - target| noise', float(np.median([abs(v['measured_noise'] - v['target_noise']) for _, v in res])))
