"""H1: blur-invariant first-moment phase fraction (label-free). Writes h1/phi1-{train,deg4}.npz. Reads images only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[4]
E = _RepoPath(__file__).resolve().parents[3]


def mode_level(a):
    h, e = np.histogram(a, bins=256, range=(0, 256))
    hs = ndi.gaussian_filter1d(h.astype(float), 2)
    return e[np.argmax(hs)] + 0.5


def measure(I):
    I = I.astype(float)
    s15 = ndi.gaussian_filter(I, 1.5)
    s25 = ndi.gaussian_filter(I, 2.5)
    M = mode_level(s15)
    thr = np.percentile(s25, 0.5)
    core = s25 <= thr
    P = float(I[core].mean())
    pore = ndi.binary_dilation(core, iterations=2)
    keep = ~pore
    den = max(M - P, 1.0)
    g = (M - I[keep].mean()) / den
    vals = []
    for by in range(4):
        for bx in range(4):
            sl = (slice(64 * by, 64 * by + 64), slice(64 * bx, 64 * bx + 64))
            kb = keep[sl]
            if kb.sum() < 500:
                continue
            Mb = mode_level(s15[sl][kb])
            vals.append(Mb - I[sl][kb].mean())
    b = float(np.mean(vals)) / den
    return [g, b, M, P, float(I[keep].mean())]


KEYS = ['phi1_global', 'phi1_block', 'h1_M', 'h1_P', 'h1_mean']


def run(name, paths, ids):
    X = np.array([measure(np.array(Image.open(_repo_io(p)))) for p in paths])
    np.savez_compressed(_repo_path(HERE, f'phi1-{name}.npz'), ids=np.array(ids), X=X, keys=np.array(KEYS))
    print(name, X.shape, np.nanmean(X, 0).round(3))


if __name__ == '__main__':
    inp = np.load(_repo_path(E, 'rmse-research-20261015/inputs.npz'), allow_pickle=True)
    ids = [str(i) for i in inp['ids']]
    run('train', [_repo_path(ROOT, f'data/train/{i}.png') for i in ids], ids)
    for deg in sys.argv[1:]:
        d = _repo_path(E, f'push-400/synth/train_{deg}')
        files = sorted(d.glob('TRAIN_*.png'))
        run(deg, files, [f.stem for f in files])
