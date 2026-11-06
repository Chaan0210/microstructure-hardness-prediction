"""Small-grain depletion check: segment simulated noisy copies of clean train images with the same segmenter used for the real noisy statistic
(ft-H300, 1 view, masks-v3 rule) and compare the per-image grain-area histogram with that of the clean source masks (paired).
Real noisy vs clean (unpaired, ft-H300 vs masks-v3): 20-40 px 0.26, 40-80 0.47, 80-160 0.71, 160-320 0.96, 640-1280 1.50.
usage: sim_sizebins.py N '{params}' [...]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from PIL import Image
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; P = _RepoPath(__file__).resolve().parents[3]
sys.path.insert(0, str(H)); sys.path.insert(0, str(_repo_path(E, 'push-300')))
from realdeg2 import degrade2
import seglib
from cellpose import models
seglib._model = models.CellposeModel(gpu=True, pretrained_model=str(_repo_path(E, 'push-300/ft-H300.pt')))
from calib_own import ids, nz, Mc
bins = np.array([1, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 10 ** 6]); pool = nz[nz >= 12]
hist = lambda m: np.histogram((lambda a: a[a > 0])(np.bincount(m.ravel().astype(np.int64))[1:]), bins)[0]
clean = [n for n in range(len(ids)) if nz[n] < 9.5][:int(sys.argv[1])]
for js in sys.argv[2:]:
    prm = json.loads(js); r = np.random.RandomState(5); Hs = np.zeros(len(bins) - 1); Hc = np.zeros(len(bins) - 1)
    for n in clean:
        a = np.array(Image.open(_repo_path(P, f'data/train/{ids[n]}.png'))).astype(np.float64); m = Mc['masks'][n]
        d = degrade2(a, m, r, prm, noise=float(r.choice(pool))).astype(np.uint8); Hs += hist(seglib.seg(d)); Hc += hist(m)
    print(json.dumps(prm), ' ratio by bin 20-40/40-80/80-160/160-320/320-640/640-1280: ' + ' '.join(f'{Hs[i] / max(Hc[i], 1e-9):.2f}' for i in range(2, 8)), flush=True)
