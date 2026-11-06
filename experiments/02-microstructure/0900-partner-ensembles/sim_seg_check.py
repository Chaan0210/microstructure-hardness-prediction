"""Is the calibrated simulation as hard as real noisy images? Segment simulated copies (realdeg2, noise from our train noise values >= 12) of clean
train images with the production noisy-row segmenter ft-a1s250 (1 view, masks-v3 rule) and compare their v3 features with the zero-shot features of
the same clean source images: paired median shift in clean-SD units. Real noisy rows show n_obj -0.33, grain_eqd_mean +0.26, grain_logA_std -0.70
(ft_diag.py, unpaired). usage: sim_seg_check.py N_IMAGES ['{json params}']"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json
import sys, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from PIL import Image
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; P = _RepoPath(__file__).resolve().parents[3]
sys.path.insert(0, str(H)); sys.path.insert(0, str(_repo_path(E, 'push-300'))); sys.path.insert(0, str(_repo_path(E, 'grain-cellpose-220')))
from realdeg2 import degrade2
import seglib
from features_v3 import extract_v3
from cellpose import models
seglib._model = models.CellposeModel(gpu=True, pretrained_model=str(_repo_path(E, 'push-300/ft-a1s250.pt')))
F = np.load(_repo_path(E, 'grain-cellpose-220/features-v3-train.npz'), allow_pickle=True); ids = [str(i) for i in F['ids']]; keys = [str(k) for k in F['keys']]
Mc = np.load(_repo_path(E, 'grain-cellpose-220/masks-v3-train.npz'), allow_pickle=True); nz = np.load(_repo_path(E, 'push-300/noise_train.npy')) * 255; pool = nz[nz >= 12]
clean = [n for n in range(len(ids)) if nz[n] < 9.5][:int(sys.argv[1])]; r = np.random.RandomState(11); PRM = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None; print('PRM', PRM)
COLS = ['n_obj', 'grain_mean_area_j', 'grain_eqd_mean', 'aw_eqd_mean', 'grain_logA_std', 'coverage', 'pore_frac', 'ar_wmean', 'neighbors_mean']
S = []
for n in clean:
    a = np.array(Image.open(_repo_path(P, f'data/train/{ids[n]}.png'))).astype(np.float64)
    d = degrade2(a, Mc['masks'][n], r, PRM, noise=float(r.choice(pool))).astype(np.uint8); f = extract_v3(d, seglib.seg(d))
    S.append([f.get(c, np.nan) if f else np.nan for c in COLS])
S = np.array(S, float); C = F['X'][clean][:, [keys.index(c) for c in COLS]].astype(float); sd = F['X'][[n for n in range(len(ids)) if nz[n] < 9.5]][:, [keys.index(c) for c in COLS]].astype(float)
print(f'ft-a1s250 on {len(clean)} calibrated sims (paired median shift / clean SD):  ' + '  '.join(f'{c} {np.nanmedian(S[:, j] - C[:, j]) / np.nanstd(sd[:, j]):+.2f}' for j, c in enumerate(COLS)))
print('paired corr sim vs clean source: ' + '  '.join(f'{c} {np.corrcoef(S[np.isfinite(S[:, j]), j], C[np.isfinite(S[:, j]), j])[0, 1]:.2f}' for j, c in enumerate(COLS)))
