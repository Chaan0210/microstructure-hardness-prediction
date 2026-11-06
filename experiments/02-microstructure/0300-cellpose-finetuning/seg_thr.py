"""Label-free threshold calibration: segment a fixed subset of noisy-half train images with a checkpoint under several
(flow_threshold, cellprob_threshold) settings and compare key feature medians with the clean half (zero-shot v3), since the
microstructure is independent of image noise. usage: seg_thr.py CKPT N_IMG"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, time, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.'); sys.path.insert(0, _repo_io('../grain-cellpose-220'))
from pathlib import Path
from PIL import Image
from features_v3 import extract_v3
from cellpose import models
ckpt, nimg = sys.argv[1], int(sys.argv[2])
M = models.CellposeModel(gpu=True, pretrained_model=ckpt)
F = np.load(_repo_io('../grain-cellpose-220/features-v3-train.npz'), allow_pickle=True); ids = [str(i) for i in F['ids']]; keys = [str(k) for k in F['keys']]
nz = F['X'][:, keys.index('noise')]; thr = np.nanmedian(nz)
noisy = np.where(nz > np.nanquantile(nz, 2 / 3))[0]; sel = noisy[np.random.RandomState(0).permutation(len(noisy))[:nimg]]
clean = np.where(nz <= thr)[0]
kk = ['n_obj', 'grain_eqd_mean', 'grain_logA_std', 'sizefield_range_s32', 'tile8_logA_std', 'sp_frac_grains', 'sp_count_frac', 'neighbors_std', 'solidity_wmean']
J = [keys.index(k) for k in kk]
ref = np.nanmedian(F['X'][clean][:, J], 0)
print('clean-half zs median', json.dumps(dict(zip(kk, np.round(ref, 4).tolist()))), flush=True)
settings = [(0.8, -2.0, True), (0.4, 0.0, False), (0.8, 0.0, False), (1.2, -2.0, False), (0.8, -1.0, False), (2.0, -2.0, False)]
for fl, cp, rule in settings:
    t0 = time.time(); rows = []
    for k in sel:
        im = np.array(Image.open(_repo_io(f'../../data/train/{ids[k]}.png')))
        m, _, _ = M.eval(im.copy(), flow_threshold=fl, cellprob_threshold=cp)
        if rule and (m > 0).mean() < 0.95:
            m2, _, _ = M.eval(im.copy(), flow_threshold=1.0, cellprob_threshold=-3.0)
            if (m2 > 0).mean() > (m > 0).mean(): m = m2
        r = extract_v3(im, m.astype(np.uint16)); rows.append([r.get(q, np.nan) if r else np.nan for q in keys])
    R = np.array(rows); med = np.nanmedian(R[:, J], 0)
    print(json.dumps(dict(flow=fl, cp=cp, rule=rule, sec=round(time.time() - t0), rel=dict(zip(kk, np.round(med / ref, 3).tolist())))), flush=True)
