"""Segment with a fine-tuned Cellpose-SAM checkpoint (same masks-v3 threshold rule as seglib.seg) and extract v3 features.
usage: seg_ft.py CKPT SPLIT HALF NVIEWS [V0=0] [aug]  (views V0..NVIEWS-1 of [id, T, UD, R90, R180, R270, LR, antiT])
  HALF: noisy | clean | all | fid  (fid = the 80 cleanest train images degraded exactly as fidelity_exp.py: blur 1.0, noise 17, contrast 0.85, rng 7;
        output holds [clean-input, degraded-input] as the two 'views')
output: ftseg-<ckpt>-<split>-<half>-n<NVIEWS>.npz with X [n, views, keys] (keys = features-v3 keys), resumable via .partial.npz
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, time, json, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.'); sys.path.insert(0, _repo_io('../grain-cellpose-220'))
from pathlib import Path
from PIL import Image
import seglib
from features_v3 import extract_v3
from degrade import degrade
ckpt, split, half, nviews = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); v0 = int(sys.argv[5]) if len(sys.argv) > 5 else 0
from cellpose import models
seglib._model = models.CellposeModel(gpu=True, pretrained_model=ckpt)
AUG = len(sys.argv) > 6 and sys.argv[6] == 'aug'   # cellpose flip-averaged flows (augment=True) before mask computation
if AUG:
    def _seg_aug(im):
        M = seglib.model()
        m, _, _ = M.eval(im.copy(), flow_threshold=0.8, cellprob_threshold=-2.0, augment=True)
        if (m > 0).mean() < 0.95:
            m2, _, _ = M.eval(im.copy(), flow_threshold=1.0, cellprob_threshold=-3.0, augment=True)
            if (m2 > 0).mean() > (m > 0).mean(): m = m2
        return m.astype(np.uint16)
    seglib.seg = _seg_aug
X2 = Path(ckpt).stem.startswith('ftx2')     # 2x-scale model (ft_cellpose_x2.py): segment the 2x bicubic upsample, return native-resolution labels
def segment(x):
    if not X2: return seglib.seg(x)
    up = np.asarray(Image.fromarray(x).resize((512, 512), Image.BICUBIC))
    return seglib.seg(up)[::2, ::2].copy()
P = Path(_repo_io('../..'))
F = np.load(_repo_io(f'../grain-cellpose-220/features-v3-{split}.npz'), allow_pickle=True); ids = [str(i) for i in F['ids']]; keys = [str(k) for k in F['keys']]
Ftr = np.load(_repo_io('../grain-cellpose-220/features-v3-train.npz'), allow_pickle=True); thr = float(np.nanmedian(Ftr['X'][:, list(Ftr['keys']).index('noise')]))
nz = F['X'][:, keys.index('noise')]
if half == 'noisy': sel = np.where(nz > thr)[0]
elif half == 'clean': sel = np.where(~(nz > thr))[0]
elif half == 'all': sel = np.arange(len(ids))
elif half == 'fid': sel = np.argsort(np.load(_repo_io('noise_train.npy')))[:80]; nviews = 2
V = [lambda a: a, lambda a: np.ascontiguousarray(a.T), lambda a: np.ascontiguousarray(a[::-1]), lambda a: np.ascontiguousarray(np.rot90(a)),
     lambda a: np.ascontiguousarray(np.rot90(a, 2)), lambda a: np.ascontiguousarray(np.rot90(a, 3)), lambda a: np.ascontiguousarray(a[:, ::-1]), lambda a: np.ascontiguousarray(np.rot90(a, 2).T)][v0:nviews]
tag = Path(ckpt).stem
name = f'ftseg-{tag}-{split}-{half}-n{nviews}' + (f'-v{v0}' if v0 else '') + ('-aug' if AUG else ''); part = Path(f'{name}.partial.npz')
out = np.full((len(sel), nviews - v0, len(keys)), np.nan); t0 = time.time(); start = 0
if part.exists():
    Z = np.load(_repo_io(part), allow_pickle=True); start = int(Z['done']); out[:start] = Z['X'][:start]; print(json.dumps(dict(resume_from=start)), flush=True)
rng = np.random.RandomState(7)
masks = np.zeros((len(sel), 256, 256), np.uint16)   # view-0 masks (kept so feature definitions can be revised without re-segmenting)
if part.exists() and 'masks' in Z.files: masks[:start] = Z['masks'][:start]
for n, k in enumerate(sel):
    im = np.array(Image.open(_repo_path(P, f'data/{split}/{ids[k]}.png')))
    if half == 'fid':
        d = degrade(im, rng, blur=1.0, noise=17, contrast=0.85)   # rng consumed in the same order as fidelity_exp.py
        if n < start: continue
        for v, x in enumerate([im, d]):
            r = extract_v3(x, segment(x)); out[n, v] = [r.get(q, np.nan) if r else np.nan for q in keys]
    else:
        if n < start: continue
        for v, f in enumerate(V):
            x = f(im); m = segment(x); r = extract_v3(x, m); out[n, v] = [r.get(q, np.nan) if r else np.nan for q in keys]
            if v == 0 and v0 == 0: masks[n] = m
    if (n + 1) % 10 == 0: np.savez(_repo_io(part), ids=np.array([ids[j] for j in sel]), X=out, keys=np.array(keys), done=n + 1, masks=masks)
    if (n + 1) % 25 == 0: print(json.dumps(dict(done=n + 1, n=len(sel), sec=round(time.time() - t0))), flush=True)
np.savez(_repo_io(f'{name}.npz'), ids=np.array([ids[j] for j in sel]), X=out, keys=np.array(keys), thr=thr)
if half != 'fid' and v0 == 0: np.savez_compressed(_repo_io(f'{name}-masks.npz'), ids=np.array([ids[j] for j in sel]), masks=masks)
part.unlink(missing_ok=True); print('DONE', flush=True)
