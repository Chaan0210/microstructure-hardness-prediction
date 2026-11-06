"""Full noisy-row feature pipeline (as candidate 030/032) on the synthetic degraded clean-train images.
Stage A (cellpose venv): ft-a1s250 4-view segmentation -> v3 features (mean over views) + 16 vstd cols + 8 stcoh cols.
output: synth/feat-ft' + SET + '.npz (ids, X [n, 88] in 030 column order). Resumable via synth/feat-ft' + SET + '.partial.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[6]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, time, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
HERE = _repo_path(Path(__file__).resolve().parents[1], 'push-400'); SET = sys.argv[1]
sys.path.insert(0, str(_repo_path(HERE.parent, 'push-300'))); sys.path.insert(0, str(_repo_path(HERE.parent, 'grain-cellpose-220')))
from PIL import Image
import seglib, stcoh
from features_v3 import extract_v3
from cellpose import models
seglib._model = models.CellposeModel(gpu=True, pretrained_model=str(_repo_path(HERE.parent, 'push-300/ft-a1s250.pt')))
F030 = np.load(_repo_path(HERE.parent, 'push-300/feat-a1s250n4vstd-stcoh-train.npz'), allow_pickle=True); keys = [str(k) for k in F030['keys']]
vk = keys[:64]; VSTD = [k[5:] for k in keys if k.startswith('vstd_')]
ids = sorted(json.load(open(_repo_path(HERE, 'synth/' + SET + '_params.json'))).keys())
V = [lambda a: a, lambda a: np.ascontiguousarray(a.T), lambda a: np.ascontiguousarray(a[::-1]), lambda a: np.ascontiguousarray(np.rot90(a))]
part = _repo_path(HERE, 'synth/feat-ft' + SET + '.partial.npz'); X = np.full((len(ids), len(keys)), np.nan); start = 0
if part.exists(): Z = np.load(_repo_io(part)); X[:int(Z['done'])] = Z['X'][:int(Z['done'])]; start = int(Z['done'])
t0 = time.time()
for n, i in enumerate(ids):
    if n < start: continue
    im = np.array(Image.open(_repo_path(HERE, f'synth/{SET}/{i}.png')))
    VX = np.full((4, 64), np.nan)
    for v, f in enumerate(V):
        x = f(im); r = extract_v3(x, seglib.seg(x)); VX[v] = [r.get(q, np.nan) if r else np.nan for q in vk]
    row = np.full(len(keys), np.nan); row[:64] = np.nanmean(VX, 0)
    for c in VSTD: row[keys.index('vstd_' + c)] = np.nanstd(VX[:, vk.index(c)])
    S = stcoh.features(im.astype(np.float64)); row[[keys.index(k) for k in stcoh.KEYS]] = S
    X[n] = row
    if (n + 1) % 10 == 0: np.savez(_repo_io(part), X=X, done=n + 1)
    if (n + 1) % 25 == 0: print(json.dumps(dict(done=n + 1, n=len(ids), sec=round(time.time() - t0))), flush=True)
np.savez(_repo_path(HERE, 'synth/feat-ft' + SET + '.npz'), ids=np.array(ids), X=X, keys=np.array(keys)); part.unlink(missing_ok=True); print('DONE', flush=True)
