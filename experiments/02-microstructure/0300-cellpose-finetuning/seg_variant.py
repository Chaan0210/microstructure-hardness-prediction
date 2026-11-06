"""Segment the noisy half of a split with a preprocessing variant and extract v3 features (single view + optional D4 views)."""
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
split, variant, nviews = sys.argv[1], sys.argv[2], int(sys.argv[3]); v0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0; half = sys.argv[5] if len(sys.argv) > 5 else 'noisy'
P = Path(_repo_io('../..'))
F = np.load(_repo_io(f'../grain-cellpose-220/features-v3-{split}.npz'), allow_pickle=True); ids = [str(i) for i in F['ids']]; keys = [str(k) for k in F['keys']]
Ftr = np.load(_repo_io('../grain-cellpose-220/features-v3-train.npz'), allow_pickle=True); thr = float(np.nanmedian(Ftr['X'][:, list(Ftr['keys']).index('noise')]))
nz = F['X'][:, keys.index('noise')]; sel = np.where(nz > thr)[0] if half == 'noisy' else np.where(~(nz > thr))[0]
fn = getattr(seglib, variant)
V = [lambda a: a, lambda a: np.ascontiguousarray(a.T), lambda a: np.ascontiguousarray(a[::-1]), lambda a: np.ascontiguousarray(np.rot90(a)),
     lambda a: np.ascontiguousarray(np.rot90(a, 2)), lambda a: np.ascontiguousarray(np.rot90(a, 3)), lambda a: np.ascontiguousarray(a[:, ::-1]), lambda a: np.ascontiguousarray(np.rot90(a, 2).T)][v0:nviews]
name = f'var-{variant}-{split}-{half}-v{v0}to{nviews}'; part = Path(f'{name}.partial.npz')
out = np.full((len(sel), nviews - v0, len(keys)), np.nan); t0 = time.time(); start = 0
if part.exists():  # resume: rows [0, done) were saved by an interrupted run
    Z = np.load(_repo_io(part), allow_pickle=True); assert [str(i) for i in Z['ids']] == [ids[k] for k in sel]; start = int(Z['done']); out[:start] = Z['X'][:start]
    print(json.dumps(dict(resume_from=start)), flush=True)
def save_partial(done):
    np.savez(_repo_io(part), ids=np.array([ids[k] for k in sel]), X=out, keys=np.array(keys), thr=thr, done=done)
for n, k in enumerate(sel):
    if n < start: continue
    im = np.array(Image.open(_repo_path(P, f'data/{split}/{ids[k]}.png')))
    for v, f in enumerate(V):
        x = f(im); r = extract_v3(x, fn(x)); out[n, v] = [r.get(q, np.nan) if r else np.nan for q in keys]
    if (n + 1) % 10 == 0: save_partial(n + 1)
    if (n + 1) % 25 == 0: print(json.dumps(dict(done=n + 1, n=len(sel), sec=round(time.time() - t0))), flush=True)
np.savez(_repo_io(f'{name}.npz'), ids=np.array([ids[k] for k in sel]), X=out, keys=np.array(keys), thr=thr)
part.unlink(missing_ok=True); print('DONE', flush=True)
