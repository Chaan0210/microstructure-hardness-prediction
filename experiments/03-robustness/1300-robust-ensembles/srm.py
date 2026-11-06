"""Statistical region merging grain measurement for boundary-free noisy renderings (label-free, per image).
background removal (Gaussian 24) -> SLIC superpixels (~1500) -> RAG on mean intensity; hierarchical merging while the two-sample statistic
|mu_a - mu_b| / (sigma * sqrt(1/n_a + 1/n_b)) < K (sigma = Haar-MAD pixel noise estimate, floored) -> region statistics.
usage: srm.py LIST(train|test|deg4) [K] -> srm-<list>-k<K>.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import slic
from skimage import graph
P = Path(_repo_io('../..')); which = sys.argv[1]; K = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
if which == 'deg4': ids = sorted(f.stem for f in (_repo_path(P, 'experiments/push-400/synth/train_deg4')).glob('*.png')); path = lambda i: _repo_path(P, f'experiments/push-400/synth/train_deg4/{i}.png')
else: ids = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/train.csv' if which == 'train' else 'data/sample_submission.csv')))]; path = lambda i: _repo_path(P, f'data/{which}/{i}.png')
def noise_sd(I):
    f = I / 255.; hh = (f[0::2, 0::2] - f[0::2, 1::2] - f[1::2, 0::2] + f[1::2, 1::2]) / 2; return np.median(np.abs(hh)) / 0.6745 * 255
def merge_w(g, src, dst, n):
    a, b = g.nodes[dst], g.nodes[n]; se = SIG * np.sqrt(1 / a['pixel count'] + 1 / b['pixel count']); return {'weight': abs(a['mean color'][0] - b['mean color'][0]) / se}
def merge_f(g, src, dst):
    g.nodes[dst]['total color'] += g.nodes[src]['total color']; g.nodes[dst]['pixel count'] += g.nodes[src]['pixel count']
    g.nodes[dst]['mean color'] = g.nodes[dst]['total color'] / g.nodes[dst]['pixel count']
rows = []
for i in ids:
    I = np.asarray(Image.open(_repo_io(path(i))).convert('L'), np.float64); SIG = max(noise_sd(I), 2.0)
    R = I - ndi.gaussian_filter(I, 24) + I.mean(); S = ndi.gaussian_filter(R, 0.7)
    lab = slic(np.dstack([S, S, S]) / 255., n_segments=1500, compactness=8, start_label=1, channel_axis=-1)
    rag = graph.rag_mean_color(np.dstack([R, R, R]), lab)
    for n in rag.nodes: rag.nodes[n]['mean color'] = rag.nodes[n]['total color'] / rag.nodes[n]['pixel count']
    for x, y_, d in rag.edges(data=True):
        a, b = rag.nodes[x], rag.nodes[y_]; d['weight'] = abs(a['mean color'][0] - b['mean color'][0]) / (SIG * np.sqrt(1 / a['pixel count'] + 1 / b['pixel count']))
    L = graph.merge_hierarchical(lab, rag, thresh=K, rag_copy=False, in_place_merge=True, merge_func=merge_f, weight_func=merge_w)
    area = np.bincount(L.ravel())[1:] if L.min() > 0 else np.bincount(L.ravel()); area = area[area >= 15]
    la = np.log(area); w = area / area.sum()
    means = ndi.mean(R, L, np.unique(L)); 
    rows.append([len(area), len(area) / (I.size / 1000), np.dot(w, la), la.mean(), la.std(), np.quantile(la, 0.9), np.std(means), SIG])
    if len(rows) % 100 == 0: print(which, len(rows), flush=True)
keys = ['srm_n', 'srm_per_kpx', 'srm_aw_logA', 'srm_mean_logA', 'srm_sd_logA', 'srm_q90_logA', 'srm_region_I_sd', 'srm_sigma']
np.savez(_repo_io(f'srm-{which}-k{K:g}.npz'), ids=np.array(ids), X=np.array(rows), keys=np.array(keys)); print('done', which, len(ids))
