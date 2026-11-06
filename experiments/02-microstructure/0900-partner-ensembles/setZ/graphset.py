"""Grain-graph variant of Study900's grain-set regressor: same object bank, skip Ridge, training/epoch selection and dev protocol, but each grain's
embedding is refined by one round of message passing over its K nearest grains (centroid distance, same image) before area-weighted pooling:
h = phi([x, g]);  m = mean_{j in kNN(i)} h_j;  z = psi([h, m, h - m]);  pool mean/sd of z.  Fixed a priori: K = 6, widths as the original.
outputs var-graph/seed<k>/oof.npz.   usage: graphset.py SEED[,SEED...]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import copy, importlib.util, sys, numpy as np, torch
from torch import nn
from pathlib import Path
from scipy.spatial import cKDTree
HERE = Path(__file__).resolve().parent; S9 = _repo_path(HERE.parents[1], 'probabilistic-grains-900')
spec = importlib.util.spec_from_file_location('grain_set_run', _repo_path(HERE, 'run.py')); run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
ORIG = copy.deepcopy(run.CONFIG); run.torch.set_num_threads(2); K = 6
def knn_table(bank):
    keys = [str(k) for k in bank['object_keys']]; cx, cy = keys.index('cx'), keys.index('cy'); off = bank['offsets']; nbr = np.zeros((len(bank['objects']), K), np.int64)
    for i in range(len(off) - 1):
        a, b = off[i], off[i + 1]; n = b - a; P = bank['objects'][a:b][:, [cx, cy]]
        if n == 1: nbr[a] = a; continue
        k = min(K + 1, n); _, J = cKDTree(P).query(P, k=k); J = J[:, 1:] if k > 1 else J
        if J.shape[1] < K: J = np.concatenate([J, np.repeat(J[:, -1:], K - J.shape[1], 1)], 1)
        nbr[a:b] = a + J
    return nbr
class GraphData(run.TensorData):
    NBR = None; LOCAL = None
    def batch(self, images, *, reorder=False, padding=False):
        images = np.asarray(images, dtype=int)
        rows = np.concatenate([np.arange(self.offsets[i], self.offsets[i + 1]) for i in images])
        counts = self.offsets[images + 1] - self.offsets[images]; group = np.repeat(np.arange(len(images)), counts)
        if reorder:
            perm = np.random.default_rng(4813).permutation(len(rows)); rows, group = rows[perm], group[perm]
        local = GraphData.LOCAL; local[rows] = np.arange(len(rows)); nb = local[GraphData.NBR[rows]]
        ri = torch.as_tensor(rows, device=self.device); bi = torch.as_tensor(group, device=self.device); ii = torch.as_tensor(images, device=self.device)
        x, w = self.objects[ri], self.weights[ri]; nbt = torch.as_tensor(nb, device=self.device)
        if padding:
            n0 = len(rows); x = torch.cat([x, torch.zeros((len(images), x.shape[1]), device=self.device)]); w = torch.cat([w, torch.zeros(len(images), device=self.device)])
            bi = torch.cat([bi, torch.arange(len(images), device=self.device)]); pad = torch.arange(n0, n0 + len(images), device=self.device)[:, None].repeat(1, K); nbt = torch.cat([nbt, pad])
        return (x, w, bi, self.globals[ii], self.skip[ii], nbt), self.y[ii]
class GraphSet(nn.Module):
    def __init__(self, object_dim, global_dim):
        super().__init__(); C = run.CONFIG; E = C['embedding_dim']
        self.phi = nn.Sequential(nn.Linear(object_dim + global_dim, C['encoder_width']), nn.SiLU(), nn.Linear(C['encoder_width'], E), nn.SiLU())
        self.psi = nn.Sequential(nn.Linear(3 * E, C['encoder_width']), nn.SiLU(), nn.Linear(C['encoder_width'], E), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(2 * E + global_dim, C['head_width']), nn.SiLU(), nn.Dropout(C['head_dropout']), nn.Linear(C['head_width'], 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
    def forward(self, objects, weights, group, globals_, skip, nbr):
        h = self.phi(torch.cat([objects, globals_[group]], dim=1)); m = h[nbr].mean(1); z = self.psi(torch.cat([h, m, h - m], 1))
        mean = torch.zeros((len(globals_), z.shape[1]), device=z.device, dtype=z.dtype); second = torch.zeros_like(mean)
        mean.index_add_(0, group, weights[:, None] * z); second.index_add_(0, group, weights[:, None] * z.square())
        sd = (second - mean.square()).clamp(min=1e-8).sqrt(); correction = self.head(torch.cat([mean, sd, globals_], dim=1)).squeeze(-1)
        return skip + correction, correction
run.TensorData = GraphData; run.GrainSet = GraphSet
if __name__ == '__main__':
    bank, base, matched = run.load_inputs(_repo_path(S9, 'object-bank-train.npz'), _repo_path(S9, 'base.npz')); GraphData.NBR = knn_table(bank); GraphData.LOCAL = np.zeros(len(bank['objects']), np.int64)
    for seed in [int(s) for s in sys.argv[1].split(',')]:
        run.CONFIG = {**copy.deepcopy(ORIG), 'seed': seed}; out = _repo_path(HERE, 'var-graph', f'seed{seed}'); out.mkdir(parents=True, exist_ok=True)
        run.run_train(bank, base, matched, _repo_path(S9, 'object-bank-train.npz'), _repo_path(S9, 'base.npz'), out, 'cpu'); print('done', seed, flush=True)
