"""Branch 1 on dev: structure-conditioned reading of spatial tokens, in the grain-set training framework (push-900/setZ/run.py: Ridge skip on
token mean/SD + structure columns, zero-initialised neural correction, inner-split epoch selection, refit; identical config and seeds).
Tokens: 16 x 16 patches of the z1 U-Net maps (tokbank-train.npz); structure = 12 core summary columns. Variants (same widths):
  struct  correction = head(structure)                                   (token encoder output ignored)
  generic phi(token), uniform mean/SD pooling, head([pooled, structure])
  cond    phi([token, structure]) and attention pooling weights softmax(a([token, structure])) per image, head([pooled, structure])
env BANK=dino -> tokbank-dino-train.npz (raw-image DINOv2 tokens), outputs var-tokdino-<VARIANT>
usage: tokset.py VARIANT SEED[,SEED...]  -> var-tok-<VARIANT>/seed<k>/oof.npz"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import copy, importlib.util, sys, numpy as np, torch
from torch import nn
from pathlib import Path
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; S9 = _repo_path(E, 'probabilistic-grains-900')
spec = importlib.util.spec_from_file_location('grain_set_run', _repo_path(E, 'push-900/setZ/run.py')); run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
ORIG = copy.deepcopy(run.CONFIG); run.torch.set_num_threads(2); VARIANT = sys.argv[1]
def segsoftmax(s, group, n):
    mx = torch.full((n,), -1e9, device=s.device).scatter_reduce(0, group, s, reduce='amax', include_self=True)
    e = torch.exp(s - mx[group]); den = torch.zeros(n, device=s.device).index_add_(0, group, e); return e / den[group]
class TokSet(nn.Module):
    def __init__(self, object_dim, global_dim):
        super().__init__(); C = run.CONFIG; Ed = C['embedding_dim']; cin = object_dim + (global_dim if VARIANT == 'cond' else 0)
        self.phi = nn.Sequential(nn.Linear(cin, C['encoder_width']), nn.SiLU(), nn.Linear(C['encoder_width'], Ed), nn.SiLU())
        self.att = nn.Sequential(nn.Linear(object_dim + global_dim, C['encoder_width']), nn.SiLU(), nn.Linear(C['encoder_width'], 1)) if VARIANT == 'cond' else None
        self.head = nn.Sequential(nn.Linear(2 * Ed + global_dim, C['head_width']), nn.SiLU(), nn.Dropout(C['head_dropout']), nn.Linear(C['head_width'], 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
    def forward(self, objects, weights, group, globals_, skip):
        n = len(globals_); g = globals_[group]
        z = self.phi(torch.cat([objects, g], 1) if VARIANT == 'cond' else objects)
        w = segsoftmax(self.att(torch.cat([objects, g], 1)).squeeze(-1), group, n) if VARIANT == 'cond' else weights
        mean = torch.zeros((n, z.shape[1]), device=z.device).index_add_(0, group, w[:, None] * z); second = torch.zeros_like(mean).index_add_(0, group, w[:, None] * z.square())
        sd = (second - mean.square()).clamp(min=1e-8).sqrt()
        if VARIANT == 'struct': mean, sd = torch.zeros_like(mean), torch.zeros_like(sd)
        corr = self.head(torch.cat([mean, sd, globals_], 1)).squeeze(-1); return skip + corr, corr
run.GrainSet = TokSet
run.invariance_audit = lambda *a, **k: {}
import os
BK = os.environ.get('BANK', ''); BF = _repo_path(H, f'tokbank-{BK}-train.npz' if BK else 'tokbank-train.npz'); OD = f'var-tok{BK}-'
if __name__ == '__main__':
    bank, base, matched = run.load_inputs(BF, _repo_path(S9, 'base.npz'))
    for seed in [int(s) for s in sys.argv[2].split(',')]:
        run.CONFIG = {**copy.deepcopy(ORIG), 'seed': seed}; out = _repo_path(H, f'{OD}{VARIANT}', f'seed{seed}'); out.mkdir(parents=True, exist_ok=True)
        run.run_train(bank, base, matched, BF, _repo_path(S9, 'base.npz'), out, 'cpu'); print('done', seed, flush=True)
