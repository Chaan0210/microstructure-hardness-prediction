"""Raw-image spatial tokens for branch 1: frozen DINOv2 ViT-B/14 (local weights of experiments/reproduction-018-176, public 2023), input as in
that study (256 -> 504 bicubic, ImageNet normalisation), last-layer normed patch tokens 36 x 36 x 768 -> 2x2 mean -> 18 x 18 -> fixed Gaussian
orthonormal projection to 32 dims (seed 20261013, no fitting, no labels) + (y, x) position. Saved in the grain-set bank format with the same
12 structure columns as tokbank-<split>.npz -> tokbank-dino-<split>.npz.  usage: dino_tokens.py"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, torch
from pathlib import Path
H = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; P0 = _RepoPath(__file__).resolve().parents[3]; R = _repo_path(E, 'reproduction-018-176')
sys.path.insert(0, str(R)); sys.path.insert(0, str(_repo_path(R, 'vendor')))
from feature_helpers import dino_input
from dinov2.hub.backbones import dinov2_vitb14
m = dinov2_vitb14(pretrained=False); m.load_state_dict(torch.load(_repo_path(R, 'weights/DINO-B.pth'), map_location='cpu', weights_only=True), strict=True); m.eval().to('mps')
Q, _ = np.linalg.qr(np.random.default_rng(20261013).standard_normal((768, 32))); Q = Q.astype(np.float32)
gy, gx = np.mgrid[0:18, 0:18] / 17.0 - 0.5; pos = np.c_[gy.ravel(), gx.ravel()].astype(np.float32)
for split in ('train', 'test'):
    T = np.load(_repo_path(H, f'tokbank-{split}.npz'), allow_pickle=True); ids = [str(i) for i in T['ids']]; objs = []
    with torch.inference_mode():
        for n, i in enumerate(ids):
            x = torch.from_numpy(dino_input(_repo_path(P0, 'data', split, f'{i}.png'))[None]).to('mps')
            t = m.get_intermediate_layers(x, n=1, norm=True)[0][0].cpu().numpy().reshape(36, 36, 768)
            t = t.reshape(18, 2, 18, 2, 768).mean((1, 3)).reshape(324, 768) @ Q; objs.append(np.c_[t, pos])
            if (n + 1) % 250 == 0: print(split, n + 1, flush=True)
    out = {k: T[k] for k in T.files}; out['objects'] = np.concatenate(objs).astype(np.float64); out['offsets'] = np.arange(0, 324 * len(ids) + 1, 324)
    out['weights'] = np.full(324 * len(ids), 1 / 324.0); out['object_keys'] = np.array([f'd{k}' for k in range(32)] + ['pos_y', 'pos_x'])
    np.savez(_repo_path(H, f'tokbank-dino-{split}.npz'), **out); print(split, 'done', out['objects'].shape, flush=True)
