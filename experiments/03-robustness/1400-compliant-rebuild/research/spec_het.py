"""Segmentation-free, noise-corrected local-scale heterogeneity. Per image: remove smooth shading; DoG band energies at log-spaced scales;
subtract the analytic white-noise energy of each band (noise SD from the Haar MAD estimate); per 4x4 block (64 px) the energy-weighted mean log scale;
het_spec = SD over blocks; also global scale and 8x8 version. Benchmark: measure on compliant degraded copies vs truth from the clean source.
usage: spec_het.py (benchmark on push-1400 copies c1..c4 vs clean sources)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json, csv, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
R = Path(str(_REPO_ROOT)); E = _repo_path(R, 'experiments')
SC = np.array([0.8, 1.2, 1.7, 2.4, 3.4, 4.8, 6.8])
def noise_sd(g):
    g = g.astype(float); hh = (g[0::2, 0::2] - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2]) / 2; return float(np.median(np.abs(hh)) / 0.6745)
_nE = {}
def noise_energy(s):   # energy of DoG(s, 1.6 s) applied to unit white noise (per pixel), by simulation once
    if s not in _nE:
        z = np.random.RandomState(0).normal(0, 1, (512, 512)); b = ndi.gaussian_filter(z, s) - ndi.gaussian_filter(z, 1.6 * s); _nE[s] = float((b[32:-32, 32:-32] ** 2).mean())
    return _nE[s]
def measure(img):
    x = img.astype(float); sd = noise_sd(img); x = x - ndi.gaussian_filter(x, 24)
    Eb = []
    for s in SC:
        b = ndi.gaussian_filter(x, s) - ndi.gaussian_filter(x, 1.6 * s); e = b ** 2 - sd ** 2 * noise_energy(s)
        Eb.append(e)
    Eb = np.array(Eb)   # [nscale, H, W]
    out = []
    for nb in (4, 8):
        s = 256 // nb; blk = Eb.reshape(len(SC), nb, s, nb, s).mean((2, 4)); blk = np.clip(blk, 1e-3, None)
        ls = (blk * np.log(SC)[:, None, None]).sum(0) / blk.sum(0); out += [ls.std(), ls.mean()]
    g = np.clip(Eb.mean((1, 2)), 1e-3, None); out.append(float((g * np.log(SC)).sum() / g.sum())); return out
if __name__ == '__main__':
    ids = [r['ID'] for r in csv.DictReader(open(_repo_path(R, 'data/train.csv')))]
    Fz = np.load(_repo_path(E, 'push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); fk = [str(k) for k in Fz['keys']]; fid = [str(i) for i in Fz['ids']]
    truth_cols = ['sizefield_std_s32', 'het4_u', 'tile_logA_std']
    src = sorted(json.load(open(_repo_path(E, 'push-1400/R/experiments/push-400/synth/train_c1_params.json'))).keys())
    Cl = np.array([measure(np.array(Image.open(_repo_path(R, f'data/train/{i}.png')))) for i in src])
    T = np.array([[Fz['X'][fid.index(i), fk.index(c)] for c in truth_cols] for i in src]).astype(float)
    keys = ['spec_het4', 'spec_mean4', 'spec_het8', 'spec_mean8', 'spec_glob']
    print('validity on CLEAN sources: corr(spec measure on clean, truth on clean)')
    for j, k in enumerate(keys): print(f'  {k:10s}', ' '.join(f'{c}:{np.corrcoef(Cl[:, j], T[:, q])[0, 1]:+.2f}' for q, c in enumerate(truth_cols)))
    for st in ('c1', 'c3'):
        Cp = np.array([measure(np.array(Image.open(_repo_path(E, f'push-1400/R/experiments/push-400/synth/train_{st}/{i}.png')))) for i in src])
        print(f'copies {st}: corr(measure on copy, same measure on clean source) and vs truth')
        for j, k in enumerate(keys): print(f'  {k:10s} self {np.corrcoef(Cp[:, j], Cl[:, j])[0, 1]:+.2f} |', ' '.join(f'{c}:{np.corrcoef(Cp[:, j], T[:, q])[0, 1]:+.2f}' for q, c in enumerate(truth_cols)))
    np.savez(_repo_path(E, 'push-1400/research/spec_bench.npz'), src=np.array(src), clean=Cl, truth=T)
