"""Robust train-only report for a noisy-row change vs a base (default 061): plain dev + partitions, non-stretch gain, label-matched (LM) noisy dMSE,
stripe-uniform (SU) noisy dMSE, improved stripes, bootstrap P(better). usage: from robust_eval import report; report(name, new, base=None)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, nsgain as N
y, g, c61, nz = N.y, N.g, N.c61, N.nz; lw = np.load(_repo_io(str(_repo_path(_REPO_ROOT, 'experiments/03-robustness/1300-robust-ensembles/lm_weights.npy'))))
sw = np.load(_repo_io(str(_repo_path(_REPO_ROOT, 'experiments/03-robustness/1300-robust-ensembles/su_weights.npy')))); nzr = np.round(nz, 1)
def report(nm, new, base=None):
    base = c61 if base is None else base; info, _ = N.decomp('', new, base=base, quiet=True); d = (new - y) ** 2 - (base - y) ** 2
    dl = (lw * d)[:, g].mean(0); ds = d[:, g].mean(0); su = (sw[g] * ds).sum() / sw[g].sum(); st = sorted(set(nzr[g])); imp = sum(ds[nzr[g] == s].mean() < 0 for s in st)
    rng = np.random.RandomState(0); n = g.sum(); bs = [dl[rng.randint(0, n, n)].mean() for _ in range(3000)]
    pp = np.sqrt(((base - y) ** 2).mean(1)) - np.sqrt(((new - y) ** 2).mean(1))
    print(f'{nm:30s} plain {N.rm(base) - N.rm(new):+.4f} ({"/".join(f"{v:+.3f}" for v in pp)}) | non-stretch {info:+.4f} | LM {dl.mean():+.2f} (SE {dl.std() / np.sqrt(n):.2f}, P {np.mean(np.array(bs) < 0):.2f}) | SU {su:+.2f} | stripes {imp}/{len(st)}')
def partners(tag, d='../push-1200'):
    from blend_dev import load
    B = load(3); S = np.load(_repo_io(f'{d}/sr-{tag}.npz')); Cx = dict(N.C); Cx['ridge'] = np.where(g, S['ridge'], B['ridge']); Cx['logr'] = np.where(g, S['logridge'], B['logridge']); return N.blend(Cx)
