"""Non-stretch dev gain of noisy-row changes vs 061 (train-only). usage: import nsgain; nsgain.decomp(name, new) or run for the audit variants."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, sys
sys.path.insert(0, _repo_io('../push-900')); from blend_dev import load
A = np.load(_repo_io(str(_repo_path(_REPO_ROOT, 'experiments/03-robustness/1300-robust-ensembles/audit061.npz')))); y, nz, fail, fold, c61 = A['y'], A['nz'], A['fail'], A['fold'], A['c61']
C = {k[2:]: A[k] for k in A.files if k.startswith('C_')}; W = dict(cnn=0.16, tab=0.384, ridge=0.128, logr=0.128, set=0.05, gmm=0.15)
g = (nz >= 12) & ~fail[0]; rm = lambda a: float(np.sqrt(((a - y) ** 2).mean()))
def blend(Cx): return np.where(fail, C['cnn'], sum(W[k] * Cx[k] for k in W))
def decomp(nm, new, base=c61, quiet=False):
    o = new.copy(); bs = []
    for k in range(3):
        x = base[k, g] - base[k, g].mean(); dd = (new - base)[k, g]; b = (x * dd).sum() / (x * x).sum(); bs.append(b); o[k, g] = base[k, g] + dd - b * x
    tot = rm(base) - rm(new); info = rm(base) - rm(o)
    dm = ((o - y) ** 2 - (base - y) ** 2)[:, g].mean(0); se = dm.std() / np.sqrt(g.sum())
    pp = np.sqrt(((base - y) ** 2).mean(1)) - np.sqrt(((o - y) ** 2).mean(1))
    if not quiet: print(f'{nm:28s} gain {tot:+.4f} = stretch {tot - info:+.4f} + NON-STRETCH {info:+.4f} (parts {"/".join(f"{v:+.3f}" for v in pp)}; noisy dMSE {dm.mean():+.2f} SE {se:.2f}) coef {np.mean(bs):+.3f}')
    return info, o
if __name__ == '__main__':
    B = load(3); R = np.load(_repo_io('ridge_audit.npz'))
    for nm in ('eqw', 'grp', 'eqw+grp'):
        Cx = dict(C); Cx['ridge'] = np.where(g, R[f'{nm}_ridge'], B['ridge']); Cx['logr'] = np.where(g, R[f'{nm}_logridge'], B['logridge']); decomp(f'partners {nm}', blend(Cx))
    T = np.load(_repo_io('tt-a1.npz'))
    for s in ('ridge', 'tab', 'tabmix'):
        for comp in ('tab', 'logr'):
            Cx = dict(C); Cx[comp] = np.where(g, T[s], C[comp]); decomp(f'TT {s} replaces {comp}', blend(Cx))
