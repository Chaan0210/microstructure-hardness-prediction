"""Stretch / non-stretch decomposition of dev gains for every candidate with a Public reading (train-only diagnostic).
stretch component of a noisy-row change d = b * (base - mean(base)) per partition; non-stretch gain = gain of base + (d - stretch)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, glob, numpy as np
src = open(_repo_io('kdiag.py')).read(); src = src[:src.index("u = (c57 - c53)")]; exec(compile(src, 'kdiag', 'exec'))
c59 = c58.copy()   # 059 = 058 with the stretch removed per partition (as built)
for k in range(3):
    x = c53[k, m] - c53[k, m].mean(); dd = (c58 - c53)[k, m]; c59[k, m] = c53[k, m] + dd - (x * dd).sum() / (x * x).sum() * x
c67 = c61 + 0.128 * m * ts
rm = lambda a: float(np.sqrt(((a - y) ** 2).mean()))
def decomp(nm, base, new, pub):
    o = new.copy(); bs = []
    for k in range(3):
        x = base[k, m] - base[k, m].mean(); dd = (new - base)[k, m]; b = (x * dd).sum() / (x * x).sum(); bs.append(b); o[k, m] = base[k, m] + dd - b * x
    tot = rm(base) - rm(new); info = rm(base) - rm(o)
    print(f'{nm:10s} dev gain {tot:+.4f} = stretch {tot - info:+.4f} + non-stretch {info:+.4f} (stretch coef {np.mean(bs):+.3f})   Public {pub}')
decomp('053->057', c53, c57, '9.0381->8.9963 (better)'); decomp('053->058', c53, c58, 'not submitted'); decomp('053->059', c53, c59, '9.0381->9.0592 (worse)')
decomp('053->061', c53, c61, '9.0381->8.9831 (better)'); decomp('061->062', c61, c62, 'n/a'); decomp('061->064', c61, c64, '8.9831->9.0166 (worse)')
decomp('061->067', c61, c67, 'not submitted'); decomp('062->064', c62, c64, 'n/a')
