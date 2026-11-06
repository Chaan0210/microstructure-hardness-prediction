"""Read only explicitly listed development400 caches; no raw CSV/holdout/test reader."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,hashlib
import numpy as np
from contracts import Features,Labels,AnchorGraph,require,sha
ROOT=_RepoPath(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent

def graph_for(k):
 p=json.loads((_repo_path(OUT, 'data-contract.json')).read_text());g=p['graphs'][k];return AnchorGraph(tuple(g['A']),g['offset'],tuple(g['cells']))
def load_development():
 p=json.loads((_repo_path(OUT, 'data-contract.json')).read_text())
 for rel,digest in p['inputs'].items():require(sha(_repo_path(ROOT, rel))==digest,'Data contract hash mismatch: '+rel)
 with np.load(_repo_path(ROOT, 'experiments/cnn-nested-204/cache/data.npz'),allow_pickle=False) as z:
  ids=z['ids'].copy();groups=z['groups'].copy();labels=Labels(np.arange(400),z['y']);arrays={'images':z['images'].copy(),'X':z['X'].copy()}
 for key,rel in p['feature_files'].items():
  with np.load(_repo_path(ROOT, rel),allow_pickle=False) as z:
   require(z['ids'].tolist()==ids.tolist(),'Feature IDs/order mismatch');arrays[key]=z['X'].reshape(400,-1).copy()
 require(arrays['K'].shape==(400,26) and arrays['X'].shape==(400,585) and arrays['L'].shape==(400,9216),'Feature schema mismatch')
 f=Features(ids,groups,arrays,p['blocks'],p['kblocks']);require(ids.tolist()==p['ids'],'Development IDs changed');return f,labels,p
