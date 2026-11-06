"""Finite zero-dimensional grayscale persistence: graph connectivity, not grains."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from skimage.morphology import max_tree
from scipy.special import ndtr

def diagram(image):
 """Lower-level 4-neighbor diagram, omit sole essential and zero-length bars."""
 x=np.asarray(image);assert x.ndim==2 and x.dtype==np.uint8 and min(x.shape)>=3
 p,order=max_tree(255-x,connectivity=1);v=(255-x).ravel().astype(np.int32);p=p.ravel();peak=v.copy();pairs=[]
 for i in order[:0:-1]:
  j=p[i];younger=min(peak[i],peak[j]);death=v[j]
  if younger>death:pairs.append((255-younger,255-death))
  peak[j]=max(peak[i],peak[j])
 return np.array(sorted(pairs),dtype=np.int16).reshape(-1,2)

def encode(pd,area):
 if not len(pd):return np.zeros(256),np.zeros(256)
 pd=np.asarray(pd,dtype=float);birth=pd[:,0]/255;life=(pd[:,1]-pd[:,0])/255;edges=np.linspace(0,1,17);sigma=1/16
 # Integrate Gaussian mass in each fixed cell; no renormalization of mass
 # outside unit square. Linear persistence weight downweights short bars.
 B=np.diff(ndtr((edges[None,:]-birth[:,None])/sigma),axis=1);L=np.diff(ndtr((edges[None,:]-life[:,None])/sigma),axis=1)
 PI=(B.T@(life[:,None]*L))/area
 delta=np.bincount(pd[:,0].astype(int),minlength=257)-np.bincount(pd[:,1].astype(int),minlength=257);curve=np.cumsum(delta)[:256]/area
 return curve,PI.ravel()

def extract(image):
 x=np.asarray(image);assert x.shape==(256,256) and x.dtype==np.uint8
 c=[];p=[];diagrams=[]
 for v in (x,255-x):
  pd=diagram(v);curve,pi=encode(pd,x.size);c.append(curve);p.append(pi);diagrams.append(pd)
 return np.concatenate(c),np.concatenate(p),diagrams
