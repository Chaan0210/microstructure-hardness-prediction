"""Finite-window signed third moments on three distinct spatial sites."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
RADII=(1,2,4,8,16,32,64)
def offsets(r,kind,k,flip):
 z=np.array([[0,0],[0,r],[0,2*r] if kind=='line' else [r,0]],dtype=int)
 for _ in range(k):z=np.column_stack([-z[:,1],z[:,0]])
 if flip:z[:,1]*=-1
 return z

def sample(x,z):
 h,w=x.shape;lo=z.min(0);hi=z.max(0);return [x[-lo[0]+dy:h-hi[0]+dy,-lo[1]+dx:w-hi[1]+dx] for dy,dx in z]

def moments(image,radii=RADII):
 a=np.asarray(image,dtype=float)/255.;mu=float(a.mean());x=a-mu;pairs=[];triples=[]
 for r in radii:
  for kind in ('line','corner'):
   pp=[];tt=[]
   for k in range(4):
    for flip in (False,True):
     a,b,c=sample(x,offsets(r,kind,k,flip));pp.append(float((a*b+a*c+b*c).mean()/3));tt.append(float((a*b*c).mean()))
   pairs.extend([np.mean(pp),np.std(pp)]);triples.extend([np.mean(tt),np.std(tt)])
 return np.array([mu,float((x*x).mean()),float((x*x*x).mean())]),np.array(pairs),np.array(triples)
def extract(image):
 a=np.asarray(image);assert a.dtype==np.uint8 and a.shape==(256,256);v=moments(a);assert all(np.isfinite(z).all() for z in v);return v

def independent(image,radii=RADII):
 x=np.asarray(image,dtype=float)/255.;mu=x.mean();x-=mu;h,w=x.shape;yy,xx=np.mgrid[:h,:w];pairs=[];trips=[]
 for r in radii:
  for kind in ('line','corner'):
   ps=[];ts=[]
   for k in range(4):
    for flip in (False,True):
     # Rotate/reflection the image instead of transforming the motif.
     q=np.rot90(x,k);q=np.fliplr(q) if flip else q
     if kind=='line':coords=((0,0),(0,r),(0,2*r))
     else:coords=((0,0),(0,r),(r,0))
     valid=np.ones((h,w),bool)
     for dy,dx in coords:valid&=(yy+dy<h)&(xx+dx<w)
     y0,x0=yy[valid],xx[valid];A,B,C=[q[y0+dy,x0+dx] for dy,dx in coords];ps.append(np.sum(A*B+A*C+B*C)/(3*len(A)));ts.append(np.sum(A*B*C)/len(A))
   pairs.extend([np.mean(ps),np.std(ps)]);trips.extend([np.mean(ts),np.std(ts)])
 return np.array([mu,np.mean(x**2),np.mean(x**3)]),np.array(pairs),np.array(trips)
