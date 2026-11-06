"""Fixed finite-window centered binary pair and triangle statistics, native pixels."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
QUANTILES=(.2,.5,.8);RADII=(1,2,4,8,16,32)
def transforms(z):
    z=np.asarray(z,dtype=int)
    for k in range(4):
        a=z.copy()
        for _ in range(k):a=np.column_stack((-a[:,1],a[:,0]))
        for flip in (False,True):
            b=a.copy()
            if flip:b[:,1]*=-1
            yield b

def triangles(radii=RADII):
    return [(r,t,np.array([[0,0],[r,0],[0,t*r]])) for r in radii for t in (1,2)]

def pair_orbits(radii=RADII):
    found=set()
    for _,_,z in triangles(radii):
        for i,j in ((0,1),(0,2),(1,2)):found.add(tuple(sorted(abs(z[i]-z[j]))))
    return tuple(sorted(found,key=lambda z:(z[0]**2+z[1]**2,z)))
ORBITS=pair_orbits()
PAIR_NAMES=tuple(f'binary_q{int(q*100)}_pair_{a}_{b}_{m}' for q in QUANTILES for a,b in ORBITS for m in ('mean','std'))
TRIPLE_NAMES=tuple(f'binary_q{int(q*100)}_triple_r{r}_aspect{t}_{m}' for q in QUANTILES for r,t,_ in triangles() for m in ('mean','std'))

def samples(x,z):
    h,w=x.shape;lo=z.min(0);hi=z.max(0);assert h>hi[0]-lo[0] and w>hi[1]-lo[1]
    return [x[-lo[0]+a:h-hi[0]+a,-lo[1]+b:w-hi[1]+b] for a,b in z]

def centered(mask,radii=RADII):
    mask=np.asarray(mask,dtype=bool);p=float(mask.mean());x=mask.astype(float)-p;pair=[];triple=[]
    for a,b in pair_orbits(radii):
        vals=[]
        for z in transforms(np.array([[0,0],[a,b]])):
            A,B=samples(x,z);vals.append(float(np.mean(A*B)))
        pair.extend((float(np.mean(vals)),float(np.std(vals))))
    for _,_,base in triangles(radii):
        vals=[]
        for z in transforms(base):
            A,B,C=samples(x,z);vals.append(float(np.mean(A*B*C)))
        triple.extend((float(np.mean(vals)),float(np.std(vals))))
    return np.array(pair),np.array(triple)

def independent(mask,radii=RADII):
    # Explicit coordinate gather after transforming the image, not motif/slice reuse.
    mask=np.asarray(mask,dtype=bool);x=mask.astype(float)-mask.mean();h,w=x.shape;pair=[];triple=[]
    for bank,bases in [(pair,[np.array([[0,0],[a,b]]) for a,b in pair_orbits(radii)]),(triple,[z for _,_,z in triangles(radii)])]:
        for base in bases:
            vals=[]
            for k in range(4):
                for flip in (False,True):
                    a=np.rot90(x,k);a=np.fliplr(a) if flip else a;yy,xx=np.mgrid[:h-int(base[:,0].max()),:w-int(base[:,1].max())];prod=np.ones_like(yy,dtype=float)
                    for dy,dx in base:prod*=a[yy+dy,xx+dx]
                    vals.append(float(prod.sum()/prod.size))
            bank.extend((np.mean(vals),np.std(vals)))
    return np.array(pair),np.array(triple)

def extract(image):
    image=np.asarray(image);assert image.shape==(256,256) and image.dtype==np.uint8;thresholds=np.quantile(image,QUANTILES,method='linear');P=[];T=[];fractions=[]
    for threshold in thresholds:
        mask=image<=threshold;p,t=centered(mask);P.extend(p);T.extend(t);fractions.append(float(mask.mean()))
    P=np.array(P);T=np.array(T);assert P.shape==(114,) and T.shape==(72,) and np.isfinite(P).all() and np.isfinite(T).all()
    return P,T,{'thresholds':thresholds.tolist(),'fractions':fractions}

def scaling(X,blocks):
    # Each bank retains its own normalization when the second bank is appended.
    X=np.asarray(X,float);blocks=np.asarray(blocks);mean=X.mean(0);std=X.std(0);scale=np.ones(X.shape[1]);floors={};weights=np.ones_like(scale)
    for b in sorted(set(blocks.tolist())):
        ix=blocks==b;floor=max(float(np.sqrt(np.mean(std[ix]**2))),1e-12);floors[b]=floor;scale[ix]=np.maximum(std[ix],floor);weights[ix]=1/np.sqrt(ix.sum())
    return {'mean':mean,'scale':scale,'std':std,'floors':floors,'weights':weights}

def transform(X,s):return (np.asarray(X,float)-s['mean'])/s['scale']*s['weights']
