"""Exact positive-length ordinary H1 of 2D top-cell sublevel cubical filtration.
Each pixel is a CLOSED square, edges/vertices enter at min incident pixel value.
Foreground corner contacts are connected (8), complement uses edge adjacency (4).
Nonperiodic rectangle in the plane; exterior component never produces a hole.
Dual max-tree is a computational method, NOT using unpadded H0 as H1.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from skimage.morphology import max_tree
from scipy.special import ndtr


def diagram(image):
    x=np.asarray(image)
    if x.ndim!=2 or x.dtype!=np.uint8: raise ValueError('2D uint8 required')
    # 256 is an immortal connected EXTERIOR, not an added primal foreground ring.
    v=np.pad(x.astype(np.int16),1,constant_values=256)
    p,order=max_tree(v,connectivity=1); values=v.ravel(); p=p.ravel()
    peak=values.copy(); pairs=[]
    for i in order[:0:-1]:
        j=p[i]; young=min(int(peak[i]),int(peak[j])); saddle=int(values[j])
        if young>saddle: pairs.append((saddle,young))
        peak[j]=max(peak[i],peak[j])
    ans=np.array(sorted(pairs),dtype=np.int16).reshape(-1,2)
    assert not len(ans) or (ans.min()>=0 and ans.max()<=255 and (ans[:,0]<ans[:,1]).all())
    return ans


def dual_union_find(image,tie_seed=0):
    """Independent pixel UF with explicit exterior vertex; randomized equal-value order."""
    x=np.asarray(image); h,w=x.shape; n=x.size; exterior=n
    values=np.r_[x.ravel().astype(int),256]; parent=np.arange(n+1); birth=values.copy(); active=np.zeros(n+1,bool);active[n]=True
    rng=np.random.default_rng(tie_seed); order=np.lexsort((rng.random(n),-values[:n]));pairs=[]
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in order:
        i=int(i);r,c=divmod(i,w);active[i]=True;near=[]
        if r:near.append(i-w)
        if r+1<h:near.append(i+w)
        if c:near.append(i-1)
        if c+1<w:near.append(i+1)
        if r in (0,h-1) or c in (0,w-1):near.append(exterior)
        for j in near:
            if not active[j]:continue
            a,b=root(i),root(j)
            if a==b:continue
            if (birth[a],-a)<(birth[b],-b):a,b=b,a
            if birth[b]>values[i]:pairs.append((int(values[i]),int(birth[b])))
            parent[b]=a
    return np.array(sorted(pairs),dtype=np.int16).reshape(-1,2)


def boundary_matrix(image,tie_seed=0):
    """Independent exhaustive cubical boundary reduction over F2 for SMALL fixtures.
    Returns positive intervals in each dimension and essential dimensions.
    No duality/max_tree/digital Euler logic used.
    """
    x=np.asarray(image);h,w=x.shape;val={};bound={};dim={}
    def add(key,t,faces,d):
        val[key]=min(val.get(key,256),int(t));bound[key]=faces;dim[key]=d
    for r in range(h):
        for c in range(w):
            t=x[r,c]
            vs=[('v',r,c),('v',r,c+1),('v',r+1,c),('v',r+1,c+1)]
            for v in vs:add(v,t,[],0)
            es=[('h',r,c),('h',r+1,c),('e',r,c),('e',r,c+1)]
            for k,faces in zip(es,[[vs[0],vs[1]],[vs[2],vs[3]],[vs[0],vs[2]],[vs[1],vs[3]]]):add(k,t,faces,1)
            add(('s',r,c),t,es,2)
    rng=np.random.default_rng(tie_seed); tie={k:rng.random() for k in val}
    order=sorted(val,key=lambda k:(val[k],dim[k],tie[k]));idx={k:i for i,k in enumerate(order)};pivots={};reduced={};zeros=set();killed=set();bars={0:[],1:[],2:[]}
    for j,k in enumerate(order):
        col={idx[f] for f in bound[k]};assert all(i<j for i in col)
        while col and max(col) in pivots:col^=reduced[pivots[max(col)]]
        if col:
            i=max(col);pivots[i]=j;reduced[j]=col;killed.add(i);birth=order[i]
            assert i in zeros and dim[k]==dim[birth]+1
            if val[birth]<val[k]:bars[dim[birth]].append((val[birth],val[k]))
        else:zeros.add(j)
    return {d:np.array(sorted(v),dtype=np.int16).reshape(-1,2) for d,v in bars.items()},sorted(dim[order[i]] for i in zeros-killed)


def encode(pd,area):
    """Fixed 4x4 birth/lifetime PI, sigma=1/4, linear lifetime/area weight.
    Exact CDF integration of Gaussian cells; diagram exact, PI lossy summary.
    No threshold/lifetime cutoff/denoising/fitted transform.
    """
    if not len(pd):return np.zeros(16)
    a=np.asarray(pd,float);b=a[:,0]/255;l=(a[:,1]-a[:,0])/255;edges=np.linspace(0,1,5)
    B=np.diff(ndtr((edges[None,:]-b[:,None])/.25),axis=1)
    L=np.diff(ndtr((edges[None,:]-l[:,None])/.25),axis=1)
    return ((B.T@(l[:,None]*L))/area).ravel()


def extract(image):
    return np.concatenate([encode(diagram(x),x.size) for x in (image,255-image)])
