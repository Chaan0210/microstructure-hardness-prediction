"""Exact sampled maximin critical edges on a32x32 undirected grid."""
import numpy as np
def grid(n=32):
    nodes=np.arange(n*n).reshape(n,n)
    return np.r_[nodes[:,:-1].ravel(),nodes[:-1,:].ravel()],np.r_[nodes[:,1:].ravel(),nodes[1:,:].ravel()]
U,V=grid()
def critical_edges(scores,pairs,n=32):
    u,v=grid(n);scores=np.asarray(scores);pairs=np.asarray(pairs,int);assert len(scores)==len(u) and np.isfinite(scores).all();assert pairs.ndim==2 and pairs.shape[1]==2 and np.all(pairs[:,0]!=pairs[:,1])
    parent=np.arange(n*n);size=np.ones(n*n,int);pending=[set() for _ in parent];out=np.full(len(pairs),-1,int)
    for j,(x,y) in enumerate(pairs):pending[x].add(j);pending[y].add(j)
    def root(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    remaining=len(pairs)
    # Stable edge index breaks ties without inventing an affinity perturbation.
    for edge in np.lexsort((np.arange(len(scores)),-scores)):
        x,y=root(u[edge]),root(v[edge])
        if x==y:continue
        if size[x]<size[y]:x,y=y,x
        shared=pending[x]&pending[y]
        for j in shared:out[j]=edge
        remaining-=len(shared);pending[x]^=pending[y];pending[y]=set();parent[y]=x;size[x]+=size[y]
        if remaining==0:break
    assert np.all(out>=0)
    return out
def signature(boundary,edges,pairs,radii):
    b=np.asarray(boundary)[2:-2,2:-2].ravel();q=np.clip(np.asarray(edges),1e-7,1-1e-7);critical=critical_edges(q,pairs);con=q[critical]
    z=[b.mean(),b.std(),*np.quantile(b,[.1,.5,.9,.95]),np.mean(b>.5),np.mean(-(np.clip(b,1e-7,1-1e-7)*np.log(np.clip(b,1e-7,1-1e-7))+(1-np.clip(b,1e-7,1-1e-7))*np.log(1-np.clip(b,1e-7,1-1e-7))))]
    z.extend([q.mean(),q.std(),*np.quantile(q,[.1,.5,.9]),np.mean(-(q*np.log(q)+(1-q)*np.log(1-q)))])
    for radius in range(4):
        value=con[radii==radius];z.extend([value.mean(),value.std()])
    for threshold in [.3,.5,.7]:
        parent=np.arange(1024);size=np.ones(1024,int)
        def root(x):
            while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
            return x
        for j in np.flatnonzero(q>=threshold):
            x,y=root(U[j]),root(V[j])
            if x==y:continue
            if size[x]<size[y]:x,y=y,x
            parent[y]=x;size[x]+=size[y]
        z.extend([np.sum(parent==np.arange(1024))/1024,size[parent==np.arange(1024)].max()/1024])
    z=np.asarray(z,float);assert z.shape==(28,) and np.isfinite(z).all();return z
def fixed_pairs(seed=2026101784,count=128):
    rng=np.random.default_rng(seed);pairs=[];radii=[]
    for j,r in enumerate([1,2,4,8]):
        for _ in range(count):
            dy,dx=[(0,r),(r,0),(r,r),(r,-r)][int(rng.integers(4))];y=int(rng.integers(max(0,-dy),min(32,32-dy)));x=int(rng.integers(max(0,-dx),min(32,32-dx)));pairs.append([32*y+x,32*(y+dy)+x+dx]);radii.append(j)
    return np.asarray(pairs,int),np.asarray(radii,int)
