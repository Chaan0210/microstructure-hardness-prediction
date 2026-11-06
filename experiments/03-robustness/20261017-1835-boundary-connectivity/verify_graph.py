"""Independent widest-path, gradient, missing-label and D4 geometry checks."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import graph as g
import targets as t
import encoder as e
import numpy as np,json,sys,time
import torch
def floyd(scores,n):
    u,v=g.grid(n);a=np.full((n*n,n*n),-np.inf);a[u,v]=scores;a[v,u]=scores;np.fill_diagonal(a,np.inf)
    for k in range(n*n):a=np.maximum(a,np.minimum(a[:,k,None],a[None,k,:]))
    return a
def main():
    rng=np.random.default_rng(17);records=[];start=time.monotonic()
    for kind in ['random','ties','bridge']:
        n=5;u,v=g.grid(n);scores=rng.normal(size=len(u)) if kind=='random' else np.full(len(u),.3)
        if kind=='bridge':scores[(u%5==1)&(v==u+1)]=-.7;scores[np.flatnonzero((u%5==1)&(v==u+1))[2]]=.8
        pairs=np.array([(x,y) for x in range(n*n) for y in range(x+1,n*n)]);j=g.critical_edges(scores,pairs,n);a=floyd(scores,n);error=float(np.max(abs(scores[j]-a[pairs[:,0],pairs[:,1]])));assert error==0.;records.append(dict(fixture=kind,pairs=len(pairs),maximum_error=error))
    scores=rng.normal(size=40);pairs=np.array([[0,24],[2,22],[4,20],[1,17]]);ix=g.critical_edges(scores,pairs,5);x=torch.tensor(scores,dtype=torch.float64,requires_grad=True);y=torch.tensor([1.,0.,1.,0.],dtype=torch.float64);loss=torch.nn.functional.binary_cross_entropy_with_logits(x[ix],y);loss.backward();autograd=x.grad.numpy();errors=[]
    def value(z):
        ids=g.critical_edges(z,pairs,5);v=z[ids];return float(np.mean(np.logaddexp(0,v)-y.numpy()*v))
    for j in range(40):
        hi=scores.copy();lo=scores.copy();hi[j]+=1e-6;lo[j]-=1e-6;numeric=(value(hi)-value(lo))/2e-6;errors.append(abs(numeric-autograd[j]))
    assert max(errors)<1e-8 and np.all(autograd[np.setdiff1d(np.arange(40),ix)]==0)
    missing=np.zeros((4,1024),np.uint16);py,pw=t.target_pairs(missing,np.array([[0,1],[2,3]]),4);assert np.all(pw==0)
    nodes=np.ones((4,1024),np.uint16);nodes[2:,1]=2;_,pw=t.target_pairs(nodes,np.array([[0,1]]),4);assert pw[0]==0
    z=dict(np.load(_repo_io(c.HERE/'targets.npz')));idx=0;id=str(z['ids'][idx]);get,_=t.sources();mask=get(id);original,_=t.make(z['pixels'][idx,0],mask);d4=[]
    for rot in range(4):
        for flip in [False,True]:
            changed,_=t.make(e.transform(z['pixels'][idx,0],rot,flip),[e.transform(m,rot,flip) for m in mask]);expect=e.transform(original['nodes'].reshape(4,32,32),rot,flip).reshape(4,1024);assert np.array_equal(changed['nodes'],expect);assert np.array_equal(changed['boundary'],e.transform(original['boundary'],rot,flip));err=float(np.max(abs(changed['boundary_weight']-e.transform(original['boundary_weight'],rot,flip))));assert err<1e-6;mapping=e.edge_mapping(rot,flip);y,w=t.target_pairs(changed['nodes'],np.c_[g.U,g.V],len(mask));assert np.array_equal(y,original['edge_target'][mapping]) and np.array_equal(w,original['edge_weight'][mapping]);d4.append(dict(rotation=rot,flip=flip,weight_error=err))
    c.save(_repo_io('graph-verification.json'),dict(all_passed=True,widest_path_fixtures=records,independent_gradient_max_error=max(errors),noncritical_gradients_zero=True,missing_not_negative=True,split_votes_unknown=True,D4_source_checks=d4,source=id,fixtures_used_for_training=False,source_sha256=c.sha(__file__),seconds=time.monotonic()-start));print('GRAPH CHECKS PASSED',max(errors),flush=True)
if __name__=='__main__':
    sys.addaudithook(c.guard)
    with c.threadpool_limits(limits=2):main()
