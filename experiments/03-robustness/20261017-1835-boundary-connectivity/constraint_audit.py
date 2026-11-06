"""Check consistency of confident local edges and sampled region-pair targets."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import graph as g
import numpy as np,time,sys
def roots(allow):
    parent=np.arange(1024);size=np.ones(1024,int)
    def root(j):
        while parent[j]!=j:parent[j]=parent[parent[j]];j=parent[j]
        return j
    for k in np.flatnonzero(allow):
        a,b=root(g.U[k]),root(g.V[k])
        if a==b:continue
        if size[a]<size[b]:a,b=b,a
        parent[b]=a;size[a]+=size[b]
    return np.array([root(j) for j in range(1024)])
def main():
    z=np.load(_repo_io(c.HERE/'targets.npz'));rows=[];start=time.monotonic()
    for i,id in enumerate(z['ids']):
        ey,ew=z['edge_target'][i],z['edge_weight'][i];py,pw=z['pair_target'][i],z['pair_weight'][i];pair=z['pairs'];pos=roots((ew>0)&(ey>0));possible=roots((ew==0)|(ey>0));negative=(py==0)&(pw>0);positive=(py>0)&(pw>0);nc=negative&(pos[pair[:,0]]==pos[pair[:,1]]);pc=positive&(possible[pair[:,0]]!=possible[pair[:,1]]);rows.append(dict(ID=str(id),negative_pairs=int(negative.sum()),negative_conflicts=int(nc.sum()),positive_pairs=int(positive.sum()),positive_conflicts=int(pc.sum()),unknown_permitted_components=int(len(np.unique(possible)))))
    summary={k:int(sum(v[k] for v in rows)) for k in ['negative_pairs','negative_conflicts','positive_pairs','positive_conflicts']};c.save(_repo_io('target-constraint-audit.json'),dict(summary=summary,rows=rows,seconds=time.monotonic()-start,hardness_used=False,parameters_changed=False,definitions=dict(negative_conflict='Confident different-region pair already joined by a path made entirely of confident same-region local edges.',positive_conflict='Confident same-region pair separated even after allowing every unknown edge and forbidding only confident different-region edges.'),limitation='Only the fixed2048 candidate pairs per source are checked. Consistency does not certify true physical grains.'));print(summary)
if __name__=='__main__':
    sys.addaudithook(c.guard)
    with c.threadpool_limits(limits=2):main()
