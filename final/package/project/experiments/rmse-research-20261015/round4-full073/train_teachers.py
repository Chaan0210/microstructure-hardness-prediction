"""Build the pre-registered label graph, then cache complete non021 teachers."""
import argparse
import json
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import full_branch as b


def graphs(d):
    old=json.loads((b.EXP/'residual-cnn-219/data-contract.json').read_text())
    out=[]
    for k,g in enumerate(old['graphs']):
        a=np.load(b.HERE/f'anchors/fold{k}.npz')
        T,A,B,V=[a[x] for x in ('T','A','B','V')]
        assert np.array_equal(d['dev'][g['A']],A)
        assert set(V)==set(d['dev'][d['fold'][0]==k])
        rng=np.random.default_rng(20261016+k);aa=rng.permutation(A)
        outer_valid=[B,aa[:107],aa[107:]]
        nodes=[dict(name='outer0',train=A.tolist(),query=B.tolist())]
        for j,cell in enumerate(g['cells']):
            C,D=d['dev'][cell['C']],d['dev'][cell['D']]
            assert not set(C)&set(D) and set(C)|set(D)==set(A)
            nodes.append(dict(name=f'deep{j}',train=C.tolist(),query=D.tolist()))
        for j,q in enumerate(outer_valid[1:],1):
            train=np.array([i for i in T if i not in set(q)])
            nodes.append(dict(name=f'outer{j}',train=train.tolist(),query=q.tolist()))
        assert sorted(np.concatenate(outer_valid).tolist())==sorted(T.tolist())
        assert sorted(sum([n['query'] for n in nodes if n['name'].startswith('deep')],[]))==sorted(A.tolist())
        for n in nodes:
            assert not set(n['train'])&set(n['query'])
            assert (set(n['train'])|set(n['query']))<=set(T)
        out.append(dict(fold=k,T=T.tolist(),A=A.tolist(),B=B.tolist(),V=V.tolist(),nodes=nodes))
    file=b.HERE/'label-graph.json'
    if file.exists():assert json.loads(file.read_text())==out
    else:b.r.save_json(file,out)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5));p.add_argument('--node')
    args=p.parse_args();torch.set_num_threads(2)
    d,y=b.load();gg=graphs(d)
    lock={str(f.relative_to(b.ROOT)):b.r.digest(f) for f in
          [b.HERE/'PLAN.md',b.HERE/'full_branch.py',b.HERE/'train_teachers.py',b.HERE/'label-graph.json']}
    lock.update(d['source_hashes'])
    path=b.HERE/'teacher-provenance.json'
    if path.exists():assert json.loads(path.read_text())==lock
    else:b.r.save_json(path,lock)
    for g in gg:
        if args.fold is not None and g['fold']!=args.fold:continue
        for node in g['nodes']:
            if args.node and node['name']!=args.node:continue
            train=np.asarray(node['train']);query=np.asarray(node['query'])
            b.fit(d,train,query,y[train].copy(),g['fold'],node['name'])
            print('TEACHER COMPLETE',g['fold'],node['name'],flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
