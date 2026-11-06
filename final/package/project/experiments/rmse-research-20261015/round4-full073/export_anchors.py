"""Read-only export of existing complete021 A-only anchors; no training."""
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'experiments/residual-cnn-219'
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(HERE.parent))
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from data import load_development
import research as r


def main():
    torch.set_num_threads(2)
    features, _, contract = load_development()
    d = dict(np.load(HERE.parent / 'inputs.npz'))
    assert np.array_equal(features.ids, d['ids'][d['dev']])
    out = HERE / 'anchors'
    out.mkdir(exist_ok=True)
    records = []
    for k, graph in enumerate(contract['graphs']):
        A, B, T, V = [np.asarray(graph[x], int) for x in ('A','B','T','V')]
        assert set(A).isdisjoint(B) and set(A)|set(B)==set(T)
        assert set(T).isdisjoint(V) and set(T)|set(V)==set(range(400))
        assert set(V)==set(np.flatnonzero(d['fold'][0]==k))
        path = OLD / f'main-20261007/fold-{k}/frozen-anchor.joblib'
        binding = json.loads(path.with_suffix('.json').read_text())
        assert r.digest(path)==binding['anchor_sha256']
        anchor = joblib.load(path)
        assert list(anchor.train)==A.tolist()==binding['train']
        assert set(anchor.root.train)==set(A)==set(anchor.short.train)==set(anchor.L.train)
        assert anchor.record['labels_only_A']
        for cell, block in zip(anchor.record['cells'], anchor.residuals):
            C,D = map(set,(cell['C'],cell['D']))
            assert C.isdisjoint(D) and C|D==set(A) and set(block.train)==D
            for split in block.record['splits']:
                u,v = map(set,(split['train'],split['validation']))
                assert u.isdisjoint(v) and u|v==D
        pB = anchor.predict(features, B)
        pV = anchor.predict(features, V)
        old = np.load(path.parent/'predictions.npz')
        assert np.array_equal(old['V'],V)
        error = float(np.max(np.abs(old['R0']-pV)))
        assert error<.002, error
        target = out / f'fold{k}.npz'
        if target.exists():
            saved=np.load(target);assert np.array_equal(saved['prediction_B'],pB)
        else:
            np.savez_compressed(target, A=d['dev'][A],B=d['dev'][B],T=d['dev'][T],V=d['dev'][V],
                                prediction_B=pB, train_ids=features.ids[A],validation_ids=features.ids[B])
        record=dict(fold=k,anchor_path=str(path.relative_to(ROOT)),sha256=r.digest(path),
                    max_reload_error=error,all_nested_labels_only_A=True,
                    target_path=str(target.relative_to(HERE)),target_sha256=r.digest(target))
        records.append(record)
        print(json.dumps(record),flush=True)
    r.save_json(out/'verification.json',dict(contract_sha256=r.digest(OLD/'data-contract.json'),folds=records))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
