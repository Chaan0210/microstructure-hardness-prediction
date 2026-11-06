"""Isolated 078 attribution and two controlled residual-CNN experiments."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
import sys
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[3]
R4 = _repo_path(ROOT, 'experiments/rmse-research-20261015/round4-full073')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '2')
sys.path.insert(0, str(R4))
sys.path.insert(0, str(R4.parent))
import argparse
import csv
import hashlib
import importlib.util
import json
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import cnn_experiment as e

ARMS = ('high_loss', 'no_teacher_context')
r = e.r


def save(name, value):
    p = _repo_path(HERE, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def protect():
    p = _repo_path(HERE, 'protected.json')
    if p.exists():
        for name, sha in json.loads(p.read_text()).items():
            assert r.digest(_repo_path(ROOT, name)) == sha, name
        return
    paths = [_repo_path(ROOT, f'artifacts/submission-candidate-{k}.csv') for k in ('061', '073', '076', '078', '080')]
    paths += [_repo_path(R4.parent, k) for k in ('inputs.npz', 'scope.npz', 'consistency-strict.npz')]
    paths += list(R4.glob('*.py')) + [_repo_path(R4, 'label-graph.json')]
    paths += list(R4.glob('teachers/*/predictions.npz')) + list((_repo_path(R4, 'anchors')).glob('*.npz'))
    paths += list(R4.glob('aux-seed*/*/model.pt')) + list(R4.glob('aux-seed*/*/predictions.npz'))
    paths += list((_repo_path(R4, 'final500')).glob('models/*/model.pt'))
    paths += [_repo_path(R4, 'final500/recipe.json'), _repo_path(R4, 'final500/test-predictions.npz'),
              _repo_path(R4, 'final500/candidate-aux-noise-addition.csv'), _repo_path(R4, 'noise-calibration/aux-oof.npz')]
    save('protected.json', {str(p.relative_to(ROOT)): r.digest(p) for p in paths})


def read_csv(path, ids):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ['ID', 'hardness']
    assert [row['ID'] for row in rows] == list(ids)
    pred = np.array([float(row['hardness']) for row in rows])
    assert np.isfinite(pred).all() and len(set(ids)) == len(ids)
    return pred


def metrics(y, pred, folds):
    return dict(rmse=float(np.sqrt(np.mean((y-pred)**2))),
                folds=[float(np.sqrt(np.mean((y[folds == k]-pred[folds == k])**2))) for k in range(5)])


def audit():
    protect()
    ids = [row['ID'] for row in csv.DictReader((_repo_path(ROOT, 'data/sample_submission.csv')).open())]
    assert len(ids) == 1000
    ps = {k: read_csv(_repo_path(ROOT, f'artifacts/submission-candidate-{k}.csv'), ids) for k in ('073', '076', '078', '080')}
    z = np.load(_repo_path(R4, 'final500/test-predictions.npz'))
    assert z['ids'].tolist() == ids
    delta = ps['080'] - ps['078']
    original = read_csv(_repo_path(R4, 'final500/candidate-aux-noise-addition.csv'), ids) - ps['073']
    assert np.max(np.abs(delta-original)) < 1e-10
    high = (z['noise'] >= 12) & ~z['failed']
    low_delta = ps['078'] - ps['076']
    assert np.array_equal(delta[~high], np.zeros((~high).sum()))
    assert np.array_equal(low_delta[high], np.zeros(high.sum()))
    assert np.max(np.abs(delta*low_delta)) == 0
    candidate = ps['076'] + delta
    assert np.array_equal(candidate[~high], ps['076'][~high])
    assert np.array_equal(candidate[high], ps['080'][high])
    out = _repo_path(HERE, 'candidate-076-plus-frozen-cnn.csv')
    if out.exists():
        assert np.array_equal(read_csv(out, ids), candidate)
    else:
        with out.open('x', newline='') as handle:
            w = csv.writer(handle); w.writerow(['ID', 'hardness'])
            w.writerows(zip(ids, map(lambda x: repr(float(x)), candidate)))
    d = dict(np.load(_repo_path(R4.parent, 'inputs.npz')))
    dev = d['dev']; y = d['y'][dev]; fold = d['fold'][0]
    q = np.load(_repo_path(R4, 'noise-calibration/aux-oof.npz'))
    assert np.array_equal(q['ids'], d['ids'][dev]) and np.array_equal(q['fold'], fold)
    corr = q['addition'] - q['baseline']
    p76 = np.load(_repo_path(ROOT, 'experiments/push-1300/dev-combo.npy'))[0]
    p78 = np.load(_repo_path(ROOT, 'experiments/push-1300/dev-078.npy'))[0]
    assert p76.shape == (400,) and np.isfinite(p76).all()
    alignment = np.load(_repo_path(ROOT, 'experiments/probabilistic-grains-900/base.npz'), allow_pickle=True)
    assert np.array_equal(alignment['ids'], d['ids'][dev]) and np.array_equal(alignment['fold'], d['fold'])
    assert np.array_equal(p76[d['noise'][dev]>=12], p78[d['noise'][dev]>=12])
    scores = {'076': 8.9646534604, '078': 9.0298958718, '080': 8.999772193}
    for k in ('076', '080'):
        record = json.loads((_repo_path(ROOT, f'reports/candidate{k}-public-submission.json')).read_text())
        assert record['public_rmse'] == scores[k]
        assert record['CSV_sha256'] == r.digest(_repo_path(ROOT, f'artifacts/submission-candidate-{k}.csv'))
    result = dict(public_scores=scores, new078_source='Direct human report in this chat on 2026-10-16; no browser verification',
                  public_contrasts=dict(low_partner_change_078_minus076=scores['078']-scores['076'],
                                        frozen_CNN_080_minus078=scores['080']-scores['078']),
                  disjoint_support=True, changed_high_rows=int(high.sum()),
                  formula='076 + (080 - 078); no new coefficient, threshold, or public inverse fit',
                  new_csv=str(out.relative_to(ROOT)), CSV_sha256=r.digest(out), uploaded=False,
                  correction_rms=float(np.sqrt(np.mean(delta**2))),
                  recorded076=metrics(y, p76, fold), recorded076_plus_frozenCNN=metrics(y, p76+corr, fold),
                  strict073=metrics(y, q['baseline'], fold), strict073_plus_frozenCNN=metrics(y, q['addition'], fold),
                  limitation='076 is historical as-produced OOF, not a fresh strict-source076 refit. No new Public score or independent holdout.',
                  old_failure_claim_corrected='CNN marginal Public effect is favorable;078 low-noise partner change was unfavorable.')
    save('attribution.json', result)
    np.savez_compressed(_repo_path(HERE, 'fixed-composition-oof.npz'), ids=d['ids'][dev], y=y, fold=fold,
                        recorded076=p76, candidate=p76+corr, correction=corr)
    protect()
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def fitter():
    """Reuse exact old fitter; change only optional main-loss weights and empty-query diagnostic."""
    src = _repo_path(R4, 'cnn_fitter.py')
    text = src.read_text()
    old = "main_loss=nn.functional.mse_loss(pred,torch.from_numpy(target[ix]));aux_loss="
    new = "main_loss=((pred-torch.from_numpy(target[ix]))**2*torch.from_numpy(data['main_weights'][ix])).mean();aux_loss="
    assert text.count(old) == 1
    text = text.replace(old, new)
    old = 'float(np.max(np.abs(rp-vp)))'
    assert text.count(old) == 1
    text = text.replace(old, 'float(np.max(np.abs(rp-vp),initial=0.))')
    path = _repo_path(HERE, 'weighted_fitter.py')
    if path.exists(): assert path.read_text() == text
    else: path.write_text(text)
    save('fitter-derivation.json', dict(source=str(src.relative_to(ROOT)), source_sha256=r.digest(src),
                                     derived_sha256=r.digest(path), changes=['optional training-only main loss weights', 'empty-query reload diagnostic']))
    spec = importlib.util.spec_from_file_location('new_weighted_fitter', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def prepare(d, labels, graph, stage):
    k = graph['fold']
    A, B, T, V = [np.asarray(graph[key], int) for key in ('A', 'B', 'T', 'V')]
    if stage == 'selector':
        train, query, kind = A, B, 'deep'
        z = np.load(_repo_path(R4, f'teachers/fold{k}-outer0/predictions.npz'))
        query_base, gap = z['base'], z['tab']-z['ridge']
        anchor = np.load(_repo_path(R4, f'anchors/fold{k}.npz'))
        assert np.array_equal(anchor['B'], B)
        base = .16*anchor['prediction_B']+.84*query_base
    else:
        train, query, kind = T, V, 'outer'
        lookup = {int(v): i for i, v in enumerate(d['dev'])}
        ix = np.array([lookup[int(v)] for v in V])
        base = np.load(_repo_path(R4.parent, 'scope.npz'))['strict_blend'][0, ix]
        query_base = (base-.16*d['C73_cnn'][0, ix])/.84
        gap = d['C73_tab'][0, ix]-np.load(_repo_path(R4.parent, 'consistency-strict.npz'))['production'][0, ix]
    teachers, sources = e.teacher_oof(d, graph, kind, train)
    raw = e.raw_context(d, graph, kind, teachers, query, query_base, gap)
    data = e.cnn_data(d, train, query, labels[train].copy(), teachers, sources, k, stage, raw)
    eligible = np.nan_to_num(d['X'][query, list(d['keys']).index('n_obj')], nan=0) >= 20
    return data, dict(train=train, query=query, base=base, eligible=eligible)


def configure(data, d, arm):
    tr = data['train']
    data['main_weights'] = np.ones(len(tr), np.float32)
    if arm == 'high_loss':
        gate = (d['noise'][tr] >= 12).astype(np.float32)
        assert gate.sum() > 0
        data['main_weights'] = gate/gate.mean()
    elif arm == 'no_teacher_context':
        data['context'] = data['context'].copy()
        data['context'][:, :2] = 0
    else: raise ValueError(arm)
    data['meta'].update(experiment_arm=arm, high_gate=12,
                        main_weight_sha256=hashlib.sha256(data['main_weights'].tobytes()).hexdigest(),
                        primary_high_noise_count=int((d['noise'][tr]>=12).sum()),
                        context_columns_zeroed=[0,1] if arm == 'no_teacher_context' else [])
    return data


def no_test(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        p = Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(_repo_path(ROOT, 'data/test')) or p.name.endswith('-test.npz'):
            raise RuntimeError('Test inputs are forbidden in training: '+str(p))


def train(arms, repeat):
    protect(); fit = fitter(); sys.addaudithook(no_test)
    d = e.c.load_data(); labels = d.pop('y')
    graphs = json.loads((_repo_path(R4, 'label-graph.json')).read_text())
    for k, graph in enumerate(graphs):
        assert graph['fold'] == k
        for arm in arms:
            for stage in ('selector', 'final'):
                path = _repo_path(HERE, 'models', f'{arm}-seed{repeat}', f'fold{k}-{stage}')
                path.parent.mkdir(parents=True, exist_ok=True)
                data, contract = prepare(d, labels, graph, stage)
                data = configure(data, d, arm)
                seed = 2026101501+100*k+100000*repeat
                if (path/'training.json').exists():
                    old = json.loads((path/'training.json').read_text())
                    assert old['meta'] == data['meta'] and old['seed'] == seed
                else:
                    fit.fit_one(data, 'aux', seed, path)
                save(str(path.relative_to(HERE) / 'contract.json'), dict(
                    arm=arm, repeat=repeat, stage=stage, fold=k, seed=seed,
                    train_ids=d['ids'][contract['train']].tolist(), query_ids=d['ids'][contract['query']].tolist(),
                    target_sha256=hashlib.sha256(labels[contract['train']].tobytes()).hexdigest(),
                    training_code_sha256=r.digest(_repo_path(HERE, 'weighted_fitter.py')), study_sha256=r.digest(__file__),
                    plan_sha256=r.digest(_repo_path(HERE, 'PLAN.md')), test_used=False, outer_labels_used=False))
                print('COMPLETE', arm, repeat, k, stage, flush=True)
    protect()
    save(f'training-seed{repeat}-complete.json', dict(arms=list(arms), repeat=repeat, complete=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['audit', 'train'])
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--repeat', type=int, choices=[0,1], default=0)
    a = p.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        audit() if a.mode == 'audit' else train(a.arms, a.repeat)


if __name__ == '__main__': main()
