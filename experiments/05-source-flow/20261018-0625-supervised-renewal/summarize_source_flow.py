"""Aggregate complete source-local new-representation checks without selection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os, sys, argparse, json
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[k] = '2'
import numpy as np


def main(a):
    d = np.load(_repo_path(HERE, 'data.npz')); dev = d['dev']; ids = d['ids'][dev]; y = d['y'][dev]
    fold = d['fold']; base = d['base083']; rows = {str(v): i for i, v in enumerate(ids)}
    per_seed = {kind: [] for kind in ['flow46', 'flow38']}; provenance = []
    for seed in a.seeds:
        values = {kind: np.full(400, np.nan) for kind in per_seed}
        for k in range(5):
            out = _repo_path(HERE, f'source-h300/proper-original400-f{k}-s{seed}')
            recipe = json.loads((out/'recipe.json').read_text())
            assert not set(recipe['source_indices']) & set(recipe['query'])
            assert np.array_equal(recipe['train'], dev[fold != k]) and np.array_equal(recipe['query'], dev[fold == k])
            for kind, name in [('flow46', 'system-predictions.npz'), ('flow38', 'system38-predictions.npz')]:
                z = np.load(_repo_io(out/name)); ix = np.array([rows[str(v)] for v in z['ids']])
                assert np.array_equal(z['y'], y[ix]) and np.array_equal(z['base083'], base[ix])
                assert np.all(fold[ix] == k); values[kind][ix] = z['system']
            provenance.append(dict(folder=str(out.relative_to(HERE)), encoder_seed=recipe['seed'], fold=k,
                                   auxiliary_sources=len(recipe['source_indices']), query_source_overlap=0,
                                   model_sha256=json.loads((out/'trained.json').read_text())['model_sha256']))
        for kind, v in values.items():
            assert np.isfinite(v).all(); per_seed[kind].append(v)
    predictions = {kind: np.mean(v, 0) for kind, v in per_seed.items()}
    draw = np.random.default_rng(2026101997).integers(400, size=(10000, 400)); old = (base-y)**2
    noise = np.load(_repo_path(HERE, 'diagnostic-pixel-noise.npy'))[dev]; nobj = d['nobj'][dev]
    groups = {'noise_low': noise < 9.5, 'noise_mid': (noise >= 9.5)&(noise < 12), 'noise_high': noise >= 12,
              'target_low': y < 186.1664820705299, 'target_mid': (y >= 186.1664820705299)&(y < 200.4092597154498),
              'target_high': y >= 200.4092597154498, 'morph_coarse': nobj < 223.0833333333,
              'morph_mid': (nobj >= 223.0833333333)&(nobj < 368.3333333333), 'morph_fine': nobj >= 368.3333333333,
              'morph_missing': ~np.isfinite(nobj)}
    result = {}
    for kind, p in predictions.items():
        new = (p-y)**2; boot = np.sqrt(old[draw].mean(1))-np.sqrt(new[draw].mean(1)); keep = np.ones(400, bool)
        keep[np.argsort(old-new)[-10:]] = False
        fs = [dict(fold=k, baseline_rmse=float(old[fold == k].mean()**.5), candidate_rmse=float(new[fold == k].mean()**.5)) for k in range(5)]
        result[kind] = dict(baseline_rmse=float(old.mean()**.5), candidate_rmse=float(new.mean()**.5),
                           gain=float(old.mean()**.5-new.mean()**.5), positive_folds=sum(r['candidate_rmse'] < r['baseline_rmse'] for r in fs),
                           ci95=np.percentile(boot, [2.5, 97.5]).tolist(), top10_positive_removed_gain=float(old[keep].mean()**.5-new[keep].mean()**.5),
                           positive_images=int(np.sum(new < old)), folds=fs,
                           per_encoder_seed_rmse=[float(np.mean((v-y)**2)**.5) for v in per_seed[kind]],
                           diagnostic_groups={name: dict(n=int(mask.sum()), baseline_rmse=float(old[mask].mean()**.5), candidate_rmse=float(new[mask].mean()**.5)) for name, mask in groups.items() if mask.any()})
    result.update(encoder_seeds=a.seeds, rows=400, new_representation_query_source_overlap=0,
                  complete_conditional_incumbent_compared=True, full_system_protocol_passed=False, provenance=provenance,
                  scope='Original400 reused development rows. New auxiliary encoder and its eligibility ranking exclude outer-query originals. Inherited P104/083 representations remain historically global. This is additional conditional evidence, not the predefined all500 complete end-to-end confirmation or a Private probability.')
    tag = '-'.join(map(str, a.seeds)); target = _repo_path(HERE, f'source-h300/proper-summary-s{tag}')
    np.savez(_repo_io(target.with_suffix('.npz')), ids=ids, y=y, fold=fold, base083=base, **predictions)
    target.with_suffix('.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: {name: v[name] for name in ['baseline_rmse', 'candidate_rmse', 'gain', 'positive_folds', 'ci95', 'top10_positive_removed_gain']} for k, v in result.items() if k in predictions}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--seeds', type=int, nargs='+', default=[1811])
    main(p.parse_args())
