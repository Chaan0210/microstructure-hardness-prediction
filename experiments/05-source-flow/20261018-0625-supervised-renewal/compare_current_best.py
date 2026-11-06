"""Matched conditional comparison with the user-submitted global F38 system.

This adds a comparator after its Public result was reported. It does not select
weights, change training recipes, or supersede the original confirmation plan.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent


def main(seeds):
    tag = '-'.join(map(str, seeds))
    path = _repo_path(HERE, f'source-h300/proper-summary-s{tag}.npz')
    source = np.load(_repo_io(path))
    current_path = _repo_path(HERE, 'flow-content-ablation/oof.npz')
    current = np.load(_repo_io(current_path))
    for key in ['ids', 'y', 'fold', 'base083']:
        assert np.array_equal(source[key], current[key]), key
    y, fold = source['y'], source['fold']
    old = (current['learned_fields38_system'] - y) ** 2
    new = (source['flow38'] - y) ** 2
    assert np.isfinite(old).all() and np.isfinite(new).all()
    draw = np.random.default_rng(2026101997).integers(len(y), size=(10000, len(y)))
    boot = np.sqrt(old[draw].mean(1)) - np.sqrt(new[draw].mean(1))
    keep = np.ones(len(y), bool)
    keep[np.argsort(old-new)[-10:]] = False
    folds = [dict(fold=k, current_best_rmse=float(old[fold == k].mean()**.5),
                  source_candidate_rmse=float(new[fold == k].mean()**.5)) for k in range(5)]
    data = np.load(_repo_path(HERE, 'data.npz'))
    noise = np.load(_repo_path(HERE, 'diagnostic-pixel-noise.npy'))[data['dev']]
    nobj = data['nobj'][data['dev']]
    groups = {
        'noise_low': noise < 9.5, 'noise_mid': (noise >= 9.5) & (noise < 12), 'noise_high': noise >= 12,
        'target_low': y < 186.1664820705299,
        'target_mid': (y >= 186.1664820705299) & (y < 200.4092597154498), 'target_high': y >= 200.4092597154498,
        'morph_coarse': nobj < 223.0833333333,
        'morph_mid': (nobj >= 223.0833333333) & (nobj < 368.3333333333), 'morph_fine': nobj >= 368.3333333333,
        'morph_missing': ~np.isfinite(nobj),
    }
    result = dict(
        comparator='candidate-flow38-v1 (global H300 F38)', encoder_seed_suffixes=seeds,
        rows=len(y), current_best_rmse=float(old.mean()**.5), source_candidate_rmse=float(new.mean()**.5),
        gain=float(old.mean()**.5-new.mean()**.5), ci95=np.percentile(boot, [2.5, 97.5]).tolist(),
        positive_folds=sum(v['source_candidate_rmse'] < v['current_best_rmse'] for v in folds),
        positive_images=int(np.sum(new < old)), folds=folds,
        top10_positive_removed_gain=float(old[keep].mean()**.5-new[keep].mean()**.5),
        diagnostic_groups={name:dict(n=int(mask.sum()), current_best_rmse=float(old[mask].mean()**.5),
            source_candidate_rmse=float(new[mask].mean()**.5)) for name,mask in groups.items() if mask.any()},
        prediction_files={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,current_path]},
        full_system_protocol_passed=False, coefficients_fit=False,
        scope='Same original400 rows/folds, 320 labeled training rows per fold, same heads/control/fallback rules. New source encoder excludes validation originals; global-H300 comparator and inherited P104/083 have historical global representation exposure. Reused-data conditional comparison only. ID bootstrap does not account for adaptive model search and is not a Private probability.',
    )
    out = _repo_path(HERE, f'source-h300/current-best-comparison-s{tag}.json')
    assert not out.exists(), 'Keep earlier comparison records immutable.'
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['current_best_rmse','source_candidate_rmse','gain','ci95','positive_folds','top10_positive_removed_gain']},indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    main(p.parse_args().seeds)
