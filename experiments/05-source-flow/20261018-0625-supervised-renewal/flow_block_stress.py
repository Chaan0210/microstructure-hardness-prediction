"""Broader matched component check with unchanged splits/seeds/block weights.

The complete incumbent is unavailable on these500-row splits. This is never
reported as a complete incumbent comparison or a pass of the full protocol.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os, sys
HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[k] = '2'
for k in ['TMPDIR', 'MPLCONFIGDIR', 'XDG_CACHE_HOME']:
    os.environ[k] = str(_repo_path(HERE, 'tmp'))
os.environ['HF_HOME'] = str(_repo_path(HERE, 'tmp/huggingface'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
import gc, hashlib, json, time
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits


def main():
    start = time.monotonic()
    while not (_repo_path(HERE, 'semantic-flow-fusion-f0/results.json')).exists():
        if time.monotonic()-start > 14400:
            raise RuntimeError('Earlier native readouts did not complete; stress check not started.')
        time.sleep(5)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    d = np.load(_repo_path(HERE, 'data.npz')); y = d['y']
    f = np.load(_repo_path(ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz'))
    pos = {str(v): i for i, v in enumerate(f['ids'])}; P = f['X'][[pos[str(v)] for v in d['ids']]].astype(float)
    f = np.load(_repo_path(HERE, 'domain-semantics/H300-features.npz')); assert np.array_equal(f['ids'], d['ids'])
    X = dict(control=P, flow46=np.c_[P, f['flow']], flow38=np.c_[P, f['flow'][:, :38]])
    out = _repo_path(HERE, 'flow-block-stress'); out.mkdir(exist_ok=True)
    spec = dict(splits=[2026101801, 2026101802], head_seeds=[2026101811, 2026101823], folds=5,
                fitting='Same original500 rows, T400/Q100. Native local TabPFN3.5 n8, headseed+fold, equal mean. Median imputer/StandardScaler/Ridge100 raw/log, exp inverse.',
                block='.6Tab+.2Ridge+.2logRidge, inherited normalized existing masses fixed before these scores.',
                primary='Report paired Tab component to match prior stress check and full changed-block combination. Do not treat either as complete083.',
                reuse='Prior control and flow46 Tab predictions reused only after exact train/query-index checks. Current immutable train features and H300 cache are the same frozen inputs.',
                scope='Conditional historical auxiliary representations and repeatedly used labels. No test inputs. Full-system confirmation protocol remains unchanged and not passed.',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'recipe.json').write_text(json.dumps(spec, indent=2))
    sys.path.insert(0, str(_repo_path(ROOT, 'experiments/push-300'))); import harness
    pred = {name+'_'+kind: np.zeros((2, 500)) for name in X for kind in ['tab', 'linear', 'block']}
    fold_id = np.zeros((2, 500), int)
    for rep, split_seed in enumerate(spec['splits']):
        for k, (tr, q) in enumerate(KFold(5, shuffle=True, random_state=split_seed).split(P)):
            fold_id[rep, q] = k
            for name, A in X.items():
                ps = []
                for seed in spec['head_seeds']:
                    if name in ['control', 'flow46']:
                        tag = 'control' if name == 'control' else 'flow'
                        path = _repo_path(HERE, f'flow-component-confirmation/{tag}-r{rep}-f{k}-s{seed}.npz')
                    else:
                        path = out/f'{name}-r{rep}-f{k}-s{seed}.npz'
                    if path.exists():
                        z = np.load(_repo_io(path)); assert np.array_equal(z['train'], tr) and np.array_equal(z['query'], q); v = z['prediction']
                    else:
                        assert name == 'flow38'
                        m = harness.tabpfn_new('v3.5', seed+k, 'cpu', n_est=8); m.fit(A[tr], y[tr]); v = m.predict(A[q]); del m; gc.collect()
                        np.savez(_repo_io(path), train=tr, query=q, prediction=v)
                    ps.append(v)
                tab = np.mean(ps, 0); linear = []
                for target in ['raw', 'log']:
                    m = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=100))
                    m.fit(A[tr], y[tr] if target == 'raw' else np.log(y[tr])); v = m.predict(A[q]); linear.append(v if target == 'raw' else np.exp(v))
                lin = np.mean(linear, 0)
                pred[name+'_tab'][rep, q] = tab; pred[name+'_linear'][rep, q] = lin; pred[name+'_block'][rep, q] = .6*tab+.4*lin
            print('FOLD', rep, k, flush=True)
    draw = np.random.default_rng(2026101997).integers(500, size=(10000, 500)); results = {}
    for kind in ['tab', 'block']:
        old = (pred['control_'+kind]-y)**2
        for name in ['flow46', 'flow38']:
            new = (pred[name+'_'+kind]-y)**2
            old_id, new_id = old.mean(0), new.mean(0)
            boot = np.sqrt(old_id[draw].mean(1))-np.sqrt(new_id[draw].mean(1))
            keep = np.ones(500, bool); keep[np.argsort(old_id-new_id)[-10:]] = False
            folds = []
            for rep in range(2):
                for k in range(5):
                    q = fold_id[rep] == k
                    folds.append(dict(repetition=rep, fold=k, gain=float(old[rep, q].mean()**.5-new[rep, q].mean()**.5)))
            results[name+'_'+kind] = dict(control_rmse=float(old.mean()**.5), candidate_rmse=float(new.mean()**.5),
                gain=float(old.mean()**.5-new.mean()**.5), repetition_gains=(np.sqrt(old.mean(1))-np.sqrt(new.mean(1))).tolist(),
                positive_folds=sum(r['gain'] > 0 for r in folds), paired_original_id_ci95=np.percentile(boot, [2.5, 97.5]).tolist(),
                top10_positive_IDs_removed_gain=float(old_id[keep].mean()**.5-new_id[keep].mean()**.5), folds=folds)
    results.update(complete_incumbent_compared=False, full_system_protocol_passed=False,
                   scope='Paired component/block comparisons only. Historical global auxiliary representations, all500labels reused; repeated observations clustered by originalID. No claim of an untouched dataset or Private probability.')
    np.savez(_repo_io(out/'predictions.npz'), ids=d['ids'], y=y, fold=fold_id, **pred)
    (out/'results.json').write_text(json.dumps(results, indent=2)); print('DONE', json.dumps(results), flush=True)


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
