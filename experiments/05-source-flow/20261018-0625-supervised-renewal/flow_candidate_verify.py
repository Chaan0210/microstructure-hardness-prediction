"""Inference-only independent feature spot check and dependency verification."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os, sys
HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[3]
OUT = _repo_path(HERE, 'candidate-flow46-v1')
sys.dont_write_bytecode = True
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[k] = '2'
for k in ['TMPDIR', 'MPLCONFIGDIR', 'XDG_CACHE_HOME', 'CELLPOSE_LOCAL_MODELS_PATH']:
    os.environ[k] = str(_repo_path(HERE, 'tmp'))
os.environ['HF_HUB_OFFLINE'] = '1'
import csv, gc, json, time
import numpy as np
import torch
from PIL import Image
from cellpose.vit import CPSAM
from threadpoolctl import threadpool_limits
from domain_semantics import descriptors, sha


def main():
    start = time.monotonic()
    while not (_repo_path(OUT, 'replay-verification.json')).exists():
        if time.monotonic()-start > 14400:
            raise RuntimeError('Candidate replay did not finish; verification remains incomplete.')
        time.sleep(5)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    r = json.loads((_repo_path(OUT, 'frozen-recipe.json')).read_text())
    meta = json.loads((_repo_path(OUT, 'fit-complete.json')).read_text())
    assert meta['recipe_sha256'] == sha(_repo_path(OUT, 'frozen-recipe.json'))
    assert sha(_repo_path(ROOT, r['physical_train'])) == r['physical_train_sha256']
    assert sha(_repo_path(ROOT, r['flow_training_cache'])) == r['flow_cache_sha256']
    context = np.load(_repo_path(OUT, 'training-context.npz'))
    labels = {v['ID']: float(v['hardness']) for v in csv.DictReader((_repo_path(ROOT, 'data/train.csv')).open())}
    assert np.array_equal(context['y'], [labels[str(v)] for v in context['ids']])
    source_flow = np.load(_repo_path(ROOT, r['flow_training_cache']))
    assert np.array_equal(context['ids'], source_flow['ids']) and np.array_equal(context['flow'], source_flow['flow'])
    original = json.loads((_repo_path(ROOT, 'experiments/082-fusion-20261018-0155/protected.json')).read_text())
    selected = [r['flow_model'], r['physical_train'], r['physical_test'], 'experiments/push-300/harness.py']
    foundation = [p for p in original if p.endswith('/tabpfn-v3.5-20260909.safetensors')]
    assert len(foundation) == 1
    selected += foundation
    dependencies = {}
    for name in selected:
        digest = sha(_repo_path(ROOT, name))
        if name in original:
            assert digest == original[name]
        dependencies[name] = dict(sha256=digest, bytes=(_repo_path(ROOT, name)).stat().st_size, matches_prior_protection_manifest=name in original)
    z = np.load(_repo_path(OUT, 'test-flow-features.npz'))
    assert z['flow'].shape == (1000, 46) and np.isfinite(z['flow']).all()
    # Fixed positions chosen before completion of test inference. These rows
    # are solely for reproduction checking, never model selection.
    positions = [0, 499, 999]
    net = CPSAM()
    state = torch.load(_repo_path(ROOT, r['flow_model']), map_location='cpu', weights_only=True, mmap=True)
    status = net.load_state_dict(state, strict=False)
    assert set(status.missing_keys) <= {'diam_labels', 'diam_mean'} and not status.unexpected_keys
    del state; gc.collect(); net.eval().requires_grad_(False)
    capture = {}; handle = net.encoder.neck.register_forward_hook(lambda m, args, v: capture.update(neck=v))
    checks = []
    try:
        with torch.no_grad():
            for j in positions:
                ident = str(z['ids'][j])
                with Image.open(_repo_path(ROOT, 'data/test', f'{ident}.png')) as image:
                    raw = np.asarray(image)
                lo, hi = np.percentile(raw, [1, 99])
                norm = ((raw.astype(np.float32)-float(lo))/max(float(hi-lo), 1.)).astype(np.float32)
                values = []
                for first in [0, 2]:
                    x = np.stack([np.ascontiguousarray(np.rot90(norm, rot)) for rot in [first, first+1]])[:, None]
                    field, _ = net(torch.tensor(x)); neck = capture.pop('neck').numpy(); field = field.numpy()
                    for k, rot in enumerate([first, first+1]):
                        _, value, keys = descriptors(neck[k], field[k], np.rot90(raw, rot)); values.append(value)
                error = float(np.max(np.abs(np.mean(values, 0)-z['flow'][j])))
                assert error < 1e-6 and keys == r['flow_keys']
                checks.append(dict(id=ident, feature_max_absolute_difference=error))
    finally:
        handle.remove()
    with (_repo_path(OUT, 'candidate-flow46-v1.csv')).open() as f:
        reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
    sample = list(csv.DictReader((_repo_path(ROOT, 'data/sample_submission.csv')).open()))
    assert fields == ['ID', 'hardness'] and len(rows) == 1000
    assert [v['ID'] for v in rows] == [v['ID'] for v in sample]
    assert len({v['ID'] for v in rows}) == 1000
    assert np.isfinite([float(v['hardness']) for v in rows]).all()
    replay = json.loads((_repo_path(OUT, 'replay-verification.json')).read_text())
    report = dict(complete=True, fixed_spotchecks=checks, dependencies=dependencies,
                  csv_sha256=sha(_repo_path(OUT, 'candidate-flow46-v1.csv')), rows=1000, schema=fields,
                  saved_state_replay_max_difference=replay['replay_max_difference'],
                  saved_state_replay_byte_identical=replay['csv_byte_identical'],
                  labels_match_original_training_csv=True,
                  training_context_sha256=sha(_repo_path(OUT, 'training-context.npz')),
                  inference_only=True, parameters_fitted=0, no_test_based_selection=True,
                  no_submission_or_selection_change=True, full_system_confirmation_passed=False)
    (_repo_path(OUT, 'final-verification.json')).write_text(json.dumps(report, indent=2))
    print('VERIFIED', json.dumps({k: v for k, v in report.items() if k != 'dependencies'}), flush=True)


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
