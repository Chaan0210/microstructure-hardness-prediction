"""Frozen full500 source-encoder inference; no fitting or test-based selection."""
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
for k in ['TMPDIR', 'MPLCONFIGDIR', 'XDG_CACHE_HOME', 'CELLPOSE_LOCAL_MODELS_PATH']:
    os.environ[k] = str(_repo_path(HERE, 'tmp'))
os.environ['HF_HUB_OFFLINE'] = '1'
import argparse, csv, gc, hashlib, json, time, fcntl
import numpy as np
import torch
from PIL import Image
from cellpose.vit import CPSAM
from threadpoolctl import threadpool_limits
from domain_semantics import normalize, sha
from flow38_fast import descriptors


def main(a):
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    assert torch.backends.mps.is_available()
    torch.mps.set_per_process_memory_fraction(.3)
    destination = Path(a.candidate).resolve()
    assert destination.is_relative_to(HERE)
    recipe_path = destination/'frozen-recipe.json'
    recipe = json.loads(recipe_path.read_text())
    assert recipe['status'].startswith('frozen_') and recipe['field_count'] == 38
    assert recipe['auxiliary_source_sha256'] == sha(_repo_path(HERE, 'source_h300_v2.py'))
    assert recipe['descriptor_source_sha256'] == sha(_repo_path(HERE, 'flow38_fast.py'))
    assert recipe['inference_source_sha256'] == sha(Path(__file__))
    source = _repo_path(HERE, 'source-h300', a.encoder_tag)
    trained = json.loads((source/'trained.json').read_text())
    assert trained['complete'] and trained['model_sha256'] == sha(source/'model.pt')
    d = np.load(_repo_path(HERE, 'data.npz'))
    ck = torch.load(_repo_io(source/'model.pt'), map_location='cpu', weights_only=False, mmap=True)
    spec = ck['recipe']
    assert spec['protocol'] == 'full500' and spec['seed'] in recipe['encoder_training_seeds']
    assert spec['train'] == list(range(500)) and spec['query'] == []
    assert spec['train_ids'] == d['ids'].tolist()
    assert spec['source_sha256'] == recipe['auxiliary_source_sha256']
    assert spec['initial_weights_sha256'] == recipe['initial_weights_sha256']
    model = CPSAM()
    state = {k: v for k, v in ck['state'].items() if k not in ['diam_labels', 'diam_mean']}
    status = model.load_state_dict(state, strict=False)
    assert set(status.missing_keys) <= {'diam_labels', 'diam_mean'} and not status.unexpected_keys
    del ck, state; gc.collect(); model.eval().requires_grad_(False).to('mps')
    if a.split == 'train':
        ids = d['ids']; pixels = np.load(_repo_path(HERE, 'train-pixels.npy'), mmap_mode='r')
    else:
        ids = np.array([r['ID'] for r in csv.DictReader((_repo_path(ROOT, 'data/sample_submission.csv')).open())])
        assert len(ids) == len(set(ids)) == 1000 and not set(ids) & set(d['ids'])
        pixels = None
    out = destination/'features'; out.mkdir(exist_ok=True)
    tag = f'{a.encoder_tag}-{a.split}'
    final = out/f'{tag}.npz'; partial = out/f'{tag}-partial.npz'
    contract = dict(model_sha256=trained['model_sha256'], recipe_sha256=sha(recipe_path),
                    inference_source_sha256=sha(Path(__file__)), descriptor_source_sha256=sha(_repo_path(HERE, 'flow38_fast.py')))
    if final.exists():
        existing = json.loads((out/f'{tag}-complete.json').read_text())
        assert all(existing[k] == v for k, v in contract.items()) and existing['feature_sha256'] == sha(final)
        return
    values, elapsed = [], 0.
    if partial.exists():
        z = np.load(_repo_io(partial)); values = list(z['flow']); elapsed = float(z['elapsed'])
        assert np.array_equal(z['ids'], ids[:len(values)])
        assert str(z['model_sha256']) == contract['model_sha256'] and str(z['recipe_sha256']) == contract['recipe_sha256']
    start = time.monotonic()
    with torch.no_grad():
        for j in range(len(values), len(ids)):
            if pixels is not None:
                raw = pixels[j, 0]
            else:
                with Image.open(_repo_path(ROOT, 'data/test', f'{ids[j]}.png')) as image:
                    assert image.mode == 'L' and image.size == (256, 256)
                    raw = np.asarray(image)
            features = []
            for first in [0, 2]:
                x = np.stack([np.ascontiguousarray(np.rot90(normalize(raw), rot)) for rot in [first, first+1]])[:, None]
                flow, _ = model(torch.tensor(x, device='mps')); flow = flow.cpu().numpy()
                for k, rot in enumerate([first, first+1]):
                    f, keys = descriptors(flow[k], np.rot90(raw, rot)); features.append(f)
            value = np.mean(features, 0)
            assert value.shape == (38,) and np.isfinite(value).all() and keys == recipe['flow_keys']
            values.append(value); torch.mps.empty_cache()
            if len(values) % 10 == 0 or len(values) == len(ids):
                np.savez(_repo_io(out/f'{tag}-partial-new.npz'), ids=ids[:len(values)], flow=np.array(values),
                         elapsed=elapsed+time.monotonic()-start, model_sha256=contract['model_sha256'], recipe_sha256=contract['recipe_sha256'])
                (out/f'{tag}-partial-new.npz').replace(partial)
                print('INFERENCE', a.encoder_tag, a.split, len(values), 'seconds', elapsed+time.monotonic()-start, flush=True)
    np.savez(_repo_io(final), ids=ids, flow=np.array(values), flow_keys=np.array(keys))
    report = dict(complete=True, rows=len(ids), split=a.split, seconds=elapsed+time.monotonic()-start,
                  feature_sha256=sha(final), inference_only=True, parameters_updated=0, hardness_labels_read=False, **contract)
    (out/f'{tag}-complete.json').write_text(json.dumps(report, indent=2)); print('DONE', tag, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--candidate', required=True); p.add_argument('--encoder-tag', required=True)
    p.add_argument('--split', choices=['train', 'test'], required=True)
    with (_repo_path(HERE, 'tmp/heavy-learning.lock')).open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        with threadpool_limits(2):
            main(p.parse_args())
