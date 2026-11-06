"""C4 inference on provided train originals after a source-local auxiliary fit."""
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
import argparse, gc, json, time, resource
import numpy as np
import torch
from cellpose.vit import CPSAM
from threadpoolctl import threadpool_limits
from domain_semantics import normalize, descriptors, sha
from flow_geometry import geometry


def main(a):
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    assert torch.backends.mps.is_available()
    torch.mps.set_per_process_memory_fraction(.3)
    out = _repo_path(HERE, 'source-h300', a.tag)
    trained = json.loads((out/'trained.json').read_text())
    assert trained['complete'] and trained['model_sha256'] == sha(out/'model.pt')
    if (out/'flow-complete.json').exists():
        return
    d = np.load(_repo_path(HERE, 'data.npz')); px = np.load(_repo_path(HERE, 'train-pixels.npy'), mmap_mode='r')
    ck = torch.load(_repo_io(out/'model.pt'), map_location='cpu', weights_only=False, mmap=True)
    r = ck['recipe']; assert not set(r['source_indices']) & set(r['query'])
    indices = np.unique(np.r_[r['train'], r['query']]).astype(int)
    ids = d['ids'][indices]
    net = CPSAM()
    state = {k: v for k, v in ck['state'].items() if k not in ['diam_labels', 'diam_mean']}
    status = net.load_state_dict(state, strict=False)
    assert set(status.missing_keys) <= {'diam_labels', 'diam_mean'} and not status.unexpected_keys
    del ck, state; gc.collect()
    net.eval().requires_grad_(False).to('mps')
    capture = {}; handle = net.encoder.neck.register_forward_hook(lambda m, args, v: capture.update(neck=v))
    values, geos, elapsed = [], [], 0.
    part = out/'flow-v2-partial.npz'
    map_path = out/'flowmaps-c4.npy'
    if part.exists():
        assert map_path.exists(), 'A partial descriptor cache requires its matching latent-map cache.'
    maps = np.lib.format.open_memmap(map_path, mode='r+' if map_path.exists() else 'w+',
                                    dtype=np.float16, shape=(len(indices), 4, 3, 128, 128))
    if part.exists():
        z = np.load(_repo_io(part)); values = list(z['flow']); geos = list(z['geometry']); elapsed = float(z['elapsed'])
        assert np.array_equal(z['ids'], ids[:len(values)]) and str(z['model_sha256']) == trained['model_sha256']
    start = time.monotonic()
    try:
        with torch.no_grad():
            for j in indices[len(values):]:
                row = len(values)
                vv, gg = [], []
                for first in [0, 2]:
                    x = np.stack([np.ascontiguousarray(np.rot90(normalize(px[j, 0]), rot)) for rot in [first, first+1]])[:, None]
                    f, _ = net(torch.tensor(x, device='mps')); n = capture.pop('neck').cpu().numpy(); f = f.cpu().numpy()
                    for m, rot in enumerate([first, first+1]):
                        im = np.rot90(px[j, 0], rot)
                        _, v, keys = descriptors(n[m], f[m], im)
                        g, gkeys = geometry(f[m], im)
                        vv.append(v); gg.append(g)
                        maps[row, rot] = f[m].reshape(3, 128, 2, 128, 2).mean((2, 4))
                values.append(np.mean(vv, 0)); geos.append(np.mean(gg, 0))
                assert np.isfinite(values[-1]).all() and np.isfinite(geos[-1]).all()
                torch.mps.synchronize(); torch.mps.empty_cache()
                if len(values) % 10 == 0 or len(values) == len(indices):
                    maps.flush()
                    np.savez(_repo_io(out/'flow-v2-partial-new.npz'), ids=ids[:len(values)], flow=np.array(values), geometry=np.array(geos), elapsed=elapsed+time.monotonic()-start, model_sha256=trained['model_sha256'])
                    (out/'flow-v2-partial-new.npz').replace(part)
                    print('SOURCE_FLOW', len(values), 'seconds', elapsed+time.monotonic()-start, flush=True)
    finally:
        handle.remove()
    dest = out/'flow-features.npz'
    np.savez(_repo_io(dest), ids=ids, indices=indices, flow=np.array(values), geometry=np.array(geos), flow_keys=np.array(keys), geometry_keys=np.array(gkeys))
    result = dict(complete=True, rows=len(ids), seconds=elapsed+time.monotonic()-start, feature_sha256=sha(dest),
                  latent_maps_sha256=sha(map_path), latent_map_shape=list(maps.shape),
                  model_sha256=trained['model_sha256'], source_sha256=sha(Path(__file__)), device='mps', inference_only=True, test_inputs_read=False)
    (out/'flow-complete.json').write_text(json.dumps(result, indent=2)); print('DONE', json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True)
    with threadpool_limits(2):
        main(p.parse_args())
