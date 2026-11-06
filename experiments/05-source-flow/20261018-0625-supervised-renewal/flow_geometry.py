"""Continuous grain geometry at several spatial scales, before thresholding.

These are features of original training images, not rendered training images.
No labels or test data enter this extractor. The H300 ancestry remains global,
so downstream development comparisons are explicitly conditional.
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
for k in ['TMPDIR', 'MPLCONFIGDIR', 'XDG_CACHE_HOME', 'CELLPOSE_LOCAL_MODELS_PATH']:
    os.environ[k] = str(_repo_path(HERE, 'tmp'))
os.environ['HF_HUB_OFFLINE'] = '1'
import argparse, gc, json, time, hashlib, resource
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.special import expit
from cellpose.vit import CPSAM
from threadpoolctl import threadpool_limits
from domain_semantics import normalize, descriptors, sha


def geometry(flow, image):
    fy, fx = flow[:2].astype(np.float64) / 5
    speed = np.sqrt(fy * fy + fx * fx + .01)
    uy, ux = fy / speed, fx / speed
    p = expit(np.clip(flow[2].astype(np.float64), -20, 20))
    im = image.astype(np.float64) / 255
    lo, hi = np.percentile(im, [10, 90])
    gray = np.clip((gaussian_filter(im, 1) - im.mean()) / max((hi-lo)/2.563, .01), -8, 8)
    values, keys = [], []
    for sigma in [1, 2, 4, 8, 16, 32]:
        w = gaussian_filter(p, sigma) + .01
        y = gaussian_filter(p * uy, sigma) / w
        x = gaussian_filter(p * ux, sigma) / w
        coherence = x * x + y * y
        yy, yx = np.gradient(y)
        xy, xx = np.gradient(x)
        div, curl = xx + yy, xy - yx
        a, b = xx - yy, xy + yx
        strain = np.sqrt(a*a+b*b) + 1e-8
        g = gaussian_filter(gray, sigma)
        fields = {
            'coherence_mean': coherence.mean(),
            'coherence_sd': coherence.std(),
            'coherence_gray': np.mean(coherence * g),
            'coherence_gray2': np.mean(coherence * g*g),
            'div_rms': np.sqrt(np.mean(div*div)),
            'curl_rms': np.sqrt(np.mean(curl*curl)),
            'strain_c4': np.mean((a*a-b*b)/strain),
            'strain_s4': np.mean(2*a*b/strain),
            'confidence_sd': w.std(),
            'speed_sd': gaussian_filter(speed, sigma).std(),
            'gray_flux': np.mean(np.gradient(g)[0]*y + np.gradient(g)[1]*x),
        }
        keys.extend(f'{k}_s{sigma}' for k in fields)
        values.extend(fields.values())
    return np.array(values, np.float32), keys


def main(a):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if a.device == 'mps':
        assert torch.backends.mps.is_available()
        torch.mps.set_per_process_memory_fraction(.3)
    d = np.load(_repo_path(HERE, 'data.npz'))
    px = np.load(_repo_path(HERE, 'train-pixels.npy'), mmap_mode='r')
    out = _repo_path(HERE, 'flow-geometry')
    out.mkdir(exist_ok=True)
    weights = _repo_path(ROOT, 'experiments/push-300/ft-H300.pt')
    spec = dict(scope='Original training images only. H300 global auxiliary ancestry; conditional comparisons.',
                weights_sha256=sha(weights), source_sha256=sha(Path(__file__)),
                views=[0, 90, 180, 270], device=a.device, descriptor_count=66,
                hypothesis='Hard masks and marginal flow quantiles discard spatial coherence and phase-size covariance. Confidence-weighted normalized flow coherence over six scales measures these continuously.',
                training=False, test_inputs_read=False)
    name = 'probe' if a.probe else 'full'
    path = out / f'{name}-recipe.json'
    if path.exists():
        assert json.loads(path.read_text()) == spec
    else:
        path.write_text(json.dumps(spec, indent=2))
    net = CPSAM()
    state = torch.load(_repo_io(weights), map_location='cpu', weights_only=True, mmap=True)
    status = net.load_state_dict(state, strict=False)
    assert set(status.missing_keys) <= {'diam_labels', 'diam_mean'} and not status.unexpected_keys
    del state
    gc.collect()
    net.eval().requires_grad_(False).to(a.device)
    capture = {}
    handle = net.encoder.neck.register_forward_hook(lambda m, args, v: capture.update(neck=v))
    inds = d['dev'][d['fold'] != 0][:a.probe] if a.probe else np.arange(500)
    vals, old, elapsed = [], [], 0.
    partial = out / 'partial.npz'
    maps = None
    if not a.probe:
        maps = np.lib.format.open_memmap(out / 'flowmaps-c4.npy', mode='r+' if (out/'flowmaps-c4.npy').exists() else 'w+', dtype=np.float16, shape=(500, 4, 3, 128, 128))
        if partial.exists():
            z = np.load(_repo_io(partial))
            assert np.array_equal(z['ids'], d['ids'][:len(z['geometry'])])
            vals, old, elapsed = list(z['geometry']), list(z['flow46']), float(z['elapsed'])
    start = time.monotonic()
    try:
        with torch.no_grad():
            for j in inds[len(vals):]:
                gs, fs = [], []
                for first in [0, 2]:
                    x = np.stack([np.ascontiguousarray(np.rot90(normalize(px[j, 0]), rot)) for rot in [first, first+1]])[:, None]
                    f, _ = net(torch.tensor(x, device=a.device))
                    n = capture.pop('neck').cpu().numpy()
                    f = f.cpu().numpy()
                    for m, rot in enumerate([first, first+1]):
                        im = np.rot90(px[j, 0], rot)
                        g, keys = geometry(f[m], im)
                        _, v, _ = descriptors(n[m], f[m], im)
                        gs.append(g); fs.append(v)
                        if maps is not None:
                            maps[j, rot] = f[m].reshape(3, 128, 2, 128, 2).mean((2, 4))
                vals.append(np.mean(gs, 0)); old.append(np.mean(fs, 0))
                assert np.isfinite(vals[-1]).all() and np.isfinite(old[-1]).all()
                if a.device == 'mps':
                    torch.mps.synchronize(); torch.mps.empty_cache()
                if len(vals) % 10 == 0 or a.probe:
                    total = elapsed + time.monotonic()-start
                    print('GEOMETRY', len(vals), 'seconds', total, flush=True)
                    if not a.probe:
                        maps.flush()
                        np.savez(_repo_io(out/'partial-new.npz'), ids=d['ids'][:len(vals)], geometry=np.array(vals), flow46=np.array(old), elapsed=total)
                        (out/'partial-new.npz').replace(partial)
    finally:
        handle.remove()
    np.savez(_repo_io(out/f'{name}-features.npz'), ids=d['ids'][inds], geometry=np.array(vals), flow46=np.array(old), geometry_keys=np.array(keys))
    ref = np.load(_repo_path(HERE, 'domain-semantics/H300-features.npz'))['flow'][inds]
    scale = np.maximum(np.std(np.load(_repo_path(HERE, 'domain-semantics/H300-features.npz'))['flow'][d['dev'][d['fold'] != 0]], 0), 1e-4)
    err = float(np.sqrt(np.mean(((np.array(old)-ref)/scale)**2)))
    result = dict(rows=len(vals), seconds=elapsed+time.monotonic()-start, cpu_reference_scaled_rms=err,
                  peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  mps_driver=torch.mps.driver_allocated_memory() if a.device == 'mps' else 0)
    assert err < .005
    (out/f'{name}-done.json').write_text(json.dumps(result, indent=2))
    print('DONE', json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    ap.add_argument('--probe', type=int, default=0)
    with threadpool_limits(2):
        main(ap.parse_args())
