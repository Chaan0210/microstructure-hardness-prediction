from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os, sys
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[k] = '2'
for k in ['TMPDIR', 'MPLCONFIGDIR', 'XDG_CACHE_HOME', 'CELLPOSE_LOCAL_MODELS_PATH']:
    os.environ[k] = str(_repo_path(HERE, 'tmp'))
os.environ['HF_HUB_OFFLINE'] = '1'
import json, time, resource
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import domain_semantics as old
from flow38_fast import descriptors


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    d = np.load(_repo_path(HERE, 'data.npz')); px = np.load(_repo_path(HERE, 'train-pixels.npy'), mmap_mode='r')
    ids = d['dev'][d['fold'] != 0][:3]; model = old.model('H300', 'cpu'); capture = {}
    hook = model.encoder.neck.register_forward_hook(lambda m, args, value: capture.update(neck=value))
    checks = []; old_time = 0.; new_time = 0.
    with torch.no_grad():
        for j in ids:
            for first in [0, 2]:
                x = np.stack([np.ascontiguousarray(np.rot90(old.normalize(px[j, 0]), k)) for k in [first, first+1]])[:, None]
                flow, _ = model(torch.tensor(x)); neck = capture.pop('neck').numpy(); flow = flow.numpy()
                for m, rot in enumerate([first, first+1]):
                    image = np.rot90(px[j, 0], rot)
                    start = time.monotonic(); _, expected, names = old.descriptors(neck[m], flow[m], image); old_time += time.monotonic()-start
                    start = time.monotonic(); actual, keys = descriptors(flow[m], image); new_time += time.monotonic()-start
                    assert keys == names[:38] and np.array_equal(actual, expected[:38])
                    checks.append(dict(id=str(d['ids'][j]),rotation=rot,max_difference=float(abs(actual-expected[:38]).max())))
    hook.remove()
    result = dict(original_training_images=3, views_checked=len(checks), all_features_bit_identical=True,
                  original_descriptor_seconds=old_time, flow38_only_descriptor_seconds=new_time,
                  optimizer_updates=0, test_inputs_read=False, checks=checks,
                  source_sha256=old.sha(_repo_path(HERE, 'flow38_fast.py')), original_source_sha256=old.sha(_repo_path(HERE, 'domain_semantics.py')),
                  peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (_repo_path(HERE, 'flow38-fast-verification.json')).write_text(json.dumps(result, indent=2))
    print('FAST_DESCRIPTOR_VERIFIED', json.dumps({k: v for k, v in result.items() if k != 'checks'}), flush=True)


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
