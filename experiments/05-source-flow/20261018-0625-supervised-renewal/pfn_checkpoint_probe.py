"""Numerical parity and memory test on source-training rows; zero updates."""
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
for k in ['TMPDIR', 'MPLCONFIGDIR', 'XDG_CACHE_HOME']:
    os.environ[k] = str(_repo_path(HERE, 'tmp'))
os.environ['HF_HOME'] = str(_repo_path(HERE, 'tmp/huggingface'))
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
import fcntl, gc, json, time, resource, subprocess
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
import pfn_checkpoint as pf
import pfn_feature_learning as fl
from flow_neural_measurement import Measurement, load


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(2026101931)
    d, X = pf.load(); z = np.load(_repo_path(pf.ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz'))
    pos = {str(v): j for j, v in enumerate(z['ids'])}; X = z['X'][[pos[str(v)] for v in d['ids']]].astype(float)
    z = np.load(_repo_path(HERE, 'domain-semantics/H300-features.npz')); X = np.c_[X, z['flow'][:, :38]]
    outer = d['dev'][d['fold'] != 0]; A, _ = train_test_split(outer, test_size=1/3, random_state=2026101933); A = np.sort(A)
    norm = fl.Features().fit(X[A]); data = load(d); model = Measurement(data, norm.transform(X), A)
    indices = A[:32]; labels = torch.tensor(d['y'][indices], dtype=torch.float32)
    values, gradients = [], []
    for flag in [False, True]:
        reg = pf.make(0); x = model(indices).detach().clone().requires_grad_(True)
        p = pf.forward(reg, x[:24], labels[:24], x[24:], recompute=flag)
        ((p-labels[24:])/20).square().mean().backward()
        values.append(p.detach().numpy()); gradients.append(x.grad.detach().numpy()); del reg, x, p; gc.collect()
    delta = float(np.max(np.abs(values[0]-values[1])))
    relative = float(np.linalg.norm(gradients[0]-gradients[1])/max(np.linalg.norm(gradients[0]), 1e-12))
    assert delta < 1e-5 and relative < 1e-4, (delta, relative)
    reg = pf.make(0); ix = np.random.default_rng(2026101931).permutation(len(A)); c, q = ix[:64], ix[64:80]
    target = torch.tensor(d['y'][A], dtype=torch.float32); start = time.monotonic(); Z = model(A[np.r_[c, q]])
    p = pf.forward(reg, Z[:64], target[c], Z[64:]); loss = ((p-target[q])/20).square().mean()+.01*model.penalty; loss.backward()
    grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)); assert np.isfinite(grad) and grad > 0
    rss = int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(os.getpid())], text=True).strip())*1024
    result = dict(output_max_difference=delta, input_gradient_relative_difference=relative,
                  context=64, queries=16, feature_columns=174, benchmark_seconds=time.monotonic()-start,
                  peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, current_rss=rss,
                  parameter_gradient_norm=grad, optimizer_updates=0, no_test_inputs=True,
                  source_training_ids=d['ids'][A[np.r_[c, q]]].tolist(), mechanism='Installed PerformanceOptions(force_recompute_layer=True), package files unchanged.')
    (_repo_path(HERE, 'pfn-checkpoint-probe.json')).write_text(json.dumps(result, indent=2))
    print('CHECKPOINT_PROBE', json.dumps({k: v for k, v in result.items() if k != 'source_training_ids'}), flush=True)


if __name__ == '__main__':
    with (_repo_path(HERE, 'tmp/heavy-learning.lock')).open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        with threadpool_limits(2):
            main()
