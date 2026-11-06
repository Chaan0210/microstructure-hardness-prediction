"""What does the 082 residual CNN use? Re-run the saved round4 aux CNNs (seeds 0/1, outer refits) on manipulated validation
images with contexts held fixed. Diagnostic only: reads round4 artifacts (no bytecode, no writes there), writes diag/cnn-probe.json."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
R4 = _repo_path(HERE.parents[1], 'rmse-research-20261015/round4-full073')
sys.path.insert(0, str(R4))
import cnn_experiment as ce  # noqa: E402
c = ce.c
torch.set_num_threads(2)

d = c.load_data(); labels = d.pop('y')
gg = json.loads((_repo_path(R4, 'label-graph.json')).read_text())
baseline = np.load(_repo_path(R4.parent, 'scope.npz'))['strict_blend'][0]
legacy = d['C73_cnn'][0]; non021 = (baseline - .16 * legacy) / .84
strict_ridge = np.load(_repo_path(R4.parent, 'consistency-strict.npz'))['production'][0]
dev = list(d['dev'])
rng = np.random.default_rng(20261017)


def shuffle_patches(x, p):
    out = x.copy()
    for n in range(len(x)):
        tiles = [x[n, 0, i:i + p, j:j + p] for i in range(0, 256, p) for j in range(0, 256, p)]
        order = rng.permutation(len(tiles)); k = 0
        for i in range(0, 256, p):
            for j in range(0, 256, p):
                out[n, 0, i:i + p, j:j + p] = tiles[order[k]]; k += 1
    return out


def phase_random(x):
    out = x.copy()
    for n in range(len(x)):
        F = np.fft.fft2(x[n, 0]); ph = np.angle(np.fft.fft2(rng.standard_normal((256, 256))))
        out[n, 0] = np.real(np.fft.ifft2(np.abs(F) * np.exp(1j * (np.angle(F) * 0 + ph)))) + 0
        out[n, 0] += x[n, 0].mean() - out[n, 0].mean()
    return out.astype(np.float32)


def standardize(x):
    m = x.mean((2, 3), keepdims=True); s = x.std((2, 3), keepdims=True)
    return ((x - m) / s * x.std() + x.mean()).astype(np.float32)


MANIP = {
    'orig': lambda x: x,
    'shuffle32': lambda x: shuffle_patches(x, 32),
    'shuffle16': lambda x: shuffle_patches(x, 16),
    'shuffle8': lambda x: shuffle_patches(x, 8),
    'blur1': lambda x: ndi.gaussian_filter(x, (0, 0, 1, 1)).astype(np.float32),
    'blur2': lambda x: ndi.gaussian_filter(x, (0, 0, 2, 2)).astype(np.float32),
    'phase_random': phase_random,
    'standardize': standardize,
}
res = {m: [] for m in MANIP}; meta = []
for g in gg:
    k = g['fold']; T, V = [np.asarray(g[key], int) for key in ('T', 'V')]
    outer_base, outer_sources = ce.teacher_oof(d, g, 'outer', T)
    ix = np.array([dev.index(i) for i in V]); failed = d['fail'][0, ix]
    raw = ce.raw_context(d, g, 'outer', outer_base, V, non021[ix], d['C73_tab'][0, ix] - strict_ridge[ix])
    final = ce.cnn_data(d, T, V, labels[T].copy(), outer_base, outer_sources, k, 'outer_refit', raw)
    ys = final['meta']['target_std']
    out = {}
    for seed in (0, 1):
        path = _repo_path(R4, f'aux-seed{seed}/fold{k}-final')
        model = c.ComplementCNN(); model.load_state_dict(torch.load(_repo_io(path / 'model.pt'), map_location='cpu', weights_only=True)['state_dict'])
        stored = np.load(_repo_io(path / 'predictions.npz'))['validation_prediction']
        imgs = final['images'].copy()
        for name, fn in MANIP.items():
            x = imgs.copy(); x[V] = fn(imgs[V])
            pred = c.predict(model, x, final['context'], V, 'aux') * ys
            if name == 'orig':
                assert np.max(np.abs(pred - stored)) < 1e-6, (k, seed, np.max(np.abs(pred - stored)))
            out.setdefault(name, []).append(pred)
    lam = float(np.load(_repo_path(R4, f'aux-seed0/fold{k}-outer.npz'))['lambda_add'])
    gate = (d['noise'][V] >= 12) & ~failed
    for name in MANIP:
        r_ = np.mean(out[name], 0); r_[failed] = 0.
        res[name].append(dict(fold=k, ids=d['ids'][V].tolist(), corr=r_.tolist(), gate=gate.tolist()))
    meta.append(dict(fold=k, lam=lam))
    print('fold', k, 'done', flush=True)

y = labels; b82 = None
ssl = np.load(_repo_path(HERE.parents[1], 'highnoise-ssl-20261016-2312/oof.npz'), allow_pickle=True)
b76 = np.load(_repo_path(HERE.parents[1], '076-complement-20261016/baseline076.npz'), allow_pickle=True)
pos = {str(i): j for j, i in enumerate(ssl['ids'])}
summary = {}
orig = np.concatenate([np.array(r['corr']) for r in res['orig']]); gate = np.concatenate([np.array(r['gate']) for r in res['orig']])
idl = sum([r['ids'] for r in res['orig']], []); jj = np.array([pos[i] for i in idl])
p76 = b76['p76'][jj]; yy = ssl['y'][jj]
lamv = np.concatenate([[m['lam']] * len(r['ids']) for m, r in zip(meta, res['orig'])])
for name in MANIP:
    cc = np.concatenate([np.array(r['corr']) for r in res[name]])
    p = p76 + np.where(gate, lamv * cc, 0)
    e0 = (p76 - yy)[gate] ** 2; e = (p - yy)[gate] ** 2
    summary[name] = dict(corr_with_orig=float(np.corrcoef(cc[gate], orig[gate])[0, 1]) if name != 'orig' else 1.0,
                         rms_ratio=float(np.sqrt(np.mean(cc[gate] ** 2)) / np.sqrt(np.mean(orig[gate] ** 2))),
                         gated_dMSE_vs_076=float(np.mean(e - e0)),
                         corr_with_resid76=float(np.corrcoef(cc[gate], (yy - p76)[gate])[0, 1]))
json.dump(dict(summary=summary, lambdas=meta), open(_repo_path(HERE, 'cnn-probe.json'), 'w'), indent=1)
for k_, v in summary.items():
    print('%-13s corr_w_orig %.3f rms_ratio %.2f gated dMSE vs 076 %+.2f corr_w_resid %.3f' % (k_, v['corr_with_orig'], v['rms_ratio'], v['gated_dMSE_vs_076'], v['corr_with_resid76']))
