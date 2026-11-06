"""H3 screen evaluation (frozen before results). Blend effect of replacing the slot-044 members (4 variants, nb rows / full rows)
by +MEAN or +RAND versions, each against the matched CPU BASE, on top of strict-082 (primary) and strict-076."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
E = _RepoPath(__file__).resolve().parents[3]
V = ['u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4']
b76 = np.load(_repo_path(E, '076-complement-20261016/baseline076.npz'), allow_pickle=True)
ssl = np.load(_repo_path(E, 'highnoise-ssl-20261016-2312/oof.npz'), allow_pickle=True)
ids = [str(i) for i in b76['ids']]; y = ssl['y']; fold = b76['fold']; nz = b76['noise']; failed = b76['failed']
gate = nz >= 12
p76 = b76['p76']; p82 = ssl['baseline082']


def load(arm):
    mem = []
    for v in V:
        h = {}
        for fs in ('nb', 'full'):
            z = np.load(_repo_path(HERE, f'tab/{arm}-{fs}-{v}.npz'), allow_pickle=True)
            pos = {str(i): j for j, i in enumerate(z['ids'])}
            g = z['g'][[pos[i] for i in ids]]
            assert np.isfinite(g).all(), (arm, fs, v)
            h[fs] = g
        mem.append(np.where(gate, h['full'], h['nb']))
    return np.array(mem)


def stored_044():
    mem = []
    for v in V:
        h = {}
        for fs in ('nb', 'full'):
            z = np.load(_repo_path(E, f'push-1300/abl/dev-h9phys-{fs}-044-{v}.npz'), allow_pickle=True)
            pos = {str(i): j for j, i in enumerate(z['original_ids'])}
            h[fs] = z['original_g'][[pos[i] for i in ids]]
        mem.append(np.where(gate, h['full'], h['nb']))
    return np.array(mem)


def rmse(p, m=None):
    m = np.ones(len(y), bool) if m is None else m
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))


def boot(base, cand, n=10000, seed=0):
    rng = np.random.default_rng(seed); d = (cand - y) ** 2 - (base - y) ** 2
    idx = rng.integers(0, len(y), (n, len(y)))
    gains = np.sqrt(np.mean((base - y)[idx] ** 2, 1)) - np.sqrt(np.mean((cand - y)[idx] ** 2, 1))
    return float(np.mean(gains > 0)), [float(np.quantile(gains, .025)), float(np.quantile(gains, .975))], d


def report(base, cand):
    r = dict(base=rmse(base), cand=rmse(cand)); r['delta'] = r['cand'] - r['base']
    r['bands'] = {b: rmse(cand, m) - rmse(base, m) for b, m in
                  (('clean', nz < 9.5), ('mid', (nz >= 9.5) & (nz < 12)), ('noisy', gate))}
    r['fold_delta'] = [rmse(cand, fold == k) - rmse(base, fold == k) for k in range(5)]
    r['folds_improved'] = int(sum(x < 0 for x in r['fold_delta']))
    pb, ci, d = boot(base, cand)
    r['p_better'] = pb; r['gain_ci95'] = ci
    o = np.argsort(d)  # most helped first
    keep = np.ones(len(y), bool); keep[o[:10]] = False
    r['delta_without_top10_helped'] = rmse(cand, keep) - rmse(base, keep)
    r['top10_share_of_net_gain'] = float(d[o[:10]].sum() / d.sum()) if d.sum() < 0 else None
    return r


if __name__ == '__main__':
    base = load('base'); out = {}
    st = stored_044()
    out['cpu_vs_stored_mps_rms'] = float(np.sqrt(np.mean((base - st) ** 2)))
    for arm in ('mean', 'rand'):
        mem = load(arm)
        dblock = (mem - base).sum(0) / 12.0
        dblock[failed] = 0.0
        out[arm] = {'on082': report(p82, p82 + .384 * dblock), 'on076': report(p76, p76 + .384 * dblock),
                    'member_rmse_gated': [rmse(m, gate) for m in mem], 'member_rmse_nongated': [rmse(m, ~gate) for m in mem]}
    out['base_member_rmse_gated'] = [rmse(m, gate) for m in base]; out['base_member_rmse_nongated'] = [rmse(m, ~gate) for m in base]
    m82 = out['mean']['on082']; r82 = out['rand']['on082']
    out['rule'] = dict(a=m82['delta'] <= -0.010, b=(m82['delta'] - r82['delta']) <= -0.010, c=m82['folds_improved'] >= 3,
                       d=m82['delta_without_top10_helped'] <= 0.0)
    out['advance'] = bool(all(out['rule'].values()))
    json.dump(out, open(_repo_path(HERE, 'screen-result.json'), 'w'), indent=1)
    print(json.dumps({k: out[k] for k in ('cpu_vs_stored_mps_rms', 'rule', 'advance')}, indent=1))
    for arm in ('mean', 'rand'):
        for on in ('on082', 'on076'):
            r = out[arm][on]
            print(f"{arm:5s} {on}: {r['base']:.4f} -> {r['cand']:.4f} (d {r['delta']:+.4f}) bands {({k: round(v, 4) for k, v in r['bands'].items()})} "
                  f"folds {[round(x, 3) for x in r['fold_delta']]} P {r['p_better']:.2f} CI {[round(x, 4) for x in r['gain_ci95']]} no-top10 {r['delta_without_top10_helped']:+.4f}")
    print('member RMSE gated base/mean/rand', np.round(out['base_member_rmse_gated'], 3), np.round(out['mean']['member_rmse_gated'], 3), np.round(out['rand']['member_rmse_gated'], 3))
    print('member RMSE nongated base/mean/rand', np.round(out['base_member_rmse_nongated'], 3), np.round(out['mean']['member_rmse_nongated'], 3), np.round(out['rand']['member_rmse_nongated'], 3))
