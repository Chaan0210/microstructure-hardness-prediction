"""H3 confirmation evaluation (written before the screen result). Full 12-member TabPFN block with +im_mean vs the stored production
block, on three partitions, assembled on 082-structure bases; plus the 400->100 holdout block check.
Bases: original partition = strict-082 (primary) and strict-076; partitions 1/2 = reproduced 076 dev OOF (stepchange c14b-orig) +
the fixed retest correction (0.5468 x mean of two seeds on gated rows), exactly as rescnn-retest. Candidate = base + 0.384 x dBlock
(non-failed rows). dBlock = (sum over members of new - base member)/12, where the base member is the CPU re-run when available
(device-matched) and otherwise the stored MPS prediction (device bias reported from slot 044)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
E = _RepoPath(__file__).resolve().parents[3]
V = ['u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4']; SLOTS = ['038', '039', '044']
PARTS = ['original', '20261031', '20261101']
inp = np.load(_repo_path(E, 'rmse-research-20261015/inputs.npz'), allow_pickle=True)
ids0 = [str(i) for i in inp['ids'][inp['dev']]]
nz = inp['noise'][inp['dev']]; gate = nz >= 12
y = inp['y'][inp['dev']]
B = np.load(_repo_path(E, 'stepchange-20261016/repro/components/c14b-orig.npz'))
assert np.allclose(B['y'], y)
W = 0.5467904360508202
R4 = _repo_path(E, 'rmse-research-20261015/round4-full073'); RT = _repo_path(E, 'rescnn-retest-20261017')
dev = list(inp['dev'])


def correction(p):
    c = np.full(400, np.nan)
    for k in range(5):
        if p == 0:
            zs = [np.load(_repo_path(R4, f'aux-seed{j}/fold{k}-outer.npz')) for j in (0, 1)]; ix = zs[0]['dev_index']
        else:
            zs = [np.load(_repo_path(RT, f'cnn/p{p}-fold{k}-seed{j}-outer.npz')) for j in (0, 1)]; ix = np.array([dev.index(i) for i in zs[0]['V']])
        c[ix] = np.mean([z['residual'] for z in zs], 0)
    g = gate & ~B['fail_dev'][p]
    return np.where(g, W * c, 0.0)


def member(arm, fs, slot, v, part):
    p = _repo_path(HERE, f'tab/{arm}-{fs}-{slot}-{v}-{part}.npz')
    if not p.exists() and arm == 'base' and slot == '044' and part == 'original':
        p = _repo_path(E, f'infogain-20261017/h3/tab/base-{fs}-{v}.npz')
    if not p.exists() and arm == 'mean' and slot == '044' and part == 'original':
        p = _repo_path(E, f'infogain-20261017/h3/tab/mean-{fs}-{v}.npz')
    if p.exists():
        z = np.load(_repo_io(p), allow_pickle=True); pos = {str(i): j for j, i in enumerate(z['ids'])}
        return z['g'][[pos[i] for i in ids0]], 'cpu'
    assert arm == 'base'
    tag = f'h9phys-{fs}-044-{v}' if slot == '044' else f'h9-{fs}-{slot}-{v}'
    z = np.load(_repo_path(E, f'push-1300/abl/dev-{tag}.npz'), allow_pickle=True)
    pos = {str(i): j for j, i in enumerate(z[f'{part}_ids'])}
    return z[f'{part}_g'][[pos[i] for i in ids0]], 'stored_mps'


def dblock(part):
    tot = np.zeros(400); src = []
    for s in SLOTS:
        for v in V:
            h = {}
            for fs in ('nb', 'full'):
                new, _ = member('mean', fs, s, v, part); old, how = member('base', fs, s, v, part); src.append(how)
                h[fs] = new - old
            tot += np.where(gate, h['full'], h['nb'])
    return tot / 12.0, src


def rmse(p, m=None):
    m = np.ones(len(y), bool) if m is None else m
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))


def compare(base, cand, fold, rng):
    d = (cand - y) ** 2 - (base - y) ** 2; o = np.argsort(d)
    idx = rng.integers(0, 400, (10000, 400))
    gains = np.sqrt(np.mean((base - y)[idx] ** 2, 1)) - np.sqrt(np.mean((cand - y)[idx] ** 2, 1))
    keep = np.ones(400, bool); keep[o[:10]] = False
    return dict(base=rmse(base), cand=rmse(cand), delta=rmse(cand) - rmse(base), p_better=float(np.mean(gains > 0)),
                ci95=[float(np.quantile(gains, .025)), float(np.quantile(gains, .975))],
                fold_delta=[rmse(cand, fold == k) - rmse(base, fold == k) for k in range(5)],
                bands={b: rmse(cand, m) - rmse(base, m) for b, m in (('clean', nz < 9.5), ('mid', (nz >= 9.5) & (nz < 12)), ('noisy', gate))},
                delta_without_top10=rmse(cand, keep) - rmse(base, keep), d=d)


if __name__ == '__main__':
    rng = np.random.default_rng(20261017); out = {}; ds = []
    ssl = np.load(_repo_path(E, 'highnoise-ssl-20261016-2312/oof.npz'), allow_pickle=True)
    b76s = np.load(_repo_path(E, '076-complement-20261016/baseline076.npz'), allow_pickle=True)
    for p, part in enumerate(PARTS):
        db, src = dblock(part); fail = B['fail_dev'][p]; db[fail] = 0.0
        fold = inp['fold'][p]
        base = B['dev'][p] + correction(p)
        r = compare(base, base + .384 * db, fold, rng); ds.append(r.pop('d')); r['base_member_source'] = sorted(set(src)); out[part] = r
        if p == 0:
            for nm, bb in (('strict082', ssl['baseline082']), ('strict076', b76s['p76'])):
                rr = compare(bb, bb + .384 * db, fold, rng); rr.pop('d'); out['original_' + nm] = rr
    dm = np.mean(ds, 0)
    out['pooled_dMSE_per_image'] = dict(mean=float(dm.mean()), se=float(dm.std(ddof=1) / np.sqrt(400)),
                                        p_better=float(np.mean([dm[rng.integers(0, 400, 400)].mean() < 0 for _ in range(10000)])))
    json.dump(out, open(_repo_path(HERE, 'confirm-result.json'), 'w'), indent=1)
    for k, r in out.items():
        if 'delta' in r:
            print(f"{k:22s} {r['base']:.4f} -> {r['cand']:.4f} d {r['delta']:+.4f} P {r['p_better']:.2f} CI {np.round(r['ci95'], 4)} folds {np.round(r['fold_delta'], 3)} "
                  f"bands { {b: round(v, 4) for b, v in r['bands'].items()} } no-top10 {r['delta_without_top10']:+.4f}")
    print('pooled per-image dMSE', out['pooled_dMSE_per_image'])
