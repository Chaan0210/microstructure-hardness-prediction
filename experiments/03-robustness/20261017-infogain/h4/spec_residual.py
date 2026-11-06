"""H4: explicit spectral replacement of the residual-CNN correction (pre-registered in PLAN.md, 15:12).
Same targets/context/protocol as round4's aux CNN; image model = RidgeCV on h3/spec-train.npz (23 cols) + 6 context cols (arm S)
or context only (arm C). Reads round4 / rescnn-retest / anchors read-only (no bytecode). Writes h4/result.json, h4/preds.npz."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
E = _RepoPath(__file__).resolve().parents[3]
R4 = _repo_path(E, 'rmse-research-20261015/round4-full073'); RT = _repo_path(E, 'rescnn-retest-20261017')
sys.path.insert(0, str(R4))
import cnn_experiment as ce  # noqa: E402
c = ce.c
W_CNN = 0.5467904360508202

d = dict(np.load(_repo_path(R4.parent, 'inputs.npz'))); labels = d.pop('y')
keys = list(d['keys']); dev = list(d['dev'])
spec = np.load(_repo_path(E, 'infogain-20261017/h3/spec-train.npz'), allow_pickle=True)
assert [str(i) for i in spec['ids']] == [str(i) for i in d['ids']]
SPEC = spec['X']
baseline = np.load(_repo_path(R4.parent, 'scope.npz'))['strict_blend']; legacy = d['C73_cnn'][0]
non021_0 = (baseline[0] - .16 * legacy) / .84
sridge = np.load(_repo_path(R4.parent, 'consistency-strict.npz'))['production']
failed_all = np.nan_to_num(d['X'][:, keys.index('n_obj')], nan=0) < 20


def model():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))


def design(arm, rows, raw):
    return np.c_[SPEC[rows], raw[rows]] if arm == 'S' else raw[rows]


def lam(r, y, base, elig):
    r, y, base = r[elig], y[elig], base[elig]
    return float(np.clip(r @ (y - base) / max(r @ r, 1e-12), 0, 1))


def ctx(base, gap):
    return np.c_[base, gap, d['noise'], d['X'][:, keys.index('u_ph')],
                 np.log1p(np.maximum(np.nan_to_num(d['X'][:, keys.index('n_obj')], nan=0), 0)), d['X'][:, keys.index('coverage')]]


res = {'original': {}}; store = {}
gg = json.loads((_repo_path(R4, 'label-graph.json')).read_text())
oof = {a: {k: np.full(400, np.nan) for k in ('rep', 'add')} for a in ('S', 'C')}
p76 = np.full(400, np.nan); p82 = np.full(400, np.nan); lams = {'S': [], 'C': []}
for g in gg:
    k = g['fold']; A, B, T, V = [np.asarray(g[x], int) for x in 'ABTV']
    an = np.load(_repo_path(E, f'highnoise-relational-20261016/anchors/fold{k}.npz'))
    assert all(np.array_equal(an[x], np.asarray(g[x])) for x in 'ABTV')
    ib = np.load(_repo_path(R4, f'teachers/fold{k}-outer0/predictions.npz')); assert np.array_equal(ib['query'], B)
    base_A, _ = ce.teacher_oof(d, g, 'deep', A)
    rawA = ce.raw_context(d, g, 'deep', base_A, B, ib['base'], ib['tab'] - ib['ridge'])
    base_T, _ = ce.teacher_oof(d, g, 'outer', T)
    ix = np.array([dev.index(i) for i in V])
    rawV = ce.raw_context(d, g, 'outer', base_T, V, non021_0[ix], d['C73_tab'][0, ix] - sridge[0, ix])
    yB = labels[B]; eligB = an['highB']
    jx = an['ix']; p76[jx] = an['p76V']; p82[jx] = an['p82V']
    assert np.array_equal(np.sort(jx), np.sort(ix)) and np.array_equal(jx, ix)
    for arm in ('S', 'C'):
        sel = model().fit(design(arm, A, rawA), labels[A] - base_A[A])
        rb = sel.predict(design(arm, B, rawA))
        l_rep = lam(rb, yB, an['p76B'], eligB); l_add = lam(rb, yB, an['p82B'], eligB)
        fin = model().fit(design(arm, T, rawV), labels[T] - base_T[T])
        rv = fin.predict(design(arm, V, rawV)); rv[failed_all[V]] = 0.
        hv = an['highV']
        oof[arm]['rep'][ix] = an['p76V'] + np.where(hv, l_rep * rv, 0)
        oof[arm]['add'][ix] = an['p82V'] + np.where(hv, l_add * rv, 0)
        lams[arm].append(dict(fold=k, rep=l_rep, add=l_add, alpha_sel=float(sel[-1].alpha_), alpha_fin=float(fin[-1].alpha_)))

y = labels[d['dev']]; fold = d['fold'][0]; nz = d['noise'][d['dev']]; gate = (nz >= 12) & ~d['fail'][0]
rng = np.random.default_rng(20261017)


def rm(p, m=None):
    m = np.ones(len(p), bool) if m is None else m
    return float(np.sqrt(np.mean((p[m] - yy[m]) ** 2)))


def cmp(base, cand, folds=None):
    dd = (cand - yy) ** 2 - (base - yy) ** 2; o = np.argsort(dd); n = len(dd)
    idx = rng.integers(0, n, (10000, n))
    gains = np.sqrt(np.mean((base - yy)[idx] ** 2, 1)) - np.sqrt(np.mean((cand - yy)[idx] ** 2, 1))
    keep = np.ones(n, bool); keep[o[:10]] = False
    out = dict(base=rm(base), cand=rm(cand), delta=rm(cand) - rm(base), p_better=float(np.mean(gains > 0)),
               ci95=[float(np.quantile(gains, .025)), float(np.quantile(gains, .975))], delta_without_top10=rm(cand, keep) - rm(base, keep))
    if folds is not None:
        out['fold_delta'] = [rm(cand, folds == f) - rm(base, folds == f) for f in range(5)]
        out['folds_improved'] = int(sum(x < 0 for x in out['fold_delta']))
    return out


yy = y
assert np.isfinite(p76).all() and np.isfinite(p82).all()
r = res['original']
r['rmse'] = dict(p76=rm(p76), p82=rm(p82), **{f'{a}_{m}': rm(oof[a][m]) for a in oof for m in oof[a]})
r['S_rep_vs_076'] = cmp(p76, oof['S']['rep'], fold); r['C_rep_vs_076'] = cmp(p76, oof['C']['rep'], fold)
r['S_rep_vs_082'] = cmp(p82, oof['S']['rep'], fold); r['S_add_vs_082'] = cmp(p82, oof['S']['add'], fold)
r['C_add_vs_082'] = cmp(p82, oof['C']['add'], fold); r['lambdas'] = lams
r['gated_rmse'] = dict(p76=rm(p76, gate), p82=rm(p82, gate), S_rep=rm(oof['S']['rep'], gate), C_rep=rm(oof['C']['rep'], gate))
store.update(ids=np.array([str(i) for i in d['ids'][d['dev']]]), y=y, p76=p76, p82=p82, **{f'{a}_{m}': oof[a][m] for a in oof for m in oof[a]})

# holdout-100 and partitions 1/2 (frozen coefficient = mean of the five original-partition fold coefficients)
lam_S = float(np.mean([x['rep'] for x in lams['S']])); lam_Sadd = float(np.mean([x['add'] for x in lams['S']]))
lam_C = float(np.mean([x['rep'] for x in lams['C']]))
RG = json.loads((_repo_path(RT, 'label-graph.json')).read_text())
strict = baseline
Bc = np.load(_repo_path(E, 'stepchange-20261016/repro/components/c14b-orig.npz'))


def rt_teacher(p, k, name):
    return np.load(_repo_path(RT, f'teachers/fold{k}-p{p}-{name}/predictions.npz'))


extra = {}
for p in (1, 2, 3):
    pr = {a: [] for a in ('S', 'C')}; cn = []; bs = []; ids_v = []; gates = []
    for g in [x for x in RG if x['partition'] == p]:
        k = g['fold']; T, V = np.array(g['T'], int), np.array(g['V'], int)
        base = np.full(500, np.nan); gap = np.full(500, np.nan)
        for n in g['nodes']:
            if n['name'] == 'full':
                continue
            z = rt_teacher(p, k, n['name']); base[z['query']] = z['base']; gap[z['query']] = z['tab'] - z['ridge']
        assert np.isfinite(base[T]).all()
        if p in (1, 2):
            ix = np.array([dev.index(i) for i in V]); leg = d['C73_cnn'][p, ix]; nv = (strict[p, ix] - .16 * leg) / .84
            fl = d['fail'][p, ix]; nv[fl] = strict[p, ix][fl]; qg = d['C73_tab'][p, ix] - sridge[p, ix]
            b = Bc['dev'][p][ix]
        else:
            z = rt_teacher(p, k, 'full'); assert np.array_equal(z['query'], V); nv = z['base']; qg = z['tab'] - z['ridge']
            fl = failed_all[V]; b = z['base']
        tb = base.copy(); tb[V] = nv; gp = gap.copy(); gp[V] = qg; raw = ctx(tb, gp)
        gv = (d['noise'][V] >= 12) & ~fl
        for arm in ('S', 'C'):
            fin = model().fit(design(arm, T, raw), labels[T] - base[T])
            rv = fin.predict(design(arm, V, raw)); rv[fl] = 0.
            pr[arm].append(np.where(gv, (lam_S if arm == 'S' else lam_C) * rv, 0))
        zs = [np.load(_repo_path(RT, f'cnn/p{p}-fold{k}-seed{j}-outer.npz')) for j in (0, 1)]
        cn.append(np.where(gv, W_CNN * np.mean([zz['residual'] for zz in zs], 0), 0)); bs.append(b); ids_v.append(V); gates.append(gv)
    V = np.concatenate(ids_v); b = np.concatenate(bs); gv = np.concatenate(gates); yy = labels[V]
    cS = np.concatenate(pr['S']); cC = np.concatenate(pr['C']); cc = np.concatenate(cn)
    out = dict(n=int(len(V)), n_gated=int(gv.sum()), rmse_base=rm(b), rmse_cnn=rm(b + cc), rmse_S=rm(b + cS), rmse_C=rm(b + cC))
    for nm, cv in (('S', cS), ('C', cC), ('CNN', cc)):
        dd = ((b + cv - yy) ** 2 - (b - yy) ** 2)[gv]
        bsm = np.array([dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(10000)])
        out[f'gated_dMSE_{nm}'] = dict(mean=float(dd.mean()), se=float(dd.std(ddof=1) / np.sqrt(len(dd))), p_better=float(np.mean(bsm < 0)))
    out['corr_S_CNN_gated'] = float(np.corrcoef(cS[gv], cc[gv])[0, 1])
    extra[{1: 'partition1', 2: 'partition2', 3: 'holdout100'}[p]] = out
res.update(extra); res['frozen_lambda'] = dict(S=lam_S, S_add=lam_Sadd, C=lam_C)

o = res['original']; h = res['holdout100']
rule_rep = dict(r1=o['rmse']['S_rep'] <= o['rmse']['p82'] + 0.010, r2=o['rmse']['S_rep'] <= o['rmse']['C_rep'] - 0.020,
                r3=h['gated_dMSE_S']['mean'] < 0 and h['gated_dMSE_S']['p_better'] >= 0.8 and h['gated_dMSE_S']['mean'] < h['gated_dMSE_CNN']['mean'],
                r4=o['S_rep_vs_076']['folds_improved'] >= 4)
a = o['S_add_vs_082']
rule_add = dict(a1=a['delta'] <= -0.020, a2=a['p_better'] >= 0.9, a3=a['folds_improved'] >= 4, a4=a['delta_without_top10'] <= 0,
                a5=not (h['gated_dMSE_S']['mean'] > 0 and (1 - h['gated_dMSE_S']['p_better']) >= 0.8))
res['rule_replacement'] = rule_rep; res['rule_addition'] = rule_add
res['verdict'] = 'supported replacement' if all(rule_rep.values()) else ('supported addition' if all(rule_add.values()) else 'rejected')
np.savez(_repo_path(HERE, 'preds.npz'), **store)
json.dump(res, open(_repo_path(HERE, 'result.json'), 'w'), indent=1)
print(json.dumps(o['rmse'], indent=1)); print('gated', o['gated_rmse'])
for kk in ('S_rep_vs_076', 'C_rep_vs_076', 'S_rep_vs_082', 'S_add_vs_082', 'C_add_vs_082'):
    q = o[kk]; print(f"{kk:14s} {q['base']:.4f}->{q['cand']:.4f} d {q['delta']:+.4f} P {q['p_better']:.2f} CI {np.round(q['ci95'], 4)} folds {np.round(q['fold_delta'], 3)} no-top10 {q['delta_without_top10']:+.4f}")
print('lambdas S', [round(x['rep'], 3) for x in lams['S']], 'add', [round(x['add'], 3) for x in lams['S']], 'C', [round(x['rep'], 3) for x in lams['C']])
for kk in ('partition1', 'partition2', 'holdout100'):
    q = res[kk]; print(kk, {x: (round(v, 4) if isinstance(v, float) else v) for x, v in q.items() if not isinstance(v, dict)},
                       {x: {kk2: round(vv, 3) for kk2, vv in v.items()} for x, v in q.items() if isinstance(v, dict)})
print('rules', rule_rep, rule_add, 'VERDICT', res['verdict'])
