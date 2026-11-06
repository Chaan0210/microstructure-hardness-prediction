"""Phase-1 diagnosis on existing strict-076 / 082 OOF (original partition, 400 dev rows). Read-only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json
from pathlib import Path
import numpy as np

ROOT = _RepoPath(__file__).resolve().parents[4]
E = _RepoPath(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

inp = np.load(_repo_path(E, 'rmse-research-20261015/inputs.npz'), allow_pickle=True)
b76 = np.load(_repo_path(E, '076-complement-20261016/baseline076.npz'), allow_pickle=True)
ssl = np.load(_repo_path(E, 'highnoise-ssl-20261016-2312/oof.npz'), allow_pickle=True)
dev = inp['dev']; ids = inp['ids'][dev]; y = inp['y'][dev]; X = inp['X'][dev]; keys = list(inp['keys'])
assert np.array_equal(ids, b76['ids']) and np.array_equal(ids, ssl['ids'])
assert np.allclose(y, ssl['y'])
nz = inp['noise'][dev]; fold = inp['fold'][0]
G = b76['G']; tab = b76['tab076']; rr = b76['ridge076']; lr = b76['logridge076']
st = inp['C73_set'][0]; gm = inp['C73_gmm'][0]; failed = b76['failed']
p76 = b76['p76']; p82 = ssl['baseline082']
rebuilt = .16 * G + .384 * tab + .128 * rr + .128 * lr + .05 * st + .15 * gm
rebuilt[failed] = G[failed]
res = {'rebuild_maxdiff': float(np.max(np.abs(rebuilt - p76)))}
band = np.where(nz < 9.5, 'clean', np.where(nz < 12, 'mid', 'noisy'))
comps = dict(p76=p76, p82=p82, tab=tab, G=G, ridge=rr, logridge=lr, set=st, gmm=gm)


def rmse(a, m=None):
    m = np.ones(len(y), bool) if m is None else m
    return float(np.sqrt(np.mean((a[m] - y[m]) ** 2)))


res['bands'] = {b: {'n': int((band == b).sum()), **{k: rmse(v, band == b) for k, v in comps.items()}}
                for b in ('clean', 'mid', 'noisy')}
res['all'] = {k: rmse(v) for k, v in comps.items()}
e82 = p82 - y
sse = e82 ** 2
res['sse_share'] = {b: float(sse[band == b].sum() / sse.sum()) for b in ('clean', 'mid', 'noisy')}

# error vs hardness (regression-to-mean signature) within band
rtm = {}
for b in ('clean', 'mid', 'noisy'):
    m = band == b
    q = np.quantile(y[m], [0, .2, .4, .6, .8, 1])
    rows = []
    for i in range(5):
        mm = m & (y >= q[i]) & (y <= q[i + 1])
        rows.append(dict(y_lo=float(q[i]), y_hi=float(q[i + 1]), n=int(mm.sum()), bias=float(e82[mm].mean()),
                         rmse=float(np.sqrt(np.mean(e82[mm] ** 2)))))
    slope = float(np.polyfit(p82[m], y[m], 1)[0])
    rtm[b] = dict(quintiles=rows, slope_y_on_pred=slope, sd_y=float(y[m].std()), sd_pred=float(p82[m].std()),
                  corr=float(np.corrcoef(p82[m], y[m])[0, 1]))
res['by_hardness'] = rtm

# noise stripes
stripes = {}
for k in np.unique(np.round(nz / 0.7413)).astype(int):
    m = np.round(nz / 0.7413).astype(int) == k
    stripes[int(k)] = dict(n=int(m.sum()), rmse82=rmse(p82, m), rmse76=rmse(p76, m), sd_y=float(y[m].std()))
res['stripes'] = stripes

# residual-feature correlations (diagnostic only; not used for selection)
fc = {}
for b in ('clean', 'noisy'):
    m = band == b
    out = []
    for j, k in enumerate(keys):
        x = X[m, j]
        ok = np.isfinite(x)
        if ok.sum() < 30 or np.std(x[ok]) == 0:
            continue
        rx = np.argsort(np.argsort(x[ok])); re = np.argsort(np.argsort(e82[m][ok]))
        out.append((k, float(np.corrcoef(rx, re)[0, 1]), float(np.corrcoef(np.argsort(np.argsort(y[m][ok])), rx)[0, 1])))
    out.sort(key=lambda t: -abs(t[1]))
    fc[b] = out[:12]
res['resid_feature_spearman_top'] = fc

# component disagreement vs error
C = np.c_[tab, G, rr, lr, st, gm]
spread = C.std(1)
for b in ('clean', 'noisy'):
    m = band == b
    res.setdefault('disagreement', {})[b] = dict(
        spearman_spread_abs_err=float(np.corrcoef(np.argsort(np.argsort(spread[m])), np.argsort(np.argsort(np.abs(e82[m]))))[0, 1]),
        resid_corr={n: float(np.corrcoef(c[m] - y[m], e82[m])[0, 1]) for n, c in zip(['tab', 'G', 'ridge', 'logridge', 'set', 'gmm'], C.T)})
# set blow-ups
res['set_abs_err_q'] = {q: float(np.quantile(np.abs(st - y), q)) for q in (.5, .9, .99, 1.0)}
res['set_minus_blend_q'] = {q: float(np.quantile(np.abs(st - p76), q)) for q in (.5, .9, .99, 1.0)}

# concentration
o = np.argsort(-sse)
res['top10_sse_share'] = float(sse[o[:10]].sum() / sse.sum())
res['top40_sse_share'] = float(sse[o[:40]].sum() / sse.sum())
res['worst20'] = [dict(id=str(ids[i]), y=float(y[i]), p82=float(p82[i]), noise=float(nz[i]), band=str(band[i]),
                       tab=float(tab[i]), G=float(G[i]), set=float(st[i]), gmm=float(gm[i])) for i in o[:20]]
# fold RMSE
res['fold_rmse82'] = [rmse(p82, fold == k) for k in range(5)]
res['fold_rmse76'] = [rmse(p76, fold == k) for k in range(5)]
# noisy: CNN correction share
corr = p82 - p76
m = band == 'noisy'
res['cnn_corr'] = dict(rms=float(np.sqrt(np.mean(corr[m] ** 2))), corr_with_neg_resid=float(np.corrcoef(corr[m], -(p76 - y)[m])[0, 1]))
json.dump(res, open(_repo_path(OUT, 'diag.json'), 'w'), indent=1)
print(json.dumps({k: res[k] for k in ('rebuild_maxdiff', 'all', 'bands', 'sse_share', 'fold_rmse82', 'top10_sse_share', 'top40_sse_share', 'cnn_corr')}, indent=1))
for b in rtm:
    print(b, 'slope', round(rtm[b]['slope_y_on_pred'], 3), 'corr', round(rtm[b]['corr'], 3), 'sd_y', round(rtm[b]['sd_y'], 1), 'sd_pred', round(rtm[b]['sd_pred'], 1))
    for r in rtm[b]['quintiles']:
        print('   y %.0f-%.0f n%d bias %+.2f rmse %.2f' % (r['y_lo'], r['y_hi'], r['n'], r['bias'], r['rmse']))
for b in fc:
    print(b, [(k, round(c, 3), round(cy, 2)) for k, c, cy in fc[b][:8]])
print('stripes', {k: (v['n'], round(v['rmse82'], 2), round(v['sd_y'], 1)) for k, v in stripes.items()})
print('disagreement', json.dumps(res['disagreement']))
print('set err q', res['set_abs_err_q'], res['set_minus_blend_q'])
