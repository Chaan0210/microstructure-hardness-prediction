"""Evaluation and decision exactly as pre-registered in PLAN.md. Correction c = 0.5468 x mean(seed0, seed1) residual on gated rows
(noise >= 12 and not failed), 0 elsewhere. Baseline = reproduced 076 dev OOF (partitions 1, 2) / full teacher (holdout).
Reference: the original partition recomputed with the same fixed 0.5468 from round4's aux-seed0/1 outer residuals.
usage: .venv/bin/python retest_eval.py -> results.json"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json, numpy as np
from pathlib import Path
OUT = Path(__file__).resolve().parent; E = _RepoPath(__file__).resolve().parents[2]; R4 = _repo_path(E, 'rmse-research-20261015/round4-full073'); W = 0.5467904360508202
B = np.load(_repo_path(E, 'stepchange-20261016/repro/components/c14b-orig.npz')); y = B['y']; nz = B['nz_dev']; fail = B['fail_dev']; b76 = B['dev']
I = np.load(_repo_path(E, 'rmse-research-20261015/inputs.npz')); dev = list(I['dev']); yall = I['y']; G = json.load(open(_repo_path(OUT, 'label-graph.json')))
rng = np.random.RandomState(0)

def correction_dev(p):
    c = np.full(400, np.nan)
    for k in range(5):
        if p == 0:
            zs = [np.load(_repo_path(R4, f'aux-seed{j}/fold{k}-outer.npz')) for j in (0, 1)]; ix = zs[0]['dev_index']; assert np.array_equal(zs[1]['dev_index'], ix)
        else:
            zs = [np.load(_repo_path(OUT, f'cnn/p{p}-fold{k}-seed{j}-outer.npz')) for j in (0, 1)]; ix = np.array([dev.index(i) for i in zs[0]['V']])
        c[ix] = np.mean([z['residual'] for z in zs], 0)
    assert np.isfinite(c).all(); g = (nz >= 12) & ~fail[p]
    return np.where(g, W * c, 0.0), g

def stats(d, folds=None, nboot=10000):
    o = np.argsort(d); bs = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(nboot)])
    out = dict(dMSE=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), P_better=float((bs < 0).mean()), P_worse=float((bs > 0).mean()),
               CI95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=int(len(d)),
               drop_top5=float(np.delete(d, o[:5]).mean()), drop_top10=float(np.delete(d, o[:10]).mean()),
               share_top10=float(d[o[:10]].sum() / d.sum()) if d.sum() != 0 else None)
    if folds is not None: out['per_fold'] = [float(d[folds == f].mean()) for f in range(5)]
    return out

res = {'weight_noisy': W}; per = {}
for p in (0, 1, 2):
    c, g = correction_dev(p); new = b76[p] + c; d_img = (new - y) ** 2 - (b76[p] - y) ** 2
    x = b76[p][g] - b76[p][g].mean(); s = float((x @ c[g]) / (x @ x)); ns = b76[p].copy(); ns[g] = b76[p][g] + c[g] - s * x
    rm = lambda a, m=slice(None): float(np.sqrt(((a - y)[m] ** 2).mean()))
    per[p] = d_img
    res[f'partition{p}'] = dict(**stats(d_img[g], I['fold'][p][g]), rmse_076=rm(b76[p]), rmse_082=rm(new), gain_all_rows=rm(b76[p]) - rm(new),
                                gated_rmse=[rm(b76[p], g), rm(new, g)], stretch_coef=s, non_stretch_gain=rm(b76[p]) - rm(ns),
                                correction_rms_gated=float(np.sqrt((c[g] ** 2).mean())))
g12 = (nz >= 12) & ~fail[1] & ~fail[2]
res['new_partitions_combined'] = stats(((per[1] + per[2]) / 2)[g12])
# holdout-100: T(400) -> 100 teacher base vs base + correction
gh = [x for x in G if x['partition'] == 3][0]; V = np.array(gh['V'], int)
zf = np.load(_repo_path(OUT, 'teachers/fold0-p3-full/predictions.npz')); assert np.array_equal(zf['query'], V)
zs = [np.load(_repo_path(OUT, f'cnn/p3-fold0-seed{j}-outer.npz')) for j in (0, 1)]; ch = np.mean([z['residual'] for z in zs], 0)
failh = zs[0]['failed']; gh_ = (I['noise'][V] >= 12) & ~failh; yh = yall[V]; bh = zf['base']; nh = bh + np.where(gh_, W * ch, 0)
dh = (nh - yh) ** 2 - (bh - yh) ** 2
res['holdout100'] = dict(**stats(dh[gh_]), rmse_teacher=float(np.sqrt(((bh - yh)[~failh] ** 2).mean())), rmse_teacher_plus_c=float(np.sqrt(((nh - yh)[~failh] ** 2).mean())))
# pre-registered decision
cmb, p1, p2, ho = res['new_partitions_combined'], res['partition1'], res['partition2'], res['holdout100']
support = cmb['dMSE'] < 0 and cmb['P_better'] >= 0.80 and not any(q['dMSE'] > 0 and q['P_worse'] >= 0.80 for q in (p1, p2))
contradict = (cmb['dMSE'] > 0 and cmb['P_worse'] >= 0.80) or (cmb['P_better'] < 0.80 and ho['dMSE'] > 0 and ho['P_worse'] >= 0.80)
res['decision'] = 'SUPPORTS 082' if support else ('CONTRADICTS 082 -> recommend reverting to 076 (user action)' if contradict else 'INCONCLUSIVE -> keep 082 (pre-registered default)')
json.dump(res, open(_repo_path(OUT, 'results.json'), 'w'), indent=1)
for k in ('partition0', 'partition1', 'partition2', 'new_partitions_combined', 'holdout100'):
    q = res[k]; print(f"{k:24s} dMSE {q['dMSE']:+7.3f} +- {q['se']:.3f} P(better) {q['P_better']:.3f} CI {np.round(q['CI95'], 2)} drop5 {q['drop_top5']:+.3f} drop10 {q['drop_top10']:+.3f}"
                      + (f" folds {np.round(q['per_fold'], 2)}" if 'per_fold' in q else '') + (f" | RMSE 076 {q['rmse_076']:.4f} -> {q['rmse_082']:.4f}, stretch {q['stretch_coef']:+.3f}, non-stretch gain {q['non_stretch_gain']:+.4f}" if 'rmse_076' in q else ''))
print('DECISION:', res['decision'])
