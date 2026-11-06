"""Compliant residual-CNN correction (round4 'aux' recipe) on top of the compliant C076 structure.
stage teachers: 3-fold cross-fit (KFold 3, shuffle, seed 2026101607) of the C076 non-021 branch on the 500 train rows -> T_oof, gap_oof
   (tab = 12 h9 member sets x 3 TabPFN seeds [nb sets for noise<12 rows, full sets for noise>=12 rows]; ridge/logRidge = nb plain partners on
    noise<12 rows, DTA arm B with copies of the fold's training sources on noise>=12 rows; gmm nb/full; grain-set model 3 seeds; failed -> label mean)
stage fit: round4 ComplementCNN 'aux' arm, residual target y - T_oof, 6 context inputs, seeds 2026101501/2026201501, 40 epochs (cnn_fitter.fit_one).
stage infer: test context from the C076 test components; correction = seed mean; C076 + w[band]*correction, w = (0, 0.5467904360508202) frozen
   from round4 (addition arm, mean of five B-only inner calibrations). Nothing is fitted on test.
usage (from R/experiments/c14rescnn): python c14_rescnn.py teachers|fit|infer DEVICE"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[6]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, json, copy, gc, time, warnings, importlib.util, numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')
H = Path(__file__).resolve().parent; E = H.parent; P = E.parent; OUT = _repo_path(H, 'out'); OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(_repo_path(E, 'push-300'))); sys.path.insert(0, str(_repo_path(H, 'lib')))
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import GroupKFold, KFold
stage = sys.argv[1]; DEV = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}; order = [r['ID'] for r in csv.DictReader(open(_repo_path(P, 'data/sample_submission.csv')))]
S, V = ['038', '039', '044'], ['u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4']; VH = [v + 'H2Z' for v in V]; SETS = ['c1', 'c2', 'c3', 'c4']
W = dict(tab=.384, ridge=.128, logr=.128, set=.05, gmm=.15)
F44 = np.load(_repo_path(E, 'push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); ids = [str(i) for i in F44['ids']]; keys = [str(k) for k in F44['keys']]
X44 = F44['X'].astype(float); y = np.array([lab[i] for i in ids]); noise = np.nan_to_num(X44[:, keys.index('noise')] * 255, nan=0)
failed = np.nan_to_num(X44[:, keys.index('n_obj')], nan=0) < 20; gate = noise >= 12
G = json.load(open(_repo_path(E, 'push-1300/abl/groups.json'))); drop = set(G['intensity'].split() + G['het'].split())
def tabset(fs, s, v, split='train'):
    z = np.load(_repo_path(E, f'push-1300/abl/feat-h9-{fs}-{s}-{v}-{split}.npz'), allow_pickle=True); assert [str(i) for i in z['ids']] == (ids if split == 'train' else order); return z['X'].astype(float)
def anghcor(v, split='train'):
    z = np.load(_repo_path(E, f'push-800/feat-{v}-anghcor-{split}.npz'), allow_pickle=True); assert [str(k) for k in z['keys']] == keys; return z['X'].astype(float)
def copies(v):
    SX, SID = [], []; pos = {i: k for k, i in enumerate(ids)}; ni = keys.index('noise')
    for s in SETS:
        z = np.load(_repo_path(E, f'push-1200/synthrows-{s}-{v}-anghcor.npz'), allow_pickle=True); assert [str(k) for k in z['keys']] == keys
        m = z['X'][:, ni].astype(float) * 255 >= 12; SX.append(z['X'].astype(float)[m]); SID += [pos[str(i)] for i in np.array(z['ids'])[m]]
    return np.vstack(SX), np.array(SID)
GR = {'ridge': [0.1, 0.3, 1, 3, 10, 30, 100, 1000], 'log': [0.3, 1, 2, 3, 10, 30, 100]}
def pipe(kind, a): return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True) if kind == 'ridge' else SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=a))
tf = lambda kind, t: np.log(t) if kind == 'log' else t; itf = lambda kind, p: np.exp(p) if kind == 'log' else p
def fitB(kind, Xr, yr, sr, nzr, Xc, yc, sc):
    Xa = np.vstack([Xr, Xc]); ya = np.r_[yr, yc]; srcs = np.r_[sr, sc]; isreal = np.r_[np.ones(len(yr), bool), np.zeros(len(yc), bool)]; nzall = np.r_[nzr, np.full(len(yc), 99.0)]; best = None
    for a in GR[kind]:
        err = []
        for tr, va in GroupKFold(5).split(Xa, ya, srcs):
            m = pipe(kind, a).fit(Xa[tr], tf(kind, ya[tr])); sel = va[isreal[va] & (nzall[va] >= 12)]
            if sel.size: err += list((itf(kind, m.predict(Xa[sel])) - ya[sel]) ** 2)
        e = np.mean(err)
        if best is None or e < best[0]: best = (e, a)
    return pipe(kind, best[1]).fit(Xa, tf(kind, ya))
src = open(_repo_path(E, 'push-900/gmm_partner.py')).read(); src = src[:src.index("mode = sys.argv[1]")]; gp = {'__file__': str(_repo_path(E, 'push-900/gmm_partner.py')), '__name__': 'gp'}; exec(compile(src, 'gp', 'exec'), gp)
S9 = _repo_path(E, 'probabilistic-grains-900')
def teachers():
    import harness
    Xg = gp['gmm_feats'](np.load(_repo_path(S9, 'object-bank-train.npz'), allow_pickle=True), ids)
    spec = importlib.util.spec_from_file_location('c14_grainset', _repo_path(S9, 'set_model/run.py')); sm = importlib.util.module_from_spec(spec); spec.loader.exec_module(sm)
    bank, _, _ = sm.load_inputs(_repo_path(S9, 'object-bank-train.npz'), _repo_path(S9, 'base.npz')); bpos = {str(i): j for j, i in enumerate(bank['ids'])}; bidx = np.array([bpos[i] for i in ids]); bank.pop('y'); cfg = copy.deepcopy(sm.CONFIG)
    VX = {v: anghcor(v) for v in VH}; CP = {v: copies(v) for v in VH}; keep = np.array([k not in drop for k in keys])
    TB = {(fs, s, v): tabset(fs, s, v) for fs in ('nb', 'full') for s in S for v in V}
    T = np.full(500, np.nan); GAP = np.full(500, np.nan); comp = {k: np.full(500, np.nan) for k in W}
    for fold, (tr, va) in enumerate(KFold(3, shuffle=True, random_state=2026101607).split(y)):
        f = _repo_path(OUT, f'teacher-fold{fold}.npz')
        if f.exists():
            z = np.load(_repo_io(f)); assert np.array_equal(z['query'], va); [comp[k].__setitem__(va, z[k]) for k in W]; continue
        t0 = time.time(); res = {k: np.full(len(va), np.nan) for k in W}
        for noisy in (False, True):
            m = gate[va] == noisy; qi = va[m]
            if not len(qi): continue
            fs = 'full' if noisy else 'nb'; mem = []
            for s in S:
                for v in V:
                    Xt = TB[(fs, s, v)]
                    for sd in range(3):
                        tm = harness.tabpfn_new('v3.5', 100 * sd + fold, DEV); tm.fit(Xt[tr], y[tr]); mem.append(tm.predict(Xt[qi])); del tm; gc.collect()
            res['tab'][m] = np.mean(mem, 0)
            cols = np.ones(len(keys), bool) if noisy else keep
            if noisy:
                for kind, nm, vs in (('ridge', 'ridge', ['u1dmavgH2Z']), ('log', 'logr', VH)):
                    ps = []
                    for v in vs:
                        SX, SID = CP[v]; kp = np.isin(SID, tr)
                        mdl = fitB(kind, VX[v][tr], y[tr], tr, noise[tr], SX[kp], y[SID[kp]], SID[kp]); ps.append(itf(kind, mdl.predict(VX[v][qi])))
                    res[nm][m] = ps[0] if kind == 'ridge' else np.exp(np.mean(np.log(ps), 0))
            else:
                mdl = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000])).fit(VX['u1dmavgH2Z'][tr][:, cols], y[tr])
                res['ridge'][m] = mdl.predict(VX['u1dmavgH2Z'][qi][:, cols]); lp = []
                for v in VH:
                    lm = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), Ridge(alpha=2.0)).fit(VX[v][tr][:, cols], np.log(y[tr])); lp.append(lm.predict(VX[v][qi][:, cols]))
                res['logr'][m] = np.exp(np.mean(lp, 0))
            xg = np.c_[VX['u1dmavgH2Z'][:, cols], Xg]; res['gmm'][m] = gp['fit_predict'](xg[tr], y[tr], xg[qi], 19001)[0]
            print(json.dumps(dict(fold=fold, noisy=noisy, n_query=int(m.sum()), sec=round(time.time() - t0))), flush=True)
        b2 = dict(bank); b2['y'] = np.full(len(bank['ids']), np.nan); bt = bidx[tr]; b2['y'][bt] = y[tr]; sp = []
        for seed in (731503, 731509, 731527):
            sm.CONFIG = {**copy.deepcopy(cfg), 'seed': seed}; stop = sm.train_with_inner_stop(b2, bt, 'cpu', seed + fold)
            model, context, data, hist = sm.refit(b2, bt, 'cpu', seed + fold, stop['selected_epochs']); sp.append(sm.predict_y(model, data, context, bidx[va])); del model, data; gc.collect()
        res['set'] = np.mean(sp, 0)
        np.savez(_repo_io(f), query=va, train=tr, **res); [comp[k].__setitem__(va, res[k]) for k in W]; print('teacher fold', fold, 'done', round(time.time() - t0), flush=True)
    T = sum(W[k] * comp[k] for k in W) / .84; T[failed] = y.mean(); GAP = comp['tab'] - comp['ridge']
    np.savez(_repo_path(OUT, 'teachers.npz'), ids=np.array(ids), T=T, gap=GAP, **comp)
    print('teacher T oof RMSE (non-failed) %.3f' % np.sqrt(((T - y)[~failed] ** 2).mean()))
def ctx(Xf, kk, T, gap, nz):
    return np.c_[T, gap, nz, Xf[:, kk.index('u_ph')], np.log1p(np.maximum(np.nan_to_num(Xf[:, kk.index('n_obj')], nan=0), 0)), Xf[:, kk.index('coverage')]]
def fit():
    import torch, cnn_fitter as cf
    from PIL import Image
    sys.path.insert(0, str(_repo_path(E, 'rmse-research-20261015/round4-full073')))
    Z = np.load(_repo_path(OUT, 'teachers.npz')); T, gap = Z['T'], Z['gap']
    images = np.stack([np.asarray(Image.open(_repo_path(P, f'data/train/{i}.png')), np.float32) / 255 for i in ids])[:, None]
    train = np.arange(500); raw = ctx(X44, keys, T, gap, noise)
    xm, xs = float(images.mean()), float(images.std()); ym, ys = float(y.mean()), float(y.std())
    med = np.nanmedian(raw, 0); rr = np.where(np.isfinite(raw), raw, med); cm = rr.mean(0); cs = np.maximum(rr.std(0), 1e-6)
    data = dict(images=((images - xm) / xs).astype(np.float32), context=((rr - cm) / cs).astype(np.float32), train=train, valid=np.array([], int), base=T,
                targets=dict(direct=((y - ym) / ys).astype(np.float32), residual=((y - T) / ys).astype(np.float32)),
                meta=dict(image_mean=xm, image_std=xs, target_mean=ym, target_std=ys, context_mean=cm.tolist(), context_std=cs.tolist(), context_median=med.tolist(), stage='c14 full500'))
    src = open(_repo_path(H, 'lib/cnn_fitter.py')).read().replace('float(np.max(np.abs(rp-vp)))', 'float(np.max(np.abs(rp-vp), initial=0.))'); (_repo_path(H, 'lib/cnn_fitter_full.py')).write_text(src)
    spec = importlib.util.spec_from_file_location('cff', _repo_path(H, 'lib/cnn_fitter_full.py')); cff = importlib.util.module_from_spec(spec); spec.loader.exec_module(cff)
    torch.set_num_threads(2)
    for j, seed in enumerate((2026101501, 2026201501)):
        d = _repo_path(OUT, f'aux-seed{j}')
        if not (d / 'training.json').exists(): cff.fit_one(data, 'aux', seed, d)
    json.dump(data['meta'], open(_repo_path(OUT, 'normalization.json'), 'w'))
    print('fit done; train residual RMSE (T) %.3f' % np.sqrt(((y - T)[~failed] ** 2).mean()))
def infer(blend_npz):
    import torch, cnn_fitter as cf
    from PIL import Image
    B = np.load(_repo_io(blend_npz)); assert [str(i) for i in B['order']] == order; meta = json.load(open(_repo_path(OUT, 'normalization.json')))
    Ft = np.load(_repo_path(E, 'push-800/feat-u1dmavgH2Z-anghcor-test.npz'), allow_pickle=True); kt = [str(k) for k in Ft['keys']]; Xt = Ft['X'].astype(float)
    nzt = np.nan_to_num(Xt[:, kt.index('noise')] * 255, nan=0); band = (nzt >= 12).astype(int); fl = B['failed_test'].astype(bool)
    p021 = B['test_cnn']; Tt = (B['test'] - .16 * p021) / .84; gapt = B['test_tab'] - B['test_ridge']; raw = ctx(Xt, kt, Tt, gapt, nzt)
    w = np.array([0.0, 0.5467904360508202]); act = (~fl) & (w[band] > 0)
    imgs = np.stack([np.asarray(Image.open(_repo_path(P, f'data/test/{order[i]}.png')), np.float32) / 255 for i in np.flatnonzero(act)])[:, None]
    im = ((imgs - meta['image_mean']) / meta['image_std']).astype(np.float32)
    clean = np.where(np.isfinite(raw), raw, np.asarray(meta['context_median'])); cc = ((clean - np.asarray(meta['context_mean'])) / np.asarray(meta['context_std'])).astype(np.float32)[act]
    corr = np.zeros((2, 1000))
    for j in range(2):
        ck = torch.load(_repo_path(OUT, f'aux-seed{j}/model.pt'), map_location='cpu', weights_only=True); m = cf.ComplementCNN(); m.load_state_dict(ck['state_dict'])
        corr[j, act] = cf.predict(m, im, cc, np.arange(len(im)), 'aux') * meta['target_std']
    c = corr.mean(0); pred = B['test'] + w[band] * c; pred[fl] = B['test'][fl]
    np.savez(_repo_path(OUT, 'test-correction.npz'), order=np.array(order), correction=c, seeds=corr, pred=pred, base=B['test'], band=band, failed=fl)
    print('correction: rows %d rms %.3f mean %+.3f | pred change rms %.3f' % (act.sum(), np.sqrt((c[act] ** 2).mean()), c[act].mean(), np.sqrt(((pred - B['test']) ** 2).mean())))
if __name__ == '__main__':
    {'teachers': teachers, 'fit': fit}.get(stage, lambda: infer(sys.argv[3]))()
