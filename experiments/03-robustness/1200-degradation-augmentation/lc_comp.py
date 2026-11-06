"""Component learning curves on the Study900 dev folds: every outer fold's 320 training rows subsampled to FRAC (seeded by partition/fold/frac;
identical subsets for every component), validation rows unchanged. PART=tab: TabPFN 044 slot (u1dmavgH2Z anghcor, 1 seed), RidgeCV, logRidge
(4 variants), gmm partner. PART=cnn: TinyCNN128 ensemble (4 seeds). -> lc-<PART>-f<FRAC>.npz ([3, 400] Study900 order per component).
usage: lc_comp.py PART FRAC DEVICE"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
part, frac, device = sys.argv[1], float(sys.argv[2]), sys.argv[3]
P = Path(_repo_io('../..')); lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open(_repo_path(P, 'data/train.csv')))}
splits = {'original': _repo_path(P, 'experiments/resolution-216/ensemble-OOF.npz'), '20261031': _repo_path(P, 'experiments/resolution-repartition-217/20261031/ensemble-OOF.npz'), '20261101': _repo_path(P, 'experiments/resolution-repartition-217/20261101/ensemble-OOF.npz')}
sid = [str(i) for i in np.load(_repo_io('../probabilistic-grains-900/base.npz'), allow_pickle=True)['ids']]
def subset(p_, k, tr):
    if frac >= 1: return tr
    rng = np.random.default_rng(7919 * p_ + 104729 * k + int(frac * 1000)); return np.sort(rng.choice(tr, int(round(frac * len(tr))), replace=False))
out = {}
if part == 'tab':
    sys.path.insert(0, _repo_io('../push-300')); import harness
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV, Ridge
    src = open(_repo_io('../push-900/gmm_partner.py')).read(); src = src[:src.index("mode = sys.argv[1]")]; gp = {'__file__': '../push-900/gmm_partner.py', '__name__': 'gp'}; exec(compile(src, 'gp', 'exec'), gp)
    V = ['u1dm1szH2Z', 'u1dm2szH2Z', 'u1dmavgH2Z', 'u1dmavg4H2Z']
    FX = {}
    for v in V:
        F = np.load(_repo_io(f'../push-800/feat-{v}-anghcor-train.npz'), allow_pickle=True); FX[v] = ({str(i): k for k, i in enumerate(F['ids'])}, F['X'].astype(float))
    Btr = np.load(_repo_io('../probabilistic-grains-900/object-bank-train.npz'), allow_pickle=True); pos0, X0 = FX['u1dmavgH2Z']; ids0 = sorted(pos0, key=pos0.get)
    XG = np.c_[X0, gp['gmm_feats'](Btr, ids0)]
    comps = ('tab', 'ridge', 'logridge', 'gmm')
else:
    import torch; from PIL import Image
    sys.path.insert(0, str(_repo_path(P, 'studies'))); from cnn_controls import TinyCNN; sys.path.pop(0); Fn = torch.nn.functional; torch.set_num_threads(2)
    img = lambda f: np.asarray(Image.open(_repo_io(f)).resize((128, 128), Image.Resampling.BOX), dtype=np.float32) / 255.
    IM = {i: img(_repo_path(P, 'data/train', f'{i}.png'))[None] for i in sid}
    comps = ('cnn',)
    def cnn_fit_predict(Xt, yt, Xv, seed):
        nm = (float(Xt.mean()), float(Xt.std()), float(yt.mean()), float(yt.std())); XN = (Xt - nm[0]) / nm[1]; yN = (yt - nm[2]) / nm[3]
        torch.manual_seed(seed); m = TinyCNN().to(device); o = torch.optim.AdamW(m.parameters(), lr=.002, weight_decay=.01); sch = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=80, eta_min=.0001)
        rng = np.random.default_rng(seed); avg = None; nb = max(1, int(np.ceil(len(yt) / 32)))
        for ep in range(80):
            perm = rng.permutation(len(yt))
            for f0 in range(0, len(yt), 32):
                ix = perm[f0:f0 + 32]; xb = np.rot90(XN[ix], int(rng.integers(4)), axes=(2, 3)); xb = np.ascontiguousarray(xb[:, :, :, ::-1] if rng.random() < .5 else xb)
                m.train(); o.zero_grad(set_to_none=True); loss = Fn.mse_loss(m(torch.from_numpy(xb).to(device)), torch.tensor(yN[ix], dtype=torch.float32, device=device)); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.); o.step()
            sch.step()
            if ep >= 75:
                st = {n: v.detach().cpu().clone() for n, v in m.state_dict().items()}; avg = {n: v / 5 for n, v in st.items()} if avg is None else {n: avg[n] + st[n] / 5 for n in st}
        m.load_state_dict(avg); m.eval()
        with torch.no_grad(): return m(torch.from_numpy((Xv - nm[0]) / nm[1]).to(device)).cpu().numpy() * nm[3] + nm[2]
for c in comps: out[c] = np.zeros((3, 400))
for p_, (name, path) in enumerate(splits.items()):
    d = np.load(_repo_io(path)); ids = [str(i) for i in d['ids']]; y = d['y']; fold = d['fold']; pp = {i: j for j, i in enumerate(ids)}; jj = [pp[i] for i in sid]; o = {c: np.zeros(400) for c in comps}
    for k in range(5):
        tr = subset(p_, k, np.flatnonzero(fold != k)); va = fold == k
        if part == 'tab':
            pos, X = FX['u1dmavgH2Z']; Xd = X[[pos[i] for i in ids]]
            m = harness.tabpfn_new('v3.5', k, device); m.fit(Xd[tr], y[tr]); o['tab'][va] = m.predict(Xd[va])
            o['ridge'][va] = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000])).fit(Xd[tr], y[tr]).predict(Xd[va])
            lg = []
            for v in V:
                pv, Xv_ = FX[v]; Xx = Xv_[[pv[i] for i in ids]]; lg.append(make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=2.0)).fit(Xx[tr], np.log(y[tr])).predict(Xx[va]))
            o['logridge'][va] = np.exp(np.mean(lg, 0)); XGd = XG[[pos0[i] for i in ids]]; o['gmm'][va], _ = gp['fit_predict'](XGd[tr], y[tr], XGd[va], 19001 + 101 * p_ + k)
        else:
            Xi = np.stack([IM[i] for i in ids]); o['cnn'][va] = np.mean([cnn_fit_predict(Xi[tr], y[tr], Xi[va], 1000 * s + 17 * k + p_) for s in range(4)], 0)
    for c in comps: out[c][p_] = o[c][jj]
    print(name, 'done', {c: round(float(np.sqrt(((out[c][p_] - y[jj]) ** 2).mean())), 3) for c in comps}, flush=True)
np.savez(_repo_io(f'lc-{part}-f{frac:g}.npz'), **out, ids=np.array(sid))
