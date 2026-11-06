"""Residual-CNN variants on the round4 full073 benchmark (original partition, 5 outer folds, cached cross-fitted teachers; A213 selector -> B107
coefficients; T320 refit -> V80). Same protocol as round4 cnn_experiment.run (aux arm), with free seed shift / epochs / width.
usage: cnnvar.py seed TAG SEEDSHIFT EPOCHS WIDTH     |    cnnvar.py calib TAG1,TAG2,...  (seed-mean + noise-regime calibration inside B)"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, os, json
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'): os.environ.setdefault(k, '2')
from pathlib import Path
import numpy as np, torch
from torch import nn
R4 = Path(_repo_io(str(_repo_path(_REPO_ROOT, 'experiments/04-residual-cnn/20261015-rmse-research/round4-full073')))); sys.path.insert(0, str(R4))
import cnn_experiment as e
from noise_calibration import select
c = e.c; r = e.r; OUT = _repo_path(Path(__file__).resolve().parent, 'cnnvar')
class Wide(c.ComplementCNN):
    def __init__(self, ch=(32, 64, 96, 128)):
        nn.Module.__init__(self); self.blocks = nn.ModuleList(); inc = 1
        for i, out in enumerate(ch):
            self.blocks.append(nn.Sequential(nn.Conv2d(inc, out, 7 if i == 0 else 3, stride=4 if i == 0 else 2, padding=3 if i == 0 else 1), nn.GroupNorm(4, out), nn.SiLU())); inc = out
        nimg = 2 + 2 * sum(ch); self.main = nn.Sequential(nn.Linear(nimg + 6, 64), nn.SiLU(), nn.Dropout(.25), nn.Linear(64, 1)); self.aux = nn.Linear(nimg, 1)
        nn.init.zeros_(self.main[-1].weight); nn.init.zeros_(self.main[-1].bias); nn.init.zeros_(self.aux.weight); nn.init.zeros_(self.aux.bias)
    def forward(self, x, context, arm):
        stats = [x.mean((2, 3)), (x.var((2, 3), correction=0) + 1e-6).sqrt()]; h = x
        for b in self.blocks: h = b(h); stats.extend([h.mean((2, 3)), (h.var((2, 3), correction=0) + 1e-6).sqrt()])
        image = torch.cat(stats, 1); return self.main(torch.cat([image, context], 1)).squeeze(1), self.aux(image).squeeze(1)
class MapCNN(c.ComplementCNN):
    def __init__(self, inch=5, ch=(16, 32, 48, 64)):
        nn.Module.__init__(self); self.blocks = nn.ModuleList(); inc = inch
        for i, out in enumerate(ch):
            self.blocks.append(nn.Sequential(nn.Conv2d(inc, out, 7 if i == 0 else 3, stride=4 if i == 0 else 2, padding=3 if i == 0 else 1), nn.GroupNorm(4, out), nn.SiLU())); inc = out
        nimg = 2 * inch + 2 * sum(ch); self.main = nn.Sequential(nn.Linear(nimg + 6, 64), nn.SiLU(), nn.Dropout(.25), nn.Linear(64, 1)); self.aux = nn.Linear(nimg, 1)
        nn.init.zeros_(self.main[-1].weight); nn.init.zeros_(self.main[-1].bias); nn.init.zeros_(self.aux.weight); nn.init.zeros_(self.aux.bias)
    forward = Wide.forward
MAPS = None
def add_maps(data, ids):
    global MAPS
    if MAPS is None:
        z = np.load(_repo_io(str(_repo_path(_REPO_ROOT, 'experiments/02-microstructure/0400-unet-maps/unet4maps-z1-train.npz')))); pos = {str(i): j for j, i in enumerate(z['ids'])}
        mp = z['maps'][[pos[str(i)] for i in ids]][:, [1, 2, 3, 7]].astype(np.float32)          # p_phase, p_pore, p_boundary, darkness z (label-free U-Net maps)
        MAPS = (np.repeat(np.repeat(mp, 2, 2), 2, 3) - 0.5).astype(np.float32) * 2.0
    data['images'] = np.concatenate([data['images'], MAPS], 1); return data
def seed_run(tag, shift, epochs, width):
    if width == 2: c.ComplementCNN = Wide
    if width == 3: c.ComplementCNN = MapCNN
    folder = _repo_path(OUT, tag); folder.mkdir(parents=True, exist_ok=True)
    d = c.load_data(); labels = d.pop('y'); gg = json.loads((_repo_path(R4, 'label-graph.json')).read_text())
    baseline = np.load(_repo_path(R4.parent, 'scope.npz'))['strict_blend'][0]; legacy = d['C73_cnn'][0]; non021 = (baseline - .16 * legacy) / .84
    strict_ridge = np.load(_repo_path(R4.parent, 'consistency-strict.npz'))['production'][0]; dev = list(d['dev'])
    for g in gg:
        k = g['fold']; A, B, T, V = [np.asarray(g[q], int) for q in ('A', 'B', 'T', 'V')]
        if (folder / f'fold{k}-outer.npz').exists(): continue
        ib = np.load(_repo_path(R4, f'teachers/fold{k}-outer0/predictions.npz'))
        sb, src = e.teacher_oof(d, g, 'deep', A); raw = e.raw_context(d, g, 'deep', sb, B, ib['base'], ib['tab'] - ib['ridge'])
        sel = e.cnn_data(d, A, B, labels[A].copy(), sb, src, k, 'weight_selector', raw); seed = 2026101501 + 100 * k + shift
        if width == 3: sel = add_maps(sel, d['ids'])
        p = folder / f'fold{k}-selector'; rb = np.load(_repo_io(p / 'predictions.npz'))['validation_prediction'] if (p / 'training.json').exists() else c.fit_one(sel, 'aux', seed, p, epochs)
        anchor = np.load(_repo_path(R4, f'anchors/fold{k}.npz')); tb = ib['base']; pb = .16 * anchor['prediction_B'] + .84 * tb
        eligible = ~d['fail'][0][np.array([dev.index(i) for i in B])]
        np.savez_compressed(_repo_io(folder / f'fold{k}-selection.npz'), B=B, base073=pb, non021=tb, residual=rb, eligible=eligible)
        ob, osrc = e.teacher_oof(d, g, 'outer', T); ix = np.array([dev.index(i) for i in V]); failed = d['fail'][0, ix]
        raw = e.raw_context(d, g, 'outer', ob, V, non021[ix], d['C73_tab'][0, ix] - strict_ridge[ix])
        fin = e.cnn_data(d, T, V, labels[T].copy(), ob, osrc, k, 'outer_refit', raw); p = folder / f'fold{k}-final'
        if width == 3: fin = add_maps(fin, d['ids'])
        rv = np.load(_repo_io(p / 'predictions.npz'))['validation_prediction'] if (p / 'training.json').exists() else c.fit_one(fin, 'aux', seed, p, epochs)
        pv = baseline[ix]; t = non021[ix]; rv = rv.copy(); rv[failed] = 0.; t[failed] = pv[failed]
        np.savez_compressed(_repo_io(folder / f'fold{k}-outer.npz'), V=V, dev_index=ix, ids=d['ids'][V], base=pv, non021=t, residual=rv)
        print('fold', k, 'done', tag, flush=True)
def calib(tags):
    d = dict(np.load(_repo_path(R4.parent, 'inputs.npz'))); y = d['y']; dev = d['dev']; out = {a: np.full(400, np.nan) for a in ('addition', 'replacement')}; base = np.full(400, np.nan); W = []
    for k in range(5):
        inner = [np.load(_repo_path(OUT, t, f'fold{k}-selection.npz')) if (_repo_path(OUT, t)).exists() else np.load(_repo_path(R4, t, f'fold{k}-selection.npz')) for t in tags]
        outer = [np.load(_repo_path(OUT, t, f'fold{k}-outer.npz')) if (_repo_path(OUT, t)).exists() else np.load(_repo_path(R4, t, f'fold{k}-outer.npz')) for t in tags]
        B = inner[0]['B']; V = outer[0]['V']; good = inner[0]['eligible']; rb = np.mean([z['residual'] for z in inner], 0); rv = np.mean([z['residual'] for z in outer], 0)
        band = d['noise'][B][good] >= 12; vband = (d['noise'][V] >= 12).astype(int); ix = outer[0]['dev_index']; base[ix] = outer[0]['base']; wr = {}
        for arm, key in (('addition', 'base073'), ('replacement', 'non021')):
            w, s = select(rb[good], (y[B] - inner[0][key])[good], band); bb = outer[0]['base'] if arm == 'addition' else outer[0]['non021']
            out[arm][ix] = bb + w[vband] * rv; wr[arm] = np.round(w, 3).tolist()
        W.append(wr)
    yd = y[dev]; rm = lambda a: float(np.sqrt(((a - yd) ** 2).mean())); nz = d['noise'][dev]
    fx = np.full(400, np.nan)
    for k in range(5):
        outer = [np.load(_repo_path(OUT, t, f'fold{k}-outer.npz')) if (_repo_path(OUT, t)).exists() else np.load(_repo_path(R4, t, f'fold{k}-outer.npz')) for t in tags]; ix = outer[0]['dev_index']
        fx[ix] = outer[0]['base'] + 0.5467904360508202 * (nz[ix] >= 12) * np.mean([z['residual'] for z in outer], 0)
    print(f'   fixed w=0.547 addition {rm(fx):.4f} (gain {rm(base) - rm(fx):+.4f}) noisy {np.sqrt(((fx - yd)[nz >= 12] ** 2).mean()):.3f}')
    print(f'{",".join(tags)}: base {rm(base):.4f} | addition {rm(out["addition"]):.4f} (gain {rm(base) - rm(out["addition"]):+.4f}) | replacement {rm(out["replacement"]):.4f} | noisy-band add RMSE {np.sqrt(((out["addition"] - yd)[nz >= 12] ** 2).mean()):.3f} | weights {[w["addition"] for w in W]}')
    return out
if __name__ == '__main__':
    torch.set_num_threads(2)
    if sys.argv[1] == 'seed': seed_run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    else: calib(sys.argv[2].split(','))
