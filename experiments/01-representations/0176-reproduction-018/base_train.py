from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np,torch,json
from pathlib import Path
from sklearn.model_selection import GroupKFold
from compact_regression import BalancedKernel
from cnn_model import TinyCNN
CNN_SEED=20261003
INNER_SEED=20261005
WIDTHS=(.25,1.,4.)
ALPHAS=(.01,.1,1.)
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
def cnn_fold(X, y, tr, va, ids, fold, device, checkpoint):
    rng = np.random.default_rng(CNN_SEED + fold)
    torch.manual_seed(CNN_SEED + fold)
    mu, sd = float(X[tr].mean()), float(X[tr].std())
    ym, ys = float(y[tr].mean()), float(y[tr].std())
    if sd <= 0 or ys <= 0:
        raise ValueError('Degenerate CNN normalization')
    Xt, yt = (X[tr]-mu)/max(sd, 1e-6), (y[tr]-ym)/ys
    model = TinyCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80, eta_min=.0001)
    avg, curve = None, []
    for epoch in range(80):
        model.train()
        order, loss_sum = rng.permutation(len(tr)), 0.
        for first in range(0, len(tr), 32):
            ix = order[first:first+32]
            xb = np.rot90(Xt[ix], int(rng.integers(4)), axes=(2, 3))
            if rng.random() < .5:
                xb = xb[:, :, :, ::-1]
            xb = torch.from_numpy(np.ascontiguousarray(xb)).to(device)
            target = torch.tensor(yt[ix], dtype=torch.float32, device=device)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(xb), target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite CNN loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            loss_sum += float(loss.detach().cpu()) * len(ix)
        scheduler.step()
        curve.append({'epoch': epoch+1,
                      'augmented_train_rmse': float(np.sqrt(loss_sum/len(tr))*ys),
                      'lr_after_step': scheduler.get_last_lr()[0]})
        if epoch >= 75:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if avg is None:
                avg = {k: v/5 for k, v in state.items()}
            else:
                for k in avg:
                    avg[k] += state[k]/5
        if (epoch+1) % 20 == 0:
            print(json.dumps({'component': 'cnn', 'fold': fold, **curve[-1]}), flush=True)
    model.load_state_dict(avg)
    model.eval()
    def predict(indices):
        result = []
        with torch.no_grad():
            for first in range(0, len(indices), 32):
                xb = torch.from_numpy((X[indices[first:first+32]]-mu)/max(sd, 1e-6)).to(device)
                result.extend((model(xb).cpu().numpy()*ys+ym).tolist())
        return np.array(result)
    prediction, train_prediction = predict(va), predict(tr)
    torch.mps.synchronize() if device=='mps' else None
    torch.save({'state_dict': avg, 'normalization': {'x_mean': mu, 'x_std': sd,
                'y_mean': ym, 'y_std': ys}, 'fold': fold, 'seed': CNN_SEED+fold,
                'architecture': 'cnn_controls.TinyCNN', 'image_size': 128,
                'epochs': 80, 'average_last_epochs': 5,
                'train_ids': [ids[i] for i in tr], 'validation_ids': [ids[i] for i in va]}, _repo_io(checkpoint))
    summary = {'train_rmse': rmse(y[tr], train_prediction),
               'validation_rmse': rmse(y[va], prediction), 'curve': curve}
    del model, opt, avg
    if device == 'mps':
        torch.mps.empty_cache()
    return prediction, summary

def kernel_fold(X, y, groups, blocks, tr, va, fold, selection=None):
    selection = tr if selection is None else np.asarray(selection)
    inner = list(GroupKFold(n_splits=3, shuffle=True, random_state=INNER_SEED+fold)
                 .split(X[selection], y[selection], groups[selection]))
    scores = []
    for width in WIDTHS:
        for alpha in ALPHAS:
            pred = np.full(len(selection), np.nan)
            for ti, vi in inner:
                if set(groups[selection[ti]]) & set(groups[selection[vi]]):
                    raise ValueError('Inner group overlap')
                model = BalancedKernel(blocks, width, alpha).fit(X[selection[ti]], y[selection[ti]])
                pred[vi] = model.predict(X[selection[vi]])
            scores.append({'bandwidth': width, 'alpha': alpha, 'inner_rmse': rmse(y[selection], pred)})
    chosen = min(scores, key=lambda z: z['inner_rmse'])
    model = BalancedKernel(blocks, chosen['bandwidth'], chosen['alpha']).fit(X[tr], y[tr])
    return model.predict(X[va]), model, {'selected': chosen, 'inner_scores': scores}