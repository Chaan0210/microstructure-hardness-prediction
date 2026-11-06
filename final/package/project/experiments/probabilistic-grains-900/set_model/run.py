from __future__ import annotations

import gc
import os
import random

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

import numpy as np
import torch
from torch import nn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CONFIG = {
    "seed": 731503,
    "n_seeds": 1,
    "encoder_width": 32,
    "embedding_dim": 16,
    "head_width": 32,
    "head_dropout": 0.1,
    "batch_size": 32,
    "lr": 0.001,
    "weight_decay": 0.01,
    "correction_penalty": 0.001,
    "max_epochs": 160,
    "patience": 24,
    "inner_validation_fraction": 0.2,
    "inner_min_improvement_standardized_mse": 0.00001,
    "ridge_alphas": [1.0, 10.0, 100.0, 1000.0],
    "ridge_inner_folds": 4,
    "objective": "One direct hardness squared-error loss per original image",
    "skip": "Fixed training-only Ridge on raw object means/stds and globals",
    "architecture": "Context-conditioned phi, area-weighted mean/std, small head",
    "normalization": "Fit within each training subset; equal total object weight per image",
    "selection": "One inner image split selects epoch, then all outer-training images refit",
    "epoch_zero": "Ridge skip is eligible if learned correction does not improve inner loss",
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def raw_summaries(bank) -> np.ndarray:
    """Per-image deterministic summaries, without fitting to another image."""
    rows = []
    x = bank["objects"]
    for i, (a, b) in enumerate(zip(bank["offsets"][:-1], bank["offsets"][1:])):
        z, w = x[a:b], bank["weights"][a:b, None]
        finite = np.isfinite(z)
        mass = (w * finite).sum(axis=0)
        mu = (np.where(finite, z, 0.0) * w).sum(axis=0) / np.maximum(mass, 1e-15)
        var = (np.where(finite, (z - mu) ** 2, 0.0) * w).sum(axis=0) / np.maximum(mass, 1e-15)
        mu[mass == 0] = np.nan
        sd = np.sqrt(np.maximum(var, 0.0))
        sd[mass == 0] = np.nan
        rows.append(np.r_[mu, sd, bank["global_X"][i]])
    return np.asarray(rows)


def fit_norm(x: np.ndarray, weights=None):
    finite = np.isfinite(x)
    w = np.ones((len(x), 1)) if weights is None else np.asarray(weights)[:, None]
    mass = (w * finite).sum(axis=0)
    mean = (w * np.where(finite, x, 0.0)).sum(axis=0) / np.maximum(mass, 1e-15)
    var = (w * np.where(finite, (x - mean) ** 2, 0.0)).sum(axis=0) / np.maximum(mass, 1e-15)
    scale = np.sqrt(np.maximum(var, 0.0))
    scale[scale < 1e-8] = 1.0
    return {"mean": mean, "scale": scale}


def transform_norm(x, norm):
    return ((np.where(np.isfinite(x), x, norm["mean"]) - norm["mean"]) / norm["scale"]).astype(np.float32)


def object_rows(bank, images):
    return np.concatenate([np.arange(bank["offsets"][i], bank["offsets"][i + 1]) for i in images])


def fit_context(bank, train_idx, all_y=None):
    """All fitted state has this single explicit image-scope boundary."""
    train_idx = np.asarray(train_idx, dtype=int)
    y = bank["y"] if all_y is None else np.asarray(all_y)
    rows = object_rows(bank, train_idx)
    obj_norm = fit_norm(bank["objects"][rows], bank["weights"][rows])
    global_norm = fit_norm(bank["global_X"][train_idx])
    pipeline = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge())
    cv = KFold(n_splits=CONFIG["ridge_inner_folds"], shuffle=True, random_state=CONFIG["seed"])
    search = GridSearchCV(pipeline, {"ridge__alpha": CONFIG["ridge_alphas"]},
                          cv=cv, scoring="neg_mean_squared_error", n_jobs=1, refit=True)
    search.fit(bank["summary"][train_idx], y[train_idx])
    return {
        "object_norm": obj_norm,
        "global_norm": global_norm,
        "ridge": search.best_estimator_,
        "ridge_alpha": float(search.best_params_["ridge__alpha"]),
        "y_mean": float(y[train_idx].mean()),
        "y_scale": float(max(y[train_idx].std(), 1e-8)),
        "training_ids": bank["ids"][train_idx].tolist(),
    }


class TensorData:
    def __init__(self, bank, context, device, all_y=None):
        self.device = device
        self.offsets = bank["offsets"]
        self.objects = torch.as_tensor(transform_norm(bank["objects"], context["object_norm"]), device=device)
        self.weights = torch.as_tensor(bank["weights"].astype(np.float32), device=device)
        self.globals = torch.as_tensor(transform_norm(bank["global_X"], context["global_norm"]), device=device)
        pred = context["ridge"].predict(bank["summary"])
        self.skip = torch.as_tensor(((pred - context["y_mean"]) / context["y_scale"]).astype(np.float32), device=device)
        y = bank["y"] if all_y is None else np.asarray(all_y)
        self.y = torch.as_tensor(((y - context["y_mean"]) / context["y_scale"]).astype(np.float32), device=device)

    def batch(self, images, *, reorder=False, padding=False):
        images = np.asarray(images, dtype=int)
        rows = np.concatenate([np.arange(self.offsets[i], self.offsets[i + 1]) for i in images])
        counts = self.offsets[images + 1] - self.offsets[images]
        group = np.repeat(np.arange(len(images)), counts)
        if reorder:
            perm = np.random.default_rng(4813).permutation(len(rows))
            rows, group = rows[perm], group[perm]
        ri = torch.as_tensor(rows, device=self.device)
        bi = torch.as_tensor(group, device=self.device)
        ii = torch.as_tensor(images, device=self.device)
        x, w = self.objects[ri], self.weights[ri]
        if padding:
            # Zero-weight inference-only padding. Never used to train.
            x = torch.cat([x, torch.zeros((len(images), x.shape[1]), device=self.device)])
            w = torch.cat([w, torch.zeros(len(images), device=self.device)])
            bi = torch.cat([bi, torch.arange(len(images), device=self.device)])
        return (x, w, bi, self.globals[ii], self.skip[ii]), self.y[ii]


class GrainSet(nn.Module):
    def __init__(self, object_dim, global_dim):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(object_dim + global_dim, CONFIG["encoder_width"]), nn.SiLU(),
                                 nn.Linear(CONFIG["encoder_width"], CONFIG["embedding_dim"]), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(2 * CONFIG["embedding_dim"] + global_dim, CONFIG["head_width"]),
                                  nn.SiLU(), nn.Dropout(CONFIG["head_dropout"]), nn.Linear(CONFIG["head_width"], 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, objects, weights, group, globals_, skip):
        z = self.phi(torch.cat([objects, globals_[group]], dim=1))
        mean = torch.zeros((len(globals_), z.shape[1]), device=z.device, dtype=z.dtype)
        second = torch.zeros_like(mean)
        mean.index_add_(0, group, weights[:, None] * z)
        second.index_add_(0, group, weights[:, None] * z.square())
        sd = (second - mean.square()).clamp(min=1e-8).sqrt()
        correction = self.head(torch.cat([mean, sd, globals_], dim=1)).squeeze(-1)
        return skip + correction, correction


def make_model(bank, device, seed):
    seed_all(seed)
    return GrainSet(bank["objects"].shape[1], bank["global_X"].shape[1]).to(device)


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])


def update_batch(model, optimizer, batch):
    model.train()
    inputs, y = batch
    optimizer.zero_grad(set_to_none=True)
    prediction, correction = model(*inputs)
    loss = (prediction - y).square().mean() + CONFIG["correction_penalty"] * correction.square().mean()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return float(loss.detach().cpu())


def epoch(model, optimizer, data, train_idx, rng):
    order = rng.permutation(train_idx)
    total = 0.0
    for k in range(0, len(order), CONFIG["batch_size"]):
        ids = order[k:k + CONFIG["batch_size"]]
        total += update_batch(model, optimizer, data.batch(ids)) * len(ids)
    return total / len(order)


@torch.no_grad()
def predict_normalized(model, data, images):
    model.eval()
    rows = []
    for k in range(0, len(images), CONFIG["batch_size"]):
        inputs, _ = data.batch(images[k:k + CONFIG["batch_size"]])
        rows.append(model(*inputs)[0].cpu().numpy())
    return np.concatenate(rows)


def predict_y(model, data, context, images):
    return predict_normalized(model, data, images) * context["y_scale"] + context["y_mean"]


def train_with_inner_stop(bank, outer_train, device, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(outer_train)
    n_validation = max(1, round(len(order) * CONFIG["inner_validation_fraction"]))
    inner_validation, inner_train = order[:n_validation], order[n_validation:]
    context = fit_context(bank, inner_train)
    data = TensorData(bank, context, device)
    model = make_model(bank, device, seed)
    optimizer = optimizer_for(model)
    y_inner = (bank["y"][inner_validation] - context["y_mean"]) / context["y_scale"]
    initial = float(np.mean((predict_normalized(model, data, inner_validation) - y_inner) ** 2))
    best_loss, best_epoch, stale = initial, 0, 0
    history = [{"epoch": 0, "inner_mse": initial, "train_loss": None}]
    for e in range(1, CONFIG["max_epochs"] + 1):
        train_loss = epoch(model, optimizer, data, inner_train, rng)
        val_loss = float(np.mean((predict_normalized(model, data, inner_validation) - y_inner) ** 2))
        history.append({"epoch": e, "inner_mse": val_loss, "train_loss": train_loss})
        if val_loss < best_loss - CONFIG["inner_min_improvement_standardized_mse"]:
            best_loss, best_epoch, stale = val_loss, e, 0
        else:
            stale += 1
        if stale >= CONFIG["patience"]:
            break
    result = {
        "selected_epochs": best_epoch,
        "inner_best_standardized_mse": best_loss,
        "inner_initial_standardized_mse": initial,
        "inner_train_ids": bank["ids"][inner_train].tolist(),
        "inner_validation_ids": bank["ids"][inner_validation].tolist(),
        "inner_ridge_alpha": context["ridge_alpha"],
        "history": history,
    }
    del data, model, optimizer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return result


def refit(bank, outer_train, device, seed, n_epochs):
    context = fit_context(bank, outer_train)
    data = TensorData(bank, context, device)
    model = make_model(bank, device, seed)
    optimizer = optimizer_for(model)
    rng = np.random.default_rng(seed)
    history = []
    for e in range(1, n_epochs + 1):
        history.append({"epoch": e, "train_loss": epoch(model, optimizer, data, outer_train, rng)})
    return model, context, data, history


def save_checkpoint(path, bank, model, context, metadata):
    torch.save({"config": CONFIG, "object_keys": bank["object_keys"].astype(str).tolist(),
                "global_keys": bank["global_keys"].astype(str).tolist(), "context": context,
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "metadata": metadata}, path)
