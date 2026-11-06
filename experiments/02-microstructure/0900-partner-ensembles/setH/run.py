"""Small train-only grain-set regression, with nested epoch selection.

Run from the repository root. Every write is scoped to this new experiment.
The object bank contains only original competition train images and frozen
zero-shot masks. No object/crop gets its own copied hardness training label.
"""

from __future__ import annotations
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

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


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
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
    "fixed_blend_new_predictor_weight": 0.2,
    "objective": "One direct hardness squared-error loss per original image",
    "skip": "Fixed training-only Ridge on raw object means/stds and globals",
    "architecture": "Context-conditioned phi, area-weighted mean/std, small head",
    "normalization": "Fit within each training subset; equal total object weight per image",
    "selection": "One inner image split selects epoch, then all outer-training images refit",
    "epoch_zero": "Ridge skip is eligible if learned correction does not improve inner loss",
}


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def load_inputs(bank_path: Path, base_path: Path):
    bank = {k: v for k, v in np.load(_repo_io(bank_path), allow_pickle=True).items()}
    base = {k: v for k, v in np.load(_repo_io(base_path), allow_pickle=True).items()}
    required = {"ids", "offsets", "objects", "object_keys", "weights", "global_X", "global_keys", "y"}
    assert required <= set(bank), required - set(bank)
    bank["ids"] = bank["ids"].astype(str)
    base["ids"] = base["ids"].astype(str)
    bank["objects"] = np.asarray(bank["objects"], dtype=np.float64)
    bank["weights"] = np.asarray(bank["weights"], dtype=np.float64)
    bank["global_X"] = np.asarray(bank["global_X"], dtype=np.float64)
    bank["offsets"] = np.asarray(bank["offsets"], dtype=np.int64)
    bank["y"] = np.asarray(bank["y"], dtype=np.float64).reshape(-1)
    base["y"] = np.asarray(base["y"], dtype=np.float64).reshape(-1)
    assert len(set(bank["ids"])) == len(bank["ids"])
    assert len(set(base["ids"])) == len(base["ids"])
    assert bank["offsets"].shape == (len(bank["ids"]) + 1,)
    assert bank["offsets"][0] == 0
    assert bank["offsets"][-1] == len(bank["objects"]) == len(bank["weights"])
    assert np.all(np.diff(bank["offsets"]) > 0), "Represent empty masks with explicit whole-image fallback"
    assert bank["objects"].shape[1] == len(bank["object_keys"])
    assert bank["global_X"].shape == (len(bank["ids"]), len(bank["global_keys"]))
    assert np.isfinite(bank["y"]).all()
    assert np.isfinite(bank["weights"]).all() and np.min(bank["weights"]) >= 0
    for a, b in zip(bank["offsets"][:-1], bank["offsets"][1:]):
        mass = bank["weights"][a:b].sum()
        assert mass > 0
        bank["weights"][a:b] /= mass
    pos = {i: j for j, i in enumerate(bank["ids"])}
    matched = np.array([pos[i] for i in base["ids"]])
    assert np.allclose(bank["y"][matched], base["y"], atol=1e-5)
    assert base["fold"].shape == (3, len(matched))
    assert all(set(row.tolist()) == set(range(5)) for row in base["fold"])
    bank["summary"] = raw_summaries(bank)
    return bank, base, matched


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


def benchmark(bank, matched, base, output):
    train_idx = matched[base["fold"][0] != 0]
    context = fit_context(bank, train_idx)
    batch_idx = train_idx[:CONFIG["batch_size"]]
    report = {"uses_outer_training_only": True, "training_ids": bank["ids"][train_idx].tolist(),
              "torch": torch.__version__, "threads": torch.get_num_threads(), "devices": {}}
    for device in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        data = TensorData(bank, context, device)
        model = make_model(bank, device, CONFIG["seed"])
        opt = optimizer_for(model)
        for _ in range(3):
            update_batch(model, opt, data.batch(batch_idx))
        synchronize(device)
        t0 = time.perf_counter()
        for _ in range(12):
            update_batch(model, opt, data.batch(batch_idx))
        synchronize(device)
        elapsed = time.perf_counter() - t0
        report["devices"][device] = {"seconds_per_batch": elapsed / 12,
                                           "parameters": sum(p.numel() for p in model.parameters())}
        del data, model, opt
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    report["selected_device"] = min(report["devices"], key=lambda d: report["devices"][d]["seconds_per_batch"])
    save_json(output / "benchmark.json", report)
    print(json.dumps({"benchmark": report["devices"], "selected_device": report["selected_device"]}), flush=True)
    return report["selected_device"]


@torch.no_grad()
def invariance_audit(model, data, images, y_scale):
    model.eval()
    ordinary = model(*data.batch(images)[0])[0]
    reordered = model(*data.batch(images, reorder=True)[0])[0]
    padded = model(*data.batch(images, padding=True)[0])[0]
    a = float((ordinary - reordered).abs().max().cpu()) * y_scale
    b = float((ordinary - padded).abs().max().cpu()) * y_scale
    assert a < 0.002 and b < 0.002, (a, b)
    return {"reorder_max_abs_hardness": a, "zero_weight_padding_max_abs_hardness": b,
            "padding_is_inference_test_only": True}


def outside_label_audit(bank, train_idx, context, device):
    changed = bank["y"].copy()
    outside = np.ones(len(changed), dtype=bool)
    outside[train_idx] = False
    changed[outside] = 10000 + np.arange(outside.sum())
    altered = fit_context(bank, train_idx, all_y=changed)
    assert context["training_ids"] == altered["training_ids"]
    assert context["y_mean"] == altered["y_mean"] and context["y_scale"] == altered["y_scale"]
    for key in ("object_norm", "global_norm"):
        for field in ("mean", "scale"):
            assert np.array_equal(context[key][field], altered[key][field])
    skip_a = context["ridge"].predict(bank["summary"])
    skip_b = altered["ridge"].predict(bank["summary"])
    assert np.array_equal(skip_a, skip_b)
    batch_idx = train_idx[:CONFIG["batch_size"]]
    states, losses = [], []
    for ctx, y in ((context, bank["y"]), (altered, changed)):
        data = TensorData(bank, ctx, device, all_y=y)
        model = make_model(bank, device, CONFIG["seed"])
        optimizer = optimizer_for(model)
        seed_all(CONFIG["seed"] + 19)
        losses.append(update_batch(model, optimizer, data.batch(batch_idx)))
        states.append({k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()})
        del model, data, optimizer
    max_diff = max(float(np.max(np.abs(states[0][k] - states[1][k]))) for k in states[0])
    assert max_diff < 2e-5 and abs(losses[0] - losses[1]) < 2e-5, (max_diff, losses)
    return {"outside_training_labels_changed": int(outside.sum()), "normalization_identical": True,
            "ridge_predictions_identical": True, "single_training_step_parameter_max_abs_difference": max_diff,
            "training_step_losses": losses, "scope": "Fitted state and representative actual training update"}


def save_checkpoint(path, bank, model, context, metadata):
    torch.save({"config": CONFIG, "object_keys": bank["object_keys"].astype(str).tolist(),
                "global_keys": bank["global_keys"].astype(str).tolist(), "context": context,
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "metadata": metadata}, _repo_io(path))


def load_checkpoint_predict(path, bank, images, device):
    checkpoint = torch.load(_repo_io(path), map_location="cpu", weights_only=False)
    assert checkpoint["config"] == CONFIG
    assert checkpoint["object_keys"] == bank["object_keys"].astype(str).tolist()
    assert checkpoint["global_keys"] == bank["global_keys"].astype(str).tolist()
    model = make_model(bank, device, CONFIG["seed"])
    model.load_state_dict(checkpoint["model_state"])
    data = TensorData(bank, checkpoint["context"], device)
    out = predict_y(model, data, checkpoint["context"], images)
    return out, checkpoint, model, data


def baseline_share(base):
    p, g, b = base["p021"], base["grain"], base["baseline"]
    failed = np.broadcast_to(base["failed"], np.shape(b))
    valid = ~failed.astype(bool) & (np.abs(p - g) > 1e-5)
    share = float(np.median(((b - g)[valid] / (p - g)[valid])))
    error = float(np.max(np.abs((share * p + (1 - share) * g - b)[~failed.astype(bool)])))
    assert 0 <= share <= 1 and error < 2e-4, (share, error)
    return share


def metrics(base, pred, ridge):
    y = base["y"][None, :]
    failed = np.broadcast_to(base["failed"], pred.shape).astype(bool)
    share = baseline_share(base)
    report = {"baseline_p021_share": share, "fixed_predictor_weight": 0.2, "metrics": {}}
    all_p = {"baseline045": base["baseline"]}
    for name, raw in (("raw_summary_ridge", ridge), ("grain_set", pred)):
        p = np.where(failed, base["p021"], raw)
        all_p[name + "_standalone"] = p
        all_p[name + "_blend20"] = 0.8 * base["baseline"] + 0.2 * p
        all_p[name + "_same021_share"] = base["baseline"] + 0.2 * (1 - share) * np.where(failed, 0, p - base["grain"])
    for name, p in all_p.items():
        squared = (p - y) ** 2
        per_fold = [float(np.sqrt(squared[part, base["fold"][part] == fold].mean())) for part in range(3) for fold in range(5)]
        report["metrics"][name] = {"pooled_rmse": float(np.sqrt(squared.mean())),
                                   "partition_rmse": np.sqrt(squared.mean(axis=1)).tolist(),
                                   "fold_rmse": per_fold,
                                   "mse_improvement_over045": float(np.mean((base["baseline"] - y) ** 2 - squared))}
    return report, all_p


def run_train(bank, base, matched, bank_path, base_path, output, device):
    manifest = {"config": CONFIG, "script_sha256": fingerprint(Path(__file__)),
                "bank_sha256": fingerprint(bank_path), "base_sha256": fingerprint(base_path),
                "bank_path": str(bank_path.resolve()), "base_path": str(base_path.resolve()),
                "object_keys": bank["object_keys"].astype(str).tolist(), "global_keys": bank["global_keys"].astype(str).tolist(),
                "matched_ids": base["ids"].tolist(), "bank_images_excluded_from_cv": sorted(set(bank["ids"]) - set(base["ids"])),
                "frozen_before_outer_scores": True, "device": device,
                "validation_limit": "Reused legacy folds, conditional comparison; not fresh independent validation"}
    save_json(output / "manifest.json", manifest)
    pred = np.full(base["fold"].shape, np.nan)
    ridge = np.full_like(pred, np.nan)
    audits = {}
    split_names = [str(s) for s in base.get("split_names", ["original", "20261031", "20261101"])]
    t0 = time.perf_counter()
    for part in range(3):
        for fold in range(5):
            stem = f"{split_names[part]}-fold{fold}"
            train_local = np.flatnonzero(base["fold"][part] != fold)
            val_local = np.flatnonzero(base["fold"][part] == fold)
            train_idx, val_idx = matched[train_local], matched[val_local]
            assert len(train_idx) == 320 and len(val_idx) == 80
            assert not (set(bank["ids"][train_idx]) & set(bank["ids"][val_idx]))
            seed = CONFIG["seed"] + 100 * part + fold
            f0 = time.perf_counter()
            stopping = train_with_inner_stop(bank, train_idx, device, seed)
            model, context, data, refit_history = refit(bank, train_idx, device, seed, stopping["selected_epochs"])
            pred[part, val_local] = predict_y(model, data, context, val_idx)
            ridge[part, val_local] = context["ridge"].predict(bank["summary"][val_idx])
            meta = {"split": split_names[part], "fold": fold, "seed": seed,
                    "training_ids": bank["ids"][train_idx].tolist(), "validation_ids": bank["ids"][val_idx].tolist(),
                    "selected_epochs": stopping["selected_epochs"], "ridge_alpha": context["ridge_alpha"],
                    "parameter_count": sum(p.numel() for p in model.parameters())}
            checkpoint_path = output / f"{stem}.pt"
            save_checkpoint(checkpoint_path, bank, model, context, meta)
            loaded_p, _, loaded_model, loaded_data = load_checkpoint_predict(checkpoint_path, bank, val_idx, device)
            diff = float(np.max(np.abs(loaded_p - pred[part, val_local])))
            assert diff < 0.002
            audit = {"reload_max_abs_hardness": diff, **invariance_audit(loaded_model, loaded_data, val_idx[:8], context["y_scale"])}
            if part == 0 and fold == 0:
                audit["outside_label_audit"] = outside_label_audit(bank, train_idx, context, device)
            audits[stem] = audit
            record = {**meta, **stopping, "refit_history": refit_history, "audit": audit,
                      "elapsed_seconds": time.perf_counter() - f0}
            save_json(output / f"{stem}-history.json", record)
            np.savez_compressed(_repo_io(output / "oof-progress.npz"), ids=base["ids"], y=base["y"], fold=base["fold"],
                                prediction=pred, ridge=ridge, split_names=np.asarray(split_names))
            print(json.dumps({"fold_completed": stem, "selected_epochs": stopping["selected_epochs"],
                              "ridge_alpha": context["ridge_alpha"], "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
            del model, data, loaded_model, loaded_data
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
    assert np.isfinite(pred).all() and np.isfinite(ridge).all()
    report, derived = metrics(base, pred, ridge)
    report["elapsed_seconds"] = time.perf_counter() - t0
    report["audits"] = audits
    report["validation_limit"] = manifest["validation_limit"]
    save_json(output / "results.json", report)
    np.savez_compressed(_repo_io(output / "oof.npz"), ids=base["ids"], y=base["y"], fold=base["fold"], prediction=pred,
                        ridge=ridge, split_names=np.asarray(split_names), **derived)
    print(json.dumps({"completed": True, "elapsed_seconds": report["elapsed_seconds"],
                      "rmse": {k: v["pooled_rmse"] for k, v in report["metrics"].items()}}), flush=True)


def run_verify(bank, base, matched, output, device):
    saved = np.load(_repo_io(output / "oof.npz"))
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["bank_sha256"] == fingerprint(Path(manifest["bank_path"]))
    assert manifest["base_sha256"] == fingerprint(Path(manifest["base_path"]))
    assert np.array_equal(saved["ids"], base["ids"])
    report = {}
    for part, split in enumerate(saved["split_names"]):
        for fold in range(5):
            mask = base["fold"][part] == fold
            images = matched[mask]
            stem = f"{split}-fold{fold}"
            pred, checkpoint, model, data = load_checkpoint_predict(output / f"{stem}.pt", bank, images, device)
            assert checkpoint["metadata"]["validation_ids"] == bank["ids"][images].tolist()
            assert not (set(checkpoint["context"]["training_ids"]) & set(bank["ids"][images]))
            diff = float(np.max(np.abs(pred - saved["prediction"][part, mask])))
            # Independent CPU reload may sum objects in a different floating-point order than MPS.
            assert diff < 0.01, (stem, diff)
            report[stem] = {"reload_max_abs_hardness": diff,
                            **invariance_audit(model, data, images[:8], checkpoint["context"]["y_scale"])}
            del model, data
    save_json(output / "independent-verification.json", {"device": device, "all_passed": True, "folds": report})
    print(json.dumps({"independent_verification_passed": True, "device": device,
                      "max_reload_abs_hardness": max(x["reload_max_abs_hardness"] for x in report.values())}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["benchmark", "train", "verify"], default="train")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--bank", type=Path, default=_repo_path(ROOT, "object-bank-train.npz"))
    parser.add_argument("--base", type=Path, default=_repo_path(ROOT, "base.npz"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    bank, base, matched = load_inputs(args.bank, args.base)
    HERE.mkdir(parents=True, exist_ok=True)
    if args.mode == "benchmark":
        benchmark(bank, matched, base, HERE)
        return
    if args.device == "auto":
        if args.mode == "verify":
            device = "cpu"
        else:
            device = benchmark(bank, matched, base, HERE)
    else:
        device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this process; use CPU or authorized MPS-capable execution")
    if args.mode == "verify":
        run_verify(bank, base, matched, HERE, device)
    else:
        run_train(bank, base, matched, args.bank, args.base, HERE, device)


if __name__ == "__main__":
    main()
