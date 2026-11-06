"""Train-only, fixed-protocol Ridge confirmation; writes only beside this file."""
from __future__ import annotations
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT

import hashlib
import json
import os
import platform
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import joblib
import numpy as np
import scipy
import sklearn
from scipy.optimize import nnls
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[4]
E = _RepoPath(__file__).resolve().parents[3]
PARTS = ("original", "20261031", "20261101")
VARIANTS = ("u1dm1szH2Z", "u1dm2szH2Z", "u1dmavgH2Z", "u1dmavg4H2Z")
ALPHAS = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
BOOTSTRAPS = 10000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def aligned_indices(source_ids, target_ids):
    source_ids = np.asarray(source_ids).astype(str)
    target_ids = np.asarray(target_ids).astype(str)
    assert len(set(source_ids)) == len(source_ids), "duplicate source ID"
    assert len(set(target_ids)) == len(target_ids), "duplicate target ID"
    pos = {s: i for i, s in enumerate(source_ids)}
    return np.array([pos[s] for s in target_ids], dtype=int)


def load_inputs():
    feature_path = _repo_path(E, "push-800/eda/resid045.npz")
    source_feature_path = _repo_path(E, "push-800/feat-u1dmavgH2Z-anghcor-train.npz")
    feature = np.load(_repo_io(feature_path), allow_pickle=False)
    ids, x, keys = feature["ids"].astype(str), feature["X"].astype(float), feature["keys"].astype(str)
    assert x.shape == (400, 98) and len(set(keys)) == len(keys)
    assert all(s.startswith("TRAIN_") for s in ids)
    source = np.load(_repo_io(source_feature_path), allow_pickle=False)
    ix = aligned_indices(source["ids"], ids)
    cx = aligned_indices(source["keys"], keys)
    assert np.array_equal(source["X"][ix][:, cx], x, equal_nan=True)
    # Simultaneously permuted rows and columns reconstruct the same design matrix.
    rng = np.random.default_rng(900)
    rp, cp = rng.permutation(len(source["ids"])), rng.permutation(len(source["keys"]))
    ix2 = aligned_indices(source["ids"][rp], ids)
    cx2 = aligned_indices(source["keys"][cp], keys)
    assert np.array_equal(source["X"][rp][:, cp][ix2][:, cx2], x, equal_nan=True)
    paths = {"038": [_repo_path(E, f"push-300/dev-{v}-v35.npz") for v in VARIANTS],
             "039": [_repo_path(E, f"push-600/dev-{v}-angh.npz") for v in VARIANTS],
             "044": [_repo_path(E, f"push-800/dev-{v}-anghcor.npz") for v in VARIANTS]}
    archives = {c: [np.load(_repo_io(p), allow_pickle=False) for p in ps] for c, ps in paths.items()}
    ref = archives["039"][0]
    ys, folds, p021s, failures, grain = [], [], [], [], []
    for part in PARTS:
        ri = aligned_indices(ref[f"{part}_ids"], ids)
        ry, rf, rq, rb = [ref[f"{part}_{s}"][ri] for s in ("y", "fold", "p021", "failed")]
        family = []
        for aa in archives.values():
            preds = []
            for a in aa:
                ai = aligned_indices(a[f"{part}_ids"], ids)
                for s, expected in (("y", ry), ("fold", rf), ("p021", rq), ("failed", rb)):
                    np.testing.assert_array_equal(a[f"{part}_{s}"][ai], expected)
                preds.append(a[f"{part}_g"][ai])
            family.append(np.mean(preds, axis=0))
        ys.append(ry)
        folds.append(rf)
        p021s.append(rq)
        failures.append(rb.astype(bool))
        grain.append(np.mean(family, axis=0))
    y, fold, p021, failed, grain = map(np.array, (ys, folds, p021s, failures, grain))
    assert np.array_equal(y[0], feature["y"])
    for yy in y[1:]:
        np.testing.assert_array_equal(yy, y[0])
    for ff in fold:
        u, n = np.unique(ff, return_counts=True)
        assert np.array_equal(u, np.arange(5)) and np.array_equal(n, np.full(5, 80))
    grain = np.where(failed, p021, grain)
    baseline = 0.2 * p021 + 0.8 * grain
    np.testing.assert_allclose(float(np.sqrt(np.mean((baseline-y)**2))), 9.9402401, atol=1e-6)
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in [feature_path, source_feature_path, *sum(paths.values(), [])]}
    return dict(ids=ids, x=x, keys=keys, y=y, fold=fold, p021=p021, failed=failed,
                grain=grain, baseline=baseline, input_hashes=hashes)


def ridge_fit(x_train, y_train, weights=None):
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          StandardScaler(), RidgeCV(alphas=ALPHAS))
    if weights is None:
        model.fit(x_train, y_train)
    else:
        model.fit(x_train, y_train, ridgecv__sample_weight=weights)
    return model


def variance_design(x_train, keys):
    nproxy = 65536.0 / np.exp(x_train[:, list(keys).index("m_la_mean")])
    noise = 255.0 * x_train[:, list(keys).index("noise")]
    design = np.c_[np.ones(len(x_train)), 1.0 / nproxy, noise**2]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    design = imputer.fit_transform(design)
    scale = np.maximum(np.mean(np.abs(design), axis=0), 1e-12)
    return design / scale, imputer, scale


def fit_fold(x_train, y_train, train_ids, keys, weighted):
    """The fitter never receives validation labels or validation covariates."""
    if not weighted:
        return {"model": ridge_fit(x_train, y_train), "weights": np.ones(len(y_train)), "inner_folds": []}
    inner_oof = np.full(len(y_train), np.nan)
    inner_folds = []
    for itr, iva in KFold(n_splits=4, shuffle=True, random_state=900).split(x_train):
        assert not set(train_ids[itr]).intersection(train_ids[iva])
        model = ridge_fit(x_train[itr], y_train[itr])
        inner_oof[iva] = model.predict(x_train[iva])
        inner_folds.append({"train_ids": train_ids[itr].tolist(), "validation_ids": train_ids[iva].tolist(),
                            "alpha": float(model[-1].alpha_)})
    assert np.isfinite(inner_oof).all()
    design, imputer, scale = variance_design(x_train, keys)
    coef, _ = nnls(design, (y_train - inner_oof)**2)
    variance = np.maximum(design @ coef, 1e-8)
    weights = 1.0 / variance
    weights /= weights.mean()
    weights = np.clip(weights, 0.25, 4.0)
    return {"model": ridge_fit(x_train, y_train, weights), "weights": weights,
            "inner_folds": inner_folds, "variance_imputer": imputer, "variance_scale": scale,
            "variance_coefficients_scaled": coef, "inner_oof": inner_oof}


def rmse(y, pred):
    return float(np.sqrt(np.mean((y-pred)**2)))


def paired_interval(base_se, other_se, mask):
    """Each resampled unit is an image ID with its three squared errors."""
    a, b = base_se[:, mask].mean(axis=0), other_se[:, mask].mean(axis=0)
    rng = np.random.default_rng(901)
    values = []
    for start in range(0, BOOTSTRAPS, 250):
        ix = rng.integers(len(a), size=(min(250, BOOTSTRAPS-start), len(a)))
        values.extend((np.sqrt(a[ix].mean(axis=1))-np.sqrt(b[ix].mean(axis=1))).tolist())
    return {"gain_rmse": float(np.sqrt(a.mean())-np.sqrt(b.mean())),
            "gain_rmse_ci95": np.quantile(values, [0.025, 0.975]).tolist(),
            "gain_mse": float((a-b).mean()), "bootstrap_unit": "image_id", "replicates": BOOTSTRAPS}


def evaluate(data, predictions):
    y, fold = data["y"], data["fold"]
    noise = data["x"][:, list(data["keys"]).index("noise")] * 255
    nproxy = 65536.0 / np.exp(data["x"][:, list(data["keys"]).index("m_la_mean")])
    threshold = float(np.nanquantile(noise, 2/3))
    worst5 = np.argsort(((y-data["baseline"])**2).mean(axis=0))[-5:]
    exclude5 = np.ones(len(noise), bool)
    exclude5[worst5] = False
    segments = {"all": np.ones(len(noise), bool),
                "high_noise": np.isfinite(noise) & (noise >= threshold),
                "high_noise_nproxy_lt300": np.isfinite(noise) & (noise >= threshold) & (nproxy < 300),
                "exclude_worst5_baseline": exclude5}
    out = {"diagnostic_definition": {"noise_threshold_gray": threshold, "high_noise_rule": ">= train-dev 2/3 quantile",
             "nproxy": "65536 / exp(m_la_mean)", "worst5_ids": data["ids"][worst5].tolist(),
             "segments_are_descriptive_only": True}, "methods": {}}
    base_se = (data["baseline"]-y)**2
    weight_se = (predictions["weight_only_084"]-y)**2
    for name, p in predictions.items():
        e = (p-y)**2
        bypart = {part: rmse(y[j], p[j]) for j, part in enumerate(PARTS)}
        byfold = {f"{part}/{k}": rmse(y[j, fold[j] == k], p[j, fold[j] == k])
                  for j, part in enumerate(PARTS) for k in range(5)}
        gains = sum(byfold[f"{part}/{k}"] < rmse(y[j, fold[j] == k], data["baseline"][j, fold[j] == k])
                    for j, part in enumerate(PARTS) for k in range(5))
        out["methods"][name] = {"rmse": rmse(y, p), "mse": float(e.mean()), "per_partition": bypart,
            "per_fold": byfold, "folds_improved_vs045": gains, "segments": {}}
        for sn, mask in segments.items():
            out["methods"][name]["segments"][sn] = {"n_image_ids": int(mask.sum()),
                "rmse": rmse(y[:, mask], p[:, mask]), "baseline_rmse": rmse(y[:, mask], data["baseline"][:, mask]),
                "vs045": paired_interval(base_se, e, mask), "vs_weight_only_084": paired_interval(weight_se, e, mask)}
    return out, segments


def main():
    start = time.time()
    model_dir = _repo_path(HERE, "models")
    if (_repo_path(HERE, "metrics.json")).exists():
        raise FileExistsError("Results already exist; refusing to overwrite this fixed experiment.")
    model_dir.mkdir(exist_ok=True)
    data = load_inputs()
    models_meta, validation_checks = [], []
    predictions = {"baseline045": data["baseline"],
                   "weight_only_084": np.where(data["failed"], data["p021"], .16*data["p021"]+.84*data["grain"])}
    for weighted in (False, True):
        name = "fgls_ridge" if weighted else "ridge"
        oof = np.full_like(data["y"], np.nan)
        for j, part in enumerate(PARTS):
            for k in range(5):
                tr, va = data["fold"][j] != k, data["fold"][j] == k
                train_ids, validation_ids = data["ids"][tr], data["ids"][va]
                assert len(train_ids) == 320 and len(validation_ids) == 80
                assert not set(train_ids).intersection(validation_ids)
                bundle = fit_fold(data["x"][tr], data["y"][j, tr], train_ids, data["keys"], weighted)
                pred = bundle["model"].predict(data["x"][va])
                oof[j, va] = pred
                bundle.update({"feature_keys": data["keys"], "train_ids": train_ids,
                    "validation_ids": validation_ids, "partition": part, "fold": k,
                    "input_hashes": data["input_hashes"]})
                path = model_dir / f"{name}_{part}_{k}.joblib"
                joblib.dump(bundle, _repo_io(path))
                loaded = joblib.load(_repo_io(path))
                np.testing.assert_array_equal(loaded["model"].predict(data["x"][va]), pred)
                # The outer validation labels are deliberately replaced and cannot
                # change fit results: fit_fold only accepts the training slice.
                poison = data["y"][j].copy()
                poison[va] = np.arange(va.sum()) * 1e8 - 1e10
                poisoned = fit_fold(data["x"][tr], poison[tr], train_ids, data["keys"], weighted)
                np.testing.assert_array_equal(poisoned["model"].predict(data["x"][va]), pred)
                expected_median = np.nanmedian(data["x"][tr], axis=0)
                expected_median = np.nan_to_num(expected_median, nan=0.0)
                np.testing.assert_array_equal(bundle["model"][0].statistics_, expected_median)
                validation_checks.append({"method": name, "partition": part, "fold": k,
                    "train_validation_disjoint": True, "reload_predictions_exact": True,
                    "heldout_label_poison_no_effect": True, "outer_training_only_median": True})
                models_meta.append({"method": name, "partition": part, "fold": k,
                    "file": str(path.relative_to(HERE)), "sha256": sha(path),
                    "alpha": float(bundle["model"][-1].alpha_), "train_ids": train_ids.tolist(),
                    "validation_ids": validation_ids.tolist(), "inner_folds": bundle["inner_folds"],
                    "weight_min": float(bundle["weights"].min()), "weight_max": float(bundle["weights"].max()),
                    "weight_mean": float(bundle["weights"].mean()),
                    "variance_coefficients_scaled": bundle.get("variance_coefficients_scaled", np.array([])).tolist()})
            print(f"{name} {part}: completed 5 folds and label-exclusion/reload checks", flush=True)
        assert np.isfinite(oof).all()
        oof = np.where(data["failed"], data["p021"], oof)
        predictions[name] = oof
        predictions[f"{name}_20"] = np.where(data["failed"], data["p021"], .8*data["baseline"]+.2*oof)
        predictions[f"{name}_same021share"] = np.where(data["failed"], data["p021"], .2*data["p021"]+.64*data["grain"]+.16*oof)
    metrics, segments = evaluate(data, predictions)
    np.savez_compressed(_repo_path(HERE, "predictions.npz"), ids=data["ids"], y=data["y"], fold=data["fold"],
        partitions=np.array(PARTS), failed=data["failed"], p021=data["p021"], grain045=data["grain"],
        **predictions, **{f"segment_{k}": v for k, v in segments.items()})
    json_write(_repo_path(HERE, "models_manifest.json"), models_meta)
    json_write(_repo_path(HERE, "validation.json"), {"all_checks_passed": True, "id_and_feature_alignment": True,
        "baseline_reproduced": True, "feature_row_column_permutation_invariance": True,
        "source_feature_archive_verified": True, "n_input_archives": len(data["input_hashes"]),
        "n_fold_models": len(models_meta), "checks": validation_checks})
    metrics.update({"protocol_sha256": sha(_repo_path(HERE, "METHOD.md")), "script_sha256": sha(Path(__file__)),
        "input_hashes": data["input_hashes"], "elapsed_seconds": time.time()-start,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
            "scipy": scipy.__version__, "device": "cpu", "threads": 1},
        "limitations": ["Reused development folds and previously selected representation; not fresh independent validation.",
            "Legacy feature extraction was not retrained separately within these folds.",
            "Paired intervals are descriptive and do not adjust for historical model/feature selection.",
            "No test predictions or leaderboard submission were made."]})
    json_write(_repo_path(HERE, "metrics.json"), metrics)
    lines = ["# Ridge confirmation results", "", "Conditional comparison on reused 400-image development data.", "",
             "| Method | Pooled RMSE | Gain vs 045 | Image-ID paired 95% interval | Better folds |",
             "|---|---:|---:|---|---:|"]
    for name, m in metrics["methods"].items():
        uncertainty = m["segments"]["all"]["vs045"]
        lo, hi = uncertainty["gain_rmse_ci95"]
        lines.append(f"| {name} | {m['rmse']:.6f} | {uncertainty['gain_rmse']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {m['folds_improved_vs045']}/15 |")
    lines.extend(["", "Positive gain means lower RMSE. All designs and blend weights were fixed in METHOD.md before scoring.",
        "The primary 20% blend decreases the old 021 component from 20% to 16%; the same021share comparator preserves it.",
        "The weight-only control measures the effect of that change without adding Ridge.", "",
        "All 30 fold models reproduced predictions exactly after reload. Held-out label poisoning left all fits unchanged.",
        "Inputs were aligned explicitly by image ID and feature name; permuting stored rows and columns reproduced the same design.",
        "Models, training/validation IDs, the weighted method's inner folds, input hashes, and all metrics are saved locally.", "",
        "These folds and legacy representations were already used in model development. The intervals do not adjust for historical selection.",
        "No public/private improvement is established; no test predictions or submission were produced."])
    (_repo_path(HERE, "RESULTS.md")).write_text("\n".join(lines)+"\n")
    for name, m in metrics["methods"].items():
        print(name, f"RMSE {m['rmse']:.7f}", m["segments"]["all"]["vs045"], flush=True)
    print(f"finished in {time.time()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
