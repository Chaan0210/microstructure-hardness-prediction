"""Stage 2: frozen measurement controls, hierarchical model, nested Ridge readouts.

Run from repository root:
  .venv/bin/python experiments/probabilistic-grains-900/probabilistic/run_stage2.py
Use --smoke for a feature-only engineering check (no hardness scoring).
"""
from __future__ import annotations
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[5]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT

import argparse
import os
import pickle
import time
from pathlib import Path

from grain_measurement import (
    ARMS, CONFIG, HERE, ExperimentData, HierarchicalMeasurement, fold_features,
    json_write, metrics, physical_readout, safe_output, sha256, transform,
)
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def ridge(alpha):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), Ridge(alpha=alpha))


def features(data, cache, arm, kind, tag):
    x = physical_readout(cache[f"{arm}_{tag}_mean"], cache[f"{arm}_{tag}_var"])
    if kind == "x98_augmented":
        x = np.column_stack([data.X98[cache[f"{tag}_idx"]], x])
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, default=HERE.parents[1]/"object-bank-train.npz")
    parser.add_argument("--base", type=Path, default=HERE.parents[1]/"base.npz")
    parser.add_argument("--output", type=Path, default=HERE/"stage2")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark-first-fold", action="store_true", help="Complete one fold without reporting outer scores; cache reused by full run")
    args = parser.parse_args()
    out = safe_output(args.output)
    data = ExperimentData(args.bank, args.base)
    start = time.time()
    json_write(out/"provenance.json", {**data.provenance, "cache_key": data.cache_key,
                                      "stage2_code_sha256": sha256(__file__),
                                      "cache_compatibility_manifest_sha256": sha256(HERE/"cache-compatibility-order-extension.json")})
    if args.smoke:
        # First 20 training images in a prescribed fold; no labels used.
        train = np.flatnonzero(data.folds[0] != 0)[:20]
        valid = np.flatnonzero(data.folds[0] == 0)[:3]
        model = HierarchicalMeasurement.fit([data.images[i] for i in train])
        results = {}
        for arm in ARMS:
            mean, var, diag = transform([data.images[i] for i in valid], arm, model)
            assert mean.shape == (3, 15) and var.shape == (3, 15)
            assert np.all(var >= 0) and np.all((mean[:, 0] >= 0) & (mean[:, 0] <= 1))
            results[arm] = {"means": mean.tolist(), "variances": var.tolist(), "diagnostics": diag}
        # Numerical PSD verification on actual measured train-image objects.
        from grain_measurement import expected_rbf
        K = expected_rbf(mean, var, mean, var, 4.)
        min_eig = float(np.linalg.eigvalsh(K).min())
        assert min_eig > -1e-9 and np.allclose(K, K.T)
        record = {"mode": "feature_only_no_hardness_score", "training_ids": data.ids[train].tolist(),
                  "validation_ids": data.ids[valid].tolist(), "fit_history": model.fit_history,
                  "measurements": results, "kernel_min_eigenvalue": min_eig,
                  "seconds": time.time()-start}
        json_write(out/"smoke.json", record)
        print({"smoke": "passed", "seconds": record["seconds"], "mu": model.mu.tolist(),
               "tau": np.sqrt(model.tau2).tolist(), "kernel_min_eigenvalue": min_eig}, flush=True)
        return
    names = [f"{arm}__{kind}" for arm in ARMS for kind in CONFIG["ridge_inputs"]]
    predictions = {n: np.full_like(data.baseline, np.nan) for n in names}
    cache_dir, model_dir = safe_output(out/"features"), safe_output(out/"models")
    all_records = []
    for s in range(3):
        for f in range(5):
            tag = f"partition{s}-fold{f}"
            tr, va = np.flatnonzero(data.folds[s] != f), np.flatnonzero(data.folds[s] == f)
            kfold = KFold(n_splits=3, shuffle=True, random_state=19001+101*s+f)
            inner_caches = []
            for k, (itr, iva) in enumerate(kfold.split(tr)):
                cache = fold_features(data, tr[itr], tr[iva], cache_dir, f"{tag}-inner{k}")
                inner_caches.append(cache)
                print(f"{tag} inner{k} features ready, elapsed {time.time()-start:.1f}s", flush=True)
            outer = fold_features(data, tr, va, cache_dir, f"{tag}-outer")
            fold_records = {}
            for arm in ARMS:
                for kind in CONFIG["ridge_inputs"]:
                    name = f"{arm}__{kind}"
                    scores = []
                    for alpha in CONFIG["alpha_ridge"]:
                        errors = []
                        for cache in inner_caches:
                            fit = ridge(alpha)
                            fit.fit(features(data, cache, arm, kind, "train"), data.y[cache["train_idx"]])
                            p = fit.predict(features(data, cache, arm, kind, "eval"))
                            errors.extend((p-data.y[cache["eval_idx"]])**2)
                        scores.append(float(np.mean(errors)))
                    best = int(np.argmin(scores))
                    alpha = CONFIG["alpha_ridge"][best]
                    model = ridge(alpha)
                    model.fit(features(data, outer, arm, kind, "train"), data.y[tr])
                    predictions[name][s, va] = model.predict(features(data, outer, arm, kind, "eval"))
                    with (model_dir/f"{tag}-{name}.pkl").open("wb") as fh:
                        pickle.dump({"model": model, "train_ids": data.ids[tr], "validation_ids": data.ids[va],
                                     "arm": arm, "kind": kind, "alpha": alpha,
                                     "measurement_cache": str(cache_dir/f"{tag}-outer.npz")}, fh)
                    fold_records[name] = {"alpha": alpha, "inner_mse": dict(zip(map(str, CONFIG["alpha_ridge"]), scores))}
            all_records.append({"partition": s, "fold": f, "selection": fold_records})
            json_write(out/"inner-selection.json", all_records)
            np.savez_compressed(_repo_io(out/"predictions-partial.npz"), ids=data.ids, y=data.y, **predictions)
            # Keep outer scores unseen until all prescribed folds finish.
            print(f"{tag} complete, elapsed {time.time()-start:.1f}s", flush=True)
            if args.benchmark_first_fold:
                json_write(out/"BENCHMARK.json", {"seconds": time.time()-start, "completed_outer_folds": 1,
                                                 "outer_scores_opened": False, "cache_key": data.cache_key})
                return
    summary = {name: metrics(data, p) for name, p in predictions.items()}
    json_write(out/"summary.json", {"models": summary, "seconds": time.time()-start,
                                    "validation_scope": "reused development partitions; nested new readouts; baseline fixed"})
    np.savez_compressed(_repo_io(out/"predictions.npz"), ids=data.ids, y=data.y, baseline=data.baseline,
                        fold=data.folds, **predictions)
    json_write(out/"COMPLETE.json", {"cache_key": data.cache_key, "stage": 2, "seconds": time.time()-start,
                                     "models": names, "predictions_sha256": sha256(out/"predictions.npz")})
    for name, values in summary.items():
        print(name, "standalone", round(values["standalone_rmse"], 6),
              "blend20", round(values["blend20_rmse"], 6),
              "same021", round(values["same021_blend20_rmse"], 6), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=int(os.environ.get("PROB_GRAIN_THREADS", "2"))):
        main()
