"""Stage 4: point vs analytic uncertain-input RBF; execute after stage3 completes."""
from __future__ import annotations
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT

import argparse
import json
import pickle
import time
from pathlib import Path

from grain_measurement import (
    CONFIG, HERE, ExperimentData, expected_rbf, json_write, metrics,
    safe_output, sha256,
)
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from distribution_kernel import DistributionKernelModel


def cache_read(data, folder, tag, tr, va):
    file = folder/f"{tag}.npz"
    cache = dict(np.load(_repo_io(file), allow_pickle=False))
    if str(cache["cache_key"]) != data.cache_key or not np.array_equal(cache["train_idx"], tr) or not np.array_equal(cache["eval_idx"], va):
        raise ValueError(f"Feature cache identity/split mismatch: {file}")
    return cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=Path, default=HERE.parent/"object-bank-train.npz")
    p.add_argument("--base", type=Path, default=HERE.parent/"base.npz")
    p.add_argument("--stage2", type=Path, default=HERE/"stage2")
    p.add_argument("--output", type=Path, default=HERE/"stage4")
    p.add_argument("--stage3-complete", action="store_true", help="Explicit execution gate after stage3 has completed")
    args = p.parse_args()
    if not args.stage3_complete:
        raise RuntimeError("Stage4 follows stage3; pass --stage3-complete only after that stage finishes")
    data = ExperimentData(args.bank, args.base)
    complete = json.loads((args.stage2/"COMPLETE.json").read_text())
    if complete["cache_key"] != data.cache_key:
        raise ValueError("Stage2 features were generated using different inputs or measurement code")
    out = safe_output(args.output)
    models_dir = safe_output(out/"models")
    json_write(out/"provenance.json", {**data.provenance, "cache_key": data.cache_key,
                                      "stage4_code_sha256": sha256(__file__),
                                      "kernel_model_code_sha256": sha256(HERE/"distribution_kernel.py"),
                                      "stage2": str(args.stage2.resolve())})
    predictions = {name: np.full_like(data.baseline, np.nan) for name in ["point_rbf", "uncertain_rbf"]}
    records, start = [], time.time()
    for s in range(3):
        for f in range(5):
            tag = f"partition{s}-fold{f}"
            tr, va = np.flatnonzero(data.folds[s] != f), np.flatnonzero(data.folds[s] == f)
            inner = []
            kfold = KFold(n_splits=3, shuffle=True, random_state=19001+101*s+f)
            for k, (itr, iva) in enumerate(kfold.split(tr)):
                inner.append(cache_read(data, args.stage2/"features", f"{tag}-inner{k}", tr[itr], tr[iva]))
            outer = cache_read(data, args.stage2/"features", f"{tag}-outer", tr, va)
            record = {"partition": s, "fold": f, "selection": {}}
            for name in predictions:
                scales = [0.] if name == "point_rbf" else CONFIG["kernel_uncertainty_scale"]
                selection = []
                for scale in scales:
                    for length in CONFIG["kernel_lengths"]:
                        # Kernel matrices are shared by all alpha values.
                        errors = {a: [] for a in CONFIG["kernel_alpha"]}
                        for cache in inner:
                            m, v = cache["hierarchical_train_mean"], cache["hierarchical_train_var"]
                            em, ev = cache["hierarchical_eval_mean"], cache["hierarchical_eval_var"]
                            location, sd = m.mean(0), m.std(0)
                            sd = np.where(sd > 1e-8, sd, 1.)
                            z, zv, ez, ezv = (m-location)/sd, v/sd**2*scale, (em-location)/sd, ev/sd**2*scale
                            K = expected_rbf(z, zv, z, zv, length*np.sqrt(m.shape[1]))
                            Ke = expected_rbf(ez, ezv, z, zv, length*np.sqrt(m.shape[1]))
                            y = data.y[cache["train_idx"]]
                            mean_y = y.mean()
                            for alpha in CONFIG["kernel_alpha"]:
                                Ka = K.copy()
                                Ka[np.diag_indices_from(Ka)] += alpha
                                coef = cho_solve(cho_factor(Ka, lower=True, check_finite=False), y-mean_y, check_finite=False)
                                pred = Ke@coef+mean_y
                                errors[alpha].extend((pred-data.y[cache["eval_idx"]])**2)
                        for alpha in CONFIG["kernel_alpha"]:
                            selection.append({"scale": scale, "length": length, "alpha": alpha,
                                              "inner_mse": float(np.mean(errors[alpha]))})
                chosen = min(selection, key=lambda x: x["inner_mse"])
                model = DistributionKernelModel.fit(outer["hierarchical_train_mean"], outer["hierarchical_train_var"], data.y[tr],
                                                     chosen["length"], chosen["alpha"], chosen["scale"])
                predictions[name][s, va] = model.predict(outer["hierarchical_eval_mean"], outer["hierarchical_eval_var"])
                with (models_dir/f"{tag}-{name}.pkl").open("wb") as fh:
                    pickle.dump({"model": model, "train_ids": data.ids[tr], "validation_ids": data.ids[va],
                                 "selected": chosen, "measurement_cache": str(args.stage2/"features"/f"{tag}-outer.npz")}, fh)
                record["selection"][name] = {"selected": chosen, "all_candidates": selection}
            records.append(record)
            json_write(out/"inner-selection.json", records)
            print(f"{tag} kernels complete, elapsed {time.time()-start:.1f}s", flush=True)
    summary = {name: metrics(data, pred) for name, pred in predictions.items()}
    json_write(out/"summary.json", {"models": summary, "seconds": time.time()-start,
                                    "validation_scope": "reused development partitions; nested kernels; baseline fixed"})
    np.savez_compressed(_repo_io(out/"predictions.npz"), ids=data.ids, y=data.y, baseline=data.baseline, fold=data.folds, **predictions)
    json_write(out/"COMPLETE.json", {"cache_key": data.cache_key, "stage": 4, "seconds": time.time()-start,
                                     "predictions_sha256": sha256(out/"predictions.npz")})
    for name, value in summary.items():
        print(name, "standalone", round(value["standalone_rmse"], 6), "blend20", round(value["blend20_rmse"], 6),
              "same021", round(value["same021_blend20_rmse"], 6), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
