"""Exactly two frozen-seed confirmations; retain the equal mean of all three.

This wrapper never changes run.py or its original fitted artifacts. The repeated
IDs and legacy partitions remain conditional development evidence.
"""
from __future__ import annotations
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("grain_set_run", _repo_path(HERE, "run.py"))
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
ORIGINAL_CONFIG = copy.deepcopy(run.CONFIG)
ALL_SEEDS = [731503, 731509, 731527]
EXTRA_SEEDS = ALL_SEEDS[1:]


def paired_gain(reference, candidate, y):
    old = ((reference - y) ** 2).mean(0)
    new = ((candidate - y) ** 2).mean(0)
    ix = np.random.default_rng(90020261012).integers(0, len(y), (5000, len(y)))
    boots = np.sqrt(old[ix].mean(1)) - np.sqrt(new[ix].mean(1))
    return {"rmse_gain": float(np.sqrt(old.mean()) - np.sqrt(new.mean())),
            "mse_gain": float((old - new).mean()),
            "conditional_id_bootstrap95": np.quantile(boots, [.025, .975]).tolist()}


def summarize(base):
    sys.path.insert(0, str(HERE.parent))
    from common import evaluate

    inputs = [_repo_path(HERE, "oof.npz")] + [_repo_path(HERE, f"seed{s}", "oof.npz") for s in EXTRA_SEEDS]
    bundles = [{k: v for k, v in np.load(_repo_io(p), allow_pickle=False).items()} for p in inputs]
    for b in bundles:
        for key in ("ids", "y", "fold"):
            assert np.array_equal(b[key], bundles[0][key])
    predictions = np.stack([b["prediction"] for b in bundles])
    ridges = np.stack([b["ridge"] for b in bundles])
    mean_prediction, mean_ridge = predictions.mean(0), ridges.mean(0)
    report, derived = run.metrics(base, mean_prediction, mean_ridge)
    q = np.nanquantile(base["noise"], [1 / 3, 2 / 3])
    groups = {"all": np.ones(len(base["y"]), bool),
              "clean": base["noise"] < q[0],
              "middle": (base["noise"] >= q[0]) & (base["noise"] < q[1]),
              "noisy": base["noise"] >= q[1],
              "noisy_coarse_medium": (base["noise"] >= q[1]) & (base["Nproxy"] < 300)}
    safe_predictions = np.where(base["failed"][None], base["p021"][None], predictions)
    seed_variance = safe_predictions.var(axis=0, ddof=0)
    report.update({"all_seeds_retained": ALL_SEEDS, "equal_weights": [1 / 3] * 3,
                   "selection": "Two predetermined extra seeds; all three always retained",
                   "validation_limit": "Same reused 400 IDs and partitions; replication is not new independent validation",
                   "evaluations": {k: evaluate(v) for k, v in derived.items()},
                   "paired_against_original_seed": {}, "seed_variability": {},
                   "per_seed_rmse": {str(s): {k: float(np.sqrt(np.mean((b[k] - base["y"]) ** 2)))
                                              for k in ("grain_set_standalone", "grain_set_blend20", "grain_set_same021_share")}
                                     for s, b in zip(ALL_SEEDS, bundles)}})
    for name in ("grain_set_standalone", "grain_set_blend20", "grain_set_same021_share"):
        report["paired_against_original_seed"][name] = paired_gain(bundles[0][name], derived[name], base["y"])
    for name, mask in groups.items():
        per_image_sd = np.sqrt(seed_variance[:, mask].mean(0))
        report["seed_variability"][name] = {
            "n_images": int(mask.sum()), "set_seed_sd_mean": float(per_image_sd.mean()),
            "set_seed_sd_p90": float(np.quantile(per_image_sd, .9)),
            "blend20_seed_sd_mean": float(.2 * per_image_sd.mean()),
            "population_seed_variance_mean": float(seed_variance[:, mask].mean())}
    errors = safe_predictions - base["y"][None, None]
    report["seed_error_correlations"] = np.corrcoef(errors.reshape(3, -1)).tolist()
    report["averaging_identity"] = {
        "average_individual_seed_mse": float(np.mean(errors ** 2)),
        "equal_mean_mse": float(np.mean((safe_predictions.mean(0) - base["y"]) ** 2)),
        "difference_equals_population_seed_variance": float(seed_variance.mean())}
    report["predictive_variance_limit"] = "Seed disagreement only; not calibrated image measurement or predictive uncertainty"
    report["standalone_learned_gain_over_equal_mean_ridge"] = paired_gain(
        derived["raw_summary_ridge_standalone"], derived["grain_set_standalone"], base["y"])
    epochs = {}
    for s, folder in zip(ALL_SEEDS, [HERE] + [_repo_path(HERE, f"seed{s}") for s in EXTRA_SEEDS]):
        histories = [json.loads(p.read_text()) for p in sorted(folder.glob("*-history.json"))]
        epochs[str(s)] = [{k: h[k] for k in ("split", "fold", "selected_epochs", "ridge_alpha")} for h in histories]
    report["epochs"] = epochs
    run.save_json(_repo_path(HERE, "replication-results.json"), report)
    np.savez_compressed(_repo_path(HERE, "replicated-oof.npz"), ids=base["ids"], y=base["y"], fold=base["fold"],
                        split_names=base["split_names"], prediction=mean_prediction, ridge=mean_ridge,
                        all_seeds=np.array(ALL_SEEDS), seed_predictions=predictions, seed_ridges=ridges, **derived)
    print(json.dumps({"three_seed_rmse": {k: v["pooled_rmse"] for k, v in report["metrics"].items()},
                      "per_seed": report["per_seed_rmse"], "all_seeds_retained": ALL_SEEDS}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "verify", "summarize"], default="train")
    args = p.parse_args()
    run.torch.set_num_threads(2)
    run.torch.set_num_interop_threads(1)
    bank_path, base_path = _repo_path(HERE, "object-bank-train.npz"), _repo_path(HERE.parents[1], "probabilistic-grains-900/base.npz")
    bank, base, matched = run.load_inputs(bank_path, base_path)
    if args.mode == "summarize":
        summarize(base)
        return
    if args.mode == "train":
        run.save_json(_repo_path(HERE, "replication-plan.json"), {
            "all_seeds": ALL_SEEDS, "additional_seeds": EXTRA_SEEDS, "original_config": ORIGINAL_CONFIG,
            "aggregation": "Equal mean of all three, regardless of individual validation scores",
            "script_sha256": run.fingerprint(Path(__file__)), "model_script_sha256": run.fingerprint(_repo_path(HERE, "run.py")),
            "architecture_grid_epoch_budget_unchanged": True,
            "stage": "Predetermined seed confirmation after original single-seed result"})
    for seed in EXTRA_SEEDS:
        run.CONFIG = {**copy.deepcopy(ORIGINAL_CONFIG), "seed": seed}
        output = _repo_path(HERE, f"seed{seed}")
        output.mkdir(parents=True, exist_ok=True)
        if args.mode == "train":
            run.run_train(bank, base, matched, bank_path, base_path, output, "cpu")
        else:
            run.run_verify(bank, base, matched, output, "cpu")
    run.CONFIG = copy.deepcopy(ORIGINAL_CONFIG)
    if args.mode == "verify":
        summarize(base)


if __name__ == "__main__":
    main()
