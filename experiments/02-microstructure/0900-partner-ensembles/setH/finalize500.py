"""Frozen three-seed full-train refit and separately authorized test inference.

The fit mode never reads test pixels/features. The infer mode never loads train
labels or calls any fitting/optimization routine. No dummy test targets exist.
"""
from __future__ import annotations
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT

import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("grain_set_run", _repo_path(HERE, "run.py"))
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
ORIGINAL_CONFIG = copy.deepcopy(run.CONFIG)
SEEDS = [731503, 731509, 731527]
OUTPUT = _repo_path(HERE, "full500")


class InferenceTensorData:
    """Label-free adapter; batch returns an unused None instead of any targets."""
    def __init__(self, bank, context):
        self.device = "cpu"
        self.offsets = bank["offsets"]
        self.objects = run.torch.as_tensor(run.transform_norm(bank["objects"], context["object_norm"]))
        self.weights = run.torch.as_tensor(bank["weights"].astype(np.float32))
        self.globals = run.torch.as_tensor(run.transform_norm(bank["global_X"], context["global_norm"]))
        predicted = context["ridge"].predict(bank["summary"])
        self.skip = run.torch.as_tensor(((predicted - context["y_mean"]) / context["y_scale"]).astype(np.float32))

    def batch(self, images):
        images = np.asarray(images, int)
        rows = np.concatenate([np.arange(self.offsets[i], self.offsets[i + 1]) for i in images])
        group = np.repeat(np.arange(len(images)), self.offsets[images + 1] - self.offsets[images])
        ri, bi, ii = (run.torch.as_tensor(a) for a in (rows, group, images))
        return (self.objects[ri], self.weights[ri], bi, self.globals[ii], self.skip[ii]), None


def fit():
    bank_path, base_path = _repo_path(HERE.parent, "object-bank-train.npz"), _repo_path(HERE.parent, "base.npz")
    bank, _, _ = run.load_inputs(bank_path, base_path)
    train_idx = np.arange(len(bank["ids"]))
    assert len(train_idx) == 500 and all(i.startswith("TRAIN_") for i in bank["ids"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {"seeds": SEEDS, "equal_weights": [1 / 3] * 3, "base_config": ORIGINAL_CONFIG,
                "stage": "Full original 500-image train refit after frozen 3-seed confirmation",
                "inner_split": "400 train / 100 stopping validation, then refit all 500",
                "training_ids": bank["ids"].tolist(), "train_bank_sha256": run.fingerprint(bank_path),
                "model_script_sha256": run.fingerprint(_repo_path(HERE, "run.py")),
                "finalizer_script_sha256": run.fingerprint(Path(__file__)),
                "test_read_during_fit": False, "fixed_new_predictor_blend_weight": .2,
                "fallback_rule": "Legacy failure OR whole-image object-bank fallback -> p021"}
    run.save_json(_repo_path(OUTPUT, "manifest.json"), manifest)
    t0 = time.perf_counter()
    records = []
    for seed in SEEDS:
        run.CONFIG = {**copy.deepcopy(ORIGINAL_CONFIG), "seed": seed}
        stopping = run.train_with_inner_stop(bank, train_idx, "cpu", seed)
        model, context, data, refit_history = run.refit(bank, train_idx, "cpu", seed, stopping["selected_epochs"])
        meta = {"seed": seed, "training_ids": bank["ids"].tolist(), "n_training_images": 500,
                "selected_epochs": stopping["selected_epochs"], "ridge_alpha": context["ridge_alpha"],
                "parameter_count": sum(p.numel() for p in model.parameters()), "validation_ids": []}
        path = _repo_path(OUTPUT, f"seed{seed}.pt")
        run.save_checkpoint(path, bank, model, context, meta)
        prediction = run.predict_y(model, data, context, train_idx)
        loaded, _, reload_model, reload_data = run.load_checkpoint_predict(path, bank, train_idx, "cpu")
        difference = float(np.max(np.abs(prediction - loaded)))
        assert difference < .002
        audit = {"reload_max_abs_hardness": difference,
                 **run.invariance_audit(reload_model, reload_data, train_idx[:8], context["y_scale"])}
        record = {**meta, **stopping, "refit_history": refit_history, "audit": audit}
        run.save_json(_repo_path(OUTPUT, f"seed{seed}-history.json"), record)
        records.append({"seed": seed, "selected_epochs": stopping["selected_epochs"], "audit": audit})
        print(json.dumps(records[-1]), flush=True)
        del model, data, reload_model, reload_data
        run.gc.collect()
    run.save_json(_repo_path(OUTPUT, "fit-summary.json"), {"seeds": records, "elapsed_seconds": time.perf_counter() - t0,
                                               "test_read_during_fit": False})


def load_test_bank(path):
    with np.load(_repo_io(path), allow_pickle=False) as data:
        assert "y" not in data.files, "Inference bank must be label-free"
        bank = {k: data[k] for k in data.files}
    bank["ids"] = bank["ids"].astype(str)
    assert len(set(bank["ids"])) == len(bank["ids"]) and all(i.startswith("TEST_") for i in bank["ids"])
    bank["offsets"] = bank["offsets"].astype(np.int64)
    bank["objects"] = bank["objects"].astype(np.float64)
    bank["global_X"] = bank["global_X"].astype(np.float64)
    bank["weights"] = bank["weights"].astype(np.float64)
    assert bank["offsets"].shape == (len(bank["ids"]) + 1,)
    assert bank["offsets"][0] == 0 and bank["offsets"][-1] == len(bank["objects"])
    assert np.all(np.diff(bank["offsets"]) > 0)
    assert bank["objects"].shape[1] == len(bank["object_keys"])
    assert bank["global_X"].shape == (len(bank["ids"]), len(bank["global_keys"]))
    assert len(bank["weights"]) == len(bank["objects"]) and np.isfinite(bank["weights"]).all()
    for a, b in zip(bank["offsets"][:-1], bank["offsets"][1:]):
        assert bank["weights"][a:b].sum() > 0 and (bank["weights"][a:b] >= 0).all()
        bank["weights"][a:b] /= bank["weights"][a:b].sum()
    bank["summary"] = run.raw_summaries(bank)
    return bank


def infer(test_path, baseline_path):
    # Read the already-frozen full-train recipe before opening test feature values.
    manifest = json.loads((_repo_path(OUTPUT, "manifest.json")).read_text())
    assert manifest["seeds"] == SEEDS and manifest["base_config"] == ORIGINAL_CONFIG
    assert manifest["fixed_new_predictor_blend_weight"] == .2
    assert manifest["model_script_sha256"] == run.fingerprint(_repo_path(HERE, "run.py"))
    assert manifest["finalizer_script_sha256"] == run.fingerprint(Path(__file__))
    bank = load_test_bank(test_path)
    assert not (set(manifest["training_ids"]) & set(bank["ids"]))
    with np.load(_repo_io(baseline_path), allow_pickle=False) as d:
        baseline = {k: d[k] for k in d.files}
    ids = baseline["ids"].astype(str)
    assert set(ids) == set(bank["ids"]) and len(set(ids)) == len(ids)
    lookup = {i: j for j, i in enumerate(ids)}
    order = np.array([lookup[i] for i in bank["ids"]])
    baseline045 = baseline["baseline045"][order].reshape(-1)
    p021 = baseline["p021"][order].reshape(-1)
    failed_legacy = baseline["failed"][order].reshape(-1).astype(bool)
    obj_col = bank["object_keys"].astype(str).tolist().index("whole_image_fallback")
    failed_objects = np.array([np.any(bank["objects"][a:b, obj_col] > .5)
                               for a, b in zip(bank["offsets"][:-1], bank["offsets"][1:])])
    failed = failed_legacy | failed_objects
    all_pred = []
    for seed in SEEDS:
        run.CONFIG = {**copy.deepcopy(ORIGINAL_CONFIG), "seed": seed}
        checkpoint = run.torch.load(_repo_path(OUTPUT, f"seed{seed}.pt"), map_location="cpu", weights_only=False)
        assert checkpoint["config"] == run.CONFIG
        assert checkpoint["context"]["training_ids"] == manifest["training_ids"]
        assert checkpoint["object_keys"] == bank["object_keys"].astype(str).tolist()
        assert checkpoint["global_keys"] == bank["global_keys"].astype(str).tolist()
        model = run.make_model(bank, "cpu", seed)
        model.load_state_dict(checkpoint["model_state"])
        data = InferenceTensorData(bank, checkpoint["context"])
        all_pred.append(run.predict_y(model, data, checkpoint["context"], np.arange(len(ids))))
        del model, data
    all_pred = np.stack(all_pred)
    safe = np.where(failed[None], p021[None], all_pred)
    mean = safe.mean(0)
    blend = .8 * baseline045 + .2 * mean
    assert np.isfinite(all_pred).all() and np.isfinite(blend).all()
    np.savez_compressed(_repo_path(OUTPUT, "predictions.npz"), ids=bank["ids"], seeds=np.array(SEEDS),
                        seed_predictions_raw=all_pred, seed_predictions_fallback=safe,
                        set_mean_raw=all_pred.mean(0), set_mean_fallback=mean,
                        baseline045=baseline045, p021=p021, failed_legacy=failed_legacy,
                        failed_objects=failed_objects, failed=failed, blend20=blend)
    with (_repo_path(OUTPUT, "submission-grain-set-3seed-blend20.csv")).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "hardness"])
        writer.writerows(zip(bank["ids"], blend))
    run.save_json(_repo_path(OUTPUT, "inference-audit.json"), {
        "n_test_images": len(ids), "n_training_images": 500, "seeds": SEEDS,
        "legacy_failure_count": int(failed_legacy.sum()), "object_failure_count": int(failed_objects.sum()),
        "union_failure_count": int(failed.sum()), "label_free_inference_adapter": True,
        "fit_or_optimization_during_inference": False, "test_bank_sha256": run.fingerprint(test_path),
        "baseline_test_sha256": run.fingerprint(baseline_path), "fixed_new_predictor_blend_weight": .2,
        "set_seed_sd_mean": float(safe.std(0).mean()), "set_seed_sd_p90": float(np.quantile(safe.std(0), .9)),
        "submission_sha256": run.fingerprint(_repo_path(OUTPUT, "submission-grain-set-3seed-blend20.csv"))})
    print(json.dumps({"inference_complete": True, "n_test_images": len(ids), "fallbacks": int(failed.sum()),
                      "output": str(_repo_path(OUTPUT, "submission-grain-set-3seed-blend20.csv"))}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fit", "infer"], required=True)
    p.add_argument("--test-bank", type=Path, default=_repo_path(HERE.parent, "object-bank-test.npz"))
    p.add_argument("--baseline-test", type=Path, default=_repo_path(HERE.parent, "ridge", "final500", "predictions.npz"))
    args = p.parse_args()
    run.torch.set_num_threads(2)
    run.torch.set_num_interop_threads(1)
    if args.mode == "fit":
        fit()
    else:
        infer(args.test_bank, args.baseline_test)


if __name__ == "__main__":
    main()
