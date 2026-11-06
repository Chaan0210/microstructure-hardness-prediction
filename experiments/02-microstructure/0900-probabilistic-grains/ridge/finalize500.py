"""Prepared finalizer. Run only after the parent task authorizes finalization.

Fits original 500 train rows, then infers fixed test features. Writes exclusively
to a new ridge/final500 directory. This script never submits or contacts a service.
"""
from __future__ import annotations
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT

import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

import joblib
import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[4]
OUT = _repo_path(HERE, "final500")
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
EXPECTED045_SHA = "7a9dbb257f9c599596769cfcade50f8dd3d988637879ff9896113c62138725c1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def align(source, target):
    assert len(set(source)) == len(source) and len(set(target)) == len(target)
    pos = {s: j for j, s in enumerate(source)}
    return np.array([pos[s] for s in target])


def read_prediction(path, order):
    with path.open() as f:
        records = list(csv.DictReader(f))
    ids = [r["ID"] for r in records]
    assert len(ids) == 1000 and set(ids) == set(order)
    values = np.array([float(r["hardness"]) for r in records])[align(ids, order)]
    assert np.isfinite(values).all()
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-authorized-finalization", action="store_true",
                        help="Execute the prepared frozen recipe after task authorization.")
    args = parser.parse_args()
    if not args.run_authorized_finalization:
        parser.error("Prepared only; execution requires --run-authorized-finalization.")
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing finalization: {OUT}")

    training_file = _repo_path(ROOT, "experiments/push-800/feat-u1dmavgH2Z-anghcor-train.npz")
    label_file = _repo_path(ROOT, "data/train.csv")
    protocol_file = _repo_path(HERE, "METHOD.md")
    with label_file.open() as f:
        records = list(csv.DictReader(f))
    train_ids = np.array([r["ID"] for r in records])
    y = np.array([float(r["hardness"]) for r in records])
    assert len(train_ids) == 500 and len(set(train_ids)) == 500
    assert all(i.startswith("TRAIN_") for i in train_ids) and np.isfinite(y).all()
    with np.load(_repo_io(training_file), allow_pickle=False) as d:
        assert set(d["ids"].astype(str)) == set(train_ids)
        keys = d["keys"].astype(str)
        x = d["X"][align(d["ids"].astype(str), train_ids)].astype(float)
    with np.load(_repo_path(ROOT, "experiments/push-800/eda/resid045.npz"), allow_pickle=False) as d:
        np.testing.assert_array_equal(keys, d["keys"].astype(str))
        np.testing.assert_array_equal(x[align(train_ids, d["ids"].astype(str))], d["X"])
    assert x.shape == (500, 98) and len(set(keys)) == 98
    assert not np.isinf(x).any()

    # Training and all preprocessing precede loading any test feature values.
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          StandardScaler(), RidgeCV(alphas=ALPHAS))
    model.fit(x, y)
    np.testing.assert_array_equal(model[0].statistics_, np.nan_to_num(np.nanmedian(x, axis=0), nan=0.0))
    trained_state = joblib.hash(model)
    bundle = {"model": model, "train_ids": train_ids, "feature_keys": keys,
              "training_features_sha256": sha(training_file), "labels_sha256": sha(label_file),
              "protocol_sha256": sha(protocol_file), "alpha_candidates": ALPHAS,
              "chosen_alpha": float(model[-1].alpha_), "training_scope": "500 original train image rows only"}

    test_file = _repo_path(ROOT, "experiments/push-800/feat-u1dmavgH2Z-anghcor-test.npz")
    sample_file = _repo_path(ROOT, "data/sample_submission.csv")
    old045_file = _repo_path(ROOT, "artifacts/submission-candidate-045.csv")
    old021_file = _repo_path(ROOT, "artifacts/submission-candidate-021.csv")
    assert sha(old045_file) == EXPECTED045_SHA, "The frozen 045 input changed"
    with sample_file.open() as f:
        test_ids = np.array([r["ID"] for r in csv.DictReader(f)])
    assert len(test_ids) == 1000 and len(set(test_ids)) == 1000
    assert all(i.startswith("TEST_") for i in test_ids)
    assert not set(test_ids).intersection(train_ids)
    with np.load(_repo_io(test_file), allow_pickle=False) as d:
        test_source_ids, test_keys = d["ids"].astype(str), d["keys"].astype(str)
        assert set(test_source_ids) == set(test_ids) and set(test_keys) == set(keys)
        xt = d["X"][align(test_source_ids, test_ids)][:, align(test_keys, keys)].astype(float)
    assert xt.shape == (1000, 98) and not np.isinf(xt).any()
    raw_ridge = model.predict(xt)
    assert joblib.hash(model) == trained_state, "Prediction changed the fitted model"
    old045 = read_prediction(old045_file, test_ids)
    old021 = read_prediction(old021_file, test_ids)
    failed = np.nan_to_num(xt[:, list(keys).index("n_obj")], nan=0.0) < 20
    np.testing.assert_allclose(old045[failed], old021[failed], rtol=0, atol=1e-8)
    ridge = np.where(failed, old021, raw_ridge)
    candidate = np.where(failed, old021, 0.8*old045 + 0.2*ridge)
    assert np.isfinite(candidate).all()

    OUT.mkdir()
    model_path = _repo_path(OUT, "ridge500.joblib")
    joblib.dump(bundle, _repo_io(model_path))
    loaded = joblib.load(_repo_io(model_path))
    np.testing.assert_array_equal(loaded["model"].predict(xt), raw_ridge)
    csv_path = _repo_path(OUT, "candidate-045-ridge20.csv")
    with csv_path.open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "hardness"])
        writer.writerows((i, repr(float(p))) for i, p in zip(test_ids, candidate))
    np.testing.assert_array_equal(read_prediction(csv_path, test_ids), candidate)
    np.savez_compressed(_repo_path(OUT, "predictions.npz"), ids=test_ids, candidate=candidate,
                        baseline045=old045, p021=old021, raw_ridge=raw_ridge,
                        ridge_with_fallback=ridge, failed=failed, train_ids=train_ids, feature_keys=keys)
    source_paths = [training_file, label_file, test_file, sample_file, old045_file, old021_file, protocol_file, Path(__file__)]
    manifest = {"formula": "where(n_obj<20, p021, 0.8*submitted045 + 0.2*RidgeCV_full500)",
        "alpha_candidates": ALPHAS, "selected_alpha": float(model[-1].alpha_), "train_images": 500,
        "test_inference_images": 1000, "fallback_images": int(failed.sum()),
        "input_hashes": {str(p.relative_to(ROOT)): sha(p) for p in source_paths},
        "model_sha256": sha(model_path), "csv_sha256": sha(csv_path), "submitted": False,
        "reload_prediction_exact": True, "csv_prediction_exact": True,
        "test_scope": "Only fixed-feature inference after train-only fitting; no labels, scoring, adaptation, or selection.",
        "synthetic_examples_generated": 0, "test_training": False,
        "limitations": ["Frozen blend recipe from reused dev folds and legacy 98-feature representation.",
                        "No public or private score is available for this candidate."],
        "prediction_change_rms": float(np.sqrt(np.mean((candidate-old045)**2))),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "device": "cpu"}}
    (_repo_path(OUT, "manifest.json")).write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"csv": str(csv_path), "alpha": manifest["selected_alpha"],
                     "fallback_images": manifest["fallback_images"], "submitted": False}, indent=2))


if __name__ == "__main__":
    main()
