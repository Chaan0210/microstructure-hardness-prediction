# Candidate018 reproduction package v176

This package preserves the frozen018 recipe and contains newly trained176 final500 models. It is a reproducibility deliverable, not a new model-selection experiment. Existing018,019,020 CSVs and v136/v155 archives remain unchanged. Nothing was uploaded or submitted.

## What was actually run

Raw500 training images and labels were processed again. Compact26, physics57, CNN128 inputs, DINO-S3456, DINO-B6912 and spatial528 features exactly matched historical500 features. All13 base nodes were retrained:39 CNNs,78 long MLPs,4 short-B heads,13 final KRRs with351 inner fits,4 affines and60 residual objects (48 non-null). No previously fitted competition learner or cached training features entered fitting. Only local official frozen DINO S/B weights, fixed source code/configuration and the original development400 ID list were reused.

All121 newly trained neural states, middle OOF predictions, calibration, residual components and final500 reload predictions matched historical018 exactly. Independent validation recomputed364 KRR fits,60 residual objects and4 affines; it reloaded121 neural models. Same-device reload was exact. CPU/MPS neural maximum difference was0.00004404042687156107, within the predeclared0.001 diagnostic tolerance. This is not an additional holdout performance estimate. All500 labels were used for final training; no formerholdout subgroup score was computed.

The top-level KRR selects its existing9-point grid on the fixed originaldevelopment400, then refits500. Nested KRRs select on each current node's full training set. This intentional historical exception is preserved. Scalers and target statistics are fit on each learner's training rows. Calibration and residual targets use the existing nested/honest partitions. Frozen DINO encoders perform no dataset fit. No test fitting or tuning is allowed.

Raw1000 test images were also processed afresh after model validation. Predictions were repeated from reloaded fresh176 models; IDs, order, finite values, CSV numeric roundtrip and historical018 CSV equality were checked. See evidence/test-comparison.json and evidence/inference-validation.json. Archive isolation testing is documented in the external176 report: raw2-image extraction smoke test plus1000 inference using the explicit fresh176 feature cache. Full500 training was not redundantly rerun from the extracted ZIP.

## Environment and data

Use the existing project uv-created virtual environment. No installation, download or payment occurred. Exact versions are recorded in requirements-current.lock and evidence/environment.json; Python3.13.3, arm64/M1 Pro, torch2.14.0, numpy2.5.3, scipy1.18.1, scikit-learn1.9.1, Pillow12.3.0, scikit-image0.26.0. CPU threads are fixed at2. Raw extraction and training require Apple MPS; --device cpu changes regression inference only, not raw DINO extraction. CPU-only raw extraction and cross-platform bitwise reproduction are not claimed.

Data are not included. Supply a directory with train.csv, sample_submission.csv, train/TRAIN_000001.png through TRAIN_000500.png, and test/TEST_000001.png through TEST_001000.png. Images must be grayscale256x256. CSV column order must be ID,hardness and IDs must follow the canonical order. Test CSV hardness strings are ignored. Original CSV/image SHA256 checks are in config/original-train-hashes.json and config/original-test-hashes.json. Development IDs and feature schema are also included. Models necessarily contain fitted training statistics, kernel support features and coefficients; raw labels, raw images and feature caches are excluded.

## Run inference

From this extracted directory, set PY to the absolute path of the already installed project .venv/bin/python, and DATA to the authorized data directory. Use a new output directory each time. Network access and fitting/backward APIs are disabled during inference.

```sh
"$PY" selftest.py --output integrity-check
"$PY" infer.py --data "$DATA" --output raw-test-reproduction
```

The output CSV is raw-test-reproduction/reproduced-018.csv. Expected SHA256:6d478f08de769ca0c54d640c0711c95b15ad4e6b707d0516022c23260f1cfc16. Models are verified against models/manifest.json. For an explicit previously validated raw-derived test cache, use --features PATH/test-features.npz instead of --data; its adjacent test-feature-validation.json is required. CPU regression replay can be requested with --device cpu; CPU/MPS differences are reported numerically, not expected to be CSV-byte identical.

## Retrain the frozen full500 recipe

The package's run directory is intentionally absent. Keep the supplied models unchanged and export a fresh model directory. These commands reproduce the executed176 training path:

```sh
"$PY" selftest.py --output before-training-integrity
"$PY" extract.py train --data "$DATA"
"$PY" train.py prepare
"$PY" train.py fit
"$PY" validate_full500.py
"$PY" export_models.py --models models-retrained
"$PY" infer.py --models models-retrained --data "$DATA" --output retrained-test
```

Completed training nodes may be reused only from this new run and are hash-checked. A partially written node stops rather than silently overwriting it; preserve the failure and use a new working copy or explicit audited recovery. The main fit caps RSS at4GB and observed MPS driver allocation at2GB, checks a30-minute fit limit at completion, and records a1-second memory trace. These are not system-level process limits. Raw500 extraction took267seconds; training490seconds; independent validation11seconds in the executed environment. Packaging and test inference are separate costs. No performance selection or parameter changes are authorized by these commands.

## Evidence and source history

MANIFEST.json covers every packaged file except itself. Its hash and ZIP hash are recorded in the external176 report. Evidence includes the frozen plan, exact environment, raw hashes, independent checks, historical comparison and test checks. Official pretrained model URLs, sizes and local hashes are in licenses/current-model-provenance.json. These local hashes establish integrity, not a publisher signature. DINOv2 source commit7764ea0f912e53c92e82eb78a2a1631e92725fc8 and Apache-2.0 notices are preserved. The copied upstream provenance's early candidate scope is historical; PACKAGE.json identifies this package's current scope. Dependency notices are metadata, and installed dependency binaries are not redistributed here. No new license grant for project-specific source is asserted.

Two implementation corrections were recorded without changing the learned recipe. First, an optional original-data-hash check was added while the first feature extraction process was already running; source-history/extract-initial.py preserves that exact executed source and numeric output matched all historical features. Second, validate.py retained a development OOF root group-disjoint assertion that cannot apply to final500 train-query reloads. It is retained as frozen historical source; use validate_full500.py, which permits only the exact root500 overlap and keeps all nested group exclusions. A comparison helper was separately changed to independently reload historical residual models on fresh500 training features. Its initial addendum incorrectly suspected a1000-row reference cache; subsequent shape inspection confirmed that block caches are500-row, while only the top-level cache is1000-row. comparison-provenance-correction.json records this correction. The original helper was never executed or used for fitting. Addenda preserve the original hashes and reasons.

For the isolated reproduction test:

```sh
"$PY" selftest.py --features "$FRESH_TEST_CACHE" --raw-data "$DATA" --output isolated-check
```

This checks all file hashes and installed versions, probes fit/backward guards, re-extracts the predeclared first/last test images, and runs the full1000 inference twice. The external executed test additionally used --project-boundary to block undeclared historical project file reads. It does not train, select a model, estimate new performance or submit anything.
