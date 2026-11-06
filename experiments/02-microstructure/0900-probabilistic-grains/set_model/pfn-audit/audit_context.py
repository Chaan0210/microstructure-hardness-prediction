"""Offline CPU replay of one saved TabPFN MPS training/query context.

No query labels, scores, fitting choices, or model-selection decisions are used.
All writes are within this audit directory; original code and weights are read-only.
"""
from __future__ import annotations
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[5]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT

from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
REPO = _RepoPath(__file__).resolve().parents[5]
for key, value in {
    "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2",
    "MKL_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "TABPFN_DISABLE_TELEMETRY": "1",
    "XDG_CACHE_HOME": str(_repo_path(HERE, "cache")), "HF_HOME": str(_repo_path(HERE, "cache", "hf")),
    "TORCH_HOME": str(_repo_path(HERE, "cache", "torch")), "TABPFN_CACHE_DIR": str(_repo_path(HERE, "cache", "tabpfn")),
    "MPLCONFIGDIR": str(_repo_path(HERE, "cache", "matplotlib")),
    "PYTHONDONTWRITEBYTECODE": "1",
}.items():
    os.environ[key] = value
sys.dont_write_bytecode = True
sys.path.insert(0, str(STUDY))

import gc
import hashlib
import importlib.metadata
import json
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits
import pfn_readout


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(name, obj):
    (_repo_path(HERE, name)).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def read_context(path):
    with np.load(_repo_io(path), allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def difference(a, b):
    delta = np.asarray(a, float) - np.asarray(b, float)
    return {"max_absolute": float(np.max(np.abs(delta))), "rms": float(np.sqrt(np.mean(delta ** 2))),
            "mean_absolute": float(np.mean(np.abs(delta))), "mean_signed": float(np.mean(delta))}


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    threadpool_limits(limits=2)
    context_path = _repo_path(STUDY, "pfn-readout-v2", "partition0-fold0-base98-context.npz")
    weights_path = _repo_path(REPO, "experiments", "tabpfn-v3", "weights", "tabpfn-v3.5-20260909.safetensors")
    c = read_context(context_path)
    assert c["X"].shape == (320, 98) and c["query"].shape == (80, 98)
    assert c["y"].shape == (320,) and c["prediction"].shape == (80,)
    assert not (set(c["train_ids"]) & set(c["query_ids"]))
    assert np.isfinite(c["y"]).all() and np.isfinite(c["prediction"]).all()
    seed = int(c["seed"])
    assert seed == 0 and pfn_readout.CONFIG["n_estimators"] == 8
    config = {"context_path": str(context_path), "context_sha256": sha(context_path),
              "weights_path": str(weights_path), "weights_sha256": sha(weights_path),
              "root_script_sha256": sha(_repo_path(STUDY, "pfn_readout.py")), "audit_script_sha256": sha(__file__),
              "torch_version": torch.__version__, "tabpfn_version": importlib.metadata.version("tabpfn"),
              "numpy_version": np.__version__, "seed": seed, "n_estimators": 8, "device": "cpu",
              "threads": torch.get_num_threads(), "network": "Offline environment plus root make() socket blocking",
              "train_query_ids_disjoint": True, "n_training": 320, "n_available_queries": 80,
              "query_labels_loaded_or_scored": False,
              "timing_policy": "First8 query CPU time times10 + initialization/fit; if >120s, repeat only8",
              "repeat_tolerance_atol": 1e-6, "device_tolerance_atol": .005, "device_tolerance_rtol": 1e-5}
    save("CONFIG.json", config)
    t0 = time.perf_counter()
    model = pfn_readout.make(seed, "cpu")
    model.fit(c["X"], c["y"])
    fit_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    first8 = np.asarray(model.predict(c["query"][:8]))
    first8_seconds = time.perf_counter() - t1
    assert np.isfinite(first8).all()
    projection = fit_seconds + 10 * first8_seconds
    n = 8 if projection > 120 else 80
    timing = {"initialization_plus_fit_seconds": fit_seconds, "first8_prediction_seconds": first8_seconds,
              "conservative_full80_projection_seconds": projection, "selected_query_count": n}
    save("timing.json", timing)
    print(json.dumps(timing), flush=True)
    if n == 80:
        t2 = time.perf_counter()
        prediction = np.asarray(model.predict(c["query"]))
        timing["full80_prediction_seconds"] = time.perf_counter() - t2
        timing["first8_vs_full80_prefix_difference"] = difference(first8, prediction[:8])
    else:
        prediction = first8
    del model
    gc.collect()
    # Fresh context read and newly constructed same-seed estimator, as final bundles would replay.
    repeat_context = read_context(context_path)
    t3 = time.perf_counter()
    repeat_model = pfn_readout.make(seed, "cpu")
    repeat_model.fit(repeat_context["X"], repeat_context["y"])
    repeated = np.asarray(repeat_model.predict(repeat_context["query"][:n]))
    timing["fresh_context_repeat_seconds"] = time.perf_counter() - t3
    assert np.isfinite(prediction).all() and np.isfinite(repeated).all()
    cpu_repeat = difference(prediction, repeated)
    device_difference = difference(prediction, c["prediction"][:n])
    cpu_match = bool(np.allclose(prediction, repeated, atol=config["repeat_tolerance_atol"], rtol=0))
    device_match = bool(np.allclose(prediction, c["prediction"][:n],
                                   atol=config["device_tolerance_atol"], rtol=config["device_tolerance_rtol"]))
    result = {**config, "timing": timing, "n_replayed_queries": n, "all_outputs_finite": True,
              "cpu_fresh_context_repeat": cpu_repeat, "cpu_repeat_within_tolerance": cpu_match,
              "cpu_vs_saved_mps": device_difference, "device_within_declared_tolerance": device_match,
              "audit_scope": "One fixed representative context; no heldout target scoring",
              "query_count_limit": None if n == 80 else "CPU audit restricted to first8 under predeclared runtime policy; MPS reference originally predicted80",
              "source_hashes_unchanged": {"context": sha(context_path) == config["context_sha256"],
                                          "weights": sha(weights_path) == config["weights_sha256"],
                                          "root_script": sha(_repo_path(STUDY, "pfn_readout.py")) == config["root_script_sha256"]}}
    save("results.json", result)
    np.savez_compressed(_repo_path(HERE, "predictions.npz"), query_ids=c["query_ids"][:n], cpu=prediction,
                        cpu_fresh_context_repeat=repeated, saved_mps=c["prediction"][:n])
    print(json.dumps({"n_queries": n, "cpu_repeat": cpu_repeat, "cpu_vs_mps": device_difference,
                      "device_within_tolerance": device_match, "all_outputs_finite": True}), flush=True)


if __name__ == "__main__":
    main()
