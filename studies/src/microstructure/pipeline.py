"""CPU-only training and prediction. Does not download data or call remote models."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "2")
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .features import FEATURE_VERSION, canonical_hash, extract, read_image

ROOT = Path(__file__).resolve().parents[3]

def local_path(value, must_exist=False):
    p = Path(value)
    if p.is_absolute():
        raise ValueError("Use a project-relative path")
    p = (_repo_path(ROOT, p)).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("Path escapes project directory")
    if must_exist and not p.exists():
        raise FileNotFoundError(p)
    return p

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rmse(y, pred):
    y, pred = np.asarray(y,dtype=float), np.asarray(pred,dtype=float)
    if y.shape != pred.shape or not y.size or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("RMSE requires matching, nonempty, finite vectors")
    return float(np.sqrt(np.mean((y-pred)**2)))

def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")

def read_rows(path, prefix, labels):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["ID","hardness"]:
            raise ValueError(f"Unexpected CSV columns: {reader.fieldnames}")
        rows = list(reader)
    ids = [r["ID"] for r in rows]
    if len(ids)!=len(set(ids)) or not ids or any(not re.fullmatch(prefix+r"_\d{6}", i) for i in ids):
        raise ValueError("Missing, duplicate, or unsafe IDs")
    y = np.array([float(r["hardness"]) for r in rows]) if labels else None
    if labels and not np.isfinite(y).all():
        raise ValueError("Non-finite labels")
    return ids, y

def features(data, ids, split):
    xs, groups, files, names = [], [], {}, None
    started = time.perf_counter()
    for i, ident in enumerate(ids):
        path = (data/split/(ident+".png")).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Image symlink escapes project")
        img = read_image(path)
        x, this_names = extract(img)
        if names is not None and names != this_names:
            raise ValueError("Feature schema changed")
        names = this_names
        xs.append(x); groups.append(canonical_hash(img)); files[ident] = digest(path)
        if (i+1)%50 == 0:
            print(f"{split}: {i+1}/{len(ids)} images", flush=True)
    return np.stack(xs), np.array(groups), files, names, time.perf_counter()-started

def candidates(seed, jobs):
    return {
        "mean": DummyRegressor(),
        "ridge_100": make_pipeline(StandardScaler(), Ridge(alpha=100)),
        "et_leaf2": ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2,
                 max_features=.8, random_state=seed, n_jobs=jobs),
        "et_leaf4": ExtraTreesRegressor(n_estimators=400, min_samples_leaf=4,
                 max_features=1., random_state=seed, n_jobs=jobs),
        "hgb_leaf8": HistGradientBoostingRegressor(max_iter=250, max_leaf_nodes=8,
                 min_samples_leaf=15, l2_regularization=10., learning_rate=.05,
                 early_stopping=False, random_state=seed),
    }

def environment():
    return {"python":sys.version, "platform":platform.platform(),
            "packages":{p:importlib.metadata.version(p) for p in
               ("numpy","scipy","Pillow","scikit-learn","scikit-image","joblib")},
            "cpu_threads":{k:os.environ[k] for k in
               ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS")}}

def train(args):
    data, out = local_path(args.data, True), local_path(args.output)
    # A unique run protects previous best artifacts and discourages holdout re-use.
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    ids,y = read_rows(data/"train.csv", "TRAIN", True)
    if len(ids)!=500:
        raise ValueError(f"Official training set requires 500 rows; got {len(ids)}")
    X,groups,hashes,names,feature_seconds = features(data,ids,"train")
    # No test files, sample submission, or test distribution are read during training.
    dev, hold = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=args.seed).split(X,y,groups))
    if len(np.unique(groups[dev]))<5:
        raise ValueError("Insufficient independent image groups")
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    folds = list(splitter.split(X[dev], y[dev], groups[dev]))
    fold_assignment = {}
    for fold, (_, va) in enumerate(folds):
        for idx in dev[va]:
            fold_assignment[ids[idx]] = fold
    write_json(out/"split_manifest.json", {
        "seed":args.seed, "holdout":[ids[i] for i in hold], "development_folds":fold_assignment,
        "group_by_id":dict(zip(ids,groups.tolist())),
        "caveat":"Exact and dihedral duplicates grouped; unknown near-duplicates/source families require review before trusting validation."})
    np.savez_compressed(_repo_io(out/"train_features.npz"), X=X, y=y, ids=np.array(ids), groups=groups)
    models = candidates(args.seed,args.jobs)
    report = {"seed":args.seed, "metric":"RMSE", "feature_version":FEATURE_VERSION,
        "n_features":X.shape[1], "n_train":len(y), "n_development":len(dev), "n_holdout":len(hold),
        "unique_groups":len(np.unique(groups)), "feature_seconds":feature_seconds,
        "environment":environment(), "train_csv_sha256":digest(data/"train.csv"),
        "image_sha256":hashes, "feature_names":names, "candidates":{}}
    oof = {}
    for name, base in models.items():
        pred = np.full(len(dev),np.nan)
        fold_scores, fit_seconds = [], []
        for tr, va in folds:
            model = clone(base); t = time.perf_counter()
            model.fit(X[dev[tr]],y[dev[tr]])
            fit_seconds.append(time.perf_counter()-t)
            pred[va] = model.predict(X[dev[va]])
            fold_scores.append(rmse(y[dev[va]],pred[va]))
        oof[name] = pred
        report["candidates"][name] = {"fold_rmse":fold_scores, "oof_rmse":rmse(y[dev],pred),
            "fit_seconds":fit_seconds, "params":{k:str(v) for k,v in base.get_params().items()}}
        print(json.dumps({"candidate":name, **report["candidates"][name]},ensure_ascii=False),flush=True)
        write_json(out/"progress.json", report)
    # Only two prespecified 50/50 blends; no holdout-based optimization.
    selections = {name:{name:1.} for name in models}
    for name, parts in {"blend_et_hgb":{"et_leaf2":.5,"hgb_leaf8":.5},
                        "blend_et_ridge":{"et_leaf2":.5,"ridge_100":.5}}.items():
        selections[name] = parts
        pred = sum(oof[n]*w for n,w in parts.items())
        report["candidates"][name] = {"oof_rmse":rmse(y[dev],pred), "weights":parts}
    winner = min(selections, key=lambda name:report["candidates"][name]["oof_rmse"])
    weights = selections[winner]
    write_json(out/"selection.json", {"selected":winner,"weights":weights,
        "selected_using":"development OOF RMSE only"})
    hold_pred = np.zeros(len(hold))
    for name, weight in weights.items():
        estimator = clone(models[name]).fit(X[dev],y[dev])
        hold_pred += weight*estimator.predict(X[hold])
    report["selected"] = winner
    report["holdout_rmse"] = rmse(y[hold],hold_pred)
    report["holdout_mean_baseline_rmse"] = rmse(y[hold],np.full(len(hold),y[dev].mean()))
    report["validation_note"] = "Single untouched grouped holdout after development-CV selection. Do not tune further on this holdout."
    errors = sorted(zip(hold,hold_pred),key=lambda z:abs(y[z[0]]-z[1]),reverse=True)
    with (out/"holdout_errors.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f); writer.writerow(["ID","target","prediction","residual","absolute_error"])
        for i,p in errors:
            writer.writerow([ids[i],y[i],p,y[i]-p,abs(y[i]-p)])
    with (out/"development_oof.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f); writer.writerow(["ID","hardness",*oof])
        for j,i in enumerate(dev):
            writer.writerow([ids[i],y[i],*[oof[n][j] for n in oof]])
    # Refit selected recipe on all TRAIN only. Test remains untouched.
    fitted = {name:clone(models[name]).fit(X,y) for name in weights}
    bundle = {"models":fitted,"weights":weights,"feature_names":names,
              "feature_version":FEATURE_VERSION,"environment":report["environment"]}
    joblib.dump(bundle,_repo_io(out/"best_model.joblib"))
    report["model_sha256"] = digest(out/"best_model.joblib")
    report["source_sha256"] = {p.name:digest(p) for p in Path(__file__).parent.glob("*.py")}
    report["elapsed_seconds"] = time.perf_counter()-started
    write_json(out/"report.json",report)
    print(json.dumps({"run":str(out.relative_to(ROOT)), "selected":winner,
        "holdout_rmse":report["holdout_rmse"], "seconds":report["elapsed_seconds"]}),flush=True)

def predict(args):
    data, run, output = local_path(args.data,True), local_path(args.run,True), local_path(args.output)
    report = json.loads((run/"report.json").read_text())
    if report["source_sha256"] != {p.name:digest(p) for p in Path(__file__).parent.glob("*.py")}:
        raise ValueError("Source changed since training; preserve original code or retrain")
    if environment()["packages"] != report["environment"]["packages"]:
        raise ValueError("Package versions differ from training")
    if digest(run/"best_model.joblib") != report["model_sha256"]:
        raise ValueError("Model integrity mismatch")
    # joblib can execute code; load only artifacts produced locally by this pipeline.
    bundle=joblib.load(_repo_io(run/"best_model.joblib"))
    if bundle["feature_version"] != FEATURE_VERSION:
        raise ValueError("Feature version mismatch")
    ids,_ = read_rows(data/"sample_submission.csv","TEST",False)
    if len(ids)!=1000:
        raise ValueError(f"Official test set requires 1000 rows; got {len(ids)}")
    X,_,hashes,names,seconds = features(data,ids,"test")
    if names != bundle["feature_names"]:
        raise ValueError("Feature schema mismatch")
    pred=sum(bundle["models"][n].predict(X)*w for n,w in bundle["weights"].items())
    if pred.shape != (len(ids),) or not np.isfinite(pred).all():
        raise ValueError("Invalid predictions")
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",newline="",encoding="utf-8") as f:
        writer=csv.writer(f); writer.writerow(["ID","hardness"])
        writer.writerows(zip(ids,pred))
    write_json(output.with_suffix(".manifest.json"), {"run":str(run.relative_to(ROOT)),
        "model_sha256":report["model_sha256"], "submission_sha256":digest(output),
        "sample_submission_sha256":digest(data/"sample_submission.csv"),
        "image_sha256":hashes, "rows":len(ids),"feature_seconds":seconds})
    print(f"Prepared {output.relative_to(ROOT)}; no upload performed.",flush=True)

def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("train")
    p.add_argument("--data",default="data"); p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=20260930); p.add_argument("--jobs",type=int,default=2)
    p.set_defaults(func=train)
    p=sub.add_parser("predict")
    p.add_argument("--data",default="data"); p.add_argument("--run",required=True)
    p.add_argument("--output",default="artifacts/submission.csv"); p.set_defaults(func=predict)
    args=parser.parse_args()
    if getattr(args,"jobs",2) not in range(1,5):
        parser.error("--jobs must be between 1 and 4")
    args.func(args)

if __name__=="__main__":
    main()
