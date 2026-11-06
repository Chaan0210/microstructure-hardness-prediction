"""Train-only input stabilization and development-CV comparison; no holdout use."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,csv,json
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, ".venv")).resolve():raise SystemExit("Use project .venv")
sys.path.insert(0,str(_repo_path(ROOT, "src")))
from microstructure_v2.pipeline import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

def main():
    out=_repo_path(ROOT, "artifacts/run-005");out.mkdir(exist_ok=False)
    a=np.load(_repo_path(ROOT, "artifacts/run-003/train_features.npz"));X,y,ids=(a[k] for k in ("X","y","ids"))
    sp=json.loads((_repo_path(ROOT, "artifacts/run-001/split_manifest.json")).read_text())
    dev=np.array([i for i,s in enumerate(ids) if s in sp["development_folds"]])
    foldid=np.array([sp["development_folds"][s] for s in ids[dev]])
    folds=[(np.flatnonzero(foldid!=k),np.flatnonzero(foldid==k)) for k in range(5)]
    with (_repo_path(ROOT, "artifacts/run-001/development_oof.csv")).open() as f:old={r["ID"]:r for r in csv.DictReader(f)}
    with (_repo_path(ROOT, "artifacts/run-004/development_oof.csv")).open() as f:last={r["ID"]:r for r in csv.DictReader(f)}
    et_oof=np.array([float(old[s]["et_leaf2"]) for s in ids[dev]])
    raw=np.array([float(last[s]["blend_morph_0.25"]) for s in ids[dev]])
    params={"base_ridge":make_pipeline(ColumnTransformer([("base","passthrough",list(range(519)))]),
                MinMaxScaler(clip=True),StandardScaler(),Ridge(alpha=100)),
            "morph_ridge":make_pipeline(MinMaxScaler(clip=True),StandardScaler(),Ridge(alpha=1000))}
    oof={};scores={};t=time.perf_counter();bounds=[]
    clipped=raw.copy()
    for tr,va in folds:
        lo,hi=float(y[dev[tr]].min()),float(y[dev[tr]].max())
        clipped[va]=np.clip(raw[va],lo,hi);bounds.append({"min":lo,"max":hi,"validation_clipped":int(((raw[va]<lo)|(raw[va]>hi)).sum())})
    for name,model in params.items():
        p=np.zeros(len(dev));times=[]
        for tr,va in folds:
            m=clone(model);start=time.perf_counter();m.fit(X[dev[tr]],y[dev[tr]]);p[va]=m.predict(X[dev[va]]);times.append(time.perf_counter()-start)
        oof[name]=p;scores[name]={"oof_rmse":rmse(y[dev],p),"fit_predict_seconds":times}
    options={"stable_base":.5*et_oof+.5*oof["base_ridge"],
             "stable_hybrid":.375*et_oof+.375*oof["base_ridge"]+.25*oof["morph_ridge"]}
    for name,p in options.items():scores[name]={"oof_rmse":rmse(y[dev],p),"fold_rmse":[rmse(y[dev[va]],p[va]) for _,va in folds]}
    scores["raw_hybrid"]={"oof_rmse":rmse(y[dev],raw)}
    scores["target_clipped_hybrid"]={"oof_rmse":rmse(y[dev],clipped),"bounds_by_fold":bounds}
    winner=min(options,key=lambda n:scores[n]["oof_rmse"])
    # Stability is addressed in input space; no target range is tuned from test.
    if scores[winner]["oof_rmse"]>scores["raw_hybrid"]["oof_rmse"]+.2:
        write_json(out/"comparison.json",scores);raise RuntimeError("Stabilization worsened CV materially; review needed")
    weights={"base_et":.5,"base_ridge":.5} if winner=="stable_base" else {"base_et":.375,"base_ridge":.375,"morph_ridge":.25}
    old_bundle=joblib.load(_repo_path(ROOT, "artifacts/run-001/best_model.joblib"))
    params["base_et"]=make_pipeline(ColumnTransformer([("base","passthrough",list(range(519)))]),clone(old_bundle["models"]["et_leaf2"]))
    write_json(out/"selection.json",{"selected":winner,"weights":weights,"criterion":"best development CV among input-stabilized recipes; no more than 0.2 RMSE worse than raw accepted","holdout_evaluated":False,"target_clipping":False})
    fitted={n:clone(params[n]).fit(X,y) for n in weights}
    report=json.loads((_repo_path(ROOT, "artifacts/run-004/report.json")).read_text())
    report.update(selected=winner,candidates=scores,holdout_rmse=None,validation_note="Input min/max learned inside each train fold. All selection uses development CV. No new holdout evaluation.",
        feature_seconds=0.,feature_cache_source="artifacts/run-003/train_features.npz",environment=environment())
    bundle={"models":fitted,"weights":weights,"feature_version":FEATURE_VERSION,"feature_names":report["feature_names"],"environment":report["environment"]}
    joblib.dump(bundle,_repo_io(out/"best_model.joblib"))
    report["model_sha256"]=digest(out/"best_model.joblib")
    report["source_sha256"]={p.name:digest(p) for p in ROOT.joinpath("src/microstructure_v2").glob("*.py")}
    report["stabilize_script_sha256"]=digest(Path(__file__));report["elapsed_seconds"]=time.perf_counter()-t
    write_json(out/"report.json",report)
    with (out/"development_oof.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["ID","hardness",*options])
        for j,i in enumerate(dev):w.writerow([ids[i],y[i],*[options[n][j] for n in options]])
    print(json.dumps({"selected":winner,"scores":scores,"elapsed_seconds":report["elapsed_seconds"]},indent=2),flush=True)

if __name__=="__main__":main()
