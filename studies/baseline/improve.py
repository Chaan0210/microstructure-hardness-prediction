"""Bounded second-stage search; selects using development folds only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, os
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, ".venv")).resolve():
    raise SystemExit("Use .venv/bin/python -I improve.py")
sys.path.insert(0,str(_repo_path(ROOT, "src")))
from microstructure.pipeline import *
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, f_regression

def improved_candidates(seed,jobs):
    result={}
    for c in (10.,100.):
        for gamma in (.001,.003):
            result[f"svr_c{c:g}_g{gamma:g}"]=TransformedTargetRegressor(
                regressor=make_pipeline(StandardScaler(),SVR(C=c,gamma=gamma,epsilon=.05)),
                transformer=StandardScaler())
    for alpha in (.1,1.):
        result[f"krr_a{alpha:g}"]=make_pipeline(StandardScaler(),KernelRidge(alpha=alpha,gamma=.001,kernel="rbf"))
    result["ridge10"]=make_pipeline(StandardScaler(),Ridge(alpha=10.))
    result["ridge1000"]=make_pipeline(StandardScaler(),Ridge(alpha=1000.))
    result["pca_svr"]=TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(),PCA(n_components=30,random_state=seed),
            SVR(C=10,gamma=.003,epsilon=.05)),transformer=StandardScaler())
    result["selected_svr"]=TransformedTargetRegressor(
        regressor=make_pipeline(SelectPercentile(f_regression,percentile=40),StandardScaler(),
            SVR(C=10,gamma=.003,epsilon=.05)),transformer=StandardScaler())
    result["et_leaf1"]=ExtraTreesRegressor(n_estimators=600,min_samples_leaf=1,
        max_features=.6,random_state=seed,n_jobs=jobs)
    return result

def main():
    base=_repo_path(ROOT, "artifacts/run-001"); out=_repo_path(ROOT, "artifacts/run-002")
    out.mkdir(exist_ok=False)
    old=json.loads((base/"report.json").read_text())
    cache=np.load(_repo_io(base/"train_features.npz"))
    X,y,ids,groups=(cache[k] for k in ("X","y","ids","groups"))
    manifest=json.loads((base/"split_manifest.json").read_text())
    hold=np.array([i for i,ident in enumerate(ids) if ident in set(manifest["holdout"])])
    dev=np.array([i for i,ident in enumerate(ids) if ident in manifest["development_folds"]])
    assignments=np.array([manifest["development_folds"][ident] for ident in ids[dev]])
    folds=[(np.flatnonzero(assignments!=k),np.flatnonzero(assignments==k)) for k in range(5)]
    models=candidates(old["seed"],2)
    models.update(improved_candidates(old["seed"],2))
    score_table={}; oof={}; recipes={}
    with (base/"development_oof.csv").open() as f:
        rows={r["ID"]:r for r in csv.DictReader(f)}
    for name in candidates(old["seed"],2):
        pred=np.array([float(rows[ident][name]) for ident in ids[dev]])
        oof[name]=pred; recipes[name]={name:1.}
        score_table[name]=old["candidates"][name]
    started=time.perf_counter()
    for name,model in improved_candidates(old["seed"],2).items():
        pred=np.full(len(dev),np.nan); scores=[]; times=[]
        for tr,va in folds:
            estimator=clone(model); t=time.perf_counter()
            estimator.fit(X[dev[tr]],y[dev[tr]])
            pred[va]=estimator.predict(X[dev[va]])
            times.append(time.perf_counter()-t)
            scores.append(rmse(y[dev[va]],pred[va]))
        score_table[name]={"oof_rmse":rmse(y[dev],pred),"fold_rmse":scores,"fit_predict_seconds":times,
            "params":{k:str(v) for k,v in model.get_params().items()}}
        oof[name]=pred; recipes[name]={name:1.}
        print(json.dumps({"candidate":name,"oof_rmse":score_table[name]["oof_rmse"],
            "seconds":sum(times)}),flush=True)
        write_json(out/"progress.json",score_table)
    # Prespecified equal blends plus an equal blend of the three best individual recipes.
    blends={"blend_et_hgb":{"et_leaf2":.5,"hgb_leaf8":.5},
            "blend_et_ridge":{"et_leaf2":.5,"ridge_100":.5},
            "blend_svr_et":{"svr_c10_g0.001":.5,"et_leaf1":.5}}
    top3=sorted(oof,key=lambda n:rmse(y[dev],oof[n]))[:3]
    blends["blend_top3"]={n:1/3 for n in top3}
    for name,weights in blends.items():
        pred=sum(oof[n]*w for n,w in weights.items())
        oof[name]=pred;recipes[name]=weights
        score_table[name]={"oof_rmse":rmse(y[dev],pred),"weights":weights}
    winner=min(score_table,key=lambda n:score_table[n]["oof_rmse"])
    weights=recipes[winner]
    write_json(out/"selection.json",{"selected":winner,"weights":weights,
        "selected_using":"development OOF only; holdout not used for selection"})
    # This is a repeated diagnostic: the earlier baseline already exposed this holdout.
    pred_hold=sum(clone(models[n]).fit(X[dev],y[dev]).predict(X[hold])*w for n,w in weights.items())
    report={k:old[k] for k in ("seed","metric","feature_version","n_features","n_train",
        "n_development","n_holdout","unique_groups","feature_names","train_csv_sha256","image_sha256")}
    report.update(environment=environment(),candidates=score_table,selected=winner,
        holdout_rmse=rmse(y[hold],pred_hold),holdout_mean_baseline_rmse=old["holdout_mean_baseline_rmse"],
        validation_note="Development OOF used for limited adaptive model selection. Holdout is a repeated diagnostic, not a fresh independent evaluation. Source-family grouping remains unverified.",
        feature_seconds=0.,feature_cache_source="artifacts/run-001/train_features.npz",
        baseline_selected=old["selected"],baseline_holdout_rmse=old["holdout_rmse"])
    fitted={n:clone(models[n]).fit(X,y) for n in weights}
    bundle={"models":fitted,"weights":weights,"feature_names":old["feature_names"],
        "feature_version":FEATURE_VERSION,"environment":report["environment"]}
    joblib.dump(bundle,_repo_io(out/"best_model.joblib"))
    report["model_sha256"]=digest(out/"best_model.joblib")
    report["source_sha256"]={p.name:digest(p) for p in ROOT.joinpath("src/microstructure").glob("*.py")}
    report["improvement_script_sha256"]=digest(Path(__file__))
    report["elapsed_seconds"]=time.perf_counter()-started
    write_json(out/"report.json",report)
    with (out/"development_oof.csv").open("w",newline="") as f:
        wr=csv.writer(f);wr.writerow(["ID","hardness",*oof])
        for j,i in enumerate(dev):wr.writerow([ids[i],y[i],*[oof[n][j] for n in oof]])
    with (out/"holdout_errors.csv").open("w",newline="") as f:
        wr=csv.writer(f);wr.writerow(["ID","target","prediction","residual"])
        for i,p in zip(hold,pred_hold):wr.writerow([ids[i],y[i],p,y[i]-p])
    print(json.dumps({"selected":winner,"development_oof_rmse":score_table[winner]["oof_rmse"],
        "repeated_holdout_rmse":report["holdout_rmse"],"seconds":report["elapsed_seconds"]}),flush=True)

if __name__=="__main__":main()
