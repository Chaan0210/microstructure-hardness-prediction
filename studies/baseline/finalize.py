"""Final bounded development-CV comparison. No holdout target access."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, time, csv
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, ".venv")).resolve():raise SystemExit("Use project .venv")
sys.path.insert(0,str(_repo_path(ROOT, "src")))
from microstructure_v2.pipeline import *
from sklearn.preprocessing import PolynomialFeatures,QuantileTransformer
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.compose import ColumnTransformer

def main():
    old=_repo_path(ROOT, "artifacts/run-001"); new=_repo_path(ROOT, "artifacts/run-003"); out=_repo_path(ROOT, "artifacts/run-004")
    out.mkdir(exist_ok=False);a=np.load(_repo_io(new/"train_features.npz"))
    X,y,ids=(a[k] for k in ("X","y","ids"))
    splits=json.loads((old/"split_manifest.json").read_text())
    dev=np.array([i for i,s in enumerate(ids) if s in splits["development_folds"]])
    foldid=np.array([splits["development_folds"][s] for s in ids[dev]])
    folds=[(np.flatnonzero(foldid!=k),np.flatnonzero(foldid==k)) for k in range(5)]
    models={};oof={};scores={};recipes={}
    for label,path,width in [("base",old,519),("morph",new,X.shape[1])]:
        bundle=joblib.load(_repo_io(path/"best_model.joblib"))
        report=json.loads((path/"report.json").read_text())
        with (path/"development_oof.csv").open() as f:rows={r["ID"]:r for r in csv.DictReader(f)}
        pred=np.zeros(len(dev));recipe={}
        for name,weight in bundle["weights"].items():
            key=label+"_"+name
            estimator=clone(bundle["models"][name])
            if width<X.shape[1]:
                estimator=make_pipeline(ColumnTransformer([("base","passthrough",list(range(width)))]),estimator)
            models[key]=estimator;recipe[key]=weight
            pred+=weight*np.array([float(rows[s][name]) for s in ids[dev]])
        oof[label]=pred;recipes[label]=recipe;scores[label]={"oof_rmse":rmse(y[dev],pred)}
    additions={}
    for k in (20,40):
        for alpha in (100.,1000.):
            additions[f"poly{k}_a{alpha:g}"]=make_pipeline(
                SelectKBest(f_regression,k=k),StandardScaler(),
                PolynomialFeatures(degree=2,include_bias=False),
                StandardScaler(),Ridge(alpha=alpha))
    for alpha in (100.,1000.):
        additions[f"quantile_a{alpha:g}"]=make_pipeline(
            QuantileTransformer(n_quantiles=100,output_distribution="normal",random_state=20260930),
            Ridge(alpha=alpha))
    t=time.perf_counter()
    for name,model in additions.items():
        pred=np.zeros(len(dev));foldscores=[];seconds=[]
        for tr,va in folds:
            m=clone(model);start=time.perf_counter()
            m.fit(X[dev[tr]],y[dev[tr]]);pred[va]=m.predict(X[dev[va]])
            seconds.append(time.perf_counter()-start);foldscores.append(rmse(y[dev[va]],pred[va]))
        models[name]=model;oof[name]=pred;recipes[name]={name:1.}
        scores[name]={"oof_rmse":rmse(y[dev],pred),"fold_rmse":foldscores,"fit_predict_seconds":seconds,
            "params":{k:str(v) for k,v in model.get_params().items()}}
        print(json.dumps({"name":name,"oof_rmse":scores[name]["oof_rmse"],"seconds":sum(seconds)}),flush=True)
    # Limited blending grid with baseline, picked solely on development predictions.
    for name in ["morph",min(additions,key=lambda n:scores[n]["oof_rmse"])]:
        for weight in (.25,.5):
            key=f"blend_{name}_{weight}"
            oof[key]=(1-weight)*oof["base"]+weight*oof[name]
            recipes[key]={n:(1-weight)*w for n,w in recipes["base"].items()}
            recipes[key].update({n:weight*w for n,w in recipes[name].items()})
            scores[key]={"oof_rmse":rmse(y[dev],oof[key]),"weights":recipes[key]}
    winner=min(scores,key=lambda n:scores[n]["oof_rmse"])
    weights=recipes[winner]
    report=json.loads((new/"report.json").read_text())
    report.update(selected=winner,candidates=scores,elapsed_seconds=time.perf_counter()-t,
        feature_seconds=0.,feature_cache_source="artifacts/run-003/train_features.npz",
        holdout_rmse=None,holdout_mean_baseline_rmse=None,
        validation_note="Chosen using development CV only. Holdout not evaluated in stages 003 or 004. Adaptive CV selection can be optimistic.")
    write_json(out/"selection.json",{"selected":winner,"weights":weights,"criterion":"development OOF RMSE only","holdout_evaluated":False})
    fitted={n:clone(models[n]).fit(X,y) for n in weights}
    bundle={"models":fitted,"weights":weights,"feature_version":FEATURE_VERSION,
        "feature_names":report["feature_names"],"environment":environment()}
    joblib.dump(bundle,_repo_io(out/"best_model.joblib"))
    report["model_sha256"]=digest(out/"best_model.joblib")
    report["source_sha256"]={p.name:digest(p) for p in ROOT.joinpath("src/microstructure_v2").glob("*.py")}
    report["finalize_script_sha256"]=digest(Path(__file__))
    report["elapsed_seconds"]=time.perf_counter()-t
    write_json(out/"report.json",report)
    with (out/"development_oof.csv").open("w",newline="") as f:
        writer=csv.writer(f);writer.writerow(["ID","hardness",*oof])
        for j,i in enumerate(dev):writer.writerow([ids[i],y[i],*[oof[n][j] for n in oof]])
    print(json.dumps({"selected":winner,"oof_rmse":scores[winner]["oof_rmse"],"elapsed_seconds":report["elapsed_seconds"]}),flush=True)

if __name__=="__main__":main()
