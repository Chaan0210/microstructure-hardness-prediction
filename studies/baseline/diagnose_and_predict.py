"""Prediction-only test feature cache and stability diagnostics; never fits test."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,os,csv,json,time,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, ".venv")).resolve():raise SystemExit("Use project .venv")
sys.path.insert(0,str(_repo_path(ROOT, "src")))
from microstructure_v2.pipeline import *
from microstructure_v2.features import extract,read_image

def one(ident):
    p=_repo_path(ROOT, "data/test", ident+".png")
    x,names=extract(read_image(p))
    return x,names,digest(p)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="artifacts/submission-candidate-002.csv")
    args=parser.parse_args();run=_repo_path(ROOT, "artifacts/run-005")
    report=json.loads((run/"report.json").read_text());bundle=joblib.load(_repo_io(run/"best_model.joblib"))
    assert report["model_sha256"]==digest(run/"best_model.joblib")
    assert report["environment"]["packages"]==environment()["packages"]
    assert report["source_sha256"]=={p.name:digest(p) for p in ROOT.joinpath("src/microstructure_v2").glob("*.py")}
    ids,_=read_rows(_repo_path(ROOT, "data/sample_submission.csv"),"TEST",False);assert len(ids)==1000
    cache=_repo_path(ROOT, "artifacts/test-prediction-features.npz");started=time.perf_counter()
    if cache.exists():
        a=np.load(_repo_io(cache));assert a["ids"].tolist()==ids;assert a["names"].tolist()==bundle["feature_names"]
        X=a["X"];hashes=a["hashes"].tolist()
        assert hashes==[digest(_repo_path(ROOT, "data/test", ident+".png")) for ident in ids]
    else:
        xs=[];hashes=[]
        with ProcessPoolExecutor(max_workers=2) as pool:
            for i,(x,names,sha) in enumerate(pool.map(one,ids,chunksize=5)):
                assert names==bundle["feature_names"];xs.append(x);hashes.append(sha)
                if (i+1)%50==0:print(f"test cache: {i+1}/1000",flush=True)
        X=np.stack(xs)
        np.savez_compressed(_repo_io(cache),X=X,ids=np.array(ids),names=np.array(bundle["feature_names"]),hashes=np.array(hashes))
    feature_seconds=time.perf_counter()-started
    original=joblib.load(_repo_path(ROOT, "artifacts/run-004/best_model.joblib"))
    ranges={};original_pred=np.zeros(len(ids))
    for n,m in original["models"].items():
        p=m.predict(X);original_pred+=original["weights"][n]*p
        ranges["original_"+n]={"min":float(p.min()),"max":float(p.max()),"negative_count":int((p<0).sum())}
    with (_repo_path(ROOT, "artifacts/submission-candidate-001.csv")).open() as f:prior=list(csv.DictReader(f))
    assert [r["ID"] for r in prior]==ids
    np.testing.assert_allclose(original_pred,[float(r["hardness"]) for r in prior],rtol=0,atol=1e-8)
    pred=np.zeros(len(ids))
    for n,m in bundle["models"].items():
        p=m.predict(X);pred+=bundle["weights"][n]*p
        ranges["stable_"+n]={"min":float(p.min()),"max":float(p.max()),"negative_count":int((p<0).sum())}
    train=np.load(_repo_path(ROOT, "artifacts/run-003/train_features.npz"))["X"]
    sd=train.std(axis=0);z=np.abs(X-train.mean(axis=0))/np.maximum(sd,1e-12)
    diagnostic={"purpose":"stability/format only; test not used for fitting or threshold estimation",
        "rows":len(ids),"feature_seconds":feature_seconds,"component_ranges":ranges,
        "original_feature_recomputation_matches_all_rows":True,
        "rows_with_any_training_abs_z_over10":int(np.any(z>10,axis=1).sum()),"max_training_abs_z":float(z.max()),
        "prediction_min":float(pred.min()),"prediction_max":float(pred.max()),"negative_count":int((pred<0).sum()),
        "posthoc_target_clipping":False}
    write_json(_repo_path(ROOT, "reports/full-test-stability-diagnostic.json"),diagnostic)
    if not np.isfinite(pred).all() or np.any(pred<0):raise ValueError("Stable candidate still has invalid/negative predictions; review required")
    output=local_path(args.output)
    with output.open("x",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["ID","hardness"]);w.writerows(zip(ids,pred))
    write_json(output.with_suffix(".manifest.json"),{"run":"artifacts/run-005","model_sha256":report["model_sha256"],
        "submission_sha256":digest(output),"rows":len(ids),"sample_submission_sha256":digest(_repo_path(ROOT, "data/sample_submission.csv")),
        "image_sha256":dict(zip(ids,hashes)),"feature_seconds":feature_seconds,"test_cache_only_for_prediction":True,
        "prediction_script_sha256":digest(Path(__file__))})
    print(json.dumps(diagnostic,indent=2),flush=True)
    print("Saved local candidate; not uploaded.",flush=True)

if __name__=="__main__":main()
