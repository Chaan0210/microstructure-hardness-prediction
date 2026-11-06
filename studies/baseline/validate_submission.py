"""Validate a local submission file without uploading it."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse,csv,hashlib,json,math,os,statistics,sys
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, ".venv")).resolve():raise SystemExit("Use project .venv")
def path(s):
    p=Path(os.path.abspath(_repo_path(ROOT, s)))
    if Path(s).is_absolute() or not p.is_relative_to(ROOT) or not p.resolve().is_relative_to(ROOT.parent):raise ValueError("Use project-relative paths")
    return p
def rows(p):
    with p.open(newline="",encoding="utf-8-sig") as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=["ID","hardness"]:raise ValueError("Columns must be ID,hardness")
        return list(reader)
def main():
    p=argparse.ArgumentParser();p.add_argument("--submission",default="artifacts/submission-candidate-002.csv")
    p.add_argument("--sample",default="data/sample_submission.csv");p.add_argument("--report",default="reports/submission-validation.json")
    a=p.parse_args();source=path(a.submission);r=rows(source);sample=rows(path(a.sample))
    ids=[v["ID"] for v in r];expected=[v["ID"] for v in sample];pred=[float(v["hardness"]) for v in r]
    if len(r)!=1000 or ids!=expected or len(set(ids))!=1000:raise ValueError("Row count, IDs or sample order mismatch")
    if not all(math.isfinite(v) for v in pred):raise ValueError("Non-finite prediction")
    result={"valid":True,"rows":len(r),"columns":["ID","hardness"],"ids_unique":True,
        "sample_order_exact":True,"all_predictions_finite":True,"negative_prediction_count":sum(v<0 for v in pred),"all_predictions_nonnegative":all(v>=0 for v in pred),"prediction_min":min(pred),
        "prediction_max":max(pred),"prediction_mean":statistics.mean(pred),
        "prediction_std":statistics.pstdev(pred),"sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "file_bytes":source.stat().st_size,"uploaded":False}
    path(a.report).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
