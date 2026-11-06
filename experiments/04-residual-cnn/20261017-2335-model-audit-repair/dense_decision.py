"""Prospective stricter calibration-attribution check; no new fitting."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import evaluate as e
import json,sys
import numpy as np
def main():
 r=json.loads((c.HERE/'dense-results.json').read_text());z=np.load(_repo_io(c.HERE/'dense-oof.npz'));m=e.compare(z['old_matched'],z['pool_matched'],z['y'],z['fold'],z['high']);passed=r['advance'] and m['family_ci'][0]>0
 c.save(_repo_io('dense-audit-decision.json'),dict(original_rule_passed=r['advance'],versus_matched_old_CNN=m,additional_rule_passed=m['family_ci'][0]>0,advance=bool(passed),additional_protocol_sha256=c.sha(c.HERE/'ADDITIONAL-CONTROL.md'),code_sha256=c.sha(__file__)));print(json.dumps(dict(advance=bool(passed),versus_old=m),indent=2))
if __name__=='__main__':sys.addaudithook(c.guard);main()
