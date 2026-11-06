"""Validate the prospective120-minute budget without extending any allowance."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import json,datetime
def main():
    p=json.loads((c.HERE/'resource-probe-b32.json').read_text());total=p['projected_matrix_features_readouts_seconds'];assert total<=6600 and total-900<=5700 and all(v>0 for v in p['gradients'].values())
    path=c.HERE/'resource-accounting.json'
    if path.exists():
        r=json.loads(path.read_text());assert r['passed'] and r['probe_sha256']==c.sha(c.HERE/'resource-probe-b32.json') and r['protocol_total_budget_seconds']==7200
    else:c.save(_repo_io('resource-accounting.json'),dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,probe_sha256=c.sha(c.HERE/'resource-probe-b32.json'),protocol_total_budget_seconds=7200,protocol_validation_reserve_seconds=600,protocol_readout_reserve_seconds=900,encoder_stage_cap_seconds=5700,measured_projected_encoder_and_features_seconds=total-900,projected_with_validation_seconds=total+600,correct_combined_fit_gate_seconds=6600,main_models_or_hardness_scores_seen=False))
    print('Budget verified:',round((total+600)/60,2),'minutes including validation, <=120')
if __name__=='__main__':main()
