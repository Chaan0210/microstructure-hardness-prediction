"""Apply the fixed total-compute budget to measured resource probes."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import json
files=['teachers-complete.json','cnn-retarget-complete.json','log-complete.json','dense-resource-probe.json','dense-mps-resource-probe.json']
spent={f:json.loads((c.HERE/f).read_text())['seconds'] for f in files}
if (c.HERE/'diagnostic-stage-times.json').exists():
 for r in json.loads((c.HERE/'diagnostic-stage-times.json').read_text()):
  if r['stage']!='dense-probe':spent[r['stage']]=r['seconds']
else:spent['audit']=json.loads((c.HERE/'audit.json').read_text())['seconds']
spent['failed_startup_and_timing_margin']=60
remaining=5400-sum(spent.values())-600
projection=json.loads((c.HERE/'dense-mps-resource-probe.json').read_text())['projected_main_seconds']
if (c.HERE/'dense-budget-decision.json').exists():
 r=json.loads((c.HERE/'dense-budget-decision.json').read_text());assert abs(r['remaining_fit_seconds']-remaining)<1e-9 and r['projected_dense_seconds']==projection;print(json.dumps(r,indent=2))
else:
 r=dict(protocol_total_seconds=5400,compute_spent=spent,verification_reserve_seconds=600,remaining_fit_seconds=remaining,dense_hard_cap_seconds=3300,projected_dense_seconds=projection,passes=projection<3300 and remaining>=3300,reason='Apply unchanged90min total with600s verification reserve and3300s dense stage hard cap; no candidate accuracy is used.');c.save(_repo_io('dense-budget-decision.json'),r);print(json.dumps(r,indent=2))
assert r['passes']
