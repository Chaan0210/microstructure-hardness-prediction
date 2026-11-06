"""Small read-only progress snapshot; no ML runtime imported."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json
H=Path(__file__).resolve().parent
n=len(list((_repo_path(H, 'teachers076')).glob('*/predictions.npz')))
t=len(list((_repo_path(H, 'teachers076')).glob('*/tab-*.npz')))
c=len(list((_repo_path(H, 'retarget-cnn')).glob('*/*/training.json')))
result=dict(completed_teacher_bundles=n,total_teacher_bundles=12,
            cached_Tab_predictions=t,total_Tab_predictions=864,
            completed_CNN_fits=c,total_CNN_fits=8,pilot_complete=(_repo_path(H, 'pilot-complete.json')).exists())
if (_repo_path(H, 'retarget-results-01.json')).exists():
    result['results']=json.loads((_repo_path(H, 'retarget-results-01.json')).read_text())
print(json.dumps(result,ensure_ascii=False))
