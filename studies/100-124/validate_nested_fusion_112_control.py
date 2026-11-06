"""Verify the new small-head wrapper reproduces incumbent011 training exactly."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import nested_fusion_112 as e
import numpy as np,torch

def main():
 e.check();torch.set_num_threads(2);d=e.data('original');tr=np.flatnonzero(d['fid']!=0);va=np.flatnonzero(d['fid']==0)
 pred,rec=e.fit(d,None,None,tr,va,0,-1,'shortB',e.OUT/'incumbent-head-control-replay')
 original=e.old.s.foldpath('original',0);saved=torch.load(_repo_io(original/'model.pt'),map_location='cpu',weights_only=True);new=torch.load(_repo_io(e.OUT/'incumbent-head-control-replay/model.pt'),map_location='cpu',weights_only=True)
 assert all(torch.equal(v,saved['state'][k]) for k,v in new['state'].items())
 np.testing.assert_array_equal(pred,d['basehead'][va]);assert saved['train_ids']==new['train_ids'] and saved['validation_ids']==new['validation_ids']
 record={'passed':True,'new_wrapper_12epoch_state_exact_incumbent':True,'prediction_reload_error':0.,'incumbent_checkpoint_sha256':e.sha(original/'model.pt'),'new_report_sha256':e.sha(e.OUT/'incumbent-head-control-replay/report.json'),'source_sha256':e.sha(Path(__file__))};e.write(e.OUT/'incumbent-head-control-validation.json',record);e.emit(record)
if __name__=='__main__':main()
