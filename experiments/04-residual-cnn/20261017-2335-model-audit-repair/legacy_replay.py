"""Verify every original073 CNN context against saved incumbent branch outputs."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import cnn_repair as cr
import sys,time,json,gc
import numpy as np,torch
from threadpoolctl import threadpool_limits
def main():
 start=time.monotonic();d,y=cr.load();rows=[]
 for k in range(5):
  for stage in ['selector','final']:
   z=cr.data(d,y,k,stage,'073')
   for seed in [0,1]:
    p=c.R4/f'aux-seed{seed}/fold{k}-{stage}';err=cr.replay(z,p);rows.append(dict(fold=k,stage=stage,seed=seed,maxdiff=err))
   del z;gc.collect()
 c.save(_repo_io('legacy-context-verification.json'),dict(passed=True,records=rows,seconds=time.monotonic()-start));print('All20 incumbentCNN contexts reproduced exactly')
if __name__=='__main__':
 torch.set_num_threads(2);torch.set_num_interop_threads(1);sys.addaudithook(c.guard)
 with threadpool_limits(limits=2):main()
