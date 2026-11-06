"""Controlled all-stage mean/std pooling experiment; scratch weights, same convolutional trunk."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, '.venv')).resolve():raise SystemExit('Use project .venv')
sys.path.insert(0,str(ROOT))
import cnn_controls as engine
import torch
from torch import nn
class MomentCNN(engine.TinyCNN):
    def __init__(self):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(418,64),nn.SiLU(),nn.Dropout(.15),nn.Linear(64,1))
    def forward(self,x):
        stats=[x.mean((2,3)),(x.var((2,3),correction=0)+1e-6).sqrt()]
        h=x
        for i,layer in enumerate(self.features):
            h=layer(h)
            if i%3==2:stats.extend([h.mean((2,3)),(h.var((2,3),correction=0)+1e-6).sqrt()])
        return self.head(torch.cat(stats,1)).squeeze(1)
if __name__=='__main__':
    plan={'variant':'MomentCNN: mean/std from every stage; identical conv widths/kernels/strides/GroupNorm','seed':20261003,'image_size':128,'epochs':40,'no_pretrained':True,'weights':{'baseline':.5,'grain':.25,'cnn':.25},'source_sha256':engine.sha(Path(__file__)),'engine_sha256':engine.sha(_repo_path(ROOT, 'cnn_controls.py')),'parameters':sum(p.numel() for p in MomentCNN().parameters()),'license':'Original project implementation using installed PyTorch BSD-3-Clause; no external model code/weights copied'}
    engine.write(_repo_path(ROOT, 'reports/cnn-moments-plan.json'),plan)
    engine.TinyCNN=MomentCNN
    engine.cv(_repo_path(ROOT, 'artifacts/run-011'),_repo_path(ROOT, 'artifacts/cnn-benchmark-001/benchmark.json'),128,40)
    p=_repo_path(ROOT, 'artifacts/run-011/report.json');r=engine.json.loads(p.read_text());r['architecture_provenance']=plan;engine.write(p,r)
