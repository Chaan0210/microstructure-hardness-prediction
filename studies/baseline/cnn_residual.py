"""Scratch amplitude-preserving residual moment network. No pretrained code/weights."""
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
class ResidualStage(nn.Module):
    def __init__(self,inc,out,first=False):
        super().__init__();self.down=nn.Conv2d(inc,out,5 if first else 3,stride=2,padding=2 if first else 1)
        self.residual=nn.Sequential(nn.Conv2d(out,out,3,padding=1),nn.SiLU(),nn.Conv2d(out,out,3,padding=1));self.act=nn.SiLU()
    def forward(self,x):
        h=self.act(self.down(x));return self.act(h+.1*self.residual(h))
class ResidualMomentCNN(nn.Module):
    def __init__(self):
        super().__init__();widths=[16,32,48,64];self.stages=nn.ModuleList([ResidualStage(1 if i==0 else widths[i-1],n,i==0) for i,n in enumerate(widths)])
        self.head=nn.Sequential(nn.Linear(322,64),nn.SiLU(),nn.Dropout(.15),nn.Linear(64,1))
    def forward(self,x):
        values=[x.mean((2,3)),(x.var((2,3),correction=0)+1e-6).sqrt()];h=x
        for stage in self.stages:
            h=stage(h);values.extend([h.mean((2,3)),(h.var((2,3),correction=0)+1e-6).sqrt()])
        return self.head(torch.cat(values,1)).squeeze(1)
if __name__=='__main__':
    plan={'variant':'ResidualMomentCNN, no activation normalization, mean/std across4stages','hypothesis':'Preserve amplitude and multiscale texture signals suppressed by GroupNorm; residual local processing','controlled_package':'Architecture changes as one prespecified package; not isolating each block effect','seed':20261003,'image_size':128,'epochs':40,'ensemble_weights':{'baseline':.5,'grain':.25,'cnn':.25},'parameters':sum(p.numel() for p in ResidualMomentCNN().parameters()),'source_sha256':engine.sha(Path(__file__)),'engine_sha256':engine.sha(_repo_path(ROOT, 'cnn_controls.py')),'license':'Original project source; PyTorch BSD3 library. No external model or weights.'}
    engine.write(_repo_path(ROOT, 'reports/cnn-residual-plan.json'),plan);engine.TinyCNN=ResidualMomentCNN
    engine.cv(_repo_path(ROOT, 'artifacts/run-013'),_repo_path(ROOT, 'artifacts/cnn-benchmark-001/benchmark.json'),128,40)
    p=_repo_path(ROOT, 'artifacts/run-013/report.json');r=engine.json.loads(p.read_text());r['architecture_provenance']=plan;engine.write(p,r)
