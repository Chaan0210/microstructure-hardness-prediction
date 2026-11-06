"""Image-only amplitude audit after training; no learned parameter or label use."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,ast,hashlib
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3];OLD=_repo_path(ROOT, 'experiments/highnoise-relational-20261016')
def main():
    d=np.load(_repo_path(OLD, 'inputs.npz'));dev=d['dev'];source=_repo_path(HERE, 'study.py');tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['regions','photo']];ns=dict(np=np,ndi=ndi);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    rows=[]
    for j in dev:
        with Image.open(_repo_path(ROOT, f"data/train/{d['ids'][j]}.png")) as im:p=np.asarray(im,np.uint8)[None,None]
        mask,means,sd,tile=ns['regions'](p);x=p.astype(np.float32)/255;r=dict(ID=str(d['ids'][j]),noise=float(d['noise'][j]),raw_std_gray255=float(sd[0]*255),region_support=float((mask>0).mean()),strengths={})
        for strength in [.03,.1,.3]:
            z=ns['photo'](x,mask,means,sd,tile,np.random.default_rng(2026101672+int(j)),strength);diff=z-x
            r['strengths'][str(strength)]=dict(RMS_gray255=float(np.sqrt(np.mean(diff**2))*255),max_gray255=float(abs(diff).max()*255),changed_fraction=float((abs(diff)>1e-7).mean()))
        rows.append(r)
    summary={}
    for name,rr in [('all',rows),('high_noise',[r for r in rows if r['noise']>=12]),('other',[r for r in rows if r['noise']<12])]:
        summary[name]=dict(n=len(rr),region_support_median=float(np.median([r['region_support'] for r in rr])),strengths={v:{metric:dict(median=float(np.median([r['strengths'][v][metric] for r in rr])),p05_p95=np.quantile([r['strengths'][v][metric] for r in rr],[.05,.95]).tolist()) for metric in ['RMS_gray255','max_gray255','changed_fraction']} for v in ['0.03','0.1','0.3']})
    with (_repo_path(HERE, 'strength-audit.json')).open('x') as f:json.dump(dict(label_use=False,selection_use=False,scope='Actual one-view amplitude on the same400 sources. Descriptive, not an independent validation or exhaustive stochastic transform test.',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),summary=summary,rows=rows),f,indent=2)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
