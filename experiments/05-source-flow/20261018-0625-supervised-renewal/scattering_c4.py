"""Retain signed angular scattering interactions lost by full rotation pooling."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
for k in ['TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME']:os.environ[k]=str(_repo_path(HERE, 'tmp'))
os.environ['HF_HOME']=str(_repo_path(HERE, 'tmp/huggingface'));os.environ['HF_HUB_OFFLINE']='1';os.environ['TABPFN_DISABLE_TELEMETRY']='1'
import json,hashlib,gc,time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from scattering_readout import reduced
PAIRS=[(0,0),(1,-1),(1,1),(2,0),(0,2),(2,-2),(2,2)]

def transform(W,shift=0):
    S1=W[:,1:49].reshape(-1,6,8);S1=np.roll(S1,shift,2);theta=2*np.pi*np.arange(8)/8
    first=np.log(np.maximum(S1,1e-8));cols=[W[:,:1]]
    for h in [0,2,4]:
        cols.append((first*np.cos(h*theta)[None,None]).mean(2))
        if h==2:cols.append((first*np.sin(h*theta)[None,None]).mean(2))
    meta=[(j,t) for j in range(6) for t in range(8)];lookup={};column=49
    for j1,t1 in meta:
        for j2,t2 in meta:
            if j2>j1:lookup[(j1,j2,t1,t2)]=column;column+=1
    for j1 in range(6):
        for j2 in range(j1+1,6):
            s=np.stack([W[:,lookup[(j1,j2,t1,t2)]] for t1 in range(8) for t2 in range(8)],1).reshape(-1,8,8);s=np.roll(np.roll(s,shift,1),shift,2)
            V=np.log(np.maximum(s,1e-9)/np.maximum(S1[:,j1,:,None],1e-8))
            for a,b in PAIRS:
                angle=a*theta[:,None]+b*theta[None,:];cols.append((V*np.cos(angle)[None]).mean((1,2))[:,None])
                if (a,b)!=(0,0):cols.append((V*np.sin(angle)[None]).mean((1,2))[:,None])
    return np.concatenate(cols,1)

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);d=np.load(_repo_path(HERE, 'data.npz'));ids=d['ids'][d['dev']];y=d['y'][d['dev']];fold=d['fold'];out=_repo_path(HERE, 'scattering-c4');out.mkdir(exist_ok=True)
    path=_repo_path(ROOT, 'artifacts/wavelet-scattering-096/development_features.npz');z=np.load(_repo_io(path));pos={str(v):i for i,v in enumerate(z['ids'])};W=z['X'][[pos[str(v)] for v in ids]];C=transform(W);S=reduced(W);assert C.shape==(400,220);error=float(abs(C-transform(W,4)).max());assert error<1e-10
    fp=_repo_path(ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz');f=np.load(_repo_io(fp));pos={str(v):i for i,v in enumerate(f['ids'])};X=f['X'][[pos[str(v)] for v in ids]].astype(float)
    spec=dict(hypothesis='Earlier58scatteringpooledallabsoluteangles andreflections, losingknownsignedfourth-orderorientationinformation. RetainC4-invariantbutreflection-sensitiveabsolutepairharmonics.',
        columns='S0+24logS1harmonics+15scalepairsx13logS2/S1harmonics=220; PAIRS='+repr(PAIRS),orientation_permutation_C4_error=error,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),feature_sha256=hashlib.sha256(fp.read_bytes()).hexdigest(),
        model='local TabPFN3.5 n8 seeds0/100/200+fold CPU',control='samephysical104alone andsamephysical104+previous58rotation/reflectionpooledfeatures',
        formula='Complete083+.384*(newphysical104+features -matchedphysical104),existingfailurefallback; conditionalhistoricalrepresentationcomparison.')
    (out/'recipe.json').write_text(json.dumps(spec,indent=2));np.savez(_repo_io(out/'features.npz'),ids=ids,C4=C,invariant=S)
    sys.path.insert(0,str(_repo_path(ROOT, 'experiments/push-300')));import harness
    ps={k:np.zeros(400) for k in ['physical','invariant','C4']}
    for f in range(5):
        tr=np.flatnonzero(fold!=f);q=np.flatnonzero(fold==f)
        controls=[]
        for seed in [0,100,200]:
            z=np.load(_repo_path(HERE, f'conductivity/readout/physical-f{f}-s{seed}.npz'));assert np.array_equal(z['train'],tr) and np.array_equal(z['query'],q);controls.append(z['prediction'])
        ps['physical'][q]=np.mean(controls,0)
        for name,A in [('invariant',np.c_[X,S]),('C4',np.c_[X,C])]:
            pp=[]
            for seed in [0,100,200]:
                cache=out/f'{name}-f{f}-s{seed}.npz'
                if cache.exists():pred=np.load(_repo_io(cache))['prediction']
                else:
                    start=time.monotonic();m=harness.tabpfn_new('v3.5',seed+f,'cpu',n_est=8);m.fit(A[tr],y[tr]);pred=m.predict(A[q]);np.savez(_repo_io(cache),train=tr,query=q,prediction=pred);del m;gc.collect();print(name,f,seed,'seconds',time.monotonic()-start,flush=True)
                pp.append(pred)
            ps[name][q]=np.mean(pp,0)
    for name in ['invariant','C4']:ps[name+'_system']=d['base083']+.384*(~d['failed'])*(ps[name]-ps['physical'])
    ps['base083']=d['base083'];results={k:dict(rmse=float(np.sqrt(np.mean((v-y)**2))),fold_rmse=[float(np.sqrt(np.mean((v[fold==f]-y[fold==f])**2))) for f in range(5)]) for k,v in ps.items()};np.savez(_repo_io(out/'oof.npz'),ids=ids,fold=fold,y=y,**ps);(out/'results.json').write_text(json.dumps(results,indent=2));print('DONE',json.dumps(results),flush=True)

if __name__=='__main__':
    with threadpool_limits(2):main()
