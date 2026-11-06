"""Evaluate an earlier segmenter on hardness, rather than an unlabeled bias proxy.

R300 was stopped after a distribution-shift diagnostic; its downstream hardness
prediction was not tested. This compares its cached single-view measurements
with matched H300 single-view measurements and the actual production bank.
"""
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
import json,hashlib,time,gc
import numpy as np
import torch
from threadpoolctl import threadpool_limits

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);d=np.load(_repo_path(HERE, 'data.npz'));ids=d['ids'][d['dev']];y=d['y'][d['dev']];fold=d['fold'];noise=d['noise'][d['dev']];out=_repo_path(HERE, 'r300-readout');out.mkdir(exist_ok=True)
    fp=_repo_path(ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz');z=np.load(_repo_io(fp));keys=z['keys'].tolist();pos={str(v):i for i,v in enumerate(z['ids'])};X=z['X'][[pos[str(v)] for v in ids]].astype(float);datasets={};sources={};common=None
    for tag in ['H300','R300']:
        file=_repo_path(ROOT, f'experiments/push-300/ftseg-ft-{tag}-train-noisy-n1.npz');r=np.load(_repo_io(file));rp={str(v):i for i,v in enumerate(r['ids'])};rk=r['keys'].tolist();columns=[k for k in keys if k in rk];A=X.copy();changed=[]
        assert all(str(i) in rp for i in ids[noise>=12])
        for j,ident in enumerate(ids):
            if noise[j]<12:continue
            for key in columns:A[j,keys.index(key)]=r['X'][rp[str(ident)],0,rk.index(key)]
            A[j,keys.index('sizefield_std_s32_x_m_el_pix')]=A[j,keys.index('sizefield_std_s32')]*A[j,keys.index('m_el_pix')];changed.append(str(ident))
        # Quality scalars are original-image properties, not segmenter outputs.
        for key in ['noise','img_std']:assert np.nanmax(abs(A[:,keys.index(key)]-X[:,keys.index(key)]))<1e-5
        if common is None:common=columns
        else:assert columns==common
        datasets[tag]=A;sources[tag]=dict(path=str(file.relative_to(ROOT)),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),rows=changed)
    spec=dict(hypothesis='PriorR300rejectionwasbasedonshiftinunlabeledgraincount/size,notactualhardness. Biasedmeasurementscanstillpredictafterreadout.',
        changed_columns=common,unchanged='ExistingU-Netandview-disagreementfeaturesretained;onlycommon64-fieldmeasurementsreplacedonnoise>=12;dependent sizefield×elongation product recomputed.',
        primary='Complete083+.384*(R300hybridTabPFN -actualphysical104TabPFN),onlynoise>=12andnonfailed.',
        controls='H300hybrid usesexactlysame1-viewreplacementcolumns/rows; R-minus-Hdifferencealsoreported.',
        model='localTabPFN3.5n8,seeds0/100/200+fold,CPU',scope='Conditionalhistoricalrepresentationcomparison. R300andH300werealreadytrainedon250cleanestprovidedtrainimages;globalaugmentationpoolusedhistoricaltrainstatistics. No newimage/modeltraining/testreads.',
        prior_training='push-300/ft_cellpose.py,300updates,lr2e-5,batch4,freeze0,holdout0;R300real-original-image-degradation,H300heavyordinarydegradation. No renderednewlayout.',sources=sources,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'recipe.json').write_text(json.dumps(spec,indent=2));sys.path.insert(0,str(_repo_path(ROOT, 'experiments/push-300')));import harness;pred={k:np.zeros(400) for k in ['physical','H300','R300']}
    for k in range(5):
        tr=np.flatnonzero(fold!=k);q=np.flatnonzero(fold==k);ps=[]
        for seed in [0,100,200]:
            z=np.load(_repo_path(HERE, f'conductivity/readout/physical-f{k}-s{seed}.npz'));assert np.array_equal(z['train'],tr) and np.array_equal(z['query'],q);ps.append(z['prediction'])
        pred['physical'][q]=np.mean(ps,0)
        for tag,A in datasets.items():
            ps=[]
            for seed in [0,100,200]:
                file=out/f'{tag}-f{k}-s{seed}.npz'
                if file.exists():p=np.load(_repo_io(file))['prediction']
                else:
                    start=time.monotonic();m=harness.tabpfn_new('v3.5',seed+k,'cpu',n_est=8);m.fit(A[tr],y[tr]);p=m.predict(A[q]);np.savez(_repo_io(file),train=tr,query=q,prediction=p);del m;gc.collect();print(tag,k,seed,time.monotonic()-start,flush=True)
                ps.append(p)
            pred[tag][q]=np.mean(ps,0)
    elig=(noise>=12)&~d['failed']
    for tag in ['H300','R300']:pred[tag+'_system']=d['base083']+.384*elig*(pred[tag]-pred['physical'])
    pred['R_minus_H_system']=d['base083']+.384*elig*(pred['R300']-pred['H300']);pred['base083']=d['base083'];r={k:dict(rmse=float(np.mean((v-y)**2)**.5),fold_rmse=[float(np.mean((v[fold==f]-y[fold==f])**2)**.5) for f in range(5)]) for k,v in pred.items()};np.savez(_repo_io(out/'oof.npz'),ids=ids,y=y,fold=fold,**pred);(out/'results.json').write_text(json.dumps(r,indent=2));print('DONE',json.dumps(r),flush=True)

if __name__=='__main__':
    with threadpool_limits(2):main()
