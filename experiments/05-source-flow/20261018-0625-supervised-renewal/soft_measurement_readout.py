"""Matched five-fold check of fixed grain-distribution measurements.

The eleven learned parameters were not responsible for the first-fold gain:
fixed measurements performed better. Test the fixed representation with both
the exact one-estimator screen and the production three-seed/eight-estimator
readout. No further measurement fitting or coefficient search is performed.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
import json,hashlib,time,gc
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import pfn_gradient_probe as pf
import pfn_feature_learning as fl
import soft_measurement as sm
from soft_pfn_learning import Measurement

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);d,X=pf.load();data,md=sm.load();assert np.array_equal(md['ids'],d['ids']);dev=d['dev'];fold=d['fold'];ids=d['ids'][dev];y=d['y'][dev];out=_repo_path(HERE, 'soft-measurement-readout');out.mkdir(exist_ok=True)
    recipe=dict(primary='Fixedsoft24features with one-estimator identityTabPFN3.5, exactly matching initial soft-pfn-f0 model.',secondary='Samefixed24features appendedtoactualphysical104 withstandardn8x3TabPFN; matchedphysical104n8x3control reused.',
        controls='Identityphysical104alone(n1seed0) andproductionphysical104(n8seeds0/100/200+fold).',
        parameters='theta=0: thresholds.15/.30/.50,width.06,shadingblend.5,varianceScale1,noiseSlope0. Nolearnedmeasurementparameters.',
        preprocessing='Exactsource-onlyFeatures/Measurement normalizationforidentitymodel;nativeTabPFNpreprocessingforn8. Allnewmeasurementsareper-imageoriginal-pixel/frozen-zero-shot-maskfunctions.',
        formula='Complete083+.384*(candidate -productionphysical104n8x3control),preserveexistingfailed-maskfallback.',scope='Conditionalhistoricalphysical104;newimagefeaturesfixedexternal/per-imageonly.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'recipe.json').write_text(json.dumps(recipe,indent=2));pred={k:np.zeros(400) for k in ['physical_n8','physical_identity','fixed_identity','fixed_n8']};sys.path.insert(0,str(_repo_path(ROOT, 'experiments/push-300')));import harness
    # Constant extra measurements for the standard readout; no cross-image statistics.
    with torch.no_grad():
        F=sm.features(data,torch.zeros(11,dtype=torch.float64))[:,16:];ar=data['core'][:,6:7];F=torch.cat([F,F[:,:3]*ar,F[:,12:15]*ar],1).numpy()
    np.savez(_repo_io(out/'fixed-features.npz'),ids=d['ids'],X=F)
    for k in range(5):
        ti=np.flatnonzero(fold!=k);qi=np.flatnonzero(fold==k);tr=dev[ti];q=dev[qi];norm=fl.Features().fit(X[tr]);base=norm.transform(X);model=Measurement(data,base,tr);yy=torch.tensor(d['y'][tr],dtype=torch.float32)
        controls=[]
        for seed in [0,100,200]:
            z=np.load(_repo_path(HERE, f'conductivity/readout/physical-f{k}-s{seed}.npz'));assert np.array_equal(z['train'],ti) and np.array_equal(z['query'],qi);controls.append(z['prediction'])
        pred['physical_n8'][qi]=np.mean(controls,0)
        for name in ['physical_identity','fixed_identity']:
            path=out/f'{name}-f{k}.npz'
            if path.exists():pr=np.load(_repo_io(path))['prediction']
            else:
                reg=pf.make(0)
                with torch.no_grad():
                    T=base[tr] if name=='physical_identity' else model(tr);Q=base[q] if name=='physical_identity' else model(q);pr=pf.forward(reg,T,yy,Q).numpy()
                np.savez(_repo_io(path),train=tr,query=q,prediction=pr);del reg;gc.collect()
            if k==0 and name=='fixed_identity':
                expected=np.load(_repo_path(HERE, 'runs/soft-pfn-f0/outer/initial.npz'))['prediction'];error=float(abs(pr-expected).max());print('INITIAL_REPRODUCTION_ERROR',error,flush=True);assert error<.001
            pred[name][qi]=pr
        vals=[]
        for seed in [0,100,200]:
            path=out/f'fixed_n8-f{k}-s{seed}.npz'
            if path.exists():pr=np.load(_repo_io(path))['prediction']
            else:
                start=time.monotonic();reg=harness.tabpfn_new('v3.5',seed+k,'cpu',n_est=8);reg.fit(np.c_[X[tr],F[tr]],d['y'][tr]);pr=reg.predict(np.c_[X[q],F[q]]);np.savez(_repo_io(path),train=tr,query=q,prediction=pr);del reg;gc.collect();print('N8',k,seed,'seconds',time.monotonic()-start,flush=True)
            vals.append(pr)
        pred['fixed_n8'][qi]=np.mean(vals,0)
        eligible=~d['failed'][qi];bs=d['base083'][qi]
        print('FOLD',k,json.dumps({name:dict(raw=float(np.mean((pred[name][qi]-y[qi])**2)**.5),system=float(np.mean((bs+.384*eligible*(pred[name][qi]-pred['physical_n8'][qi])-y[qi])**2)**.5)) for name in ['physical_identity','fixed_identity','fixed_n8']}),flush=True)
    for name in ['physical_identity','fixed_identity','fixed_n8']:pred[name+'_system']=d['base083']+.384*(~d['failed'])*(pred[name]-pred['physical_n8'])
    pred['base083']=d['base083'];r={k:dict(rmse=float(np.mean((v-y)**2)**.5),fold_rmse=[float(np.mean((v[fold==f]-y[fold==f])**2)**.5) for f in range(5)]) for k,v in pred.items()};np.savez(_repo_io(out/'oof.npz'),ids=ids,y=y,fold=fold,**pred);(out/'results.json').write_text(json.dumps(r,indent=2));print('DONE',json.dumps(r),flush=True)

if __name__=='__main__':
    with threadpool_limits(2):main()
