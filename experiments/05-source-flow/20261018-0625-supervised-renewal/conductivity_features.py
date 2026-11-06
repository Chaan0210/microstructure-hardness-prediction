"""Nonlocal descriptors of supplied-image phase maps; no generated training images."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
import hashlib,json,time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg,LinearOperator,spsolve
from threadpoolctl import threadpool_limits

def solve(k):
    n=k.shape[0];assert k.shape==(n,n) and np.isfinite(k).all() and k.min()>0
    ix=np.arange(n*n).reshape(n,n);right=np.roll(ix,-1,axis=1);down=np.roll(ix,-1,axis=0)
    kr=np.roll(k,-1,axis=1);kd=np.roll(k,-1,axis=0);wx=2*k*kr/(k+kr);wy=2*k*kd/(k+kd)
    u=np.r_[ix.ravel(),ix.ravel()];v=np.r_[right.ravel(),down.ravel()];w=np.r_[wx.ravel(),wy.ravel()]
    L=sparse.coo_matrix((np.r_[w,w,-w,-w],(np.r_[u,v,u,v],np.r_[u,v,v,u])),shape=(n*n,n*n)).tocsr()
    A=LinearOperator(L.shape,matvec=lambda x:L@x+x.mean(),dtype=np.float64)
    diagonal=L.diagonal()+1/(n*n);M=LinearOperator(L.shape,matvec=lambda x:x/diagonal,dtype=np.float64)
    columns=[];energies=[];amplitudes=[];iterations=[];fallbacks=0
    for axis in [1,0]:
        face=wx if axis==1 else wy;b=(face-np.roll(face,1,axis=axis)).ravel();b-=b.mean();count=[0]
        def callback(_):count[0]+=1
        chi,info=cg(A,b,rtol=1e-8,atol=1e-10,maxiter=1500,M=M,callback=callback)
        if info:
            chi=np.r_[0.,spsolve(L[1:,1:],b[1:])];fallbacks+=1
        chi=chi.reshape(n,n);ex=(1. if axis==1 else 0.)+np.roll(chi,-1,axis=1)-chi;ey=(1. if axis==0 else 0.)+np.roll(chi,-1,axis=0)-chi
        columns.append([float((wx*ex).mean()),float((wy*ey).mean())]);er=wx*ex*ex;ed=wy*ey*ey
        energies.append(.5*(er+np.roll(er,1,axis=1)+ed+np.roll(ed,1,axis=0)))
        xr=ex*ex;yd=ey*ey;amplitudes.append(.5*(xr+np.roll(xr,1,axis=1)+yd+np.roll(yd,1,axis=0)))
        iterations.append(count[0])
    K=np.array(columns).T;asym=float(abs(K[0,1]-K[1,0]));K=(K+K.T)/2;ev=np.linalg.eigvalsh(K)
    assert asym<1e-5*max(1.,np.trace(K));assert ev.min()>=1/np.mean(1/k)-1e-5 and ev.max()<=k.mean()+1e-5
    energy=np.mean(energies,0);amplitude=np.mean(amplitudes,0);eff=np.trace(K)/2
    assert abs(energy.mean()-eff)<1e-5*max(1.,eff)
    diff=K[0,0]-K[1,1];cross=2*K[0,1];trace=np.trace(K)
    top=np.sort(energy.ravel())[-int(np.ceil(.1*n*n)):].sum()/energy.sum()
    tile=energy.reshape(4,n//4,4,n//4).mean((1,3))
    features=[eff/k.mean(),np.log(eff),(ev[1]-ev[0])/trace,(diff*diff-cross*cross)/(trace*trace),2*diff*cross/(trace*trace),
              energy.std()/eff,*np.quantile(energy,[.9,.99])/eff,top,tile.std()/eff,amplitude.mean(),np.quantile(amplitude,.99)/amplitude.mean()]
    return np.array(features),dict(tensor=K.tolist(),iterations=iterations,asymmetry=asym,fallbacks=fallbacks)

def verify():
    # Mathematical coefficient fixtures only; never images or regression training rows.
    uniform=np.full((16,16),3.);_,u=solve(uniform);assert np.max(abs(np.array(u['tensor'])-np.eye(2)*3))<1e-10
    layered=np.ones((16,16))*2;layered[:,8:]=8;_,l=solve(layered)
    expected=np.diag([1/np.mean(1/layered),layered.mean()]);error=float(abs(np.array(l['tensor'])-expected).max());assert error<1e-7
    maps=np.load(_repo_path(ROOT, 'experiments/push-400/unet4maps-z1-train.npz'));p=maps['maps'][0,1].astype(float);k=1+7*p.reshape(64,2,64,2).mean((1,3));f,a=solve(k);g,b=solve(np.rot90(k));rot_error=float(abs(f-g).max());assert rot_error<1e-5
    return dict(uniform_exact=True,layered_tensor_error=error,original_train_field_C4_descriptor_error=rot_error,fixture_training_rows=0)

def main():
    out=_repo_path(HERE, 'conductivity');out.mkdir(exist_ok=True);check=verify();(out/'verification.json').write_text(json.dumps(check,indent=2));print('VERIFIED',check,flush=True)
    source=_repo_path(ROOT, 'experiments/push-400/unet4maps-z1-train.npz');z=np.load(_repo_io(source));X=[];S=[];records=[];start=time.monotonic()
    for j,ident in enumerate(z['ids']):
        m=z['maps'][j,:4].astype(float).reshape(4,64,2,64,2).mean((2,4));phase=np.clip(m[1]/np.maximum(m[0]+m[1],1e-6),0,1);pore=np.clip(m[2],0,1)
        marginal=np.r_[phase.mean(),phase.std(),np.quantile(phase,[.1,.25,.5,.75,.9]),pore.mean(),pore.std()]
        features=[];audit=[]
        for ratio in [2.,8.,32.]:
            k=(1+(ratio-1)*phase)*(1-.9*pore);v,r=solve(k);features.extend(v);audit.append(r)
        X.append(features);S.append(marginal);records.append(dict(id=str(ident),solves=audit))
        if (j+1)%50==0:print('EXTRACTED',j+1,'seconds',round(time.monotonic()-start,1),flush=True)
    np.savez(_repo_io(out/'features.npz'),ids=z['ids'],pde=np.array(X),marginal=np.array(S))
    spec=dict(rows=500,grid=64,contrast_ratios=[2,8,32],pore_factor=.1,pde_dimension=36,marginal_dimension=9,seconds=time.monotonic()-start,
        source_map_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Deterministic per-image extraction, no target fitting. Inherited maps have historical train-image exposure: downstream comparison is conditional.',
        interpretation='Conductivity is a nonlocal structural descriptor, not a calibrated physical hardness simulation. Periodicity and artificial contrast ratios are feature definitions.',
        reference='https://www.sciencedirect.com/science/article/abs/pii/S0020768324001045',solver='Finite-volume harmonic face conductance, periodic correctors, conjugate gradients with mean-zero gauge; not an FFT-paper replication.',
        no_generated_training_images=True,solves=records)
    (out/'provenance.json').write_text(json.dumps(spec,indent=2));print('DONE',len(X),round(time.monotonic()-start,1),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
