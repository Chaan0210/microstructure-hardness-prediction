"""Isolated conditional screens; no test inputs, no old-output writes."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
os.environ['MPLCONFIGDIR']=str(_repo_path(HERE, 'mpl-cache'))
os.environ['HF_HUB_OFFLINE']='1'
import ast, argparse, hashlib, json, time, csv
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from skimage import measure
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import joblib

def guard(event,args):
    if event=='open':
        p,mode,flags=args
        if not isinstance(p,(str,bytes,os.PathLike)): return
        p=Path(os.fsdecode(p)).absolute()
        writing=(mode is not None and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing and not p.is_relative_to(HERE): raise RuntimeError('Write outside experiment: '+str(p))
        if p.is_relative_to(_repo_path(ROOT, 'data/test')) or (p.is_relative_to(ROOT) and ('-test.' in p.name or p.name.startswith('submission-'))):
            raise RuntimeError('Test input blocked during research: '+str(p))

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def save(name,obj):
    p=_repo_path(HERE, name);p.parent.mkdir(exist_ok=True,parents=True)
    with p.open('x') as f: json.dump(obj,f,indent=2,allow_nan=False)

def arrays(name,**kw):
    p=_repo_path(HERE, name);assert not p.exists(),p
    np.savez_compressed(_repo_io(p),**kw)

def load_data():
    path=_repo_path(ROOT, 'experiments/rmse-research-20261015/inputs.npz')
    d=dict(np.load(_repo_io(path)));y=d.pop('y')
    base=dict(np.load(_repo_path(ROOT, 'experiments/076-complement-20261016/baseline076.npz')))
    assert np.array_equal(base['ids'],d['ids'][d['dev']]) and np.array_equal(base['fold'],d['fold'][0])
    gx=np.load(_repo_path(ROOT, 'experiments/rmse-research-20261015/round2/gmm-features500.npz'))
    assert np.array_equal(gx['ids'],d['ids']);d['gmm']=gx['X'];d['base']=base
    assert set(d['src'])<=set(range(500))
    return d,y

def target_function():
    p=_repo_path(ROOT, 'experiments/push-400/phase_unet4.py');tree=ast.parse(p.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='targets')
    ns=dict(np=np,ndi=ndi,measure=measure)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns)
    return ns['targets']

def weighted_quantile(x,w,quantiles):
    ix=np.argsort(x,kind='stable');ww=w[ix];c=np.cumsum(ww)-.5*ww
    return np.interp(quantiles,c/c[-1],x[ix])

def map_summary(p):
    g=p[0]+p[1];a=p[:4].argmax(0)
    return np.array([p[1].sum()/max(g.sum(),1e-8),
        (a==1).sum()/max(((a==0)|(a==1)).sum(),1),
        (ndi.gaussian_filter(p[1],16)/np.maximum(ndi.gaussian_filter(g,16),1e-3)).std()])

def extract():
    assert (_repo_path(HERE, 'incumbent.json')).exists(),'Freeze must complete first'
    start=time.monotonic();mfile=_repo_path(ROOT, 'experiments/grain-cellpose-220/masks-v3-train.npz')
    M=np.load(_repo_io(mfile));ids=M['ids'].astype(str);masks=M['masks']
    noise=np.load(_repo_path(ROOT, 'experiments/push-300/noise_train.npy'))
    clean=np.flatnonzero(noise<np.median(noise));val=clean[np.random.RandomState(0).permutation(len(clean))[:30]]
    tr=np.setdiff1d(clean,val);targets=target_function();counts=np.zeros(4,np.int64);truth={};used={str(mfile.relative_to(ROOT)):digest(mfile)}
    for k in clean:
        p=_repo_path(ROOT, f'data/train/{ids[k]}.png');a=np.asarray(Image.open(_repo_io(p)),np.float32)
        t,_,_=targets(a,masks[k])
        if k in tr: counts+=np.bincount(t.ravel(),minlength=4)
        else: truth[int(k)]=t[::2,::2]
        used[str(p.relative_to(ROOT))]=digest(p)
    freq=counts/counts.sum();weights=1/np.sqrt(freq+.001);weights/=weights.mean()
    assert len(tr)==219 and len(val)==30
    del masks,M
    p=_repo_path(ROOT, 'experiments/push-400/unet4maps-z1-train.npz');u=np.load(_repo_io(p))
    assert np.array_equal(ids,u['ids'].astype(str));maps=u['maps'];used[str(p.relative_to(ROOT))]=digest(p)
    rng=np.random.default_rng(2026101601);dirs=rng.normal(size=(16,4));dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
    dirs=np.r_[np.eye(4),dirs];quantiles=np.linspace(.025,.975,17)
    signature=[];raw=[];corrected=[];errors=[]
    for k in range(len(ids)):
        z=maps[k].astype(float);pr=z[:4];pr=pr/np.maximum(pr.sum(0),1e-10)
        pc=pr/weights[:,None,None];pc/=pc.sum(0)
        raw.append(map_summary(pr));corrected.append(map_summary(pc))
        g=np.maximum(pr[0]+pr[1],1e-6);w=g.ravel()
        physical=np.stack([(z[4]-5.636784553527832)/.9691801071166992,
            (z[7]-.15)/.15,np.hypot(z[5],z[6])/.3,2*pr[1]/g],-1).reshape(-1,4)
        proj=physical@dirs.T
        signature.append(np.array([weighted_quantile(proj[:,j],w,quantiles) for j in range(20)]))
        if k in truth:
            t=truth[k];f=(t==1).sum()/max(((t==0)|(t==1)).sum(),1)
            record=dict(id=str(ids[k]),phase_true=float(f))
            for name,q in [('raw',pr),('corrected',pc)]:
                record[name]=dict(phase=float(map_summary(q)[0]),phase_error2=float((map_summary(q)[0]-f)**2),
                    brier=[float(np.mean((q[c]-(t==c))**2)) for c in range(4)],
                    logloss=float(-np.log(np.maximum(np.take_along_axis(q,t[None],axis=0),1e-10)).mean()))
            errors.append(record)
        if (k+1)%100==0: print('extracted',k+1,'seconds',round(time.monotonic()-start,1),flush=True)
    stats={name:dict(phase_mse=float(np.mean([e[name]['phase_error2'] for e in errors])),
        brier=np.mean([e[name]['brier'] for e in errors],axis=0).tolist(),logloss=float(np.mean([e[name]['logloss'] for e in errors]))) for name in ['raw','corrected']}
    gate=stats['corrected']['phase_mse']<=.95*stats['raw']['phase_mse'] and all(stats['corrected']['brier'][j]<=1.05*stats['raw']['brier'][j] for j in [2,3])
    arrays('measurements.npz',ids=ids,signature=np.array(signature),raw=np.array(raw),corrected=np.array(corrected),directions=dirs,quantiles=quantiles,unet_val=val)
    save('measurement-screen.json',dict(class_frequencies=freq.tolist(),class_weights=weights.tolist(),n_unet_training=len(tr),validation_ids=ids[val].tolist(),
        h1_advance=bool(gate),stats=stats,per_image=errors,source_sha256=used,code_sha256=digest(__file__),
        seconds=time.monotonic()-start,hardness_used=False,limits='Weak mask targets; existing downsampled float16 TTA-mean maps. Validation 30 is auxiliary-model-heldout, not untouched hardness data.'))
    print('MEASUREMENT SCREEN',json.dumps(stats),'advance',gate,flush=True)

def standardize(x,q):
    imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(x)
    sc=StandardScaler().fit(imp.transform(x))
    return sc.transform(imp.transform(x)),sc.transform(imp.transform(q)),imp,sc

def center_kernel(K,Q):
    mean=K.mean(0);grand=float(mean.mean())
    return K-mean[None,:]-mean[:,None]+grand,Q-Q.mean(1)[:,None]-mean[None,:]+grand

def solve_grid(K,Q,y,alphas):
    K,Q=center_kernel(K,Q);w,v=eigh((K+K.T)/2,check_finite=False)
    w=np.maximum(w,0);coef=v.T@(y-y.mean());Qv=Q@v
    return np.stack([Qv@(coef/(w+a))+y.mean() for a in alphas])

H2A=[.001,.01,.1,1.,10.];H2S=[0.,.1,1.]

def h2_grid(x,q,signature,qsignature,y,kind):
    Z,Q,_,_=standardize(x,q);K=Z@Z.T/Z.shape[1];V=Q@Z.T/Z.shape[1]
    if kind=='linear': return solve_grid(K,V,y,H2A)
    n=4 if kind=='marginal' else 20
    s=signature[:,:n].reshape(len(x),-1);t=qsignature[:,:n].reshape(len(q),-1)
    D=cdist(s,s,'sqeuclidean')/s.shape[1];E=cdist(t,s,'sqeuclidean')/s.shape[1]
    bw=max(float(np.median(D[np.triu_indices(len(D),1)])),1e-8)
    return np.concatenate([solve_grid(K+a*np.exp(-D/(2*bw)),V+a*np.exp(-E/(2*bw)),y,H2A) for a in H2S])

def select_h2(x,s,y,q,qs,seed,kind):
    losses=[]
    for a,b in KFold(3,shuffle=True,random_state=seed).split(x):
        p=h2_grid(x[a],x[b],s[a],s[b],y[a],kind)
        losses.append(np.sum((p-y[b])**2,axis=1))
    loss=np.sum(losses,axis=0);j=int(np.argmin(loss));pred=h2_grid(x,q,s,qs,y,kind)[j]
    return pred,dict(index=j,alpha=H2A[j%5],extra_scale=0 if kind=='linear' else H2S[j//5],inner_mse=(loss/len(x)).tolist())

def gmm_columns(d,ix,noisy):
    cols=np.ones(len(d['keys']),bool) if noisy else d['keep']
    return np.c_[d['X'][ix][:,cols],d['gmm'][ix]]

def run_h2():
    d,y=load_data();m=np.load(_repo_path(HERE, 'measurements.npz'));assert np.array_equal(m['ids'],d['ids'])
    sig=m['signature'];base=d['base'];dev=d['dev'];fold=base['fold'];result={k:np.full(400,np.nan) for k in ['linear','marginal','joint']};records=[]
    start=time.monotonic()
    for k in range(5):
        tr=dev[fold!=k]
        for noisy in [False,True]:
            j=np.flatnonzero((fold==k)&((d['noise'][dev]>=12)==noisy));q=dev[j]
            if not len(q):continue
            for kind in result:
                p,meta=select_h2(gmm_columns(d,tr,noisy),sig[tr],y[tr].copy(),gmm_columns(d,q,noisy),sig[q],2026101635+100*k,kind)
                result[kind][j]=p;records.append(dict(fold=k,noisy=noisy,kind=kind,train_ids=d['ids'][tr].tolist(),query_ids=d['ids'][q].tolist(),**meta))
        print('H2 fold',k,'seconds',round(time.monotonic()-start,1),flush=True)
    assert all(np.isfinite(p).all() for p in result.values())
    arrays('h2-oof.npz',ids=d['ids'][dev],fold=fold,**result)
    save('h2-selection.json',dict(records=records,code_sha256=digest(__file__),plan_sha256=digest(_repo_path(HERE, 'PLAN.md')),seconds=time.monotonic()-start))

H3A=[.01,.1,1.,10.];H3G=[.03,.3,3.]

def training_rows(d,tr,labels):
    keep=np.isin(d['src'],tr);pos={int(i):j for j,i in enumerate(tr)}
    src=np.r_[tr,d['src'][keep]];X=np.r_[d['X'][tr],d['sx'][keep]]
    yy=np.r_[labels,[labels[pos[int(i)]] for i in d['src'][keep]]]
    assert set(src)<=set(tr)
    return X,yy,src

def h3_grid(X,y,src,realtr,q,kind,rotation):
    Z,Q,imp,sc=standardize(X,q)
    index={int(i):j for j,i in enumerate(realtr)};n=len(realtr)
    assert np.array_equal(src[:n],realtr)
    D=Z[n:]-Z[[index[int(i)] for i in src[n:]]]
    if kind=='identity':T=np.eye(Z.shape[1])
    else:
        # Each original source contributes equally to the nuisance second moment.
        ids,ct=np.unique(src[n:],return_counts=True);count=dict(zip(ids,ct));wt=np.array([1/count[i] for i in src[n:]])
        C=(D*wt[:,None]).T@D/max(wt.sum(),1)
        if kind=='rotated': C=rotation@C@rotation.T
        w,v=eigh(C,check_finite=False);T=(v*(1/np.sqrt(1+np.maximum(w,0))))@v.T
    Z=Z@T;Q=Q@T;DD=cdist(Z,Z,'sqeuclidean');EE=cdist(Q,Z,'sqeuclidean')
    predictions=np.concatenate([solve_grid(np.exp(-g*DD/Z.shape[1]),np.exp(-g*EE/Z.shape[1]),y,H3A) for g in H3G])
    return predictions

def select_h3(d,tr,labels,q,seed,kind,rotation):
    loss=np.zeros(12);count=0
    for a,b in KFold(3,shuffle=True,random_state=seed).split(tr):
        train=tr[a];val=tr[b];val=val[d['noise'][val]>=12]
        X,yy,src=training_rows(d,train,labels[a])
        assert set(src).isdisjoint(val)
        p=h3_grid(X,yy,src,train,d['X'][val],kind,rotation)
        pos={int(i):j for j,i in enumerate(tr)};yv=np.array([labels[pos[int(i)]] for i in val])
        loss+=np.sum((p-yv)**2,axis=1);count+=len(val)
    j=int(np.argmin(loss));X,yy,src=training_rows(d,tr,labels)
    p=h3_grid(X,yy,src,tr,d['X'][q],kind,rotation)[j]
    return p,dict(index=j,alpha=H3A[j%4],gamma=H3G[j//4],inner_mse=(loss/count).tolist(),n_rows=len(yy),n_sources=len(np.unique(src)))

def run_h3():
    d,y=load_data();dev=d['dev'];base=d['base'];fold=base['fold'];out={n:base['ridge076'].copy() for n in ['identity','invariant','rotated']}
    rotation=np.linalg.qr(np.random.default_rng(2026101602).normal(size=(98,98)))[0];records=[];start=time.monotonic()
    for k in range(5):
        tr=dev[fold!=k];j=np.flatnonzero((fold==k)&(d['noise'][dev]>=12));q=dev[j]
        for kind in out:
            p,meta=select_h3(d,tr,y[tr].copy(),q,2026101635+100*k,kind,rotation);out[kind][j]=p
            records.append(dict(fold=k,kind=kind,train_ids=d['ids'][tr].tolist(),query_ids=d['ids'][q].tolist(),**meta))
        print('H3 fold',k,'seconds',round(time.monotonic()-start,1),flush=True)
    arrays('h3-oof.npz',ids=d['ids'][dev],fold=fold,**out)
    save('h3-selection.json',dict(records=records,seconds=time.monotonic()-start,code_sha256=digest(__file__),plan_sha256=digest(_repo_path(HERE, 'PLAN.md'))))

def run_h1():
    record=json.loads((_repo_path(HERE, 'measurement-screen.json')).read_text())
    if not record['h1_advance']:
        save('h1-stopped.json',dict(reason='Predeclared physical calibration gate failed; no hardness model selected.',screen='measurement-screen.json'));return
    d,y=load_data();m=np.load(_repo_path(HERE, 'measurements.npz'));dev=d['dev'];fold=d['base']['fold'];keys=list(d['keys'])
    ix=[keys.index(k) for k in ['u_ph','u_ph_hard','u_ph_f32']]
    out={n:np.zeros(400) for n in ['raw','corrected']};records=[];alphas=[.1,1,10,100,1000]
    for k in range(5):
        tr=dev[fold!=k]
        for noisy in [False,True]:
            jj=np.flatnonzero((fold==k)&((d['noise'][dev]>=12)==noisy));q=dev[jj]
            cols=np.ones(98,bool) if noisy else d['keep']
            for kind in out:
                x=d['X'].copy();x[:,ix]=m[kind];x=np.c_[x[:,cols],d['gmm']];loss=np.zeros(5)
                for a,b in KFold(3,shuffle=True,random_state=2026101635+100*k).split(tr):
                    Z,Q,_,_=standardize(x[tr[a]],x[tr[b]])
                    for ai,alpha in enumerate(alphas):loss[ai]+=np.sum((Ridge(alpha=alpha).fit(Z,y[tr[a]]).predict(Q)-y[tr[b]])**2)
                j=int(np.argmin(loss));Z,Q,imp,sc=standardize(x[tr],x[q]);model=Ridge(alpha=alphas[j]).fit(Z,y[tr])
                out[kind][jj]=model.predict(Q)
                path=_repo_path(HERE, f'h1-fold{k}-{int(noisy)}-{kind}.joblib');joblib.dump(dict(imputer=imp,scaler=sc,model=model,columns=cols),_repo_io(path))
                assert np.array_equal(joblib.load(_repo_io(path))['model'].predict(Q),out[kind][jj])
                records.append(dict(fold=k,noisy=noisy,kind=kind,alpha=alphas[j],inner_mse=(loss/len(tr)).tolist(),train_ids=d['ids'][tr].tolist(),query_ids=d['ids'][q].tolist()))
        print('H1 fold',k,flush=True)
    arrays('h1-oof.npz',ids=d['ids'][dev],fold=fold,**out);save('h1-selection.json',records)

def selfcheck():
    # Kernel centering and eigensolve agree with an independent augmented
    # linear system including an unpenalized intercept.
    rng=np.random.default_rng(12);z=rng.normal(size=(20,5));q=rng.normal(size=(7,5));y=rng.normal(size=20)
    K=z@z.T;Q=q@z.T;alpha=.3;actual=solve_grid(K,Q,y,[alpha])[0]
    expected=Ridge(alpha=alpha).fit(z,y).predict(q)
    assert np.max(np.abs(actual-expected))<1e-10
    p=np.array([.6,.2,.03,.17]);w=np.array([.5,1,4,.8]);soft=w*p/(w@p);recovered=soft/w;recovered/=recovered.sum()
    assert np.max(abs(recovered-p))<1e-15
    save('selfcheck.json',dict(kernel_max_error=float(np.max(abs(actual-expected))),weighted_probability_inversion=True,code_sha256=digest(__file__)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['selfcheck','extract','h1','h2','h3']);args=parser.parse_args()
    sys.addaudithook(guard)
    with threadpool_limits(limits=1):
        {'selfcheck':selfcheck,'extract':extract,'h1':run_h1,'h2':run_h2,'h3':run_h3}[args.action]()
