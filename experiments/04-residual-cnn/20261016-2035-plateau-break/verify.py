"""Independent checks and portable saved models for the completed screens."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import study as s
import math,json,sys
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist

def model_bundle(X,labels,kind,alpha,signature=None,scale=0.,sources=None,realtr=None):
    Z,_,imp,sc=s.standardize(X,X[:1]);obj=dict(kind=kind,alpha=alpha,imputer=imp,scaler=sc,y_mean=float(labels.mean()))
    if kind=='joint':
        ss=signature.reshape(len(X),-1);D=cdist(ss,ss,'sqeuclidean')/ss.shape[1]
        bw=max(float(np.median(D[np.triu_indices(len(D),1)])),1e-8)
        K=Z@Z.T/Z.shape[1]+scale*np.exp(-D/(2*bw));obj.update(signature=ss,bw=bw,scale=scale)
    else:
        n=len(realtr);pos={int(i):j for j,i in enumerate(realtr)};D=Z[n:]-Z[[pos[int(i)] for i in sources[n:]]]
        ids,ct=np.unique(sources[n:],return_counts=True);count=dict(zip(ids,ct));wt=np.array([1/count[i] for i in sources[n:]])
        C=(D*wt[:,None]).T@D/wt.sum();v,V=eigh(C,check_finite=False);T=(V*(1/np.sqrt(1+np.maximum(v,0))))@V.T
        Z=Z@T;K=np.exp(-scale*cdist(Z,Z,'sqeuclidean')/Z.shape[1]);obj.update(T=T,gamma=scale,covariance=C)
    mean=K.mean(0);grand=float(mean.mean());KC=K-mean[:,None]-mean[None,:]+grand
    w,v=eigh((KC+KC.T)/2,check_finite=False);coef=v@((v.T@(labels-labels.mean()))/(np.maximum(w,0)+alpha))
    obj.update(Z=Z,mean_kernel=mean,grand=grand,coef=coef,training_label_sha256=s.hashlib.sha256(labels.tobytes()).hexdigest())
    return obj

def predict(obj,Q,signature=None):
    z=obj['scaler'].transform(obj['imputer'].transform(Q));Z=obj['Z']
    if obj['kind']=='joint':
        sq=signature.reshape(len(Q),-1);D=cdist(sq,obj['signature'],'sqeuclidean')/sq.shape[1]
        K=z@Z.T/Z.shape[1]+obj['scale']*np.exp(-D/(2*obj['bw']))
    else:K=np.exp(-obj['gamma']*cdist(z@obj['T'],Z,'sqeuclidean')/Z.shape[1])
    centered=K-K.mean(1)[:,None]-obj['mean_kernel'][None,:]+obj['grand']
    return centered@obj['coef']+obj['y_mean']

def main():
    lock=json.loads((s.HERE/'pre-score-lock.json').read_text())['sha256']
    assert all(s.digest(s.HERE/p)==h for p,h in lock.items())
    d,y=s.load_data();dev=d['dev'];fold=d['base']['fold'];sig=np.load(_repo_io(s.HERE/'measurements.npz'))['signature'];records=[]
    h2=np.load(_repo_io(s.HERE/'h2-oof.npz'));h3=np.load(_repo_io(s.HERE/'h3-oof.npz'))
    h2meta=json.loads((s.HERE/'h2-selection.json').read_text())['records'];h3meta=json.loads((s.HERE/'h3-selection.json').read_text())['records']
    folder=s.HERE/'models';folder.mkdir(exist_ok=True)
    for k in range(5):
        tr=dev[fold!=k]
        for noisy in [False,True]:
            ix=np.flatnonzero((fold==k)&((d['noise'][dev]>=12)==noisy));q=dev[ix]
            meta=next(r for r in h2meta if r['fold']==k and r['noisy']==noisy and r['kind']=='joint')
            obj=model_bundle(s.gmm_columns(d,tr,noisy),y[tr].copy(),'joint',meta['alpha'],sig[tr],meta['extra_scale']);obj.update(train_ids=d['ids'][tr],query_ids=d['ids'][q])
            path=folder/f'h2-joint-fold{k}-gate{int(noisy)}.joblib';assert not path.exists();s.joblib.dump(obj,_repo_io(path))
            p=predict(s.joblib.load(_repo_io(path)),s.gmm_columns(d,q,noisy),sig[q]);error=float(np.max(abs(p-h2['joint'][ix])));assert error<1e-8,error
            records.append(dict(fold=k,kind='h2',gate=int(noisy),model=str(path.relative_to(s.HERE)),reload_max_error=error))
        ix=np.flatnonzero((fold==k)&(d['noise'][dev]>=12));q=dev[ix]
        meta=next(r for r in h3meta if r['fold']==k and r['kind']=='invariant');X,yy,src=s.training_rows(d,tr,y[tr].copy())
        assert set(src).isdisjoint(q) and set(src)<=set(tr)
        obj=model_bundle(X,yy,'invariant',meta['alpha'],scale=meta['gamma'],sources=src,realtr=tr);obj.update(train_ids=d['ids'][tr],query_ids=d['ids'][q],source_ids=d['ids'][src])
        path=folder/f'h3-invariant-fold{k}.joblib';assert not path.exists();s.joblib.dump(obj,_repo_io(path))
        p=predict(s.joblib.load(_repo_io(path)),d['X'][q]);error=float(np.max(abs(p-h3['invariant'][ix])));assert error<1e-8,error
        records.append(dict(fold=k,kind='h3',model=str(path.relative_to(s.HERE)),reload_max_error=error))
    # Actual nested-procedure replay. Training functions receive only the exact
    # training labels; changing every other label cannot alter their result.
    k=0;tr=dev[fold!=k];q=dev[fold==k];poison=y.copy();poison[np.setdiff1d(np.arange(500),tr)]+=10000
    meta=next(r for r in h2meta if r['fold']==0 and not r['noisy'] and r['kind']=='joint')
    q2=q[d['noise'][q]<12];ix2=np.flatnonzero((fold==0)&(d['noise'][dev]<12))
    p,m=s.select_h2(s.gmm_columns(d,tr,False),sig[tr],poison[tr].copy(),s.gmm_columns(d,q2,False),sig[q2],2026101635,'joint')
    assert np.array_equal(p,h2['joint'][ix2]) and m['index']==meta['index']
    q3=q[d['noise'][q]>=12];ix3=np.flatnonzero((fold==0)&(d['noise'][dev]>=12));rotation=np.linalg.qr(np.random.default_rng(2026101602).normal(size=(98,98)))[0]
    # Strip all historical OOF/label-bearing arrays from the data capability.
    slim={key:d[key] for key in ['X','sx','src','noise']}
    p,m=s.select_h3(slim,tr,poison[tr].copy(),q3,2026101635,'invariant',rotation)
    assert np.array_equal(p,h3['invariant'][ix3])
    # Verify every recorded outer source/query and inner source grouping.
    for rec in h2meta+h3meta:
        assert set(rec['train_ids']).isdisjoint(rec['query_ids'])
    for k in range(5):
        tr=dev[fold!=k]
        for a,b in s.KFold(3,shuffle=True,random_state=2026101635+100*k).split(tr):
            _,_,src=s.training_rows(slim,tr[a],y[tr[a]])
            assert set(src)<=set(tr[a]) and set(src).isdisjoint(tr[b])
    pred=np.load(_repo_io(s.HERE/'all-predictions.npz'));scored=json.loads((s.HERE/'results.json').read_text());independent={}
    for name,r in scored.items():
        value=math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(pred[name],pred['y']))/400)
        assert abs(value-r['rmse'])<1e-12;independent[name]=value
    rng=np.random.default_rng(42);a=rng.normal(size=100);w=rng.uniform(size=100);perm=rng.permutation(100);qs=np.linspace(.025,.975,17)
    assert np.array_equal(s.weighted_quantile(a,w,qs),s.weighted_quantile(a[perm],w[perm],qs))
    s.save(_repo_io('verification.json'),dict(all_passed=True,pre_score_lock_unchanged=True,saved_model_replays=records,
        heldout_label_poison_nested_H2_H3_exact=True,all_inner_outer_source_groups_disjoint=True,
        independent_scalar_rmse=independent,weighted_distribution_permutation_invariant=True,
        limits='Audits new conditional heads only; frozen upstream auxiliary representations are not certified as fully nested.'))
    print('VERIFIED',len(records),'saved kernel models, nested label poison, source grouping, independent RMSE',flush=True)

if __name__=='__main__':
    sys.addaudithook(s.guard)
    with s.threadpool_limits(limits=1):main()
