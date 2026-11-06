"""Fresh all-500 matched CPU benchmark and fully nested ensemble-aware selection.

Uses no CNN, held-out label, test input or cached full500 supervised prediction.
Resume is per completed fold; caches contain explicit train/validation IDs.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('TABPFN_DISABLE_TELEMETRY','1')
os.environ.setdefault('MPLCONFIGDIR',str(_repo_path(Path(__file__).resolve().parent, 'mpl-cache')))
os.environ.setdefault('HF_HUB_OFFLINE','1')
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from models import HERE,inputs,groups,EvidenceRidge
from cumulants import select
import research as r

sys.path.insert(0,str(r.EXP/'push-300'))
import harness
SEEDS=(2026101501,2026101502,2026101503)
VARIANTS=('u1dm1szH2Z','u1dm2szH2Z','u1dmavgH2Z','u1dmavg4H2Z')
W=np.array([.384,.128,.128,.15])/.79
GRID=np.array([0.,.25,.5,1.])


def load():
    d=inputs();ids=d['ids'];pos={i:j for j,i in enumerate(ids)}
    d['variants']={}
    for v in VARIANTS:
        f=np.load(_repo_io(r.EXP/f'push-800/feat-{v}-anghcor-train.npz'),allow_pickle=True)
        fids=list(f['ids'].astype(str));x=f['X'].astype(float)[[fids.index(i)for i in ids]]
        xx=[];source=[]
        for s in ('deg','deg2','deg3','deg4'):
            a=np.load(_repo_io(r.EXP/f'push-1200/synthrows-{s}-{v}-anghcor.npz'),allow_pickle=True)
            ni=list(a['keys']).index('noise');m=a['X'][:,ni].astype(float)*255>=12
            xx.extend(a['X'].astype(float)[m]);source.extend(pos[str(i)]for i in a['ids'][m])
        d['variants'][v]=(x,np.array(xx),np.array(source))
    source=r.EXP/'push-900/gmm_partner.py';gp={'__file__':str(source),'__name__':'gmm_reuse'}
    exec(compile(source.read_text().split('mode = sys.argv[1]')[0],str(source),'exec'),gp)
    d['gmm_module']=gp
    cache=HERE/'gmm-features500.npz'
    if cache.exists():
        f=np.load(_repo_io(cache));assert np.array_equal(f['ids'],ids);d['gmm_features']=f['X']
    else:
        d['gmm_features']=gp['gmm_feats'](np.load(_repo_io(r.EXP/'probabilistic-grains-900/object-bank-train.npz'),allow_pickle=True),list(ids))
        np.savez_compressed(_repo_io(cache),ids=ids,X=d['gmm_features'])
    c=np.load(_repo_io(HERE/'cumulants-train.npz'));assert np.array_equal(c['ids'],ids);d['cumulants']=c['X']
    d['random']=[np.random.default_rng(r.SEED+j).normal(size=c['X'].shape)for j in range(3)]
    d['failed500']=np.nan_to_num(d['X'][:,list(d['keys']).index('n_obj')],nan=0)<20
    return d


def fit_bundle(d,tr,va,seed,extra=False):
    """Every supervised parameter, including inner GMM selection, sees only tr."""
    assert not set(tr)&set(va)
    y=d['y'];nz=d['noise'];n=len(va)
    pred={k:np.zeros(n)for k in ('tab','ridge','logridge','gmm','group','cumulants','random0','random1','random2')}
    details=[]
    for noisy in (False,True):
        cols=np.ones(len(d['keys']),bool)if noisy else d['keep']
        mask=(nz[va]>=12)==noisy;tv=va[mask]
        if not len(tv):continue
        x,yy,domain,src,isreal,pairs,_=r.training(d,tr,va,noisy,True)
        assert set(src)<=set(tr)
        control=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),RidgeCV(alphas=[.1,1,10,100,1000])).fit(x,yy)
        q=d['X'][tv][:,cols];pred['ridge'][mask]=control.predict(q)
        gg,levels=groups(d['keys'][cols]);group=EvidenceRidge(gg,max_iter=2000).fit(x,yy);pred['group'][mask]=group.predict(q)
        logs=[]
        for variant in VARIANTS:
            vx,sx,source=d['variants'][variant]
            if noisy:
                use=np.isin(source,tr);xt=np.vstack([vx[tr],sx[use]])[:,cols];yt=np.r_[y[tr],y[source[use]]]
            else:xt=vx[tr][:,cols];yt=y[tr]
            m=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=2.)).fit(xt,np.log(yt))
            logs.append(m.predict(vx[tv][:,cols]))
        pred['logridge'][mask]=np.exp(np.mean(logs,axis=0))
        xg=np.c_[d['X'][:,cols],d['gmm_features']]
        pg,alpha=d['gmm_module']['fit_predict'](xg[tr],y[tr],xg[tv],19001+seed%997)
        pred['gmm'][mask]=pg
        if extra:
            for name,ex in [('cumulants',d['cumulants'])]+[(f'random{j}',v)for j,v in enumerate(d['random'])]:
                (null,candidate),meta=select(xg[tr],ex[tr],y[tr],19001+seed%997)
                assert np.max(np.abs(null.predict(xg[tv],ex[tv])-pg))<1e-8
                pred[name][mask]=candidate.predict(xg[tv],ex[tv])
                details.append(dict(noisy=noisy,kind=name,**meta))
        t=time.monotonic()
        tab=harness.tabpfn_new('v3.5',seed,'cpu',n_est=4)
        tab.fit(d['X'][tr][:,cols],y[tr]);pred['tab'][mask]=tab.predict(d['X'][tv][:,cols])
        details.append(dict(noisy=noisy,n_real=len(tr),n_aug=int((~isreal).sum()),n_sources=len(set(src)),
            ridge_alpha=float(control[-1].alpha_),group_iterations=group.iterations,group_last_change=group.last_change,
            tab_seconds=time.monotonic()-t,tab_seed=seed,tab_n_estimators=4))
        del tab,group,control;gc.collect()
    baseline=W[0]*pred['tab']+W[1]*pred['ridge']+W[2]*pred['logridge']+W[3]*pred['gmm']
    pred['base']=baseline;pred['group_delta']=W[1]*(pred['group']-pred['ridge'])
    # There is no honest CNN prediction for these new folds. A common train-mean
    # fallback is used for the two feature-failure rows, identically in every arm.
    fail=d['failed500'][va]
    pred['base'][fail]=y[tr].mean();pred['group_delta'][fail]=0.
    for key in ('cumulants','random0','random1','random2'):
        if extra:
            p=baseline+W[3]*(pred[key]-pred['gmm']);p[fail]=y[tr].mean();pred[key+'_blend']=p
    return pred,details


def run_partition(rep,nested=True):
    start=time.monotonic();d=load();root=HERE/f'full500-{SEEDS[rep]}';root.mkdir(exist_ok=True)
    folds=np.zeros(500,int)
    for k,(_,va)in enumerate(KFold(5,shuffle=True,random_state=SEEDS[rep]).split(d['X'])):folds[va]=k
    for k in range(5):
        path=root/f'fold{k}.npz'
        if path.exists():continue
        tr,va=np.where(folds!=k)[0],np.where(folds==k)[0]
        # Fit validation predictions before reading/using validation labels.
        pred,details=fit_bundle(d,tr,va,SEEDS[rep]+k,extra=True)
        print('outer',rep,k,'ready elapsed',round(time.monotonic()-start,1),flush=True)
        lambdas=[0.,0.];inner_records=[]
        if nested:
            ibase=np.zeros(len(tr));idelta=np.zeros(len(tr))
            for j,(ia,ib)in enumerate(KFold(3,shuffle=True,random_state=SEEDS[rep]+100+k).split(tr)):
                cache=root/f'fold{k}-inner{j}.npz'
                if cache.exists():
                    cc=np.load(_repo_io(cache));assert np.array_equal(cc['train_ids'],d['ids'][tr[ia]])and np.array_equal(cc['validation_ids'],d['ids'][tr[ib]])
                    ip={key:cc[key]for key in ('base','group_delta')}
                else:
                    ip,meta=fit_bundle(d,tr[ia],tr[ib],SEEDS[rep]+1000+10*k+j)
                    np.savez_compressed(_repo_io(cache),train_ids=d['ids'][tr[ia]],validation_ids=d['ids'][tr[ib]],**ip)
                    r.save_json(cache.with_suffix('.json'),meta)
                ibase[ib]=ip['base'];idelta[ib]=ip['group_delta']
                assert not set(tr[ia])&set(va)and not set(tr[ib])&set(va)
                print('inner',rep,k,j,'ready elapsed',round(time.monotonic()-start,1),flush=True)
            for band in (0,1):
                mask=((d['noise'][tr]>=12)==bool(band))&~d['failed500'][tr]
                loss=[float(np.mean((ibase[mask]+v*idelta[mask]-d['y'][tr[mask]])**2))for v in GRID]
                lambdas[band]=float(GRID[int(np.argmin(loss))]);inner_records.append(dict(band=band,grid=GRID.tolist(),mse=loss,selected=lambdas[band]))
        pred['group_all_blend']=pred['base']+pred['group_delta']
        pred['group_low_blend']=pred['base']+(d['noise'][va]<12)*pred['group_delta']
        pred['nested_blend']=pred['base']+np.where(d['noise'][va]>=12,lambdas[1],lambdas[0])*pred['group_delta']
        np.savez_compressed(_repo_io(path),train_ids=d['ids'][tr],validation_ids=d['ids'][va],validation_index=va,**pred)
        r.save_json(path.with_suffix('.json'),dict(details=details,inner_selection=inner_records,lambdas=lambdas,seed=SEEDS[rep],fold=k,
            base_rmse=r.rmse(pred['base'],d['y'][va]),nested_rmse=r.rmse(pred['nested_blend'],d['y'][va]),
            label_provenance='every source belongs to actual train IDs; inner bases retrained without outer held-out labels'))
        print('fold complete',rep,k,'lambdas',lambdas,'base',round(r.rmse(pred['base'],d['y'][va]),5),'nested',round(r.rmse(pred['nested_blend'],d['y'][va]),5),flush=True)
    summarize()


def summarize():
    d=inputs();arms=['base','group_all_blend','group_low_blend','nested_blend','cumulants_blend','random0_blend','random1_blend','random2_blend']
    arrays={a:[]for a in arms};folds=[];parts=[]
    for seed in SEEDS:
        root=HERE/f'full500-{seed}'
        if not all((root/f'fold{k}.npz').exists()for k in range(5)):continue
        out={a:np.zeros(500)for a in arms};f=np.zeros(500,int)
        for k in range(5):
            z=np.load(_repo_io(root/f'fold{k}.npz'));va=z['validation_index'];f[va]=k
            assert not set(z['train_ids'])&set(z['validation_ids'])
            for a in arms:out[a][va]=z[a]
        for a in arms:arrays[a].append(out[a])
        folds.append(f);parts.append(seed)
    if not parts:return
    arrays={a:np.array(v)for a,v in arrays.items()};folds=np.array(folds)
    result={a:r.compare(arrays['base'],p,d['y'],folds)for a,p in arrays.items()if a!='base'}
    result['seeds']=parts;result['n_unique_ids']=500;result['benchmark_scope']='CPU tabular ensemble; one avg TabPFN feature bank, n_estimators=4, neural partners omitted; weights normalized by .79'
    result['weights']=W.tolist();result['code_sha256']=r.digest(__file__)
    np.savez_compressed(_repo_io(HERE/'full500-summary.npz'),ids=d['ids'],fold=folds,**arrays)
    r.save_json(HERE/'full500-summary.json',result)
    print('completed partitions',parts,flush=True)
    for a in arms[1:]:
        q=result[a];print(a,'rmse',q['candidate']['pooled'],'gain',q['gain'],'CI',q['conditional_id_bootstrap_gain_ci95'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--partition',type=int,choices=[0,1,2]);ap.add_argument('--summary',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        summarize()if a.summary else run_partition(a.partition if a.partition is not None else 0)
