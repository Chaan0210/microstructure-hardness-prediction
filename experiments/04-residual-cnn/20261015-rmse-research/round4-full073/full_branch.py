"""Full073 non021 branch, with a separate training-label capability per fit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import copy
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[4]
EXP = _RepoPath(__file__).resolve().parents[3]
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name,'2')
os.environ.setdefault('TABPFN_DISABLE_TELEMETRY','1')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('MPLCONFIGDIR',str(_repo_path(HERE, 'mpl-cache')))
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(_repo_path(EXP, 'push-300')))
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import harness
import research as r

VARIANTS = ('u1dm1szH2Z','u1dm2szH2Z','u1dmavgH2Z','u1dmavg4H2Z')
SLOTS = {'038':('push-500',''), '039':('push-700','-angh'), '044':('push-800','-anghcor')}
SET_SEEDS = (731503,731509,731527)
WEIGHTS = {'tab':.384,'ridge':.128,'logridge':.128,'set':.05,'gmm':.15}


def load():
    raw=dict(np.load(_repo_path(HERE.parent, 'inputs.npz')))
    ids=raw['ids'];pos={str(i):j for j,i in enumerate(ids)}
    d={k:raw[k] for k in ('ids','X','keys','keep','noise','dev','fold','fail')}
    labels=raw['y'].copy()
    d['failed']=np.nan_to_num(d['X'][:,list(d['keys']).index('n_obj')],nan=0)<20
    assert np.array_equal(d['failed'][d['dev']],d['fail'][0])
    groups=json.loads((_repo_path(EXP, 'push-1300/abl/groups.json')).read_text())
    drop=set(groups['het'].split()+groups['intensity'].split())
    inputs=[_repo_path(HERE.parent, 'inputs.npz'),_repo_path(EXP, 'push-1300/abl/groups.json')]
    d['banks']={};d['variants']={}
    for slot,(folder,suffix) in SLOTS.items():
        for v in VARIANTS:
            path=_repo_path(EXP, f'{folder}/feat-{v}{suffix}-train.npz');inputs.append(path)
            z=np.load(_repo_io(path),allow_pickle=True);order={str(i):j for j,i in enumerate(z['ids'])}
            x=z['X'].astype(float)[[order[str(i)] for i in ids]]
            cols=np.array([str(k) not in drop for k in z['keys']])
            d['banks'][slot,v]=(x,cols)
    for v in VARIANTS:
        x,cols=d['banks']['044',v]
        assert np.array_equal(cols,d['keep'])
        xx=[];source=[]
        for suffix in ('deg','deg2','deg3','deg4'):
            path=_repo_path(EXP, f'push-1200/synthrows-{suffix}-{v}-anghcor.npz');inputs.append(path)
            z=np.load(_repo_io(path),allow_pickle=True);mask=z['X'][:,list(z['keys']).index('noise')].astype(float)*255>=12
            xx.extend(z['X'].astype(float)[mask]);source.extend(pos[str(i)] for i in z['ids'][mask])
        d['variants'][v]=(x,np.array(xx),np.array(source))
    path=_repo_path(HERE.parent, 'round2/gmm-features500.npz');inputs.append(path)
    z=np.load(_repo_io(path));assert np.array_equal(z['ids'],ids);d['gmm_X']=z['X']
    source=_repo_path(EXP, 'push-900/gmm_partner.py');inputs.append(source)
    gp={'__file__':str(source),'__name__':'gmm_reuse'}
    exec(compile(source.read_text().split('mode = sys.argv[1]')[0],str(source),'exec'),gp)
    d['gmm']=gp
    path=_repo_path(EXP, 'probabilistic-grains-900/set_model/run.py');inputs.append(path)
    spec=importlib.util.spec_from_file_location('round4_grainset',path)
    sm=importlib.util.module_from_spec(spec);spec.loader.exec_module(sm)
    bankpath=_repo_path(EXP, 'probabilistic-grains-900/object-bank-train.npz')
    basepath=_repo_path(EXP, 'probabilistic-grains-900/base.npz');inputs.extend([bankpath,basepath])
    bank,_,_=sm.load_inputs(bankpath,basepath)
    bpos={str(i):j for j,i in enumerate(bank['ids'])}
    d['bank_index']=np.array([bpos[str(i)] for i in ids]);bank.pop('y')
    d['set_module']=sm;d['set_config']=copy.deepcopy(sm.CONFIG);d['object_bank']=bank
    d['source_hashes']={str(p.relative_to(ROOT)):r.digest(p) for p in inputs}
    return d,labels


def linear_training(d,train,labels,variant,noisy):
    x,sx,source=d['variants'][variant]
    cols=np.ones(len(d['keys']),bool) if noisy else d['keep']
    if not noisy:return x[train][:,cols],labels.copy(),train.copy(),cols
    keep=np.isin(source,train);lookup={int(i):j for j,i in enumerate(train)}
    sy=np.array([labels[lookup[int(i)]] for i in source[keep]])
    src=np.r_[train,source[keep]];assert set(src)<=set(train)
    return np.vstack([x[train],sx[keep]])[:,cols],np.r_[labels,sy],src,cols


def fit(d,train,query,labels,fold,node):
    """No global labels exist in d; labels is exactly len(train)."""
    train=np.asarray(train,int);query=np.asarray(query,int);labels=np.asarray(labels,float)
    assert labels.shape==train.shape and np.isfinite(labels).all()
    assert not set(train)&set(query)
    path=_repo_path(HERE, f'teachers/fold{fold}-{node}');path.mkdir(parents=True,exist_ok=True)
    contract=dict(train_ids=d['ids'][train].tolist(),query_ids=d['ids'][query].tolist(),
                  target_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
                  code_sha256=r.digest(__file__),source_hashes=d['source_hashes'],fold=fold,
                  weights=WEIGHTS,device='cpu',threads=2,tab_seeds=[100*s+fold for s in range(3)],
                  tab_n_estimators='default-auto',set_seeds=list(SET_SEEDS))
    if (path/'contract.json').exists():
        assert json.loads((path/'contract.json').read_text())==contract,'Teacher contract changed'
    else:r.save_json(path/'contract.json',contract)
    if (path/'predictions.npz').exists():
        z=dict(np.load(_repo_io(path/'predictions.npz')))
        assert np.array_equal(z['train'],train) and np.array_equal(z['query'],query)
        return z
    begin=time.monotonic();n=len(query)
    result={k:np.full(n,np.nan) for k in WEIGHTS};details=[]
    for noisy in (False,True):
        mask=(d['noise'][query]>=12)==noisy;qi=query[mask]
        if not len(qi):continue
        x,y,src,cols=linear_training(d,train,labels,VARIANTS[2],noisy)
        model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),
                            RidgeCV(alphas=[.1,1,10,100,1000])).fit(x,y)
        result['ridge'][mask]=model.predict(d['X'][qi][:,cols])
        logs=[]
        for v in VARIANTS:
            xx,yy,ss,cc=linear_training(d,train,labels,v,noisy)
            lm=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=2.)).fit(xx,np.log(yy))
            logs.append(lm.predict(d['variants'][v][0][qi][:,cc]))
        result['logridge'][mask]=np.exp(np.mean(logs,axis=0))
        xg=np.c_[d['X'][:,cols],d['gmm_X']]
        result['gmm'][mask],alpha=d['gmm']['fit_predict'](xg[train],labels,xg[qi],19001)
        details.append(dict(noisy=noisy,n_real=len(train),n_aug=len(src)-len(train),
                            augmentation_source_ids=d['ids'][np.unique(src)].tolist(),
                            ridge_alpha=float(model[-1].alpha_),gmm_alpha=float(alpha)))
        members=[]
        for (slot,v),(xx,nbcols) in d['banks'].items():
            cc=np.ones(xx.shape[1],bool) if noisy else nbcols
            for s in range(3):
                file=path/f'tab-{int(noisy)}-{slot}-{v}-seed{s}.npz'
                if file.exists():
                    z=np.load(_repo_io(file));pp=z['prediction']
                    assert np.array_equal(z['query'],qi)
                else:
                    t=time.monotonic()
                    tm=harness.tabpfn_new('v3.5',100*s+fold,'cpu')
                    tm.fit(xx[train][:,cc],labels);pp=tm.predict(xx[qi][:,cc])
                    assert np.isfinite(pp).all()
                    np.savez_compressed(_repo_io(file),query=qi,prediction=pp,train=train,
                                        seconds=time.monotonic()-t,seed=100*s+fold)
                    del tm;gc.collect()
                members.append(pp)
            print(json.dumps(dict(teacher=node,fold=fold,gate=int(noisy),slot=slot,variant=v,
                                  completed_tab_members=len(members),seconds=round(time.monotonic()-begin,1))),flush=True)
        assert len(members)==36
        result['tab'][mask]=np.mean(members,axis=0)
    sm=d['set_module'];bank=dict(d['object_bank'])
    bank['y']=np.full(len(bank['ids']),np.nan)
    bt=d['bank_index'][train];bq=d['bank_index'][query];bank['y'][bt]=labels
    seed_predictions=[]
    for seed in SET_SEEDS:
        file=path/f'grain-set-{seed}.npz'
        sm.CONFIG={**copy.deepcopy(d['set_config']),'seed':seed}
        if file.exists():
            z=np.load(_repo_io(file));assert np.array_equal(z['query'],query);pp=z['prediction']
        else:
            stopping=sm.train_with_inner_stop(bank,bt,'cpu',seed+fold)
            model,context,data,history=sm.refit(bank,bt,'cpu',seed+fold,stopping['selected_epochs'])
            assert set(context['training_ids'])==set(d['ids'][train])
            for kind in ('inner_train_ids','inner_validation_ids'):
                assert set(stopping[kind])<=set(d['ids'][train])
            pp=sm.predict_y(model,data,context,bq)
            sm.save_checkpoint(path/f'grain-set-{seed}.pt',bank,model,context,
                               dict(seed=seed+fold,training_ids=d['ids'][train].tolist(),query_ids=d['ids'][query].tolist()))
            r.save_json(path/f'grain-set-{seed}.json',dict(stopping=stopping,refit_history=history))
            np.savez_compressed(_repo_io(file),query=query,train=train,prediction=pp,selected_epochs=stopping['selected_epochs'])
            del model,data;gc.collect()
        seed_predictions.append(pp)
        print(json.dumps(dict(teacher=node,fold=fold,set_seed=seed,seconds=round(time.monotonic()-begin,1))),flush=True)
    result['set']=np.mean(seed_predictions,axis=0)
    for p in result.values():assert p.shape==(n,) and np.isfinite(p).all()
    result['base']=sum(WEIGHTS[k]*result[k] for k in WEIGHTS)/.84
    result['base'][d['failed'][query]]=labels.mean()
    result.update(train=train,query=query,train_ids=d['ids'][train],query_ids=d['ids'][query])
    np.savez_compressed(_repo_io(path/'predictions.npz'),**result)
    r.save_json(path/'details.json',dict(linear=details,elapsed_seconds=time.monotonic()-begin,
                                        complete_tab_members=72,set_seeds=list(SET_SEEDS)))
    return result
