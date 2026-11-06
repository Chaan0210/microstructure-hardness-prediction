"""Actual076 nested teachers; reuse only unchanged, identical-scope components."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse
import gc
import hashlib
import json
import signal
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import probe as p
import full_branch as b
HERE=p.HERE


def fit(d,train,query,labels,fold,node,extra):
    source=p.R4/f'teachers/fold{fold}-{node}/predictions.npz'
    prior=np.load(_repo_io(source));old_contract=json.loads(source.with_name('contract.json').read_text())
    assert np.array_equal(prior['train'],train) and np.array_equal(prior['query'],query)
    assert set(train).isdisjoint(query) and len(labels)==len(train)
    target_sha=hashlib.sha256(labels.tobytes()).hexdigest()
    assert old_contract['target_sha256']==target_sha
    path=HERE/f'teachers076/fold{fold}-{node}';path.mkdir(parents=True,exist_ok=True)
    contract=dict(fold=fold,node=node,train_ids=d['ids'][train].tolist(),query_ids=d['ids'][query].tolist(),
        target_sha256=target_sha,old_prediction_sha256=p.r.digest(source),source_sha256=d['source_hashes'],
        cumulant_sha256=p.r.digest(_repo_path(p.ROOT, 'experiments/push-1300/cumul-train.npz')),
        code_sha256=p.r.digest(__file__),plan_sha256=p.r.digest(HERE/'RETARGET-PLAN.md'),
        changed_components=['TabPFN H9 all12banks x3seeds x2regimes','source-grouped high-noise linear alpha'],
        unchanged_components=['low-noise Ridge and logRidge','GMM-Ridge','grain-set3seeds'],
        tab_n_estimators='default-auto',device='cpu',threads=2,weights=b.WEIGHTS)
    if (path/'contract.json').exists():assert json.loads((path/'contract.json').read_text())==contract
    else:p.save(_repo_io(str(path.relative_to(HERE)/'contract.json')),contract)
    if (path/'predictions.npz').exists():return dict(np.load(_repo_io(path/'predictions.npz')))
    out={key:prior[key].copy() for key in b.WEIGHTS};start=time.monotonic();records=[]
    for noisy in (False,True):
        mask=(d['noise'][query]>=12)==noisy;qi=query[mask]
        if not len(qi):continue
        if noisy:
            rp,rm=p.grouped_linear(d,train,qi,labels,b.VARIANTS[2],False);logs=[];details=[]
            for variant in b.VARIANTS:
                lp,lm=p.grouped_linear(d,train,qi,labels,variant,True);logs.append(np.log(lp));details.append(lm)
            out['ridge'][mask]=rp;out['logridge'][mask]=np.exp(np.mean(logs,axis=0))
            records.append(dict(ridge=rm,logridge=details))
        members=[]
        for (slot,variant),(raw,nbcols) in d['banks'].items():
            X=np.c_[raw,extra];cols=np.ones(X.shape[1],bool) if noisy else np.r_[nbcols,True,True]
            for j in range(3):
                file=path/f'tab-{int(noisy)}-{slot}-{variant}-seed{j}.npz'
                if file.exists():
                    z=np.load(_repo_io(file));assert np.array_equal(z['train'],train) and np.array_equal(z['query'],qi)
                    pred=z['prediction']
                else:
                    model=b.harness.tabpfn_new('v3.5',100*j+fold,'cpu')
                    model.fit(X[train][:,cols],labels);pred=model.predict(X[qi][:,cols])
                    assert np.isfinite(pred).all()
                    np.savez_compressed(_repo_io(file),train=train,query=qi,prediction=pred,seed=100*j+fold)
                    del model;gc.collect()
                members.append(pred)
            print('076 TEACHER',fold,node,'gate',int(noisy),'bank',slot,variant,
                  'members',len(members),'seconds',round(time.monotonic()-start,1),flush=True)
        assert len(members)==36;out['tab'][mask]=np.mean(members,axis=0)
    out['base']=sum(b.WEIGHTS[k]*out[k] for k in b.WEIGHTS)/.84
    out['base'][d['failed'][query]]=labels.mean()
    assert all(np.isfinite(out[k]).all() for k in out)
    out.update(train=train,query=query,train_ids=d['ids'][train],query_ids=d['ids'][query])
    np.savez_compressed(_repo_io(path/'predictions.npz'),**out)
    p.save(_repo_io(str(path.relative_to(HERE)/'details.json')),dict(linear=records,seconds=time.monotonic()-start,
        mean_shift_vs073=float(np.mean(out['base']-prior['base'])),rms_shift_vs073=float(np.sqrt(np.mean((out['base']-prior['base'])**2)))))
    return out


def main(folds):
    p.protect();p.sys.addaudithook(p.old.no_test)
    def timeout(*unused):raise TimeoutError('70min pilot teacher budget reached; completed fits retained')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(4200)
    d,y=b.load();c=np.load(_repo_path(p.ROOT, 'experiments/push-1300/cumul-train.npz'))
    cp={str(i):j for j,i in enumerate(c['ids'])};extra=c['X'][[cp[str(i)] for i in d['ids']]][:,[1,4]]
    graphs=json.loads((p.R4/'label-graph.json').read_text());completed=[]
    for k in folds:
        for node in graphs[k]['nodes']:
            tr=np.array(node['train']);query=np.array(node['query'])
            assert set(tr).isdisjoint(graphs[k]['V']) and set(query).isdisjoint(graphs[k]['V'])
            fit(d,tr,query,y[tr].copy(),k,node['name'],extra)
            completed.append(dict(fold=k,node=node['name']))
            p.save(_repo_io('teacher076-progress.json'),dict(folds=folds,completed=completed))
            print('076 TEACHER COMPLETE',k,node['name'],flush=True)
    signal.alarm(0);p.protect()
    p.save(_repo_io('teachers076-complete-'+''.join(map(str,folds))+'.json'),dict(complete=True,folds=folds,teachers=completed))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folds',type=int,nargs='+',default=[0,1]);args=parser.parse_args()
    assert len(set(args.folds))==len(args.folds) and set(args.folds)<=set(range(5))
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main(args.folds)
