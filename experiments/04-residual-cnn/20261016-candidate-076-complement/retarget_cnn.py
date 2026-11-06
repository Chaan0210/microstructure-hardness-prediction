"""Same CNN, honest076 residual targets; internal-only matched calibration."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse
import json
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import probe as p
import noise_calibration as nc
HERE=p.HERE;e=p.old.e;r=p.r


def data_for(d,labels,graph,stage):
    k=graph['fold'];A,B,T,V=[np.array(graph[n],int) for n in ('A','B','T','V')]
    train,query,kind=(A,B,'deep') if stage=='selector' else (T,V,'outer')
    teacher=np.full(500,np.nan);gap=np.full(500,np.nan);count=np.zeros(500,int);sources=[]
    for node in graph['nodes']:
        if not node['name'].startswith(kind):continue
        path=HERE/f"teachers076/fold{k}-{node['name']}/predictions.npz";z=np.load(_repo_io(path))
        tr,va=z['train'],z['query'];assert set(tr).isdisjoint(va) and set(tr)|set(va)==set(train)
        assert set(tr).isdisjoint(query) and set(va).isdisjoint(query)
        teacher[va]=z['base'];gap[va]=z['tab']-z['ridge'];count[va]+=1
        sources.append(dict(train_ids=d['ids'][tr].tolist(),validation_ids=d['ids'][va].tolist(),
                            file=str(path.relative_to(p.ROOT)),sha256=r.digest(path)))
    assert np.all(count[train]==1)
    if stage=='selector':
        q=np.load(_repo_io(HERE/f'teachers076/fold{k}-outer0/predictions.npz'))
        assert np.array_equal(q['query'],query)
        tq=q['base'];gq=q['tab']-q['ridge']
    else:
        z=np.load(_repo_io(HERE/'baseline076.npz'));ix=np.flatnonzero(z['fold']==k)
        assert np.array_equal(z['dev'][ix],V)
        tq=(z['p76'][ix]-.16*z['G'][ix])/.84;tq[z['failed'][ix]]=z['p76'][ix][z['failed'][ix]]
        gq=z['tab076'][ix]-z['ridge076'][ix]
    teacher[query]=tq;gap[query]=gq;keys=list(d['keys']);X=d['X']
    raw=np.c_[teacher,gap,d['noise'],X[:,keys.index('u_ph')],
              np.log1p(np.maximum(np.nan_to_num(X[:,keys.index('n_obj')],nan=0),0)),X[:,keys.index('coverage')]]
    data=e.cnn_data(d,train,query,labels[train].copy(),teacher,sources,k,stage,raw)
    data['meta']['training_objective']='y minus strict076 non021 branch, honest nested teachers'
    return data


def train(folds):
    p.protect();p.sys.addaudithook(p.old.no_test)
    d=e.c.load_data();labels=d.pop('y');graphs=json.loads((p.R4/'label-graph.json').read_text())
    for k in folds:
        assert all((HERE/f"teachers076/fold{k}-{n['name']}/predictions.npz").exists() for n in graphs[k]['nodes'])
        for stage in ('selector','final'):
            data=data_for(d,labels,graphs[k],stage)
            changed=labels.copy();changed[data['valid']]+=10000
            check=data_for(d,changed,graphs[k],stage)
            assert data['meta']==check['meta']
            for key in ('images','context','train','valid'):
                assert np.array_equal(data[key],check[key]),key
            for key in data['targets']:assert np.array_equal(data['targets'][key],check['targets'][key])
            del check
            for repeat in (0,1):
                seed=2026101501+100*k+100000*repeat
                path=HERE/f'retarget-cnn/seed{repeat}/fold{k}-{stage}';path.parent.mkdir(parents=True,exist_ok=True)
                if not (path/'training.json').exists():e.c.fit_one(data,'aux',seed,path)
                record=json.loads((path/'training.json').read_text());assert record['seed']==seed and record['meta']==data['meta']
                ck=torch.load(_repo_io(path/'model.pt'),map_location='cpu',weights_only=True)
                model=e.c.ComplementCNN();model.load_state_dict(ck['state_dict'])
                pred=e.c.predict(model,data['images'],data['context'],data['valid'],'aux')*data['meta']['target_std']
                ref=np.load(_repo_io(path/'predictions.npz'));assert np.array_equal(pred,ref['validation_prediction'])
                p.save(_repo_io(str(path.relative_to(HERE)/'verification.json')),dict(checkpoint_replay_exact=True,
                    query_label_poison_training_inputs_exact=True,code_sha256=r.digest(__file__),
                    fitter_sha256=r.digest(p.R4/'cnn_fitter.py'),plan_sha256=r.digest(HERE/'RETARGET-PLAN.md')))
                print('RETARGET CNN COMPLETE',k,stage,repeat,flush=True)
    p.protect();p.save(_repo_io('retarget-cnn-complete-'+''.join(map(str,folds))+'.json'),dict(folds=folds,complete=True))


def corrections(k,stage,new):
    ps=[]
    for j in (0,1):
        path=HERE/f'retarget-cnn/seed{j}/fold{k}-{stage}/predictions.npz' if new else p.R4/f'aux-seed{j}/fold{k}-{stage}/predictions.npz'
        z=np.load(_repo_io(path));ps.append(z['validation_prediction'])
        if j==0:query=z['validation_index']
        else:assert np.array_equal(query,z['validation_index'])
    return query,np.mean(ps,axis=0)


def evaluate(folds):
    p.protect();d=dict(np.load(_repo_io(p.R4.parent/'inputs.npz')));z=dict(np.load(_repo_io(HERE/'baseline076.npz')))
    chosen=np.isin(z['fold'],folds);ix=np.flatnonzero(chosen);pos={int(q):j for j,q in enumerate(z['dev'])}
    fixed=json.loads((p.R4/'noise-calibration/summary.json').read_text())['aux']['selection']
    names=('old_fixed','old_matched','retarget_fixed','retarget_matched')
    pred={n:z['p76'].copy() for n in names};records=[]
    for k in folds:
        inner=np.load(_repo_io(HERE/f'teachers076/fold{k}-outer0/predictions.npz'))
        anchor=np.load(_repo_io(p.R4/f'anchors/fold{k}.npz'));B=inner['query']
        assert np.array_equal(B,anchor['B'])
        baseB=.16*anchor['prediction_B']+.84*inner['base']
        eligible=np.nan_to_num(d['X'][B,list(d['keys']).index('n_obj')],nan=0)>=20
        baseB[~eligible]=anchor['prediction_B'][~eligible]
        row=dict(fold=k,calibration_ids=d['ids'][B].tolist(),selection={})
        for new in (False,True):
            qb,rb=corrections(k,'selector',new);V,rv=corrections(k,'final',new)
            assert np.array_equal(qb,B) and set(B).isdisjoint(V)
            weight,meta=nc.select(rb[eligible],(d['y'][B]-baseB)[eligible],d['noise'][B][eligible]>=12)
            altered=d['y'].copy();altered[V]+=10000
            ww,mm=nc.select(rb[eligible],(altered[B]-baseB)[eligible],d['noise'][B][eligible]>=12)
            assert np.array_equal(weight,ww) and meta==mm
            jx=np.array([pos[int(q)] for q in V]);good=(z['noise'][jx]>=12)&~z['failed'][jx]
            prefix='retarget' if new else 'old'
            pred[prefix+'_matched'][jx[good]]+=float(weight[1])*rv[good]
            pred[prefix+'_fixed'][jx[good]]+=float(fixed[k]['addition']['weights'][1])*rv[good]
            row['selection'][prefix]=dict(weights=weight.tolist(),**meta,outer_label_poison_exact=True)
            row['outer_ids']=d['ids'][V].tolist()
        records.append(row)
    reference=np.load(_repo_io(HERE/'probe-oof.npz'))['old4']
    assert np.max(np.abs(pred['old_fixed'][ix]-reference[ix]))<1e-10
    y=d['y'][z['dev']][ix];ff=z['fold'][ix]
    def compare(a,b):
        a,b=a[ix],b[ix];ae=(a-y)**2;be=(b-y)**2
        samples=np.random.default_rng(2026101629).integers(0,len(y),(10000,len(y)))
        gains=np.sqrt(ae[samples].mean(1))-np.sqrt(be[samples].mean(1))
        ordered=np.argsort(ae-be)[::-1];remain=ordered[10:]
        return dict(base_rmse=float(np.sqrt(ae.mean())),candidate_rmse=float(np.sqrt(be.mean())),
            gain=float(np.sqrt(ae.mean())-np.sqrt(be.mean())),ci95=np.quantile(gains,[.025,.975]).tolist(),
            fold_gains={str(k):float(np.sqrt(ae[ff==k].mean())-np.sqrt(be[ff==k].mean())) for k in folds},
            gain_without_largest10=float(np.sqrt(ae[remain].mean())-np.sqrt(be[remain].mean())),
            error_correlation=float(np.corrcoef(a-y,b-y)[0,1]))
    matched=compare(pred['old_matched'],pred['retarget_matched'])
    frozen=compare(pred['old_fixed'],pred['retarget_fixed'])
    comparisons={n:compare(pred['old_fixed'],pr) for n,pr in pred.items()}
    if folds==[0,1]:
        advance=matched['gain']>=.01 and min(matched['fold_gains'].values())>=-.05 and frozen['gain']>0
    else:
        advance=len(folds)==5 and matched['gain']>0 and sum(g>0 for g in matched['fold_gains'].values())>=3 and matched['ci95'][0]>0
    scores={n:dict(rmse=r.rmse(pr[ix],y),folds={str(k):r.rmse(pr[ix][ff==k],y[ff==k]) for k in folds}) for n,pr in pred.items()}
    tag=''.join(map(str,folds))
    p.save(_repo_io('retarget-results-'+tag+'.json'),dict(folds=folds,n_ids=len(ix),scores=scores,matched_training_effect=matched,
        frozen_weight_training_effect=frozen,vs_current=comparisons,selections=records,
        advance=bool(advance),pilot=(len(folds)<5),
        limitation='Preselected folds of previously explored development data. Pilot scores are not full400 OOF scores.'))
    np.savez_compressed(_repo_io(HERE/f'retarget-oof-{tag}.npz'),ids=z['ids'][ix],fold=ff,y=y,**{n:pr[ix] for n,pr in pred.items()})
    print(json.dumps(dict(scores=scores,matched=matched,frozen=frozen,advance=bool(advance)),indent=2),flush=True)
    p.protect()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['train','evaluate'])
    parser.add_argument('--folds',type=int,nargs='+',default=[0,1]);args=parser.parse_args()
    assert sorted(set(args.folds))==args.folds and set(args.folds)<=set(range(5))
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):train(args.folds) if args.mode=='train' else evaluate(args.folds)

