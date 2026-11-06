"""Full073 matched outer evaluation; every new weight is selected inside T."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse
import json
import os
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name,'2')
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import cnn_fitter as c
HERE=Path(__file__).resolve().parent
r=c.r
SEED_SHIFTS=(0,100000)


class StatsOnly(c.ComplementCNN):
    def forward(self,x,context,arm):
        image=torch.zeros((len(x),322),dtype=x.dtype,device=x.device)
        image[:,:2]=torch.cat([x.mean((2,3)),(x.var((2,3),correction=0)+1e-6).sqrt()],1)
        return self.main(torch.cat([image,torch.zeros_like(context)],1)).squeeze(1),self.aux(image).squeeze(1)


def teacher_oof(d,g,kind,train):
    values=np.full(len(d['ids']),np.nan);coverage=np.zeros(len(d['ids']),int);sources=[]
    for node in g['nodes']:
        if not node['name'].startswith(kind):continue
        path=_repo_path(HERE, f"teachers/fold{g['fold']}-{node['name']}/predictions.npz")
        z=np.load(_repo_io(path));tr=z['train'];va=z['query']
        assert set(tr).isdisjoint(va) and set(tr)|set(va)==set(train)
        assert set(tr).isdisjoint(g['V']) and set(va).isdisjoint(g['V'])
        assert np.array_equal(tr,node['train']) and np.array_equal(va,node['query'])
        values[va]=z['base'];coverage[va]+=1
        sources.append(dict(train_ids=d['ids'][tr].tolist(),validation_ids=d['ids'][va].tolist(),
                            file=str(path.relative_to(HERE)),sha256=r.digest(path)))
    assert np.all(coverage[train]==1) and np.isfinite(values[train]).all()
    return values,sources


def raw_context(d,g,kind,teachers,query,query_base,query_gap):
    t=teachers.copy();t[query]=query_base;gap=np.full(len(t),np.nan)
    for node in g['nodes']:
        if node['name'].startswith(kind):
            z=np.load(_repo_path(HERE, f"teachers/fold{g['fold']}-{node['name']}/predictions.npz"))
            gap[z['query']]=z['tab']-z['ridge']
    gap[query]=query_gap;keys=list(d['keys'])
    return np.c_[t,gap,d['noise'],d['X'][:,keys.index('u_ph')],
                 np.log1p(np.maximum(np.nan_to_num(d['X'][:,keys.index('n_obj')],nan=0),0)),
                 d['X'][:,keys.index('coverage')]]


def cnn_data(d,train,query,labels,teachers,sources,fold,stage,context_raw=None):
    """labels is a copied train-only vector; no query label is accepted."""
    train=np.asarray(train,int);query=np.asarray(query,int);labels=np.asarray(labels,float)
    assert len(train)==len(labels) and set(train).isdisjoint(query)
    assert np.isfinite(teachers[train]).all()
    xm=float(d['images'][train].mean());xs=float(d['images'][train].std())
    ym=float(labels.mean());ys=float(labels.std())
    assert xs>0 and ys>0
    images=((d['images']-xm)/xs).astype(np.float32)
    targets=dict(direct=((labels-ym)/ys).astype(np.float32),
                 residual=((labels-teachers[train])/ys).astype(np.float32))
    context=np.zeros((len(images),6),np.float32)
    meta=dict(fold=fold,stage=stage,train_ids=d['ids'][train].tolist(),validation_ids=d['ids'][query].tolist(),
              image_mean=xm,image_std=xs,target_mean=ym,target_std=ys,context_is_zero=context_raw is None,
              inner_sources=sources,training_objective='y minus full073 non021 branch (normalized by .84)')
    if context_raw is not None:
        med=np.nanmedian(context_raw[train],axis=0);raw=np.where(np.isfinite(context_raw),context_raw,med)
        cm=raw[train].mean(0);cs=np.maximum(raw[train].std(0),1e-6)
        context=((raw-cm)/cs).astype(np.float32)
        meta.update(context_mean=cm.tolist(),context_std=cs.tolist(),context_median=med.tolist())
    return dict(images=images,context=context,targets=targets,
                train=train,valid=query,base=teachers,meta=meta)


def coefficient(base,residual,labels,eligible):
    b,p,y=np.asarray(base)[eligible],np.asarray(residual)[eligible],np.asarray(labels)[eligible]
    denominator=float(p@p)
    unconstrained=float(p@(y-b)/denominator) if denominator>1e-12 else 0.
    value=float(np.clip(unconstrained,0.,1.))
    return value,dict(selected=value,unconstrained=unconstrained,n=int(eligible.sum()),
                       correction_rms=float(np.sqrt(np.mean(p*p))),
                       base_rmse=r.rmse(b,y),selected_rmse=r.rmse(b+value*p,y))


def fitted(data,seed,path,arm='residual'):
    if (path/'training.json').exists():
        meta=json.loads((path/'training.json').read_text())
        assert meta['seed']==seed and meta['meta']==data['meta'] and meta['arm']==arm
        z=np.load(_repo_io(path/'predictions.npz'))
        assert np.array_equal(z['train_index'],data['train']) and np.array_equal(z['validation_index'],data['valid'])
        return z['validation_prediction']
    return c.fit_one(data,arm,seed,path)


def run(repeat=0,stats=False,aux=False,context_only=False,fold_filter=None):
    d=c.load_data();labels=d.pop('y');gg=json.loads((_repo_path(HERE, 'label-graph.json')).read_text())
    baseline=np.load(_repo_path(HERE.parent, 'scope.npz'))['strict_blend'][0]
    legacy=d['C73_cnn'][0];non021=(baseline-.16*legacy)/.84
    assert len(baseline)==400 and np.array_equal(baseline[d['fail'][0]],legacy[d['fail'][0]])
    tag=('stats' if stats else 'aux' if aux else 'context' if context_only else 'cnn')+f'-seed{repeat}'
    arm='aux' if aux else 'context_only' if context_only else 'residual'
    conditioned=aux or context_only
    strict_ridge=np.load(_repo_path(HERE.parent, 'consistency-strict.npz'))['production'][0]
    folder=_repo_path(HERE, tag);folder.mkdir(exist_ok=True)
    lock={str(p.relative_to(r.ROOT)):r.digest(p) for p in
          (Path(__file__),_repo_path(HERE, 'cnn_fitter.py'),_repo_path(HERE, 'label-graph.json'),_repo_path(HERE, 'PLAN.md'),
           _repo_path(HERE.parent, 'inputs.npz'),_repo_path(HERE.parent, 'scope.npz'),_repo_path(HERE, 'anchors/verification.json'),
           _repo_path(HERE.parent, 'consistency-strict.npz'),_repo_path(HERE, 'AMENDMENT-01.md'))}
    lock.update({str(p.relative_to(r.ROOT)):r.digest(p) for p in sorted((_repo_path(HERE, 'anchors')).glob('fold*.npz'))})
    path=folder/'provenance.json'
    if path.exists():assert json.loads(path.read_text())==lock
    else:r.save_json(path,lock)
    if stats:c.ComplementCNN=StatsOnly
    for g in gg:
        if fold_filter is not None and g['fold']!=fold_filter:continue
        k=g['fold'];A,B,T,V=[np.asarray(g[key],int) for key in ('A','B','T','V')]
        ib=np.load(_repo_path(HERE, f'teachers/fold{k}-outer0/predictions.npz'))
        selected_base,sources=teacher_oof(d,g,'deep',A)
        raw=raw_context(d,g,'deep',selected_base,B,ib['base'],ib['tab']-ib['ridge']) if conditioned else None
        selector=cnn_data(d,A,B,labels[A].copy(),selected_base,sources,k,'weight_selector',raw)
        seed=2026101501+100*k+SEED_SHIFTS[repeat]
        rb=fitted(selector,seed,folder/f'fold{k}-selector',arm)
        anchor=np.load(_repo_path(HERE, f'anchors/fold{k}.npz'))
        assert np.array_equal(ib['query'],B) and np.array_equal(anchor['B'],B)
        tb=ib['base'];gb=anchor['prediction_B'];pb=.16*gb+.84*tb
        eligible=~d['fail'][0][np.array([list(d['dev']).index(i) for i in B])]
        lam_add,add_selection=coefficient(pb,rb,labels[B].copy(),eligible)
        lam_replace,replace_selection=coefficient(tb,rb,labels[B].copy(),eligible)
        selection=dict(add=add_selection,replace=replace_selection,weight_validation_ids=d['ids'][B].tolist(),
                        selector_training_ids=d['ids'][A].tolist(),outer_validation_ids=d['ids'][V].tolist())
        r.save_json(folder/f'fold{k}-selection.json',selection)
        np.savez_compressed(_repo_io(folder/f'fold{k}-selection.npz'),B=B,base073=pb,non021=tb,residual=rb,eligible=eligible)
        # Lambdas are now frozen. The final refit may use all T, including B.
        outer_base,outer_sources=teacher_oof(d,g,'outer',T)
        ix=np.array([list(d['dev']).index(i) for i in V]);failed=d['fail'][0,ix]
        raw=raw_context(d,g,'outer',outer_base,V,non021[ix],d['C73_tab'][0,ix]-strict_ridge[ix]) if conditioned else None
        final=cnn_data(d,T,V,labels[T].copy(),outer_base,outer_sources,k,'outer_refit',raw)
        rv=fitted(final,seed,folder/f'fold{k}-final',arm)
        p=baseline[ix];t=non021[ix];rv[failed]=0.;t[failed]=p[failed]
        np.savez_compressed(_repo_io(folder/f'fold{k}-outer.npz'),V=V,dev_index=ix,ids=d['ids'][V],
                            base=p,non021=t,residual=rv,addition=p+lam_add*rv,
                            replacement=t+lam_replace*rv,lambda_add=lam_add,lambda_replace=lam_replace)
        print('CNN FOLD COMPLETE',tag,k,'inner weights',lam_add,lam_replace,flush=True)
    if all((folder/f'fold{k}-outer.npz').exists() for k in range(5)):summarize(folder,d,labels)


def summarize(folder,d=None,labels=None):
    if d is None:d=dict(np.load(_repo_path(HERE.parent, 'inputs.npz')));labels=d.pop('y')
    y=labels[d['dev']];fold=d['fold'][:1];names=('base','non021','addition','replacement')
    out={key:np.full(400,np.nan) for key in names};out['residual']=np.full(400,np.nan)
    selections=[]
    for k in range(5):
        z=np.load(_repo_io(folder/f'fold{k}-outer.npz'));ix=z['dev_index']
        assert set(ix)==set(np.flatnonzero(fold[0]==k))
        for key in out:out[key][ix]=z[key]
        selections.append(dict(fold=k,addition=float(z['lambda_add']),replacement=float(z['lambda_replace'])))
    assert all(np.isfinite(p).all() for p in out.values())
    result={key:r.compare(out['base'][None],out[key][None],y,fold) for key in names[1:]}
    result['replacement_vs_deletion']=r.compare(out['non021'][None],out['replacement'][None],y,fold)
    result['inner_weights']=selections
    result['scope']='Full073 original five folds; complete021 inner anchor; all non021 components; source-restricted DTA'
    result['weight_selection']='A213 CNN, B107 constrained least squares; then refit CNN on T320; V80 never selects'
    r.save_json(folder/'summary.json',result)
    np.savez_compressed(_repo_io(folder/'oof.npz'),ids=d['ids'][d['dev']],y=y,fold=fold[0],**out)
    for name in ('addition','replacement','replacement_vs_deletion'):
        q=result[name];print(name,'rmse',q['candidate']['pooled'],'gain',q['gain'],
                             'improved_folds',q['improved_folds'],'CI',q['conditional_id_bootstrap_gain_ci95'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repeat',type=int,choices=(0,1),default=0)
    ap.add_argument('--stats',action='store_true');ap.add_argument('--summary',type=Path)
    ap.add_argument('--aux',action='store_true')
    ap.add_argument('--context-only',action='store_true')
    ap.add_argument('--fold',type=int,choices=range(5),help='Run one fold for resumable execution; no partial summary')
    args=ap.parse_args();torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.summary:summarize(args.summary)
        else:
            assert sum((args.stats,args.aux,args.context_only))<=1
            run(args.repeat,args.stats,args.aux,args.context_only,args.fold)
