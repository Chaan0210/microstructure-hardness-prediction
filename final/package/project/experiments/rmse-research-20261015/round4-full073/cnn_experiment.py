"""Original final-CNN normalization and training-target construction."""
import numpy as np

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

