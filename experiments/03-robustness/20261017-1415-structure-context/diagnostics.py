"""Describe learned field structure and failures; no candidate selection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import encoder as e
import numpy as np,json,sys
def rank(x):
    x=x.astype(float);sv=np.linalg.svd(x-x.mean(0),compute_uv=False);v=sv**2;v/=max(v.sum(),1e-30);return float(np.exp(-(v*np.log(np.maximum(v,1e-30))).sum()))
def main():
    r=json.loads((e.HERE/'results.json').read_text());o=np.load(_repo_io(e.HERE/'oof.npz'));z=np.load(_repo_io(e.OLD/'inputs.npz'));graphs=json.loads((e.OLD/'graphs.json').read_text());pos={int(j):i for i,j in enumerate(z['dev'])};geo={a:np.zeros((400,3)) for a in ['spatial','permuted']};ref={a:np.zeros((400,3)) for a in geo};rows=[]
    for k,g in enumerate(graphs):
        for stage in ['selector','refit']:
            for repeat in range(2):
                for arm in ['spatial','permuted','random']:
                    path=e.HERE/(f'encoders/fold{k}-{stage}-seed{repeat}-'+('spatial' if arm=='random' else arm));f=np.load(_repo_io(path/('random-features.npz' if arm=='random' else 'features.npz')));tr=f['train_features'];scale=tr.std(0);scale=np.where(scale>1e-12,scale,1);ctx=np.load(_repo_io(e.HERE/f'readouts/fold{k}-{stage}/{arm}-seed{repeat}/context.npz'));record=dict(fold=k,stage=stage,repeat=repeat,arm=arm,raw_effective_rank=rank(tr),standardized_effective_rank=rank((tr-tr.mean(0))/scale),PCA16_fraction=float(ctx['pca_explained_variance_ratio'].sum()))
                    if arm!='random':
                        t=json.loads((path/'training.json').read_text());record.update(first_train_loss=t['curve'][0]['loss'],last_train_loss=t['curve'][-1]['loss'])
                        if stage=='refit':
                            ix=np.array([pos[int(j)] for j in f['query']]);factor=(t['normalization']['image_std']*255)**2;geo[arm][ix]+=f['geometry']*factor/2;ref[arm][ix]+=f['geometry_reference']*factor/2
                    rows.append(record)
    geometry={};rng=np.random.default_rng(2026101785)
    for arm in geo:
        geometry[arm]={}
        for band,mask in [('all',np.ones(400,bool)),('high',o['high'])]:
            model=geo[arm][mask];base=ref[arm][mask];draw=rng.integers(0,len(model),(10000,len(model)));delta=model-base;boot=delta[draw].mean(1);geometry[arm][band]=dict(n=int(mask.sum()),units='squared gray levels on0..255; columns DC, centered-field, gradient',model_MSE=model.mean(0).tolist(),reference_MSE=base.mean(0).tolist(),relative_reduction=(1-model.mean(0)/base.mean(0)).tolist(),dMSE_CI95=np.quantile(boot,[.025,.975],axis=0).tolist())
    sse=(o['baseline082']-o['y'])**2-(o['spatial']-o['y'])**2;worst=[]
    for i in np.argsort(sse)[:10]:worst.append(dict(ID=str(o['ids'][i]),fold=int(o['fold'][i]),noise=float(o['noise'][i]),hardness=float(o['y'][i]),baseline082=float(o['baseline082'][i]),spatial=float(o['spatial'][i]),delta=float(o['spatial_delta'][i]),SSE_gain=float(sse[i])))
    e.save(_repo_io('diagnostics.json'),dict(geometry=geometry,representation=rows,worst_spatial_changes=worst,limitations=['A coarse masked-patch conditional mean can improve without reconstructing individual boundaries.','P has a different physical mask footprint; interpret geometry against its own reference and do not claim a pure topology causal effect.','Effective rank/PCA variance do not certify preservation of hardness-relevant information.','These diagnostics never modify the fixed candidate or decision rule.']))
    e.arrays('geometry-per-image.npz',ids=o['ids'],high=o['high'],spatial=geo['spatial'],permuted=geo['permuted'],spatial_reference=ref['spatial'],permuted_reference=ref['permuted'])
    print('GEOMETRY',json.dumps(geometry));print('WORST',worst[:3])
if __name__=='__main__':
    sys.addaudithook(e.guard)
    with e.threadpool_limits(limits=2):main()
