"""Descriptive mechanism, influence and held-out map panels after fixed scoring."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import graph as g
import numpy as np,json,sys
from PIL import Image,ImageDraw
def components(q):
    parent=np.arange(1024);size=np.ones(1024,int)
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for j in np.flatnonzero(q>=.5):
        a,b=root(g.U[j]),root(g.V[j])
        if a==b:continue
        if size[a]<size[b]:a,b=b,a
        parent[b]=a;size[a]+=size[b]
    labels=np.array([root(i) for i in range(1024)]);_,labels=np.unique(labels,return_inverse=True);palette=np.random.default_rng(712).integers(35,230,(int(labels.max())+1,3),dtype=np.uint8);return palette[labels.reshape(32,32)]
def main():
    o=np.load(_repo_io(c.HERE/'oof.npz'));d=c.inputs(labels=True);z=np.load(_repo_io(c.HERE/'targets.npz'));assert np.array_equal(z['dev'],d['dev']) and np.array_equal(z['ids'],o['ids']);results=json.loads((c.HERE/'results.json').read_text());gain=(o['baseline082']-o['y'])**2-(o['topology']-o['y'])**2;worst=np.argsort(gain)[:10];best=np.argsort(-gain)[:10]
    def cases(indices):return [dict(ID=str(o['ids'][j]),fold=int(o['fold'][j]),noise=float(o['noise'][j]),y=float(o['y'][j]),baseline082=float(o['baseline082'][j]),candidate=float(o['topology'][j]),SSE_gain=float(gain[j])) for j in indices]
    raw=json.loads((c.HERE/'target-audit.json').read_text())['sources'];rank=[]
    for k in range(5):
        for stage in ['selector','refit']:
            for arm in ['topology','local','hardness','random']:
                for repeat in [0,1]:
                    src=c.HERE/(f'encoders/fold{k}-{stage}-seed{repeat}-'+('topology' if arm=='random' else arm));f=np.load(_repo_io(src/('random-features.npz' if arm=='random' else 'features.npz')));X=f['train_latent'].astype(float);X-=X.mean(0);sd=X.std(0);X/=np.where(sd>0,sd,1);values=np.linalg.svd(X,compute_uv=False)**2;values=values/values.sum();positive=values>0;er=float(np.exp(-np.sum(values[positive]*np.log(values[positive]))));ctx=np.load(_repo_io(c.HERE/f'readouts/fold{k}-{stage}/{arm}-seed{repeat}/context.npz'));entry=dict(fold=k,stage=stage,arm=arm,seed=repeat,standardized_latent_effective_rank=er,PCA16_fraction=float(ctx['pca_explained_variance_ratio'].sum()))
                    if arm!='random':entry['training_loss_first_last']=[json.loads((src/'training.json').read_text())['curve'][j]['loss'] for j in [0,-1]]
                    rank.append(entry)
    high=o['high'];weak={}
    for arm in ['topology','local','hardness']:
        metric=o[arm+'_weak'];reference=np.full_like(metric,np.log(2));draw=np.random.default_rng(2026101796).integers(0,int(high.sum()),(10000,int(high.sum())));delta=(metric-reference)[high];weak[arm]=dict(high_mean_BCE=metric[high].mean(0).tolist(),high_difference_from_constant_CI95=np.quantile(delta[draw].mean(1),[.025,.975],axis=0).tolist())
    paired=o['topology_weak'][high]-o['local_weak'][high];weak['topology_minus_local']=dict(high_mean=paired.mean(0).tolist(),conditional_CI95=np.quantile(paired[draw].mean(1),[.025,.975],axis=0).tolist())
    fixed=[int(np.flatnonzero(o['ids']==id)[0]) for id in ['TRAIN_000010','TRAIN_000025','TRAIN_000030']];chosen=fixed+[int(i) for i in worst if int(i) not in fixed][:3];canvas=Image.new('RGB',(1280,6*282+36),'white');drawimage=ImageDraw.Draw(canvas)
    for col,title in enumerate(['Original','Weak boundary','Local-only predicted','Maximin predicted','Maximin graph components@.5']):drawimage.text((col*256+4,5),title,fill='black')
    panelrows=[]
    for row,j in enumerate(chosen):
        k=int(o['fold'][j]);an=np.load(_repo_io(c.OLD/f'anchors/fold{k}.npz'));loc=int(np.flatnonzero(an['V']==d['dev'][j])[0]);pred={}
        for arm in ['topology','local']:
            ff=[np.load(_repo_io(c.HERE/f'encoders/fold{k}-refit-seed{r}-{arm}/features.npz')) for r in [0,1]];pred[arm]=(np.mean([f['query_boundary'][loc].astype(float) for f in ff],0),np.mean([f['query_edges'][loc].astype(float) for f in ff],0))
        arrays=[z['pixels'][j,0],np.uint8(255*z['boundary'][j]),np.uint8(255*np.clip(pred['local'][0],0,1)),np.uint8(255*np.clip(pred['topology'][0],0,1)),components(pred['topology'][1])]
        for col,x in enumerate(arrays):canvas.paste(Image.fromarray(x).convert('RGB').resize((256,256),Image.Resampling.NEAREST),(col*256,row*282+58))
        label=f"{o['ids'][j]} | {'fixed label-blind example' if row<3 else 'selected failure'} | y={o['y'][j]:.2f},082={o['baseline082'][j]:.2f},new={o['topology'][j]:.2f}";drawimage.text((4,row*282+39),label,fill='black');panelrows.append(dict(ID=str(o['ids'][j]),selection='fixed before outcomes' if row<3 else 'largest post-score harm',outer_fold=k,predicted_without_source_training=True))
    canvas.save(_repo_io(c.HERE/'heldout-boundaries.png'));c.save(_repo_io('diagnostics.json'),dict(worst=cases(worst),best=cases(best),weak_structure=weak,representation=rank,panel_sources=panelrows,panel_limit='Prediction maps storedfloat16 before plotting, so the graph@.5 illustration may differ at near-threshold edges from fullfloat32 readout signatures. These are auxiliary-mask targets, not actual grain annotations.',hardness_selected_parameters=False,source_sha256=c.sha(__file__)));print('POST-SCORE DIAGNOSTICS SAVED',flush=True)
if __name__=='__main__':
    sys.addaudithook(c.guard)
    with c.threadpool_limits(limits=2):main()
