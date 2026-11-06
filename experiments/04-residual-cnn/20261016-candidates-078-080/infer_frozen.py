"""Rebuild076 from component caches and infer the preserved full500 CNNs."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse
import csv
import json
import os
import numpy as np
from PIL import Image
import torch
from threadpoolctl import threadpool_limits
import study as s


def no_train(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=s.Path(os.fsdecode(args[0])).absolute()
        if p==_repo_path(s.ROOT, 'data/train.csv') or p.is_relative_to(_repo_path(s.ROOT, 'data/train')) or p==s.R4.parent/'inputs.npz':
            raise RuntimeError('Inference must not read training targets or pixels: '+str(p))


def main(model_root=None):
    # Check protected input hashes before entering inference-only access scope.
    s.protect();s.sys.addaudithook(no_train)
    used={};ids=[r['ID'] for r in csv.DictReader((_repo_path(s.ROOT, 'data/sample_submission.csv')).open())]
    def track(p):
        p=s.Path(p);used[str(p.relative_to(s.ROOT))]=s.r.digest(p);return p
    def csvpred(tag):return s.read_csv(track(_repo_path(s.ROOT, f'artifacts/submission-candidate-{tag}.csv')),ids)
    def npz(path):
        z=dict(np.load(_repo_io(track(path)),allow_pickle=True))
        assert z['ids'].astype(str).tolist()==ids
        return z
    exp=_repo_path(s.ROOT, 'experiments');h=exp/'push-1300';final=s.R4/'final500'
    model_root=final/'models' if model_root is None else s.Path(model_root).resolve()
    p73,p21=csvpred('073'),csvpred('021')
    G=npz(exp/'push-1200/fin-d1234n12.npz');B=npz(h/'fin-srcwB.npz')
    R=npz(exp/'probabilistic-grains-900/ridge/final500/predictions.npz')
    S=npz(exp/'probabilistic-grains-900/set_model/full500/predictions.npz')
    failed=R['failed']|S['failed'];noise=np.nan_to_num(G['noise'],nan=0)
    high=(noise>=12)&~failed;low=(noise<12)&~failed
    p74=p73+high*(.128*(B['ridge']-G['ridge'])+.128*(B['logridge']-G['logridge']))
    assert np.array_equal(p74,csvpred('074'))
    T9=npz(h/'fin-h9-tab-cpu.npz');Tn=npz(h/'fin-nb-tab-cpu.npz')
    full={tag:(csvpred(tag)-.25*p21)/.75 for tag in ('038','039','044')}
    old_tab=np.where(low,np.mean([Tn['nb_'+c] for c in full],axis=0),np.mean(list(full.values()),axis=0))
    h9_tab=np.where(low,np.mean([T9['nb_'+c] for c in full],axis=0),np.mean([T9['full_'+c] for c in full],axis=0))
    p76=np.where(failed,p74,p74+.384*(h9_tab-old_tab))
    assert np.array_equal(p76,csvpred('076'))
    features=npz(exp/'push-800/feat-u1dmavgH2Z-anghcor-test.npz')
    X=features['X'].astype(float);keys=features['keys'].astype(str).tolist();band=(noise>=12).astype(int)
    pn=npz(h/'fin-nb-partners.npz');ridge=np.where(band,G['ridge'],pn['ridge_nb'])
    tab=np.where(band,np.mean(list(full.values()),axis=0),np.mean([Tn['nb_'+c] for c in full],axis=0))
    T=(p73-.16*p21)/.84;T[failed]=p73[failed]
    raw=np.c_[T,tab-ridge,noise,X[:,keys.index('u_ph')],
              np.log1p(np.maximum(np.nan_to_num(X[:,keys.index('n_obj')],nan=0),0)),X[:,keys.index('coverage')]]
    saved=npz(final/'test-predictions.npz');assert np.array_equal(raw,saved['context_raw'],equal_nan=True)
    recipe=json.loads(track(final/'recipe.json').read_text());weights=np.array(recipe['production_weights']['addition'])
    assert weights[0]==0 and np.isclose(weights[1],.5467904360508202)
    active=np.flatnonzero(high);images=[]
    for i in active:
        with Image.open(_repo_io(track(_repo_path(s.ROOT, f'data/test/{ids[i]}.png')))) as image:
            assert image.mode=='L' and image.size==(256,256)
            images.append(np.asarray(image,np.float32)/255.)
    images=np.stack(images)[:,None];corrections=np.zeros((2,len(ids)))
    for j in (0,1):
        ck=torch.load(_repo_io(track(model_root/f'aux-seed{j}/model.pt')),map_location='cpu',weights_only=True)
        meta=ck['normalization'];assert meta==recipe['normalization']
        model=s.e.c.ComplementCNN();model.load_state_dict(ck['state_dict'])
        im=((images-meta['image_mean'])/meta['image_std']).astype(np.float32)
        clean=np.where(np.isfinite(raw),raw,np.array(meta['context_median']))
        ctx=((clean-np.array(meta['context_mean']))/np.array(meta['context_std'])).astype(np.float32)[active]
        corrections[j,active]=s.e.c.predict(model,im,ctx,np.arange(len(active)),'aux')*meta['target_std']
    replay_error=float(np.max(np.abs(corrections-saved['seed_corrections'])))
    assert replay_error<1e-5
    # Preserve the original float arithmetic of the frozen submitted difference.
    corrected73=p73+weights[band]*corrections.mean(0);corrected73[failed]=p73[failed]
    pred=p76+(corrected73-p73)
    reference=s.read_csv(s.HERE/'candidate-076-plus-frozen-cnn.csv',ids)
    error=float(np.max(np.abs(pred-reference)));assert error<1e-5
    out=s.HERE/'verification/inferred-from-models.csv'
    out.parent.mkdir(exist_ok=True)
    if out.exists():assert np.array_equal(s.read_csv(out,ids),pred)
    else:
        with out.open('x',newline='') as f:
            w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,map(lambda x:repr(float(x)),pred)))
    s.save(_repo_io('frozen-inference-verification.json'),dict(complete=True,rebuilt074_and076_exact=True,
        context_rebuild_exact=True,CNN_replay_max_error=replay_error,candidate_max_error=error,
        candidate_bitwise_exact=bool(np.array_equal(pred,reference)),CSV_sha256=s.r.digest(out),
        model_root=str(model_root.relative_to(s.ROOT)),
        training_labels_or_pixels_read=False,fit_during_inference=False,source_sha256=used,
        production_coefficient=weights.tolist(),uploaded=False))
    print('FROZEN INFERENCE VERIFIED',replay_error,error,'exact',np.array_equal(pred,reference),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--models',type=s.Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main(args.models)
