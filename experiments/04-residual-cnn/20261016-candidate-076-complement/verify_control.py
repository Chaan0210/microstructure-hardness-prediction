"""Show duplicate-channel models remain in the original CNN function space."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json
import numpy as np
from PIL import Image
import torch
from threadpoolctl import threadpool_limits
import stream_cnn as s
p=s.p


def main():
    p.protect();z=dict(np.load(_repo_io(p.HERE/'baseline076.npz')));source=np.load(_repo_io(p.R4.parent/'inputs.npz'))
    X=source['X'][z['dev']];keys=list(source['keys'])
    T=(z['p73']-.16*z['G'])/.84;T[z['failed']]=z['p73'][z['failed']]
    raw=np.c_[T,z['tab073']-z['ridge073'],z['noise'],X[:,keys.index('u_ph')],
              np.log1p(np.maximum(np.nan_to_num(X[:,keys.index('n_obj')],nan=0),0)),X[:,keys.index('coverage')]]
    records=[]
    for k in range(5):
        ix=np.flatnonzero(z['fold']==k)
        images=np.stack([np.asarray(Image.open(_repo_path(p.ROOT, f'data/train/{i}.png')),np.float32)/255. for i in z['ids'][ix]])[:,None]
        for seed in (0,1):
            path=p.HERE/f'stream/duplicate-seed{seed}/fold{k}-final/model.pt'
            ck=torch.load(_repo_io(path),map_location='cpu',weights_only=True);meta=ck['normalization']
            im=((images-meta['image_mean'])/meta['image_std']).astype(np.float32)
            rr=np.where(np.isfinite(raw[ix]),raw[ix],np.array(meta['context_median']))
            ctx=((rr-np.array(meta['context_mean']))/np.array(meta['context_std'])).astype(np.float32)
            twin=s.TwoStream('duplicate');twin.load_state_dict(ck['state_dict'])
            single=s.Original();state={name:ck['state_dict'][name].clone() for name in single.state_dict()}
            state['blocks.0.0.weight']=state['blocks.0.0.weight'].sum(1,keepdim=True)
            single.load_state_dict(state)
            indices=np.arange(len(ix));a=s.e.c.predict(twin,im,ctx,indices,'aux')*meta['target_std']
            b=s.e.c.predict(single,im,ctx,indices,'aux')*meta['target_std']
            ref=np.load(_repo_io(path.with_name('predictions.npz')))
            assert np.array_equal(ref['validation_index'],z['dev'][ix])
            assert np.array_equal(a,ref['validation_prediction'])
            error=float(np.max(np.abs(a-b)));assert error<2e-5,error
            records.append(dict(fold=k,seed=seed,two_channel_replay_exact=True,collapsed_max_error=error,
                                original_checkpoint_sha256=s.r.digest(path)))
    p.save(_repo_io('collapsed-control-verification.json'),dict(all_passed=True,records=records,
        formula='Conv([x,x],[W0,W1],b) = Conv(x,W0+W1,b)',
        limitation='Shows equal function space, not whether initialization or optimizer dynamics caused score differences.',
        collapsed_models_deployed=False))
    p.protect();print('Verified10duplicate models and their single-channel equivalents',max(q['collapsed_max_error'] for q in records),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
