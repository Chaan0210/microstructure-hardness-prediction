"""Post-fit descriptive mechanisms; never select training settings here."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import experiment as e
import numpy as np,json,sys,argparse

def main(repeats):
    tag=''.join(map(str,repeats));z=np.load(_repo_io(e.HERE/f'oof-{tag}.npz'));d=dict(np.load(_repo_io(e.HERE/'inputs.npz')));records=[];geometry=[]
    for arm in e.ARMS:
        p=z[arm];b=z['baseline082'];y=z['y'];high=(z['noise']>=12)&~d['failed'];delta=p-b
        D=np.c_[np.ones(high.sum()),b[high]-b[high].mean()];beta=np.linalg.lstsq(D,delta[high],rcond=None)[0]
        affine=np.zeros(400);affine[high]=D@beta;orth=delta-affine
        mse=lambda x:float(np.mean((x-y)**2));total=mse(b)-mse(p);cal=mse(b)-mse(b+affine);signal=mse(b)-mse(b+orth)
        assert abs(total-cal-signal)<1e-10
        records.append(dict(arm=arm,high_noise_affine_intercept=float(beta[0]),high_noise_slope_change=float(beta[1]),total_MSE_gain=total,affine_MSE_gain=cal,orthogonal_MSE_gain=signal,
            orthogonal_RMSE_gain=e.rmse(b,y)-e.rmse(b+orth,y),status='Diagnostic projection of fixed OOF changes; not a new fitted or selectable predictor.'))
        for repeat in repeats:
            for k in range(5):
                root=e.HERE/f'models/seed{repeat}/fold{k}-refit-{arm}';q=np.load(_repo_io(root/'predictions.npz'));rec=json.loads((root/'training.json').read_text());meta=rec['normalization'];target=(d['y'][q['query']]-meta['target_mean'])/meta['target_std']
                emb=e.torch.tensor(q['embedding'],dtype=e.torch.float32);labels=e.torch.tensor(target,dtype=e.torch.float32)
                kl=float(e.neighborhood_loss(emb,emb,labels));Z=q['embedding'].astype(float);Z-=Z.mean(0);Z/=np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-6)
                ix=np.triu_indices(len(Z),1);x=(1-Z@Z.T)[ix];dist=abs(target[:,None]-target[None,:])[ix]
                correlation=float(np.corrcoef(x,dist)[0,1]) if x.std()>0 else None
                geometry.append(dict(repeat=repeat,fold=k,arm=arm,heldout_neighborhood_KL=kl,heldout_pair_distance_correlation=correlation,final_train_image_RMSE=rec['train_rmse'],heldout_image_RMSE=e.rmse(q['prediction'],d['y'][q['query']]),embedding_effective_rank=rec['embedding_effective_rank'],embedding_std_mean=rec['embedding_std_mean'],encoder_gradient_norms=rec['gradients']))
    e.save(_repo_io(f'mechanisms-{tag}.json'),dict(spread_decomposition=records,representation_diagnostics=geometry,
        limitations=['Pair distances are dependent, descriptive quantities; no pair-count significance test.',
            'Heldout geometry uses four-view averaged embeddings and a single whole validation fold, not dropout training minibatches; do not compare its absolute KL to online training KL.',
            'Orthogonal change scores are diagnosis only and do not authorize deploying an OOF-derived adjustment.']))
    print(json.dumps(records,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,nargs='+',default=[0]);args=p.parse_args();sys.addaudithook(e.guard)
    with e.threadpool_limits(limits=2):main(args.repeats)
