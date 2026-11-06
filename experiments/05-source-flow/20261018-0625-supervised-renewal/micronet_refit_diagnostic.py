from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
import micronet_direct as m
import numpy as np,torch,json,time
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

@torch.no_grad()
def features(body,px,ix,views):
    body.eval();out=[];mu=torch.tensor([.485,.456,.406])[None,:,None,None];sd=torch.tensor([.229,.224,.225])[None,:,None,None]
    for first in range(0,len(ix),8):
        gray=torch.tensor(np.array(px[ix[first:first+8]],dtype=np.float32)/255.);ff=[]
        for rot in range(views):
            x=torch.rot90(gray,rot,(2,3));f=body((x.expand(-1,3,-1,-1)-mu)/sd);r=torch.stack([x.pow(k).mean((1,2,3)) for k in [1,2,3,4]],1);ff.append(torch.cat([f,r],1).numpy())
        out.append(np.stack(ff,1))
    return np.concatenate(out)

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);root=_repo_path(HERE, 'runs/micronet-direct-f0-s01');ck=torch.load(_repo_io(root/'resume.pt'),map_location='cpu',weights_only=False)
    d=np.load(_repo_path(HERE, 'data.npz'));px=np.load(_repo_path(HERE, 'train-pixels.npy'),mmap_mode='r');tr=np.array(ck['recipe']['train']);q=np.array(ck['recipe']['query'])
    body,_=m.backbone('micronet');body.load_state_dict({k.removeprefix('body.'):v for k,v in ck['state'].items() if k.startswith('body.')});start=time.monotonic()
    T=features(body,px,tr,1)[:,0];Q=features(body,px,q,4);norm=m.RMSNorm().fit(T);ridge=Ridge(alpha=ck['recipe']['head_alpha']).fit(norm.transform(T),d['y'][tr])
    pred=np.mean(ridge.predict(norm.transform(Q.reshape(-1,Q.shape[-1]))).reshape(len(q),4),1)
    oldnorm=np.clip((Q-ck['state']['mean'].numpy())/ck['state']['scale'].numpy(),-8,8)
    neural=ck['ym']+ck['ys']*(oldnorm@ck['state']['head.weight'].numpy()[0]+ck['state']['head.bias'].item()).mean(1)
    base=d['base083'][d['fold']==0];r=dict(epoch=ck['epoch'],head_alpha=ck['recipe']['head_alpha'],original_head_rmse=m.rmse(neural,d['y'][q]),refit_head_rmse=m.rmse(pred,d['y'][q]),refit_training_rmse=m.rmse(ridge.predict(norm.transform(T)),d['y'][tr]),quarter_blend_rmse=m.rmse(.75*base+.25*pred,d['y'][q]),seconds=time.monotonic()-start,query_rows_used_for_fit=False)
    (_repo_path(HERE, f'micronet-refit-epoch{ck["epoch"]}.json')).write_text(json.dumps(r,indent=2));np.savez(_repo_path(HERE, f'micronet-refit-epoch{ck["epoch"]}.npz'),train=tr,query=q,prediction=pred,original_head=neural);print(json.dumps(r),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
