from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
import phase_joint_v3 as p
import numpy as np,torch,json,time
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    p.tensor=lambda a,dtype=torch.float32:torch.tensor(a,dtype=dtype)
    out=_repo_path(HERE, 'runs/phase-joint-b4-f0-s81');d=dict(np.load(_repo_path(HERE, 'data.npz')));px=np.load(_repo_path(HERE, 'train-pixels.npy'),mmap_mode='r');z=np.load(_repo_io(out/'frozen-features.npz'));tr=z['train']
    aux=p.prepare(d,px,tr,out);X=np.c_[z['train_features'],z['train_globals']];mean=X.mean(0);scale=np.maximum(X.std(0),.01);ym=d['y'][tr].mean();ys=d['y'][tr].std();y=(d['y'][tr]-ym)/ys
    ridge=Ridge(alpha=10.).fit((X-mean)/scale,y);frozen_train=ridge.predict((X-mean)/scale)*ys+ym
    ck=torch.load(_repo_io(out/'hardness-checkpoint.pt'),map_location='cpu',weights_only=False);net=p.UNet(32)
    net.load_state_dict({k.removeprefix('net.'):v for k,v in ck['state'].items() if k.startswith('net.')});net.eval();start=time.monotonic()
    f=p.extract(net,px,aux,tr);newX=np.c_[f,z['train_globals']];beta=ck['state']['head.weight'].numpy()[0];bias=ck['state']['head.bias'].numpy()[0]
    pred=ym+ys*(((newX-np.array(ck['mean']))/np.array(ck['scale']))@beta+bias)
    record=dict(epoch=ck['epoch'],frozen_training_rmse=p.ga.rmse(frozen_train,d['y'][tr]),checkpoint_training_rmse=p.ga.rmse(pred,d['y'][tr]),training_prediction_RMS_change=float(np.sqrt(np.mean((pred-frozen_train)**2))),query_labels_used=False)
    init=torch.load(_repo_io(out/'pretrained.pt'),map_location='cpu',weights_only=True);net.load_state_dict(init['state']);net.eval();rng=np.random.default_rng(2026101887)
    pos={int(j):i for i,j in enumerate(tr)};probes=[]
    for repeat in range(6):
        ix=rng.choice(tr,4,replace=False);rows=np.array([pos[int(j)] for j in ix]);x=p.tensor(np.array([p.norm(px[j,0]) for j in ix])[:,None])
        ff=p.physical_features(net(x),aux);gg=p.tensor(z['train_globals'][rows]);xx=torch.cat([ff,gg],1)
        pp=((xx-p.tensor(mean))/p.tensor(scale))@p.tensor(ridge.coef_)+float(ridge.intercept_)
        hard=(pp-p.tensor((d['y'][ix]-ym)/ys)).square().mean();gh=torch.autograd.grad(hard,list(net.parameters()))
        al=p.aux_loss(net,p.aux_batch(aux,px,rng,batch=2),aux);ga=torch.autograd.grad(al,list(net.parameters()))
        hn=float(torch.sqrt(sum(g.square().sum() for g in gh)));an=float(torch.sqrt(sum(g.square().sum() for g in ga)));dot=float(sum((a*b).sum() for a,b in zip(gh,ga)))
        probes.append(dict(hardness_encoder_gradient=hn,auxiliary_encoder_gradient=an,ratio_hardness_to_auxiliary=hn/max(an,1e-12),cosine=dot/max(hn*an,1e-12)))
    record.update(gradient_probes=probes,median_gradient_ratio=float(np.median([r['ratio_hardness_to_auxiliary'] for r in probes])),seconds=time.monotonic()-start,optimizer_updates=0)
    (_repo_path(HERE, f'phase-optimization-epoch{ck["epoch"]}.json')).write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
