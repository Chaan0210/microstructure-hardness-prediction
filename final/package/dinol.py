"""DINOv2-L feature extraction and three-seed MLP training for G021."""
from pathlib import Path
import argparse,csv,gc,hashlib,importlib.util,json,os,sys,time
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
os.environ['XFORMERS_DISABLED']='1';os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
from PIL import Image
from threadpoolctl import threadpool_limits
PACKAGE=Path(__file__).resolve().parent;PROJECT=PACKAGE/'project'
S=PROJECT/'experiments/candidate021-178'
sys.path.insert(0,str(PROJECT/'experiments/reproduction-018-176/vendor'))

def features(a,out):
    from dinov2.hub.backbones import dinov2_vitl14
    data=Path(a.data).resolve();rows=list(csv.DictReader((data/('train.csv' if a.split=='train' else 'sample_submission.csv')).open()));ids=np.array([r['ID'] for r in rows])
    m=dinov2_vitl14(pretrained=False);m.load_state_dict(torch.load(PROJECT/'experiments/dinol-177/dinov2_vitl14_pretrain.pth',map_location='cpu',weights_only=True),strict=True)
    m.requires_grad_(False).eval().to(a.device);values=[];start=time.monotonic()
    for i in ids[:a.limit or len(ids)]:
        with Image.open(data/a.split/f'{i}.png') as im:
            assert im.mode=='L' and im.size==(256,256);raw=np.asarray(im.resize((504,504),Image.Resampling.BICUBIC),dtype=np.float32)/255.
        x=np.ascontiguousarray((np.repeat(raw[None],3,axis=0)-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None])
        with torch.inference_mode():
            layers=m.get_intermediate_layers(torch.tensor(x[None],device=a.device),n=[11,17,23],return_class_token=True,norm=True)
            value=torch.cat([torch.cat([c,p.mean(1),p.std(1,correction=0)],1) for p,c in layers],1).cpu().numpy()[0]
        assert value.shape==(9216,) and np.isfinite(value).all();values.append(value)
        if len(values)%25==0:print('DINO-L',len(values),flush=True)
    X=np.array(values);target=out/f'{a.split}-features.npz';payload=dict(ids=ids[:len(X)],X=X)
    if a.split=='train':payload['y']=np.array([float(r['hardness']) for r in rows[:len(X)]])
    np.savez_compressed(target,**payload)
    error=None
    if not a.retrained:
        ref=np.load(S/f'{a.split}-features.npz');error=float(abs(X-ref['X'][:len(X)]).max())
    (out/'verification.json').write_text(json.dumps(dict(rows=len(X),max_abs_feature_difference=error,seconds=time.monotonic()-start,scope='Frozen DINO-L raw-image inference; no encoder fitting.'),indent=2)+'\n')

def train(a,out):
    spec=importlib.util.spec_from_file_location('selected_lhead',S/'mlp_head.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    z=np.load(a.features or S/'train-features.npz');ids,X=z['ids'],z['X']
    lab={r['ID']:float(r['hardness']) for r in csv.DictReader((Path(a.data)/'train.csv').open())};y=np.array([lab[str(i)] for i in ids])
    assert len(ids)==500 and X.shape==(500,9216) and np.array_equal(z['y'],y)
    sc,ym,ys,_,bins=h.preprocess(X,y);xt=torch.tensor(sc.transform(X),dtype=torch.float32);yt=torch.tensor((y-ym)/ys,dtype=torch.float32);bt=torch.tensor(bins)
    rows=[]
    for seed in [20261128,20261205,20261219]:
        m,log=h.fit(xt,yt,bt,seed,0.)
        obj=dict(model=m.state_dict(),input_dim=9216,scaler_mean=torch.tensor(sc.mean_),scaler_scale=torch.tensor(sc.scale_),scaler_var=torch.tensor(sc.var_),target_mean=ym,target_std=ys,train_ids=ids.tolist(),seed=seed,config=h.CONFIG,effective_aux_weight=0.)
        torch.save(obj,out/f'L-MLP-{seed}.pt');identical=None
        if not a.retrained:
            ref=torch.load(S/f'models/L-MLP-{seed}.pt',map_location='cpu',weights_only=True)
            identical=all(torch.equal(v,ref['model'][k]) for k,v in obj['model'].items())
        rows.append(dict(seed=seed,all_state_tensors_exact=identical,training_log=log));print('TRAINED L',seed,'exact',identical,flush=True)
    (out/'verification.json').write_text(json.dumps(dict(training_rows=500,tests_used=False,states=rows,scope='Three MLPs retrained on fixed DINO-L training features, not foundation-model retraining.'),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['features','train']);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--split',choices=['train','test'],default='train');p.add_argument('--device',choices=['cpu','mps'],default='mps');p.add_argument('--limit',type=int,default=0);p.add_argument('--features')
    p.add_argument('--project',type=Path,default=PROJECT)
    p.add_argument('--retrained',action='store_true')
    a=p.parse_args();PROJECT=a.project.resolve();S=PROJECT/'experiments/candidate021-178'
    sys.path.insert(0,str(PROJECT/'experiments/reproduction-018-176/vendor'))
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    with threadpool_limits(2):features(a,out) if a.action=='features' else train(a,out)
