"""Learn local continuous-flow measurements through whole-image hardness prediction.

This tests direct supervised feature learning on continuous flow maps after
fixed multiscale flow summaries failed. The original H300 feature extractor
has global auxiliary ancestry, so the exploratory comparison is conditional.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
import argparse,copy,hashlib,json,math,resource,time,gc,fcntl,tempfile,subprocess
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
import pfn_checkpoint as pf
import pfn_feature_learning as fl
import soft_measurement as sm

from flow_neural_measurement import Measurement,load as load_bags,KEYS

@torch.no_grad()
def prediction(reg,model,train,query,y):
    A=model.extra(train).numpy();B=model.extra(query).numpy()
    with tempfile.TemporaryDirectory(prefix='flow-native-',dir=_repo_path(HERE, 'tmp')) as directory:
        directory=Path(directory);source=directory/'input.npz';target=directory/'output.npz'
        np.savez(_repo_io(source),X=np.c_[model.data['raw_physical'][train],A],y=y.numpy(),Q=np.c_[model.data['raw_physical'][query],B],fold=model.data['fold_index'])
        subprocess.run([sys.executable,str(_repo_path(HERE, 'native_readout_worker.py')),'--input',str(source),'--output',str(target)],cwd=_repo_io(pf.ROOT),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),check=True,timeout=600)
        z=np.load(_repo_io(target));result=z['prediction'].copy();print('NATIVE_READOUT_PEAK',int(z['peak_rss']),flush=True);return result

def fit(a,d,X,data,train,query,out,steps,select=False,probe=False):
    out.mkdir(exist_ok=True);torch.manual_seed(a.seed);norm=fl.Features().fit(X[train]);base=norm.transform(X);model=Measurement(data,base,train);reg=pf.make(0);rng=np.random.default_rng(a.seed);y=torch.tensor(d['y'][train],dtype=torch.float32)
    spec=vars(a)|dict(train_ids=d['ids'][train].tolist(),query_ids=d['ids'][query].tolist(),stage_steps=steps,select=select,
        measurement='Small4→12→16→8convolutionalmeasurementnetwork over canonicalC4-averaged H300flow/confidence plus original grayscale;32mean/SD/gray-covariance/tile-variationfeatures. Whole-imagehardness only. No rendered images; these are latent features of provided originals. Parameter-anchor penalty.01.',
        predictor='Actual104productioncolumns plus the stronger38flowstatistics plus32learnedmeasurements. GradienttrainingusesfrozenlocalTabPFN3.5n1identity,noFingerprint;innerselectionandfinalpredictionsusenativeTabPFNn8x3seeds. Conditionalphysical104.',
        fitting='Onlytraining originals inall optimization contexts/queries. Source-onlyinitialfeaturemean/scale. InnerA213/B107select0/25/50/100/200, thenT320refitwithsameLRhorizon. Originalorientation retained after inverse-aligning/averaging4encoder views; nohardness-labelrotationaugmentationorrotation-pooledneuralreadout. H300globalauxiliaryancestry remains conditional. No new images.',
        prior=.01,checkpointing='InstalledPerformanceOptionsforce_recompute_layer=True;output/gradientparityrequired. Nativevalidationheadsruninaseparateprocesstolimitretainedallocations;samemodels/seeds/settings.',gradient_source_sha256=hashlib.sha256((_repo_path(HERE, 'pfn_checkpoint.py')).read_bytes()).hexdigest(),worker_source_sha256=hashlib.sha256((_repo_path(HERE, 'native_readout_worker.py')).read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),measurement_source_sha256=hashlib.sha256((_repo_path(HERE, 'flow_neural_measurement.py')).read_bytes()).hexdigest())
    fl.save(_repo_io(out/'recipe.json'),spec);initial=prediction(reg,model,train,query,y)
    previous=_repo_path(HERE, 'runs/flow-neural-pfn-f0/probe/initial.npz')
    if a.fold==0 and select and previous.exists():
        old=np.load(_repo_io(previous));assert np.array_equal(old['query'],query);difference=float(abs(old['prediction']-initial).max());fl.save(_repo_io(out/'readout-worker-parity.json'),dict(max_difference=difference));assert difference<1e-5
    np.savez(_repo_io(out/'initial.npz'),query=query,prediction=initial);bestloss=fl.rmse(initial,d['y'][query]);beststep=0;best=copy.deepcopy(model.state_dict());curve=[dict(step=0,query_rmse=bestloss)];start=time.monotonic();first=0;opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.001)
    if (out/'resume.pt').exists() and not probe:
        ck=torch.load(_repo_io(out/'resume.pt'),map_location='cpu',weights_only=False);assert ck['recipe']==spec;model.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer']);rng.bit_generator.state=ck['rng'];first=ck['step'];bestloss=ck['bestloss'];beststep=ck['beststep'];best=ck['best'];curve=ck['curve']
    for step in range(first,steps):
        ix=rng.permutation(len(train));nctx=min(a.context,len(train)-a.queries);c=ix[:nctx];q=ix[nctx:nctx+a.queries];assert not set(c)&set(q)
        opt.zero_grad(set_to_none=True);model.step=step;Z=model(train[np.r_[c,q]]);pred=pf.forward(reg,Z[:nctx],y[c],Z[nctx:]);loss=((pred-y[q])/20).square().mean()+.01*model.penalty;assert torch.isfinite(loss);loss.backward();grad=float(nn.utils.clip_grad_norm_(model.parameters(),1.));assert np.isfinite(grad)
        memory=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;current_rss=int(subprocess.check_output(['ps','-o','rss=','-p',str(os.getpid())],text=True).strip())*1024
        if probe:
            r=dict(seconds=time.monotonic()-start,memory=memory,current_rss=current_rss,gradient_norm=grad,optimizer_updates=0,context=nctx,queries=len(q),trainable_parameters=sum(p.numel() for p in model.parameters()));fl.save(_repo_io(out/'probe.json'),r);print('PROBE',json.dumps(r),flush=True);return r
        
        if current_rss>=6*2**30 or memory>=8*2**30:
            fl.save(_repo_io(out/'memory-guard.json'),dict(step=step,optimizer_updates=step,current_rss=current_rss,peak_rss=memory));raise RuntimeError('Declared isolated-learner memory budget exceeded.')
        for g in opt.param_groups:g['lr']=a.lr*(.1+.9*.5*(1+math.cos(math.pi*step/a.horizon)))
        opt.step()
        if step==0 or (step+1)%10==0:print('TRAIN',out.name,step+1,float(loss.detach()),'seconds',time.monotonic()-start,'rss',memory,flush=True)
        if step+1 in [25,50,100,200] or step+1==steps:
            pr=prediction(reg,model,train,query,y);score=fl.rmse(pr,d['y'][query]);row=dict(step=step+1,query_rmse=score,parameter_norm=float(sum(v.detach().square().sum() for v in model.parameters()).sqrt()),seconds=time.monotonic()-start);curve.append(row);print('EVAL',out.name,json.dumps(row),flush=True);fl.save(_repo_io(out/'curve.json'),curve)
            if select and score<bestloss:bestloss=score;beststep=step+1;best=copy.deepcopy(model.state_dict())
        if (step+1)%10==0 or step+1==steps:
            ck=dict(state=model.state_dict(),optimizer=opt.state_dict(),recipe=spec,step=step+1,rng=rng.bit_generator.state,bestloss=bestloss,beststep=beststep,best=best,curve=curve);torch.save(ck,_repo_io(out/'resume-new.pt'));(out/'resume-new.pt').replace(out/'resume.pt');gc.collect()
    if select:model.load_state_dict(best)
    pred=prediction(reg,model,train,query,y);torch.save(dict(state=model.state_dict(),normalization=norm.state(),recipe=spec,selected_step=beststep if select else steps),_repo_io(out/'model.pt'));np.savez(_repo_io(out/'predictions.npz'),train=train,query=query,prediction=pred,initial=initial)
    with torch.no_grad():np.savez(_repo_io(out/'measurement-features.npz'),train=train,query=query,train_features=model.extra(train).numpy(),query_features=model.extra(query).numpy())
    r=dict(rmse=fl.rmse(pred,d['y'][query]),initial_rmse=fl.rmse(initial,d['y'][query]),selected_step=beststep if select else steps,seconds=time.monotonic()-start);fl.save(_repo_io(out/'done.json'),r);return r

def main(a):
    torch.set_num_threads(2);torch.set_num_interop_threads(1);d,X=pf.load();physical=np.load(_repo_path(pf.ROOT, 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz'));position={str(v):i for i,v in enumerate(physical['ids'])};X=physical['X'][[position[str(v)] for v in d['ids']]].astype(float);flow=np.load(_repo_path(HERE, 'domain-semantics/H300-features.npz'));assert np.array_equal(flow['ids'],d['ids']);X=np.c_[X,flow['flow'][:,:38]];data=load_bags(d);data['raw_physical']=X;data['fold_index']=a.fold;train=d['dev'][d['fold']!=a.fold];query=d['dev'][d['fold']==a.fold];A,B=train_test_split(train,test_size=1/3,random_state=2026101933+a.fold);A=np.sort(A);B=np.sort(B);out=_repo_path(HERE, 'runs', a.tag);out.mkdir(exist_ok=True)
    if a.action=='probe':return fit(a,d,X,data,A,B,out/'probe',1,probe=True)
    s=out/'inner/done.json';selected=json.loads(s.read_text()) if s.exists() else fit(a,d,X,data,A,B,out/'inner',a.horizon,select=True);gc.collect();r=fit(a,d,X,data,train,query,out/'outer',selected['selected_step']);z=np.load(_repo_io(out/'outer/predictions.npz'));pr=z['prediction'];ix=np.flatnonzero(d['fold']==a.fold);base=d['base083'][ix];eligible=~d['failed'][ix]
    ctrl=np.mean([np.load(_repo_path(HERE, f'conductivity/readout/physical-f{a.fold}-s{s}.npz'))['prediction'] for s in [0,100,200]],0);linear=np.load(_repo_path(HERE, 'flow-linear-check/oof.npz'));lead=np.load(_repo_path(HERE, 'flow-content-ablation/oof.npz'));system=base+.384*eligible*(pr-ctrl)+.256*eligible*(lead['learned_fields38_linear'][ix]-linear['control'][ix]);flow_system=lead['learned_fields38_system'][ix]
    r.update(flow38_system_rmse=fl.rmse(flow_system,d['y'][query]),physical104_n8x3_control_rmse=fl.rmse(ctrl,d['y'][query]),base083_rmse=fl.rmse(base,d['y'][query]),system_rmse=fl.rmse(system,d['y'][query]),quarter_blend_rmse=fl.rmse(base+.25*eligible*(pr-base),d['y'][query]),comparison='Complete frozen flow38 system +.384*(new learned-field TabPFN -flow38 TabPFN), with the existing fixed flow38 linear delta unchanged. SameT320/Q80; global H300/physical representations remain conditional. Inner-selected step is refit on outerT only.')
    np.savez(_repo_io(out/'predictions.npz'),train=train,query=query,prediction=pr,control=ctrl,system=system);fl.save(_repo_io(out/'done.json'),r);print('DONE',json.dumps(r),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['probe','train']);ap.add_argument('--fold',type=int,default=0);ap.add_argument('--tag',default='flow-neural-pfn-f0');ap.add_argument('--seed',type=int,default=2026101931);ap.add_argument('--horizon',type=int,default=200);ap.add_argument('--context',type=int,default=64);ap.add_argument('--queries',type=int,default=16);ap.add_argument('--lr',type=float,default=.001)
    with (_repo_path(HERE, 'tmp/heavy-learning.lock')).open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        with threadpool_limits(2):main(ap.parse_args())
