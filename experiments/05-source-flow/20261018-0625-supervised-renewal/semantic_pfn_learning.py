"""Learn shared soft pore/phase assignments through whole-image hardness prediction.

Earlier soft-measurement work used only sixteen morphology controls and a Ridge
head (fixed model RMSE13.74). This uses the actual104-column production bank
and a frozen TabPFN query loss. No grain receives an independent hardness label.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
import argparse,copy,hashlib,json,math,resource,time,gc
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
import pfn_gradient_probe as pf
import pfn_feature_learning as fl
import soft_measurement as sm

from semantic_grains import Measurement,load as load_bags,KEYS

@torch.no_grad()
def prediction(reg,model,train,query,y):
    A=model.extra(train).numpy();B=model.extra(query).numpy();ps=[]
    for seed in [0,100,200]:
        m=pf.harness.tabpfn_new('v3.5',seed+model.data['fold_index'],'cpu',n_est=8);m.fit(np.c_[model.data['raw_physical'][train],A],y.numpy());ps.append(m.predict(np.c_[model.data['raw_physical'][query],B]));del m;gc.collect()
    return np.mean(ps,0)

def fit(a,d,X,data,train,query,out,steps,select=False,probe=False):
    out.mkdir(exist_ok=True);torch.manual_seed(a.seed);norm=fl.Features().fit(X[train]);base=norm.transform(X);model=Measurement(data,base,train);reg=pf.make(0);rng=np.random.default_rng(a.seed);y=torch.tensor(d['y'][train],dtype=torch.float32)
    spec=vars(a)|dict(train_ids=d['ids'][train].tolist(),query_ids=d['ids'][query].tolist(),stage_steps=steps,select=select,
        measurement='Shared17→16→2networkadjustssoftporeandphaseprobabilitiesforeachoriginalgrain;21area/count/size/shape/spatialmomentsrecomputed. Hardporeexclusionisremoved. Whole-imagehardnesslabels only. Logitcorrectionsbounded±6, initializedzero;outputpenalty.01.',
        predictor='Actual104productioncolumns plus21semanticmeasurements. GradienttrainingusesfrozenlocalTabPFN3.5n1identity,noFingerprint;innerselectionandfinalpredictionsusenativeTabPFNn8x3seeds. Conditionalphysical104.',
        fitting='Onlytraining originals inall optimization contexts/queries. Source-onlyinitialfeaturemean/scale. InnerA213/B107select0/25/50/100/200, thenT320refitwithsameLRhorizon. Allmeasurementsfromoriginalzero-shotmasks, no newimages.',
        prior=.01,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),measurement_source_sha256=hashlib.sha256((_repo_path(HERE, 'semantic_grains.py')).read_bytes()).hexdigest())
    fl.save(_repo_io(out/'recipe.json'),spec);initial=prediction(reg,model,train,query,y);np.savez(_repo_io(out/'initial.npz'),query=query,prediction=initial);bestloss=fl.rmse(initial,d['y'][query]);beststep=0;best=copy.deepcopy(model.state_dict());curve=[dict(step=0,query_rmse=bestloss)];start=time.monotonic();first=0;opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.001)
    if (out/'resume.pt').exists() and not probe:
        ck=torch.load(_repo_io(out/'resume.pt'),map_location='cpu',weights_only=False);assert ck['recipe']==spec;model.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer']);rng.bit_generator.state=ck['rng'];first=ck['step'];bestloss=ck['bestloss'];beststep=ck['beststep'];best=ck['best'];curve=ck['curve']
    for step in range(first,steps):
        ix=rng.permutation(len(train));nctx=min(a.context,len(train)-a.queries);c=ix[:nctx];q=ix[nctx:nctx+a.queries];assert not set(c)&set(q)
        opt.zero_grad(set_to_none=True);Z=model(train[np.r_[c,q]]);pred=pf.forward(reg,Z[:nctx],y[c],Z[nctx:]);loss=((pred-y[q])/20).square().mean()+.01*model.penalty;assert torch.isfinite(loss);loss.backward();grad=float(nn.utils.clip_grad_norm_(model.parameters(),1.));assert np.isfinite(grad)
        memory=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if probe:
            r=dict(seconds=time.monotonic()-start,memory=memory,gradient_norm=grad,optimizer_updates=0,context=nctx,queries=len(q),trainable_parameters=sum(p.numel() for p in model.parameters()));fl.save(_repo_io(out/'probe.json'),r);print('PROBE',json.dumps(r),flush=True);return r
        assert memory<6*2**30
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
    torch.set_num_threads(2);torch.set_num_interop_threads(1);d,X=pf.load();data=load_bags(d);data['raw_physical']=X;data['fold_index']=a.fold;train=d['dev'][d['fold']!=a.fold];query=d['dev'][d['fold']==a.fold];A,B=train_test_split(train,test_size=1/3,random_state=2026101933+a.fold);A=np.sort(A);B=np.sort(B);out=_repo_path(HERE, 'runs', a.tag);out.mkdir(exist_ok=True)
    if a.action=='probe':return fit(a,d,X,data,A,B,out/'probe',1,probe=True)
    s=out/'inner/done.json';selected=json.loads(s.read_text()) if s.exists() else fit(a,d,X,data,A,B,out/'inner',a.horizon,select=True);gc.collect();r=fit(a,d,X,data,train,query,out/'outer',selected['selected_step']);z=np.load(_repo_io(out/'outer/predictions.npz'));pr=z['prediction'];ix=np.flatnonzero(d['fold']==a.fold);base=d['base083'][ix];eligible=~d['failed'][ix]
    ctrl=np.mean([np.load(_repo_path(HERE, f'conductivity/readout/physical-f{a.fold}-s{s}.npz'))['prediction'] for s in [0,100,200]],0);system=base+.384*eligible*(pr-ctrl)
    r.update(physical104_n8x3_control_rmse=fl.rmse(ctrl,d['y'][query]),base083_rmse=fl.rmse(base,d['y'][query]),system_rmse=fl.rmse(system,d['y'][query]),quarter_blend_rmse=fl.rmse(base+.25*eligible*(pr-base),d['y'][query]),comparison='Actualcomplete083+.384*(newmeasurementpredictor -existingphysical104n8x3control). Differentpredictorconfiguration;sameT320/V80, conditionalhistoricalrepresentations.')
    np.savez(_repo_io(out/'predictions.npz'),train=train,query=query,prediction=pr,control=ctrl,system=system);fl.save(_repo_io(out/'done.json'),r);print('DONE',json.dumps(r),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['probe','train']);ap.add_argument('--fold',type=int,default=0);ap.add_argument('--tag',default='semantic-pfn-f0');ap.add_argument('--seed',type=int,default=2026101931);ap.add_argument('--horizon',type=int,default=200);ap.add_argument('--context',type=int,default=192);ap.add_argument('--queries',type=int,default=32);ap.add_argument('--lr',type=float,default=.003)
    with threadpool_limits(2):main(ap.parse_args())
