"""Small scratch CNN: dev-only fixed-epoch CV, no external weights/data or test reads."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
os.environ['TORCH_HOME']=str(_repo_path(__import__('pathlib').Path(__file__).resolve().parent, '.torch-cache'))
import sys,json,csv,time,resource,subprocess,gc,hashlib,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, '.venv')).resolve():raise SystemExit('Use project uv .venv')
import numpy as np
from PIL import Image
import torch
from torch import nn

def write(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
def sync(d):
    if d=='mps':torch.mps.synchronize()
def thermal():
    p=subprocess.run(['pmset','-g','therm'],capture_output=True,text=True)
    return (p.stdout+p.stderr).strip()
def memory(d):
    m={'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if d=='mps':
        for key in ('current_allocated_memory','driver_allocated_memory','recommended_max_memory'):
            if hasattr(torch.mps,key):m[key]=int(getattr(torch.mps,key)())
    return m
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__();layers=[];inc=1
        for i,out in enumerate((16,32,64,96)):
            layers.extend([nn.Conv2d(inc,out,5 if i==0 else 3,stride=2,padding=2 if i==0 else 1),nn.GroupNorm(4,out),nn.SiLU()]);inc=out
        self.features=nn.Sequential(*layers);self.head=nn.Sequential(nn.Linear(194,64),nn.SiLU(),nn.Dropout(.15),nn.Linear(64,1))
    def forward(self,x):
        h=self.features(x)
        stats=torch.cat([h.mean((2,3)),(h.var((2,3),correction=0)+1e-6).sqrt(),x.mean((2,3)),(x.var((2,3),correction=0)+1e-6).sqrt()],1)
        return self.head(stats).squeeze(1)
def load_dev(size=128):
    sp=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text());mapping=sp['development_folds'];ids=sorted(mapping)
    with (_repo_path(ROOT, 'data/train.csv')).open() as f:labels={r['ID']:float(r['hardness']) for r in csv.DictReader(f) if r['ID'] in mapping}
    assert len(ids)==400 and not set(ids)&set(sp['holdout'])
    X=np.stack([np.asarray(Image.open(_repo_path(ROOT, 'data/train', f'{s}.png')).resize((size,size),Image.Resampling.BOX),dtype=np.float32)/255. for s in ids])[:,None]
    return ids,X,np.array([labels[s] for s in ids]),np.array([mapping[s] for s in ids]),sp

def bench(out):
    out.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(20261003)
    ids,X,y,fid,sp=load_dev(256);tr=np.flatnonzero(fid!=0)[:32];X=X[tr];y=y[tr]
    result={'torch':torch.__version__,'mps_built':torch.backends.mps.is_built(),'mps_available':torch.backends.mps.is_available(),'threads':2,'image_size':256,'batch_size':32,'thermal_before':thermal(),'no_pretrained_weights':True,'devices':{}}
    for d in (['mps'] if result['mps_available'] else ['cpu']):
        torch.manual_seed(20261003);model=TinyCNN().to(d);opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
        x=torch.from_numpy((X-X.mean())/max(X.std(),1e-6)).to(d);t=torch.tensor((y-y.mean())/y.std(),dtype=torch.float32,device=d)
        times=[];losses=[]
        for step in range(23 if d=='mps' else 6):
            sync(d);start=time.perf_counter();opt.zero_grad(set_to_none=True);loss=nn.functional.mse_loss(model(x),t);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();sync(d)
            if step>=3:times.append(time.perf_counter()-start);losses.append(float(loss.detach().cpu()))
        result['devices'][d]={'measured_steps':len(times),'median_step_seconds':float(np.median(times)),'images_per_second':32/float(np.median(times)),'first_half_median':float(np.median(times[:max(1,len(times)//2)])),'second_half_median':float(np.median(times[len(times)//2:])),'memory':memory(d),'finite_losses':bool(np.isfinite(losses).all()),'loss_first':losses[0],'loss_last':losses[-1]}
        result['parameters']=sum(p.numel() for p in model.parameters());del model,opt,x,t;gc.collect()
        if d=='mps':torch.mps.empty_cache()
    result['thermal_after']=thermal();result['thermal_note']='Temperature sensors unavailable if pmset reports errors; timing alone cannot certify thermal safety.'
    write(out/'benchmark.json',result);print(json.dumps(result,indent=2),flush=True)

def cv(out,benchmark,size,epochs):
    out.mkdir(exist_ok=False);torch.set_num_threads(2)
    b=json.loads(benchmark.read_text());d='mps' if b['mps_available'] else 'cpu'
    if not b['devices'][d]['finite_losses']:raise RuntimeError('Benchmark failed')
    if size==256 and (b['devices'][d]['median_step_seconds']>.12 or b['devices'][d]['memory']['process_peak_rss_bytes']>2_000_000_000):raise RuntimeError('256px resource estimate exceeded; report before proceeding')
    ids,X,y,fid,sp=load_dev(size);basepath=_repo_path(ROOT, 'artifacts/run-005/best_model.joblib');csvpath=_repo_path(ROOT, 'artifacts/submission-candidate-002.csv')
    plan={'seed':20261003,'epochs':epochs,'average_last_epochs':5,'image_size':size,'batch_size':32,'lr':.002,'weight_decay':.01,'device':d,'augmentation':'random batch D4 rotations/reflections, training only; no intensity jitter','normalization':'pixel and target mean/std fit only on fold training rows','selection':'fixed endpoint, last five weight average; epoch validation diagnostic only, no early stop','ensemble_weights':{'baseline':.5,'grain':.25,'cnn':.25},'interpretation':'No arbitrary improvement cutoff; compare fold deltas and uncertainty; reused development CV is optimistic','no_holdout_evaluation':True,'no_test_reads':True,'pretrained':False,'baseline_model_sha256':sha(basepath),'baseline_csv_sha256':json.loads((_repo_path(ROOT, 'artifacts/run-007/plan.json')).read_text())['baseline_csv_sha256'],'torch':torch.__version__,'source_sha256':sha(Path(__file__))}
    write(out/'plan.json',plan);start=time.perf_counter();oof=np.zeros(len(y));folds=[];curves=[]
    for fold in range(5):
        tr=np.flatnonzero(fid!=fold);va=np.flatnonzero(fid==fold)
        assert not {sp['group_by_id'][ids[i]] for i in tr}&{sp['group_by_id'][ids[i]] for i in va}
        rng=np.random.default_rng(plan['seed']+fold);torch.manual_seed(plan['seed']+fold)
        mu,sd=float(X[tr].mean()),float(X[tr].std());ym,ys=float(y[tr].mean()),float(y[tr].std())
        Xt=(X[tr]-mu)/max(sd,1e-6);yt=(y[tr]-ym)/ys
        model=TinyCNN().to(d);opt=torch.optim.AdamW(model.parameters(),lr=plan['lr'],weight_decay=plan['weight_decay']);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=.0001)
        avg=None;ft=time.perf_counter();curve=[]
        for epoch in range(epochs):
            model.train();order=rng.permutation(len(tr));loss_sum=0.
            for first in range(0,len(tr),32):
                ix=order[first:first+32];xb=Xt[ix];xb=np.rot90(xb,int(rng.integers(4)),axes=(2,3))
                if rng.random()<.5:xb=xb[:,:,:,::-1]
                xb=torch.from_numpy(np.ascontiguousarray(xb)).to(d);target=torch.tensor(yt[ix],dtype=torch.float32,device=d)
                opt.zero_grad(set_to_none=True);pred=model(xb);loss=nn.functional.mse_loss(pred,target)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();loss_sum+=float(loss.detach().cpu())*len(ix)
            schedule.step();model.eval();val_pred=[]
            with torch.no_grad():
                for vf in range(0,len(va),32):
                    vx=torch.from_numpy((X[va[vf:vf+32]]-mu)/max(sd,1e-6)).to(d)
                    val_pred.extend((model(vx).cpu().numpy()*ys+ym).tolist())
            curve.append({'epoch':epoch+1,'augmented_train_rmse':float(np.sqrt(loss_sum/len(tr))*ys),'validation_rmse_diagnostic':rmse(y[va],np.array(val_pred)),'lr_after_step':schedule.get_last_lr()[0]})
            if epoch>=epochs-5:
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                if avg is None:avg={k:v/5 for k,v in state.items()}
                else:
                    for k in avg:avg[k]+=state[k]/5
            if (epoch+1)%10==0:print(json.dumps({'fold':fold,**curve[-1],'seconds':time.perf_counter()-ft}),flush=True)
            if time.perf_counter()-start>1800:raise RuntimeError('30 minute total budget exceeded; stop without promotion')
        model.load_state_dict(avg);model.eval()
        def predict(indices):
            result=[]
            with torch.no_grad():
                for f in range(0,len(indices),32):
                    v=torch.from_numpy((X[indices[f:f+32]]-mu)/max(sd,1e-6)).to(d);result.extend((model(v).cpu().numpy()*ys+ym).tolist())
            return np.array(result)
        oof[va]=predict(va);train_pred=predict(tr);sync(d)
        # CPU portable tensors only; no custom pickle classes or optimizer state.
        torch.save({'state_dict':avg,'normalization':{'x_mean':mu,'x_std':sd,'y_mean':ym,'y_std':ys},'fold':fold,'train_ids':[ids[i] for i in tr]},_repo_io(out/f'fold-{fold}.pt'))
        rec={'fold':fold,'train_rmse':rmse(y[tr],train_pred),'validation_rmse':rmse(y[va],oof[va]),'seconds':time.perf_counter()-ft,'memory':memory(d)};folds.append(rec);curves.append(curve)
        write(out/'progress.json',{'folds':folds,'curves':curves});print(json.dumps(rec),flush=True)
        del model,opt,avg,state;gc.collect()
        if d=='mps':torch.mps.empty_cache()
    with (_repo_path(ROOT, 'artifacts/run-005/development_oof.csv')).open() as f:old={r['ID']:float(r['stable_hybrid']) for r in csv.DictReader(f)}
    baseline=np.array([old[s] for s in ids])
    with (_repo_path(ROOT, 'artifacts/run-006/development_oof.csv')).open() as f:grows={r['ID']:float(r['grain_ridge100']) for r in csv.DictReader(f)}
    grain=np.array([grows[s] for s in ids]);preds={'baseline':baseline,'cnn':oof,'fixed_three':.5*baseline+.25*grain+.25*oof}
    scores={n:{'oof_rmse':rmse(y,p),'fold_rmse':[rmse(y[fid==f],p[fid==f]) for f in range(5)]} for n,p in preds.items()}
    best='fixed_three';improvement=scores['baseline']['oof_rmse']-scores[best]['oof_rmse'];better=sum(a>b for a,b in zip(scores['baseline']['fold_rmse'],scores[best]['fold_rmse']))
    rng=np.random.default_rng(20261003);delta=[]
    for _ in range(2000):
        ix=rng.integers(0,len(y),len(y));delta.append(rmse(y[ix],baseline[ix])-rmse(y[ix],preds[best][ix]))
    with (out/'development_oof.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','hardness',*preds]);w.writerows([s,y[i],*[p[i] for p in preds.values()]] for i,s in enumerate(ids))
    report={**plan,'scores':scores,'best_new':best,'improvement':improvement,'better_folds':better,'advance':None,'folds':folds,'curves':curves,'conditional_improvement_95':np.percentile(delta,[2.5,97.5]).tolist(),'residual_correlation':float(np.corrcoef(y-baseline,y-oof)[0,1]),'elapsed_seconds':time.perf_counter()-start,'thermal_after':thermal(),'uncertainty':'OOF repeatedly used for model exploration; bootstrap conditional on fitted models and selection. No holdout evaluation. MPS exact bitwise reproducibility not guaranteed.'}
    assert sha(basepath)==plan['baseline_model_sha256']
    write(out/'report.json',report);print(json.dumps({k:report[k] for k in ['scores','advance','elapsed_seconds','conditional_improvement_95']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['bench256','128x80','256x40']);args=p.parse_args()
    if args.mode=='bench256':bench(_repo_path(ROOT, 'artifacts/cnn-benchmark-256'))
    elif args.mode=='128x80':cv(_repo_path(ROOT, 'artifacts/run-009'),_repo_path(ROOT, 'artifacts/cnn-benchmark-001/benchmark.json'),128,80)
    else:cv(_repo_path(ROOT, 'artifacts/run-010'),_repo_path(ROOT, 'artifacts/cnn-benchmark-256/benchmark.json'),256,40)
