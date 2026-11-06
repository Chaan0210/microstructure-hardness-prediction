"""Predeclared small CNN structure and matched-resolution experiments."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
import sys,json,csv,time,gc,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, '.venv')).resolve():raise SystemExit('Use project uv .venv')
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from torch import nn
import joblib
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import make_pipeline
import cnn_controls as engine
import frozen_validation as fv
from cnn_texture_fusion import CovarianceCNN,covariance_descriptor
OUT=_repo_path(ROOT, 'artifacts/cnn-structure-035')
VARIANTS=('tiny128','cov128','covmorph128','tiny256')
EPOCHS=80
SEED=20261003

def make_model(name):
    return CovarianceCNN(name=='covmorph128') if name.startswith('cov') else engine.TinyCNN()
def call(model,name,x,morph):return model(x,morph) if name=='covmorph128' else model(x)
def load_data():
    ids,y,groups,Xbase,recipes,weights,panels,provenance=fv.read_inputs(['run-012'])
    data={128:[],256:[]};hashes={}
    for ident in ids:
        p=_repo_path(ROOT, 'data/train', f'{ident}.png')
        with Image.open(_repo_io(p)) as im:
            assert im.mode=='L' and im.size==(256,256)
            data[256].append(np.asarray(im,dtype=np.float32)/255.)
            data[128].append(np.asarray(im.resize((128,128),Image.Resampling.BOX),dtype=np.float32)/255.)
        hashes[ident]=fv.sha(p)
    data={s:np.stack(v)[:,None] for s,v in data.items()}
    return ids,y,groups,Xbase,recipes,weights,panels['run-012'],data,{**provenance,'development_image_sha256':hashes}
def morphology_scale(Z,tr):
    scale=make_pipeline(MinMaxScaler(clip=True),StandardScaler()).fit(Z[tr])
    return scale,scale.transform(Z).astype(np.float32)
def unit_checks():
    torch.manual_seed(20261115)
    z=torch.randn(1,3,2,2,dtype=torch.double,requires_grad=True);idx=torch.triu_indices(3,3);w=torch.where(idx[0]==idx[1],1.,2.**.5).double()
    assert torch.autograd.gradcheck(lambda a:covariance_descriptor(a,idx[0],idx[1],w),(z,),eps=1e-6,atol=1e-4)
    for name in VARIANTS:
        m=make_model(name);x=torch.zeros(2,1,128,128);f=torch.zeros(2,26);p=call(m,name,x,f);loss=(p-torch.tensor([-.3,.7])).square().mean();loss.backward()
        assert torch.isfinite(p).all() and all(torch.isfinite(t.grad).all() for t in m.parameters() if t.grad is not None)
    return {'covariance_cpu_double_gradcheck':True,'zero_input_forward_backward_all_variants_finite':True,'parameters':{n:sum(p.numel() for p in make_model(n).parameters()) for n in VARIANTS}}
def benchmark():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(2)
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; report before long CPU training')
    checks=unit_checks();ids,y,groups,Xbase,recipes,weights,panel,images,provenance=load_data()
    sp=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text());train=np.array([j for j,i in enumerate(ids) if sp['development_folds'][i]!=0]);ix=train[:32];Z=panel['X'];scale,morph=morphology_scale(Z,train)
    results={};device='mps'
    for name in VARIANTS:
        size=256 if name=='tiny256' else 128;X=images[size];mu=float(X[train].mean());sd=float(X[train].std());ym=float(y[train].mean());ys=float(y[train].std());torch.manual_seed(SEED);model=make_model(name).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
        xb=torch.from_numpy((X[ix]-mu)/max(sd,1e-6)).to(device);mb=torch.from_numpy(morph[ix]).to(device);target=torch.tensor((y[ix]-ym)/ys,dtype=torch.float32,device=device);times=[];losses=[]
        for step in range(13):
            engine.sync(device);start=time.perf_counter();opt.zero_grad(set_to_none=True);loss=nn.functional.mse_loss(call(model,name,xb,mb),target);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();engine.sync(device)
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if step>=3:times.append(time.perf_counter()-start);losses.append(float(loss.detach().cpu()))
        results[name]={'median_step_seconds':float(np.median(times)),'measured_steps':len(times),'finite_loss_and_gradients':True,'last_loss':losses[-1],'memory':engine.memory(device),'estimated_five_fold_train_seconds':float(np.median(times))*10*80*5}
        del model,opt,xb,mb,target;gc.collect();torch.mps.empty_cache()
    report={'device':device,'torch':torch.__version__,'checks':checks,'variants':results,'thermal':engine.thermal(),'thermal_limit':'pmset may not expose sensors; timings and allocator memory do not measure temperature.','data_scope':'Only original fold0 training320 used for normalization/benchmark;32 training images benchmarked, no validation-driven selection.','source_sha256':{n:fv.sha(_repo_path(ROOT, n)) for n in ('cnn_structure_experiment.py','cnn_texture_fusion.py','cnn_controls.py','frozen_validation.py','compact_regression.py')},'feature_cache_sha256':fv.sha(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'))}
    fv.write(_repo_path(OUT, 'benchmark.json'),report);fv.write(_repo_path(OUT, 'input-provenance.json'),provenance);print(json.dumps(report),flush=True)

def predict(model,name,X,morph,indices,mu,sd,ym,ys,device):
    out=[];model.eval()
    with torch.no_grad():
        for first in range(0,len(indices),32):
            ix=indices[first:first+32];xb=torch.from_numpy((X[ix]-mu)/max(sd,1e-6)).to(device);mb=torch.from_numpy(morph[ix]).to(device)
            out.extend((call(model,name,xb,mb).cpu().numpy()*ys+ym).tolist())
    return np.array(out)

def train_fold(name,X,Z,y,tr,va,ids,fold,device,dest):
    start=time.perf_counter();rng=np.random.default_rng(SEED+fold);torch.manual_seed(SEED+fold)
    mu,sd=float(X[tr].mean()),float(X[tr].std());ym,ys=float(y[tr].mean()),float(y[tr].std());assert sd>0 and ys>0
    scale,morph=morphology_scale(Z,tr);Xt=(X[tr]-mu)/max(sd,1e-6);yt=(y[tr]-ym)/ys;model=make_model(name).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=.0001);avg=None;curves=[]
    for epoch in range(EPOCHS):
        model.train();order=rng.permutation(len(tr));sse=0.
        for first in range(0,len(tr),32):
            local=order[first:first+32];global_ix=tr[local];xb=np.rot90(Xt[local],int(rng.integers(4)),axes=(2,3))
            if rng.random()<.5:xb=xb[:,:,:,::-1]
            xb=torch.from_numpy(np.ascontiguousarray(xb)).to(device);mb=torch.from_numpy(morph[global_ix]).to(device);target=torch.tensor(yt[local],dtype=torch.float32,device=device)
            opt.zero_grad(set_to_none=True);loss=nn.functional.mse_loss(call(model,name,xb,mb),target)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite train loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();sse+=float(loss.detach().cpu())*len(local)
        schedule.step();record={'epoch':epoch+1,'augmented_train_rmse':float(np.sqrt(sse/len(tr))*ys),'lr':schedule.get_last_lr()[0]}
        # Diagnostic curve only; endpoint remains predeclared80/last5average.
        if (epoch+1)%10==0:record['validation_rmse_diagnostic']=fv.rmse(y[va],predict(model,name,X,morph,va,mu,sd,ym,ys,device))
        curves.append(record)
        if epoch>=75:
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if avg is None:avg={k:v/5 for k,v in state.items()}
            else:
                for k in avg:avg[k]+=state[k]/5
        if (epoch+1)%20==0:print(json.dumps({'variant':name,'fold':fold,**record,'seconds':time.perf_counter()-start}),flush=True)
        if time.perf_counter()-start>600:raise RuntimeError('Fold exceeds10min budget; report before extending')
    model.load_state_dict(avg);model.eval();vp=predict(model,name,X,morph,va,mu,sd,ym,ys,device);tp=predict(model,name,X,morph,tr,mu,sd,ym,ys,device);engine.sync(device)
    checkpoint={'state_dict':avg,'architecture':name,'seed':SEED+fold,'image_size':X.shape[-1],'epochs':EPOCHS,'normalization':{'x_mean':mu,'x_std':sd,'y_mean':ym,'y_std':ys},'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va]}
    torch.save(checkpoint,_repo_io(dest/f'{name}.pt'));joblib.dump(scale,_repo_io(dest/f'{name}-morph-scaler.joblib'))
    # Fresh-model reload verifies nonpersistent covariance indices and portable state.
    reloaded=make_model(name).to(device);reloaded.load_state_dict(torch.load(_repo_io(dest/f'{name}.pt'),map_location='cpu',weights_only=True)['state_dict']);rp=predict(reloaded,name,X,morph,va,mu,sd,ym,ys,device);np.testing.assert_allclose(rp,vp,rtol=0,atol=2e-4)
    record={'train_rmse':fv.rmse(y[tr],tp),'validation_rmse':fv.rmse(y[va],vp),'seconds':time.perf_counter()-start,'curve':curves,'memory':engine.memory(device),'reload_max_abs_difference':float(np.max(abs(rp-vp))),'parameters':sum(p.numel() for p in model.parameters())}
    del model,reloaded,opt,avg;gc.collect();torch.mps.empty_cache();return vp,record

def run(split):
    bench=json.loads((_repo_path(OUT, 'benchmark.json')).read_text());assert torch.backends.mps.is_available();torch.set_num_threads(2)
    for n,h in bench['source_sha256'].items():assert fv.sha(_repo_path(ROOT, n))==h
    for info in bench['variants'].values():
        if info['median_step_seconds']>.2 or info['memory']['process_peak_rss_bytes']>4_000_000_000:raise RuntimeError('Benchmark outside expected time/RAM envelope; report first')
    dest=_repo_path(OUT, split);dest.mkdir(exist_ok=False);ids,y,groups,Xbase,recipes,weights,panel,images,provenance=load_data();Z=panel['X'];assert fv.sha(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'))==bench['feature_cache_sha256']
    source=_repo_path(ROOT, 'artifacts/run-001/split_manifest.json') if split=='original' else _repo_path(ROOT, f'artifacts/frozen-validation-021/seed-{split}/split_manifest.json')
    sp=json.loads(source.read_text());assert set(sp['development_folds'])==set(ids);fid=np.array([sp['development_folds'][i] for i in ids])
    plan={'split':split,'variants':list(VARIANTS),'hypotheses':{'pooling':'tiny128→cov128 isolates added rank12 covariance statistics','fusion':'cov128→covmorph128 adds26 train-fold-normalized compact features','resolution':'tiny128→tiny256 keeps architecture,seed,optimizer,80epochs,800steps/fold,batch32 equal'},'fixed_schedule':{'seed':SEED,'epochs':80,'batch':32,'optimizer':'AdamW','lr':.002,'weight_decay':.01,'cosine_min_lr':.0001,'last_weight_average':5,'clip_grad_norm':5.},'augmentation':'Same seeded NumPy batch D4 order; no resizecrop/intensity jitter. Compact features are D4-summary features, no augmentation-dependent recomputation.','morphology_scaler':'MinMaxScaler(clip=True)→StandardScaler fit ONLY outer train; joint network uses standardized26inputs','comparison':'All baseline and inner-selected compactKRR components freshly refit per outerfold; all4CNNs freshly trained. Fixed ensemble .5baseline+.25compactKRR+.25CNN. Compare newly trained Tiny128 reference plus recorded003 for drift.','data_scope':'development400 only, same original and two predefined repartitions. No holdout/test or externaldata.','selection':'No earlystop/checkpoint selection by validation curves; no tuned ensemble weights','source_sha256':bench['source_sha256'],'protected':{n:fv.sha(_repo_path(ROOT, n)) for n in ('artifacts/submission-candidate-003.csv','artifacts/final-model-017/report.json')},'split_manifest_sha256':fv.sha(source),'repeated_selection_caveat':fv.CAVEAT}
    fv.write(dest/'plan.json',plan);fv.write(dest/'input-provenance.json',provenance);fv.write(dest/'split_manifest.json',{'development_folds':dict(zip(ids,fid.tolist())),'groups':dict(zip(ids,groups.tolist()))})
    names=['baseline','compact_kernel',*VARIANTS,*[n+'_blend' for n in VARIANTS]];pred={n:np.full(400,np.nan) for n in names};logs=[];started=time.perf_counter()
    for k in range(5):
        tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(groups[tr])&set(groups[va]);d=dest/f'fold-{k}';d.mkdir();t=time.perf_counter()
        fitted={n:clone(m).fit(Xbase[tr],y[tr]) for n,m in recipes.items()};base=sum(weights[n]*m.predict(Xbase[va]) for n,m in fitted.items());pred['baseline'][va]=base;joblib.dump({'models':fitted,'weights':weights,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va]},_repo_io(d/'baseline.joblib'));del fitted
        kp,km,kd=fv.kernel_fold(Z,y,groups,panel['blocks'],tr,va,k);pred['compact_kernel'][va]=kp;joblib.dump({'model':km,'features':panel['names'],'blocks':panel['blocks'],'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va]},_repo_io(d/'kernel.joblib'));del km
        log={'fold':k,'baseline_kernel_seconds':time.perf_counter()-t,'kernel_selection':kd,'models':{},'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va]}
        for name in VARIANTS:
            size=256 if name=='tiny256' else 128;pred[name][va],record=train_fold(name,images[size],Z,y,tr,va,ids,k,'mps',d);pred[name+'_blend'][va]=.5*base+.25*kp+.25*pred[name][va];log['models'][name]=record
            print(json.dumps({'split':split,'fold':k,'completed':name,'cnn_rmse':record['validation_rmse'],'blend_rmse':fv.rmse(y[va],pred[name+'_blend'][va]),'seconds':record['seconds']}),flush=True)
        log['checkpoint_sha256']={p.name:fv.sha(p) for p in d.iterdir()};logs.append(log);fv.write(dest/f'fold-{k}-report.json',log)
        with (dest/'progress.json').open('w') as f:json.dump({'completed_folds':len(logs),'last_fold':k,'elapsed_seconds':time.perf_counter()-started},f)
    # Recorded003 is reporting-only. It never supplies training targets or inputs.
    prior=_repo_path(ROOT, 'artifacts/run-012/development_oof.csv') if split=='original' else _repo_path(ROOT, f'artifacts/frozen-validation-021/seed-{split}/development_oof.csv')
    with prior.open() as f:rows=list(csv.DictReader(f))
    by={r['ID']:r for r in rows};assert len(rows)==len(by)==400 and set(by)==set(ids);np.testing.assert_array_equal(y,[float(by[i]['hardness']) for i in ids]);column='nested_kernel_cnnblend' if split=='original' else 'candidate003';pred['recorded003']=np.array([float(by[i][column]) for i in ids])
    scores={n:{'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]} for n,p in pred.items()};rng=np.random.default_rng(20261116);ix=rng.integers(0,400,size=(2000,400))
    for name in VARIANTS:
        p=pred[name+'_blend'];paired={}
        for reference in ('tiny128_blend','recorded003'):
            ref=pred[reference];delta=np.sqrt(np.mean((y-ref)[ix]**2,axis=1))-np.sqrt(np.mean((y-p)[ix]**2,axis=1));paired[reference]={'rmse_gain':fv.rmse(y,ref)-fv.rmse(y,p),'conditional_ci95':np.percentile(delta,[2.5,97.5]).tolist(),'fold_gains':[fv.rmse(y[fid==k],ref[fid==k])-fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]}
        scores[name+'_blend']['paired']=paired
    with (dest/'development_oof.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','hardness','fold',*pred]);w.writerows([i,y[j],fid[j],*[v[j] for v in pred.values()]] for j,i in enumerate(ids))
    for n,h in plan['source_sha256'].items():assert fv.sha(_repo_path(ROOT, n))==h
    for n,h in plan['protected'].items():assert fv.sha(_repo_path(ROOT, n))==h
    report={'plan':plan,'scores':scores,'folds':logs,'elapsed_seconds':time.perf_counter()-started,'thermal':engine.thermal(),'limits':fv.CAVEAT+' MPS seeded execution is not guaranteed bitwise identical across runs. No fulltrain/test inference.'};fv.write(dest/'report.json',report);print(json.dumps({'split':split,'scores':scores,'elapsed_seconds':report['elapsed_seconds']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['benchmark','original','20261031','20261101']);a=p.parse_args();benchmark() if a.mode=='benchmark' else run(a.mode)
