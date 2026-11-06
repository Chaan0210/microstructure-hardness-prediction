"""One fixed identity-versus-D4 CNN control; development400 only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,datetime,argparse,gc,resource,hashlib
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cnn_update_budget_128 as prior
import numpy as np,torch
q=prior.q;e=prior.e;cal=prior.cal;engine=prior.engine
OUT=_repo_path(ROOT, 'artifacts/cnn-orientation-129');ARMS=('D4','identity');SEED=20261003

def check():
    prior.check();p=e.read(_repo_path(OUT, 'plan.json'))
    for name,hh in {**p['sources'],**p['inputs'],**p['protected']}.items():assert e.sha(_repo_path(ROOT, name))==hh,name
    return p

def prepare():
    prior.check();assert torch.backends.mps.is_available();OUT.mkdir(exist_ok=False)
    inputs={}
    names=['reports/problem-definition-111-summary.json','reports/dino-046-summary.json','reports/input-noise-110-summary.json','reports/cnn-controls-summary.json','reports/cnn-structure-035-summary.json','reports/research-batch-105-108-summary.json','reports/cnn-update-budget-128-summary.json','reports/base-learning-curve-127-summary.json','reports/cnn-update-budget-128-preservation.json','artifacts/nested-calibration-074/development_features.npz','artifacts/nested-calibration-074/extraction.json']
    for name in names:inputs[name]=e.sha(_repo_path(ROOT, name))
    for split in e.SPLITS:
        for k in range(5):
            sp=q.spec(split,k,2,0)
            for f in [sp['source_report'],sp['paths']['cnn']]:inputs[str(f.relative_to(ROOT))]=e.sha(f)
    p={'study':'129 removal of unverified CNN D4 label-preservation assumption','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hypothesis':'Synthetic hardness proxy has no documented D4 invariance guarantee in the inspected official evidence recorded111. Current CNN assigns unchanged labels to rotated/reflected training images. Removing this augmentation may preserve useful absolute-orientation information. This tests a training assumption, not proof that the generator depends on direction.','novelty':{'111':'Identified lack of invariance guarantee but did not train a removal control.','009_011_035_106_128':'CNN size/pooling/amplitude/update-budget trials retained D4. No CNN identity-only counterpart found in inspected training sources and research summaries.','046':'DINO D4 averaged frozen features, not removal of train-time CNN D4.','110':'Added weak noise to DINO input, not orientation control.','057_088_089_090':'Not another head epoch/dropout/seed/width regularization sweep.'},'arms':list(ARMS),'configuration':'Existing TinyCNN128 BOX; same 91,905 parameters, GroupNorm/SiLU, global feature and input mean/std, hidden64 dropout.15. Exactly80epochs,320train/80validation,800updates,32rows/batch,no partial/drop_last; AdamW .002,wd.01,clip5,MSE,epoch cosine to.0001,last5epoch weight average. Identity inference for both.','pairing':'Same seed20261003+fold and legacy single NumPy RNG. BOTH arms draw the same permutation and each batch rotation/reflection variables; identity discards transform after drawing, so future sample order and RNG consumption stay equal. Torch initialization and dropout RNG states checked. Only pixel transform differs.','normalization':'Pixel scalar mean/std and target mean/std from320foldtrain only; no per-image contrast normalization, new scaler/PCA or label-mixing.','primary':'Identity versus D4 standalone untrimmed OOF MSE. Secondary prespecified raw4way=.3compact26+.3CNN+.2Smean3+.2Bmean3 (same honest cached outer predictions). This is NOT full014: affine/shortB/physics-residual excluded, no stale downstream refits claimed.','gate':'Original5fold: after validation, advance ONLY if identity has strictly lower standalone AND raw4way MSE. Then run both existing resplits with unchanged recipe. Further integration justified only if both endpoints improve in each ofall3partitions. No minimum/fold-count/CI gate, no rescue blend or alternate augmentation. Positive is compute screen, not significance.','scope_after_positive':'This study assesses the base branch and raw mixture. Any full014 replacement needs new nested identity CNN predictions and refit affine/residual pipeline; never directly replace with old calibrators. No production/full500/test/submission in129.','validation':'Fresh D4 control must reproduce frozen legacy CNN predictions<1e-3,state<1e-4. Reload every model MPS/CPU<1e-3; independently reconstruct RNG orders, per-update LR,5weight-average states,train normalization and group/ID exclusion. Verify actual input hashes atsteps0,99,399,799. Refit identity fold0 with all80outer targets changed; compare states/predictions.','resource':'Existinguv CPU2/MPS,10mainfits original then onlyifpositive20more; firstD4 usedasbenchmark,reused. Projection10fits<600sec,RSS/MPSdriver each<3GB,eachfit<180sec. No installs/downloads/externaldata/payment.','uncertainty':'10000 paired400-ID bootstrap seed20271111; resplit predictions clustered bysame400IDs. Conditionalfixedfit interval excludes repeated historical selection and refitting. Onepairedseed per fold; no physical-invariance conclusion from prediction differences.','sources':{n:e.sha(_repo_path(ROOT, n)) for n in ['cnn_orientation_129.py','validate_cnn_orientation_129.py','frozen_validation.py','cnn_controls.py']},'inputs':inputs,'protected':{str(f.relative_to(ROOT)):e.sha(f) for f in [q.p.CSV,_repo_path(ROOT, 'reports/candidate014-summary.json'),_repo_path(ROOT, 'reports/candidate014-verification.json'),_repo_path(ROOT, 'reports/readme-update-128.json')]},'cost':0,'uploaded':False}
    e.write(_repo_path(OUT, 'plan.json'),p);e.emit({'prepared':True,'plan_sha256':e.sha(_repo_path(OUT, 'plan.json')),'hypothesis':p['hypothesis'],'gate':p['gate']})

def fit(d,T,V,k,arm,folder):
    folder.mkdir(exist_ok=False);start=time.perf_counter();seed=SEED+k;rng=np.random.default_rng(seed);torch.manual_seed(seed)
    norm={'x_mean':float(d['images'][T].mean()),'x_std':float(d['images'][T].std()),'y_mean':float(d['y'][T].mean()),'y_std':float(d['y'][T].std())}
    X=(d['images'][T]-norm['x_mean'])/max(norm['x_std'],1e-6);y=(d['y'][T]-norm['y_mean'])/norm['y_std'];model=engine.TinyCNN().to('mps');init=prior.statehash(model);rng0=prior.mps_rng_hash();opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=80,eta_min=.0001)
    avg=None;orders=[];rots=[];flips=[];lrs=[];losses=[];input_hash={};states={};step=0
    e.emit({'fit_started':str(folder.relative_to(ROOT)),'arm':arm,'n':len(T)})
    for epoch in range(80):
        model.train();order=rng.permutation(len(T))
        for first in range(0,len(T),32):
            ix=order[first:first+32];rot=int(rng.integers(4));flip=bool(rng.random()<.5);xb=X[ix]
            if arm=='D4':
                xb=np.rot90(xb,rot,axes=(2,3))
                if flip:xb=xb[:,:,:,::-1]
            else:assert arm=='identity'
            xb=np.ascontiguousarray(xb)
            if step in (0,99,399,799):input_hash[str(step)]=prior.ah(xb)
            orders.append(ix.copy());rots.append(rot);flips.append(flip);lrs.append(opt.param_groups[0]['lr']);xb=torch.from_numpy(xb).to('mps');target=torch.tensor(y[ix],dtype=torch.float32,device='mps')
            opt.zero_grad(set_to_none=True);loss=torch.nn.functional.mse_loss(model(xb),target);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(float(loss.detach().cpu()));step+=1
        scheduler.step()
        if epoch>=75:
            state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()};states[str(epoch+1)]=cal.h.state_hash(state);torch.save(state,_repo_io(folder/f'epoch-{epoch+1}.pt'))
            if avg is None:avg={n:v/5 for n,v in state.items()}
            else:
                for n,v in state.items():avg[n]+=v/5
        if (epoch+1)%20==0:e.emit({'arm':arm,'fold':k,'epoch':epoch+1,'seconds':time.perf_counter()-start});assert time.perf_counter()-start<180
    rngend=prior.mps_rng_hash();model.load_state_dict(avg);vp=prior.predict(model,d['images'],V,norm,'mps');tp=prior.predict(model,d['images'],T,norm,'mps');engine.sync('mps')
    payload={'state_dict':avg,'normalization':norm,'train_indices':T.tolist(),'validation_indices':V.tolist(),'train_ids':[d['ids'][i] for i in T],'validation_ids':[d['ids'][i] for i in V],'seed':seed,'epochs':80,'average_last_epochs':5,'arm':arm,'architecture':'cnn_controls.TinyCNN'};torch.save(payload,_repo_io(folder/'model.pt'))
    with (folder/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=T,validation_indices=V,train_predictions=tp,validation_predictions=vp)
    with (folder/'streams.npz').open('xb') as f:np.savez_compressed(_repo_io(f),order=np.stack(orders),rot=np.array(rots),flip=np.array(flips),lr=np.array(lrs),loss=np.array(losses))
    rec={'arm':arm,'fold':k,'seed':seed,'updates':step,'n_train':len(T),'train_RMSE':e.rmse(d['y'][T],tp),'validation_RMSE':e.rmse(d['y'][V],vp),'initial_state_sha256':init,'rng_start':rng0,'rng_end':rngend,'average_state_sha256':cal.h.state_hash(avg),'checkpoint_state_sha256':states,'actual_input_sha256':input_hash,'model_sha256':e.sha(folder/'model.pt'),'predictions_sha256':e.sha(folder/'predictions.npz'),'streams_sha256':e.sha(folder/'streams.npz'),'seconds':time.perf_counter()-start,'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'MPS_driver_bytes':torch.mps.driver_allocated_memory()};e.write(folder/'report.json',rec);del model,opt,avg;gc.collect();torch.mps.empty_cache();return vp,rec

def run(split):
    check()
    if split!='original':assert e.read(_repo_path(OUT, 'screen-decision.json'))['advance']
    d=cal.inputs(split);old=q.a.load(split);assert old['ids']==d['ids'];np.testing.assert_array_equal(old['y'],d['y']);np.testing.assert_array_equal(old['fid'],d['fid']);path=_repo_path(OUT, split);path.mkdir(exist_ok=False);P=np.full((2,400),np.nan);records=[];start=time.perf_counter()
    for k in range(5):
        T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);assert len(T)==320 and len(V)==80 and not set(d['g'][T])&set(d['g'][V]);fold=[]
        for ai,arm in enumerate(ARMS):
            pred,rec=fit(d,T,V,k,arm,path/f'fold-{k}-{arm}');P[ai,V]=pred;fold.append(rec)
            if arm=='D4':
                saved=torch.load(_repo_io(q.spec(split,k,2,0)['paths']['cnn']),map_location='cpu',weights_only=True);new=torch.load(_repo_io(path/f'fold-{k}-{arm}/model.pt'),map_location='cpu',weights_only=True);err=float(abs(pred-old['P'][V,2]).max());stateerr=max(float(abs(v-saved['state_dict'][n]).max()) for n,v in new['state_dict'].items());assert err<1e-3 and stateerr<1e-4,(err,stateerr);e.write(path/f'fold-{k}-legacy.json',{'passed':True,'prediction_max_error':err,'state_max_error':stateerr})
            if split=='original' and k==0 and ai==0:
                bench={'projected10fit_seconds':rec['seconds']*10,'RSS_bytes':rec['RSS_bytes'],'MPS_driver_bytes':rec['MPS_driver_bytes'],'first_fit_reused':True};bench['passed']=bench['projected10fit_seconds']<600 and bench['RSS_bytes']<3e9 and bench['MPS_driver_bytes']<3e9;e.write(_repo_path(OUT, 'benchmark.json'),bench);assert bench['passed']
        assert fold[0]['initial_state_sha256']==fold[1]['initial_state_sha256'] and fold[0]['rng_start']==fold[1]['rng_start'] and fold[0]['rng_end']==fold[1]['rng_end'];records.append({'fold':k,'arms':dict(zip(ARMS,fold))})
    components=old['P'][:,1:];raw=np.stack([.3*components[:,0]+.3*P[ai]+.2*components[:,2]+.2*components[:,3] for ai in range(2)])
    with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],cnn=P,raw=raw,old_components=components)
    scores={arm:{'CNN_RMSE':e.rmse(d['y'],P[ai]),'raw4way_RMSE':e.rmse(d['y'],raw[ai])} for ai,arm in enumerate(ARMS)};rec={'split':split,'scores':scores,'folds':records,'predictions_sha256':e.sha(path/'predictions.npz'),'seconds':time.perf_counter()-start};e.write(path/'report.json',rec);check();e.emit({'split':split,'completed':True,'scores':scores,'seconds':rec['seconds']})

def decide():
    check();r=e.read(_repo_path(OUT, 'original/report.json'));v=e.read(_repo_path(OUT, 'validation-original.json'));assert v['passed'] and v['report_sha256']==e.sha(_repo_path(OUT, 'original/report.json'));s=r['scores'];g={m:s['D4'][m]-s['identity'][m] for m in ['CNN_RMSE','raw4way_RMSE']};result={'advance':all(x>0 for x in g.values()),'gains':g,'rule':'Both standalone CNN and prespecified raw4way strictly improve, after independent validation.','new_candidate':False};e.write(_repo_path(OUT, 'screen-decision.json'),result);e.emit(result)
if __name__=='__main__':
    torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','decide',*e.SPLITS]);x=p.parse_args();{'prepare':prepare,'decide':decide}.get(x.mode,lambda:run(x.mode))()
