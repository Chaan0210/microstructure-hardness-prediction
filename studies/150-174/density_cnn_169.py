"""Single mildly density-weighted CNN versus paired ordinaryMSE control."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='2'
import sys,time,json,gc,resource,hashlib
from pathlib import Path
import numpy as np,torch
from threadpoolctl import threadpool_limits
import cnn_update_budget_128 as prior
import label_density_169 as density
R=_RepoPath(__file__).resolve().parents[1];OUT=_repo_path(R, 'artifacts/label-density-169');e=prior.e;cal=prior.cal;engine=prior.engine;ARMS=('mse','density');SEED=20261003;BLEND=.2
torch.set_num_threads(2);threadpool_limits(2)
def write(p,z):
    with p.open('x') as f:json.dump(z,f,indent=2,allow_nan=False)
def oldpath(split,k):return _repo_path(R, 'artifacts/run-009', f'fold-{k}.pt') if split=='original' else _repo_path(R, 'artifacts/frozen-validation-021', f'seed-{split}', f'fold-{k}', f'cnn-seed{SEED}.pt')
def load(split):
    rec=e.read(_repo_path(R, 'artifacts/bootstrap-selector-144', split, 'report.json'));p=_repo_path(R, 'artifacts/bootstrap-selector-144', split, 'predictions.npz');assert e.sha(p)==rec['predictions_sha256']
    with np.load(_repo_io(p)) as z:ids=z['ids'].tolist();y=z['y'].copy();fid=z['fold'].copy();ref=z['primary'].copy()
    man=e.read(_repo_path(R, 'artifacts/run-001/split_manifest.json'));assert ids==sorted(man['development_folds']) and len(ids)==400;g=np.array([man['group_by_id'][i] for i in ids]);splitman=man if split=='original' else e.read(_repo_path(R, f'artifacts/frozen-validation-021/seed-{split}/split_manifest.json'));np.testing.assert_array_equal(fid,[splitman['development_folds'][i] for i in ids])
    p=_repo_path(R, 'artifacts/nested-calibration-074/development_features.npz');assert e.sha(p)==e.read(p.with_name('extraction.json'))['feature_sha256']
    with np.load(_repo_io(p)) as z:assert z['ids'].tolist()==ids;images=z['images'].copy()
    with np.load(_repo_path(R, 'artifacts/cnn-seed-mean-133/audit', f'{split}.npz')) as z:assert z['ids'].tolist()==ids;np.testing.assert_array_equal(y,z['y']);legacy=z['seeds'][0].copy()
    assert images.shape==(400,1,128,128);return {'ids':ids,'y':y,'fid':fid,'g':g,'images':images,'reference018':ref,'legacy':legacy}
def prepare():
    assert torch.backends.mps.is_available();diag=e.read(_repo_path(OUT, 'diagnostic.json'));assert diag['large33_sparse_in_all3']==26
    files=['density_cnn_169.py','validate_density_cnn_169.py','label_density_169.py','cnn_controls.py','cnn_update_budget_128.py','artifacts/label-density-169/diagnostic.json','artifacts/label-density-169/diagnostic-plan.json','artifacts/nested-calibration-074/development_features.npz','artifacts/nested-calibration-074/extraction.json','artifacts/cnn-seed-mean-133/audit/report.json','reports/optimization-057-058-summary.json','reports/ranking-objective-120-summary.json']
    for split in e.SPLITS:
        files.extend([f'artifacts/bootstrap-selector-144/{split}/predictions.npz',f'artifacts/bootstrap-selector-144/{split}/report.json',f'artifacts/cnn-seed-mean-133/audit/{split}.npz'])
        files.extend(str(oldpath(split,k).relative_to(R)) for k in range(5))
    plan={'study':169,'hypothesis':'A mild training-only targetdensity weighting canreduce CNN scarce-region optimization bias and improve ordinaryRMSE/complementarity, despite weakcausal diagnostic. Not assumed true.','evidence':'26of33persistentlarge cases inlowtrainingdensity inall3splits; CNN train sparseRMSE14.34 vsall9.83. However018predictioncalibration slopes1.009-1.051,withinIDdensity-errorr=-.025; DINO fitstrain nearlyexactly. Commonerrornotestablished asdensitycaused or universaloptimizerunderfit.','nonduplication':'638files inspected; no density/LDSperexampleweights.058MSEvsSmoothL1/120ranking/051hardbinaux/131RNC notdensityweight. CurrentCNN alreadyMSE.','density':'ContinuousGaussianKDE bw=.4*trainlabelpopulationSD, train leave-one-out densities. mediantrainrho/rho exponent1/4; densityfloor.1median; rawclip[.8,1.25]; dividefulltrainrawmean. Weightmax/min<=1.5625. Originalactualweights~.90-1.20,ESS~317/320. No histogram quantilebins, no labelnoiseestimate, notexactpublishedLDSreplication.','loss':'mean(w_i*(pred_i-standardized_y_i)^2), fixedglobaltrainnormalization; no perbatchweightsum normalization/resampling or weightsatprediction. Ordinary control torchMSE.','paired_recipe':'Existing91,905parameterTinyCNN128BOX,80epochs/800updates,32batch,AdamW.002,wd.01,clip5,epochcosine to.0001,last5weightaverage;D4train,identityinfer. Seed20261003+fold. Identicalinit/order/D4/dropoutRNG/budget inpair. Pixel/targetmeanstd computedunweightedonlyT. No validationduringtraining/checkpointselection.','endpoint':'Unweightedwhole400RMSE primary. Both standaloneCNN andfixed.8*complete018+.2*newCNN computed forweighted/MSE. This fixedadditionalblend preserves018; it isnot replacingCNNinside018withstalecalibrators. No learnedglobalOOFweight or newnestedcalibrator.','gate':'OriginalweightedstandaloneRMSE<MSE AND weighted20%blendRMSE<both018andMSE20%blend, >=3/5 weightedblendfoldgainsvs018, validationpass. Onlythenbothknownresplits; pooleduntrimmedMSEbelowallmatchedreferences andreporteverypartition. No rescueweight/seed/lossstrength/architecturesearch.','subgroups':'TrainKDEq20threshold sparse/dense, trainyq10/q90 tails, outoftrainrange,166large33. Showcounts/bias/unweightedRMSE incldense/tailharms. Labelsusedonlypostfitdiagnostics,nevermodelinput.','resources':'ExistingCPU2/MPS. Firstunweightedfit reusedbenchmark10fits<600sec,RSS/MPSdriver<3GB,eachfit<180sec. Original10fits+1weightedlabelpoisonreplay; conditionalresplits20fits. No newinstall/download/cost.','validation':'EverymodelCPU/MPSreload,trainIDs/stats,independentKDE/weightsandlossgradient,800pairedstreams/LR and5averagedstates. MSEcontrolmustreproduceoldcheckpoint to1e-4weights/1e-3prediction. Wholeweightedfold outside80labels+10000 replay modelstateexact/predtol1e-3.','scope':'400developmentonly,noholdout/test/full500/newCSV/upload. Preserve018/ZIP.','limitations':['LossweightsdependonY andchangeconditional-meanestimand; raretailgaincancostgeneralRMSE.','Same400adaptiveexploration; no causal proof/no independentvalidation.','Onefixedseedpairperfold; no robustnessclaim acrossnewseeds.','Existingtrainingresiduals reflect in-samplefit, not validation; ytailtrendhasselectioncomponent.'],'source_sha256':{v:e.sha(_repo_path(R, v)) for v in files},'protected':e.read(_repo_path(R, 'artifacts/cubical-h1-168/plan.json'))['protected']}
    write(_repo_path(OUT, 'plan.json'),plan);print('PLAN_LOCKED',e.sha(_repo_path(OUT, 'plan.json')),flush=True)
def check():
    p=e.read(_repo_path(OUT, 'plan.json'))
    for f,h in {**p['source_sha256'],**p['protected']}.items():assert e.sha(_repo_path(R, f))==h,f
    return p

def fit(d,T,V,k,arm,folder):
    folder.mkdir(exist_ok=False);start=time.perf_counter();seed=SEED+k;rng=np.random.default_rng(seed);torch.manual_seed(seed)
    norm={'x_mean':float(d['images'][T].mean()),'x_std':float(d['images'][T].std()),'y_mean':float(d['y'][T].mean()),'y_std':float(d['y'][T].std())}
    X=(d['images'][T]-norm['x_mean'])/max(norm['x_std'],1e-6);y=(d['y'][T]-norm['y_mean'])/norm['y_std'];w,wm=density.weights(d['y'][T]);w=w if arm=='density' else np.ones(len(T));model=engine.TinyCNN().to('mps');initial=prior.statehash(model);rng0=prior.mps_rng_hash();opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=80,eta_min=.0001)
    avg=None;orders=[];rots=[];flips=[];lrs=[];losses=[];unweighted=[];hashes={};states={};sentinels={};step=0
    print('FIT_START',folder.name,arm,'weights',float(w.min()),float(w.max()),flush=True)
    for epoch in range(80):
        model.train();order=rng.permutation(len(T))
        for first in range(0,len(T),32):
            ix=order[first:first+32];rot=int(rng.integers(4));flip=bool(rng.random()<.5);xb=np.rot90(X[ix],rot,axes=(2,3));xb=xb[:,:,:,::-1] if flip else xb;xb=np.ascontiguousarray(xb)
            if step in (0,99,399,799):hashes[str(step)]=prior.ah(xb)
            orders.append(ix.copy());rots.append(rot);flips.append(flip);lrs.append(opt.param_groups[0]['lr']);xb=torch.from_numpy(xb).to('mps');target=torch.tensor(y[ix],dtype=torch.float32,device='mps');weight=torch.tensor(w[ix],dtype=torch.float32,device='mps')
            opt.zero_grad(set_to_none=True);pred=model(xb);mse=torch.nn.functional.mse_loss(pred,target);loss=(weight*(pred-target)**2).mean() if arm=='density' else mse;assert torch.isfinite(loss)
            if step in (0,399,799):sentinels[str(step)]={'prediction':pred.detach().cpu().tolist(),'target':target.cpu().tolist(),'weight':weight.cpu().tolist(),'loss':float(loss.detach().cpu())}
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(float(loss.detach().cpu()));unweighted.append(float(mse.detach().cpu()));step+=1
        sched.step()
        if epoch>=75:
            state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()};states[str(epoch+1)]=cal.h.state_hash(state);torch.save(state,_repo_io(folder/f'epoch-{epoch+1}.pt'))
            if avg is None:avg={n:v/5 for n,v in state.items()}
            else:
                for n,v in state.items():avg[n]+=v/5
        if (epoch+1)%20==0:print('EPOCH',folder.name,epoch+1,round(time.perf_counter()-start,2),flush=True);assert time.perf_counter()-start<180
    rngend=prior.mps_rng_hash();model.load_state_dict(avg);vp=prior.predict(model,d['images'],V,norm,'mps');tp=prior.predict(model,d['images'],T,norm,'mps');engine.sync('mps')
    torch.save({'state_dict':avg,'normalization':norm,'train_indices':T.tolist(),'validation_indices':V.tolist(),'train_ids':[d['ids'][i] for i in T],'validation_ids':[d['ids'][i] for i in V],'seed':seed,'epochs':80,'average_last_epochs':5,'arm':arm,'architecture':'cnn_controls.TinyCNN'},_repo_io(folder/'model.pt'))
    np.savez_compressed(_repo_io(folder/'predictions.npz'),train_indices=T,validation_indices=V,train_predictions=tp,validation_predictions=vp)
    np.savez_compressed(_repo_io(folder/'streams.npz'),order=np.stack(orders),rot=np.array(rots),flip=np.array(flips),lr=np.array(lrs),loss=np.array(losses),unweighted_loss=np.array(unweighted))
    np.savez_compressed(_repo_io(folder/'weights.npz'),T=T,y=d['y'][T],weights=w,density=np.array(wm['density']),raw_weights=np.array(wm['raw_weights']))
    rec={'arm':arm,'fold':k,'seed':seed,'updates':step,'n_train':len(T),'train_RMSE':e.rmse(d['y'][T],tp),'validation_RMSE':e.rmse(d['y'][V],vp),'initial_state_sha256':initial,'rng_start':rng0,'rng_end':rngend,'average_state_sha256':cal.h.state_hash(avg),'checkpoint_state_sha256':states,'actual_input_sha256':hashes,'loss_sentinels':sentinels,'weight_summary':{k:v for k,v in wm.items() if k not in ['density','raw_weights']},'model_sha256':e.sha(folder/'model.pt'),'predictions_sha256':e.sha(folder/'predictions.npz'),'streams_sha256':e.sha(folder/'streams.npz'),'weights_sha256':e.sha(folder/'weights.npz'),'seconds':time.perf_counter()-start,'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'MPS_driver_bytes':torch.mps.driver_allocated_memory()};write(folder/'report.json',rec);del model,opt,avg;gc.collect();torch.mps.empty_cache();print('FIT_DONE',folder.name,rec['train_RMSE'],rec['validation_RMSE'],rec['seconds'],flush=True);return vp,rec

def run(split):
    check()
    if split!='original':assert e.read(_repo_path(OUT, 'original/report.json'))['advance'] and e.read(_repo_path(OUT, 'validation-original.json'))['passed']
    d=load(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=True);P=np.full((2,400),np.nan);records=[];start=time.perf_counter()
    for k in range(5):
        T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);assert len(T)==320 and len(V)==80 and not set(d['g'][T])&set(d['g'][V]);pair=[]
        for ai,arm in enumerate(ARMS):
            folder=out/f'fold-{k}-{arm}'
            if (folder/'report.json').exists():r=e.read(folder/'report.json');assert e.sha(folder/'predictions.npz')==r['predictions_sha256'];p=np.load(_repo_io(folder/'predictions.npz'))['validation_predictions']
            else:p,r=fit(d,T,V,k,arm,folder)
            P[ai,V]=p;pair.append(r)
            if arm=='mse':
                old=torch.load(_repo_io(oldpath(split,k)),map_location='cpu',weights_only=True);new=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);err=float(abs(p-d['legacy'][V]).max());serr=max(float(abs(v-old['state_dict'][n]).max()) for n,v in new['state_dict'].items());assert err<1e-3 and serr<1e-4,(err,serr)
                if not (out/f'legacy-{k}.json').exists():write(out/f'legacy-{k}.json',{'passed':True,'prediction_max_error':err,'state_max_error':serr,'old_checkpoint_sha256':e.sha(oldpath(split,k))})
            if split=='original' and k==0 and ai==0 and not (_repo_path(OUT, 'benchmark.json')).exists():
                b={'first_fit_seconds':r['seconds'],'projected10fits_seconds':10*r['seconds'],'RSS_bytes':r['RSS_bytes'],'MPS_driver_bytes':r['MPS_driver_bytes']};b['passed']=b['projected10fits_seconds']<600 and b['RSS_bytes']<3e9 and b['MPS_driver_bytes']<3e9;write(_repo_path(OUT, 'benchmark.json'),b);assert b['passed']
        for key in ['initial_state_sha256','rng_start','rng_end','actual_input_sha256']:assert pair[0][key]==pair[1][key],key
        records.append({'fold':k,'arms':dict(zip(ARMS,pair))});print('PAIRED_FOLD_DONE',split,k,flush=True)
    blend=.8*d['reference018'][None,:]+.2*P;scores={'reference018':e.rmse(d['y'],d['reference018']),'mse_CNN':e.rmse(d['y'],P[0]),'density_CNN':e.rmse(d['y'],P[1]),'mse_blend':e.rmse(d['y'],blend[0]),'density_blend':e.rmse(d['y'],blend[1])};gain=[e.rmse(d['y'][d['fid']==k],d['reference018'][d['fid']==k])-e.rmse(d['y'][d['fid']==k],blend[1,d['fid']==k]) for k in range(5)]
    advance=scores['density_CNN']<scores['mse_CNN'] and scores['density_blend']<min(scores['reference018'],scores['mse_blend']) and sum(g>0 for g in gain)>=3
    np.savez_compressed(_repo_io(out/'predictions.npz'),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],reference018=d['reference018'],cnn=P,blend=blend)
    r={'split':split,'RMSE':scores,'fold_gains_weighted_blend_vs018':gain,'advance':advance,'folds':records,'predictions_sha256':e.sha(out/'predictions.npz'),'seconds':time.perf_counter()-start};write(out/'report.json',r);check();print('RESULT',json.dumps({k:v for k,v in r.items() if k!='folds'}),flush=True)
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':prepare()
    else:run(mode)
