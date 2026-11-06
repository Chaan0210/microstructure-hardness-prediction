"""One CNN dropout-only control, fold0 screened before remaining original folds."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='2'
import sys,json,time,gc,resource,hashlib
from pathlib import Path
import numpy as np,torch
from threadpoolctl import threadpool_limits
import cnn_orientation_129 as old
import density_cnn_169 as loader
import cnn_memorization_audit_170 as audit
import label_density_169 as density
ROOT=_RepoPath(__file__).resolve().parents[1];OUT=_repo_path(ROOT, 'artifacts/cnn-dropout-171');prior=old.prior;cal=old.cal;engine=old.engine;e=old.e;SEED=20261003
threadpool_limits(2);torch.set_num_threads(2)
def write(p,z):
    with p.open('x') as f:json.dump(z,f,indent=2,allow_nan=False)
def ref(k):return old.OUT/'original'/f'fold-{k}-D4'
def load():return loader.load('original')
def prepare():
    assert torch.backends.mps.is_available();OUT.mkdir(exist_ok=False);sources=['dropout_cnn_171.py','validate_dropout_cnn_171.py','cnn_orientation_129.py','cnn_controls.py','density_cnn_169.py','cnn_memorization_audit_170.py','label_density_169.py','reports/cnn-memorization-170-summary.json','artifacts/cnn-orientation-129/validation-original.json','artifacts/cnn-orientation-129/original/report.json','artifacts/cnn-memorization-170/replay.npz','artifacts/cnn-memorization-170/report.json','artifacts/bootstrap-selector-144/original/predictions.npz','artifacts/bootstrap-selector-144/original/report.json','artifacts/failure-review-166/analysis.npz','artifacts/nested-calibration-074/development_features.npz','artifacts/nested-calibration-074/extraction.json']
    for k in range(5):sources.extend(str((ref(k)/file).relative_to(ROOT)) for file in ['report.json','model.pt','predictions.npz','streams.npz'])
    p={'study':171,'hypothesis':'Dropout.15 maylimit fitofaugmentedCNN; removingit alone mayimproveD4-trainingfit withoutlosingvalidation/complementarity. Singlefixedtest,notassuminggeneralizationbenefit.','factor':'Only nn.Dropout inhead after194->64Linear/SiLU changesp.15->0. Same91,905trainableparameters. GroupNorm/inputtargetnormalization unchanged.','training':'Both usemodel.train(); batchD4 rotation/reflection appliedbeforeforward; dropout.15 isstochasticinhead,p0identity. Allinference model.eval(), bothdropoutidentity. D4 anddropoutdifferent operations.','paired_recipe':'Exactold129fitfunction withp0assignment andmetadataonly. seed20261003+k,80epochs/800updates,batch32,AdamW.002,weightdecay.01,clip5,epochcosineto.0001,last5weightaverage. ExistingoriginalT320/V80 IDs. UnweightedMSE. No label-densityweights.','RNG':'SameTorchinit andNumPysample/D4stream. Removingdropout consumesnoMPSdropoutRNG; endMPSRNGneednotmatchp.15. NootherstochasticTorchlayers. No demandformeaninglessidenticaldropoutmasks whenp0.','control':'Reuseverified129D4 checkpoints/traces, independentlyreproducedby169MSE. Recomputepredictions,verifyinit/inputhashes/order/LR/norm/labels/scopes; no duplicatemaincontroltraining.','normalization':'OnlyT320unweightedpixelmean/std andlabelmean/std. Cached400images128BOX with170raw32checks andmappedlabels. No newimage preprocessing.','screen_fixed_before_training':'Fold0 ONLY first. Allaudits pass AND p0 finalmean8D4view trainRMSE<p.15 AND p0identityvalidation80RMSE<p.15 AND fixed.8*full018+.2*p0CNN validationRMSE<matchedp.15blend. Allstrictgains,nostatisticalsignificanceclaim. Ifnot,stopwithoutfold1-4 oralternativep/wd/seed/augmentation.','after_positive_screen':'Runfold1-4samep0recipeandvalidate. Reportpooled400unweightedRMSE,foldgains,allsubgroups,D4consistency. UsefuldevelopmentcandidateonlyifCNNbelowp.15 AND fixed20%blendbelowboth018andmatchedcontrol. No resplits/full500/test/CSV/upload authorizedin171.','subgroups':'TrainKDEq20thresholds for sparse/dense and166large33. All80validationrowsprimary,subgroupsdescriptiveonly. Labelsusedtodeterminepostfitgroups,neverinput/predictor. Showharmsaswellasgains.','D4_diagnostic':'Fixedfinalaveragedmodel evalmode: identitytrain/validation,all8D4mean squaredloss,meanwithinimagepredictionstd. NoTTAaveragedpredictorcandidate orbestviewselection.','validation':'CPU/MPSreloadandindependentfunctionalCNN; trainIDs/norm/paramcount; .15trainrandom/evalidentityandp0train/evalequalfixtures;full800sample/D4/LRstreams/4inputsentinels;5stateaverage;fold0freshp0refitafterchangingall80outsideTlabels mustpreservestate/predictions.','resources':'ExistingCPU2/MPS;onefold~15sec,cap180sec,RSS/driver<3GB;atmost5mainp0fitsplus1poisonreplay. Free,nonewinstall/download.','limits':['Screenfold0alreadyreusedinmanyresearchsteps; expansionnotindependentconfirmation.','Onepairedseedperfold; trainfitisnotsynonymouswithgeneralization.','Negativefixedrecipe doesnotdisproveeverydropoutmechanism; no sweep.','Complete018fixedblend isnotreplacingitsCNNwithstalecalibration.'],'protected':e.read(_repo_path(ROOT, 'artifacts/cubical-h1-168/plan.json'))['protected'],'source_sha256':{v:e.sha(_repo_path(ROOT, v)) for v in sources}}
    write(_repo_path(OUT, 'plan.json'),p);print('PLAN_LOCKED',e.sha(_repo_path(OUT, 'plan.json')),flush=True)
def check():
    p=e.read(_repo_path(OUT, 'plan.json'))
    for f,h in {**p['source_sha256'],**p['protected']}.items():assert e.sha(_repo_path(ROOT, f))==h,f
    return p

def mode_fixture():
    check();d=load();T=np.flatnonzero(d['fid']!=0);obj=torch.load(_repo_io(ref(0)/'model.pt'),map_location='cpu',weights_only=True);m=engine.TinyCNN();m.load_state_dict(obj['state_dict']);assert sum(p.numel() for p in m.parameters())==91905;assert len([v for v in m.modules() if isinstance(v,torch.nn.Dropout)])==1
    norm=obj['normalization'];xx=torch.tensor((d['images'][T[:8]]-norm['x_mean'])/norm['x_std']);m.eval()
    with torch.inference_mode():
        eval15=m(xx);m.train();torch.manual_seed(171);train15a=m(xx);train15b=m(xx);assert not torch.equal(train15a,train15b);m.head[2].p=0.;train0=m(xx);m.eval();eval0=m(xx);assert torch.equal(train0,eval0) and torch.equal(eval15,eval0)
        h=torch.ones((32,64));dr=torch.nn.Dropout(.15).train();out=dr(h);assert (out==0).any() and (out!=0).any();np.testing.assert_allclose(out[out!=0].numpy(),1/.85,atol=1e-7,rtol=0);dr.eval();assert torch.equal(dr(h),h)
    r={'passed':True,'dropout_location':'head.2 betweenSiLUandLinear64->1','parameter_count':91905,'only_one_dropout_module':True,'p15_train_output_changes_between_calls':True,'p15_eval_equals_p0_eval_sameweights':True,'p0_train_equals_eval_sameweights':True,'inverted_dropout_nonzero_scale':1/.85,'GroupNorm_remains':True};write(_repo_path(OUT, 'mode-fixture.json'),r);print('MODE_FIXTURE_PASS',json.dumps(r),flush=True)

def fit(d,T,V,k,arm,folder):
    folder.mkdir(exist_ok=False);start=time.perf_counter();seed=SEED+k;rng=np.random.default_rng(seed);torch.manual_seed(seed)
    norm={'x_mean':float(d['images'][T].mean()),'x_std':float(d['images'][T].std()),'y_mean':float(d['y'][T].mean()),'y_std':float(d['y'][T].std())}
    X=(d['images'][T]-norm['x_mean'])/max(norm['x_std'],1e-6);y=(d['y'][T]-norm['y_mean'])/norm['y_std'];model=engine.TinyCNN().to('mps');model.head[2].p=0.;init=prior.statehash(model);rng0=prior.mps_rng_hash();opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=80,eta_min=.0001)
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
    payload={'state_dict':avg,'normalization':norm,'train_indices':T.tolist(),'validation_indices':V.tolist(),'train_ids':[d['ids'][i] for i in T],'validation_ids':[d['ids'][i] for i in V],'seed':seed,'epochs':80,'average_last_epochs':5,'arm':arm,'dropout':0.,'architecture':'cnn_controls.TinyCNN'};torch.save(payload,_repo_io(folder/'model.pt'))
    with (folder/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=T,validation_indices=V,train_predictions=tp,validation_predictions=vp)
    with (folder/'streams.npz').open('xb') as f:np.savez_compressed(_repo_io(f),order=np.stack(orders),rot=np.array(rots),flip=np.array(flips),lr=np.array(lrs),loss=np.array(losses))
    rec={'dropout':0.,'arm':arm,'fold':k,'seed':seed,'updates':step,'n_train':len(T),'train_RMSE':e.rmse(d['y'][T],tp),'validation_RMSE':e.rmse(d['y'][V],vp),'initial_state_sha256':init,'rng_start':rng0,'rng_end':rngend,'average_state_sha256':cal.h.state_hash(avg),'checkpoint_state_sha256':states,'actual_input_sha256':input_hash,'model_sha256':e.sha(folder/'model.pt'),'predictions_sha256':e.sha(folder/'predictions.npz'),'streams_sha256':e.sha(folder/'streams.npz'),'seconds':time.perf_counter()-start,'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'MPS_driver_bytes':torch.mps.driver_allocated_memory()};e.write(folder/'report.json',rec);del model,opt,avg;gc.collect();torch.mps.empty_cache();return vp,rec

def run(k):
    check();assert e.read(_repo_path(OUT, 'mode-fixture.json'))['passed']
    if k>0:assert e.read(_repo_path(OUT, 'screen.json'))['advance'] and e.read(_repo_path(OUT, 'fold-0/validation.json'))['passed']
    d=load();T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);assert len(T)==320 and len(V)==80 and not set(d['g'][T])&set(d['g'][V]);path=_repo_path(OUT, f'fold-{k}');pp,r=fit(d,T,V,k,'D4',path);assert r['RSS_bytes']<3e9 and r['MPS_driver_bytes']<3e9 and r['seconds']<180
    oldr=e.read(ref(k)/'report.json');assert r['initial_state_sha256']==oldr['initial_state_sha256'] and r['rng_start']==oldr['rng_start'];assert r['actual_input_sha256']==oldr['actual_input_sha256']
    with np.load(_repo_io(ref(k)/'streams.npz')) as a,np.load(_repo_io(path/'streams.npz')) as b:
        for key in ['order','rot','flip','lr']:np.testing.assert_array_equal(a[key],b[key])
    print('TRAINING_COMPLETE',k,r['train_RMSE'],r['validation_RMSE'],flush=True)

def stats(y,p,mask):
    if not mask.any():return {'n':0}
    e=y[mask]-p[mask];return {'n':int(mask.sum()),'RMSE':float(np.sqrt(np.mean(e*e))),'bias_y_minus_pred':float(e.mean())}
def evaluate(k):
    check();assert e.read(_repo_path(OUT, f'fold-{k}/validation.json'))['passed'];d=load();T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);train=[];valid=[]
    for path,drop in [(ref(k),.15),(_repo_path(OUT, f'fold-{k}'),0.)]:
        obj=torch.load(_repo_io(path/'model.pt'),map_location='cpu',weights_only=True);model=engine.TinyCNN().to('mps');model.head[2].p=drop;model.load_state_dict(obj['state_dict']);norm=obj['normalization']
        train.append(np.stack([audit.predict(model,d['images'][T],norm,r,f) for r in range(4) for f in [False,True]]));valid.append(np.stack([audit.predict(model,d['images'][V],norm,r,f) for r in range(4) for f in [False,True]]));del model
    train=np.stack(train);valid=np.stack(valid)
    # Reusedcontrol mustmatch all8own-trainingview predictions from170.
    p=_repo_path(ROOT, 'artifacts/cnn-memorization-170');rr=e.read(p/'report.json');assert e.sha(p/'replay.npz')==rr['replay_sha256']
    with np.load(_repo_io(p/'replay.npz')) as z:np.testing.assert_array_equal(z['train_row_indices'][k*320:(k+1)*320],T);np.testing.assert_allclose(train[0],z['predictions'][0,:,k*320:(k+1)*320],rtol=0,atol=1e-3)
    with np.load(_repo_path(ROOT, 'artifacts/failure-review-166/analysis.npz')) as z:assert z['ids'].tolist()==d['ids'];large=z['persistent_large'].copy()
    rho=density.density(d['y'][T],d['y'][T],True);q20=np.quantile(rho,.2);smT=rho<q20;smV=density.density(d['y'][T],d['y'][V])<q20;groups={};yp=d['y'];ref18=d['reference018'][V];blend=.8*ref18[None,:]+.2*valid[:,0]
    for phase,ix,P,sp in [('train',T,train,smT),('validation',V,valid,smV)]:
        groups[phase]={}
        for name,mask in [('all',np.ones(len(ix),bool)),('sparse',sp),('dense',~sp),('large33',large[ix])]:
            groups[phase][name]={arm:{**stats(yp[ix],P[ai,0],mask),'all8view_RMSE':float(np.sqrt(np.mean((P[ai][:,mask]-yp[ix][mask])**2))) if mask.any() else None,'mean_view_std':float(np.std(P[ai][:,mask],axis=0).mean()) if mask.any() else None} for ai,arm in enumerate(['dropout15','dropout0'])}
            if phase=='validation':groups[phase][name].update({'reference018':stats(yp[ix],ref18,mask),'blend15':stats(yp[ix],blend[0],mask),'blend0':stats(yp[ix],blend[1],mask)})
    scores={'CNN15':e.rmse(yp[V],valid[0,0]),'CNN0':e.rmse(yp[V],valid[1,0]),'reference018':e.rmse(yp[V],ref18),'blend15':e.rmse(yp[V],blend[0]),'blend0':e.rmse(yp[V],blend[1]),'train8view15':float(np.sqrt(np.mean((train[0]-yp[T])**2))),'train8view0':float(np.sqrt(np.mean((train[1]-yp[T])**2)))}
    np.savez_compressed(_repo_path(OUT, f'fold-{k}/evaluation.npz'),T=T,V=V,train_views=train,validation_views=valid,blend=blend,reference018=ref18,train_sparse=smT,validation_sparse=smV)
    rr={'fold':k,'scores':scores,'groups':groups,'evaluation_sha256':e.sha(_repo_path(OUT, f'fold-{k}/evaluation.npz')),'validation_report_sha256':e.sha(_repo_path(OUT, f'fold-{k}/validation.json')),'train_q20_density':float(q20)};write(_repo_path(OUT, f'fold-{k}/evaluation.json'),rr)
    if k==0:
        gate={'advance':scores['train8view0']<scores['train8view15'] and scores['CNN0']<scores['CNN15'] and scores['blend0']<scores['blend15'],'checks':{'train8view_improves':scores['train8view0']<scores['train8view15'],'validation_CNN_improves':scores['CNN0']<scores['CNN15'],'matched018blend_improves':scores['blend0']<scores['blend15']},'scores':scores,'single_fold_is_not_confirmatory':True,'criterion_unchanged':True};write(_repo_path(OUT, 'screen.json'),gate);print('SCREEN',json.dumps(gate),flush=True)
    print('EVALUATION',json.dumps({'fold':k,'scores':scores,'groups':groups}),flush=True);check()
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':prepare();mode_fixture()
    elif mode.startswith('eval-'):evaluate(int(mode[5:]))
    else:run(int(mode))
