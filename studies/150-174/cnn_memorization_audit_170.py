"""Reuse-only CNN memorization/augmentation audit; no new training or candidate."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import json,csv,time,hashlib,math,resource
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image
from threadpoolctl import threadpool_limits
import cnn_controls as engine
import label_density_169 as density
R=_RepoPath(__file__).resolve().parents[1];O=_repo_path(R, 'artifacts/cnn-memorization-170');SOURCE=_repo_path(R, 'artifacts/cnn-orientation-129');ARMS=['D4','identity'];SOURCES={}
torch.set_num_threads(2);threadpool_limits(2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):SOURCES[str(p.relative_to(R))]=sha(p);return json.loads(p.read_text())
def write(p,z):
    with p.open('x') as f:json.dump(z,f,indent=2,allow_nan=False)
def npz(p,h=None):
    if h:assert sha(p)==h
    SOURCES[str(p.relative_to(R))]=sha(p)
    with np.load(_repo_io(p)) as z:return {k:z[k].copy() for k in z.files}
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
def predict(model,images,norm,rot=0,flip=False,device='mps',batch=32):
    model.eval();out=[]
    with torch.inference_mode():
        for st in range(0,len(images),batch):
            v=(images[st:st+batch]-norm['x_mean'])/max(norm['x_std'],1e-6);v=np.rot90(v,rot,axes=(2,3));v=v[:,:,:,::-1] if flip else v
            p=model(torch.from_numpy(np.ascontiguousarray(v)).to(device)).cpu().numpy()*norm['y_std']+norm['y_mean'];out.extend(p.tolist())
    return np.array(out)
def functional(state,x):
    h=x
    for i in range(4):
        j=3*i;h=F.conv2d(h,state[f'features.{j}.weight'],state[f'features.{j}.bias'],stride=2,padding=2 if i==0 else 1);h=F.group_norm(h,4,state[f'features.{j+1}.weight'],state[f'features.{j+1}.bias'],eps=1e-5);h=F.silu(h)
    stats=torch.cat([h.mean((2,3)),(h.var((2,3),correction=0)+1e-6).sqrt(),x.mean((2,3)),(x.var((2,3),correction=0)+1e-6).sqrt()],1)
    return F.linear(F.silu(F.linear(stats,state['head.0.weight'],state['head.0.bias'])),state['head.3.weight'],state['head.3.bias']).squeeze(1)
def main():
    start=time.perf_counter();assert torch.backends.mps.is_available();O.mkdir(exist_ok=False)
    # Lock sample IDs BEFORE reading any y/error/image cache.
    manifest=read(_repo_path(R, 'artifacts/run-001/split_manifest.json'));ids=sorted(manifest['development_folds']);assert len(ids)==400
    selected=sorted(ids,key=lambda s:hashlib.sha256(('170:'+s).encode()).hexdigest())[:32]
    plan={'study':170,'purpose':'Trainingpipeline/capacity sanity diagnosis only; notcandidate/generalizationexperiment.','new_training':False,'reason_to_reuse':'129alreadyfits320trainingimages toRMSE.30656 withsameCNN/800updates byremovingD4only, whiledropout.15/wd.01remain. Stronger fitting-capability evidence thannew32imageunregularizedfit. Exact32/no-regularizationexperimentwasnotperformed andisnotclaimed.','selected32_ids':selected,'selection_rule':'SmallestSHA256(170:+ID)amongexisting400developmentIDs; lockedbeforelabels/errorsread; selectionnotusedfortraining.','mapping':'Compareexisting400-only run009CSV labels,129OOF,018OOF,166labels byID. Verify32rawimageSHA/256gray/PILBOX128 cache equality. Do not readmixed500labelCSV/holdoutlabels. Priorupstream provenance remains source fororiginalgroundtruth mapping.','reuse':'10saved129models,5folds×D4/identity,80epochs,800updates,last5average; sameinit/batches/LR/dropoutdraws accordingaudited129. TrainingstatsfromeachT320. Current169 MSEcontrols alreadyreproducedsameD4weights.','additional_inference':'Bothmodels onownT320 underall8exactD4views in evalmode/dropoutoff. Reportidentity andmean8view squaredloss,notbestview oraveragedpredictorcandidate. IndependentCPUfunctionalCNN forward+MPS/CPUreload; identicaltrain/validationnorm; no newtraining.','subgroups':'TrainKDEq20density sparse/dense;166persistentlarge33. Posthoc explanatory groups, notfit/selection. Trackoverlapping1600trainrowevaluations,not1600uniqueimages.','curve_limits':'Savedonline80epochbatchloss includesdropout,changingweights andD4whereapplied; compareasdescriptiveonly. All8view inference usesfixedfinalaveragedmodel, no newclean80epochtrajectory claimed.','budget':'10cachedCNNmodelreplays,32rawimagechecks,existingCPU2/MPS,stopiftotalinference>180sec orRSS/driver>3GB. No install/download/cost/full500/test/holdout/newCSV.','protected':read(_repo_path(R, 'artifacts/cubical-h1-168/plan.json'))['protected']}
    write(_repo_path(O, 'plan.json'),plan)
    diag169=read(_repo_path(R, 'reports/label-density-169-summary.json'));summary129=read(_repo_path(R, 'reports/cnn-orientation-129-summary.json'));summary128=read(_repo_path(R, 'reports/cnn-update-budget-128-summary.json'));val=read(_repo_path(SOURCE, 'validation-original.json'));assert val['passed'] and val['report_sha256']==sha(_repo_path(SOURCE, 'original/report.json'));report=read(_repo_path(SOURCE, 'original/report.json'));z=npz(_repo_path(SOURCE, 'original/predictions.npz'),report['predictions_sha256']);assert z['ids'].tolist()==ids;y=z['y'];fid=z['fold'];np.testing.assert_array_equal(fid,[manifest['development_folds'][i] for i in ids])
    reference=read(_repo_path(R, 'artifacts/bootstrap-selector-144/original/report.json'));p18=npz(_repo_path(R, 'artifacts/bootstrap-selector-144/original/predictions.npz'),reference['predictions_sha256']);np.testing.assert_array_equal(p18['ids'],ids);np.testing.assert_array_equal(p18['y'],y)
    c=npz(_repo_path(R, 'artifacts/failure-review-166/analysis.npz'));np.testing.assert_array_equal(c['ids'],ids);np.testing.assert_array_equal(c['y'],y);large=c['persistent_large'];csvpath=_repo_path(R, 'artifacts/run-009/development_oof.csv');SOURCES[str(csvpath.relative_to(R))]=sha(csvpath)
    with csvpath.open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==400 and {r['ID'] for r in rows}==set(ids);byid={r['ID']:float(r['hardness']) for r in rows};np.testing.assert_array_equal([byid[i] for i in ids],y)
    meta=read(_repo_path(R, 'artifacts/nested-calibration-074/extraction.json'));cache=npz(_repo_path(R, 'artifacts/nested-calibration-074/development_features.npz'),meta['feature_sha256']);assert cache['ids'].tolist()==ids;images=cache['images'];assert images.shape==(400,1,128,128)
    sample=[]
    for ident in selected:
        i=ids.index(ident);path=_repo_path(R, 'data/train', f'{ident}.png');assert sha(path)==meta['image_sha256'][ident]
        with Image.open(_repo_io(path)) as im:assert im.mode=='L' and im.size==(256,256);x=np.asarray(im.resize((128,128),Image.Resampling.BOX),dtype=np.float32)/255
        np.testing.assert_array_equal(x,images[i,0]);sample.append({'ID':ident,'cached_row':i,'label':float(y[i]),'image_sha256':sha(path),'pixel_cache_exact':True})
    PP=[];YY=[];II=[];SP=[];records=[];online=[];maxmps=maxcpu=maxfunctional=0.;weights=[]
    for k in range(5):
        T=np.flatnonzero(fid!=k);V=np.flatnonzero(fid==k);groups=np.array([manifest['group_by_id'][i] for i in ids]);assert not set(groups[T])&set(groups[V]);rho=density.density(y[T],y[T],True);sparse=rho<np.quantile(rho,.2);YY.extend(y[T]);II.extend(T);SP.extend(sparse);pair=[];curves=[]
        for ai,arm in enumerate(ARMS):
            folder=_repo_path(SOURCE, 'original', f'fold-{k}-{arm}');r=read(folder/'report.json');assert r==report['folds'][k]['arms'][arm]
            for file,key in [('model.pt','model_sha256'),('predictions.npz','predictions_sha256'),('streams.npz','streams_sha256')]:assert sha(folder/file)==r[key];SOURCES[str((folder/file).relative_to(R))]=r[key]
            obj=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);assert obj['train_indices']==T.tolist() and obj['validation_indices']==V.tolist() and obj['train_ids']==[ids[i] for i in T];norm=obj['normalization'];assert norm=={'x_mean':float(images[T].mean()),'x_std':float(images[T].std()),'y_mean':float(y[T].mean()),'y_std':float(y[T].std())};assert obj['seed']==20261003+k and obj['epochs']==80 and obj['average_last_epochs']==5
            saved=npz(folder/'predictions.npz',r['predictions_sha256']);np.testing.assert_array_equal(saved['train_indices'],T);np.testing.assert_array_equal(saved['validation_indices'],V)
            model=engine.TinyCNN().to('mps');model.load_state_dict(obj['state_dict']);assert model.head[2].p==.15;train_views=np.stack([predict(model,images[T],norm,rot,flip) for rot in range(4) for flip in [False,True]]);err=float(abs(train_views[0]-saved['train_predictions']).max());assert err<1e-3;maxmps=max(maxmps,err);mps_train_error=err
            vp=predict(model,images[V],norm);assert abs(vp-saved['validation_predictions']).max()<1e-3;np.testing.assert_array_equal(saved['validation_predictions'],z['cnn'][ai,V]);model=model.cpu();cpu=predict(model,images[T],norm,device='cpu');err=float(abs(cpu-saved['train_predictions']).max());assert err<1e-3;maxcpu=max(maxcpu,err)
            # Functional CPU independently specifiesarchitecture; no model.forward reuse.
            xx=torch.tensor((images[T[:32]]-norm['x_mean'])/max(norm['x_std'],1e-6));model.eval()
            with torch.inference_mode():manual=functional(obj['state_dict'],xx).numpy()*norm['y_std']+norm['y_mean']
            ferr=float(abs(manual-cpu[:32]).max());assert ferr<1e-3;maxfunctional=max(maxfunctional,ferr)
            # Useoriginalcachedidentitypredictions for exact publishedscorecomparison.
            train_views[0]=saved['train_predictions'];pair.append(train_views)
            streams=npz(folder/'streams.npz',r['streams_sha256']);assert streams['order'].shape==(800,32);curve=np.sqrt(streams['loss'].reshape(80,10).mean(1))*norm['y_std'];curves.append(curve)
            rec={'fold':k,'arm':arm,'train_identity_RMSE':rmse(y[T],train_views[0]),'train_mean8view_loss_RMSE':float(np.sqrt(np.mean((train_views-y[T])**2))),'mean_withinimage_view_std':float(np.std(train_views,axis=0).mean()),'sparse_identity_RMSE':rmse(y[T][sparse],train_views[0,sparse]),'large_identity_RMSE':rmse(y[T][large[T]],train_views[0,large[T]]),'validation_identity_RMSE_reused':rmse(y[V],saved['validation_predictions']),'online_last10_epochs_RMSE':float(np.sqrt(np.mean(curve[-10:]**2))),'online_prev10_epochs_RMSE':float(np.sqrt(np.mean(curve[-20:-10]**2))),'MPS_identity_max_error':mps_train_error,'CPU_saved_train_max_error':err,'CPU_functional_max_error':ferr};records.append(rec);del model
            assert time.perf_counter()-start<180 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<3e9 and torch.mps.driver_allocated_memory()<3e9
        for field in ['initial_state_sha256','rng_start','rng_end']:
            assert report['folds'][k]['arms']['D4'][field]==report['folds'][k]['arms']['identity'][field]
        PP.append(np.stack(pair));online.append(np.stack(curves));print('REUSED_FOLD',k,'identity_train',[v['train_identity_RMSE'] for v in records[-2:]],'allviews',[v['train_mean8view_loss_RMSE'] for v in records[-2:]],flush=True)
    P=np.concatenate(PP,axis=2);Y=np.array(YY);I=np.array(II);sparse=np.array(SP);masks={'all':np.ones(len(Y),bool),'sparse':sparse,'dense':~sparse,'persistent_large33':large[I]};stats={}
    for name,mask in masks.items():
        stats[name]={'train_row_evaluations_overlapping':int(mask.sum()),'unique_IDs':int(len(set(I[mask].tolist()))),'arms':{arm:{'identity_RMSE':rmse(Y[mask],P[ai,0,mask]),'identity_bias_y_minus_pred':float(np.mean(Y[mask]-P[ai,0,mask])),'identity_max_absolute_error':float(abs(Y[mask]-P[ai,0,mask]).max()),'mean8view_loss_RMSE':float(np.sqrt(np.mean((P[ai][:,mask]-Y[mask])**2))),'mean_withinimage_view_std':float(np.std(P[ai][:,mask],axis=0).mean())} for ai,arm in enumerate(ARMS)}}
    for arm in ARMS:assert abs(stats['all']['arms'][arm]['identity_RMSE']-summary129['pooled_training_RMSE'][arm])<1e-12
    np.savez_compressed(_repo_path(O, 'replay.npz'),train_row_indices=I,train_y=Y,train_sparse=sparse,predictions=P,online_curves=np.stack(online))
    proposal={'status':'proposal_only_not_executed','single_control':'Oneoriginalfold0 paired diagnosis: existingD4/MSE/dropout.15 reference from129/169 versus one freshsameD4 model withdropout0 only. Keepweight_decay.01,architecture,unweightedMSE,seed20261003,800updates,32batch,LRschedule,last5average andsample/D4streamunchanged.','why':'129alreadyisolatesD4removal andshowsmemorizationwithworsevalidation; do notrepeatidentity/noaugmentation asnewcandidate.128alreadyshowsextraupdates reducefiterror butcanhurtvalidation. Remaining explicitdropout interaction withD4 notisolated bythese controls.089dropout study wasDINOhead,notTinyCNN.','scope':'One320train/80developmentvalidation fold,one newfit withcachedcontrol; nofull5fold/resplit/full018candidate. Trainingdiagnostic toallocatefurtherresearch only.','readouts':'Cleanidentity andall8D4view trainRMSE, sparse/large33trainerrors, finalordinary80validationRMSE. Noepochorparameterchoicefromvalidation.','interpretation':'Iftraininglossfallsbutvalidationworsens,retainregularizedrecipe. Iftrainingdoesnotfall,dropoutaloneinsufficient. Neitheroutcome establishesphysicalD4invariance orall400capacity limit.','limit':'Singlefold/singleintervention weakgeneralizationevidence; requiresparentdirection before newwork.'}
    out={'study':170,'status':'completed_reuse_only_no_new_training','same_exact32_unregularized_test_found':False,'substitution':'Existing129320imageidentitytrainRMSE.30656 despite retaineddropout.15/wd.01 establishes fittingcapability; avoidredundant32imagefit.','selected32':sample,'selected32_labelblind_before_read':True,'label_ID_mapping':{'four_existing400onlysources_equal':True,'sources':['run009development_oof.csv','129predictions','018predictions','166analysis'],'original500CSV_read':False,'limit':'Cache-to-cacheconsistency pluspriorupstreamprovenance; notfreshreviewofunreadholdout/source500labels.'},'raw32_pixel_preprocessing_verified':True,'stats':stats,'fold_records':records,'validation':{'passed':True,'models_reused':10,'MPS_saved_train_max_error':maxmps,'CPU_saved_train_max_error':maxcpu,'independent_CPU_functional_max_error':maxfunctional,'normalization_ID_group_scope_verified':True,'all129init_order_budget_control_certificate_reused':sha(_repo_path(SOURCE, 'validation-original.json')),'169control_reproduction_and_labelpoison_certificate':sha(_repo_path(R, 'artifacts/label-density-169/validation-original.json'))},'past_budget_evidence':{'128_train':summary128['pooled_training_RMSE'],'128_validation':summary128['pooled_validation_RMSE'],'limitation':'142/143smallrows400->800 usescompletednormalizedschedule,not320rows800->1600. Cannotclaimlargerdatasetneeds1600.'},'past_augmentation_evidence':{'129_train':summary129['pooled_training_RMSE'],'129_validation':summary129['pooled_validation_RMSE'],'limitation':'D4changestransforminvarianceconstraint/effectiveinputvariety andregularization together; doesnotprovegeneratorinvarianceorincorrectlabels.'},'next_minimal_control':proposal,'interpretation':'SameCNN readilyfitsoriginaltrainingimages includingrare/largeerrorcases whenD4removed, evenwithdropout/weightdecay. Grosslabelalignment/evaluationnormalization orcapacityfailure isnot supported. D4 recipe imposesfit/generalizationtradeoff; trainfitimprovement alonefailedvalidation. Noexclusivecausalallocation toaugmentation/regularization/budget; explicitdropoutnotisolated.','limits':['32images usedonlylabelblindmapping/pixelcheck; no new32fit orclaimallpossiblelabelslearnable.','In-samplememorizationdoesnotvalidate generalization, labelquality, physicalinvariance, or018performanceceiling.','All8views referfixedaveragedmodel evalmode; onlinecurvesincludechangingmodels/dropout andarenotdirectcleanloss trajectories.','GroupNorm retained asarchitecturalnormalization; no claimremovingallnormalization.','Same400historicaladaptation; no new independentvalidation.'],'new_fit_count':0,'new_installs_downloads_cost':0,'holdout_test_full500_CSV_upload':False,'protected':plan['protected'],'sources_sha256':SOURCES,'source_sha256':sha(_repo_path(R, 'cnn_memorization_audit_170.py')),'replay_sha256':sha(_repo_path(O, 'replay.npz')),'seconds':time.perf_counter()-start,'remaining_blockers':[]}
    for file,h in plan['protected'].items():assert sha(_repo_path(R, file))==h,file
    write(_repo_path(O, 'report.json'),out);write(_repo_path(R, 'reports/cnn-memorization-170-summary.json'),out);print('AUDIT170_PASS',json.dumps({'stats':stats,'verification':out['validation'],'seconds':out['seconds'],'new_fits':0}),flush=True)
if __name__=='__main__':main()
