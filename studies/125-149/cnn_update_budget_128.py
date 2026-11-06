"""Paired CNN data-size/update-budget control, development only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,datetime,argparse,resource,hashlib,gc
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import resume_base_learning_curve_127 as previous
import numpy as np,torch
q=previous;e=q.e;cal=q.cal;engine=cal.fv.engine;OUT=_repo_path(ROOT, 'artifacts/cnn-update-budget-128')
ARMS=('small400','small800','large800');BATCH=32;MAX_STEPS=800;SEED=20261003

def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def statehash(m):return cal.h.state_hash(m.state_dict())
def mps_rng_hash():return ah(torch.mps.get_rng_state().cpu().numpy())
def schedule(steps):return .0001+.5*(.002-.0001)*(1+np.cos(np.pi*np.arange(steps+1)/steps))
def streams(n,seed):
    rng=np.random.default_rng(seed+4100);parts=[];count=0
    while count<MAX_STEPS*BATCH:
        part=rng.permutation(n);parts.append(part);count+=n
    order=np.concatenate(parts)[:MAX_STEPS*BATCH].reshape(MAX_STEPS,BATCH)
    rng=np.random.default_rng(seed+4200);rot=rng.integers(0,4,MAX_STEPS);flip=rng.random(MAX_STEPS)<.5
    assert np.ptp(np.bincount(order.ravel(),minlength=n))<=1
    return order,rot,flip

def indices(d,split,k):
    sp=q.spec(split,k,0,0);small=np.array(sp['report']['train_indices']);large=np.flatnonzero(d['fid']!=k);val=np.flatnonzero(d['fid']==k)
    assert len(small) in (142,143) and len(large)==320 and len(val)==80 and set(small)<=set(large) and not set(d['g'][large])&set(d['g'][val])
    return small,large,val

def check():
    previous.check();plan=e.read(_repo_path(OUT, 'plan.json'))
    for f,h in {**plan['sources_sha256'],**plan['inputs_sha256'],**plan['protected_sha256']}.items():assert e.sha(_repo_path(ROOT, f))==h,f
    return plan

def prepare():
    previous.check();assert torch.backends.mps.is_available();OUT.mkdir(exist_ok=False);inputs={}
    for split in e.SPLITS:
        for k in range(5):
            sp=q.spec(split,k,0,0)
            for f in [sp['source_report'],sp['paths']['cnn'],sp['cache']]:inputs[str(f.relative_to(ROOT))]=e.sha(f)
    for f in ['reports/base-learning-curve-127-summary.json','reports/base-learning-curve-127-preservation.json','artifacts/nested-calibration-074/development_features.npz','artifacts/nested-calibration-074/extraction.json']:inputs[f]=e.sha(_repo_path(ROOT, f))
    plan={'study':'128 pairedCNN data-size/updatebudget control','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hypothesis':'127samplecurve changedbothtrainingn andoptimizersteps. Testsmall400vs800updates atfixed142/143rows, andsmall800vslarge800 atmatchedcompute, toseparatebudgetrecipe effectfromdataset-size effect undera matchednew samplingprotocol.','arms':{'small400':{'rows':'112deep-0 within074inner-0','updates':400},'small800':{'rows':'same small subset','updates':800},'large800':{'rows':'entireouter320','updates':800}},'replication':'Original5outerfolds, onefixednestedpathj0/deep0 andcommonseed20261003+outerfold; same80outervalidationimages for3arms. No choiceofsubset/seed/foldfromoutcomes. Iforiginalsmall800MSEstrictlybelow small400 andchecks pass, runbothknownresplitswithidentical15models/partition. No bestseed,epoch,weight orrepresentationsearch.','primary':'Pairedsmall800-minus-small400 untrimmedMSE/RMSE; secondarylarge800-minus-small800 data-sizecontrast. No newdeployablemodel or014composition evaluated. Originalpositivegain iscompute-screen only,not significance.','CNN':'ExistingTinyCNN128BOX (91,905parameters),sameGroupNorm/SiLU/globalmeanstd/dropout.15,AdamWlr.002->.0001,weight_decay.01,gradientclip5,MSE,D4trainingonly; identityinference. Foldtrainpixel/targetmean/std only.','sampling':'Allarmsfull32examplebatches. Concatenate independentlyshuffled completepermutations oftrainindices, carry remainder into nextpermutation. No drop_last,oversizepadding,or partialbatch; boundarybatchmaycontain an ID twice acrosspermutations. EverycompletepasscoversalltrainIDs; exposurecountdiff<=1. Same seed namespace butpermutationsdependonn. Thisdiffersfromlegacyvariablelastbatch142protocol; do notattribute legacy-to-newdifference tosteps.','pairing':'CommonTorchinitializationseed across3arms. Independent NumPy namespaces forsampleorder(seed+4100) andD4(seed+4200). Precompute800step augmentation sequence for everyarm;small400 exactlyprefixofsmall800 order andD4. Fullbatchshapeskeepdropout consumption matched; saveMPSrnghash atstart,400,800; checkcommonprefix.','scheduler':'Explicit per-update LR l_U(t)=.0001+.5*(.002-.0001)*(1+cos(pi*t/U)), t=0..U; optimizerusesl_U(t) atzero-basedupdatet. No epoch-basedscheduler orwarmup. Bothbudgetsfinishatminlr; normalizedprogress identical:l400(t)=l800(2t). The400run isnot anunfinished800schedule. StretchingnormalizedschedulechangesabsoluteLRsequence/integrateddecay; contrastestimatesbudget-with-normalized-schedule recipe,notpurestepcount underidenticalprefixLR. Small800andlarge800haveidenticalentireLRsequence.','weight_average':'Average5states at95%,96.25%,97.5%,98.75%,100%ofupdates:400[380,385,390,395,400],800[760,770,780,790,800]. Same normalizedlate window andnumberofstates; no endpoint/chosencheckpoint fromvalidation.','legacy_control':'Beforemainscreen, reproduceexistingoriginalfold0deep0 CNN80training viaexactcal.fv.cnn_fold withstoredseedoffset200,variablelastbatch,epochcosine,last5epochaverage. Compareold inner71predictions to1e-3 andsource-modelstate to1e-4; newprotocolneverclaimedbyteequivalent tolegacy. Reproductionalso resourcebenchmark; preservecached originalandall failures.','selection_scope':'Noouterlabelsenterfit/scalers/scheduler/seed/earlystop; no validationduringtraining. Outerresultsareonlyresearchscreenbetweentheonefixedbudgetcomparison; not a modelselector fitonglobalOOF. Historical400repeateduse acknowledged.','validation':'FreshMPSandCPUreloadallnewmodels; independentlyverifybatchstream/permutationcoverage,LReveryupdate,averagedcheckpointtimes,trainIDs/stats,commoninit/D4/dropoutRNGprefix. Refitoriginalsmall400withalloutside-traininglabelsperturbed,comparepredictions<=1e-3. Validatebeforefollowup decision.','resource':'FreeMPS/CPU2,existinguvpackages,legacy400updatesplus15screenmodels(total10000updates). Projectscreenfromlegacytime*25<600sec;RSS<3GB,MPSdriver<3GB. Eachnewfit<180sec. Ifscreenpositive, twoidentical15modelpartitions. No downloads/install/paidcompute/holdout/test/full500/submission.','uncertainty':'10000pairedIDbootstrapseed20271109; withresplits clusterall3predictiontriplets by400uniqueID. Conditionalfixedfitintervals omitadaptivehistory/refitting.','sources_sha256':{f:e.sha(_repo_path(ROOT, f)) for f in ['cnn_update_budget_128.py','validate_cnn_update_budget_128.py','cnn_controls.py','frozen_validation.py']},'inputs_sha256':inputs,'protected_sha256':{str(f.relative_to(ROOT)):e.sha(f) for f in [q.p.CSV,_repo_path(ROOT, 'reports/candidate014-summary.json'),_repo_path(ROOT, 'reports/candidate014-verification.json'),_repo_path(ROOT, 'reports/base-learning-curve-127-preservation.json')]},'cost':0,'uploaded':False}
    e.write(_repo_path(OUT, 'plan.json'),plan);e.emit({'prepared':True,'arms':plan['arms'],'plan_sha256':e.sha(_repo_path(OUT, 'plan.json'))})

def legacy():
    check();d=cal.inputs('original');sp=q.spec('original',0,0,0);rec=sp['report'];T=np.array(rec['train_indices']);V=np.array(rec['validation_indices']);folder=_repo_path(OUT, 'legacy-control');folder.mkdir(exist_ok=False);start=time.perf_counter()
    pp,log=cal.fv.cnn_fold(d['images'],d['y'],T,V,d['ids'],rec['offset'],'mps',folder/'model.pt')
    old=torch.load(_repo_io(sp['paths']['cnn']),weights_only=True,map_location='cpu');new=torch.load(_repo_io(folder/'model.pt'),weights_only=True,map_location='cpu');z=q.a.readnp(sp['cache']);err=float(abs(pp-z['cnn']).max());stateerr=max(float(abs(old['state_dict'][n]-v).max()) for n,v in new['state_dict'].items());assert err<1e-3 and stateerr<1e-4
    out={'passed':True,'legacy_training_reproduced':True,'validation_prediction_max_error':err,'state_max_error':stateerr,'seconds':time.perf_counter()-start,'projected15mainfit_seconds':25*log['seconds'] if 'seconds' in log else 25*(time.perf_counter()-start),'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'MPS_driver_bytes':torch.mps.driver_allocated_memory(),'model_sha256':e.sha(folder/'model.pt'),'log':log}
    out['resource_pass']=out['projected15mainfit_seconds']<600 and out['RSS_bytes']<3e9 and out['MPS_driver_bytes']<3e9;e.write(folder/'report.json',out);assert out['resource_pass'];check();e.emit({k:v for k,v in out.items() if k!='log'})

def predict(model,images,ix,norm,device):
    model.eval();out=[]
    with torch.inference_mode():
        for st in range(0,len(ix),32):
            X=(images[ix[st:st+32]]-norm['x_mean'])/max(norm['x_std'],1e-6);out.extend((model(torch.tensor(X,device=device)).cpu().numpy()*norm['y_std']+norm['y_mean']).tolist())
    return np.array(out)

def fit(d,T,V,steps,seed,folder):
    folder.mkdir(exist_ok=False);start=time.perf_counter();norm={'x_mean':float(d['images'][T].mean()),'x_std':float(d['images'][T].std()),'y_mean':float(d['y'][T].mean()),'y_std':float(d['y'][T].std())};X=(d['images'][T]-norm['x_mean'])/max(norm['x_std'],1e-6);y=(d['y'][T]-norm['y_mean'])/norm['y_std'];order,rot,flip=streams(len(T),seed);lr=schedule(steps);points=np.round(steps*np.array([.95,.9625,.975,.9875,1.])).astype(int);torch.manual_seed(seed);torch.mps.manual_seed(seed);model=engine.TinyCNN().to('mps');initial=statehash(model);rnghash={'0':mps_rng_hash()};opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01);avg=None;losses=[];checkpoint_hashes={}
    e.emit({'training_started':str(folder.relative_to(ROOT)),'rows':len(T),'updates':steps,'seed':seed})
    for t in range(steps):
        model.train();local=order[t];xb=np.rot90(X[local],int(rot[t]),axes=(2,3));xb=xb[:,:,:,::-1] if flip[t] else xb;xb=torch.from_numpy(np.ascontiguousarray(xb)).to('mps');target=torch.tensor(y[local],dtype=torch.float32,device='mps')
        for group in opt.param_groups:group['lr']=float(lr[t])
        opt.zero_grad(set_to_none=True);loss=torch.nn.functional.mse_loss(model(xb),target);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(float(loss.detach().cpu()))
        for group in opt.param_groups:group['lr']=float(lr[t+1])
        if t+1 in points:
            state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()};checkpoint_hashes[str(t+1)]=cal.h.state_hash(state)
            if avg is None:avg={n:v/5 for n,v in state.items()}
            else:
                for n,v in state.items():avg[n]+=v/5
            torch.save(state,_repo_io(folder/f'average-step-{t+1}.pt'))
        if t+1 in (400,800):rnghash[str(t+1)]=mps_rng_hash()
        if (t+1)%200==0:
            e.emit({'updates_done':t+1,'arm':folder.name,'seconds':time.perf_counter()-start});assert time.perf_counter()-start<180
    assert avg is not None and opt.param_groups[0]['lr']==float(lr[-1]);model.load_state_dict(avg);pp=predict(model,d['images'],V,norm,'mps');tp=predict(model,d['images'],T,norm,'mps');engine.sync('mps');payload={'state_dict':avg,'normalization':norm,'train_ids':[d['ids'][i] for i in T],'validation_ids':[d['ids'][i] for i in V],'train_indices':T.tolist(),'validation_indices':V.tolist(),'seed':seed,'updates':steps,'batch_size':32,'initial_state_sha256':initial,'architecture':'cnn_controls.TinyCNN','average_steps':points.tolist()};torch.save(payload,_repo_io(folder/'model.pt'))
    with (folder/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=T,validation_indices=V,train_predictions=tp,validation_predictions=pp)
    with (folder/'streams.npz').open('xb') as f:np.savez_compressed(_repo_io(f),order=order[:steps],rot=rot[:steps],flip=flip[:steps],lr=lr,loss_standardized=np.array(losses))
    rec={'n_train':len(T),'updates':steps,'seed':seed,'train_RMSE':e.rmse(d['y'][T],tp),'validation_RMSE':e.rmse(d['y'][V],pp),'initial_state_sha256':initial,'rng_state_sha256':rnghash,'sample_order_sha256':ah(order[:steps]),'augmentation_sha256':ah(np.column_stack([rot[:steps],flip[:steps]])),'average_state_sha256':cal.h.state_hash(avg),'average_steps':points.tolist(),'checkpoint_state_sha256':checkpoint_hashes,'model_sha256':e.sha(folder/'model.pt'),'predictions_sha256':e.sha(folder/'predictions.npz'),'streams_sha256':e.sha(folder/'streams.npz'),'seconds':time.perf_counter()-start};e.write(folder/'report.json',rec);del model,opt,avg;gc.collect();torch.mps.empty_cache();e.emit({'training_finished':folder.name,'updates':steps,'seconds':rec['seconds']});return pp,rec

def run(split):
    check();assert e.read(_repo_path(OUT, 'legacy-control/report.json'))['resource_pass']
    if split!='original':assert e.read(_repo_path(OUT, 'screen-decision.json'))['advance']
    d=cal.inputs(split);path=_repo_path(OUT, split);path.mkdir(exist_ok=False);P=np.full((3,400),np.nan);records=[];start=time.perf_counter()
    for k in range(5):
        small,large,V=indices(d,split,k);fold=[]
        for i,arm in enumerate(ARMS):
            T=large if arm=='large800' else small;steps=400 if arm=='small400' else 800;pred,rec=fit(d,T,V,steps,SEED+k,path/f'fold-{k}-{arm}');P[i,V]=pred;fold.append(rec)
        assert len(set(x['initial_state_sha256'] for x in fold))==1;assert len(set(x['rng_state_sha256']['0'] for x in fold))==1;assert len(set(x['rng_state_sha256']['400'] for x in fold))==1;assert fold[1]['rng_state_sha256']['800']==fold[2]['rng_state_sha256']['800'];records.append({'fold':k,'arms':dict(zip(ARMS,fold))})
    with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],predictions=P)
    scores={arm:e.rmse(d['y'],P[i]) for i,arm in enumerate(ARMS)};rec={'split':split,'RMSE':scores,'small_budget_gain':scores['small400']-scores['small800'],'matched800_data_gain':scores['small800']-scores['large800'],'folds':records,'predictions_sha256':e.sha(path/'predictions.npz'),'seconds':time.perf_counter()-start};e.write(path/'report.json',rec);check();e.emit({k:v for k,v in rec.items() if k!='folds'})

def decide():
    check();r=e.read(_repo_path(OUT, 'original/report.json'));v=e.read(_repo_path(OUT, 'validation-original.json'));assert v['passed'] and v['report_sha256']==e.sha(_repo_path(OUT, 'original/report.json'));d={'advance':r['small_budget_gain']>0,'rule':'Originalsmall800 untrimmedMSEstrictlybelow small400,afterindependentvalidation; nofoldcount/CI/minimumgain gate.','small_budget_RMSE_gain':r['small_budget_gain'],'candidate_selection':False};e.write(_repo_path(OUT, 'screen-decision.json'),d);e.emit(d)
if __name__=='__main__':
    torch.set_num_threads(2);ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','legacy','decide',*e.SPLITS]);args=ap.parse_args();{'prepare':prepare,'legacy':legacy,'decide':decide}.get(args.mode,lambda:run(args.mode))()
