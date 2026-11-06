"""172 fixed fold0 exact D4 structural experiment; train-only development cache."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
import sys,json,time,math,gc,re,datetime
from pathlib import Path
import numpy as np,torch
import d4_cnn_172 as net
import density_cnn_169 as loader
import dropout_cnn_171 as previous
import label_density_169 as density
R=_RepoPath(__file__).resolve().parents[1];OUT=_repo_path(R, 'artifacts/d4-cnn-172');e=loader.e;engine=loader.engine;prior=loader.prior;SEED=20261003
SOURCE=['d4_cnn_172.py','test_d4_172.py','study_d4_172.py','cnn_controls.py','density_cnn_169.py','cnn_update_budget_128.py','dropout_cnn_171.py','label_density_169.py']
torch.set_num_threads(2)
def write(p,z):
    with p.open('x') as f:json.dump(z,f,indent=2,ensure_ascii=False,allow_nan=False)
def prepare():
    assert torch.backends.mps.is_available();OUT.mkdir(exist_ok=False)
    files=sorted(set([*_study_sources(R),*R.glob('reports/*summary.json'),*R.glob('artifacts/*/plan.json')]))
    pat=re.compile(r'equivariant|group.?conv|weight.?tying|weight.?sharing|등변',re.I);hits={};scanned={}
    for p in files:
        if '172' in str(p.relative_to(R)):continue
        text=p.read_text();scanned[str(p.relative_to(R))]=e.sha(p);lines=[{'line':i+1,'text':x[:500]} for i,x in enumerate(text.splitlines()) if pat.search(x)]
        if lines:hits[str(p.relative_to(R))]=lines
    novelty={'scanned_files':len(scanned),'hashes':scanned,'keyword_hits':hits,'conclusion':'No true D4 group-equivariant learned convolution or group-composition kernel tying found. Reviewed129 augmentation/removal,151 viewmean inference and directional/statistical descriptors; distinct from learned orientation-mixing group convolution. Keyword hits include invariance disclaimers and unrelated equilateral geometry.','not_absolute_exhaustiveness':'Top-level research Python, summary JSON and top-level artifact plans scanned; vendored software, virtualenv and package copies excluded.'};write(_repo_path(OUT, 'novelty.json'),novelty)
    inputs=['artifacts/nested-calibration-074/development_features.npz','artifacts/nested-calibration-074/extraction.json','artifacts/bootstrap-selector-144/original/predictions.npz','artifacts/bootstrap-selector-144/original/report.json','artifacts/failure-review-166/analysis.npz','artifacts/cnn-dropout-171/fold-0/evaluation.json','artifacts/cnn-memorization-170/report.json','artifacts/cnn-memorization-170/replay.npz','artifacts/run-001/split_manifest.json','artifacts/cnn-seed-mean-133/audit/original.npz']
    for f in ['model.pt','report.json','streams.npz','predictions.npz']:inputs.append(str((previous.ref(0)/f).relative_to(R)))
    plan={'study':172,'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hypothesis':'Exact D4 weight sharing may lower augmented training error versus ordinary CNN with same approximate parameter count and optimizer updates; neither physical D4 invariance nor generalization gain assumed.','novelty_sha256':e.sha(_repo_path(OUT, 'novelty.json')),'architecture':'Lift Z2->D4 with 8 rotated/reflected tied kernels; each subsequent D4->D4 convolution mixes ALL8 input orientations by K_g[h,u]=K[g^-1*h,g^-1*u]. Four width4/12/24/36 stages, kernels5/3/3/3. Finite square zero padding, 2x2 averagepool BEFORE stride1 conv, dimensions128->64->32->16->8. GroupNorm4 with perbasechannel affine tied across8 orientations, SiLU; global orientation+space mean/std plus inputmean/std;74->64->1head/dropout.15. One image forward, not 8 independently processed images or outputTTA.','controls':'Reuse validated legacy129 D4 TinyCNN91,905params. New PoolCNN uses identical91905weights/initialstate, replaces stride2conv by avgpool2+stride1conv. It controls changed downsampling. GroupCNN91,593params(-0.3395%); headhidden64,dropout.15,GroupNorm4 retained, fewer basechannels replicatedover8 orientations. Width/pooling statistics change jointly with representation; not a pure single-coordinate architecture causal effect.','equivariance_scope':'Exact discrete90degree rotations/reflections on compatible square even pyramids only, numerical tolerance. No arbitraryangle/translationequivariance, physical target guarantee, or DINO property. Dropout occurs only after invariant pooling; eval predictions invariant, train path invariant for fixed dropoutmask.','training':'Originalfold0T320/V80. Same seed20261003,80epochs/800updates,batch32,unweightedMSE,AdamWlr.002,wd.01,clip5,epochcosineeta.0001,last5weightaverage. SameNumPyorder/D4draws/LR andactualinputhashes as129. AllnormalizationfromT320only; nocachedfullpopulationfit. No validationduringtraining or earlyselection. cleantrain20epochdiagnostics only.','compute_control':'Parameter and update counts matched; arithmetic cannot simultaneously match when tied kernels expanded over8 orientations. Report exact convMACs, measuredsamebatchstep times, driver/RSS andwallseconds. PoolCNN MACs sameaslegacy, group~7.3x. No claim equal FLOPs. Onefixedwidth, no speed/accuracy sweep.','engineering_gate':'Beforedevelopmentfit:64compositionchecks; independent slow groupconv forward/backprop and8actions; all8orientations influenceoutput; everyfeaturestage permutations andpooled prediction CPU/MPS; controlledinitialstate;16noise-free32px toy examples300updates, finalstandardizedMSE<.001 all8views; syntheticbenchmark projection<600s/fit andRSS/driver<3GB.','fold0_scope':'Two newmainfits (pool_control,group), plusonefullgroup outside80label-poison replay. Compare legacy129 and171p0 context, finalidentity/all8viewtrainingloss andall80validationRMSE; fixed80%complete018+20%CNN complementarity. No learnedblendratio, no staleCNNreplacement inside018.','decision':'Only recommend later5fold confirmation if group train8viewRMSE<bothlegacy andpoolcontrol AND group80validationRMSE<both AND fixed20%blendRMSE<018 andbothmatchedblends, withalltechnicalaudits passing. Currenttask stopsafterfold0 irrespectiveofgain; no automaticfold1-4/resplits/full500/test/newCSV/upload. Singlefold repeatedadaptive400development not confirmatory.','diagnostics':'Sparse/dense labelKDEbw.4trainSD/q20 threshold and166persistentlarge33 postfit only, not modelinputs or trainingweights. All80primary, subgroup harmsgains reported.','resources':'existinguv.venv CPU2/MPS,zero cost,nonewpackages/downloads;2mainfits+1poison,each<600s,RSSeach/driver<3GB. No holdout data or test access.','protected':e.read(_repo_path(R, 'artifacts/cubical-h1-168/plan.json'))['protected'],'source_sha256':{n:e.sha(_repo_path(R, n)) for n in SOURCE},'inputs':{n:e.sha(_repo_path(R, n)) for n in inputs}}
    write(_repo_path(OUT, 'plan.json'),plan);print('PLAN_LOCKED',e.sha(_repo_path(OUT, 'plan.json')),'files_reviewed',len(scanned),flush=True)
def check():
    p=e.read(_repo_path(OUT, 'plan.json'))
    for n,h in {**p['source_sha256'],**p['inputs'],**p['protected']}.items():assert e.sha(_repo_path(R, n))==h,n
    return p

def predict(model,images,norm,device='mps',g=0):
    model.eval();pp=[]
    with torch.inference_mode():
        for start in range(0,len(images),32):
            x=torch.from_numpy(np.ascontiguousarray((images[start:start+32]-norm['x_mean'])/max(norm['x_std'],1e-6))).to(device);p=model(net.action(x,g)).cpu().numpy()*norm['y_std']+norm['y_mean'];pp.extend(p.tolist())
    return np.array(pp)

def fit(xtrain,ytrain,train_ids,arm,path):
    # Validation arrays/labels and predictions are deliberately absent from the fit interface.
    assert xtrain.shape==(320,1,128,128) and len(set(train_ids))==320;path.mkdir(exist_ok=False)
    start=time.perf_counter();rng=np.random.default_rng(SEED);torch.manual_seed(SEED);norm={'x_mean':float(xtrain.mean()),'x_std':float(xtrain.std()),'y_mean':float(ytrain.mean()),'y_std':float(ytrain.std())};X=(xtrain-norm['x_mean'])/max(norm['x_std'],1e-6);y=(ytrain-norm['y_mean'])/norm['y_std'];model=net.MODELS[arm]().to('mps');init=prior.statehash(model);rng0=prior.mps_rng_hash();opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=80,eta_min=.0001);avg=None;orders=[];rots=[];flips=[];lrs=[];losses=[];hashes={};states={};curve=[];step=0
    print('FIT_START',arm,str(path.relative_to(R)),flush=True)
    for epoch in range(80):
        model.train();order=rng.permutation(320)
        for first in range(0,320,32):
            ix=order[first:first+32];rot=int(rng.integers(4));flip=bool(rng.random()<.5);xb=np.rot90(X[ix],rot,axes=(2,3));xb=xb[:,:,:,::-1] if flip else xb;xb=np.ascontiguousarray(xb)
            if step in (0,99,399,799):hashes[str(step)]=prior.ah(xb)
            orders.append(ix.copy());rots.append(rot);flips.append(flip);lrs.append(opt.param_groups[0]['lr']);xb=torch.from_numpy(xb).to('mps');target=torch.tensor(y[ix],dtype=torch.float32,device='mps');opt.zero_grad(set_to_none=True);loss=torch.nn.functional.mse_loss(model(xb),target);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach().cpu()));step+=1
        schedule.step()
        if epoch>=75:
            state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()};states[str(epoch+1)]=loader.cal.h.state_hash(state);torch.save(state,_repo_io(path/f'epoch-{epoch+1}.pt'))
            if avg is None:avg={n:v/5 for n,v in state.items()}
            else:
                for n,v in state.items():avg[n]+=v/5
        if (epoch+1)%20==0:
            clean=predict(model,xtrain,norm);rec={'epoch':epoch+1,'online_augmented_train_RMSE':float(np.sqrt(np.mean(losses[-10:]))*norm['y_std']),'clean_identity_train_RMSE':e.rmse(ytrain,clean),'elapsed_seconds':time.perf_counter()-start};curve.append(rec);print('EPOCH',arm,json.dumps(rec),flush=True);assert time.perf_counter()-start<600
    rngend=prior.mps_rng_hash();model.load_state_dict(avg);tp=predict(model,xtrain,norm);torch.mps.synchronize();payload={'state_dict':avg,'normalization':norm,'train_ids':train_ids,'seed':SEED,'epochs':80,'average_last_epochs':5,'arm':arm,'architecture':'d4_cnn_172.'+type(model).__name__};torch.save(payload,_repo_io(path/'model.pt'));np.savez_compressed(_repo_io(path/'streams.npz'),order=np.stack(orders),rot=np.array(rots),flip=np.array(flips),lr=np.array(lrs),loss=np.array(losses));np.savez_compressed(_repo_io(path/'train_predictions.npz'),predictions=tp)
    result={'arm':arm,'updates':step,'parameters':sum(p.numel() for p in model.parameters()),'initial_state_sha256':init,'rng_start':rng0,'rng_end':rngend,'normalization':norm,'average_state_sha256':loader.cal.h.state_hash(avg),'checkpoint_state_sha256':states,'actual_input_sha256':hashes,'model_sha256':e.sha(path/'model.pt'),'streams_sha256':e.sha(path/'streams.npz'),'train_prediction_sha256':e.sha(path/'train_predictions.npz'),'train_RMSE':e.rmse(ytrain,tp),'curve':curve,'memory':engine.memory('mps'),'seconds':time.perf_counter()-start};write(path/'report.json',result);assert result['seconds']<600 and result['memory']['process_peak_rss_bytes']<3e9 and result['memory']['driver_allocated_memory']<3e9;del model,opt,avg;gc.collect();torch.mps.empty_cache();print('FIT_COMPLETE',arm,result['train_RMSE'],result['seconds'],flush=True);return result

def run(arm):
    check()
    for mode in ['unit','synthetic','benchmark']:assert e.read(_repo_path(OUT, f'{mode}.json'))['passed']
    assert arm in net.MODELS;d=loader.load('original');T=np.flatnonzero(d['fid']!=0);V=np.flatnonzero(d['fid']==0);assert not set(d['g'][T])&set(d['g'][V]);r=fit(d['images'][T],d['y'][T],[d['ids'][i] for i in T],arm,_repo_path(OUT, arm));old=e.read(previous.ref(0)/'report.json');assert r['actual_input_sha256']==old['actual_input_sha256'];assert r['rng_start']==old['rng_start'] and r['rng_end']==old['rng_end']
    if arm=='pool_control':assert r['initial_state_sha256']==old['initial_state_sha256']
    with np.load(_repo_io(previous.ref(0)/'streams.npz')) as a,np.load(_repo_path(OUT, arm, 'streams.npz')) as b:
        for key in ['order','rot','flip','lr']:np.testing.assert_array_equal(a[key],b[key])
    check();print('SAME_INPUT_STREAMS_PASS',arm,flush=True)

def functional(state,x,arm):
    from test_d4_172 import slow
    if arm=='pool_control':
        h=x
        for i in range(0,12,3):h=torch.nn.functional.silu(torch.nn.functional.group_norm(torch.nn.functional.conv2d(torch.nn.functional.avg_pool2d(h,2),state[f'features.{i}.weight'],state[f'features.{i}.bias'],padding=2 if i==0 else 1),4,state[f'features.{i+1}.weight'],state[f'features.{i+1}.bias']))
        dims=(2,3)
    else:
        h=x
        for i,(inc,c) in enumerate([(1,4),(4,12),(12,24),(24,36)]):
            if i:
                b,cc,g,H,W=h.shape;h=torch.nn.functional.avg_pool2d(h.flatten(1,2),2).reshape(b,cc,g,H//2,W//2)
            else:h=torch.nn.functional.avg_pool2d(h,2)
            layer=net.D4Conv(inc,c,5 if i==0 else 3,i==0);layer.weight.data.copy_(state[f'convs.{i}.weight']);layer.bias.data.copy_(state[f'convs.{i}.bias']);h=slow(layer,h);h=torch.nn.functional.group_norm(h.flatten(1,2),4).reshape_as(h);h=torch.nn.functional.silu(h*state[f'norms.{i}.weight'][None,:,None,None,None]+state[f'norms.{i}.bias'][None,:,None,None,None])
        dims=(2,3,4)
    h=torch.cat([h.mean(dims),(h.var(dims,correction=0)+1e-6).sqrt(),x.mean((2,3)),(x.var((2,3),correction=0)+1e-6).sqrt()],1);return torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(h,state['head.0.weight'],state['head.0.bias'])),state['head.3.weight'],state['head.3.bias']).squeeze(1)

def validate():
    check();start=time.perf_counter();d=loader.load('original');T=np.flatnonzero(d['fid']!=0);V=np.flatnonzero(d['fid']==0);old=torch.load(_repo_io(previous.ref(0)/'model.pt'),map_location='cpu',weights_only=True);norm=old['normalization'];records={}
    for arm in net.MODELS:
        path=_repo_path(OUT, arm);r=e.read(path/'report.json');obj=torch.load(_repo_io(path/'model.pt'),map_location='cpu',weights_only=True);assert obj['train_ids']==old['train_ids']==[d['ids'][i] for i in T];assert obj['normalization']==norm=={'x_mean':float(d['images'][T].mean()),'x_std':float(d['images'][T].std()),'y_mean':float(d['y'][T].mean()),'y_std':float(d['y'][T].std())};assert e.sha(path/'model.pt')==r['model_sha256'];assert e.sha(path/'streams.npz')==r['streams_sha256'];assert e.sha(path/'train_predictions.npz')==r['train_prediction_sha256']
        with np.load(_repo_io(path/'streams.npz')) as z:O=z['order'].copy();rot=z['rot'].copy();flip=z['flip'].copy();lr=z['lr'].copy();assert np.isfinite(z['loss']).all()
        assert O.shape==(800,32);rng=np.random.default_rng(SEED);step=0
        for epoch in range(80):
            perm=rng.permutation(320)
            for first in range(0,320,32):
                np.testing.assert_array_equal(O[step],perm[first:first+32]);assert rot[step]==int(rng.integers(4)) and flip[step]==bool(rng.random()<.5);assert abs(lr[step]-(.0001+.5*(.002-.0001)*(1+math.cos(math.pi*epoch/80))))<1e-17
                if step in (0,99,399,799):
                    x=(d['images'][T[O[step]]]-norm['x_mean'])/norm['x_std'];x=net.action(torch.from_numpy(x),int(rot[step])*2+int(flip[step])).numpy();assert prior.ah(x)==r['actual_input_sha256'][str(step)]
                step+=1
        avg=None
        for epoch in range(76,81):
            state=torch.load(_repo_io(path/f'epoch-{epoch}.pt'),map_location='cpu',weights_only=True);assert loader.cal.h.state_hash(state)==r['checkpoint_state_sha256'][str(epoch)]
            if avg is None:avg={n:v/5 for n,v in state.items()}
            else:
                for n,v in state.items():avg[n]+=v/5
        assert all(torch.equal(v,obj['state_dict'][n]) for n,v in avg.items());model=net.MODELS[arm]().to('mps');model.load_state_dict(obj['state_dict']);mp=predict(model,d['images'],norm);saved=np.load(_repo_io(path/'train_predictions.npz'))['predictions'];em=float(abs(saved-mp[T]).max());assert em<1e-3;model=model.cpu();cp=predict(model,d['images'],norm,'cpu');ec=float(abs(cp-mp).max());assert ec<1e-3
        with torch.inference_mode():
            x=torch.tensor((d['images'][V[:8]]-norm['x_mean'])/norm['x_std']);p=functional(obj['state_dict'],x,arm).numpy()*norm['y_std']+norm['y_mean'];ef=float(abs(p-cp[V[:8]]).max());assert ef<1e-3
        np.savez_compressed(_repo_io(path/'verified_predictions.npz'),ids=np.array(d['ids']),predictions=mp,T=T,V=V);records[arm]={'passed':True,'MPS_reload_max_error':em,'CPU_vs_MPS_max_error':ec,'independent_functional_CPU_max_error':ef,'train_only_IDs_normalization':True,'800_streams_and_4_actual_input_hashes':True,'last5_average_exact':True,'verified_prediction_sha256':e.sha(path/'verified_predictions.npz')};del model
    # Full outside-label perturbation replay; only training rows enter fit, including normalization.
    alt=d['y'].copy();alt[V]+=10000;rr=fit(d['images'][T],alt[T],[d['ids'][i] for i in T],'group',_repo_path(OUT, 'outside-label-replay'));original=e.read(_repo_path(OUT, 'group/report.json'));assert rr['average_state_sha256']==original['average_state_sha256'];a=torch.load(_repo_path(OUT, 'group/model.pt'),map_location='cpu',weights_only=True);b=torch.load(_repo_path(OUT, 'outside-label-replay/model.pt'),map_location='cpu',weights_only=True);assert all(torch.equal(v,b['state_dict'][n]) for n,v in a['state_dict'].items());assert a['normalization']==b['normalization'];records['outside80label_poison']={'changed_labels':80,'full800_update_refit':True,'state_exact':True,'normalization_exact':True,'inference_equal_by_identical_state':True};check();result={'passed':True,'arms':records,'seconds':time.perf_counter()-start};write(_repo_path(OUT, 'validation.json'),result);print('VALIDATION_PASS',json.dumps(result),flush=True)

def stats(y,p,mask):
    if not mask.any():return {'n':0}
    residual=y[mask]-p[mask];return {'n':int(mask.sum()),'RMSE':float(np.sqrt(np.mean(residual**2))),'bias_y_minus_pred':float(residual.mean())}
def finish():
    check();assert e.read(_repo_path(OUT, 'validation.json'))['passed'];d=loader.load('original');T=np.flatnonzero(d['fid']!=0);V=np.flatnonzero(d['fid']==0);views={};score={};resources={};norm=None
    for arm,path in [('legacy',previous.ref(0)),*[(a,_repo_path(OUT, a)) for a in net.MODELS]]:
        obj=torch.load(_repo_io(path/'model.pt'),map_location='cpu',weights_only=True);norm=obj['normalization'];model=(engine.TinyCNN() if arm=='legacy' else net.MODELS[arm]()).to('mps');model.load_state_dict(obj['state_dict']);p=np.stack([predict(model,d['images'],norm,g=g) for g in range(8)]);views[arm]=p
        if arm=='legacy':np.testing.assert_allclose(p[0,V],d['legacy'][V],rtol=0,atol=1e-3)
        else:
            z=np.load(_repo_io(path/'verified_predictions.npz'));np.testing.assert_allclose(p[0],z['predictions'],rtol=0,atol=1e-3)
        score[arm]={'train_identity_RMSE':e.rmse(d['y'][T],p[0,T]),'train8view_RMSE':float(np.sqrt(np.mean((p[:,T]-d['y'][T])**2))),'validation_identity_RMSE':e.rmse(d['y'][V],p[0,V]),'validation8view_RMSE':float(np.sqrt(np.mean((p[:,V]-d['y'][V])**2))),'train_mean_view_std':float(p[:,T].std(0).mean()),'validation_mean_view_std':float(p[:,V].std(0).mean()),'all400_max_view_prediction_difference':float(abs(p-p[0]).max()),'fixed20blend_validation_RMSE':e.rmse(d['y'][V],.8*d['reference018'][V]+.2*p[0,V])};resources[arm]=e.read(path/'report.json');del model;torch.mps.empty_cache()
    assert score['group']['all400_max_view_prediction_difference']<1e-3;reference=e.rmse(d['y'][V],d['reference018'][V]);rho=density.density(d['y'][T],d['y'][T],True);th=np.quantile(rho,.2);sparse={'train':rho<th,'validation':density.density(d['y'][T],d['y'][V])<th};large=np.load(_repo_path(R, 'artifacts/failure-review-166/analysis.npz'))['persistent_large'];groups={}
    for phase,ix in [('train',T),('validation',V)]:
        groups[phase]={}
        for group,mask in [('all',np.ones(len(ix),bool)),('sparse',sparse[phase]),('dense',~sparse[phase]),('persistent_large33',large[ix])]:
            groups[phase][group]={arm:{**stats(d['y'][ix],p[0,ix],mask),'all8view_RMSE':float(np.sqrt(np.mean((p[:,ix][:,mask]-d['y'][ix][mask])**2))) if mask.any() else None} for arm,p in views.items()}
            if phase=='validation':
                groups[phase][group]['018']=stats(d['y'][ix],d['reference018'][ix],mask)
                for arm,p in views.items():groups[phase][group]['blend_'+arm]=stats(d['y'][ix],.8*d['reference018'][ix]+.2*p[0,ix],mask)
    checks={'group_train8_below_both':all(score['group']['train8view_RMSE']<score[c]['train8view_RMSE'] for c in ['legacy','pool_control']),'group_validation_below_both':all(score['group']['validation_identity_RMSE']<score[c]['validation_identity_RMSE'] for c in ['legacy','pool_control']),'group_blend_below_all':score['group']['fixed20blend_validation_RMSE']<min(reference,*[score[c]['fixed20blend_validation_RMSE'] for c in ['legacy','pool_control']])};recommend=all(checks.values());error=d['y'][V]-d['reference018'][V];correction=views['group'][0,V]-d['reference018'][V]
    # Count convolution multiply-accumulates at exact output shapes; pooling/norm/head omitted explicitly.
    def mac(widths,isgroup):
        inc=1;s=128;total=0
        for i,c in enumerate(widths):
            s//=2;total+=s*s*(c*8 if isgroup else c)*(inc*8 if isgroup and i else inc)*(5 if i==0 else 3)**2;inc=c
        return total
    macs={'legacy':mac([16,32,64,96],False),'pool_control':mac([16,32,64,96],False),'group':mac([4,12,24,36],True)}
    np.savez_compressed(_repo_path(OUT, 'evaluation.npz'),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],T=T,V=V,reference018=d['reference018'],**views)
    result={'study':172,'status':'completed_fold0_only','recommend_remaining4fold_confirmation':recommend,'candidate':False,'technical_validation':e.read(_repo_path(OUT, 'validation.json')),'gate_checks':checks,'fold0_reference018_RMSE':reference,'scores':score,'groups':groups,'fold0_dropout0_context_only':e.read(_repo_path(R, 'artifacts/cnn-dropout-171/fold-0/evaluation.json'))['scores'],'conv_MACs_per128image_excludes_pool_norm_head':macs,'group_vs_legacy_conv_MAC_ratio':macs['group']/macs['legacy'],'parameters':e.read(_repo_path(OUT, 'unit.json'))['parameters'],'benchmark':e.read(_repo_path(OUT, 'benchmark.json')),'fit_resources':{a:{k:v for k,v in r.items() if k in ['seconds','memory','RSS_bytes','MPS_driver_bytes','updates']} for a,r in resources.items()},'018_correction_alignment':{'mean_e_times_d':float(np.mean(error*correction)),'mean_d_squared':float(np.mean(correction**2)),'MSE_gain_at_weight_point2':float(np.mean(error**2)-np.mean((error-.2*correction)**2))},'scope':{'development_ids':400,'training_rows':320,'validation_rows':80,'main_fits':2,'poison_replays':1,'no_holdout_test_full500_CSV_upload':True,'new_install_download_cost':False},'limits':['Single historically reused fold0 cannot establish generalization. No otherfold expansion performed.','Approximate parameter/update matching, not equal arithmetic or walltime. Groupwidth and pooling overorientations also change representation.','Exact finite D4 equivariance does not validate physical labelinvariance or arbitraryrotation.','Sparse and persistentlargegroups uselabels/posthocoldresiduals for reportingonly.'],'plan_sha256':e.sha(_repo_path(OUT, 'plan.json')),'evaluation_sha256':e.sha(_repo_path(OUT, 'evaluation.npz')),'sources':check()['source_sha256'],'protected_verified':check()['protected']}
    write(_repo_path(OUT, 'final.json'),result);write(_repo_path(R, 'reports/d4-cnn-172-summary.json'),result);backup=_repo_path(R, 'reports/README-before-d4-172.md');assert not backup.exists();backup.write_text((_repo_path(R, 'README.md')).read_text());text='\n\n## Study172: exact D4 group CNN, engineering tests and originalfold0 only\n\nTrue weight-tied lifting/group convolutions mixall8 orientations, not outputTTA. Width4/12/24/36; tiedGroupNorm4; averagepool2 beforestride1conv preservesfinite-squareD4sampling. New ordinaryPoolCNN controls this downsampling change withsame91905initialweights; D4model91593params. Same80epochs/800updates/seed20261003/D4/dropout.15/AdamWwd.01/last5average. TrainonlyT320normalization andmatchinginput/order/LRstreams verified.64groupcomposition and independent slowconv/gradient checks, allstagepermutations CPU/MPS, noiseless16example fit andresourcebenchmark passed.\n\nFold0CNN validation: legacy='+str(score['legacy']['validation_identity_RMSE'])+', poolcontrol='+str(score['pool_control']['validation_identity_RMSE'])+', D4group='+str(score['group']['validation_identity_RMSE'])+'. Reference018='+str(reference)+', fixed20percentgroupblend='+str(score['group']['fixed20blend_validation_RMSE'])+'. Recommendremaining4foldconfirmation='+str(recommend)+'. Currentstudy stopsafterfold0; no candidate/promotion/generalizationclaim. All8viewtrainloss/subgroupharms andcompute/memorydifferences retained.\n\nTwo mainfits plusfull800update outside80labelpoison replay; stateexact,5stateaverage andCPU/MPS/independentfunctionalreload validated. Best018/ZIPs preserved. No holdout/test/full500/CSV/upload, package/download/cost. Evidence:`reports/d4-cnn-172-summary.json`, `artifacts/d4-cnn-172/plan.json`, unit/synthetic/benchmark/validation andmodelartifacts.\n';(_repo_path(R, 'README.md')).write_text((_repo_path(R, 'README.md')).read_text()+text);check();print('FINAL',json.dumps({k:result[k] for k in ['status','recommend_remaining4fold_confirmation','gate_checks','scores','fold0_reference018_RMSE','parameters','group_vs_legacy_conv_MAC_ratio']}),flush=True)
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':prepare()
    elif mode=='validate':validate()
    elif mode=='finish':finish()
    else:run(mode)
