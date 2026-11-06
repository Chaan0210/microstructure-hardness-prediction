"""One fixed weak pixel-noise augmentation; frozenB same-budget head control."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,time,datetime,hashlib,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));assert Path(sys.prefix).resolve()==_repo_path(ROOT, '.venv')
import numpy as np,torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
import frozen_seed_088 as old
import dinob_062 as b
import amplitude_cnn_106 as a
e=old.e;CFG=e.CFG;OUT=_repo_path(ROOT, 'artifacts/input-noise-110');SPLITS=('original','20261031','20261101');ARMS=('control','noise');SIGMA_UINT8=2.;DEADLINE='2026-10-03T18:34:00+00:00';FIXED5=('TRAIN_000438','TRAIN_000331','TRAIN_000227','TRAIN_000071','TRAIN_000168');read=a.read;write=a.write;sha=a.sha;emit=a.emit;rmse=a.rmse

def timecheck():assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(DEADLINE),'18:34UTC batchdeadline'
def check():
 p=read(_repo_path(OUT, 'plan.json'));timecheck()
 for n,h in {**p['sources'],**p['inputs'],**p['protected011']}.items():assert sha(_repo_path(ROOT, n))==h,n
 return p

def noise_seed(ident):return int.from_bytes(hashlib.sha256(('noise110:'+ident).encode()).digest()[:8],'little')
def noisy_pixels(raw,ident):
 z=raw.astype(np.float64)+np.random.default_rng(noise_seed(ident)).normal(0,SIGMA_UINT8,raw.shape);return np.rint(np.clip(z,0,255)).astype(np.uint8)
def image_input(raw):
 x=np.asarray(Image.fromarray(raw).resize((504,504),Image.Resampling.BICUBIC),dtype=np.float32)/255;return np.ascontiguousarray((np.repeat(x[None],3,axis=0)-b.ds.base.MEAN)/b.ds.base.STD)
def phases(ids):return np.array([noise_seed(i)%2 for i in ids],dtype=np.int64)

def prepare():
 OUT.mkdir(exist_ok=False);timecheck();src=['input_noise_110.py','validate_input_noise_110.py','frozen_seed_088.py','dinob_joint_084.py','dinob_062.py'];inp=['artifacts/dropout-control-089/plan.json','artifacts/mixup-061/plan.json','artifacts/dinob-denoised-071/plan.json','artifacts/denoise-complement-072/plan.json','artifacts/amplitude-cnn-106/representation-review.json','artifacts/amplitude-cnn-106/images.npy','artifacts/amplitude-cnn-106/image-cache.json','artifacts/dinob-062/development_features.npz','artifacts/dinob-062/extraction.json','reports/frozen-complement-085-validation.json']+[f'artifacts/frozen-complement-085/{s}/development_predictions.npz' for s in SPLITS]
 p={'study':'110 single weaknativepixelnoise trainingaugmentation','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_utc':DEADLINE,'config':CFG,'sigma_uint8':SIGMA_UINT8,'hypothesis':'Weak inputperturbation duringfrozenBheadtraining mayreduce sensitivity tosmall stochasticpixel changes. Noise isnotknown to be labelinvariant andactualnoise neednotbeiid; effectisonlyempirical.',
 'duplicate_audit':{'089':'Changeshidden dropout.50to.25;110retains.50.','061':'Mixesdifferentimagefeatures/targets usingBeta.4;110keepssameimage andunchangedtarget.','071_072':'TVdenoisingbeforeDINO orraw/TVpredictionmixture;110doesnotdenoiseoraverageinferenceviews.','106':'GroupNormremoval inscratchCNN;110retainsfrozenofficialB andexistinghead.','046_104':'D4orpatch-phase featureaveraging;110norotation/crop/patchshift.'},
 'perturbation':'Image/pixelspace, BEFORE504resize/backbone. For eachnative256uint8 raw image, independentPCG64seed=first8SHA256(noise110:ID)littleendian; addGaussianN(0,2^2)inuint8intensityunits,clip0..255,rinttointeger,uint8. Onefixedview perimage. Sigma2/255 is0.784%fullscale, chosen asweak2code-level perturbation, notfromlabelorvalidation optimization. No intensityshift/contrastjitter/noiselevel sweep. Clipping/quantization can change mean; recordactualpixelchanges.',
 'features':'SamePILbicubic504,grayRGB,ImageNetfixedaffine,frozenB6912block6/9/12 CLSmean/popstd. Reuseidentity062cache, exactidentityextractionchecks onfirst3; extractonlyone noisyview per400. Cachedimage-localfrozenfeatures safeacrossfolds; noiseseed usesIDonly.',
 'learning_budget':'Freshcontrolandcandidate foreachfold. Same01132wideGELUdropout.5,12epochs,seed20270215+k, effective16/micro2,AdamWwd.1,cosine3e-4to1e-5,SmoothL1beta1,clip1. 320*12=3840presentations,240updates. Sameinit/order/dropoutstream. Controlallidentity; candidateview=(epoch+IDhashparity)%2: exactly6identity/6noise pertrainimage. No doubleddataset/extraepochs/earlystop.',
 'normalization':'Botharms originalidentityfeature StandardScaler fitoutertrainonly; sametrainlabelmean/std. Noisyfeatures transformedthroughthisfixedscaler. Validationpredictionidentityonly. Frozenbackbone notupdated. No fittedensembleweight orcalibration.',
 'formula':'.9*samefold009+.1*head. Freshcontrolmustreproduceexisting011 andfullsavedmodelstatesexactly. Candidate replacesonly011freshhead branch.',
 'gate':'Originalpooledgain>=.02RMSE versus011 AND>=3/5positivefolds AND positivegain afterdeleting5largestpositive squared-error-gain contributors. Operationalmeaningfulnesscriteria,not significance. Freezeoriginaltop5IDsthenrun BOTHexistingresplits unchangediforiginalandindependentverificationpass. Eachresplitmustmeetgain>=.02,>=3folds,andpositivegainafterdeletingSAMEoriginaltop5. All3requiredforpromotion. No strength/weights/seed/endpointsearch.',
 'diagnostics':'ConditionalpairedbootstrapCI,perIDerrorgains,originaltop5deletion andhistorical092fixed5deletion. Exclusions onlydiagnostic/gate,notnewtrainingfilter; everymodeltrainsallfoldtrainrows. Same400reusedandconditionalCIignoresadaptiveselection.',
 'resource':'CPU2/MPSexistingvenv,cost0. First3identity+noisyimagesbenchmarkproject400<900s,RSS/MPS<4GB. Firstpairedfoldbench<120sreusedonce. Stopat18:34UTC; noinstall/newweights/externaltransfer.',
 'verification':'All10modelsreloadexact,CPUheadspotchecks,allfreshcontrolstatesmatchsaved011,matchedinit/order/dropout hashes. Fullfold0candidate replay withoutervallabelsandoutervalNOISYfeaturesperturbed; identityvalidationfeaturesunchanged. Three noisyfeature/pixel replays andoneCPUbackbonecheck. Best011protected.',
 'limits':'Noactualphysicalstructurepreservation,realnoiseindependenceorlabelinvariance proved. Weakaugmentation maystilldiscardusefulvariation. No holdout/testselection.',
 'sources':{n:sha(_repo_path(ROOT, n)) for n in src},'inputs':{n:sha(_repo_path(ROOT, n)) for n in inp},'development_ids':a.check()['development_ids'],'protected011':a.check()['protected011'],'cost':0};write(_repo_path(OUT, 'plan.json'),p);check();emit({'prepared':True,'plan_sha256':sha(_repo_path(OUT, 'plan.json')),'gate':p['gate']})

def extract():
 p=check();torch.set_num_threads(2);ids=p['development_ids'];images=np.load(_repo_io(a.OUT/'images.npy'),mmap_mode='r',allow_pickle=False)
 with np.load(_repo_path(ROOT, 'artifacts/dinob-062/development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;original=z['X'].copy()
 model=b.model().to('mps');before=e.h.state_hash(model.state_dict());start=time.perf_counter();arr=[];stats=[]
 for i,ident in enumerate(ids):
  timecheck();raw=np.asarray(images[i,0]);noisy=noisy_pixels(raw,ident);delta=noisy.astype(float)-raw.astype(float);x=image_input(noisy)
  if i<3:
   identity=b.feature(model,torch.from_numpy(image_input(raw)[None]).to('mps')).cpu().numpy()[0];np.testing.assert_array_equal(identity,original[i])
  out=b.feature(model,torch.from_numpy(x[None]).to('mps')).cpu().numpy()[0];arr.append(out);stats.append({'ID':ident,'seed':noise_seed(ident),'pixel_mean_change':float(delta.mean()),'pixel_RMS_change':float(np.sqrt(np.mean(delta**2))),'max_abs_pixel_change':float(abs(delta).max()),'fraction_changed':float((delta!=0).mean()),'noisy_pixel_sha256':hashlib.sha256(noisy.tobytes()).hexdigest(),'feature_delta_RMS':float(np.sqrt(np.mean((out.astype(float)-original[i])**2)))})
  if i==2:
   sec=time.perf_counter()-start;r={'first3_identity_and_noisy_seconds':sec,'conservative_projected400_seconds':sec*400/3,'RSS_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'MPS_driver_bytes':torch.mps.driver_allocated_memory()};r['passed']=r['conservative_projected400_seconds']<900 and max(r['RSS_peak_bytes'],r['MPS_driver_bytes'])<4e9;write(_repo_path(OUT, 'extraction-benchmark.json'),r);emit(r);assert r['passed']
  if (i+1)%40==0:emit({'extracted':i+1,'seconds':time.perf_counter()-start})
 N=np.stack(arr);assert N.shape==(400,6912) and np.isfinite(N).all();assert e.h.state_hash(model.state_dict())==before and all(t.grad is None for t in model.parameters())
 with (_repo_path(OUT, 'noisy_features.npz')).open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(ids),N=N)
 r={'shape':[400,6912],'identity3_exact':True,'backbone_unchanged':True,'features_sha256':sha(_repo_path(OUT, 'noisy_features.npz')),'seconds':time.perf_counter()-start,'records':stats,'labels_used':False};write(_repo_path(OUT, 'extraction.json'),r);check();emit({'extraction_complete':True,'seconds':r['seconds'],'pixel_RMS_range':[min(z['pixel_RMS_change'] for z in stats),max(z['pixel_RMS_change'] for z in stats)]})

def inputs(split):
 p=check();d=old.inputs(split);assert d['ids']==p['development_ids'];r=read(_repo_path(OUT, 'extraction.json'));assert sha(_repo_path(OUT, 'noisy_features.npz'))==r['features_sha256']
 with np.load(_repo_path(OUT, 'noisy_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==d['ids'];d['N']=z['N'].copy()
 return d

def fit(d,m,k,arm,path):
 timecheck();path.mkdir(parents=True,exist_ok=False);start=time.perf_counter();tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);assert not set(d['g'][tr])&set(d['g'][va]);sc=StandardScaler().fit(d['B'][tr]);ym=float(d['y'][tr].mean());ys=float(d['y'][tr].std());actual=CFG['seed']+k;model=e.Model(m,sc,actual,False).to('mps');initial=e.h.state_hash(model.state_dict());fixed=e.fixed_hash(model);opt=e.optimizer(model);orders=np.random.default_rng(actual).permuted(np.tile(np.arange(len(tr)),(12,1)),axis=1);curve=[];seeds=[];phase=phases(d['ids']);noise_count=np.zeros(len(d['ids']),dtype=int)
 for ep,order in enumerate(orders):
  timecheck();e.schedule(opt,ep);model.training_mode();losses=[]
  for first in range(0,len(tr),16):
   group=tr[order[first:first+16]];opt.zero_grad(set_to_none=True)
   for offset in range(0,len(group),2):
    ii=group[offset:offset+2];rng=actual+10000*(ep+1)+first+offset;torch.manual_seed(rng);seeds.append(rng);view=(ep+phase[ii])%2 if arm=='noise' else np.zeros(len(ii),dtype=int);noise_count[ii]+=view;X=d['B'][ii].copy();X[view==1]=d['N'][ii[view==1]];p=model(e.tensor(X));loss=F.smooth_l1_loss(p,e.tensor((d['y'][ii]-ym)/ys),beta=1.);assert torch.isfinite(loss);(loss*(len(ii)/len(group))).backward();losses.append(float(loss.detach()))
   gn=torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.);assert torch.isfinite(gn);opt.step()
  tp=e.predict(model,d,None,tr)*ys+ym;curve.append({'epoch':ep+1,'online_train_loss':float(np.mean(losses)),'train_RMSE':rmse(d['y'][tr],tp),'head_lr':opt.param_groups[0]['lr']})
 assert np.all(noise_count[va]==0) and np.all(noise_count[tr]==(6 if arm=='noise' else 0));pred=e.predict(model,d,None,va)*ys+ym;assert e.fixed_hash(model)==fixed;state={n:v.detach().cpu() for n,v in model.state_dict().items()};torch.save({'state':state,'arm':arm,'train_ids':[d['ids'][i] for i in tr],'validation_ids':[d['ids'][i] for i in va],'config':CFG,'seed':actual,'target_mean':ym,'target_std':ys,'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'scaler_var':torch.tensor(sc.var_),'initial_state_sha256':initial,'fixed_state_sha256':fixed},_repo_io(path/'model.pt'))
 with (path/'predictions.npz').open('xb') as file:np.savez_compressed(_repo_io(file),train_indices=tr,validation_indices=va,train_predictions=tp,predictions=pred,noise_count=noise_count)
 r={'arm':arm,'fold':k,'initial_state_sha256':initial,'fixed_state_sha256':fixed,'final_state_sha256':e.h.state_hash(state),'batch_order_sha256':hashlib.sha256(orders.tobytes()).hexdigest(),'dropout_seeds_sha256':hashlib.sha256(np.array(seeds,dtype=np.int64).tobytes()).hexdigest(),'noise_presentations':int(noise_count.sum()),'total_presentations':3840,'optimizer_updates':240,'curve':curve,'checkpoint_sha256':sha(path/'model.pt'),'predictions_sha256':sha(path/'predictions.npz'),'seconds':time.perf_counter()-start};write(path/'report.json',r);emit({'arm':arm,'fold_complete':k,'seconds':r['seconds']});return pred,r

def statistics(d,ps,split):
 y=d['y'];ref=d['reference011'];candidate=ps['primary'];comp=a.compare(y,ref,candidate,d['fid']);sq=(y-ref)**2-(y-candidate)**2;order=np.argsort(-sq,kind='stable');top=[d['ids'][i] for i in order[:5]] if split=='original' else read(_repo_path(OUT, 'original/report.json'))['sample_diagnostics']['original_top5_IDs'];keep=~np.isin(d['ids'],top);fixed=~np.isin(d['ids'],FIXED5);trim=rmse(y[keep],ref[keep])-rmse(y[keep],candidate[keep]);fg=rmse(y[fixed],ref[fixed])-rmse(y[fixed],candidate[fixed]);draw=np.random.default_rng(20271010).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-ref)[draw]**2,1))-np.sqrt(np.mean((y-candidate)[draw]**2,1));diag={'original_top5_IDs':top,'gain_without_original_top5':trim,'historical092_fixed5_gain':fg,'current_top5_diagnostic_only':[d['ids'][i] for i in order[:5]],'improved_rows':int((sq>0).sum()),'conditional_gain_CI95':np.percentile(boot,[2.5,97.5]).tolist(),'prediction_difference_RMS':float(np.sqrt(np.mean((candidate-ref)**2)))};gate=comp['gain']>=.02 and comp['positive_folds']>=3 and trim>0;return comp,diag,bool(gate),sq

def run(split):
 check();torch.set_num_threads(2)
 if split!='original':assert read(_repo_path(OUT, 'original/report.json'))['advance'] and read(_repo_path(ROOT, 'reports/input-noise-110-original-validation.json'))['passed']
 d=inputs(split);dest=_repo_path(OUT, split);dest.mkdir(exist_ok=False);m=b.model();start=time.perf_counter();pred={arm:np.full(400,np.nan) for arm in ARMS};records={}
 for k in range(5):
  rr={};pair_start=time.perf_counter()
  for arm in ARMS:pred[arm][d['fid']==k],rr[arm]=fit(d,m,k,arm,dest/f'{arm}-fold{k}');torch.mps.empty_cache()
  for key in ('initial_state_sha256','fixed_state_sha256','batch_order_sha256','dropout_seeds_sha256'):assert rr['control'][key]==rr['noise'][key]
  saved=torch.load(_repo_io(old.path_for(split,CFG['seed'],k)/'model.pt'),map_location='cpu',weights_only=True);assert rr['control']['final_state_sha256']==e.h.state_hash(saved['state']);np.testing.assert_array_equal(pred['control'][d['fid']==k],d['basehead'][d['fid']==k]);records[str(k)]={arm:sha(dest/f'{arm}-fold{k}/report.json') for arm in ARMS}
  if k==0:
   bench={'first_paired_fold_seconds':time.perf_counter()-pair_start,'RSS_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'MPS_driver_bytes':torch.mps.driver_allocated_memory()};bench['passed']=bench['first_paired_fold_seconds']<120 and max(bench['RSS_peak_bytes'],bench['MPS_driver_bytes'])<4e9;write(dest/'training-benchmark.json',bench);assert bench['passed']
 ps={'reference009':d['reference009'],'reference011':d['reference011'],'control_head':pred['control'],'noise_head':pred['noise'],'control':.9*d['reference009']+.1*pred['control'],'primary':.9*d['reference009']+.1*pred['noise']};np.testing.assert_array_equal(ps['control'],d['reference011']);comp,diag,advance,sq=statistics(d,ps,split)
 with (dest/'development_predictions.npz').open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],squared_error_gain=sq,**ps)
 r={'split':split,'scores':{n:rmse(d['y'],v) for n,v in ps.items()},'comparison':comp,'sample_diagnostics':diag,'advance':advance,'all5controlstates_reproduce011_exact':True,'fold_report_sha256':records,'predictions_sha256':sha(dest/'development_predictions.npz'),'seconds':time.perf_counter()-start,'plan_sha256':sha(_repo_path(OUT, 'plan.json')),'candidate_created':False};write(dest/'report.json',r);check();emit(r)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','extract',*SPLITS]);x=ap.parse_args();{'prepare':prepare,'extract':extract}.get(x.mode,lambda:run(x.mode))()
