"""Learned local interactions on frozen DINO grids, with a paired shuffled-grid control."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[name]='2'
import sys,time,json,hashlib,datetime,argparse,resource,math
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];assert Path(sys.prefix).resolve()==_repo_path(ROOT, '.venv');OUT=_repo_path(ROOT, 'artifacts/spatial-token-cnn-161')
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
SPLITS=('original','20261031','20261101');ARMS=('permuted','spatial');SEEDS=(20271222,20271242,20271262)
CFG={'epochs':30,'batch':16,'lr':.001,'min_lr':.00001,'weight_decay':.1,'dropout':.25,'clip':1.,'blend':.2,'global_hidden':32,'local_hidden':16,'projection_width':32,'permutation_seed':20271221}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,z):
 with Path(p).open('x') as f:json.dump(z,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def emit(z):print(json.dumps(z,ensure_ascii=False),flush=True)
def sh(state):
 h=hashlib.sha256()
 for n,t in sorted(state.items()):h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def rmse(y,p):return float(np.sqrt(np.mean((np.asarray(y)-p)**2)))
def seed_all(seed):torch.manual_seed(seed);torch.mps.manual_seed(seed)
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.global_fc=nn.Linear(6912,32);self.stem=nn.Conv2d(32,16,1);self.conv1=nn.Conv2d(16,16,3,padding=1);self.conv2=nn.Conv2d(16,16,3,padding=1);self.drop=nn.Dropout(.25);self.readout=nn.Linear(64,1)
 def forward(self,x,z):
  g=F.gelu(self.global_fc(x));h=F.gelu(self.stem(z));h=F.gelu(self.conv1(h));h=F.gelu(self.conv2(h));v=torch.cat([g,h.mean((2,3)),(h.var((2,3),correction=0)+1e-6).sqrt()],1);return self.readout(self.drop(v)).squeeze(1)

def prepare():
 assert torch.backends.mps.is_available();OUT.mkdir(exist_ok=False)
 sources=['spatial_token_cnn_161.py','patch_attention_092.py','spatial_joint_140.py','cnn_structure_experiment.py','reports/official-reference-review-018.json','reports/cnn-structure-035-summary.json','reports/patch-attention-092-summary.json','reports/spatial-joint-140-summary.json','reports/cluster-pair-159-summary.json','reports/layout-dependency-160-summary.json','reports/candidate020-public-submission.json','artifacts/patch-attention-092/cache.json','artifacts/spatial-joint-140/features.npz','artifacts/spatial-joint-140/extraction.json','artifacts/dinob-062/development_features.npz','artifacts/dinob-062/extraction.json','artifacts/run-001/split_manifest.json']
 for split in SPLITS:sources += [f'artifacts/bootstrap-selector-144/{split}/report.json',f'artifacts/bootstrap-selector-144/{split}/predictions.npz',f'artifacts/bootstrap-selector-144/validation-{split}.json']
 p={'study':'161 supervised spatial interaction learning on frozenBgrid','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hypothesis':'Officialdomainreview namesgrain/phase/porelayout andheterogeneity. Existing140adjacentjointstatistics had modestcomplementarity;fixedglobalmoments andpointwiseattention omit trainable nonlinearneighbor interactions. Learnlocalsemanticregionarrangement onfrozenpatchgrid plus retainedglobalB descriptors. Potentialbenefit unproven; no truegrain/phaseidentification claim.','nonduplicate':{'035/earlyCNN':'scratchrawpixelCNN covariance/resolution/multistagechanges,notCNNoverpretrainedsemanticgrid','092':'tokenwiseMLP+gatedweightedaverage,no spatialconvolution/neighbormixing','140':'fixedquadraticnearestneighbordifference,not learnednonlinear local interaction','149':'fixed colocatedintensitytokenmoments,notCNNgridmodel','159/160':'fixedbinaryconnectivity/labelblinddependencyaudit;not reused or retuned'},'inputs':'Immutable400 B lastnormalizedpatch1296x768 from092. Samefixedlabel-independent140QRprojection768x32,reshape36x36x32. Retainall6912BmultilayerCLS/mean/std asglobalbranch. NofitPCA/newpretrainedweights/backboneadaptation.','architecture':'Global6912->32GELU; local32->16 1x1GELU ->16 3x3padding1GELU ->16 3x3padding1GELU. Concatenate global32/localmean16/localpopulationstd16(sqrtvar+1e-6),dropout.25,linear64->1. Nomaxpool/GroupNorm/learnedattention. Effective5x5token neighborhood; tokensalreadyhaveglobalcontext.','control':'Samearchitecture/parameters/wholeinitialstate/orders/dropoutseeds/optimizer/epochs. Permutedcontrol uses ONEfixed1296positionpermutation sharedallimages/folds/epochs,aftertrain-channelstandardization. Retainsidenticalperimage projectedtokenmultiset/globalB; destroysmosttrueadjacency,notallpositionalinformationinsideDINOtokens. Onlyspatialcanadvance. No extra pointwise or global-onlyfallback.','config':CFG,'seeds':SEEDS,'prediction':'Mean3fixedseedrawpredictions eacharm. PRIMARY=.8unchanged018+.2spatialmean;matched=.8unchanged018+.2permutedmean. Weight20%fixedbeforetraining,comparabletocurrentS/Bbranchweights,neveroptimizedonOOF/Public. Base018calibrator/residualmodels untouched; newbranchdirecttargetfitouterT320,notglobalOOFresidualtargets.','training':'PerfoldglobalStandardScaler andprojectedchannelmean/std overONLY320trainimages. Targetmean/stdONLYtrain. MSE onstandardizedtarget,AdamW,epochcosine fixed30endpoint,noearlystop,bestepoch,SWA,augmentationorlabelbinning. All3seeds paired. No validation loss inspectedwhilefitting.','selection':'All3existingpartitions (15outerfolds) completedbeforedecision. Primarypooled1200rowuntrimmedMSE mustbeatBOTH018 andmatchedcontrol plusindependentverification. Noeachfold/CI/minimumgaingate; reportconditional10000IDgroupCIseed20271223 andeffectsize. Negative closesonefixedarchitecture,notlayer/width/epoch/projection/weightpermutation sweep. Positive onlyprospectivedevelopmentcandidate withfull500/testworkaftervalidatedlock.','resources':'ExistinguvCPU2/MPSonly.90smallfits(2arms*3seeds*15folds) plus2firstseedouterlabelpoisonreplays. First12stepmicrobenchmark projected90fitswith30%overhead<1800sec;RSS/MPSdriver<3GB. Firstcompletefold6fits rechecked/reused; stopresourcefailurewithoutsilentconfigchange. Noinstall/download/cost/externaldata.','validation':'Projectedcacheidentity/explicitchannelproducts;normalizersrawtrainstatistics;parameter/gradient/pairingchecks; all90savedmodelsMPSreload/CPUfunctionalforward; onecompleteouter80labelpoisonfitperarm; all3seedpredictionmeans/formula/IDs/finite/groups andprotectionhashes.','limits':'No guaranteedrankgain. Same400adaptivehistory andoverlappingresplits,CIconditionsonfixedfits. Randomprojectiondiscardschannels;permutationchangesboundarycontext;onefixedpermutationisnotuniversalnull. GlobalBmeanstdbranchmaydominate. No physicalabsolute-scale or truephase labels.','hashes':{n:sha(_repo_path(ROOT, n)) for n in sources},'protected':{str(q.relative_to(ROOT)):sha(q) for q in [_repo_path(ROOT, 'artifacts/submission-candidate-018.csv'),_repo_path(ROOT, 'artifacts/submission-candidate-019.csv'),_repo_path(ROOT, 'artifacts/submission-candidate-020.csv'),_repo_path(ROOT, 'deliverables/reproduction-018-v155.zip'),_repo_path(ROOT, 'deliverables/reproduction-011-014-015-v136.zip')]},'cost':0,'holdout_test_selection':False,'upload':'Parentonly;workerdoesnotupload'}
 write(_repo_path(OUT, 'plan.json'),p);write(_repo_path(ROOT, 'reports/spatial-token-cnn-161-design.json'),p);emit({'locked161':True,'parameters':sum(t.numel() for t in Net().parameters()),'fits':90,'config':CFG,'plan_sha256':sha(_repo_path(OUT, 'plan.json'))})
def check():
 p=read(_repo_path(OUT, 'plan.json'));assert p['config']==CFG and p['seeds']==list(SEEDS)
 for file,h in {**p['hashes'],**p['protected']}.items():assert sha(_repo_path(ROOT, file))==h,file
 return p

def cache():
 check();r=read(_repo_path(ROOT, 'artifacts/patch-attention-092/cache.json'));f=_repo_path(ROOT, 'artifacts/patch-attention-092/patch_tokens.npy');assert sha(f)==r['token_sha256'];tokens=np.load(_repo_io(f),mmap_mode='r');ex=read(_repo_path(ROOT, 'artifacts/spatial-joint-140/extraction.json'));file=_repo_path(ROOT, 'artifacts/spatial-joint-140/features.npz');assert sha(file)==ex['features_sha256']
 with np.load(_repo_io(file),allow_pickle=False) as z:P=z['projection'];assert z['ids'].tolist()==r['ids']
 np.testing.assert_allclose(P.T@P,np.eye(32),atol=1e-14,rtol=0);path=_repo_path(OUT, 'projected.npy');dest=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(400,32,36,36));err=0.;start=time.perf_counter()
 for i,t in enumerate(tokens):
  zz=np.asarray(t,dtype=np.float64)@P;dest[i]=zz.reshape(36,36,32).transpose(2,0,1).astype(np.float32)
  if i in [0,199,399]:
   manual=np.array([sum(float(t[kk,j])*float(P[j,c]) for j in range(768)) for kk,c in [(0,0),(647,13),(1295,31)]]);v=zz[[0,647,1295],[0,13,31]];err=max(err,float(abs(manual-v).max()));assert err<1e-12
 dest.flush();del dest;perm=np.random.default_rng(CFG['permutation_seed']).permutation(1296)
 with (_repo_path(OUT, 'feature-meta.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(r['ids']),projection=P,permutation=perm)
 write(_repo_path(OUT, 'cache.json'),{'shape':[400,32,36,36],'ids':r['ids'],'cache_sha256':sha(path),'metadata_sha256':sha(_repo_path(OUT, 'feature-meta.npz')),'source_token_sha256':r['token_sha256'],'explicit_projection_max_error':err,'seconds':time.perf_counter()-start,'training':False});check();emit({'cache_complete':True,'shape':[400,32,36,36],'seconds':time.perf_counter()-start})

def data(split):
 r=read(_repo_path(OUT, 'cache.json'));assert sha(_repo_path(OUT, 'projected.npy'))==r['cache_sha256'];assert sha(_repo_path(OUT, 'feature-meta.npz'))==r['metadata_sha256'];Z=np.load(_repo_path(OUT, 'projected.npy'),mmap_mode='r');er=read(_repo_path(ROOT, 'artifacts/dinob-062/extraction.json'));file=_repo_path(ROOT, 'artifacts/dinob-062/development_features.npz');assert sha(file)==er['feature_sha256']
 with np.load(_repo_io(file),allow_pickle=False) as z:assert z['ids'].tolist()==r['ids'];X=z['X'].copy()
 with np.load(_repo_path(OUT, 'feature-meta.npz'),allow_pickle=False) as z:perm=z['permutation'].copy();assert sorted(perm.tolist())==list(range(1296))
 path=_repo_path(ROOT, f'artifacts/bootstrap-selector-144/{split}');rr=read(path/'report.json');assert sha(path/'predictions.npz')==rr['predictions_sha256']
 with np.load(_repo_io(path/'predictions.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==r['ids'];y=z['y'].copy();fid=z['fold'].copy();ref=z['primary'].copy()
 manifest=read(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json'));assert set(r['ids'])==set(manifest['development_folds']);g=np.array([manifest['group_by_id'][i] for i in r['ids']]);return {'ids':r['ids'],'X':X,'Z':Z,'perm':perm,'y':y,'fid':fid,'g':g,'reference018':ref}

def normalize(d,T):
 sc=StandardScaler().fit(d['X'][T]);zz=np.array(d['Z'][T],dtype=float);mu=zz.mean((0,2,3));sd=zz.std((0,2,3));sd=np.maximum(sd,1e-6);return {'global_mean':sc.mean_,'global_scale':sc.scale_,'map_mean':mu,'map_scale':sd,'target_mean':float(d['y'][T].mean()),'target_std':float(d['y'][T].std())}
def arrays(d,norm,arm):
 X=((d['X'].astype(float)-norm['global_mean'])/norm['global_scale']).astype(np.float32);Z=((np.asarray(d['Z'],dtype=float)-norm['map_mean'][None,:,None,None])/norm['map_scale'][None,:,None,None]).astype(np.float32)
 if arm=='permuted':Z=Z.reshape(len(Z),32,1296)[:,:,d['perm']].reshape(len(Z),32,36,36)
 return np.ascontiguousarray(X),np.ascontiguousarray(Z)
def tensor(x,device):return torch.tensor(x,dtype=torch.float32,device=device)
def predict(m,X,Z,ix,device='mps',manual=False):
 m.eval();ps=[]
 with torch.inference_mode():
  for first in range(0,len(ix),16):
   j=ix[first:first+16];x=tensor(X[j],device);z=tensor(Z[j],device)
   if manual:
    state=m.state_dict();g=F.gelu(F.linear(x,state['global_fc.weight'],state['global_fc.bias']));h=F.gelu(F.conv2d(z,state['stem.weight'],state['stem.bias']));h=F.gelu(F.conv2d(h,state['conv1.weight'],state['conv1.bias'],padding=1));h=F.gelu(F.conv2d(h,state['conv2.weight'],state['conv2.bias'],padding=1));flat=h.flatten(2);mu=flat.sum(2)/1296;sd=((flat-mu[:,:,None]).square().sum(2)/1296+1e-6).sqrt();v=torch.cat([g,mu,sd],1);p=F.linear(v,state['readout.weight'],state['readout.bias']).squeeze(1)
   else:p=m(x,z)
   ps.append(p.detach().cpu().numpy().astype(float))
 return np.concatenate(ps)

def benchmark():
 check();d=data('original');T=np.flatnonzero(d['fid']!=0);norm=normalize(d,T);times={};stats={}
 for arm in ARMS:
  X,Z=arrays(d,norm,arm);seed_all(SEEDS[0]);m=Net().to('mps');initial=sh(m.state_dict());opt=torch.optim.AdamW(m.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay']);tt=[];m.train();grad=None
  for step in range(12):
   seed_all(SEEDS[0]+1000+step);torch.mps.synchronize();st=time.perf_counter();opt.zero_grad(set_to_none=True);ii=T[:16];loss=F.mse_loss(m(tensor(X[ii],'mps'),tensor(Z[ii],'mps')),tensor((d['y'][ii]-norm['target_mean'])/norm['target_std'],'mps'));loss.backward()
   if step==0:grad={name:float(p.grad.norm()) for name,p in m.named_parameters()};assert all(np.isfinite(v) and v>0 for v in grad.values())
   nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();torch.mps.synchronize()
   if step>=2:tt.append(time.perf_counter()-st)
  times[arm]=float(np.median(tt));stats[arm]={'initial_state':initial,'parameters':sum(p.numel() for p in m.parameters()),'first_gradient_norms':grad,'median_step_seconds':times[arm]};del m,opt;torch.mps.empty_cache()
 assert stats[ARMS[0]]['initial_state']==stats[ARMS[1]]['initial_state'];projected=sum(times.values())*3*15*20*30*1.3;r={'arms':stats,'projected90fits_seconds_with30pct':projected,'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'MPS_driver_bytes':torch.mps.driver_allocated_memory()};r['passed']=projected<1800 and r['RSS_bytes']<3e9 and r['MPS_driver_bytes']<3e9;write(_repo_path(OUT, 'benchmark.json'),r);emit(r);assert r['passed']

def fit(d,split,k,arm,base_seed,path):
 path.mkdir(exist_ok=False);start=time.perf_counter();T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);assert len(T)==320 and len(V)==80 and not set(d['g'][T])&set(d['g'][V]);norm=normalize(d,T);X,Z=arrays(d,norm,arm);y=(d['y']-norm['target_mean'])/norm['target_std'];seed=base_seed+10000*SPLITS.index(split)+k;seed_all(seed);m=Net().to('mps');initial=sh(m.state_dict());opt=torch.optim.AdamW(m.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay']);rng=np.random.default_rng(seed);orders=np.stack([rng.permutation(T) for _ in range(30)]);stream=[];curve=[];grad=None
 for ep,order in enumerate(orders):
  lr=CFG['min_lr']+.5*(CFG['lr']-CFG['min_lr'])*(1+math.cos(math.pi*ep/30))
  for g in opt.param_groups:g['lr']=lr
  m.train();loss_sum=0.
  for first in range(0,len(T),16):
   ii=order[first:first+16];ss=seed+100000*(ep+1)+first;seed_all(ss);stream.append(ss);opt.zero_grad(set_to_none=True);p=m(tensor(X[ii],'mps'),tensor(Z[ii],'mps'));loss=F.mse_loss(p,tensor(y[ii],'mps'));assert torch.isfinite(loss);loss.backward()
   if grad is None:grad={name:float(par.grad.norm()) for name,par in m.named_parameters()};assert all(np.isfinite(v) and v>0 for v in grad.values())
   gn=nn.utils.clip_grad_norm_(m.parameters(),CFG['clip']);assert torch.isfinite(gn);opt.step();loss_sum+=float(loss.detach().cpu())*len(ii)
  curve.append({'epoch':ep+1,'train_online_MSE':loss_sum/len(T),'lr':lr})
 pred=predict(m,X,Z,V)*norm['target_std']+norm['target_mean'];tp=predict(m,X,Z,T)*norm['target_std']+norm['target_mean'];state={name:t.detach().cpu().clone() for name,t in m.state_dict().items()};torch.save({'state':state,'normalization':{name:torch.tensor(v) if isinstance(v,np.ndarray) else v for name,v in norm.items()},'train_indices':T.tolist(),'validation_indices':V.tolist(),'train_ids':[d['ids'][j] for j in T],'validation_ids':[d['ids'][j] for j in V],'arm':arm,'seed':seed,'base_seed':base_seed,'config':CFG},_repo_io(path/'model.pt'))
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train=T,validation=V,predictions=pred,train_predictions=tp)
 r={'arm':arm,'split':split,'fold':k,'base_seed':base_seed,'seed':seed,'initial_state_sha256':initial,'final_state_sha256':sh(state),'order_sha256':ah(orders),'dropout_seed_sha256':ah(np.array(stream)),'train_RMSE':rmse(d['y'][T],tp),'curve':curve,'first_gradient_norms':grad,'model_sha256':sha(path/'model.pt'),'predictions_sha256':sha(path/'predictions.npz'),'seconds':time.perf_counter()-start};write(path/'report.json',r);del m,opt;torch.mps.empty_cache();emit({'fit_complete':str(path.relative_to(ROOT)),'seconds':r['seconds']});return pred,r

def run(split):
 check();assert read(_repo_path(OUT, 'benchmark.json'))['passed'];d=data(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);pred={arm:np.full((3,400),np.nan) for arm in ARMS};hashes={};start=time.perf_counter()
 for k in range(5):
  V=np.flatnonzero(d['fid']==k)
  for si,seed in enumerate(SEEDS):
   rec={}
   for arm in ARMS:
    path=out/f'fold-{k}-{arm}-seed-{seed}';pred[arm][si,V],rec[arm]=fit(d,split,k,arm,seed,path);hashes[path.name]=sha(path/'report.json')
   for key in ['initial_state_sha256','order_sha256','dropout_seed_sha256']:assert rec['spatial'][key]==rec['permuted'][key]
  if split=='original' and k==0:
   b={'first6fits_seconds':time.perf_counter()-start,'projected90fits_seconds':15*(time.perf_counter()-start),'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'MPS_driver_bytes':torch.mps.driver_allocated_memory()};b['passed']=b['projected90fits_seconds']<1800 and b['RSS_bytes']<3e9 and b['MPS_driver_bytes']<3e9;write(_repo_path(OUT, 'wholefold-benchmark.json'),b);emit(b);assert b['passed']
  emit({'split':split,'fold_complete':k})
 P={arm:pred[arm].mean(0) for arm in ARMS};B={arm:.8*d['reference018']+.2*v for arm,v in P.items()}
 with (out/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],reference018=d['reference018'],**{arm+'_seeds':v for arm,v in pred.items()},**{arm+'_raw':v for arm,v in P.items()},**B)
 r={'split':split,'RMSE':{'reference018':rmse(d['y'],d['reference018']),**{arm:rmse(d['y'],v) for arm,v in B.items()},**{arm+'_raw':rmse(d['y'],v) for arm,v in P.items()}},'model_report_hashes':hashes,'predictions_sha256':sha(out/'predictions.npz'),'seconds':time.perf_counter()-start};write(out/'report.json',r);check();emit({'split':split,'RMSE':r['RMSE'],'seconds':r['seconds']})

def validate(split):
 check();d=data(split);out=_repo_path(OUT, split);rr=read(out/'report.json');assert sha(out/'predictions.npz')==rr['predictions_sha256']
 with np.load(_repo_io(out/'predictions.npz'),allow_pickle=False) as z:saved={k:z[k].copy() for k in z.files}
 assert saved['ids'].tolist()==d['ids'];np.testing.assert_array_equal(saved['y'],d['y']);maxcpu=0.;maxfunctional=0.;count=0
 for k in range(5):
  T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);expected=normalize(d,T)
  for si,seed in enumerate(SEEDS):
   paired={}
   for arm in ARMS:
    path=out/f'fold-{k}-{arm}-seed-{seed}';r=read(path/'report.json');assert sha(path/'report.json')==rr['model_report_hashes'][path.name];assert sha(path/'model.pt')==r['model_sha256'] and sha(path/'predictions.npz')==r['predictions_sha256'];obj=torch.load(_repo_io(path/'model.pt'),weights_only=True,map_location='cpu');assert obj['train_indices']==T.tolist() and obj['validation_indices']==V.tolist();assert obj['train_ids']==[d['ids'][j] for j in T];norm={name:v.numpy() if isinstance(v,torch.Tensor) else v for name,v in obj['normalization'].items()}
    for name,v in expected.items():np.testing.assert_array_equal(norm[name],v)
    # Independent raw mean/variance for all fitted global features.
    np.testing.assert_allclose(norm['global_mean'],d['X'][T].mean(0),atol=1e-6,rtol=1e-6) if d['X'].dtype==np.float32 else np.testing.assert_allclose(norm['global_mean'],d['X'][T].mean(0),atol=1e-12,rtol=0)
    X,Z=arrays(d,norm,arm);m=Net();m.load_state_dict(obj['state']);assert sh(m.state_dict())==r['final_state_sha256'];m.to('mps');mp=predict(m,X,Z,V)*norm['target_std']+norm['target_mean'];np.testing.assert_array_equal(mp,saved[arm+'_seeds'][si,V]);m.to('cpu');cpu=predict(m,X,Z,V,'cpu')*norm['target_std']+norm['target_mean'];manual=predict(m,X,Z,V,'cpu',True)*norm['target_std']+norm['target_mean'];err=float(abs(cpu-mp).max());fe=float(abs(cpu-manual).max());assert err<.001 and fe<.001;maxcpu=max(maxcpu,err);maxfunctional=max(maxfunctional,fe);paired[arm]=r;count+=1;del m;torch.mps.empty_cache()
   for key in ['initial_state_sha256','order_sha256','dropout_seed_sha256']:assert paired['spatial'][key]==paired['permuted'][key]
 for arm in ARMS:
  np.testing.assert_array_equal(saved[arm+'_raw'],saved[arm+'_seeds'].mean(0));np.testing.assert_array_equal(saved[arm],.8*d['reference018']+.2*saved[arm+'_raw']);assert np.isfinite(saved[arm]).all();assert rmse(d['y'],saved[arm])==rr['RMSE'][arm]
 v={'passed':True,'models_verified':count,'max_CPU_MPS_error':maxcpu,'max_CPU_functional_forward_error':maxfunctional,'train_only_normalizers_and_groups':True,'paired_initialization_orders_dropout':True,'report_sha256':sha(out/'report.json')};write(_repo_path(OUT, f'validation-{split}.json'),v);check();emit({'split':split,**v})

def poison():
 check();d=data('original');V=np.flatnonzero(d['fid']==0);d['y']=d['y'].copy();d['y'][V]+=100000;records={}
 for arm in ARMS:
  pp,r=fit(d,'original',0,arm,SEEDS[0],_repo_path(OUT, f'outer-poison-{arm}'));path=_repo_path(OUT, 'original', f'fold-0-{arm}-seed-{SEEDS[0]}');old=read(path/'report.json');assert r['final_state_sha256']==old['final_state_sha256']
  with np.load(_repo_io(path/'predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(pp,z['predictions'])
  records[arm]={'state_exact':True,'predictions_exact':True,'train_statistics_excludeV':True}
 write(_repo_path(OUT, 'poison.json'),{'passed':True,'outer80_labels_changed':True,'complete30epoch_replays':records});check();emit({'outer_poison_passed':True})

def summary():
 check();assert read(_repo_path(OUT, 'poison.json'))['passed'];E={k:[] for k in ['reference018',*ARMS]};scores={};ids=None
 for split in SPLITS:
  r=read(_repo_path(OUT, split, 'report.json'));v=read(_repo_path(OUT, f'validation-{split}.json'));assert v['passed'] and v['report_sha256']==sha(_repo_path(OUT, split, 'report.json'));assert sha(_repo_path(OUT, split, 'predictions.npz'))==r['predictions_sha256']
  with np.load(_repo_path(OUT, split, 'predictions.npz'),allow_pickle=False) as z:
   if ids is None:ids=z['ids'].tolist()
   else:assert ids==z['ids'].tolist()
   for k in E:E[k].append((z[k]-z['y'])**2)
  scores[split]=r['RMSE']
 E={k:np.stack(v) for k,v in E.items()};M={k:float(v.mean()) for k,v in E.items()};groups=data('original')['g'];assert len(set(groups))==400;rng=np.random.default_rng(20271223);gains={k:[] for k in ['reference018','permuted']}
 for _ in range(10000):
  ii=rng.integers(0,400,400)
  for k in gains:gains[k].append(float(np.sqrt(E[k][:,ii].mean())-np.sqrt(E['spatial'][:,ii].mean())))
 r={'study':'161 learned spatial token CNN','advance':M['spatial']<min(M['reference018'],M['permuted']),'validation_passed':True,'RMSE_by_partition':scores,'pooled_RMSE':{k:float(np.sqrt(v)) for k,v in M.items()},'pooled_MSE':M,'conditional_ID_group_gain_CI95':{k:np.quantile(v,[.025,.975]).tolist() for k,v in gains.items()},'models_trained':90,'extra_outer_poison_models':2,'current_best':'018','Public_used_for_tuning':False,'same400_adaptive_history':True,'holdout_test_access':False,'new_CSV':False,'cost':0,'plan_sha256':sha(_repo_path(OUT, 'plan.json'))};write(_repo_path(OUT, 'decision.json'),r);write(_repo_path(ROOT, 'reports/spatial-token-cnn-161-summary.json'),r);check();emit(r)
if __name__=='__main__':
 torch.set_num_threads(2);ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','cache','benchmark','poison','summary',*SPLITS,*['validate-'+v for v in SPLITS]]);v=ap.parse_args().mode
 if v in ['prepare','cache','benchmark','poison','summary']:globals()[v]()
 elif v.startswith('validate-'):validate(v[9:])
 else:run(v)
