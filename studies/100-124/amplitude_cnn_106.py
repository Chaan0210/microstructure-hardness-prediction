"""Matched native CNN: samplewise GroupNorm versus channel affine only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,hashlib,time,datetime,argparse,resource,copy
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));assert Path(sys.prefix).resolve()==_repo_path(ROOT, '.venv')
import numpy as np,torch
from torch import nn
from PIL import Image
import cnn_controls as old
OUT=_repo_path(ROOT, 'artifacts/amplitude-cnn-106');SPLITS=('original','20261031','20261101');ARMS=('groupnorm','affine');CFG={'epochs':80,'batch':32,'lr':.002,'min_lr':.0001,'weight_decay':.01,'clip':5.,'average_last':5,'seed':20261003,'size':256,'blend':.1}
DEADLINE='2026-10-03T18:34:55+00:00'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def emit(x):print(json.dumps(x,ensure_ascii=False),flush=True)
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
def statehash(st):
 h=hashlib.sha256()
 for n,v in sorted(st.items()):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def timecheck():assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(DEADLINE),'Authorized2h batchdeadline reached'
class ChannelAffine(nn.Module):
 def __init__(self,c):super().__init__();self.weight=nn.Parameter(torch.ones(c));self.bias=nn.Parameter(torch.zeros(c))
 def forward(self,x):return x*self.weight[None,:,None,None]+self.bias[None,:,None,None]
class Net(old.TinyCNN):
 def __init__(self,arm):
  super().__init__();self.arm=arm
  if arm=='affine':
   for i,m in enumerate(self.features):
    if isinstance(m,nn.GroupNorm):self.features[i]=ChannelAffine(m.num_channels)
  elif arm!='groupnorm':raise ValueError(arm)
 def descriptor(self,x):
  h=self.features(x);return torch.cat([h.mean((2,3)),(h.var((2,3),correction=0)+1e-6).sqrt(),x.mean((2,3)),(x.var((2,3),correction=0)+1e-6).sqrt()],1)
 def forward(self,x):return self.head(self.descriptor(x)).squeeze(1)

def check():
 p=read(_repo_path(OUT, 'plan.json'))
 for n,h in {**p['sources'],**p['inputs'],**p['protected011']}.items():assert sha(_repo_path(ROOT, n))==h,n
 return p

def prepare():
 OUT.mkdir(exist_ok=False);timecheck();manifest=read(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json'));ids=sorted(manifest['development_folds']);assert len(ids)==400 and not set(ids)&set(manifest['holdout'])
 sources=['amplitude_cnn_106.py','validate_amplitude_cnn_106.py','cnn_controls.py'];inputs=['artifacts/run-001/split_manifest.json','reports/candidate011-public-submission.json','reports/cnn-structure-035-summary.json','reports/cnn-cov-controls-036-summary.json','reports/frozen-complement-085-validation.json']+[f'artifacts/frozen-complement-085/{s}/development_predictions.npz' for s in SPLITS]
 p={'study':'106 matchedsamplewise normalization removal in native scratchCNN','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_utc':DEADLINE,'config':CFG,'hypothesis':'AllpreviousTinyCNN/moment/covariance controls retainGroupNorm. Its per-imagefeature centering/scaling can attenuate absoluteactivationamplitude andintensity-dependent spatialinteractions; rawmean/std skips do notinvert allhidden normalization. ReplaceonlyGroupNorm by learnedchannel-affine scalebias toretain activationamplitude. This mayinstead destabilizetraining orencode nuisance; targetutility isunknown.',
 'duplicate_audit':'009/010/035 testedresolution;011 testedallstage moments;035/036 covariance/rank/shrinkage;alluseGroupNorm. No paired GroupNorm-versus-affine-only scratchCNN found. Raw pixels, globaltrainfold inputnormalization, andmean/std skipalreadyexisted; donotclaim those asnew.',
 'architecture':'SameTinyCNN Conv(1,16,k5,stride2),thenConv16->32->64->96 k3,stride2. Aftereachconv GroupNorm4+SiLU versus ChannelAffine(gamma,beta)+SiLU. Same194 mean/std+rawmean/std descriptor, Linear194->64 SiLU Dropout.15 Linear64->1. Identical trainableparametercount andinitialparameter tensors; onlynormalization function differs. Native256 unresizedgrayscale, no intensity jitter orperimageinput normalization.',
 'training':'Both freshsame seed20261003+outerfold.80epochs,AdamW.002/.01,cosineeta_min.0001,clip5,batch32,MSEstandardtarget. Last5stateaverage fixedendpoint. SameNumPy orders andbatchD4augmentation, sameperbatchtorchdropoutseed. Allinput/target mean/std onlyoutertrain. No innergrid,epochselection ornewpretraining.',
 'formula':'Forboth arms .9*samefold011+.1*CNN; onlyaffine isprimary. Nofittedfusion. DINO-backed011 remainsfixed.',
 'original_rule':'Affineblend muststrictlyimprove011 andgroupnormblend pooledRMSE; standalone andfoldcounts reported, notextra gates. Ifpass andindependentvalidationpass runbothknownresplits withbotharms unchanged. Allthreeprimarycriteria required forpromotion. No fallbackgroupnormcandidate ornormalization/lr/epoch sweep.',
 'representation_review':'CNNstride16 andglobalmoments can lose fineorder; DINOglobalmoments retaincontextualposition butnotinvertiblegeometry. No realgraintruth establishestruegeometrypreservation. Onexisting6 knowngeometryfixtures, compare learned descriptors betweensamegeometry/noisechanged versus differentgeometry atmatchednoise usingfoldtrain descriptorstandardization. Diagnostic only, no synthetictraining or selection.',
 'resources':'100MPS steps perarm benchmark onoriginalfold0train32; estimatesinclude20%overhead. Bound scope by2hdeadline; MPS/RSS4GB practicallimit. CPUthreads2,existinguvpackagesonly.',
 'development_ids':ids,'sources':{n:sha(_repo_path(ROOT, n)) for n in sources},'inputs':{n:sha(_repo_path(ROOT, n)) for n in inputs},'protected011':read(_repo_path(ROOT, 'reports/candidate011-public-submission.json'))['protected_sha256'],'scope':'No train.csv labels outside400,noholdout/test,nonewpackage,externaldata,rawtransmission,cost. Syntheticfixturesdiagnosticsonly.','bias':'Same400 repeatedadaptive research; matchedcontrol doesnoterasehistoricalselectionbias orprovephysicalcausality.','cost':0}
 write(_repo_path(OUT, 'plan.json'),p);oldhash=read(_repo_path(ROOT, 'artifacts/dinob-062/extraction.json'))['image_sha256'];arr=[]
 for ident in ids:
  path=_repo_path(ROOT, 'data/train', f'{ident}.png');assert sha(path)==oldhash[ident]
  with Image.open(_repo_io(path)) as im:assert im.mode=='L' and im.size==(256,256);arr.append(np.asarray(im,dtype=np.uint8).copy())
 with (_repo_path(OUT, 'images.npy')).open('xb') as f:np.save(_repo_io(f),np.stack(arr)[:,None],allow_pickle=False)
 write(_repo_path(OUT, 'image-cache.json'),{'shape':[400,1,256,256],'sha256':sha(_repo_path(OUT, 'images.npy')),'ids':ids,'labels_used':False});check();emit({'prepared':True,'plan_sha256':sha(_repo_path(OUT, 'plan.json'))})

def inputs(split):
 p=check();timecheck();m=read(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json'));ids=p['development_ids'];v=read(_repo_path(ROOT, 'reports/frozen-complement-085-validation.json'));path=_repo_path(ROOT, f'artifacts/frozen-complement-085/{split}/development_predictions.npz');assert sha(path)==v['results'][split]['predictions_sha256']
 with np.load(_repo_io(path),allow_pickle=False) as z:assert z['ids'].tolist()==ids;y=z['y'].copy();fid=z['fold'].copy();ref=z['primary'].copy()
 g=np.array([m['group_by_id'][i] for i in ids]);sp=m if split=='original' else read(_repo_path(ROOT, f'artifacts/frozen-validation-021/seed-{split}/split_manifest.json'));np.testing.assert_array_equal(fid,[sp['development_folds'][i] for i in ids]);meta=read(_repo_path(OUT, 'image-cache.json'));assert sha(_repo_path(OUT, 'images.npy'))==meta['sha256'];X=np.load(_repo_path(OUT, 'images.npy'),allow_pickle=False,mmap_mode='r')
 for k in range(5):assert np.sum(fid==k)==80 and not set(g[fid==k])&set(g[fid!=k])
 return {'ids':ids,'X':X,'y':y,'fid':fid,'g':g,'ref':ref}
def norms(d,tr):
 x=np.asarray(d['X'][tr],dtype=np.float64)/255;return {'x_mean':float(x.mean()),'x_std':float(x.std()),'y_mean':float(d['y'][tr].mean()),'y_std':float(d['y'][tr].std())}
def batch(X,ix,n,device='mps',rot=0,flip=False):
 x=(np.asarray(X[ix],dtype=np.float32)/np.float32(255)-np.float32(n['x_mean']))/np.float32(n['x_std']);x=np.rot90(x,rot,axes=(2,3));x=x[:,:,:,::-1] if flip else x;return torch.tensor(np.ascontiguousarray(x),device=device)
def predict(m,d,ix,n,device='mps',descriptors=False):
 m.eval();out=[]
 with torch.no_grad():
  for first in range(0,len(ix),32):
   x=batch(d['X'],ix[first:first+32],n,device);z=m.descriptor(x) if descriptors else m(x);out.append(z.cpu().numpy().astype(float))
 arr=np.concatenate(out);return arr if descriptors else arr*n['y_std']+n['y_mean']
def benchmark():
 check();d=inputs('original');tr=np.flatnonzero(d['fid']!=0);n=norms(d,tr);torch.set_num_threads(2);records={};states={}
 for arm in ARMS:
  torch.manual_seed(CFG['seed']);m=Net(arm).to('mps');states[arm]=statehash(m.state_dict());opt=torch.optim.AdamW(m.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay']);x=batch(d['X'],tr[:32],n);t=torch.tensor((d['y'][tr[:32]]-n['y_mean'])/n['y_std'],dtype=torch.float32,device='mps');times=[];m.train()
  for step in range(100):
   torch.manual_seed(CFG['seed']+step);torch.mps.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True);loss=nn.functional.mse_loss(m(x),t);assert torch.isfinite(loss);loss.backward();gn=nn.utils.clip_grad_norm_(m.parameters(),5);assert torch.isfinite(gn);opt.step();torch.mps.synchronize()
   if step>=5:times.append(time.perf_counter()-start)
  records[arm]={'steps':100,'median_step_seconds':float(np.median(times)),'parameters':sum(z.numel() for z in m.parameters()),'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory(),'finite':True};del m,opt,x,t;torch.mps.empty_cache();emit({'benchmark_arm':arm,**records[arm]})
 assert states['groupnorm']==states['affine'];assert records['groupnorm']['parameters']==records['affine']['parameters'];estimate=sum(z['median_step_seconds'] for z in records.values())*800*5*1.2;remaining=(datetime.datetime.fromisoformat(DEADLINE)-datetime.datetime.now(datetime.timezone.utc)).total_seconds();r={'arms':records,'initial_state_identical':True,'projected_original_seconds':estimate,'remaining_batch_seconds':remaining,'passed':estimate<remaining-120 and max(max(z['rss_peak_bytes'],z['mps_driver_bytes']) for z in records.values())<4e9};write(_repo_path(OUT, 'benchmark.json'),r);emit(r);assert r['passed']

def schedule(k):
 rng=np.random.default_rng(CFG['seed']+k);orders=[];augs=[]
 for ep in range(CFG['epochs']):
  order=rng.permutation(320);orders.append(order);augs.append([(int(rng.integers(4)),bool(rng.random()<.5)) for _ in range(10)])
 return np.array(orders),np.array(augs)

def fit(d,k,arm,path):
 timecheck();path.mkdir(exist_ok=False);torch.set_num_threads(2);tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);n=norms(d,tr);torch.manual_seed(CFG['seed']+k);m=Net(arm).to('mps');initial=statehash(m.state_dict());opt=torch.optim.AdamW(m.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay']);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=80,eta_min=.0001);orders,augs=schedule(k);start=time.perf_counter();avg=None;curve=[]
 for ep in range(80):
  timecheck();m.train();losses=[]
  for first in range(0,320,32):
   ix=tr[orders[ep,first:first+32]];rot,flip=augs[ep,first//32];x=batch(d['X'],ix,n,rot=int(rot),flip=bool(flip));target=torch.tensor((d['y'][ix]-n['y_mean'])/n['y_std'],dtype=torch.float32,device='mps');torch.manual_seed(CFG['seed']+k+10000*(ep+1)+first);opt.zero_grad(set_to_none=True);loss=nn.functional.mse_loss(m(x),target);assert torch.isfinite(loss);loss.backward();gn=nn.utils.clip_grad_norm_(m.parameters(),5.);assert torch.isfinite(gn);opt.step();losses.append(float(loss.detach()))
  sched.step();curve.append({'epoch':ep+1,'augmented_train_MSE':float(np.mean(losses)),'lr':sched.get_last_lr()[0]})
  if ep>=75:
   state={a:v.detach().cpu().clone() for a,v in m.state_dict().items()}
   if avg is None:avg={a:v/5 for a,v in state.items()}
   else:
    for a in avg:avg[a]+=state[a]/5
  if (ep+1)%20==0:emit({'split':path.parent.name,'fold':k,'arm':arm,'epoch':ep+1,'seconds':time.perf_counter()-start})
 m.load_state_dict(avg);pv=predict(m,d,va,n);trainp=predict(m,d,tr,n);assert np.isfinite(pv).all();torch.save({'state':avg,'arm':arm,'config':CFG,'normalization':n,'train_ids':[d['ids'][i] for i in tr],'validation_ids':[d['ids'][i] for i in va],'initial_state_sha256':initial},_repo_io(path/'model.pt'))
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=tr,validation_indices=va,train_predictions=trainp,predictions=pv)
 record={'arm':arm,'fold':k,'initial_state_sha256':initial,'final_state_sha256':statehash(avg),'order_sha256':hashlib.sha256(orders.tobytes()).hexdigest(),'augmentation_sha256':hashlib.sha256(augs.tobytes()).hexdigest(),'train_rmse':rmse(d['y'][tr],trainp),'curve':curve,'seconds':time.perf_counter()-start,'checkpoint_sha256':sha(path/'model.pt'),'predictions_sha256':sha(path/'predictions.npz'),'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory()};write(path/'report.json',record);del m,opt;torch.mps.empty_cache();return pv,record

def compare(y,old,new,fid):
 gains=[rmse(y[fid==k],old[fid==k])-rmse(y[fid==k],new[fid==k]) for k in range(5)];return {'gain':rmse(y,old)-rmse(y,new),'fold_gains':gains,'positive_folds':sum(z>0 for z in gains)}
def run(split):
 check();assert read(_repo_path(OUT, 'benchmark.json'))['passed']
 if split!='original':assert read(_repo_path(OUT, 'original/report.json'))['advance'] and read(_repo_path(ROOT, 'reports/amplitude-cnn-106-original-validation.json'))['passed']
 d=inputs(split);dest=_repo_path(OUT, split);dest.mkdir(exist_ok=False);start=time.perf_counter();pred={a:np.full(400,np.nan) for a in ARMS};records={}
 for k in range(5):
  rr={}
  for arm in ARMS:pred[arm][d['fid']==k],rr[arm]=fit(d,k,arm,dest/f'{arm}-fold{k}')
  for key in ('initial_state_sha256','order_sha256','augmentation_sha256'):assert rr['groupnorm'][key]==rr['affine'][key]
  records[str(k)]={a:sha(dest/f'{a}-fold{k}/report.json') for a in ARMS};emit({'paired_fold_complete':k,'split':split,'seconds':time.perf_counter()-start})
 ps={'reference011':d['ref'],**pred,**{a+'_blend':.9*d['ref']+.1*z for a,z in pred.items()}};comp=compare(d['y'],ps['reference011'],ps['affine_blend'],d['fid']);pair=compare(d['y'],ps['groupnorm_blend'],ps['affine_blend'],d['fid']);advance=comp['gain']>0 and pair['gain']>0
 with (dest/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],**ps)
 r={'split':split,'scores':{a:{'rmse':rmse(d['y'],z),'fold_rmse':[rmse(d['y'][d['fid']==k],z[d['fid']==k]) for k in range(5)]} for a,z in ps.items()},'affineblend_vs011':comp,'affineblend_vsgroupnormblend':pair,'affine_vsgroupnorm':compare(d['y'],pred['groupnorm'],pred['affine'],d['fid']),'advance':advance,'fold_report_sha256':records,'predictions_sha256':sha(dest/'development_predictions.npz'),'plan_sha256':sha(_repo_path(OUT, 'plan.json')),'seconds':time.perf_counter()-start,'candidate_created':False};write(dest/'report.json',r);check();emit(r)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','benchmark',*SPLITS]);a=ap.parse_args();{'prepare':prepare,'benchmark':benchmark}.get(a.mode,lambda:run(a.mode))()
