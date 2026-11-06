"""Approved original400 D2 main+34 verification fits. No partial-fold scoring."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,time,json,hashlib,resource,subprocess,re,traceback,math
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
O=Path(__file__).resolve().parent;S=O.parent;R=_RepoPath(__file__).resolve().parents[4];sys.path.insert(0,str(S));os.environ['TMPDIR']=str(_repo_path(O, 'tmp'));os.environ['TORCH_HOME']=str(_repo_path(O, 'tmp/torch'))
import numpy as np,torch,joblib
from threadpoolctl import threadpool_limits
from contracts import Labels,require,sha,lock_manifest
from data import load_development,graph_for
import backend as b,anchor as a,orchestrate as orch
from bundle import freeze_anchor,residual_targets,save_bundle,load_bundle
threadpool_limits(2);torch.set_num_threads(2)
START=time.perf_counter();PHASE='initial';COUNTS={};RECORDS=[];LAST_CHECK=0.;LAST_HEARTBEAT=0.;MAXMEM={'RSS_peak_bytes':0,'MPS_driver_bytes':0};READS=set();FORBIDDEN=[]
CONTRACT=json.loads((_repo_path(S, 'data-contract.json')).read_text());CODE_LOCK=json.loads((_repo_path(S, 'code-lock.json')).read_text());PROTOCOL_SHA=sha(_repo_path(S, 'protocol.json'));SPLIT_SHA=sha(_repo_path(S, 'data-contract.json'))
for name,fix in json.loads((_repo_path(O, 'authorized-source-fix.json')).read_text()).items():
 require(CODE_LOCK['sources'][name]==fix['old_sha256'],'Unexpected original source');CODE_LOCK['sources'][name]=fix['new_sha256']
def emit(x):print(json.dumps(x,allow_nan=False),flush=True)
def write(path,obj):lock_manifest(path,obj)
def status(extra=None):
 data={'phase':PHASE,'elapsed_seconds':time.perf_counter()-START,'counts':COUNTS,'memory':MAXMEM,'pid':os.getpid(),'complete':False};data.update(extra or {});p=_repo_path(O, 'status.pending');p.write_text(json.dumps(data,indent=2)+'\n');p.replace(_repo_path(O, 'status.json'))
def swap_bytes():
 out=subprocess.check_output(['/usr/sbin/sysctl','vm.swapusage'],text=True);m=re.search(r'used = ([0-9.]+)([MG])',out);require(m is not None,'Cannot read swap usage');return float(m[1])*(1024**2 if m[2]=='M' else 1024**3)
INITIAL_SWAP=swap_bytes();require(INITIAL_SWAP==0,'Swap already used; resource review required')
ORIG_RESOURCE=b.resource_check
def monitor(device='mps'):
 global LAST_CHECK,LAST_HEARTBEAT
 mem=ORIG_RESOURCE(device);MAXMEM.update({k:max(MAXMEM[k],v) for k,v in mem.items()});now=time.perf_counter();require(now-START<1500,'Fixed25minute execution budget exceeded; preserve partial artifacts and report')
 if now-LAST_CHECK>10:require(swap_bytes()<=INITIAL_SWAP,'Swap usage grew; stop');LAST_CHECK=now
 if now-LAST_HEARTBEAT>30:status();emit({'heartbeat':True,'phase':PHASE,'counts':COUNTS,'elapsed_seconds':now-START,'memory':MAXMEM});LAST_HEARTBEAT=now
 return mem
b.resource_check=monitor

def audit(event,args):
 if event.startswith('socket.'):raise RuntimeError('219 main: network forbidden')
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).absolute();mode=args[1];flags=args[2];wr=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
  if wr and not p.is_relative_to(O):raise RuntimeError('Main writes only inside'+str(O))
  if p.is_relative_to(R):
   rel=str(p.relative_to(R));allow=p.is_relative_to(S) or p.is_relative_to(_repo_path(R, '.venv')) or rel in CONTRACT['inputs'] or rel in CONTRACT['protected'] or any(rel==f['path'] for e in CONTRACT['registry'] for f in e['files'].values())
   if not allow:FORBIDDEN.append(rel);raise RuntimeError('Unplanned main project read:'+rel)
   READS.add(rel)
  elif str(p).startswith('/Users/'):raise RuntimeError('Other personal directory forbidden')
sys.addaudithook(audit)

def counter(key,n=1):COUNTS[key]=COUNTS.get(key,0)+n

def schedule_hash(seed,n):
 rng=np.random.default_rng(seed);h=hashlib.sha256()
 for ep in range(80):
  order=rng.permutation(n);h.update(order.tobytes())
  for first in range(0,n,32):h.update(np.array([int(rng.integers(4)),int(rng.random()<.5)],dtype=np.int64).tobytes())
 return h.hexdigest()
ORIG_CNN=b.fit_cnn

def tracked_cnn(f,labels,seed,device='mps',residual=False,anchor_sha256='',target_stats=None):
 monitor();start=time.perf_counter();number=COUNTS.get(PHASE+'_CNN',0)+1;emit({'stage':'CNN_START','phase':PHASE,'number':number,'train_n':len(labels.ids),'seed':seed,'residual':residual});model,curve=ORIG_CNN(f,labels,seed,device,residual,anchor_sha256,target_stats);torch.mps.synchronize();counter(PHASE+'_CNN');counter(PHASE+'_CNN_updates',80*math.ceil(len(labels.ids)/32));path=_repo_path(O, 'CNN', PHASE, f'{number:03d}-{seed}.joblib');path.parent.mkdir(parents=True,exist_ok=True);require(not path.exists(),'Refuse CNN overwrite');meta={'phase':PHASE,'seed':seed,'train':labels.ids.tolist(),'train_ids':f.ids[labels.ids].tolist(),'target_sha256':hashlib.sha256(labels.values.tobytes()).hexdigest(),'normalization':model.normalization,'schedule_sha256':schedule_hash(seed,len(labels.ids)),'state_sha256':b.h.state_hash(model.state),'seconds':time.perf_counter()-start,'curve':curve,'residual':residual};joblib.dump({'model':model,'meta':meta},_repo_io(path));rec={'path':str(path.relative_to(O)),'sha256':sha(path),**{k:v for k,v in meta.items() if k not in ['curve','normalization','train_ids']}};RECORDS.append(rec);emit({'stage':'CNN_DONE','phase':PHASE,'number':number,'seconds':meta['seconds']});return model,curve
b.fit_cnn=tracked_cnn;orch.fit_cnn=tracked_cnn
class Backend(b.ProductionBackend):
 def base(self,f,labels,offset):
  fresh=self.entry(labels,offset) is None;start=time.perf_counter();m=super().base(f,labels,offset)
  if fresh:counter(PHASE+'_KRR',28);counter(PHASE+'_KRR_final');emit({'stage':'BASE_NODE_DONE','phase':PHASE,'train_n':len(labels.ids),'offset':offset,'seconds':time.perf_counter()-start})
  else:counter(PHASE+'_cached_base')
  return m
 def long(self,f,labels,offset,arm):
  monitor();start=time.perf_counter();m=super().long(f,labels,offset,arm);counter(PHASE+'_long_heads',3);monitor();emit({'stage':'LONG_HEAD_DONE','phase':PHASE,'arm':arm,'train_n':len(labels.ids),'seconds':time.perf_counter()-start});return m
 def short(self,f,labels,offset):
  monitor();entry=self.entry(labels,offset);fresh=entry is None or 'short' not in entry['files'];m=super().short(f,labels,offset)
  if fresh:counter(PHASE+'_short_heads')
  else:counter(PHASE+'_cached_short')
  monitor();return m
ORIG_BLOCK=a.fit_block

def tracked_block(*args,**kw):
 m=ORIG_BLOCK(*args,**kw);counter(PHASE+'_classical_nonzero',16);counter(PHASE+'_zero_objects',4);return m
a.fit_block=tracked_block

def equal(x,y,path='root'):
 require(type(x)==type(y),f'Type mismatch {path}: {type(x)} vs{type(y)}')
 if torch.is_tensor(x):require(torch.equal(x,y),'Tensor replay mismatch '+path)
 elif isinstance(x,np.ndarray):require(np.array_equal(x,y,equal_nan=True),'Array replay mismatch '+path)
 elif isinstance(x,dict):
  require(x.keys()==y.keys(),'Dict keys mismatch '+path)
  for k in x:equal(x[k],y[k],path+'.'+str(k))
 elif isinstance(x,(list,tuple)):
  require(len(x)==len(y),'Length mismatch '+path)
  for i,(xx,yy) in enumerate(zip(x,y)):equal(xx,yy,path+f'[{i}]')
 elif hasattr(x,'__dict__'):equal(vars(x),vars(y),path+'.state')
 else:require(bool(x==y),'Value replay mismatch '+path)

def reload_fold(f,k):
 cell=CONTRACT['graphs'][k];folder=_repo_path(O, f'fold-{k}');A,B,V=map(np.array,(cell['A'],cell['B'],cell['V']));anchor=joblib.load(_repo_io(folder/'frozen-anchor.joblib'));require(tuple(anchor.train)==tuple(A),'Anchor training IDs');require(sha(folder/'frozen-anchor.joblib')==json.loads((folder/'frozen-anchor.json').read_text())['anchor_sha256'],'Anchor binding checksum');p0=anchor.predict(f,V);arms={}
 with np.load(_repo_io(folder/'predictions.npz'),allow_pickle=False) as z:require(z['V'].tolist()==V.tolist(),'Fold query IDs');np.testing.assert_array_equal(p0,z['R0']);saved={key:z[key].copy() for key in z.files}
 for arm,key in [('true','R1'),('permuted','Rperm')]:
  package=load_bundle(folder/arm,PROTOCOL_SHA,SPLIT_SHA);m=package.manifest;require(m['anchor_ids']==f.ids[A].tolist() and m['residual_ids']==f.ids[B].tolist(),'Persisted sample ID mapping');np.testing.assert_array_equal(package.predict(f,V),saved[key]);arms[arm]=m
 report={'passed':True,'fold':k,'V':V.tolist(),'anchor_sha256':sha(folder/'frozen-anchor.joblib'),'predictions_sha256':sha(folder/'predictions.npz'),'training_report_sha256':sha(folder/'training-report.json'),'bundles':{arm:sha(folder/arm/'manifest.json') for arm in arms},'reload_exact':True,'sample_ID_mapping_checked':True};write(folder/'validation.json',report);return report

def run_fold(f,labels,k):
 global PHASE
 PHASE='main';g=CONTRACT['graphs'][k];out=_repo_path(O, f'fold-{k}');bk=Backend(R,CONTRACT['registry'],'mps',_repo_path(O, f'nodes-fold-{k}'));start=time.perf_counter();orch.fit_pair(f,labels.restrict(g['T']),graph_for(k),g['B'],g['V'],bk,out,k,PROTOCOL_SHA,SPLIT_SHA);counter('main_affines',4);report=reload_fold(f,k);emit({'stage':'FOLD_COMPLETE_NO_SCORE','fold':k,'seconds':time.perf_counter()-start,'counts':COUNTS});return report

def verify_first(f,labels):
 global PHASE
 PHASE='verification';g=CONTRACT['graphs'][0];A,B,V=map(np.array,(g['A'],g['B'],g['V']));p=_repo_path(O, 'fold-0');original=joblib.load(_repo_io(p/'frozen-anchor.joblib'));binding=json.loads((p/'frozen-anchor.json').read_text());folder=_repo_path(O, 'verification');folder.mkdir();poison=labels.values.copy();poison[B]=12345.;poison[V]=np.nan;restricted=Labels(A,poison[A]);bk=Backend(R,CONTRACT['registry'],'mps',folder/'poison-nodes');start=time.perf_counter();replay=a.fit_complete_anchor(f,restricted,graph_for(0),bk);counter('verification_affines',4);equal(original,replay);np.testing.assert_array_equal(original.predict(f,np.r_[B,V]),replay.predict(f,np.r_[B,V]));freeze_anchor(replay,folder/'poison-anchor.joblib',PROTOCOL_SHA);emit({'stage':'FULL_ANCHOR_POISON_EXACT','seconds':time.perf_counter()-start,'counts':COUNTS})
 r=residual_targets(f,B,labels.restrict(B),p/'frozen-anchor.joblib',binding);stats=(float(r.values.mean()),max(float(r.values.std()),1e-6));perm=np.random.default_rng(20264050).permutation(len(B));targets={'true':r,'permuted':Labels(B,r.values[perm])}
 for arm in ['true','permuted']:
  outerpoison=labels.values.copy();outerpoison[V]=np.nan;rp=residual_targets(f,B,Labels(B,outerpoison[B]),p/'frozen-anchor.joblib',binding)
  if arm=='permuted':rp=Labels(B,rp.values[perm])
  np.testing.assert_array_equal(rp.values,targets[arm].values);m,curve=tracked_cnn(f,rp,20264000,'mps',True,binding['anchor_sha256'],stats);saved=joblib.load(_repo_io(p/arm/'residual.joblib'));equal(saved,m);np.testing.assert_array_equal(saved.predict(f,V),m.predict(f,V));joblib.dump(m,_repo_io(folder/f'{arm}-outer-poison.joblib'))
 # Three representative deepest CNN-only clean repeats; no extra KRR or readout fits.
 e=graph_for(0).cells[0]['deep'][0];E=np.array(e['E']);key=hashlib.sha256(E.tobytes()).hexdigest()[:16];saved=joblib.load(_repo_path(O, 'nodes-fold-0', f"base-{e['offset']}-{key}.joblib"))['model'];clean=[]
 for seed,ref in zip(b.CNN_SEEDS,saved.cnn.models):
  m,curve=tracked_cnn(f,labels.restrict(E),seed+e['offset'],'mps');equal(ref,m);np.testing.assert_array_equal(ref.predict(f,np.array(e['F'])),m.predict(f,np.array(e['F'])));clean.append(seed)
 for arm in ['true','permuted']:
  m,curve=tracked_cnn(f,targets[arm],20264000,'mps',True,binding['anchor_sha256'],stats);ref=joblib.load(_repo_io(p/arm/'residual.joblib'));equal(ref,m);np.testing.assert_array_equal(ref.predict(f,V),m.predict(f,V));joblib.dump(m,_repo_io(folder/f'{arm}-clean.joblib'))
 require(COUNTS['verification_CNN']==34 and COUNTS['verification_CNN_updates']==8480,'Verification CNN counts');report={'passed':True,'complete_anchor_B_finite_V_NaN_poison_exact':True,'all_anchor_supervised_stages_retrained':True,'cached_nodes_membership_hash_certificates_reused':True,'residual_outerV_NaN_poison_both_arms_exact':True,'deepest_clean_CNN_seeds':clean,'residual_clean_both_arms_exact':True,'semantic_states_and_normalizations_exact':True,'prediction_exact':True,'tolerance':0.,'counts':{k:v for k,v in COUNTS.items() if k.startswith('verification_')},'seconds':time.perf_counter()-start};write(folder/'validation.json',report);emit({'stage':'ALL34_VERIFICATION_PASSED','seconds':report['seconds'],'counts':COUNTS})

def main():
 global PHASE
 require(not (_repo_path(O, 'started.json')).exists(),'Run already started: inspect logs/process before any restart');require(json.loads((_repo_path(O, 'probe.json')).read_text())['within_25min_forecast'],'Probe resource gate');require(Path(sys.prefix).resolve()==_repo_path(R, '.venv'),'Wrong Python environment')
 for rel,digest in CODE_LOCK['sources'].items():require(sha(_repo_path(S, rel))==digest,'219 original source changed')
 write(_repo_path(O, 'started.json'),{'pid':os.getpid(),'UTC':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'authorization':'Parent explicitly authorized original5fold145CNN+34verification localCPU/MPS only on2026-10-07; no new installation/cost/test/full500','source_lock_sha256':sha(_repo_path(O, 'main-code-lock.json')),'initial_swap_bytes':INITIAL_SWAP,'resource_budget_seconds':1500,'same_device_replay_tolerance':0.});b.MAIN_TRAINING_ENABLED=True
 try:
  f,labels,contract=load_development();allreports=[];allreports.append(run_fold(f,labels,0));verify_first(f,labels)
  for k in range(1,5):allreports.append(run_fold(f,labels,k))
  PHASE='final-validation';expected={'main_CNN':145,'main_CNN_updates':35600,'verification_CNN':34,'verification_CNN_updates':8480,'main_long_heads':285,'main_short_heads':15,'main_KRR':1260,'main_KRR_final':45,'main_classical_nonzero':240,'main_affines':20,'verification_long_heads':57,'verification_short_heads':3,'verification_KRR':252,'verification_KRR_final':9,'verification_classical_nonzero':48,'verification_affines':4}
  for key,n in expected.items():require(COUNTS.get(key)==n,f'Count mismatch {key}:{COUNTS.get(key)}!={n}')
  seen=[];preds={k:np.full(400,np.nan) for k in ['R0','R1','Rperm']};fold=np.full(400,-1)
  for k,report in enumerate(allreports):
   V=np.array(report['V']);require(V.tolist()==CONTRACT['graphs'][k]['V'],'Locked V mismatch');seen.extend(V);fold[V]=k
   with np.load(_repo_path(O, f'fold-{k}', 'predictions.npz'),allow_pickle=False) as z:
    for key in preds:preds[key][V]=z[key]
  require(sorted(seen)==list(range(400)),'Not complete unique development400');require(all(np.isfinite(p).all() for p in preds.values()),'Missing OOF');np.savez_compressed(_repo_path(O, 'oof-unscored.npz'),ids=f.ids,fold=fold,**preds)
  for rel,digest in CONTRACT['protected'].items():require(sha(_repo_path(R, rel))==digest,'Protected best artifact changed')
  for rel,digest in CODE_LOCK['sources'].items():require(sha(_repo_path(S, rel))==digest,'Original219 source changed')
  write(_repo_path(O, 'fit-completion.json'),{'passed':True,'counts':COUNTS,'fold_validation_sha256':{str(k):sha(_repo_path(O, f'fold-{k}', 'validation.json')) for k in range(5)},'poison_validation_sha256':sha(_repo_path(O, 'verification/validation.json')),'oof_sha256':sha(_repo_path(O, 'oof-unscored.npz')),'seconds':time.perf_counter()-START,'memory':MAXMEM,'swap_final_bytes':swap_bytes(),'independent_replay_and_scoring_pending':True,'original400_only':True,'forbidden_reads':FORBIDDEN});write(_repo_path(O, 'CNN-records.json'),RECORDS);write(_repo_path(O, 'access.json'),{'reads':sorted(READS),'forbidden':FORBIDDEN});status({'complete':True});emit({'stage':'ALL_FITS_AND_INTERNAL_VALIDATION_COMPLETE','counts':COUNTS,'seconds':time.perf_counter()-START,'memory':MAXMEM})
 except Exception as ex:
  write(_repo_path(O, 'failure.json'),{'error':repr(ex),'traceback':traceback.format_exc(),'phase':PHASE,'counts':COUNTS,'seconds':time.perf_counter()-START,'memory':MAXMEM});status({'failed':True});raise
if __name__=='__main__':main()
