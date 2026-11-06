"""Fixed rank4 patch-token adapter: post-embedding versus post-block12 control."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,time,math,resource,hashlib,argparse,datetime,gc
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import frozen_seed_088 as prior
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from sklearn.preprocessing import StandardScaler
e=prior.e;fv=e.fv;backbone=e.backbone;h=e.h;OUT=_repo_path(ROOT, 'artifacts/early-adapter-095')
CFG={'rank':4,'hidden':32,'dropout':.5,'epochs':12,'microbatch':1,'effective_batch':16,'head_lr':3e-4,'head_min_lr':1e-5,'head_weight_decay':.1,'adapter_lr':1e-4,'adapter_min_lr':1e-5,'adapter_weight_decay':0.,'clip_head':1.,'clip_adapter':1.,'seed':20270215,'prediction_anchor':0.,'blend_new':.1}
ARMS=('frozen','late','early');TOP=('TRAIN_000438','TRAIN_000331','TRAIN_000227','TRAIN_000071','TRAIN_000168')

def inputs(split):return prior.inputs(split)
def seed_for(k,ep,first):return CFG['seed']+k+10000*(ep+1)+first

def prepare():
 pub=prior.diag.check_best();OUT.mkdir(exist_ok=False)
 p={'study':'095 early versus late rank4 adapter','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':'One fixedearlyexpression adaptation hypothesis, benchmarkthenoriginal,only qualifyingoriginal advances toboth knownresplits.','novelty':'080 insertsrank4 inlastattention withfixedoldheads;084 trainslastblockwithfreshhead. Bothfreeze/detach first11blocks. Here pretrainedblockweightsstayfrozen but earlyadapterchanges inputtoall12blocks and receivesgradient throughall12.','config':CFG,'architecture':'Bias-free V:768->4,U:4->768; z+U(Vz),6144parameters, no extra scaling/nonlinearity. V Kaiminguniform,Uzero. Botharms identicaladapterstate withseed20270215+fold+500000. Patchtokens only inbotharms: earlyimmediatelyafterpatch_embed beforeCLS/position;lateafterblock12 beforefinalLayerNorm,CLSleftunchanged. Allpretrainedweights/norms/embeddings frozen/eval. Headidentical084 Linear6912->32 GELU Dropout.5 Linear32->1. Block6/9/12 normalizedCLS,patchmean,populationstd.','fixed_optimization_choice':'Head/12epochs/loss/seed/order same084. Bothadapters lr1e-4 cosine->1e-5,wd0,clip1, fixedbeforebenchmark; not a sweep. Micro1accumulate16 allthree arms peruserresourceinstruction; therefore frozencontrol refit,notassumed equaltohistoricalmicro2 011. Noaugment/anchor/earlystop.','zero_init_note':'Uzero makesfunctionidentity. Vgradient expectedzero atstep0; U mustreceivegradient, thenbothmustreceivegradientafter firstupdate. Never requirebothgradientsnonzero beforeU moves.','normalization':'Same foldtrainfrozenB StandardScaler and foldtrain targetmean/std heldfixed. Neverfit scaler tovalidation/adaptedfullcohort.','graph_and_cache':'Early always rawimage ->patch_embed->adapter->all12blocks withautograd; no inference_mode,no prefixdetachcache,no detachedintermediates. Lateonly mayuseverified080 prelastblockprefix andfrozenfirst6/9 descriptors: upstreamfixed and proven equivalentto fullrawlatepath. FrozenusesverifiedBfeatures. no_grad onlyforlegitimatelyfixedlateprefix or prediction;neverearlytraining.','comparison':'.9*samefold009+.1*newbranch;replace011branch,notblendagainwith011. No calibrationfit orratiochoice.','original_gate':'Early standalone andblend muststrictlybeat BOTH late and frozen; earlyblend improves011 RMSE by>=0.10 and>=4/5positivefolds. Otherwiseterminate thisdirection. This is practicalresource/promotionrule,notstatisticalproof. Latecannotbecome fallbackcandidate.','resplit_gate':'Iforiginalpasses, unchangedbothresplits musteach satisfy thesameoriginalgate pluspositive earlyblend-vs011 gain onfixed395excluding original093top5; requirefixed395gainpositive inoriginaltoo foranypromotion. No newtop5selection. All3reusesame400,notindependentconfirmation.','resources':'ExistingCPU2/MPS/uv. Benchmarkmicro1/effective16,12epochs,320train/fold. Project all3arms5fold train plus prediction with25%margin. If >10800s (3hours) orconservativeRSSpeak+MPSdriver>8e9 stop. Checkpointonlymemoryfallback allowed ifneeded; neverrank/layer/epoch/lrsearch. Fullbenchmark dataonlyoriginalfold0trainrows; noouterperformanceused.','sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['early_adapter_095.py','dinob_joint_084.py','dinob_adapter_080.py','frozen_seed_088.py']},'evidence':{n:fv.sha(_repo_path(ROOT, n)) for n in ['reports/candidate011-public-submission.json','reports/dinob-062-validation.json','artifacts/dinob-adapter-080/prefix.json','reports/patch-replication-093-summary.json']},'protected011':pub['protected_sha256'],'scope':'Development400only;noholdout/testselection,install/download/cost/submission. Best011preserved.'};fv.write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'config':CFG,'gate':p['original_gate']}),flush=True)

def check():
 p=json.loads((_repo_path(OUT, 'plan.json')).read_text());assert p['config']==CFG
 for n,hh in {**p['sources'],**p['evidence'],**p['protected011']}.items():assert fv.sha(_repo_path(ROOT, n))==hh,n
 return p

class Adapter(nn.Module):
 def __init__(self,seed):
  super().__init__();torch.manual_seed(seed+500000);self.V=nn.Linear(768,4,bias=False);self.U=nn.Linear(4,768,bias=False);nn.init.zeros_(self.U.weight)
 def forward(self,z):return z+self.U(self.V(z))

class Model(nn.Module):
 def __init__(self,m,sc,seed,arm,checkpointing=False):
  super().__init__();self.backbone=m;self.arm=arm;self.checkpointing=checkpointing;torch.manual_seed(seed);self.head=e.Head();self.adapter=Adapter(seed);self.register_buffer('mean',torch.tensor(sc.mean_,dtype=torch.float32));self.register_buffer('scale',torch.tensor(sc.scale_,dtype=torch.float32));self.backbone.requires_grad_(False);self.adapter.requires_grad_(arm!='frozen');self.eval();self.traces=[]
 def training_mode(self):self.eval();self.head.train()
 def pool(self,z):
  zz=self.backbone.norm(z);p=zz[:,1:];return torch.cat((zz[:,0],p.mean(1),p.std(1,correction=0)),1)
 def raw_features(self,x,trace=False):
  m=self.backbone;w,hh=x.shape[-2:];z=m.patch_embed(x)
  if self.arm=='early':z=self.adapter(z)
  if trace:z.retain_grad();self.traces=[z]
  z=torch.cat((m.cls_token.expand(z.shape[0],-1,-1),z),1);z=z+m.interpolate_pos_encoding(z,w,hh);assert m.register_tokens is None
  feats=[]
  for j,block in enumerate(m.blocks):
   z=checkpoint(block,z,use_reentrant=False,preserve_rng_state=True) if self.checkpointing and torch.is_grad_enabled() and self.arm=='early' else block(z)
   if trace:z.retain_grad();self.traces.append(z)
   if j==11 and self.arm=='late':z=torch.cat((z[:,:1],self.adapter(z[:,1:])),1)
   if j in (5,8,11):feats.append(self.pool(z))
  return torch.cat(feats,1)
 def features(self,raw=None,B=None,prefix=None):
  if self.arm=='early':assert raw is not None and prefix is None;return self.raw_features(raw)
  if self.arm=='frozen':assert B is not None;return B
  assert B is not None and prefix is not None
  with torch.no_grad():z=self.backbone.blocks[-1](prefix)
  z=torch.cat((z[:,:1],self.adapter(z[:,1:])),1);return torch.cat((B[:,:4608],self.pool(z)),1)
 def forward(self,raw=None,B=None,prefix=None):return self.head((self.features(raw,B,prefix)-self.mean)/self.scale)

def tensor(x):return torch.tensor(np.array(x,copy=True),dtype=torch.float32,device='mps')
def rawimage(d,i):
 assert d['ids'][i] in d['ids'];return torch.tensor(backbone.ds.inputs(d['ids'][i],'full504'),device='mps')
def batch(model,d,tokens,i):
 if model.arm=='early':return model(raw=rawimage(d,i))
 if model.arm=='late':return model(B=tensor(d['B'][i:i+1]),prefix=tensor(tokens[i:i+1]))
 return model(B=tensor(d['B'][i:i+1]))

def optimizer(model):
 groups=[{'params':list(model.head.parameters()),'lr':CFG['head_lr'],'weight_decay':CFG['head_weight_decay']}]
 if model.arm!='frozen':groups.append({'params':list(model.adapter.parameters()),'lr':CFG['adapter_lr'],'weight_decay':CFG['adapter_weight_decay']})
 return torch.optim.AdamW(groups)
def schedule(opt,ep):
 for g,n in zip(opt.param_groups,('head','adapter')):g['lr']=CFG[n+'_min_lr']+.5*(CFG[n+'_lr']-CFG[n+'_min_lr'])*(1+math.cos(math.pi*ep/12))
def memory():
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;mps=torch.mps.driver_allocated_memory();return {'rss_peak_bytes':rss,'mps_driver_bytes':mps,'conservative_sum_bytes':rss+mps}
def checkpoint_state(model):return {n:v.detach().cpu().clone() for n,v in model.state_dict().items() if not n.startswith('backbone.')}

def benchmark(use_checkpoint=False):
 check();torch.set_num_threads(2);d=inputs('original');tr=np.flatnonzero(d['fid']!=0);sc=StandardScaler().fit(d['B'][tr]);ym=d['y'][tr].mean();ys=d['y'][tr].std();tokens=e.get_cache(d);m=backbone.model().to('mps');frozenhash=h.state_hash(m.state_dict());records={};inithead=None;initadapter=None
 # Only train images; initialfunction and the trainedinputgraph are tested separately.
 for arm in ARMS:
  model=Model(m,sc,CFG['seed'],arm,use_checkpoint).to('mps');ih=h.state_hash(model.head.state_dict());ia=h.state_hash(model.adapter.state_dict());assert sum(p.numel() for p in model.adapter.parameters())==6144
  if inithead is None:inithead=ih;initadapter=ia
  assert ih==inithead and ia==initadapter
  with torch.no_grad():
   xx=rawimage(d,int(tr[0]));full=model.raw_features(xx);np.testing.assert_allclose(full.cpu().numpy(),d['B'][tr[0]:tr[0]+1],atol=1e-5,rtol=0);rawzero=float(abs(full.cpu().numpy()-d['B'][tr[0]:tr[0]+1]).max());fast=model.features(raw=xx if arm=='early' else None,B=tensor(d['B'][tr[0]:tr[0]+1]),prefix=tensor(tokens[tr[0]:tr[0]+1]) if arm=='late' else None);np.testing.assert_allclose(fast.cpu().numpy(),full.cpu().numpy(),atol=1e-5,rtol=0);ref=model.head((tensor(d['B'][tr[0]:tr[0]+1])-model.mean)/model.scale);p0=model.head((full-model.mean)/model.scale);zeroerr=float(abs(p0-ref).max());assert zeroerr<1e-5;frozenfeature=full.detach().cpu().numpy().copy()
  del xx,full,fast,ref,p0
  # Real effective16 updates, sameorders and per-image dropoutseedstream, 1warm+2measured.
  opt=optimizer(model);orders=np.random.default_rng(CFG['seed']).permuted(np.tile(np.arange(len(tr)),(12,1)),axis=1);times=[];forwardtimes=[];gradrecords=[];graph=None;maxmem=memory()
  for step in range(3):
   group=tr[orders[0,step*16:(step+1)*16]];model.training_mode();opt.zero_grad(set_to_none=True);torch.mps.synchronize();start=time.perf_counter()
   for off,i in enumerate(group):
    torch.manual_seed(seed_for(0,0,step*16+off));p=batch(model,d,tokens,int(i));loss=F.smooth_l1_loss(p,tensor([(d['y'][i]-ym)/ys]),beta=1.);assert torch.isfinite(loss);(loss/16).backward()
   grads={n:float(p.grad.norm()) if p.grad is not None else None for n,p in model.adapter.named_parameters()};gradrecords.append(grads)
   torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.)
   if arm!='frozen':torch.nn.utils.clip_grad_norm_(model.adapter.parameters(),1.)
   opt.step();torch.mps.synchronize();elapsed=time.perf_counter()-start;times.append(elapsed);mem=memory();maxmem={k:max(maxmem[k],mem[k]) for k in mem};print(json.dumps({'benchmark_arm':arm,'effective16_step':step,'seconds':elapsed,'memory':mem,'adapter_gradients':grads}),flush=True)
   if mem['conservative_sum_bytes']>8e9:raise RuntimeError('RESOURCE_MEMORY_8GB')
  if arm!='frozen':assert gradrecords[0]['U.weight']>0 and gradrecords[0]['V.weight']==0 and gradrecords[1]['U.weight']>0 and gradrecords[1]['V.weight']>0
  if arm=='early':
   model.eval();model.zero_grad(set_to_none=True);xx=rawimage(d,int(tr[0]));f=model.raw_features(xx,trace=True);loss=F.smooth_l1_loss(model.head((f-model.mean)/model.scale),tensor([(d['y'][tr[0]]-ym)/ys]));loss.backward();gg=[float(z.grad.norm()) for z in model.traces];assert len(gg)==13 and all(v>0 and np.isfinite(v) for v in gg);graph={'embedding_adapter_output_and12block_output_gradient_norms':gg,'all12blocks_carry_input_gradient':True,'backbone_parameter_grads_absent':all(p.grad is None for p in m.parameters())};assert graph['backbone_parameter_grads_absent'];model.traces=[];del xx,f,loss
  model.eval()
  with torch.no_grad():
   xx=rawimage(d,int(tr[0]));after=model.raw_features(xx).cpu().numpy();deltas=[float(np.linalg.norm(after[:,j*2304:(j+1)*2304]-frozenfeature[:,j*2304:(j+1)*2304])) for j in range(3)]
   if arm=='early':assert all(v>0 for v in deltas)
   if arm=='late':assert deltas[0]==deltas[1]==0 and deltas[2]>0;fast=model.features(B=tensor(d['B'][tr[0]:tr[0]+1]),prefix=tensor(tokens[tr[0]:tr[0]+1]));np.testing.assert_allclose(fast.cpu().numpy(),after,atol=1e-5,rtol=0)
   for i in tr[:6]:
    torch.mps.synchronize();start=time.perf_counter();batch(model,d,tokens,int(i));torch.mps.synchronize();forwardtimes.append(time.perf_counter()-start)
  rec={'arm':arm,'checkpointing':use_checkpoint,'adapter_parameters':6144,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'initial_head_sha256':ih,'initial_adapter_sha256':ia,'zero_init_descriptor_max_error':rawzero,'zero_init_standardized_prediction_error':zeroerr,'effective16_step_seconds':times,'median_measured_effective16_seconds':float(np.median(times[1:])),'max_measured_effective16_seconds':max(times[1:]),'prediction_image_seconds_median':float(np.median(forwardtimes[1:])),'adapter_gradients_first3updates':gradrecords,'block6_9_12_descriptor_delta_L2_after3updates':deltas,'early_graph':graph,'memory':maxmem};records[arm]=rec;fv.write(_repo_path(OUT, f'benchmark-{arm}')+Path('') if False else _repo_path(OUT, f'benchmark-{arm}.json'),rec);assert h.state_hash(m.state_dict())==frozenhash and all(p.grad is None for p in m.parameters());del model,opt,p,loss;gc.collect();torch.mps.empty_cache()
 # 240effectiveupdates/arm/fold; endpoint400prediction/arm/fold. No validationinsideepochs.
 point=sum(q['median_measured_effective16_seconds']*240+q['prediction_image_seconds_median']*400 for q in records.values())*5
 conservative=sum(q['max_measured_effective16_seconds']*240+q['prediction_image_seconds_median']*400 for q in records.values())*5*1.25
 peak=max(q['memory']['conservative_sum_bytes'] for q in records.values());r={'arms':records,'estimated_all3arms_5fold_seconds':point,'resource_gate_estimate_seconds_with25pct_margin':conservative,'projected_hours_with_margin':conservative/3600,'working_memory_conservative_bytes':peak,'resource_pass':conservative<=10800 and peak<=8e9,'all_frozen_backbone_weights_unchanged':h.state_hash(m.state_dict())==frozenhash,'original_training_started':False,'benchmark_train_rows_only':True,'train_ids_used':[d['ids'][i] for i in tr[orders[0,:48]]],'time_estimate_scope':'All3arms,5fold,12epochs,240effectiveupdates/arm/fold; plus400endpointpredictions. Actualmayvary,25%marginnotguarantee. Earlygradientchecks andstartupvalidation excluded fromfulltraining estimate.','memory_policy':'RSSpeak + MPSdriver is conservative and may double-count unifiedmemory; not independent allocations.','checkpointing':use_checkpoint};fv.write(_repo_path(OUT, 'benchmark.json'),r);check();print(json.dumps({'BENCHMARK_COMPLETE':True,'resource_pass':r['resource_pass'],'point_seconds':point,'seconds_with_margin':conservative,'working_memory_bytes':peak}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','benchmark']);ap.add_argument('--checkpoint',action='store_true');a=ap.parse_args();prepare() if a.mode=='prepare' else benchmark(a.checkpoint)
