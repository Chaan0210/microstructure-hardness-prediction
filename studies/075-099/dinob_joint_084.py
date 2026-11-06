"""Fresh regularized head paired with frozen B or jointly trained last B block."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,copy,argparse,resource,hashlib,math
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import dinob_lastblock_083 as priorstudy
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
base=priorstudy.base;fv=base.fv;h=base.h;backbone=base.backbone
OUT=_repo_path(ROOT, 'artifacts/dinob-joint-084');CFG={'hidden':32,'dropout':.5,'epochs':12,'microbatch':2,'effective_batch':16,'head_lr':3e-4,'head_min_lr':1e-5,'head_weight_decay':.1,'block_lr':1e-6,'block_min_lr':1e-7,'block_weight_decay':0.,'clip_head':1.,'clip_block':1.,'seed':20270215,'prediction_anchor':0.,'blend_new':.1};ARMS=('frozen','joint')
inputs=priorstudy.inputs;get_cache=base.get_cache;tensor=base.tensor

def prepare():
 old=priorstudy.check();OUT.mkdir(exist_ok=False)
 p={'hypothesis':'082/083 fixed B heads already nearly interpolate trainlabels, leaving smallbackbone signal. Fresh joint head training exposes the backbone to unsatisfied regression signal fromstep1. Isolate this by comparing a freshhead on frozenB against the identicalfreshhead jointly training onlylastBblock. No predictionpreservation penalty.',
 'duplicate_audit':'063/064 newheads onfrozenB only;043S/080B/083B fine-tuned withalreadyfit frozenheads;059 two-stageSadaptedfeatures thenhead.047jointkernel is not neuralbackbone/headjointtraining. No matchedfreshhead+B lastblock experiment found.',
 'config':CFG,'architecture':'Full504 B6912 normalizedCLS/patchmean/std blocks6/9/12. Fresh Linear6912x32,GELU,Dropout.5,Linear32x1. Onefixedseed20270215+outerfold inbotharms. Modelparameterstate initialization identical. Jointarm onlyblock11 andheadtrainable;frozenarmonlyheadtrainable. Finalnorm andallprefixweightsfrozen/eval.',
 'regularization':'Both heads AdamWwd.1,dropout.5,12fixedepochs. Headlr3e-4 cosine->1e-5; jointblocklr1e-6 cosine->1e-7,wd0,originalpretrainedinitialization,11blocks frozen. Separateclip1 forhead/block prevents backbonegradient fromchangingheadclip rule. No predictionanchor, noforcedminimumoutputchange, noearlystopping/epochchoice.',
 'pairing':'Samefoldtrain-only StandardScaler onfrozenB features andtargetmean/std, heldconstantduringjointtraining. Identicalheadinit, NumPytrainingorders, effectivebatch16/micro2. Reset torchseed per epoch+microbatch identically inbotharms to pair dropoutmasks. Bothuseidentityimageaugmentation;none additional. Jointprefixsameimmutable080cache. Frozenarm usesexactmatchingcachedBdescriptors.',
 'single_blend':'Eacharmblend=.9*verifiedsamefold009+.1*itsrawtargetprediction. 009calibrator alreadyfitstrictlyinsideoutertrain; neverfit newcalibration orweights onouterOOF. Reference009 values enteronlyheldoutcomparison/linearcombination,nevertraininputs/targets.',
 'selection':'Originaljoint standaloneRMSE muststrictlybeatmatchedfrozenhead; jointblend muststrictlybeatfrozenblend,009,and010. Thenbothknownresplits unchanged. Candidateeligibleonlyif all3partitions meetthesecriteria andstrictverificationpasses. No fallbackfrozenheadcandidate orweight/epoch/lr/headseedsearch. CI/foldcountsdiagnosticonly.',
 'resources':'ExistingCPU2/MPS,noinstall/download/cost.6stepmicrobenchmarkprojectpaired5fold<1800s,RSS/MPS<4GB. Firstcompletepairedfold<360s andRSS/MPS<4GB, reusedonce.12epochs imply240optimizerupdates perarm320trainrows. Onefixedseedperarm; this is a pairedmethodcontrol,not equalcapacitytoexisting3seedBensemble.',
 'sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['dinob_joint_084.py','validate_dinob_joint_084.py','dinob_lastblock_083.py','dinob_adapter_080.py']},'inputs':{n:fv.sha(_repo_path(ROOT, n)) for n in ['reports/adapter-diagnostics-082-lastblock-083-summary.json','reports/dinob-lastblock-083-validation.json','reports/dinob-adapter-080-validation.json','artifacts/dinob-adapter-080/prefix.json']},'protected010':old['protected010'],'scope':'Development400only;holdout/testselectionforbidden. Preserve009/010; no upload.','bias':fv.CAVEAT+' Fixedshortschedule mayunderfit andone seed leaves seed uncertainty. Results test thisoneprotocol,not alljointfinetuning.'};fv.write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'config':CFG,'selection':p['selection']}),flush=True)

def check():
 priorstudy.check();p=json.loads((_repo_path(OUT, 'plan.json')).read_text());assert p['config']==CFG
 for n,s in {**p['sources'],**p['inputs']}.items():assert fv.sha(_repo_path(ROOT, n))==s,n
 return p

class Head(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(6912,CFG['hidden']),nn.GELU(),nn.Dropout(CFG['dropout']),nn.Linear(CFG['hidden'],1))
 def forward(self,x):return self.net(x).squeeze(-1)

class Model(nn.Module):
 def __init__(self,m,sc,seed,joint):
  super().__init__();self.block=copy.deepcopy(m.blocks[-1]);self.norm=copy.deepcopy(m.norm);torch.manual_seed(seed);self.head=Head();self.register_buffer('mean',torch.tensor(sc.mean_,dtype=torch.float32));self.register_buffer('scale',torch.tensor(sc.scale_,dtype=torch.float32));self.requires_grad_(False);self.head.requires_grad_(True);self.block.requires_grad_(joint);self.joint=joint;self.eval()
 def forward(self,X,z=None):
  if self.joint:
   zz=self.norm(self.block(z));p=zz[:,1:];late=torch.cat((zz[:,0],p.mean(1),p.std(1,correction=0)),1);X=torch.cat((X[:,:4608],late),1)
  return self.head((X-self.mean)/self.scale)
 def training_mode(self):self.eval();self.head.train()

def fixed_hash(model):return h.state_hash({n:v for n,v in model.state_dict().items() if not n.startswith('head.') and not (model.joint and n.startswith('block.'))})
def seed_for(k,epoch,first):return CFG['seed']+k+10000*(epoch+1)+first

def predict(model,d,tokens,ix):
 model.eval();out=[]
 with torch.no_grad():
  for first in range(0,len(ix),2):
   ii=ix[first:first+2];out.append(model(tensor(d['B'][ii]),tensor(tokens[ii]) if model.joint else None).cpu().numpy().astype(float))
 return np.concatenate(out)

def optimizer(model):
 groups=[{'params':list(model.head.parameters()),'lr':CFG['head_lr'],'weight_decay':CFG['head_weight_decay']}]
 if model.joint:groups.append({'params':list(model.block.parameters()),'lr':CFG['block_lr'],'weight_decay':CFG['block_weight_decay']})
 return torch.optim.AdamW(groups)

def schedule(opt,ep):
 for group,kind in zip(opt.param_groups,('head','block')):
  lo=CFG[kind+'_min_lr'];hi=CFG[kind+'_lr'];group['lr']=lo+.5*(hi-lo)*(1+math.cos(math.pi*ep/CFG['epochs']))

def gradnorm(params):return float(torch.nn.utils.clip_grad_norm_(params,float('inf')))

def fit_arm(d,tokens,m,k,arm,path):
 path.mkdir(exist_ok=False);start=time.perf_counter();tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);assert not set(d['g'][tr])&set(d['g'][va]);sc=StandardScaler().fit(d['B'][tr]);ym=float(d['y'][tr].mean());ys=float(d['y'][tr].std());model=Model(m,sc,CFG['seed']+k,arm=='joint').to('mps');initial=h.state_hash(model.state_dict());headinit=h.state_hash(model.head.state_dict());frozen=fixed_hash(model);initialblock={n:p.detach().cpu().clone() for n,p in model.block.state_dict().items()};warm=predict(model,d,tokens,tr[:8])*ys+ym;opt=optimizer(model);orders=np.random.default_rng(CFG['seed']+k).permuted(np.tile(np.arange(len(tr)),(CFG['epochs'],1)),axis=1);curve=[];firstgrad=None;maxhg=0.;maxbg=0.;maskseeds=[]
 for ep,order in enumerate(orders):
  schedule(opt,ep);model.training_mode();losses=[]
  for first in range(0,len(tr),CFG['effective_batch']):
   group=tr[order[first:first+CFG['effective_batch']]];opt.zero_grad(set_to_none=True)
   for offset in range(0,len(group),2):
    ii=group[offset:offset+2];seed=seed_for(k,ep,first+offset);torch.manual_seed(seed);maskseeds.append(seed);p=model(tensor(d['B'][ii]),tensor(tokens[ii]) if model.joint else None);loss=F.smooth_l1_loss(p,tensor((d['y'][ii]-ym)/ys),beta=1.);assert torch.isfinite(loss);(loss*(len(ii)/len(group))).backward();losses.append(float(loss.detach()))
   hg=gradnorm(model.head.parameters());bg=gradnorm(model.block.parameters()) if model.joint else 0.;assert np.isfinite([hg,bg]).all();maxhg=max(maxhg,hg);maxbg=max(maxbg,bg)
   if firstgrad is None:firstgrad={'head':hg,'block':bg,'train_loss_standardized':float(np.mean(losses))}
   torch.nn.utils.clip_grad_norm_(model.head.parameters(),CFG['clip_head'])
   if model.joint:torch.nn.utils.clip_grad_norm_(model.block.parameters(),CFG['clip_block'])
   opt.step()
  trainp=predict(model,d,tokens,tr)*ys+ym;rec={'epoch':ep+1,'online_train_loss':float(np.mean(losses)),'train_RMSE':fv.rmse(d['y'][tr],trainp),'head_lr':opt.param_groups[0]['lr'],'block_lr':opt.param_groups[1]['lr'] if model.joint else 0.,'seconds':time.perf_counter()-start};curve.append(rec);print(json.dumps({'arm':arm,'fold':k,**rec}),flush=True)
 p=predict(model,d,tokens,va)*ys+ym;assert fixed_hash(model)==frozen and all(p.grad is None for p in m.parameters());state={n:v.detach().cpu() for n,v in model.state_dict().items()};delta=sum(float((state['block.'+n]-v).square().sum()) for n,v in initialblock.items())**.5;assert delta>0 if model.joint else delta==0
 checkpoint={'state':state,'config':CFG,'arm':arm,'train_ids':[d['ids'][i] for i in tr],'validation_ids':[d['ids'][i] for i in va],'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'scaler_var':torch.tensor(sc.var_),'target_mean':ym,'target_std':ys,'initial_state_sha256':initial,'initial_head_sha256':headinit,'fixed_state_sha256':frozen};torch.save(checkpoint,_repo_io(path/'model.pt'))
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=tr,validation_indices=va,warm_train8=warm,train_predictions=trainp,predictions=p)
 r={'arm':arm,'fold':k,'config':CFG,'initial_state_sha256':initial,'initial_head_sha256':headinit,'fixed_state_sha256':frozen,'final_state_sha256':h.state_hash(state),'batch_order_sha256':hashlib.sha256(orders.tobytes()).hexdigest(),'dropout_seeds_sha256':hashlib.sha256(np.array(maskseeds,dtype=np.int64).tobytes()).hexdigest(),'curve':curve,'first_update_gradients':firstgrad,'max_head_grad_before_clip':maxhg,'max_block_grad_before_clip':maxbg,'lastblock_delta_L2':delta,'checkpoint_sha256':fv.sha(path/'model.pt'),'predictions_sha256':fv.sha(path/'predictions.npz'),'seconds':time.perf_counter()-start,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory()};fv.write(path/'report.json',r);return p,r

def fit_pair(d,tokens,m,k,path):
 path.mkdir(exist_ok=False);out={};records={};start=time.perf_counter()
 for arm in ARMS:out[arm],records[arm]=fit_arm(d,tokens,m,k,arm,path/arm);torch.mps.empty_cache()
 for key in ['initial_state_sha256','initial_head_sha256','batch_order_sha256','dropout_seeds_sha256']:assert records['frozen'][key]==records['joint'][key],key
 with np.load(_repo_io(path/'frozen/predictions.npz'),allow_pickle=False) as a,np.load(_repo_io(path/'joint/predictions.npz'),allow_pickle=False) as b:np.testing.assert_allclose(a['warm_train8'],b['warm_train8'],atol=1e-4,rtol=0);warmerr=float(abs(a['warm_train8']-b['warm_train8']).max())
 r={'fold':k,'arms':records,'warm_pair_max_error':warmerr,'same_initial_state_and_masks':True,'seconds':time.perf_counter()-start};fv.write(path/'report.json',r);return out,r

def microbenchmark():
 check();torch.set_num_threads(2);d=inputs('original');tokens=get_cache(d);m=backbone.model();tr=np.flatnonzero(d['fid']!=0);sc=StandardScaler().fit(d['B'][tr]);ix=tr[:2];yy=tensor((d['y'][ix]-d['y'][tr].mean())/d['y'][tr].std());recs={}
 for arm in ARMS:
  model=Model(m,sc,CFG['seed'],arm=='joint').to('mps');initial=h.state_hash(model.state_dict());opt=optimizer(model);times=[];model.training_mode()
  for step in range(6):
   torch.manual_seed(seed_for(0,0,step*2));torch.mps.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True);p=model(tensor(d['B'][ix]),tensor(tokens[ix]) if model.joint else None);loss=F.smooth_l1_loss(p,yy,beta=1.);loss.backward();torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.)
   if model.joint:torch.nn.utils.clip_grad_norm_(model.block.parameters(),1.)
   opt.step();torch.mps.synchronize()
   if step:times.append(time.perf_counter()-start)
  recs[arm]={'median_step_seconds':float(np.median(times)),'initial_state_sha256':initial,'head_parameters':sum(p.numel() for p in model.head.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory()};del model,opt;torch.mps.empty_cache()
 assert recs['frozen']['initial_state_sha256']==recs['joint']['initial_state_sha256'];projected=sum(r['median_step_seconds'] for r in recs.values())*160*CFG['epochs']*5;r={'arms':recs,'projected_paired5fold_train_seconds':projected,'resource_pass':projected<1800 and max(v['rss_peak_bytes'] for v in recs.values())<4e9 and max(v['mps_driver_bytes'] for v in recs.values())<4e9};fv.write(_repo_path(OUT, 'microbenchmark.json'),r);print(json.dumps(r),flush=True);assert r['resource_pass']

def benchmark_fold():
 check();assert json.loads((_repo_path(OUT, 'microbenchmark.json')).read_text())['resource_pass'];torch.set_num_threads(2);d=inputs('original');tokens=get_cache(d);m=backbone.model();out=_repo_path(OUT, 'original');out.mkdir(exist_ok=False);p,r=fit_pair(d,tokens,m,0,out/'fold-0');b={'first_paired_fold_seconds':r['seconds'],'projected5fold_seconds':r['seconds']*5,'rss_peak_bytes':max(a['rss_peak_bytes'] for a in r['arms'].values()),'mps_driver_bytes':max(a['mps_driver_bytes'] for a in r['arms'].values()),'first_pair_reused':True,'fold_report_sha256':fv.sha(out/'fold-0/report.json')};b['resource_pass']=b['first_paired_fold_seconds']<360 and b['rss_peak_bytes']<4e9 and b['mps_driver_bytes']<4e9;fv.write(_repo_path(OUT, 'benchmark.json'),b);print(json.dumps(b),flush=True);assert b['resource_pass']

def evaluate(d,out,preds,seconds):
 ps={'reference009':d['reference009'],'reference010':d['reference010'],**preds};ps.update({arm+'_blend':.9*d['reference009']+.1*preds[arm] for arm in ARMS});y=d['y'];fid=d['fid'];pairs={'joint_vs_frozen':(ps['frozen'],ps['joint']),'jointblend_vs_frozenblend':(ps['frozen_blend'],ps['joint_blend']),'jointblend_vs009':(ps['reference009'],ps['joint_blend']),'jointblend_vs010':(ps['reference010'],ps['joint_blend'])};comp={n:base.prior.old.comparison(y,a,b,fid) for n,(a,b) in pairs.items()};advance=all(c['gain']>0 for c in comp.values())
 with (out/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=y,fold=fid,**ps)
 r={'scores':{n:{'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]} for n,p in ps.items()},'comparisons':comp,'advance':advance,'fold_report_sha256':{f'fold-{k}/report.json':fv.sha(out/f'fold-{k}/report.json') for k in range(5)},'predictions_sha256':fv.sha(out/'development_predictions.npz'),'seconds_excluding_reused_pair':seconds,'new_candidate':False};fv.write(out/'report.json',r);return r

def run(split):
 check();b=json.loads((_repo_path(OUT, 'benchmark.json')).read_text());assert b['resource_pass'];torch.set_num_threads(2)
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 d=inputs(split);tokens=get_cache(d);m=backbone.model();out=_repo_path(OUT, split)
 if split=='original':assert out.is_dir() and not (out/'report.json').exists()
 else:out.mkdir(exist_ok=False)
 preds={a:np.full(400,np.nan) for a in ARMS};start=time.perf_counter()
 for k in range(5):
  fd=out/f'fold-{k}';va=np.flatnonzero(d['fid']==k)
  if split=='original' and k==0:
   assert fv.sha(fd/'report.json')==b['fold_report_sha256'];pair=json.loads((fd/'report.json').read_text());p={}
   for arm in ARMS:
    rr=pair['arms'][arm];assert fv.sha(fd/arm/'predictions.npz')==rr['predictions_sha256']
    with np.load(_repo_io(fd/arm/'predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(z['validation_indices'],va);p[arm]=z['predictions'].copy()
  else:p,_=fit_pair(d,tokens,m,k,fd)
  for arm in ARMS:preds[arm][va]=p[arm]
 r=evaluate(d,out,preds,time.perf_counter()-start)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':r['advance'],'comparisons':r['comparisons']})
 check();print(json.dumps(r),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','microbenchmark','benchmark-fold','original','20261031','20261101']);a=ap.parse_args();{'prepare':prepare,'microbenchmark':microbenchmark,'benchmark-fold':benchmark_fold}.get(a.mode,lambda:run(a.mode))()
