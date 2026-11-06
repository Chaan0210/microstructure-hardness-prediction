"""Single weak full-last-block B control after adapter diagnostics; fixed heads/009."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,copy,argparse,resource,hashlib
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import diagnose_adapter_082 as diagnostics
import dinob_adapter_080 as base
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
fv=base.fv;h=base.h;prior=base.prior;backbone=base.backbone;previous=base.previous;SEEDS=base.SEEDS
OUT=_repo_path(ROOT, 'artifacts/dinob-lastblock-083');CFG={'epochs':8,'microbatch':2,'effective_batch':16,'lr':1e-6,'weight_decay':0.,'anchor_weight':1.,'clip':1.,'seed':20270201}
tensor=base.tensor;predict=base.predict;head_states=base.head_states;prefix=base.prefix;get_cache=base.get_cache

def inputs(split):
 d=base.inputs(split);v=json.loads((_repo_path(ROOT, 'reports/dinob-adapter-080-validation.json')).read_text());p=base.OUT/split/'development_predictions.npz';assert fv.sha(p)==v['results'][split]['predictions_sha256']
 with np.load(_repo_io(p),allow_pickle=False) as z:assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['y'],d['y']);np.testing.assert_array_equal(z['fold'],d['fid']);d['reference010']=z['primary'].copy()
 return d

def prepare():
 old=diagnostics.check();r=json.loads((diagnostics.OUT/'report.json').read_text());assert r['passed'] and not r['bug_found'];OUT.mkdir(exist_ok=False)
 p={'hypothesis':'082 found livegradients and mean74.8% of available prediction-loss reduction, not a dead/identity adapter. Remaining limitation may involve the attention-only rank4 update subspace and fixed nearly-interpolatinghead, not simply LR. Test one native full lastblock update, allowing FFN/normalization/attention directions while retaining the same functional anchor. Do not force larger function changes.',
 'duplicate_audit':'043 fullblock was ViT-S with a Ridgehead;059 reused that S representation;080 was B attention-only rank4. No prior B full-native-lastblock control with these fixed3MLPheads.',
 'config':CFG,'parameters':'Only all parameters of B block11 trainable; blocks0..10, embeddings, finalnorm, all3MLPheads/scalers and009affinecoefficients frozen. Allmodules eval. Same504pixels,6912descriptor,3seedmean. No head continuation or alternate pooling.',
 'regularization':'Same080 SmoothL1beta1 +1.0MSE to frozenB training prediction. LR1e-6 from historical043 weakbackbone recipe,8fixedepochs,clip1. Nativepretrainedweights use weightdecay0 to avoid shrinking the pretrainedorigin; preservation is enforced functionally by the unchanged anchor. LR differs from080 because factorized adapter and nativeweight updates have different parameterization; not a pure causal rank-only experiment.',
 'primary':'Keep009weights andtrain-onlyaffinecoefficients. Replace Bmean only. Require originalpooledRMSE strictly below both009 and existing010; onlythen bothknownresplits unchanged. Eligible only if all3partitions improve both controls and strictverificationpasses. Bootstrap/foldcounts diagnostic only; no alternative LR/epoch/anchor/rank/mixture iffails.',
 'resources':'CPU2/MPS existingvenv;reuse immutable080 development400prefix.6-stepmicrobenchmark:projected5fold<1800s,RSS/MPS<4GB. Firstcompletefold<360s andRSS/MPS<4GB reused once before remainingfolds. No install/download/cost.',
 'sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['dinob_lastblock_083.py','validate_dinob_lastblock_083.py','diagnose_adapter_082.py','dinob_adapter_080.py']},'inputs':{n:fv.sha(_repo_path(ROOT, n)) for n in ['artifacts/adapter-diagnostics-082/report.json','reports/dinob-adapter-080-validation.json','artifacts/dinob-adapter-080/prefix.json','reports/dino-043-validation.json']},'protected010':old['protected010'],'scope':'Development400only. No holdout/testselection. Preserve009 andunsubmitted010. Any futurecandidate requires a newlock, neverautomaticupload.','bias':fv.CAVEAT};fv.write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'config':CFG,'primary':p['primary']}),flush=True)

def check():
 diagnostics.check();p=json.loads((_repo_path(OUT, 'plan.json')).read_text());assert p['config']==CFG
 for n,s in {**p['sources'],**p['inputs']}.items():assert fv.sha(_repo_path(ROOT, n))==s,n
 return p

class Tail(nn.Module):
 def __init__(self,m,states,seed):
  super().__init__();torch.manual_seed(seed);self.block=copy.deepcopy(m.blocks[-1]);self.norm=copy.deepcopy(m.norm);self.heads=nn.ModuleList()
  for a in states:
   hh=h.Head(6912);hh.load_state_dict(a['model']);self.heads.append(hh)
  self.requires_grad_(False);self.block.requires_grad_(True);self.register_buffer('mean',states[0]['scaler_mean'].float());self.register_buffer('scale',states[0]['scaler_scale'].float());self.eval()
 def forward(self,z,early):
  z=self.norm(self.block(z));p=z[:,1:];late=torch.cat((z[:,0],p.mean(1),p.std(1,correction=0)),1);x=(torch.cat((early,late),1)-self.mean)/self.scale;return torch.stack([a(x)[0] for a in self.heads]).mean(0)

def frozen_hash(t):return h.state_hash({n:v for n,v in t.state_dict().items() if not n.startswith('block.')})

def microbenchmark():
 check();torch.set_num_threads(2);d=inputs('original');tr=np.flatnonzero(d['fid']!=0);sel=tr[:2];states,old=head_states(d,0);m=backbone.model().to('mps');z=prefix(m,torch.tensor(np.concatenate([backbone.ds.inputs(d['ids'][i],'full504') for i in sel]),device='mps'));t=Tail(m,states,CFG['seed']).to('mps');yy=tensor((d['y'][sel]-states[0]['target_mean'])/states[0]['target_std']);anchor=tensor((old[sel]-states[0]['target_mean'])/states[0]['target_std']);early=tensor(d['B'][sel,:4608]);warm=t(z,early).detach().cpu().numpy().astype(float)*states[0]['target_std']+states[0]['target_mean'];np.testing.assert_allclose(warm,old[sel],atol=.003,rtol=0);frozen=frozen_hash(t);params=[p for p in t.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=CFG['lr'],weight_decay=CFG['weight_decay']);times=[]
 for step in range(6):
  torch.mps.synchronize();now=time.perf_counter();opt.zero_grad(set_to_none=True);p=t(z,early);loss=F.smooth_l1_loss(p,yy,beta=1.)+CFG['anchor_weight']*F.mse_loss(p,anchor);loss.backward();torch.nn.utils.clip_grad_norm_(params,CFG['clip']);opt.step();torch.mps.synchronize()
  if step:times.append(time.perf_counter()-now)
 assert frozen_hash(t)==frozen and all(p.grad is None for p in m.parameters());r={'trainable_parameters':sum(p.numel() for p in params),'median_step_seconds':float(np.median(times)),'projected_original5fold_train_seconds':float(np.median(times)*160*CFG['epochs']*5),'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory(),'warm_control_max_error':float(abs(warm-old[sel]).max()),'frozen_parameters_unchanged':True,'training_ids':[d['ids'][i] for i in sel]};r['resource_pass']=r['projected_original5fold_train_seconds']<1800 and r['rss_peak_bytes']<4e9 and r['mps_driver_bytes']<4e9;fv.write(_repo_path(OUT, 'microbenchmark.json'),r);print(json.dumps(r),flush=True);assert r['resource_pass']


def fit_fold(d,tokens,m,k,path):
 path.mkdir(exist_ok=False);start=time.perf_counter();tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);assert not set(d['g'][tr])&set(d['g'][va]);states,old=head_states(d,k);ym=states[0]['target_mean'];ys=states[0]['target_std'];t=Tail(m,states,CFG['seed']+k).to('mps');fh=frozen_hash(t);initial=h.state_hash(t.state_dict());warm=predict(t,d,tokens,va)*ys+ym;np.testing.assert_allclose(warm,old[va],atol=.003,rtol=0);params=[p for p in t.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=CFG['lr'],weight_decay=CFG['weight_decay']);orders=np.random.default_rng(CFG['seed']+k).permuted(np.tile(np.arange(len(tr)),(CFG['epochs'],1)),axis=1);curves=[]
 for epoch,order in enumerate(orders):
  vals=[]
  for startbatch in range(0,len(tr),CFG['effective_batch']):
   group=tr[order[startbatch:startbatch+CFG['effective_batch']]];opt.zero_grad(set_to_none=True)
   for first in range(0,len(group),CFG['microbatch']):
    ii=group[first:first+CFG['microbatch']];p=t(tensor(tokens[ii]),tensor(d['B'][ii,:4608]));loss=F.smooth_l1_loss(p,tensor((d['y'][ii]-ym)/ys),beta=1.)+CFG['anchor_weight']*F.mse_loss(p,tensor((old[ii]-ym)/ys));assert torch.isfinite(loss);(loss*(len(ii)/len(group))).backward();vals.append(float(loss.detach()))
   gn=torch.nn.utils.clip_grad_norm_(params,CFG['clip']);assert torch.isfinite(gn);opt.step()
  curves.append({'epoch':epoch+1,'train_objective':float(np.mean(vals)),'seconds':time.perf_counter()-start});print(json.dumps({'fold':k,**curves[-1]}),flush=True)
 assert frozen_hash(t)==fh and all(p.grad is None for p in m.parameters()) and all(not x.training for x in t.modules());p=predict(t,d,tokens,va)*ys+ym;tp=predict(t,d,tokens,tr)*ys+ym;state={n:v.detach().cpu() for n,v in t.state_dict().items()};torch.save({'state':state,'config':CFG,'train_ids':[d['ids'][i] for i in tr],'validation_ids':[d['ids'][i] for i in va],'target_mean':ym,'target_std':ys,'initial_state_sha256':initial,'frozen_state_sha256':fh},_repo_io(path/'model.pt'));delta=sum(float((state['block.'+n]-v.detach().cpu()).square().sum()) for n,v in m.blocks[-1].state_dict().items())**.5
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=tr,validation_indices=va,adapted=p,frozen=old[va],warm=warm,train_predictions=tp)
 r={'fold':k,'train_ids':[d['ids'][i] for i in tr],'validation_ids':[d['ids'][i] for i in va],'curve':curves,'config':CFG,'initial_state_sha256':initial,'frozen_state_sha256':fh,'final_state_sha256':h.state_hash(state),'batch_order_sha256':hashlib.sha256(orders.tobytes()).hexdigest(),'block_parameter_delta_L2':delta,'warm_control_max_error':float(abs(warm-old[va]).max()),'train_rmse_before':fv.rmse(d['y'][tr],old[tr]),'train_rmse_after':fv.rmse(d['y'][tr],tp),'checkpoint_sha256':fv.sha(path/'model.pt'),'predictions_sha256':fv.sha(path/'predictions.npz'),'seconds':time.perf_counter()-start,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory()};fv.write(path/'report.json',r);print(json.dumps({'completed_fold':k,'seconds':r['seconds'],'train_rmse_before':r['train_rmse_before'],'train_rmse_after':r['train_rmse_after']}),flush=True);return p,r


def benchmark_fold():
 check();torch.set_num_threads(2);d=inputs('original');tokens=get_cache(d);m=backbone.model();out=_repo_path(OUT, 'original');out.mkdir(exist_ok=False);p,r=fit_fold(d,tokens,m,0,out/'fold-0');b={'first_fold_seconds':r['seconds'],'projected_original5fold_seconds':r['seconds']*5,'rss_peak_bytes':r['rss_peak_bytes'],'mps_driver_bytes':r['mps_driver_bytes'],'first_fold_reused':True,'fold_report_sha256':fv.sha(out/'fold-0/report.json')};b['resource_pass']=r['seconds']<360 and r['rss_peak_bytes']<4e9 and r['mps_driver_bytes']<4e9;fv.write(_repo_path(OUT, 'benchmark.json'),b);print(json.dumps(b),flush=True);assert b['resource_pass']


def run(split):
 check();b=json.loads((_repo_path(OUT, 'benchmark.json')).read_text());assert b['resource_pass'];torch.set_num_threads(2)
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 d=inputs(split);tokens=get_cache(d);m=backbone.model();out=_repo_path(OUT, split)
 if split=='original':assert out.is_dir() and not (out/'report.json').exists()
 else:out.mkdir(exist_ok=False)
 adapted=np.full(400,np.nan);new=np.full(400,np.nan);slopes=[];start=time.perf_counter()
 for k in range(5):
  path=out/f'fold-{k}';va=np.flatnonzero(d['fid']==k)
  if split=='original' and k==0:
   assert fv.sha(path/'report.json')==b['fold_report_sha256'];r=json.loads((path/'report.json').read_text());assert fv.sha(path/'predictions.npz')==r['predictions_sha256']
   with np.load(_repo_io(path/'predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(z['validation_indices'],va);p=z['adapted'].copy()
  else:p,r=fit_fold(d,tokens,m,k,path)
  adapted[va]=p;fr=json.loads((previous.base.OUT/split/f'fold-{k}/report.json').read_text());s=fr['slope'];slopes.append(s);new[va]=d['reference009'][va]+s*.125*(p-d['Bmean'][va]);torch.mps.empty_cache()
 ps={'reference010':d['reference010'],'reference009':d['reference009'],'frozenB':d['Bmean'],'adaptedB':adapted,'primary':new};comp=prior.old.comparison(d['y'],d['reference009'],new,d['fid']);comp010=prior.old.comparison(d['y'],d['reference010'],new,d['fid'])
 with (out/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],**ps)
 r={'split':split,'scores':{n:{'rmse':fv.rmse(d['y'],p),'fold_rmse':[fv.rmse(d['y'][d['fid']==k],p[d['fid']==k]) for k in range(5)]} for n,p in ps.items()},'comparison':comp,'comparison_vs010':comp010,'unchanged009_slopes':slopes,'fold_report_sha256':{f'fold-{k}/report.json':fv.sha(out/f'fold-{k}/report.json') for k in range(5)},'predictions_sha256':fv.sha(out/'development_predictions.npz'),'seconds_excluding_reused_benchmark':time.perf_counter()-start,'new_candidate':False};fv.write(out/'report.json',r)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':comp['gain']>0 and comp010['gain']>0,'comparison':comp,'comparison_vs010':comp010})
 check();print(json.dumps(r),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','microbenchmark','benchmark-fold','original','20261031','20261101']);a=ap.parse_args();{'prepare':prepare,'microbenchmark':microbenchmark,'benchmark-fold':benchmark_fold}.get(a.mode,lambda:run(a.mode))()
