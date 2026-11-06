"""Single fixed weak last-block adaptation; development400 only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time,resource,copy,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
sys.path.insert(0,str(ROOT))
import numpy as np,torch,joblib
import dino_spatial_039 as ds
import frozen_validation as fv
from dino_regression_037 import FrozenRegressor
OUT=_repo_path(ROOT, 'artifacts/dino-lastblock-043');BATCH=4;EPOCHS=8;LR=1e-6;ANCHOR=.1

def load():
 ids,_=ds.base.scope()
 with np.load(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;y=z['y'].copy();groups=z['groups'].copy()
 with np.load(_repo_path(ROOT, 'artifacts/dino-spatial-039/development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;X=z['full504'].astype(np.float64)
 return ids,y,groups,X

def rows(split,ids):
 p=_repo_path(ROOT, 'artifacts/dino-spatial-039/regression', split, 'development_oof.csv')
 v=json.loads((_repo_path(ROOT, 'reports/dino-039-validation.json')).read_text());assert fv.sha(p)==v['results'][split]['oof_sha256']
 with p.open() as f:a=list(csv.DictReader(f))
 assert [r['ID'] for r in a]==ids
 return np.array([int(r['fold']) for r in a]),{n:np.array([float(r[n]) for r in a]) for n in ('hardness','reference003','full504','full504_blend')}

def prefix(m,x):
 assert not m.training and not any(b.training for b in m.blocks[:-1])
 with torch.no_grad():
  z=m.prepare_tokens_with_masks(x)
  for b in m.blocks[:-1]:z=b(z)
 return z.detach()

class Tail(torch.nn.Module):
 def __init__(self,m,r):
  super().__init__();self.block=copy.deepcopy(m.blocks[-1]);self.norm=copy.deepcopy(m.norm);self.requires_grad_(False);self.block.requires_grad_(True)
  for n,v in {'mean':r.scaler.mean_,'scale':r.scaler.scale_,'coef':r.regressor.coef_,'intercept':r.regressor.intercept_}.items():self.register_buffer(n,torch.tensor(v,dtype=torch.float32))
  self.eval() #eval disables stochastic depth/dropout but preserves gradients
 def forward(self,z):
  z=self.norm(self.block(z));p=z[:,1:];f=torch.cat((z[:,0],p.mean(1),p.std(1,correction=0)),1)
  return ((f-self.mean)/self.scale)@self.coef+self.intercept

def step(tail,opt,z,y,anchor):
 assert not any(m.training for m in tail.modules());opt.zero_grad(set_to_none=True);p=tail(z);loss=(p-y).square().mean()+ANCHOR*(p-anchor).square().mean();assert torch.isfinite(loss)
 loss.backward();norm=torch.nn.utils.clip_grad_norm_(tail.block.parameters(),1.);assert torch.isfinite(norm);opt.step();return float(loss.detach()),float(norm)

def predict(tail,cache,indices):
 out=[]
 with torch.no_grad():
  for j in range(0,len(indices),BATCH):out.append(tail(torch.from_numpy(np.array(cache[indices[j:j+BATCH]],copy=True)).to('mps')).cpu().numpy())
 return np.concatenate(out)

def bench():
 OUT.mkdir(exist_ok=False);torch.set_num_threads(2);ids,y,g,X=load();fid,ref=rows('original',ids);tr=np.flatnonzero(fid!=0);sel=tr[:BATCH]
 r=joblib.load(_repo_path(ROOT, 'artifacts/dino-spatial-039/regression/original/fold-0/full504.joblib'))['model'];np.testing.assert_allclose(r.scaler.mean_,X[tr].mean(0),atol=1e-10,rtol=0)
 m=ds.base.model().to('mps');assert len(m.blocks)==12 and not m.chunked_blocks
 x=torch.from_numpy(np.concatenate([ds.inputs(ids[j],'full504') for j in sel])).to('mps');z=prefix(m,x);tail=Tail(m,r).to('mps');warm=tail(z).detach().cpu().numpy()*r.ystd+r.ymean;np.testing.assert_allclose(warm,r.predict(X[sel]),atol=2e-3,rtol=0)
 frozen={n:v.detach().cpu().clone() for n,v in tail.state_dict().items() if not n.startswith('block.')};original={n:v.detach().cpu().clone() for n,v in tail.block.state_dict().items()};opt=torch.optim.AdamW(tail.block.parameters(),lr=LR,weight_decay=0.)
 yy=torch.tensor((y[sel]-r.ymean)/r.ystd,dtype=torch.float32,device='mps');anchor=torch.tensor((r.predict(X[sel])-r.ymean)/r.ystd,dtype=torch.float32,device='mps');times=[]
 for j in range(6):
  torch.mps.synchronize();t=time.perf_counter();loss,norm=step(tail,opt,z,yy,anchor);torch.mps.synchronize()
  if j:times.append(time.perf_counter()-t)
 assert all(torch.equal(v,tail.state_dict()[n].cpu()) for n,v in frozen.items());assert any(not torch.equal(v,tail.block.state_dict()[n].cpu()) for n,v in original.items());assert all(p.grad is None for p in m.parameters())
 report={'batch':BATCH,'resolution':504,'median_train_step_seconds':float(np.median(times)),'estimated_original5fold_train_seconds':float(np.median(times)*80*EPOCHS*5),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory(),'trainable_parameters':sum(p.numel() for p in tail.parameters() if p.requires_grad),'warm_start_max_prediction_error':float(abs(warm-r.predict(X[sel])).max()),'frozen_tensors_unchanged':True,'last_block_updated':True,'all_modules_eval':True,'training_ids':[ids[j] for j in sel],'loss_last':loss,'gradient_norm_last':norm}
 fv.write(_repo_path(OUT, 'benchmark.json'),report);print(json.dumps(report),flush=True)

def prepare():
 b=json.loads((_repo_path(OUT, 'benchmark.json')).read_text());assert b['mps_driver_bytes']<4_000_000_000 and b['peak_rss_bytes']<4_000_000_000 and b['estimated_original5fold_train_seconds']<1800
 protected={str(p.relative_to(ROOT)):fv.sha(p) for p in [*ROOT.glob('artifacts/submission-candidate-005*'),_repo_path(ROOT, 'artifacts/submission-candidate-003.csv'),_repo_path(ROOT, 'artifacts/dino-spatial-039/candidate-design.json'),*(_repo_path(ROOT, 'artifacts/final-model-041')).glob('*')] if p.is_file()}
 plan={'hypothesis':'Frozen natural-image patch representation may not align with synthetic hardness. Weak last-block adaptation changes representation using only outer-train labels while holding original Ridge head fixed, isolating representation adaptation.','resolution':504,'backbone':'Pinned original DINOv2 ViT-S/14; blocks0..10, patch/pos embeddings, final norm and regression head frozen; only block11 parameters trained. All modules stay eval, gradients enabled only block11.','pooling':'Unchanged normalized CLS+patch mean+population std1152','preprocessing':'Same full grayscale RGB bicubic504 and official ImageNet normalization. No augmentation. Fold-only fixed StandardScaler and targetmean/std from matched039 Ridge.','initial_head':'Samefold039 selected Ridge,lambda(.1,1,10) inside inner3fold; frozen throughout adaptation. Initial predictions must reproduce005 DINO component.','optimizer':{'name':'AdamW','learning_rate':LR,'weight_decay':0,'batch_size':BATCH,'epochs':EPOCHS,'gradient_clip_norm':1,'anchor_prediction_penalty':ANCHOR,'seed':'20261127+outerfold','schedule':'constant;fixed8epochs,no early stopping'},'loss':'MSE on fold-standardized training target plus0.1 MSE to initial frozenRidge training prediction. No outer labels or validation curves used for schedule.','prefix_cache':'Deterministic frozen eval blocks0..10; development400 memmapfloat32 tokens400x1297x384 ~797MB; no fit/normalization across rows.','ensemble':'Fixed .75samefold003+.25adaptedDINO, compare samefold005 and unchanged frozenDINO.','screen':'Original5fold only first. Advance same single setting to both existing repartitions if pooled blendRMSE improves005,atleast3/5folds improve,and gain exceeds paired-bootstrap standard deviation. This is a compute gate,not statistical significance.','limits':'Historical development400 and reused folds create repeated-selection optimism. No holdout/test/public tuning.','sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ('dino_lastblock_043.py','dino_spatial_039.py','dino_frozen_037.py','dino_regression_037.py')},'protected':protected}
 fv.write(_repo_path(OUT, 'plan.json'),plan);print(json.dumps(plan),flush=True)

def cache():
 torch.set_num_threads(2);plan=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,h in plan['sources'].items():assert fv.sha(_repo_path(ROOT, n))==h
 ids,y,g,X=load();m=ds.base.model().to('mps');path=_repo_path(OUT, 'prefix.npy');assert not path.exists();arr=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(400,1297,384));t=time.perf_counter();err=0.
 for j,i in enumerate(ids):
  x=torch.from_numpy(ds.inputs(i,'full504')).to('mps');z=prefix(m,x);arr[j]=z.cpu().numpy()[0]
  if j<2:
   with torch.no_grad():a=m.norm(m.blocks[-1](z));p=a[:,1:];f=torch.cat((a[:,0],p.mean(1),p.std(1,correction=0)),1).cpu().numpy()[0]
   np.testing.assert_allclose(f,X[j],atol=2e-3,rtol=0);err=max(err,float(abs(f-X[j]).max()))
  if (j+1)%80==0:print(json.dumps({'cache_rows':j+1,'seconds':time.perf_counter()-t}),flush=True)
 arr.flush();del arr;fv.write(_repo_path(OUT, 'prefix.json'),{'ids':ids,'shape':[400,1297,384],'seconds':time.perf_counter()-t,'sha256':fv.sha(path),'descriptor_error':err,'no_grad_and_eval':True})

def run(split):
 torch.set_num_threads(2);plan=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,h in {**plan['sources'],**plan['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==h
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 d=_repo_path(OUT, split);d.mkdir(exist_ok=True);ids,y,groups,X=load();fid,ref=rows(split,ids);np.testing.assert_array_equal(y,ref['hardness']);meta=json.loads((_repo_path(OUT, 'prefix.json')).read_text());assert meta['ids']==ids and fv.sha(_repo_path(OUT, 'prefix.npy'))==meta['sha256'];tokens=np.load(_repo_path(OUT, 'prefix.npy'),mmap_mode='r');pred=np.full(400,np.nan);logs=[];t0=time.perf_counter();m=ds.base.model().to('mps')
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(groups[tr])&set(groups[va]);fd=d/f'fold-{k}';fd.mkdir(exist_ok=True)
  if (fd/'report.json').exists():
   rec=json.loads((fd/'report.json').read_text());assert rec['validation_ids']==[ids[j] for j in va];pred[va]=rec['predictions'];logs.append(rec);continue
  torch.manual_seed(20261127+k);rng=np.random.default_rng(20261127+k);rpath=_repo_path(ROOT, 'artifacts/dino-spatial-039/regression', split, f'fold-{k}', 'full504.joblib');r=joblib.load(_repo_io(rpath))['model'];np.testing.assert_allclose(r.scaler.mean_,X[tr].mean(0),atol=1e-10,rtol=0);assert abs(r.ymean-y[tr].mean())<1e-10 and abs(r.ystd-y[tr].std())<1e-10
  tail=Tail(m,r).to('mps');initial=predict(tail,tokens,va)*r.ystd+r.ymean;np.testing.assert_allclose(initial,ref['full504'][va],atol=2e-3,rtol=0)
  frozen={n:v.detach().cpu().clone() for n,v in tail.state_dict().items() if not n.startswith('block.')};initialblock={n:v.detach().cpu().clone() for n,v in tail.block.state_dict().items()};opt=torch.optim.AdamW(tail.block.parameters(),lr=LR,weight_decay=0.);yy=(y-r.ymean)/r.ystd;anchor=(r.predict(X[tr])-r.ymean)/r.ystd;curve=[];t=time.perf_counter()
  for ep in range(EPOCHS):
   ix=rng.permutation(len(tr));losses=[]
   for j in range(0,len(tr),BATCH):
    positions=ix[j:j+BATCH];batch=tr[positions];z=torch.from_numpy(np.array(tokens[batch],copy=True)).to('mps');loss,gn=step(tail,opt,z,torch.tensor(yy[batch],dtype=torch.float32,device='mps'),torch.tensor(anchor[positions],dtype=torch.float32,device='mps'));losses.append(loss)
   rec={'epoch':ep+1,'train_objective':float(np.mean(losses)),'seconds':time.perf_counter()-t};curve.append(rec);print(json.dumps({'split':split,'fold':k,**rec}),flush=True)
  # Evaluate once at fixed final epoch; do not pick epochs using outer labels.
  p=predict(tail,tokens,va)*r.ystd+r.ymean;pred[va]=p;trainp=predict(tail,tokens,tr)*r.ystd+r.ymean
  assert all(torch.equal(v,tail.state_dict()[n].cpu()) for n,v in frozen.items());assert all(p.grad is None for p in m.parameters());assert all(not a.training for a in tail.modules())
  state={n:v.detach().cpu() for n,v in tail.state_dict().items()};checkpoint={'state':state,'ymean':r.ymean,'ystd':r.ystd,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va],'epoch':EPOCHS};torch.save(checkpoint,_repo_io(fd/'model.pt'));loaded=Tail(m,r);loaded.load_state_dict(torch.load(_repo_io(fd/'model.pt'),weights_only=True)['state']);loaded=loaded.to('mps');p2=predict(loaded,tokens,va)*r.ystd+r.ymean;np.testing.assert_allclose(p,p2,atol=1e-5,rtol=0)
  delta=sum(float((state['block.'+n]-v).square().sum()) for n,v in initialblock.items())**.5
  rec={'fold':k,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va],'predictions':p.tolist(),'curve':curve,'train_rmse':fv.rmse(y[tr],trainp),'validation_rmse':fv.rmse(y[va],p),'block_delta_l2':delta,'seconds':time.perf_counter()-t,'warm_start_error':float(abs(initial-ref['full504'][va]).max()),'reload_error':float(abs(p-p2).max()),'frozen_verified':True,'head_source_sha256':fv.sha(rpath),'checkpoint_sha256':fv.sha(fd/'model.pt')};fv.write(fd/'report.json',rec);logs.append(rec);del tail,loaded,opt;torch.mps.empty_cache()
 assert np.isfinite(pred).all();blend=.75*ref['reference003']+.25*pred;old=ref['full504_blend'];ix=np.random.default_rng(20261128).integers(0,400,(3000,400));diff=np.sqrt(np.mean((y-old)[ix]**2,axis=1))-np.sqrt(np.mean((y-blend)[ix]**2,axis=1));gain=fv.rmse(y,old)-fv.rmse(y,blend);fg=[fv.rmse(y[fid==k],old[fid==k])-fv.rmse(y[fid==k],blend[fid==k]) for k in range(5)]
 report={'split':split,'standalone_rmse':fv.rmse(y,pred),'frozen_standalone_rmse':fv.rmse(y,ref['full504']),'blend_rmse':fv.rmse(y,blend),'candidate005_rmse':fv.rmse(y,old),'gain':gain,'fold_gains':fg,'bootstrap_sd':float(diff.std()),'conditional_ci95':np.percentile(diff,[2.5,97.5]).tolist(),'folds':logs,'seconds_this_execution':time.perf_counter()-t0}
 with (d/'development_oof.csv').open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','hardness','fold','adapted','adapted_blend','candidate005','reference003']);w.writerows([i,y[j],fid[j],pred[j],blend[j],old[j],ref['reference003'][j]] for j,i in enumerate(ids))
 fv.write(d/'report.json',report)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':bool(gain>diff.std() and sum(v>0 for v in fg)>=3),'gain':gain,'bootstrap_sd':float(diff.std()),'positive_folds':sum(v>0 for v in fg),'gate':plan['screen']})
 for n,h in plan['protected'].items():assert fv.sha(_repo_path(ROOT, n))==h
 print(json.dumps({k:v for k,v in report.items() if k!='folds'}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('bench','prepare','cache','original','20261031','20261101'));a=p.parse_args();{'bench':bench,'prepare':prepare,'cache':cache}.get(a.mode,lambda:run(a.mode))()
