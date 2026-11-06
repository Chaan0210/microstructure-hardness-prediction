"""Complete teacher alignment and compare matched dense sampling policies."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import sys,json,time,gc,resource,copy
import numpy as np,torch
from torch import nn
from PIL import Image
from types import SimpleNamespace
from threadpoolctl import threadpool_limits

def save_path(path,obj):c.save(_repo_io(str(path.relative_to(c.HERE))),obj)
r=SimpleNamespace(rmse=lambda a,b:float(np.sqrt(np.mean((a-b)**2))),save_json=save_path)
ns=c.definitions(c.R4/'cnn_fitter.py',['ComplementCNN','predict','fit_one'],dict(torch=torch,nn=nn,np=np,gc=gc,time=time,resource=resource,r=r,EPOCHS=40))
Base=ns['ComplementCNN'];predict=ns['predict'];fit=ns['fit_one']
data_ns=c.definitions(c.OLD/'lowmem_cnn_data.py',['cnn_data'],dict(np=np,normalization_checks=[]));normalize=data_ns['cnn_data']
coef_ns=c.definitions(c.R4/'noise_calibration.py',['fit_weights','select'],dict(np=np,STRENGTHS=('global',100,30,10,0)));select=coef_ns['select']

class DenseFirst(Base):
 def __init__(self,kind):
  super().__init__();self.kind=kind;self.blocks[0][0].stride=(1,1)
 def forward(self,x,context,arm):
  stats=[x.mean((2,3)),(x.var((2,3),correction=0)+1e-6).sqrt()];h=x
  for j,block in enumerate(self.blocks):
   h=block(h)
   if j==0:h=nn.functional.avg_pool2d(h,4,4) if self.kind=='pool' else h[:,:,::4,::4]
   stats.extend([h.mean((2,3)),(h.var((2,3),correction=0)+1e-6).sqrt()])
  image=torch.cat(stats,1)
  return self.main(torch.cat([image,context],1)).squeeze(1),self.aux(image).squeeze(1)

def load():
 d=c.inputs();y=d.pop('y');d['images']=np.stack([np.asarray(Image.open(_repo_path(c.ROOT, f'data/train/{i}.png')),np.float32)/255 for i in d['ids']])[:,None];return d,y

def data(d,labels,k,stage,version):
 g=d['graphs'][k];tr=np.array(g['A' if stage=='selector' else 'T']);q=np.array(g['B' if stage=='selector' else 'V']);prefix='deep' if stage=='selector' else 'outer';teacher=np.full(500,np.nan);gap=teacher.copy();cov=np.zeros(500,int);sources=[]
 for n in g['nodes']:
  if not n['name'].startswith(prefix):continue
  p=c.teacher_path(k,n['name']) if version=='076' else c.R4/f"teachers/fold{k}-{n['name']}/predictions.npz";z=np.load(_repo_io(p));a,b=z['train'],z['query'];assert set(a).isdisjoint(b) and set(a)|set(b)==set(tr) and (set(a)|set(b)).isdisjoint(q);teacher[b]=z['base'];gap[b]=z['tab']-z['ridge'];cov[b]+=1
  sources.append(dict(train_ids=d['ids'][a].tolist(),validation_ids=d['ids'][b].tolist(),file=str(p.relative_to(c.ROOT)),sha256=c.sha(p)))
 assert np.all(cov[tr]==1)
 if stage=='selector':
  p=c.teacher_path(k,'outer0') if version=='076' else c.R4/f'teachers/fold{k}-outer0/predictions.npz';z=np.load(_repo_io(p));assert np.array_equal(z['query'],q);teacher[q]=z['base'];gap[q]=z['tab']-z['ridge']
 else:
  z=np.load(_repo_io(c.OLD/'baseline076.npz'));ix=np.flatnonzero(z['fold']==k);assert np.array_equal(z['dev'][ix],q);base=z['p76' if version=='076' else 'p73'][ix];teacher[q]=(base-.16*z['G'][ix])/.84;teacher[q[z['failed'][ix]]]=base[z['failed'][ix]];gap[q]=z['tab'+version][ix]-z['ridge'+version][ix]
 keys=list(d['keys']);raw=np.c_[teacher,gap,d['noise'],d['X'][:,keys.index('u_ph')],np.log1p(np.maximum(np.nan_to_num(d['X'][:,keys.index('n_obj')],nan=0),0)),d['X'][:,keys.index('coverage')]]
 z=normalize(d,tr,q,labels[tr].copy(),teacher,sources,k,stage,raw)
 if version=='076':z['meta']['training_objective']='y minus strict076 non021 branch, honest nested teachers'
 return z

def check(d,z,k,stage,version):
 # Data constructor accepts only source-indexed training labels. Poison all other labels.
 y=np.full(500,-999999.);tr=z['train'];y[tr]=z['targets']['direct'].astype(float)*z['meta']['target_std']+z['meta']['target_mean']
 # Float32 targets cannot restore exact original labels: caller compares with original labels separately.
 assert set(z['train']).isdisjoint(z['valid'])
 for x in ['images','context']:assert np.isfinite(z[x]).all()
 for x in z['targets']:assert np.isfinite(z['targets'][x]).all()

def replay(z,path,kind='base'):
 ck=torch.load(_repo_io(path/'model.pt'),map_location='cpu',weights_only=True);m=Base() if kind=='base' else DenseFirst(kind);m.load_state_dict(ck['state_dict']);p=predict(m,z['images'],z['context'],z['valid'],'aux')*z['meta']['target_std'];old=np.load(_repo_io(path/'predictions.npz'));assert np.array_equal(old['validation_index'],z['valid']);err=float(abs(p-old['validation_prediction']).max());assert err==0,err;del m;return err

def train_retarget():
 assert json.loads((c.HERE/'teachers-complete.json').read_text())['complete'];d,y=load();start=time.monotonic();records=[]
 c.save(_repo_io('cnn-retarget-code-lock.json'),dict(code=c.sha(__file__),common=c.sha(c.HERE/'common.py'),original_fitter=c.sha(c.R4/'cnn_fitter.py'),normalization=c.sha(c.OLD/'lowmem_cnn_data.py'),plan=c.sha(c.HERE/'PLAN.md'),device='cpu',threads=2))
 for k in range(5):
  for stage in ['selector','final']:
   z=data(d,y,k,stage,'076');check(d,z,k,stage,'076');altered=y.copy();altered[np.setdiff1d(np.arange(500),z['train'])]=999999.;zz=data(d,altered,k,stage,'076');assert z['meta']==zz['meta']
   for n in ['images','context']:assert np.array_equal(z[n],zz[n])
   for n in z['targets']:assert np.array_equal(z['targets'][n],zz['targets'][n])
   del zz
   for seed in [0,1]:
    old=c.OLD/f'retarget-cnn/seed{seed}/fold{k}-{stage}';path=old if (old/'training.json').exists() else c.HERE/f'cnn-retarget/seed{seed}/fold{k}-{stage}';s=2026101501+100*k+100000*seed
    if path==old:
     meta=json.loads((path/'training.json').read_text());assert meta['seed']==s and meta['meta']==z['meta'],(k,stage,'cached training contract')
    else:path.parent.mkdir(parents=True,exist_ok=True);fit(z,'aux',s,path)
    err=replay(z,path);records.append(dict(fold=k,stage=stage,seed=seed,source=str(path.relative_to(c.ROOT)),prediction_sha256=c.sha(path/'predictions.npz'),model_sha256=c.sha(path/'model.pt'),replay_maxdiff=err,poison_outside_labels_exact=True,reused=path==old));print('CNN RETARGET',k,stage,seed,round(time.monotonic()-start,1),flush=True);c.runtime()
   del z;gc.collect()
 c.save(_repo_io('cnn-retarget-complete.json'),dict(complete=True,records=records,seconds=time.monotonic()-start,new_fits=sum(not x['reused'] for x in records),reused=sum(x['reused'] for x in records)))

def probe_dense():
 d,y=load();z=data(d,y,0,'selector','073');z['valid']=z['valid'][:1];start=time.monotonic();ns['ComplementCNN']=lambda:DenseFirst('pool');fit(z,'aux',2026101501,c.HERE/'dense-resource-probe',epochs=5);elapsed=time.monotonic()-start;projection=elapsed*8*40*1.25
 c.save(_repo_io('dense-resource-probe.json'),dict(seconds=elapsed,projected_main_seconds=projection,main_fits=40,device='cpu',probe_has_no_query_labels=True,memory=c.runtime(),passes=projection<3000));print('DENSE PROJECTION',projection,flush=True)

def train_dense():
 probe=json.loads((c.HERE/'dense-resource-probe.json').read_text());assert probe['passes'],'fixed resource gate';d,y=load();start=time.monotonic();records=[]
 c.save(_repo_io('dense-code-lock.json'),dict(code=c.sha(__file__),plan=c.sha(c.HERE/'PLAN.md'),device='cpu',threads=2,arms=['pick','pool'],seeds=[2026101501,2026201501]))
 for k in range(5):
  for stage in ['selector','final']:
   z=data(d,y,k,stage,'073');check(d,z,k,stage,'073')
   for repeat in [0,1]:
    for kind in ['pick','pool']:
     assert time.monotonic()-start<3300,'dense hard55min cap';ns['ComplementCNN']=lambda:DenseFirst(kind);path=c.HERE/f'cnn-dense-{kind}/seed{repeat}/fold{k}-{stage}';path.parent.mkdir(parents=True,exist_ok=True);fit(z,'aux',2026101501+100*k+100000*repeat,path);err=replay(z,path,kind);records.append(dict(fold=k,stage=stage,seed=repeat,kind=kind,path=str(path.relative_to(c.HERE)),replay_maxdiff=err));print('DENSE',k,stage,repeat,kind,round(time.monotonic()-start,1),flush=True);c.runtime()
   del z;gc.collect()
 c.save(_repo_io('dense-complete.json'),dict(complete=True,records=records,seconds=time.monotonic()-start))

if __name__=='__main__':
 torch.set_num_threads(2);torch.set_num_interop_threads(1);sys.addaudithook(c.guard)
 with threadpool_limits(limits=2):{'retarget':train_retarget,'probe':probe_dense,'dense':train_dense}[sys.argv[1]]()
