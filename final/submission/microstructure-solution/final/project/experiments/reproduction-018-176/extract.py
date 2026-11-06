import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='2'
os.environ['XFORMERS_DISABLED']='1'
import sys,csv,json,hashlib,time,resource,socket,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vendor'))
def blocked(*a,**kw):raise RuntimeError('Network disabled')
socket.create_connection=blocked;socket.socket.connect=blocked;socket.socket.connect_ex=blocked
import numpy as np,torch,joblib
from PIL import Image
from candidate_features import extract_candidate
from denoised_grain_features import extract_denoised
from feature_helpers import dino_input,spatial_descriptor
from threadpoolctl import threadpool_limits
threadpool_limits(2);torch.set_num_threads(2)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def one(data,split,i):
 p=data/split/f'{i}.png';assert p.resolve().is_relative_to(data/split)
 with Image.open(p) as im:
  assert im.mode=='L' and im.size==(256,256);raw=np.array(im);cnn=np.asarray(im.resize((128,128),Image.Resampling.BOX),dtype=np.float32)[None]/255.
 K,n=extract_candidate(raw);tv,tn,_=extract_denoised(raw);group=min(hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for k in range(4) for a in (np.rot90(raw,k),np.fliplr(np.rot90(raw,k))))
 return K,np.r_[K,tv],cnn,n,n+tn,sha(p),group

def main():
 ap=argparse.ArgumentParser();ap.add_argument('split',choices=['train','test']);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--run',type=Path,default=ROOT/'run');args=ap.parse_args();data=args.data.resolve();out=args.run.resolve();out.mkdir(exist_ok=True);split=args.split
 if split=='test':assert json.loads((out/'training-validation.json').read_text())['passed'],'Lock/validate model before anytest access'
 start=time.perf_counter();source=data/('train.csv' if split=='train' else 'sample_submission.csv')
 expected_file=ROOT/'config'/f'original-{split}-hashes.json';expected=json.loads(expected_file.read_text()) if expected_file.exists() else None
 if expected is not None:assert sha(source)==expected['CSV_sha256']
 with source.open(newline='',encoding='utf-8-sig') as f:r=csv.DictReader(f);assert r.fieldnames==['ID','hardness'];rows=list(r)
 ids=[r['ID'] for r in rows];count=500 if split=='train' else 1000;assert len(ids)==len(set(ids))==count;assert all(i==f'{"TRAIN" if split=="train" else "TEST"}_{k+1:06d}' for k,i in enumerate(ids));y=np.array([float(r['hardness']) for r in rows]) if split=='train' else None
 if y is not None:assert np.isfinite(y).all()
 schema=json.loads((ROOT/'config/schema.json').read_text());dev=json.loads((ROOT/'config/development-ids.json').read_text());assert len(dev)==len(set(dev))==400
 plan={'split':split,'IDs':ids,'raw_CSV_sha256':sha(source),'target_read':split=='train','test_values_ignored':split=='test','frozen_config':{name:sha(ROOT/'config'/name) for name in ['schema.json','development-ids.json']},'weights':{a:sha(ROOT/'weights'/f'DINO-{a}.pth') for a in ['S','B']},'all_features_fresh':True,'extractor_sha256':sha(Path(__file__))};write(out/f'{split}-feature-plan.json',plan);records=[]
 for st in range(0,count,20):
  records.extend(joblib.Parallel(n_jobs=2)(joblib.delayed(one)(data,split,i) for i in ids[st:st+20]));print(json.dumps({'stage':split+'_morphology','rows':len(records),'seconds':time.perf_counter()-start}),flush=True)
 if expected is not None:
  assert {i:r[5] for i,r in zip(ids,records)}==expected['image_sha256']
 assert all(r[3]==schema['compact_names'] and r[4]==schema['physics_names'] for r in records)
 d={'ids':np.array(ids),'K':np.stack([r[0] for r in records]),'X57':np.stack([r[1] for r in records]),'images':np.stack([r[2] for r in records])};times={'morphology':time.perf_counter()-start};hashes={i:r[5] for i,r in zip(ids,records)}
 if split=='train':d.update(y=y,g=np.array([r[6] for r in records]),selection=np.array([ids.index(i) for i in dev]))
 from dinov2.hub.backbones import dinov2_vits14,dinov2_vitb14
 maxmps=0
 for arm,factory,width in [('S',dinov2_vits14,3456),('B',dinov2_vitb14,6912)]:
  st=time.perf_counter();m=factory(pretrained=False);m.load_state_dict(torch.load(ROOT/'weights'/f'DINO-{arm}.pth',map_location='cpu',weights_only=True),strict=True);m.requires_grad_(False);m.eval();m=m.to('mps');features=[];sp=[]
  for j,i in enumerate(ids):
   with torch.inference_mode():
    x=torch.from_numpy(dino_input(data/split/f'{i}.png')[None]).to('mps');layers=m.get_intermediate_layers(x,n=[5,8,11],return_class_token=True,norm=True);features.append(torch.cat([torch.cat([c,p.mean(1),p.std(1,correction=0)],1) for p,c in layers],1).cpu().numpy()[0])
    if arm=='B':sp.append(spatial_descriptor(layers[-1][0].cpu().numpy()[0]))
   maxmps=max(maxmps,torch.mps.driver_allocated_memory())
   if (j+1)%100==0:print(json.dumps({'stage':split+'_DINO_'+arm,'rows':j+1}),flush=True)
  d[arm]=np.stack(features).astype(float);assert d[arm].shape==(count,width);times[arm]=time.perf_counter()-st
  if arm=='B':d['spatial']=np.stack(sp);d['X']=np.column_stack([d['X57'],d['spatial']])
  del m;torch.mps.empty_cache();assert sha(ROOT/'weights'/f'DINO-{arm}.pth')==plan['weights'][arm]
 assert all(np.isfinite(v).all() for k,v in d.items() if k not in ['ids','g'])
 with (out/f'{split}-features.npz').open('xb') as f:np.savez_compressed(f,**d)
 rec={'passed':True,'split':split,'rows':count,'feature_sha256':sha(out/f'{split}-features.npz'),'plan_sha256':sha(out/f'{split}-feature-plan.json'),'image_sha256':hashes,'times':times,'seconds':time.perf_counter()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'max_observed_MPS_driver':maxmps,'fitted_regressors_loaded':0,'raw_original_rows':True};write(out/f'{split}-feature-validation.json',rec);print(json.dumps({k:v for k,v in rec.items() if k!='image_sha256'}),flush=True)
if __name__=='__main__':main()
