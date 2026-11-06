from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='2'
os.environ['XFORMERS_DISABLED']='1'
import sys,json,hashlib,time,resource,socket
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=_RepoPath(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(_repo_path(ROOT, 'vendor')))
def blocked(*a,**kw):raise RuntimeError('Network forbidden')
socket.create_connection=blocked;socket.socket.connect=blocked
import numpy as np,torch,joblib
from PIL import Image
from candidate_features import extract_candidate
from denoised_grain_features import extract_denoised
from feature_helpers import dino_input,spatial_descriptor
from threadpoolctl import threadpool_limits
threadpool_limits(2);torch.set_num_threads(2)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def one(i):
 p=_repo_path(PROJECT, 'data/train', f'{i}.png')
 with Image.open(_repo_io(p)) as im:
  assert im.mode=='L' and im.size==(256,256);raw=np.array(im);cnn=np.asarray(im.resize((128,128),Image.Resampling.BOX),dtype=np.float32)[None]/255.
 K,n=extract_candidate(raw);tv,tn,_=extract_denoised(raw);return K,np.r_[K,tv],cnn,n,n+tn,sha(p)
def main():
 out=_repo_path(ROOT, 'run');out.mkdir(exist_ok=False);start=time.perf_counter();source=_repo_path(PROJECT, 'artifacts/run-012/development_features.npz');split=_repo_path(PROJECT, 'artifacts/bootstrap-selector-144/original/predictions.npz')
 with np.load(_repo_io(source),allow_pickle=False) as z:ids=z['ids'].tolist();y=z['y'].copy();g=z['groups'].copy()
 with np.load(_repo_io(split),allow_pickle=False) as z:assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['y'],y);fold=z['fold'].copy()
 assert len(ids)==len(set(ids))==400 and set(fold)==set(range(5))
 schema=json.loads((_repo_path(PROJECT, 'deliverables/reproduction-018-v155/config/schema.json')).read_text());sources={str(p.relative_to(PROJECT)):sha(p) for p in [source,split,_repo_path(PROJECT, 'deliverables/reproduction-018-v155/config/schema.json')]}
 lock={'ids':ids,'y':y.tolist(),'groups':g.tolist(),'fold':fold.tolist(),'sources':sources,'metadata_only_from_018':['ids','y','fold'],'OOF_predictions_not_loaded':True,'schema':schema,'fresh_features':True,'raw_scope':'Only these400 named training images; no train.csv or holdout/test access','weights':{a:{'path':f'deliverables/reproduction-018-v155/weights/DINO-{a}.pth','sha256':sha(_repo_path(PROJECT, f'deliverables/reproduction-018-v155/weights/DINO-{a}.pth'))} for a in ['S','B']},'sources_code':{str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if '__pycache__' not in str(p)}}
 write(out/'feature-plan.json',lock);rows=[]
 for st in range(0,400,20):
  rows.extend(joblib.Parallel(n_jobs=2)(joblib.delayed(one)(i) for i in ids[st:st+20]));print(json.dumps({'stage':'fresh_morphology','rows':len(rows),'seconds':time.perf_counter()-start}),flush=True)
 assert all(r[3]==schema['compact_names'] and r[4]==schema['physics_names'] for r in rows)
 d={'ids':np.array(ids),'y':y,'g':g,'fold':fold,'K':np.stack([r[0] for r in rows]),'X57':np.stack([r[1] for r in rows]),'images':np.stack([r[2] for r in rows])};times={'morphology':time.perf_counter()-start};imagehash={i:r[-1] for i,r in zip(ids,rows)}
 from dinov2.hub.backbones import dinov2_vits14,dinov2_vitb14
 for arm,factory,width in [('S',dinov2_vits14,3456),('B',dinov2_vitb14,6912)]:
  st=time.perf_counter();m=factory(pretrained=False);m.load_state_dict(torch.load(_repo_path(PROJECT, lock['weights'][arm]['path']),map_location='cpu',weights_only=True),strict=True);m.requires_grad_(False);m.eval();m=m.to('mps');feats=[];sp=[]
  for j,i in enumerate(ids):
   with torch.inference_mode():
    xx=torch.from_numpy(dino_input(_repo_path(PROJECT, 'data/train', f'{i}.png'))[None]).to('mps');layers=m.get_intermediate_layers(xx,n=[5,8,11],return_class_token=True,norm=True);feats.append(torch.cat([torch.cat([c,p.mean(1),p.std(1,correction=0)],1) for p,c in layers],1).cpu().numpy()[0])
    if arm=='B':sp.append(spatial_descriptor(layers[-1][0].cpu().numpy()[0]))
   assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4e9 and torch.mps.driver_allocated_memory()<2e9
   if (j+1)%100==0:print(json.dumps({'stage':'fresh_DINO_'+arm,'rows':j+1}),flush=True)
  d[arm]=np.stack(feats).astype(float);assert d[arm].shape==(400,width);times[arm]=time.perf_counter()-st
  if arm=='B':d['spatial']=np.stack(sp);d['X']=np.column_stack([d['X57'],d['spatial']])
  del m;torch.mps.empty_cache();assert sha(_repo_path(PROJECT, lock['weights'][arm]['path']))==lock['weights'][arm]['sha256']
 with (out/'features.npz').open('xb') as f:np.savez_compressed(_repo_io(f),**d)
 # Comparisons only after immutable fresh feature output exists.
 references={'K':('artifacts/run-012/development_features.npz','X'),'X57':('artifacts/run-022/development_features.npz','X'),'images':('artifacts/nested-calibration-074/development_features.npz','images'),'S':('artifacts/dino-multilayer-044/development_features.npz','X'),'B':('artifacts/dinob-062/development_features.npz','X'),'spatial':('artifacts/spatial-joint-140/features.npz','adjacent')};errors={}
 for name,(file,key) in references.items():
  with np.load(_repo_path(PROJECT, file),allow_pickle=False) as z:
   assert z['ids'].tolist()==ids;old=z[key].astype(float)
   if name=='S':old=old.reshape(400,-1)
   errors[name]={'max_abs':float(abs(d[name]-old).max()),'exact':bool(np.array_equal(d[name],old)),'reference_sha256':sha(_repo_path(PROJECT, file))}
  assert errors[name]['max_abs']<(1e-7 if name in ['K','X57','spatial'] else 1e-4)
 rec={'passed':True,'times':times,'seconds':time.perf_counter()-start,'feature_sha256':sha(out/'features.npz'),'plan_sha256':sha(out/'feature-plan.json'),'comparisons':errors,'image_sha256':imagehash,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'all_features_freshly_extracted':True,'fitted_regressors_loaded':0};write(out/'feature-validation.json',rec);print(json.dumps({'FEATURES_COMPLETE':rec}),flush=True)
if __name__=='__main__':main()
