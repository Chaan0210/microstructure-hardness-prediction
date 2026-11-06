"""Only after112passes: locked alpha fusion, full500 attention fit, test1000 inference."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
import sys,json,csv,time,math,argparse,datetime,hashlib
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import nested_fusion_112 as e
import candidate_pipeline as cp
import dinob_062 as backbone
import numpy as np,torch
from PIL import Image
from torch.nn import functional as F
OUT=_repo_path(ROOT, 'artifacts/final-model-113');CSV=_repo_path(ROOT, 'artifacts/submission-candidate-012.csv')

def prepare():
 e.check();d=e.read(e.OUT/'decision.json');assert d['eligible_for_locked_full500_candidate'];assert all(e.read(e.OUT/f'validation-{s}.json')['passed'] for s in e.SPLITS);assert e.read(e.OUT/'validation-replay.json')['passed'];OUT.mkdir(exist_ok=False)
 with np.load(_repo_io(e.OUT/'original/predictions.npz'),allow_pickle=False) as z:m=e.meta_fit(z['outerP'][:,[0,2]],z['y']);assert len(z['y'])==400
 assert 0<m['alpha']<=1
 p={'candidate':'012','locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'alpha':m['alpha'],'alpha_fit_scope':'Originaldevelopment400OOF011/attention ONLY,after112procedurelocked. Itsfitobjectiveisnotreported asvalidation performance. No holdout target or testinput accessed tochoosealpha.','formula':'candidate011+alpha*(full500_attention-candidate011)','config':e.pa.CFG,'full500_seed':e.pa.CFG['seed'],'selection_bias':d['limits'],'development_results':d['results'],'sources':{n:e.sha(_repo_path(ROOT, n)) for n in ['candidate012_pipeline.py','nested_fusion_112.py','patch_attention_092.py','dinob_062.py']},'evidence_sha256':{str(p.relative_to(ROOT)):e.sha(p) for p in [e.OUT/'decision.json',e.OUT/'original/predictions.npz',_repo_path(ROOT, 'reports/nested-fusion-112-summary.json')]},'protected011':e.read(e.OUT/'plan.json')['protected011'],'full500_rule':'Retrainoneunchanged20epochattentionhead,base seed20270322,all500trainingrows;normalizers/targetstatsfit500. Frozenofficialbackboneunchanged. Retainfull500011as-is. Testonlyafterheadandalpha locked.','upload':False,'cost':0};e.write(_repo_path(OUT, 'decision.json'),p);e.emit({'candidate012_locked':True,'alpha':m['alpha']})
def check():
 e.check();p=e.read(_repo_path(OUT, 'decision.json'))
 for n,h in {**p['sources'],**p['evidence_sha256'],**p['protected011']}.items():assert e.sha(_repo_path(ROOT, n))==h,n
 return p

def extract(split):
 check();train=split=='train'
 if not train:assert (_repo_path(OUT, 'fit.json')).exists()
 ids,y=cp.read_csv(_repo_path(ROOT, 'data', 'train.csv' if train else 'sample_submission.csv'),'TRAIN' if train else 'TEST',500 if train else 1000,train)
 base=e.read(_repo_path(ROOT, 'artifacts/final-model-065/report.json' if train else 'artifacts/submission-candidate-008.manifest.json'));cache=_repo_path(ROOT, base['train_B_cache' if train else 'test_B_cache']);assert e.sha(cache)==base['train_B_sha256' if train else 'test_B_sha256']
 with np.load(_repo_io(cache),allow_pickle=False) as z:assert z['ids'].tolist()==ids;reference=z['X'][:,-2304:].copy()
 hashes=base['training_image_sha256' if train else 'test_image_sha256'];path=_repo_path(OUT, f'{split}_tokens.npy');assert not path.exists();dest=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(ids),1296,768));mom=[];model=backbone.model().to('mps');state=e.old.e.h.state_hash(model.state_dict());start=time.perf_counter();maxerr=0.
 for i,ident in enumerate(ids):
  file=_repo_path(ROOT, 'data', split, f'{ident}.png');assert e.sha(file)==hashes[ident]
  with Image.open(_repo_io(file)) as im:
   assert im.mode=='L' and im.size==(256,256);a=np.asarray(im.resize((504,504),Image.Resampling.BICUBIC),dtype=np.float32)/255
  inp=np.ascontiguousarray((np.repeat(a[None],3,axis=0)-backbone.ds.base.MEAN)/backbone.ds.base.STD)
  with torch.no_grad():
   patch,cls=model.get_intermediate_layers(torch.from_numpy(inp[None]).to('mps'),n=[11],return_class_token=True,norm=True)[0];desc=torch.cat((cls,patch.mean(1),patch.std(1,correction=0)),1).cpu().numpy()[0];arr=patch.cpu().numpy()[0]
  err=float(abs(desc-reference[i]).max());maxerr=max(maxerr,err);assert err<2e-4;dest[i]=arr
  if train:
   b=arr.astype(float);mom.append(np.stack([b.mean(0),np.square(b).mean(0)]))
  if (i+1)%100==0:dest.flush();del dest;dest=np.load(_repo_io(path),mmap_mode='r+');e.emit({'extracted':split,'rows':i+1,'seconds':time.perf_counter()-start})
 dest.flush();del dest;assert e.old.e.h.state_hash(model.state_dict())==state;del model;torch.mps.empty_cache()
 if train:
  with (_repo_path(OUT, 'train_moments.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),moments=np.stack(mom))
 r={'split':split,'ids':ids,'token_sha256':e.sha(path),'image_sha256':hashes,'B_descriptor_replay_max_error':maxerr,'backbone_unchanged':True,'seconds':time.perf_counter()-start,'no_holdout_subgroup_score':True};e.write(_repo_path(OUT, f'{split}_extraction.json'),r);e.emit({'extraction_complete':split,'seconds':r['seconds']})

def training_data():
 check();ids,y=cp.read_csv(_repo_path(ROOT, 'data/train.csv'),'TRAIN',500,True);r=e.read(_repo_path(OUT, 'train_extraction.json'));assert r['ids']==ids and e.sha(_repo_path(OUT, 'train_tokens.npy'))==r['token_sha256']
 with np.load(_repo_path(OUT, 'train_moments.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;mom=z['moments'].copy()
 return ids,y,np.load(_repo_path(OUT, 'train_tokens.npy'),mmap_mode='r'),mom

def train(ids,y,tokens,mom,path):
 path.mkdir(exist_ok=False);seed=e.pa.CFG['seed'];tr=np.arange(500);mean,scale,var=e.pa.normalize(mom,tr);ym=float(y.mean());ys=float(y.std());m=e.pa.Pool(mean,scale,seed,'attention').to('mps');opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.1);orders=np.random.default_rng(seed).permuted(np.tile(tr,(20,1)),axis=1);start=time.perf_counter();curve=[]
 for ep,order in enumerate(orders):
  e.pa.schedule(opt,ep);m.train();losses=[]
  for first in range(0,500,16):
   group=order[first:first+16];opt.zero_grad(set_to_none=True)
   for off in range(0,len(group),2):
    ii=group[off:off+2];torch.manual_seed(seed+10000*(ep+1)+first+off);loss=F.smooth_l1_loss(m(e.pa.tensor(tokens,ii)),torch.tensor((y[ii]-ym)/ys,dtype=torch.float32,device='mps'),beta=1.);assert torch.isfinite(loss);(loss*len(ii)/len(group)).backward();losses.append(float(loss.detach()))
   gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(gn);opt.step()
  curve.append({'epoch':ep+1,'train_online_loss':float(np.mean(losses)),'lr':opt.param_groups[0]['lr']})
 state={n:v.detach().cpu() for n,v in m.state_dict().items()};torch.save({'state':state,'seed':seed,'config':e.pa.CFG,'mean':torch.tensor(mean),'scale':torch.tensor(scale),'var':torch.tensor(var),'target_mean':ym,'target_std':ys,'train_ids':ids},_repo_io(path/'model.pt'));r={'model_sha256':e.sha(path/'model.pt'),'state_sha256':e.old.e.h.state_hash(state),'curve':curve,'seconds':time.perf_counter()-start,'order_sha256':hashlib.sha256(orders.tobytes()).hexdigest()};e.write(path/'report.json',r);del m,opt;torch.mps.empty_cache();return r

def fit():
 check();ids,y,tokens,mom=training_data();r=train(ids,y,tokens,mom,_repo_path(OUT, 'head'));e.write(_repo_path(OUT, 'fit.json'),{'head_report_sha256':e.sha(_repo_path(OUT, 'head/report.json')),'model_sha256':r['model_sha256'],'train_ids':ids,'n_train':500,'train_csv_sha256':e.sha(_repo_path(ROOT, 'data/train.csv')),'seconds':r['seconds'],'holdout_subgroup_scored':False});e.emit({'full500_fit_complete':True,'seconds':r['seconds']})

def loadmodel(device='mps'):
 a=torch.load(_repo_path(OUT, 'head/model.pt'),map_location='cpu',weights_only=True);assert e.sha(_repo_path(OUT, 'head/model.pt'))==e.read(_repo_path(OUT, 'fit.json'))['model_sha256'];m=e.pa.Pool(a['mean'].numpy(),a['scale'].numpy(),a['seed'],'attention');m.load_state_dict(a['state']);return m.to(device).eval(),a

def outputs():
 p=check();ids,_=cp.read_csv(_repo_path(ROOT, 'data/sample_submission.csv'),'TEST',1000,False);r=e.read(_repo_path(OUT, 'test_extraction.json'));assert r['ids']==ids and e.sha(_repo_path(OUT, 'test_tokens.npy'))==r['token_sha256'];tokens=np.load(_repo_path(OUT, 'test_tokens.npy'),mmap_mode='r');model,a=loadmodel();raw=e.pa.predict(model,tokens,np.arange(1000))[0]*a['target_std']+a['target_mean']
 with (_repo_path(ROOT, 'artifacts/submission-candidate-011.csv')).open(newline='') as f:rows=list(csv.DictReader(f))
 assert [v['ID'] for v in rows]==ids;old=np.array([float(v['hardness']) for v in rows]);pred=old+p['alpha']*(raw-old);assert pred.shape==(1000,) and np.isfinite(pred).all() and len(set(ids))==1000;del model;torch.mps.empty_cache();return ids,old,raw,pred

def predict():
 ids,old,raw,pred=outputs()
 with CSV.open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,pred))
 with CSV.with_suffix('.components.csv').open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','incumbent011','attention','candidate012']);w.writerows(zip(ids,old,raw,pred))
 with (_repo_path(OUT, 'predictions.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),incumbent011=old,attention=raw,primary=pred)
 e.write(CSV.with_suffix('.manifest.json'),{'candidate':'012','alpha':check()['alpha'],'rows':1000,'columns':['ID','hardness'],'submission_sha256':e.sha(CSV),'components_sha256':e.sha(CSV.with_suffix('.components.csv')),'predictions_sha256':e.sha(_repo_path(OUT, 'predictions.npz')),'model_sha256':e.sha(_repo_path(OUT, 'head/model.pt')),'sample_order_exact':True,'finite':True,'duplicates':0,'test_fit_or_selection':False,'holdout_scored':False,'uploaded':False,'cost':0});e.emit({'CSV_ready':str(CSV),'sha256':e.sha(CSV)})

def verify():
 p=check();ids,y,tokens,mom=training_data();r=e.read(_repo_path(OUT, 'head/report.json'));a=torch.load(_repo_path(OUT, 'head/model.pt'),map_location='cpu',weights_only=True);mean,scale,var=e.pa.normalize(mom,np.arange(500))
 for key,value in [('mean',mean),('scale',scale),('var',var)]:np.testing.assert_array_equal(a[key].numpy(),value)
 assert a['train_ids']==ids and a['target_mean']==y.mean() and a['target_std']==y.std();rr=train(ids,y,tokens,mom,_repo_path(OUT, 'validation-replay'));assert rr['state_sha256']==r['state_sha256'] and rr['order_sha256']==r['order_sha256'];del tokens,mom
 ids,old,raw,pred=outputs()
 with CSV.open(newline='') as f:reader=csv.DictReader(f);assert reader.fieldnames==['ID','hardness'];rows=list(reader)
 assert [v['ID'] for v in rows]==ids;np.testing.assert_array_equal(pred,[float(v['hardness']) for v in rows])
 with (_repo_path(OUT, 'predictions.npz')).open('rb') as f:
  with np.load(_repo_io(f),allow_pickle=False) as z:assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['attention'],raw);np.testing.assert_array_equal(z['primary'],pred)
 with CSV.with_suffix('.reproduced.csv').open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,pred))
 assert e.sha(CSV.with_suffix('.reproduced.csv'))==e.sha(CSV);m,a=loadmodel('cpu');tokens=np.load(_repo_path(OUT, 'test_tokens.npy'),mmap_mode='r');cp=[]
 with torch.no_grad():
  for j in range(0,1000,2):cp.extend(m(torch.tensor(np.array(tokens[j:j+2],copy=True))).numpy().astype(float)*a['target_std']+a['target_mean'])
 err=float(abs(np.array(cp)-raw).max());assert err<.003
 for split in ['train','test']:
  for ident,h in e.read(_repo_path(OUT, f'{split}_extraction.json'))['image_sha256'].items():assert e.sha(_repo_path(ROOT, 'data', split, f'{ident}.png'))==h
 check();v={'valid':True,'candidate':'012','rows':1000,'finite':True,'ID_order_exact':True,'duplicates':0,'MPS_reload_exact':True,'CPU_MPS_max_error':err,'full500_replay_state_exact':True,'CSV_byte_exact':True,'CSV_SHA256':e.sha(CSV),'alpha':p['alpha'],'all1500_image_hashes_verified':True,'test_only_inference':True,'holdout_scored':False,'uploaded':False,'cost':0,'source_sha256':e.sha(Path(__file__))};e.write(_repo_path(ROOT, 'reports/candidate012-verification.json'),v);e.emit(v)
if __name__=='__main__':
 torch.set_num_threads(2);ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','train_tokens','fit','test_tokens','predict','verify']);a=ap.parse_args();{'prepare':prepare,'train_tokens':lambda:extract('train'),'fit':fit,'test_tokens':lambda:extract('test'),'predict':predict,'verify':verify}[a.mode]()
