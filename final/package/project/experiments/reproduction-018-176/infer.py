import os,sys,json,csv,hashlib,argparse,socket,subprocess,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='2'
def forbidden(*args,**kwargs):raise RuntimeError('Fitting/backward/network forbidden during inference')
socket.create_connection=forbidden;socket.socket.connect=forbidden;socket.socket.connect_ex=forbidden
import numpy as np,torch,joblib
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from compact_regression import BalancedKernel
from cnn_model import TinyCNN
from feature_helpers import Short
import mlp_head as h,residual_models as n
from threadpoolctl import threadpool_limits
threadpool_limits(2);torch.set_num_threads(2)
def sha(p):
 hh=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):hh.update(b)
 return hh.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,r):
 with Path(p).open('x') as f:json.dump(r,f,indent=2,allow_nan=False)
def guards():
 StandardScaler.fit=forbidden;StandardScaler.partial_fit=forbidden;MinMaxScaler.fit=forbidden;MinMaxScaler.partial_fit=forbidden;BalancedKernel.fit=forbidden;n.fit=forbidden;n.r.fit_model=forbidden;h.fit=forbidden;torch.Tensor.backward=forbidden

def cnn(X,obj,device):
 norm=obj['normalization'];m=TinyCNN().to(device);m.load_state_dict(obj['state_dict']);m.eval();p=[]
 with torch.no_grad():
  for st in range(0,len(X),32):p.extend((m(torch.from_numpy((X[st:st+32]-norm['x_mean'])/max(norm['x_std'],1e-6)).to(device)).cpu().numpy()*norm['y_std']+norm['y_mean']).tolist())
 del m;return np.array(p)
def head(X,obj):
 m=h.Head(X.shape[1]);m.load_state_dict(obj['model']);return h.predict(m,(X.astype(float)-obj['scaler_mean'].numpy())/obj['scaler_scale'].numpy(),obj['target_mean'],obj['target_std'])
def short(X,obj,device):
 m=Short(obj['scaler_mean'].numpy(),obj['scaler_scale'].numpy(),obj.get('seed',20270215)).to(device);m.load_state_dict({k:obj['state'][k] for k in m.state_dict()});m.eval();p=[]
 with torch.no_grad():
  for st in range(0,len(X),2):p.extend(m(torch.tensor(X[st:st+2],dtype=torch.float32,device=device)).cpu().numpy().astype(float))
 del m;return np.array(p)*obj['target_std']+obj['target_mean']
def predict(d,models,device='mps'):
 manifest=read(models/'manifest.json');assert manifest['training_validated'] and manifest['all_fitted_models_fresh176']
 for f,hsh in manifest['files'].items():assert sha(models/f)==hsh,f
 guards();kp=joblib.load(models/'kernel.joblib')['model'].predict(d['K']);cp=np.stack([cnn(d['images'],torch.load(models/f'CNN-{s}.pt',map_location='cpu',weights_only=True),device) for s in [20261003,20261023,20261043]]);heads={}
 for arm in ['S','B']:heads[arm]=np.stack([head(d[arm],torch.load(models/f'{arm}-{s}.pt',map_location='cpu',weights_only=True)) for s in [20261128,20261205,20261219]])
 sh=short(d['B'],torch.load(models/'short.pt',map_location='cpu',weights_only=True),device);P=np.column_stack([np.zeros(len(kp)),kp,cp.mean(0),heads['S'].mean(0),heads['B'].mean(0)]);w=np.array([0.,.1875,.1875,.125,.125]);raw=P@(w/w.sum());cal=read(models/'calibrator.json');base=.9*(cal['target_center']+cal['slope']*(raw-cal['prediction_center']))+.1*sh;cor=[];full=[]
 for j in range(3):
  ww=np.array(read(models/f'weights-{j}.json')['weights']);pp=np.stack([n.predict(joblib.load(models/f'residual-{j}-{kind}.joblib')['model'],d['X']) for kind in n.KINDS]);full.append(pp);cor.append(ww@pp)
 pred=base+np.mean(cor,0);assert pred.shape==(len(d['ids']),) and np.isfinite(pred).all()
 return {'P':P,'raw':raw,'short':sh,'base':base,'corrections':np.stack(cor),'residual_candidates':np.stack(full),'primary':pred,'CNN_seeds':cp,'S_seeds':heads['S'],'B_seeds':heads['B']}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--models',type=Path,default=ROOT/'models');ap.add_argument('--features',type=Path);ap.add_argument('--data',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',choices=['mps','cpu'],default='mps');args=ap.parse_args();start=time.perf_counter();models=args.models.resolve();out=args.output.resolve();out.mkdir(exist_ok=False)
 manifest=read(models/'manifest.json')
 for file,hsh in manifest['files'].items():assert sha(models/file)==hsh
 if args.features is None:
  assert args.data is not None
  shutil.copyfile(models/'training-validation.json',out/'training-validation.json')
  code="import sys;sys.path.insert(0,sys.argv[1]);import infer;infer.guards();import extract;sys.argv=[str(extract.ROOT/'extract.py'),'test','--data',sys.argv[2],'--run',sys.argv[3]];extract.main()"
  subprocess.run([sys.executable,'-c',code,str(ROOT),str(args.data.resolve()),str(out)],check=True,cwd=ROOT)
  feature=out/'test-features.npz'
 else:feature=args.features.resolve()
 meta=read(feature.with_name('test-feature-validation.json'));assert meta['passed'] and meta['rows']==1000 and sha(feature)==meta['feature_sha256']
 with np.load(feature,allow_pickle=False) as z:d={key:z[key].copy() for key in z.files}
 assert 'y' not in d and 'g' not in d and len(set(d['ids']))==1000;assert d['ids'].tolist()==[f'TEST_{j:06d}' for j in range(1,1001)]
 result=predict(d,models,args.device);repeat=predict(d,models,args.device)
 for key in result:np.testing.assert_array_equal(result[key],repeat[key])
 csvfile=out/'reproduced-018.csv'
 with csvfile.open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(d['ids'],result['primary']))
 with csvfile.open(newline='') as f:rows=list(csv.DictReader(f))
 assert [r['ID'] for r in rows]==d['ids'].tolist();np.testing.assert_array_equal([float(r['hardness']) for r in rows],result['primary'])
 with (out/'predictions.npz').open('xb') as f:np.savez_compressed(f,ids=d['ids'],**result)
 r={'passed':True,'rows':1000,'device':args.device,'finite':True,'ID_order_exact':True,'same_device_reload_exact':True,'CSV_roundtrip_exact':True,'CSV_sha256':sha(csvfile),'predictions_sha256':sha(out/'predictions.npz'),'feature_sha256':sha(feature),'model_manifest_sha256':sha(models/'manifest.json'),'fit_backward_guards':True,'test_labels_read':False,'test_fit_tuning':False,'seconds':time.perf_counter()-start};write(out/'inference-validation.json',r);print(json.dumps(r),flush=True)
if __name__=='__main__':main()
