"""Three fresh500 Lheads under immutable177 recipe, with fullseed replay."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import time
import numpy as np,torch
import runtime as f,mlp_head as h
R=f.R;P=f.P

def check():
 plan=f.check()
 for name,hh in f.read(R/'training-code-lock.json')['sources'].items():assert f.sha(R/name)==hh
 assert h.CONFIG==plan['recipe']['head_config'];assert f.read(R/'train-feature-validation.json')['passed'];assert f.sha(R/'train-features.npz')==f.read(R/'train-feature-validation.json')['feature_sha256'];return plan

def predict(obj,X,device='cpu'):
 m=h.Head(9216);m.load_state_dict(obj['model']);Z=(X-obj['scaler_mean'].numpy())/obj['scaler_scale'].numpy()
 if device=='cpu':return h.predict(m,Z,obj['target_mean'],obj['target_std'])
 m=m.to(device);m.eval()
 with torch.inference_mode():p=m(torch.tensor(Z,dtype=torch.float32,device=device))[0].cpu().numpy().astype(float)*obj['target_std']+obj['target_mean']
 del m;torch.mps.empty_cache();return p

def main():
 plan=check();out=R/'models';out.mkdir(exist_ok=False);start=time.perf_counter()
 with np.load(_repo_io(R/'train-features.npz'),allow_pickle=False) as z:ids=z['ids'].copy();X=z['X'].copy();y=z['y'].copy()
 assert ids.tolist()==[f'TRAIN_{j:06d}' for j in range(1,501)] and X.shape==(500,9216) and np.isfinite(y).all()
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,f.os.PathLike)):
   p=f.Path(f.os.fsdecode(args[0])).absolute()
   if p.is_relative_to(P/'data') or p.is_relative_to(P/'artifacts'):raise RuntimeError('Finalhead fitcannotreadrawdata orhistoricalartifacts: '+str(p))
 f.sys.addaudithook(audit);logs={};pred=[];cpu_mps=[];replay=[]
 with f.Monitor('final-training',plan['resources']):
  sc,ym,ys,_,bins=h.preprocess(X,y);xt=torch.tensor(sc.transform(X),dtype=torch.float32);yt=torch.tensor((y-ym)/ys,dtype=torch.float32);bt=torch.tensor(bins);np.testing.assert_allclose(sc.mean_,X.mean(0),atol=1e-12,rtol=0);np.testing.assert_allclose(sc.scale_,X.std(0),atol=1e-12,rtol=0);assert ym==float(y.mean()) and ys==float(y.std())
  for seed in plan['recipe']['seeds']:
   m,log=h.fit(xt,yt,bt,seed,0.);pp=h.predict(m,sc.transform(X),ym,ys);obj={'model':m.state_dict(),'input_dim':9216,'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'scaler_var':torch.tensor(sc.var_),'target_mean':ym,'target_std':ys,'train_ids':ids.tolist(),'seed':seed,'config':h.CONFIG,'effective_aux_weight':0.};path=out/f'L-MLP-{seed}.pt';torch.save(obj,_repo_io(path));del m;loaded=torch.load(_repo_io(path),map_location='cpu',weights_only=True);actual=predict(loaded,X);np.testing.assert_array_equal(pp,actual);pm=predict(loaded,X,'mps');err=float(abs(actual-pm).max());assert err<plan['tolerances']['CPU_MPS_prediction_abs'];cpu_mps.append(err);pred.append(actual);logs[str(seed)]={**log,'file_sha256':f.sha(path),'same_device_reload_max_error':0.,'CPU_MPS_max_error':err};f.emit({'fresh500_seed_complete':seed,'seconds':time.perf_counter()-start,'CPU_MPS_max_error':err})
  for seed in plan['recipe']['seeds']:
   m,log=h.fit(xt,yt,bt,seed,0.);old=torch.load(_repo_io(out/f'L-MLP-{seed}.pt'),map_location='cpu',weights_only=True);assert all(torch.equal(value,old['model'][name]) for name,value in m.state_dict().items());np.testing.assert_array_equal(h.predict(m,sc.transform(X),ym,ys),predict(old,X));replay.append({'seed':seed,'all_tensor_states_exact':True,'predictions_exact':True,'batch_order_exact':log['batch_order_sha256']==logs[str(seed)]['batch_order_sha256']});assert replay[-1]['batch_order_exact'];del m
  with (R/'training-reload-predictions.npz').open('xb') as ff:np.savez_compressed(_repo_io(ff),ids=ids,seeds=np.stack(pred),mean3=np.mean(pred,0))
 rec={'passed':True,'rows':500,'fresh_main_fits':3,'full_seed_training_replays':replay,'seeds':logs,'scaler_and_target_statistics_all500_verified':True,'same_device_reload_max_error':0.,'CPU_MPS_max_error':max(cpu_mps),'train_target_range':[float(y.min()),float(y.max())],'no_holdout_subgroup_score':True,'test_read':False,'selection_changed':False,'feature_sha256':f.sha(R/'train-features.npz'),'training_code_lock_sha256':f.sha(R/'training-code-lock.json'),'seconds':time.perf_counter()-start};f.write(R/'training-validation.json',rec)
 manifest={'candidate':'021','L_head_weight':.1,'018_fixed_weight':.9,'fresh_L_main_heads':3,'seed_order':plan['recipe']['seeds'],'validation_sha256':f.sha(R/'training-validation.json'),'plan_sha256':f.sha(R/'plan.json'),'L_weight_provenance_sha256':f.sha(R/'L-provenance.json'),'files':{p.name:f.sha(p) for p in out.glob('*.pt')}};f.write(out/'manifest.json',manifest);f.emit({k:v for k,v in rec.items() if k!='seeds'})
if __name__=='__main__':
 try:main()
 except Exception as e:f.write(R/'training-failure.json',{'error':repr(e)});raise
