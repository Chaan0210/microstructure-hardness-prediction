"""021 frozencomponent replay and1000-row CSV creation; no fitting/tuning."""
import argparse,csv,time,subprocess,shutil
from pathlib import Path
import numpy as np,torch
import runtime as f,mlp_head as h
from train import predict
R=f.R;P=f.P;OLD=P/'experiments/reproduction-018-176'

def table(path):
 with path.open(newline='',encoding='utf-8-sig') as ff:
  r=csv.DictReader(ff);assert r.fieldnames==['ID','hardness'];rows=list(r)
 return [v['ID'] for v in rows],np.array([float(v['hardness']) for v in rows])
def stats(x):return {'min':float(x.min()),'max':float(x.max()),'mean':float(x.mean()),'std':float(x.std())}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--candidate',action='store_true');args=ap.parse_args();out=args.output.resolve();out.mkdir(exist_ok=False);start=time.perf_counter();plan=f.check()
 for name,hh in f.read(R/'inference-code-lock.json')['sources'].items():assert f.sha(R/name)==hh
 cert=f.read(R/'training-validation.json');assert cert['passed'];manifest=f.read(R/'models/manifest.json');assert manifest['validation_sha256']==f.sha(R/'training-validation.json')
 for name,hh in manifest['files'].items():assert f.sha(R/'models'/name)==hh
 meta=f.read(R/'test-feature-validation.json');assert meta['passed'] and meta['rows']==1000 and f.sha(R/'test-features.npz')==meta['feature_sha256'];f.inference_guards();h.fit=f.blocked
 with np.load(R/'test-features.npz',allow_pickle=False) as z:assert z.files==['ids','X'];ids=z['ids'].tolist();X=z['X'].copy()
 assert ids==[f'TEST_{j:06d}' for j in range(1,1001)];assert len(set(ids))==1000 and X.shape==(1000,9216) and np.isfinite(X).all()
 refids,reference=table(P/'artifacts/submission-candidate-018.csv');assert refids==ids and np.isfinite(reference).all();assert f.sha(P/'artifacts/submission-candidate-018.csv')==plan['protected']['artifacts/submission-candidate-018.csv'];base={};basecert={}
 # Explicit reuse of176 already-validated018models + fresh176 testfeaturecache.
 for device in ['mps','cpu']:
  dest=out/f'018-{device}-replay';cmd=[f.sys.executable,'-B',str(OLD/'infer.py'),'--features',str(OLD/'test-run/test-features.npz'),'--models',str(OLD/'models'),'--output',str(dest),'--device',device]
  with (out/f'018-{device}.log').open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=P)
  v=f.read(dest/'inference-validation.json');assert v['passed'];ii,bb=table(dest/'reproduced-018.csv');assert ii==ids;base[device]=bb;basecert[device]=v
 np.testing.assert_array_equal(base['mps'],reference);assert basecert['mps']['CSV_sha256']==f.sha(P/'artifacts/submission-candidate-018.csv');L={};spot_results=[]
 with f.Monitor('inference-'+out.name,plan['resources']):
  for device in ['cpu','mps']:
   values=[]
   for seed in manifest['seed_order']:
    obj=torch.load(R/'models'/f'L-MLP-{seed}.pt',map_location='cpu',weights_only=True);pp=predict(obj,X,device);repeat=predict(torch.load(R/'models'/f'L-MLP-{seed}.pt',map_location='cpu',weights_only=True),X,device);np.testing.assert_array_equal(pp,repeat);values.append(pp)
   L[device]=np.stack(values);assert np.isfinite(L[device]).all()
  # Canonical021 is original018 + CPU-trained/readout Lheads on frozenMPSfeatures.
  new=.9*reference+.1*L['cpu'].mean(0);alternate=.9*base['cpu']+.1*L['mps'].mean(0);drift=float(abs(new-alternate).max());head_drift=float(abs(L['cpu']-L['mps']).max());assert max(drift,head_drift)<plan['tolerances']['CPU_MPS_prediction_abs'];assert np.isfinite(new).all()
  with np.load(R/'test-CPU-feature-spotcheck.npz',allow_pickle=False) as spot:
   for seed in manifest['seed_order']:
    obj=torch.load(R/'models'/f'L-MLP-{seed}.pt',map_location='cpu',weights_only=True);a=predict(obj,spot['CPU'].astype(float),'cpu');b=predict(obj,spot['MPS'].astype(float),'cpu');spot_results.append(float(abs(a-b).max()))
  assert max(spot_results)<plan['tolerances']['CPU_MPS_prediction_abs'];lo,hi=cert['train_target_range'];span=hi-lo;assert new.min()>0 and new.min()>=lo-span and new.max()<=hi+span
  file=out/'submission-candidate-021.csv'
  with file.open('x',newline='') as ff:w=csv.writer(ff);w.writerow(['ID','hardness']);w.writerows(zip(ids,new))
  ii,rr=table(file);assert ii==ids;np.testing.assert_array_equal(rr,new)
  with (out/'predictions.npz').open('xb') as ff:np.savez_compressed(ff,ids=ids,reference018=reference,L_CPU_seeds=L['cpu'],L_MPS_seeds=L['mps'],primary=new,alternate_CPU_MPS=alternate)
 protected={name:f.sha(P/name) for name in plan['protected']};assert protected==plan['protected'];sha=f.sha(file)
 rec={'passed':True,'candidate':'021','CSV_sha256':sha,'rows':1000,'columns':['ID','hardness'],'ID_unique':True,'ID_order_exact':True,'finite':True,'CSV_roundtrip_exact':True,'fixed_formula':'.9*unchanged018CSV+.1*mean3fresh500LMLP','same_device_L_model_reload_exact':True,'018_model_replay_CSV_byte_exact':True,'018_CPU_MPS_max_error':float(abs(base['mps']-base['cpu']).max()),'L_readout_CPU_MPS_max_error':head_drift,'composite_CPU_MPS_max_error':drift,'CPU_MPS_comparison_scope':'All1000regressionpredictions;LencoderCPUspot3testimages separately,notfull1000CPUencoderreextraction. 018encoderfeatures reusedfrom176.','L_feature_CPU_spot':meta['test_CPU_spot'],'L_prediction_CPU_feature_spot_max_error':max(spot_results),'prediction_range':stats(new),'L_prediction_range':stats(L['cpu'].mean(0)),'018_prediction_range':stats(reference),'training_target_range':[lo,hi],'below_train_min':int((new<lo).sum()),'above_train_max':int((new>hi).sum()),'no_clipping':True,'candidate_delta_vs018':{'max_abs':float(abs(new-reference).max()),'RMS':float(np.sqrt(np.mean((new-reference)**2)))},'new_L_head_manifest_sha256':f.sha(R/'models/manifest.json'),'reused_018_model_manifest_sha256':f.sha(OLD/'models/manifest.json'),'reused_018_CSV_sha256':f.sha(P/'artifacts/submission-candidate-018.csv'),'fresh_L_test_feature_sha256':meta['feature_sha256'],'reused_018_test_feature_sha256':basecert['mps']['feature_sha256'],'fit_backward_guards':True,'test_labels_parsed':False,'test_fitting_or_selection':False,'protected_files_unchanged':protected,'seconds':time.perf_counter()-start,'source_sha256':f.sha(Path(__file__))};f.write(out/'validation.json',rec)
 if args.candidate:
  target=P/'artifacts/submission-candidate-021.csv';assert not target.exists();assert not (P/'artifacts/submission-candidate-021.manifest.json').exists()
  with file.open('rb') as src,target.open('xb') as dest:shutil.copyfileobj(src,dest)
  assert f.sha(target)==sha;f.write(P/'artifacts/submission-candidate-021.manifest.json',{'candidate':'021','CSV_absolute_path':str(target),'CSV_sha256':sha,'validation':str((out/'validation.json').relative_to(P)),'validation_sha256':f.sha(out/'validation.json'),'selection_audit':str((R/'selection-audit.json').relative_to(P)),'selection_audit_sha256':f.sha(R/'selection-audit.json'),'plan_sha256':f.sha(R/'plan.json'),'uploaded':False,'cost':0})
 f.emit({k:v for k,v in rec.items() if k!='protected_files_unchanged'})
if __name__=='__main__':
 try:main()
 except Exception as e:f.write(R/f'inference-failure-{int(time.time())}.json',{'error':repr(e)});raise
