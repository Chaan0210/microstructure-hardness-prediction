"""Savedmodel, matched-control, leakage and independent pixel/feature replay."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,time,hashlib,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
import input_noise_110 as s

def main(split):
 start=time.perf_counter();s.check();torch.set_num_threads(2);d=s.inputs(split);r=s.read(s.OUT/split/'report.json');p=s.OUT/split/'development_predictions.npz';assert s.sha(p)==r['predictions_sha256']
 with np.load(_repo_io(p),allow_pickle=False) as z:arr={k:z[k].copy() for k in z.files}
 np.testing.assert_array_equal(arr['ids'],d['ids']);np.testing.assert_array_equal(arr['y'],d['y']);np.testing.assert_array_equal(arr['fold'],d['fid']);np.testing.assert_array_equal(arr['reference011'],d['reference011']);np.testing.assert_array_equal(arr['reference009'],d['reference009']);np.testing.assert_array_equal(arr['control'],d['reference011']);backbone=s.b.model();cpuerr=0.;reload=0.;controlstates=[]
 for k in range(5):
  tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);sc=StandardScaler().fit(d['B'][tr]);rr={};assert not set(d['g'][tr])&set(d['g'][va])
  for arm in s.ARMS:
   fd=s.OUT/split/f'{arm}-fold{k}';rec=s.read(fd/'report.json');assert s.sha(fd/'report.json')==r['fold_report_sha256'][str(k)][arm];assert s.sha(fd/'model.pt')==rec['checkpoint_sha256'];a=torch.load(_repo_io(fd/'model.pt'),map_location='cpu',weights_only=True);assert a['config']==s.CFG and a['seed']==s.CFG['seed']+k;assert a['train_ids']==[d['ids'][i] for i in tr] and a['validation_ids']==[d['ids'][i] for i in va];np.testing.assert_array_equal(a['scaler_mean'],sc.mean_);np.testing.assert_array_equal(a['scaler_scale'],sc.scale_);np.testing.assert_array_equal(a['scaler_var'],sc.var_);assert a['target_mean']==d['y'][tr].mean() and a['target_std']==d['y'][tr].std();assert s.e.h.state_hash(a['state'])==rec['final_state_sha256'];model=s.e.Model(backbone,sc,s.CFG['seed']+k,False);model.load_state_dict(a['state']);model=model.to('mps');q=s.e.predict(model,d,None,va)*a['target_std']+a['target_mean'];expected=arr[arm+'_head'];np.testing.assert_array_equal(q,expected[va]);reload=max(reload,float(abs(q-expected[va]).max()));model=model.cpu().eval()
   with torch.no_grad():cpu=model(torch.tensor(d['B'][va[:8]],dtype=torch.float32)).numpy().astype(float)*a['target_std']+a['target_mean']
   err=float(abs(cpu-q[:8]).max());cpuerr=max(cpuerr,err);np.testing.assert_allclose(cpu,q[:8],rtol=0,atol=.0001)
   assert s.sha(fd/'predictions.npz')==rec['predictions_sha256']
   with np.load(_repo_io(fd/'predictions.npz'),allow_pickle=False) as z:
    np.testing.assert_array_equal(z['validation_indices'],va);np.testing.assert_array_equal(z['train_indices'],tr);assert np.all(z['noise_count'][va]==0) and np.all(z['noise_count'][tr]==(6 if arm=='noise' else 0))
   assert rec['total_presentations']==3840 and rec['optimizer_updates']==240 and rec['noise_presentations']==(1920 if arm=='noise' else 0);rr[arm]=rec;del model;s.torch.mps.empty_cache()
   if arm=='control':
    old=torch.load(_repo_io(s.old.path_for(split,s.CFG['seed'],k)/'model.pt'),map_location='cpu',weights_only=True);assert s.e.h.state_hash(old['state'])==rec['final_state_sha256'];np.testing.assert_array_equal(q,d['basehead'][va]);controlstates.append(k)
  for name in ('initial_state_sha256','fixed_state_sha256','batch_order_sha256','dropout_seeds_sha256'):assert rr['control'][name]==rr['noise'][name]
 np.testing.assert_array_equal(arr['primary'],.9*d['reference009']+.1*arr['noise_head']);comp,diag,advance,sq=s.statistics(d,arr,split);assert comp==r['comparison'] and diag==r['sample_diagnostics'] and advance==r['advance'];np.testing.assert_array_equal(sq,arr['squared_error_gain'])
 # Refit full noiseheadfold0. Change BOTH heldoutlabels andheldoutNOISYfeatures;
 # normaloriginalvalidationinputs remainunchanged forprediction comparison.
 dd=dict(d);dd['y']=d['y'].copy();dd['y'][d['fid']==0]+=12345.;dd['N']=d['N'].copy();dd['N'][d['fid']==0]+=1000.;q,rr=s.fit(dd,backbone,0,'noise',s.OUT/f'validation-replay-{split}');np.testing.assert_array_equal(q,arr['noise_head'][d['fid']==0]);assert rr['final_state_sha256']==s.read(s.OUT/split/'noise-fold0/report.json')['final_state_sha256']
 features=None
 if split=='original':
  model=backbone.to('mps');ex=s.read(s.OUT/'extraction.json');images=np.load(_repo_io(s.a.OUT/'images.npy'),mmap_mode='r',allow_pickle=False);maxf=0.;inputs=[]
  for i in (0,199,399):
   ident=d['ids'][i];seed=int.from_bytes(hashlib.sha256(('noise110:'+ident).encode()).digest()[:8],'little');raw=np.asarray(images[i,0]);noise=np.random.default_rng(seed).standard_normal(raw.shape)*2.;pixels=np.around(np.minimum(np.maximum(raw.astype(float)+noise,0),255)).astype(np.uint8);assert hashlib.sha256(pixels.tobytes()).hexdigest()==ex['records'][i]['noisy_pixel_sha256'];z=np.array(Image.fromarray(pixels).resize((504,504),Image.Resampling.BICUBIC),dtype=np.float32)/np.float32(255);x=np.stack([(z-np.float32(mu))/np.float32(sd) for mu,sd in zip([.485,.456,.406],[.229,.224,.225])]);np.testing.assert_array_equal(x,s.image_input(pixels));q=s.b.feature(model,torch.from_numpy(x[None]).to('mps')).cpu().numpy()[0];np.testing.assert_array_equal(q,d['N'][i]);maxf=max(maxf,float(abs(q-d['N'][i]).max()));inputs.append(x)
  model=model.cpu();q=s.b.feature(model,torch.from_numpy(inputs[0][None])).cpu().numpy()[0];cpu_feature_err=float(abs(q-d['N'][0]).max());np.testing.assert_allclose(q,d['N'][0],atol=.0003,rtol=0);features={'independent3_pixel_preprocessing_and_MPS_feature_replay_exact':True,'MPS_feature_max_error':maxf,'oneCPU_backbone_feature_max_error':cpu_feature_err};del model
 s.check();out={'passed':True,'split':split,'models_reloaded':10,'MPS_reload_max_error':reload,'CPU_MPS_head_prediction_max_error':cpuerr,'all5_controlstates_exactly_match011':controlstates,'matched_initialization_order_dropout_hashes':True,'train_only_original_scalers_targets_verified':True,'equal_presentations_and_optimizer_updates':True,'candidate_each_train_sample6noisy6identity_verified':True,'fullfold0_replay_with_heldoutlabels_and_noisyfeatures_perturbed_state_exact':True,'noisy_feature_replay':features,'top5_diagnostics_gate_recomputed':True,'report_sha256':s.sha(s.OUT/split/'report.json'),'source_sha256':s.sha(Path(__file__)),'seconds':time.perf_counter()-start,'holdout_test_used':False,'cost':0};s.write(_repo_path(ROOT, f'reports/input-noise-110-{split}-validation.json'),out);s.emit(out)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('split',choices=s.SPLITS);main(ap.parse_args().split)
