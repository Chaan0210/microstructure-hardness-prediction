"""Reload, fold-only statistics and rawview provenance verification for104."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,time,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
import patch_phase_104 as s

def main(split):
 start=time.perf_counter();s.check();torch.set_num_threads(2);d=s.inputs(split);r=s.read(s.OUT/split/'report.json');path=s.OUT/split/'development_predictions.npz';assert s.sha(path)==r['predictions_sha256']
 with np.load(_repo_io(path),allow_pickle=False) as z:arr={k:z[k].copy() for k in z.files}
 np.testing.assert_array_equal(arr['ids'],d['ids']);np.testing.assert_array_equal(arr['y'],d['y']);np.testing.assert_array_equal(arr['fold'],d['fid']);np.testing.assert_array_equal(arr['reference011'],d['reference011']);np.testing.assert_array_equal(arr['reference009'],d['reference009'])
 with np.load(_repo_io(s.OUT/'development_features.npz'),allow_pickle=False) as z:V=z['views'];np.testing.assert_array_equal(d['B'],V.mean(1,dtype=np.float64))
 with np.load(_repo_path(ROOT, 'artifacts/dinob-062/development_features.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(V[:,0],z['X'])
 backbone=s.b.model();reload=0.;cpuerr=0.
 for k in range(5):
  fd=s.OUT/split/f'fold-{k}';rec=s.read(fd/'report.json');assert s.sha(fd/'report.json')==r['fold_report_sha256'][str(k)];assert s.sha(fd/'model.pt')==rec['checkpoint_sha256'];a=torch.load(_repo_io(fd/'model.pt'),map_location='cpu',weights_only=True);tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);sc=StandardScaler().fit(d['B'][tr]);assert a['train_ids']==[d['ids'][i] for i in tr] and a['validation_ids']==[d['ids'][i] for i in va] and a['config']==s.e.CFG
  for name,value in [('scaler_mean',sc.mean_),('scaler_scale',sc.scale_),('scaler_var',sc.var_)]:np.testing.assert_array_equal(a[name].numpy(),value)
  assert a['target_mean']==float(d['y'][tr].mean()) and a['target_std']==float(d['y'][tr].std());m=s.e.Model(backbone,sc,s.e.CFG['seed']+k,False);m.load_state_dict(a['state']);m=m.to('mps');pv=s.e.predict(m,d,None,va)*a['target_std']+a['target_mean'];reload=max(reload,float(abs(pv-arr['phase_head'][va]).max()));np.testing.assert_array_equal(pv,arr['phase_head'][va]);m=m.cpu().eval()
  # Independent head-only CPU calculation, without Model.forward or helper inverse transform.
  with torch.no_grad():
   x=torch.tensor(d['B'][va],dtype=torch.float32);x=(x-a['state']['mean'])/a['state']['scale'];st=a['state'];x=torch.nn.functional.gelu(torch.nn.functional.linear(x,st['head.net.0.weight'],st['head.net.0.bias']));q=torch.nn.functional.linear(x,st['head.net.3.weight'],st['head.net.3.bias']).squeeze(1).numpy().astype(float)*a['target_std']+a['target_mean']
  cpuerr=max(cpuerr,float(abs(q-pv).max()));np.testing.assert_allclose(q,pv,rtol=0,atol=.003);del m;torch.mps.empty_cache()
 np.testing.assert_array_equal(arr['primary'],.9*d['reference009']+.1*arr['phase_head']);assert s.comparison(d['y'],arr['reference011'],arr['primary'],d['fid'])==r['comparison'];assert (r['comparison']['gain']>0)==r['advance']
 replay=None;featureerr=None
 if split=='original':
  pert=dict(d);pert['y']=d['y'].copy();pert['y'][d['fid']==0]+=12345.;pv,rr=s.e.fit_arm(pert,None,backbone,0,'frozen',s.OUT/'validation-replay-original');rec=s.read(s.OUT/'original/fold-0/report.json');assert rr['final_state_sha256']==rec['final_state_sha256'];np.testing.assert_array_equal(pv,arr['phase_head'][d['fid']==0]);replay={'full12epoch_state_exact':True,'outer_validation_labels_perturbed':True}
  backbone=backbone.to('mps');featureerr=0.
  for i in (0,199,399):
   a=s.p6.input_image(_repo_path(ROOT, 'data/train', f'{d["ids"][i]}.png'))
   for j,(dy,dx) in enumerate(s.OFFSETS):
    yy=np.arange(dy,dy+504);xx=np.arange(dx,dx+504);yy=np.where(yy<504,yy,1006-yy);xx=np.where(xx<504,xx,1006-xx);view=np.ascontiguousarray(a[:,yy[:,None],xx[None,:]]);np.testing.assert_array_equal(view,s.views(a)[j]);z=s.b.feature(backbone,torch.from_numpy(view[None]).to('mps')).cpu().numpy()[0];featureerr=max(featureerr,float(abs(z-V[i,j]).max()));np.testing.assert_array_equal(z,V[i,j])
 s.check();result={'passed':True,'split':split,'models_verified':5,'MPS_reload_max_error':reload,'independent_head_CPU_MPS_max_error':cpuerr,'raw_reflection_view_feature_max_error':featureerr,'raw_feature_replay_images':3 if split=='original' else 0,'foldtrain_scalers_targets_ids_verified':True,'formula_verified':True,'full_fold_replay':replay,'advance':r['advance'],'report_sha256':s.sha(s.OUT/split/'report.json'),'source_sha256':s.sha(Path(__file__)),'seconds':time.perf_counter()-start,'holdout_test_used':False,'cost':0};s.write(_repo_path(ROOT, f'reports/patch-phase-104-{split}-validation.json'),result);s.emit(result)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('split',choices=s.SPLITS);main(ap.parse_args().split)
