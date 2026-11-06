"""Scope, savedstate, train-only normalization and CPU/MPS replay audit for106."""
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
import amplitude_cnn_106 as s

def main(split):
 start=time.perf_counter();s.check();torch.set_num_threads(2);d=s.inputs(split);r=s.read(s.OUT/split/'report.json');path=s.OUT/split/'development_predictions.npz';assert s.sha(path)==r['predictions_sha256']
 with np.load(_repo_io(path),allow_pickle=False) as z:arr={k:z[k].copy() for k in z.files}
 np.testing.assert_array_equal(arr['ids'],d['ids']);np.testing.assert_array_equal(arr['y'],d['y']);np.testing.assert_array_equal(arr['fold'],d['fid']);np.testing.assert_array_equal(arr['reference011'],d['ref']);reload=0.;cpuerr=0.;paired=[]
 for k in range(5):
  tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);assert not set(d['g'][tr])&set(d['g'][va]);n=s.norms(d,tr);recs={}
  for arm in s.ARMS:
   fd=s.OUT/split/f'{arm}-fold{k}';rec=s.read(fd/'report.json');assert s.sha(fd/'report.json')==r['fold_report_sha256'][str(k)][arm];assert s.sha(fd/'model.pt')==rec['checkpoint_sha256'];a=torch.load(_repo_io(fd/'model.pt'),map_location='cpu',weights_only=True);assert a['normalization']==n and a['config']==s.CFG and a['train_ids']==[d['ids'][i] for i in tr] and a['validation_ids']==[d['ids'][i] for i in va];assert s.statehash(a['state'])==rec['final_state_sha256'];m=s.Net(arm);m.load_state_dict(a['state']);m=m.to('mps');pred=s.predict(m,d,va,n);err=float(abs(pred-arr[arm][va]).max());reload=max(reload,err);np.testing.assert_array_equal(pred,arr[arm][va]);m=m.cpu().eval()
   # Raw unaugmented forward with separately constructed affine input and target inverse.
   x=np.asarray(d['X'][va[:8]],dtype=np.float32)/np.float32(255);x=(x-np.float32(n['x_mean']))/np.float32(n['x_std'])
   with torch.no_grad():q=m(torch.from_numpy(x)).numpy().astype(float)*n['y_std']+n['y_mean']
   err=float(abs(q-pred[:8]).max());cpuerr=max(cpuerr,err);np.testing.assert_allclose(q,pred[:8],rtol=0,atol=.005);recs[arm]=rec;del m;torch.mps.empty_cache()
  for key in ('initial_state_sha256','order_sha256','augmentation_sha256'):assert recs['groupnorm'][key]==recs['affine'][key]
  paired.append(k)
 for arm in s.ARMS:np.testing.assert_array_equal(arr[arm+'_blend'],.9*d['ref']+.1*arr[arm])
 assert s.compare(d['y'],arr['reference011'],arr['affine_blend'],d['fid'])==r['affineblend_vs011'];assert s.compare(d['y'],arr['groupnorm_blend'],arr['affine_blend'],d['fid'])==r['affineblend_vsgroupnormblend']
 assert (r['affineblend_vs011']['gain']>0 and r['affineblend_vsgroupnormblend']['gain']>0)==r['advance']
 # Full replay is reserved for original fold0 both arms; perturb outervalidation labels.
 replay={}
 if split=='original':
  dd=dict(d);dd['y']=d['y'].copy();dd['y'][d['fid']==0]+=12345.
  for arm in s.ARMS:
   p,rec=s.fit(dd,0,arm,s.OUT/f'validation-replay-{arm}');old=s.read(s.OUT/'original'/f'{arm}-fold0/report.json');assert rec['final_state_sha256']==old['final_state_sha256'];np.testing.assert_array_equal(p,arr[arm][d['fid']==0]);replay[arm]={'full80epoch_state_exact':True,'outer_validation_labels_perturbed':True}
 s.check();out={'passed':True,'split':split,'models_verified':10,'reload_max_error':reload,'CPU_MPS_prediction_max_error':cpuerr,'foldtrain_normalizations_ids_verified':True,'identical_initial_states_order_augmentation_verified_folds':paired,'full_replay':replay,'advance':r['advance'],'report_sha256':s.sha(s.OUT/split/'report.json'),'source_sha256':s.sha(Path(__file__)),'seconds':time.perf_counter()-start,'holdout_test_used':False,'cost':0};s.write(_repo_path(ROOT, f'reports/amplitude-cnn-106-{split}-validation.json'),out);s.emit(out)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('split',choices=s.SPLITS);main(ap.parse_args().split)
