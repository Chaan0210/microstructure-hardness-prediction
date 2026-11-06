"""Independent quantile interpolation and train-only reconstruction checks."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,time,csv
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import sliced_distribution_099 as s
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from scipy.special import ndtri

def manualstats(v):
 sort=np.sort(v,axis=0);ix=s.P*(len(v)-1);lo=np.floor(ix).astype(int);hi=np.ceil(ix).astype(int);w=ix-lo;q=(sort[lo]*(1-w[:,None])+sort[hi]*w[:,None]).T;mu=v.mean(0);sd=v.std(0);return q,(q-mu[:,None])/np.maximum(sd[:,None],1e-12)-ndtri(s.P)[None,:]
def run():
 start=time.perf_counter();p=s.check();r=s.read(s.OUT/'diagnostic-report.json');ex=s.read(s.OUT/'extraction.json');assert s.sha(s.OUT/'summaries.npz')==ex['summary_sha256'] and s.sha(s.CACHE/'patch_tokens.npy')==p['token_sha256'];D=s.directions();np.testing.assert_allclose(np.sum(D*D,axis=0),1,rtol=0,atol=1e-14)
 with np.load(_repo_io(s.OUT/'summaries.npz'),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 ids=a['ids'].tolist();T=np.load(_repo_io(s.CACHE/'patch_tokens.npy'),mmap_mode='r');yy,xx=np.indices((36,36));masks=[(yy+xx)%2==0,(yy+xx)%2==1,xx<18,xx>=18,(yy<18)&(xx<18),(yy<18)&(xx>=18),(yy>=18)&(xx<18),(yy>=18)&(xx>=18)];maxq=0.
 for i in [0,199,399]:
  raw=np.asarray(T[i],dtype=float);v=raw@D;q,h=manualstats(v);maxq=max(maxq,float(abs(q-a['Q'][i]).max()));np.testing.assert_allclose(q,a['Q'][i],rtol=0,atol=1e-12);np.testing.assert_allclose(h,a['H'][i],rtol=0,atol=1e-12);np.testing.assert_array_equal(np.r_[raw.mean(0),raw.std(0)],a['M'][i])
  for ri,mask in enumerate(masks):
   q,h=manualstats(v[mask.ravel()]);np.testing.assert_allclose(q,a['regionQ'][i,ri],rtol=0,atol=1e-12);np.testing.assert_allclose(h,a['regionH'][i,ri],rtol=0,atol=1e-12)
 del T
 with np.load(_repo_io(s.OUT/'reconstruction_predictions.npz'),allow_pickle=False) as z:pred={k:z[k].copy() for k in z.files}
 audit=s.read(s.OUT/'reconstruction-audit.json');fid=pred['fold'];sp=s.read(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json'));assert ids==sorted(sp['development_folds']);np.testing.assert_array_equal(fid,[sp['development_folds'][i] for i in ids]);g=np.array([sp['group_by_id'][i] for i in ids])
 with (_repo_path(ROOT, 'artifacts/grain-visual-audit-097/all400_noise_contrast.csv')).open() as f:noise={z['ID']:[float(z['noise']),float(z['contrast_q95_q05'])] for z in csv.DictReader(f)}
 N=np.array([noise[i] for i in ids]);maxpred=0.;recomputed={}
 for name,X,Y,kind,reportkey in [('M_to_H',a['M'],a['H'].reshape(400,-1),'linear','M_to_H'),('M_to_Q',a['M'],a['Q'].reshape(400,-1),'linear','M_to_Q'),('N_to_H',N,a['H'].reshape(400,-1),'rbf','noise_contrast_to_H')]:
  direct=np.empty_like(Y);null=np.empty_like(Y)
  for k in range(5):
   tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(g[tr])&set(g[va]);d=audit[name][k];assert tr.tolist()==d['train_indices'] and va.tolist()==d['validation_indices'];mu=X[tr].mean(0);sd=X[tr].std(0);sd=np.where(sd==0,1,sd);np.testing.assert_allclose(mu,d['scaler_mean'],rtol=1e-12,atol=1e-12);np.testing.assert_allclose(sd,d['scaler_scale'],rtol=1e-12,atol=1e-12);A=(X[tr]-mu)/sd;B=(X[va]-mu)/sd;center=Y[tr].mean(0)
   if kind=='linear':coef=solve(A@A.T+1536*np.eye(len(tr)),Y[tr]-center,assume_a='pos');direct[va]=B@A.T@coef+center
   else:
    K=np.exp(-.5*cdist(A,A,'sqeuclidean'));coef=solve(K+np.eye(len(tr)),Y[tr]-center,assume_a='pos');direct[va]=np.exp(-.5*cdist(B,A,'sqeuclidean'))@coef+center
   null[va]=center
  err=float(abs(direct-pred[name]).max());maxpred=max(maxpred,err);np.testing.assert_allclose(direct,pred[name],rtol=0,atol=1e-9);ratio=float(np.sum((Y-direct)**2)/np.sum((Y-null)**2));assert abs(ratio-r[reportkey]['residual_fraction'])<1e-10;recomputed[name]=ratio
 rms=lambda z:np.sqrt(np.mean(z*z,axis=tuple(range(1,z.ndim))));H=a['H'].reshape(400,-1);Q=a['Q'].reshape(400,-1);checker=rms(a['regionH'][:,0]-a['regionH'][:,1]);betweenH=rms(H-np.roll(H,1,axis=0));betweenQ=rms(Q-np.roll(Q,1,axis=0));lr=rms(a['regionQ'][:,2]-a['regionQ'][:,3]);dm=cdist(a['M'],a['M'],'sqeuclidean')/1536;np.fill_diagonal(dm,np.inf);near=dm.argmin(1);mh=rms(H-H[near]);ratios=[np.median(checker)/np.median(betweenH),np.median(lr)/np.median(betweenQ),np.median(mh)/np.median(checker)]
 np.testing.assert_allclose(ratios,[r['region_ratios'][k] for k in ['checker_H_to_between_H','left_right_Q_to_between_Q','nearestmoment_H_to_checker_H']],rtol=0,atol=1e-12)
 gate=p['diagnostic_gate'];passed=(r['synthetic']['quantile_RMS_difference']>=gate['synthetic_same_moments_shape_difference_min'] and recomputed['M_to_H']>=.1 and recomputed['N_to_H']>=.25 and ratios[0]<=.5 and ratios[1]<=1.25 and ratios[2]>=2)
 assert bool(passed)==r['labelblind_gate_pass'];assert all(np.isfinite(z).all() for k,z in a.items() if k!='ids');s.check()
 out={'passed':True,'diagnostic_gate_reproduced':bool(passed),'quantile_and8regions_3images_checked':True,'manual_sorted_quantile_max_error':maxq,'trainonly15reconstructions_checked':True,'independent_kernel_or_dualridge_prediction_max_error':maxpred,'noise_contrast_only_inputs_confirmed':True,'cache_sha256_verified':True,'protected011_files_verified':len(p['protected011']),'hardness_labels_accessed':False,'holdout_test_accessed':False,'uploaded':False,'seconds':time.perf_counter()-start,'diagnostic_report_sha256':s.sha(s.OUT/'diagnostic-report.json'),'source_sha256':s.sha(Path(__file__)),'cost':0};s.write(_repo_path(ROOT, 'reports/sliced-distribution-099-validation.json'),out);s.emit(out)
if __name__=='__main__':run()
