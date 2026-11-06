from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,math,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
P=Path(__file__).resolve().parents[4];R=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
import diagnose as d
d.protect();d.guard();report=d.read(_repo_path(R, 'L1.json'));q=d.load(_repo_path(R, 'L1-paired-losses.npz'));y=q['y'];checks=[];maxerror=0
for c in report['candidates']:
 p=q[c['key']+'_prediction'];b=q[c['key']+'_baseline'];change=q[c['key']+'_paired_SE_change'];delta=p-b;r=y-b
 A=2*math.fsum((r*delta).tolist())/400;pen=math.fsum((delta*delta).tolist())/400
 assert abs(A-c['A_gain'])<1e-10 and abs(pen-c['P'])<1e-10
 np.testing.assert_allclose(change,(y-p)**2-(y-b)**2,rtol=0,atol=1e-10)
 for v in c['grid']:
  l=v['lambda'];direct=math.fsum(((y-(b+l*delta))**2-(y-b)**2).tolist())/400;err=abs(direct-(-l*A+l*l*pen));assert err<1e-10;maxerror=max(maxerror,err)
 if A<=0:assert c['lambda_nonnegative_oracle']==0
 if 0<c['lambda_nonnegative_oracle']<1:assert abs(-A+2*pen*c['lambda_nonnegative_oracle'])<1e-10
checks.append({'L1_comparisons':len(report['candidates']),'independent_fsum_max_MSE_identity_error':maxerror})
validation_sources=[]
for n in range(205,218):
 root=next(x for x in (_repo_path(P, 'experiments')).iterdir() if x.name.endswith('-'+str(n)));f=root/'validation.json';d.manifest(root,f);assert d.read(f)['passed'];validation_sources.append({'study':n,'path':str(f.relative_to(P)),'sha256':d.sha(f)})
for split in ['original','20261031','20261101']:
 z=d.data(split);plan=d.read(_repo_path(P, 'experiments/resolution-216/plan.json' if split=='original' else 'experiments/resolution-repartition-217/plan.json'));folds=plan['folds'];folds=folds[split] if split in folds else folds
 for k in range(5):
  np.testing.assert_array_equal(np.flatnonzero(z['fold']==k),folds[str(k)]['validation']);np.testing.assert_array_equal(np.flatnonzero(z['fold']!=k),folds[str(k)]['train'])
X=d.load(_repo_path(P, 'experiments/dinol-177/features.npz'))['X'][:,6144:7168];dist=cdist(X,X,'cosine');np.fill_diagonal(dist,np.inf);ix=np.argsort(dist,axis=1,kind='stable')[:,:10];nn=d.load(_repo_path(R, 'L2-neighbors.npz'));np.testing.assert_array_equal(ix,nn['indices']);err=float(np.max(abs(np.take_along_axis(dist,ix,axis=1)-nn['distances'])));assert err<1e-12
l3=d.read(_repo_path(R, 'L3.json'));pooled=[]
for split in ['original','20261031','20261101']:
 z=d.data(split);e=(z['y']-z['blend_L_MLP'])**2;pooled.append(e);s=l3['results'][split];assert abs(sum(t['SSE'] for t in s['deciles'])-e.sum())<1e-8;assert len(set(i for t in s['deciles'] for i in t['IDs']))==400
s=l3['results']['pooled_meanSE'];assert abs(s['total_SSE']-np.mean(pooled,axis=0).sum())<1e-8
m=d.read(_repo_path(R, 'L4-measurements.json'));cache=d.load(_repo_path(R, 'L4-measurements.npz'));old=d.load(_repo_path(P, 'artifacts/run-022/development_features.npz'));np.testing.assert_allclose(cache['descriptors'],old['X'][:,-31:],rtol=0,atol=1e-12);errcount=0
for i,row in enumerate(m['rows']):
 for a in row['configs']:
  assert a['all_count']==a['inside_count']+a['border_count'];assert a['empty0']==(a['inside_count']==0)
  assert abs(a['border_area_fraction']+a['small_area_fraction']+a['interior_area_fraction']+a['watershed_line_fraction']-1)<1e-12
 a=row['configs'][-1];sup=cache['support'][i]
 for code,key in [(0,'watershed_line_fraction'),(1,'interior_area_fraction'),(2,'border_area_fraction'),(3,'small_area_fraction')]:assert abs(np.mean(sup==code)-a[key])<1e-12
result={'passed':True,'new_training_fits':0,'L1':checks,'source_validation_artifacts':validation_sources,'allthree_folds_match_saved_plans':True,'L2_scipy_cosine_maxerror':err,'L3_partition_and_pooled_SSE_verified':True,'L4_images':400,'L4_all1600_config_support_sums_and_count_partition':True,'L4_cached_descriptors_exact':True,'L4_final_config_support_maps_verified':400,'protected_files':len(d.read(_repo_path(R, 'plan.json'))['protected']),'limitations':'No model retraining replay; current validation checks stored records, arithmetic, identifiers and current extraction measurements.'}
d.protect();d.write(_repo_path(R, 'verification.json'),result);print(json.dumps(result))
