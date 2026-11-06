"""Official Kymatio CPU compatibility and resource checks; no labels or fitting."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
import sys,time,json,resource,hashlib
from pathlib import Path
import numpy as np
from kymatio.numpy import Scattering2D
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==_repo_path(ROOT, '.venv')
OUT=_repo_path(ROOT, 'artifacts/wavelet-scattering-096')
def emit(x):print(json.dumps(x),flush=True)
start=time.perf_counter();small=Scattering2D(J=2,shape=(32,32),L=8,max_order=2)
rng=np.random.default_rng(20270320);x=rng.random((32,32),dtype=np.float32)
a=small(x);assert a.shape==(81,8,8) and np.isfinite(a).all()
zero=small(np.zeros((32,32),np.float32));assert np.max(abs(zero))==0
constant=small(np.ones((32,32),np.float32));assert np.max(abs(constant[1:]))<1e-5
np.testing.assert_array_equal(a,small(x));np.testing.assert_allclose(small(2*x),2*a,atol=1e-6,rtol=1e-6)
emit({'small_test_pass':True,'shape':list(a.shape),'seconds':time.perf_counter()-start})
t=time.perf_counter();sc=Scattering2D(J=6,shape=(256,256),L=8,max_order=2);build=time.perf_counter()-t;emit({'native_filter_build_seconds':build})
x=rng.random((256,256),dtype=np.float32);times=[]
for i in range(3):
 t=time.perf_counter();a=sc(x);times.append(time.perf_counter()-t);assert a.shape==(1009,4,4) and np.isfinite(a).all()
 if i==0:first=a.copy()
 else:np.testing.assert_array_equal(first,a)
 emit({'native_repetition':i+1,'seconds':times[-1]})
sc.max_order=1;t=time.perf_counter();b=sc(x);s1time=time.perf_counter()-t
assert b.shape==(49,4,4) and np.isfinite(b).all();np.testing.assert_array_equal(a[:49],b)
f=a.mean((-2,-1),dtype=np.float64);f1=b.mean((-2,-1),dtype=np.float64)
assert f.shape==(1009,) and f1.shape==(49,)
r={'passed':True,'device':'CPU','backend':'official kymatio.numpy.Scattering2D','dtype':'float32 input','small_shape':[81,8,8],'zero_exact':True,'constant_nonzero_order_max':float(abs(constant[1:]).max()),'positive_homogeneity_pass':True,'native_shape':list(a.shape),'order1_shape':list(b.shape),'pooled_dims':[len(f1),len(f)],'order1_prefix_exact':True,'repeat_exact':True,'finite':True,'filter_build_seconds':build,'native_order2_seconds':times,'native_order1_seconds':s1time,'projected400_seconds_from_synthetic':float(np.median(times)*400),'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'total_seconds':time.perf_counter()-start,'real_images_or_labels_accessed':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
r['resource_pass']=r['rss_peak_bytes']<2e9 and r['projected400_seconds_from_synthetic']<1800
(_repo_path(OUT, 'smoke.json')).write_text(json.dumps(r,indent=2)+'\n');emit(r)
