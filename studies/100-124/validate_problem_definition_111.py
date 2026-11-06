"""Independent formula checks for the label-free representation audit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,hashlib,importlib.metadata
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
import numpy as np
import torch
OUT=_repo_path(ROOT, 'artifacts/problem-definition-111')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(2**20),b''):h.update(c)
 return h.hexdigest()
def main():
 torch.set_num_threads(2);p=json.loads((_repo_path(OUT, 'plan.json')).read_text());r=json.loads((_repo_path(OUT, 'numerical-audit.json')).read_text());check={}
 for n,h in {**p['protected_sha256'],**p['source_sha256'],**p['existing_report_sha256']}.items():assert sha(_repo_path(ROOT, n))==h,n
 for arm,folder in [('S','dinov2-official'),('B','dinov2-base-062')]:
  filename=f'dinov2_vit{arm.lower()}14_pretrain.pth';path=_repo_path(ROOT, 'vendor', folder, filename)
  state=torch.load(_repo_io(path),map_location='cpu',weights_only=True);w=state['patch_embed.proj.weight'].numpy().astype(float)
  std=np.array([.229,.224,.225],dtype=np.float32).astype(float);K=np.zeros((w.shape[0],196))
  for c in range(3):K+=w[:,c].reshape(w.shape[0],196)/std[c]
  eig=np.linalg.eigvalsh(K.T@K);assert eig.min()>0
  expected=np.array([r['patch_projection'][arm]['smallest_singular'],r['patch_projection'][arm]['largest_singular']]);err=float(abs(np.sqrt(eig[[0,-1]])-expected).max());assert err<1e-9
  q=np.linspace(0,1,196);sol=np.linalg.lstsq(K,K@q,rcond=None)[0];recovery=float(abs(q-sol).max());assert recovery<1e-10
  check[arm]={'gram_positive_eigenvalues':int((eig>0).sum()),'gram_vs_SVD_extreme_singular_max_error':err,'linear_patch_fixture_recovery_max_error':recovery,'weight_sha256':sha(path)}
 yy,xx=np.indices((256,256));fixtures=[((xx+yy)%2*255).astype(np.uint8),(xx%2*255).astype(np.uint8)]
 down=[np.floor(x.reshape(128,2,128,2).mean((1,3))+.5).astype(np.uint8) for x in fixtures];assert np.array_equal(*down);assert np.unique(down[0]).tolist()==r['box_downsampling']['output_unique_values']
 # Direct GroupNorm formula, no call to its implementation.
 z=np.arange(256,dtype=float).reshape(1,4,8,8)/100;alt=z.copy();alt[0,0,-1,-1]+=10
 def norm(x):
  g=x.reshape(1,2,-1);return ((g-g.mean(-1,keepdims=True))/np.sqrt(g.var(-1,keepdims=True)+1e-5)).reshape(x.shape)
 effect=float(abs(norm(z)[0,0,0,0]-norm(alt)[0,0,0,0]));assert abs(effect-r['CNN_support']['GN_distant_fixture_effect'])<1e-12
 t=np.array([0.,2.,4.,6.]);u=t[[0,2,1,3]]
 assert np.array_equal(np.sort(t),np.sort(u));assert float(np.mean(np.diff(t)**2))==r['token_moments']['adjacent_squared_difference_original'];assert float(np.mean(np.diff(u)**2))==r['token_moments']['adjacent_squared_difference_permuted']
 packages={d.metadata['Name'].lower().replace('_','-'):d.version for d in importlib.metadata.distributions()};assert packages==p['packages']
 result={'passed':True,'independent_patch_checks':check,'BOX_manual_two_by_two_average_exact':True,'GN_manual_formula_effect_error':abs(effect-r['CNN_support']['GN_distant_fixture_effect']),'token_moment_collision_checked':True,'protected20_unchanged':len(p['protected_sha256'])==20,'previous200_reports_unchanged':len(p['existing_report_sha256'])==200,'packages_unchanged':len(packages),'source_sha256':sha(Path(__file__)),'numerical_report_sha256':sha(_repo_path(OUT, 'numerical-audit.json')),'scope':'No real images, labels, data caches, model fitting or predictions.'}
 with (_repo_path(OUT, 'validation.json')).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
