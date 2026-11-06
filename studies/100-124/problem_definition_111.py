"""Label-free numerical audit of representation claims; no training or real images."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os, sys, json, hashlib, time, importlib.metadata
from pathlib import Path
from datetime import datetime, timezone
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
sys.path.insert(0,str(ROOT))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[k]='2'
import numpy as np
import torch
from PIL import Image
import dinob_062 as b
OUT=_repo_path(ROOT, 'artifacts/problem-definition-111')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for v in iter(lambda:f.read(2**20),b''):h.update(v)
 return h.hexdigest()
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def packages():return {d.metadata['Name'].lower().replace('_','-'):d.version for d in importlib.metadata.distributions()}
def main():
 start=time.perf_counter();torch.set_num_threads(2);OUT.mkdir(exist_ok=False)
 protected=json.loads((_repo_path(ROOT, 'reports/candidate011-public-submission.json')).read_text())['protected_sha256']
 source_names=['problem_definition_111.py','candidate_pipeline.py','candidate008_pipeline.py','candidate009_pipeline.py','candidate011_pipeline.py','cnn_experiment.py','grain_features.py','compact_regression.py','latent_features.py','native_texture_features.py','denoised_grain_features.py','connected_grain_features.py','dino_frozen_037.py','dino_spatial_039.py','dinob_062.py','dino_texture_042.py','dino_multilayer_044.py','src/microstructure/features.py','src/microstructure_v2/features.py','src/microstructure_v2/extra.py']
 sources={n:sha(_repo_path(ROOT, n)) for n in source_names}
 evidence={str(p.relative_to(ROOT)):sha(p) for p in sorted((_repo_path(ROOT, 'reports')).rglob('*.json'))}
 for n,h in protected.items():assert sha(_repo_path(ROOT, n))==h,n
 environment=json.loads((_repo_path(ROOT, 'artifacts/wavelet-scattering-096/environment-before.json')).read_text());expected={**environment['packages'],**environment['allowed_additions']};assert packages()==expected
 write(_repo_path(OUT, 'plan.json'),{'utc':datetime.now(timezone.utc).isoformat(),'scope':'No real image/label/cache decoding, training, test inference, package installation or upload. Pure fixtures and existing local weights; synthetic arrays are algorithm checks only, never training data.','checks':['Effective grayscale learned patch projection rank and direct convolution agreement','Exact BOX128 high-frequency collision','Token moment permutation collision, with contextual-token caveat','Elongation-weighted alignment semantics','Convolutional support versus GroupNorm global dependence'],'source_sha256':sources,'existing_report_sha256':evidence,'protected_sha256':protected,'packages':packages()})
 yy,xx=np.indices((14,14));a=((xx+yy)%2).astype(float);c=(xx%2).astype(float)
 assert a.mean()==c.mean() and a.std()==c.std()
 patch={}
 for name,maker in [('S',b.ds.base.model),('B',b.model)]:
  m=maker();w=m.patch_embed.proj.weight.detach().double().cpu().numpy();bias=m.patch_embed.proj.bias.detach().double().cpu().numpy();std=b.ds.base.STD.astype(float);mean=b.ds.base.MEAN.astype(float)
  K=(w/std[None]).sum(1).reshape(w.shape[0],-1)
  singular=np.linalg.svd(K,compute_uv=False)
  ranks={}
  for dtype in [np.float32,np.float64]:
   tol=max(K.shape)*np.finfo(dtype).eps*singular[0];ranks[np.dtype(dtype).name]={'rank':int((singular>tol).sum()),'tolerance':float(tol)}
  inp=(np.repeat(np.stack([a,c])[:,None],3,axis=1)-mean)/std
  actual=torch.nn.functional.conv2d(torch.from_numpy(inp),torch.from_numpy(w),torch.from_numpy(bias),stride=14).numpy()[:,:,0,0]
  offset=bias-(w*mean[None]/std[None]).sum((1,2,3));manual=np.stack([K@a.ravel()+offset,K@c.ravel()+offset])
  err=float(abs(actual-manual).max());assert err<1e-10
  difference=float(np.linalg.norm(actual[0]-actual[1]));assert difference>1e-6
  patch[name]={'shape':list(K.shape),'rank_tests':ranks,'largest_singular':float(singular[0]),'smallest_singular':float(singular[-1]),'condition':float(singular[0]/singular[-1]),'equal_pixel_mean_std_fixture_embedding_L2_difference':difference,'manual_vs_conv_max_error':err,'patch_norm':type(m.patch_embed.norm).__name__,'meaning':'Learned pre-transformer linear patch map, using stored float32 normalization constants cast to float64. Full column rank distinguishes all real-valued 14x14 grayscale patches in exact arithmetic at this stage; not full network invertibility or predictive sufficiency.'}
  del m
 yy,xx=np.indices((256,256));checker=((xx+yy)%2*255).astype(np.uint8);stripes=(xx%2*255).astype(np.uint8)
 ca=np.asarray(Image.fromarray(checker).resize((128,128),Image.Resampling.BOX));sa=np.asarray(Image.fromarray(stripes).resize((128,128),Image.Resampling.BOX));assert np.array_equal(ca,sa)
 box={'input_unequal_pixels':int((checker!=stripes).sum()),'output_max_difference':int(abs(ca.astype(int)-sa.astype(int)).max()),'output_unique_values':np.unique(ca).tolist(),'meaning':'Exact collision for two binary patterns under current BOX256to128 downsampling. Does not prove grain-size loss across all branches; DINO and native descriptors receive other representations.'}
 t=np.array([[0.,1.],[2.,3.],[4.,5.],[6.,7.]])
 p=t[[0,2,1,3]];assert np.array_equal(t.mean(0),p.mean(0)) and np.array_equal(t.std(0),p.std(0))
 moment={'mean_equal':True,'std_equal':True,'adjacent_squared_difference_original':float(np.diff(t,axis=0).var()+np.mean(np.diff(t,axis=0))**2),'adjacent_squared_difference_permuted':float(np.mean(np.diff(p,axis=0)**2)),'meaning':'Abstract tokens only: moment pooling forgets token order. DINO tokens already encode position and global context; input patch permutations need not yield token permutations, and CLS is also retained.'}
 areas=np.array([1.,2.,3.]);angles=np.zeros(3);major=np.ones(3);minor=major.copy()
 old=float(abs(np.sum(areas*np.exp(2j*angles)))/areas.sum());new=float(abs(np.sum(areas*(major-minor)/(major+minor)*np.exp(2j*angles)))/areas.sum());assert old==1 and new==0
 angles=np.array([.1,.2,.8]);elong=np.array([.2,.4,.7]);q=lambda ang:np.sum(areas*elong*np.exp(2j*ang))/areas.sum();rotation_error=float(abs(abs(q(angles))-abs(q(angles+.731))));assert rotation_error<1e-14
 alignment={'circular_region_arbitrary_equal_angle_unweighted_order':old,'elongation_weighted_order':new,'rotation_of_angles_magnitude_error':rotation_error,'current_compact26_source':'grain_features.py:43 / compact_regression.py:44 uses area-only weighting','already_tested_weighted_sources':['denoised_grain_features.py:78','connected_grain_features.py:123'],'meaning':'No useful orientation for circles; weighted order encodes both shape anisotropy and coherence, not pure angular concentration. Absolute orientation rotates while order magnitude stays constant; no claim target invariant.'}
 r,j=1,1
 for k in [5,3,3,3]:r+=(k-1)*j;j*=2
 assert (r,j)==(33,16)
 z=torch.arange(4*8*8,dtype=torch.float64).reshape(1,4,8,8)/100
 alt=z.clone();alt[0,0,-1,-1]+=10
 gn=torch.nn.GroupNorm(2,4,affine=False).double();delta=float(abs(gn(z)[0,0,0,0]-gn(alt)[0,0,0,0]));assert delta>1e-6
 support={'conv_only_receptive_width_at_input_resolution':r,'stride_at_input_resolution':j,'current128_map_side':8,'nominal_native_width_excluding_norm':66,'native_stride':32,'GN_distant_fixture_effect':delta,'qualification':'33 input pixels is convolution-only support. GroupNorm aggregates across spatial positions per sample/group, coupling distant locations even before global pooling; full effective dependence is not restricted to this width. BOX support yields nominal 66 native pixels excluding global normalization.'}
 result={'utc':datetime.now(timezone.utc).isoformat(),'patch_projection':patch,'box_downsampling':box,'token_moments':moment,'alignment':alignment,'CNN_support':support,'DINO_scale':{'native_image_side':256,'resized_side':504,'common_scale':504/256,'token_grid_side':36,'patch_stride_native_equivalent':14*256/504,'meaning':'Same fixed scale across all images preserves relative pixel-scale differences, not per-image size normalization. Bicubic interpolation adds no independent detail; physical length calibration unknown.'},'source_sha256':sources,'seconds':time.perf_counter()-start}
 for n,h in {**protected,**sources,**evidence}.items():assert sha(_repo_path(ROOT, n))==h,n
 assert packages()==expected
 result['preservation']={'protected_files':len(protected),'previous_reports':len(evidence),'packages':len(expected),'unchanged':True};write(_repo_path(OUT, 'numerical-audit.json'),result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
