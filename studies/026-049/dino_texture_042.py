"""Frozen DINO patch spatial variograms; one predeclared texture panel."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,time,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import dino_spatial_039 as ds
OUT=_repo_path(ROOT, 'artifacts/dino-texture-042')
LAGS=(1,4)
def variograms(p):
 assert p.ndim==4 #batch,height,width,channel
 var=p.var((1,2),correction=0);out=[]
 for lag in LAGS:
  v=(p[:,lag:]-p[:,:-lag]).square().mean((1,2));h=(p[:,:,lag:]-p[:,:,:-lag]).square().mean((1,2))
  out.append((h+v)/(4*(var+1e-4)))
 return torch.cat(out,1)
def features(model,x):
 with torch.inference_mode():
  f=model.forward_features(x);p=f['x_norm_patchtokens'];assert p.shape[1:]==(1296,384)
  global_=torch.cat((f['x_norm_clstoken'],p.mean(1),p.std(1,correction=0)),1);texture=variograms(p.reshape(len(p),36,36,384))
  a=torch.cat((global_,texture),1)
 assert a.shape==(len(x),1920) and torch.isfinite(a).all();return a

def smoke():
 OUT.mkdir(exist_ok=False);torch.set_num_threads(2);assert torch.backends.mps.is_available()
 zero=torch.zeros(2,36,36,3);assert torch.equal(variograms(zero),torch.zeros(2,6))
 yy,xx=torch.meshgrid(torch.arange(36),torch.arange(36),indexing='ij');checker=((xx+yy)%2).float()[None,:,:,None];stripes=(xx%2).float()[None,:,:,None]
 assert torch.equal(checker.mean((1,2)),stripes.mean((1,2))) and torch.equal(checker.std((1,2)),stripes.std((1,2)))
 assert not torch.allclose(variograms(checker),variograms(stripes))
 torch.manual_seed(20261125);z=torch.randn(2,36,36,3);v=variograms(z);assert torch.isfinite(v).all();torch.testing.assert_close(v,variograms(z+7),atol=1e-5,rtol=1e-5)
 for k in range(4):
  torch.testing.assert_close(v,variograms(torch.rot90(z,k,(1,2))),atol=1e-6,rtol=1e-6)
  torch.testing.assert_close(v,variograms(torch.rot90(z,k,(1,2)).flip(2)),atol=1e-6,rtol=1e-6)
 ids,sp=ds.base.scope();ident=next(i for i in ids if sp['development_folds'][i]!=0);x=torch.from_numpy(ds.inputs(ident,'full504'));model=ds.base.model();outputs={};timing={}
 for device in ('cpu','mps'):
  model=model.to(device);xx=x.to(device);times=[]
  for j in range(3):
   if device=='mps':torch.mps.synchronize()
   t=time.perf_counter();a=features(model,xx)
   if device=='mps':torch.mps.synchronize()
   if j:times.append(time.perf_counter()-t)
  outputs[device]=a.cpu().numpy();timing[device]={'median_image_seconds':float(np.median(times)),'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
  if device=='mps':timing[device]['driver_bytes']=torch.mps.driver_allocated_memory()
 np.testing.assert_allclose(outputs['cpu'],outputs['mps'],atol=2e-3,rtol=2e-3)
 protected={str(p.relative_to(ROOT)):ds.base.sha(p) for p in [_repo_path(ROOT, 'artifacts/submission-candidate-005.csv'),_repo_path(ROOT, 'artifacts/submission-candidate-005.manifest.json'),_repo_path(ROOT, 'artifacts/submission-candidate-003.csv'),_repo_path(ROOT, 'artifacts/dino-spatial-039/candidate-design.json'),*(_repo_path(ROOT, 'artifacts/final-model-041')).glob('*')] if p.is_file()}
 plan={'hypothesis':'005globalCLS/mean/std discards where patchfeatures change. Per-channel normalized semivariograms atlags1,4 encode local-boundary coherence/scale without learning newbackboneparameters.','descriptor':'Existing504global1152 + normalizedhorizontal/verticalsquaredpatchdifferences atlags1,4 (2x384) =1920. Averageaxes preservesalignmentmagnitudebutdoesnotaddabsoluteorientationfeatures.','normalization':'(horizontalMSE+verticalMSE)/(4*(perchannelpatchvariance+1e-4)); globalpatchstd retained separately.','physical_scale':'504upsamplingof256means14pixelmodelpatch≈7.1nativepixels;lags1/4≈7.1/28.4nativepixels. No physicalunitsavailable.','architecture':'SameofficialfrozenoriginalDINOv2;finalnormalizedpatchtokens36x36;no finetuning ornewweights','anti_overfit':'Onefeaturepanelonly; StandardScaler+Ridgealpha=lambda*p;lambda(.1,1,10) selectedinner3foldonly; no extraoptimizer/epochs,ensembleweighttuning ordataaugmentation','comparison':'005reference=.75samefold003+.25matchedglobal504Ridge;new=.75samefold003+.25newtextureRidge. Sameoriginal+bothpredefinedrepartitions;no optionalgridexpansion.','inner':'GroupKFold3shuffle seed20261120+outerfold,as039;allpreprocessingfitinner/outertrainonly','scope':'Development400only, noholdout/test/Publicmodelselection. Syntheticunitfixturesonlytestdescriptorandnevertrainregressors.','rotation_note':'Descriptor aggregation invariantforrotation/reflectionofalreadycomputedtokenmaps; DINOinput orientation invariance isnotclaimedorassumed.','limits':'Reused400andpartitionscarryhistoricalselectionbias;these arenotindependentvalidation.','source_sha256':{n:ds.base.sha(_repo_path(ROOT, n)) for n in ('dino_texture_042.py','dino_spatial_039.py','dino_frozen_037.py')},'protected':protected}
 ds.base.write(_repo_path(OUT, 'plan.json'),plan);report={'device_timings':timing,'cpu_mps_max_abs_feature_diff':float(abs(outputs['cpu']-outputs['mps']).max()),'checks':{'zero_finite':True,'sameglobalstats_differentlocaltexture_distinguished':True,'token_map_D4_invariance':True,'translation_invariance':True},'fixture_scope':'Algorithmunitchecks only,not trainingdata','smoke_training_id':ident};ds.base.write(_repo_path(OUT, 'smoke.json'),report);print(json.dumps(report),flush=True)

def extract():
 plan=json.loads((_repo_path(OUT, 'plan.json')).read_text());smoke=json.loads((_repo_path(OUT, 'smoke.json')).read_text());d=smoke['device_timings']['mps'];assert d['median_image_seconds']<1 and d['process_peak_rss_bytes']<3_000_000_000 and d['driver_bytes']<3_000_000_000
 for n,h in plan['source_sha256'].items():assert ds.base.sha(_repo_path(ROOT, n))==h
 official=json.loads((_repo_path(ROOT, 'artifacts/dino-frozen-037/smoke.json')).read_text())
 for n,h in official['official_source_sha256'].items():assert ds.base.sha(ds.base.SOURCE/n)==h
 torch.set_num_threads(2);ids,_=ds.base.scope();model=ds.base.model().to('mps');before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};values=[];start=time.perf_counter()
 for j,ident in enumerate(ids):
  x=torch.from_numpy(ds.inputs(ident,'full504')).to('mps');values.append(features(model,x).cpu().numpy()[0])
  if (j+1)%80==0:print(json.dumps({'completed':j+1,'seconds':time.perf_counter()-start}),flush=True)
 X=np.stack(values);assert X.shape==(400,1920) and np.isfinite(X).all();assert all(torch.equal(before[k],v.detach().cpu()) for k,v in model.state_dict().items()) and all(p.grad is None for p in model.parameters())
 with np.load(_repo_path(ROOT, 'artifacts/dino-spatial-039/development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;np.testing.assert_allclose(X[:,:1152],z['full504'],rtol=0,atol=2e-4);delta=float(abs(X[:,:1152]-z['full504']).max())
 with (_repo_path(OUT, 'development_features.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),X=X)
 for n,h in plan['protected'].items():assert ds.base.sha(_repo_path(ROOT, n))==h
 report={'seconds':time.perf_counter()-start,'weights_unchanged':True,'shape':list(X.shape),'original504_descriptor_max_difference':delta,'feature_sha256':ds.base.sha(_repo_path(OUT, 'development_features.npz')),'image_sha256':{i:ds.base.sha(_repo_path(ROOT, 'data/train', f'{i}.png')) for i in ids},'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};ds.base.write(_repo_path(OUT, 'extraction.json'),report);print(json.dumps(report|{'image_sha256':'saved in artifact'}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('smoke','extract'));a=p.parse_args();smoke() if a.mode=='smoke' else extract()
