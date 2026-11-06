"""Verify package integrity, raw two-image extraction and guarded1000-row inference."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse,ast,json,os,sys,time,importlib.metadata
from pathlib import Path
import infer as inf
ROOT=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--features',type=Path);ap.add_argument('--raw-data',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--project-boundary',type=Path);args=ap.parse_args();start=time.perf_counter();out=args.output.resolve();out.mkdir(exist_ok=False)
 manifest=inf.read(_repo_path(ROOT, 'MANIFEST.json'))
 for name,h in manifest['files'].items():assert inf.sha(_repo_path(ROOT, name))==h,name
 for p in ROOT.glob('*.py'):ast.parse(p.read_text())
 env=inf.read(_repo_path(ROOT, 'evidence/environment.json'));versions={}
 for key,v in env['packages'].items():
  actual=importlib.metadata.version(key);assert actual==v,(key,actual,v);versions[key]=actual
 provenance=inf.read(_repo_path(ROOT, 'licenses/current-model-provenance.json'))
 for w in provenance['weights']:assert inf.sha(_repo_path(ROOT, w['path']))==w['sha256']
 cert=inf.read(_repo_path(ROOT, 'models/training-validation.json'));assert cert['passed']
 inf.guards();blocked=[]
 for name,fn in [('scaler.fit',inf.StandardScaler().fit),('kernel.fit',inf.BalancedKernel().fit),('head.fit',inf.h.fit),('residual.fit',inf.n.fit),('backward',inf.torch.tensor(1.,requires_grad=True).backward)]:
  try:fn()
  except RuntimeError as e:assert 'forbidden' in str(e);blocked.append(name)
  else:raise AssertionError(name)
 touched=set();boundary=args.project_boundary.resolve() if args.project_boundary else None
 features=args.features.resolve() if args.features else None;data=args.raw_data.resolve() if args.raw_data else None
 explicit=set() if features is None else {features,features.with_name('test-feature-validation.json')}
 rawids=['TEST_000001','TEST_001000']
 if data is not None:explicit|={data/'sample_submission.csv',*[data/'test'/f'{i}.png' for i in rawids]}
 def audit(event,a):
  if event!='open' or not isinstance(a[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(a[0])).resolve()
  if boundary and p.is_relative_to(boundary) and not (p.is_relative_to(ROOT) or p.is_relative_to(out) or p.is_relative_to(boundary/'.venv') or p in explicit):raise RuntimeError('Isolated package attempted undeclared project file: '+str(p))
  if boundary and p.is_relative_to(boundary) and not p.is_relative_to(boundary/'.venv'):touched.add(str(p.relative_to(boundary)))
 sys.addaudithook(audit);raw={}
 if data is not None:
  assert features is not None
  import extract
  import numpy as np,torch
  expected=inf.read(_repo_path(ROOT, 'config/original-test-hashes.json'));assert inf.sha(data/'sample_submission.csv')==expected['CSV_sha256']
  with np.load(_repo_io(features),allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
  indices=[0,999]
  for j,i in zip(indices,rawids):
   K,X57,cnn,_,_,h,_=extract.one(data,'test',i);assert h==expected['image_sha256'][i]
   for name,v in [('K',K),('X57',X57),('images',cnn)]:np.testing.assert_array_equal(v,d[name][j])
  from dinov2.hub.backbones import dinov2_vits14,dinov2_vitb14
  for arm,factory in [('S',dinov2_vits14),('B',dinov2_vitb14)]:
   m=factory(pretrained=False);m.load_state_dict(torch.load(_repo_path(ROOT, 'weights', f'DINO-{arm}.pth'),map_location='cpu',weights_only=True));m.requires_grad_(False);m.eval();m=m.to('mps')
   for j,i in zip(indices,rawids):
    with torch.inference_mode():
     x=torch.from_numpy(extract.dino_input(data/'test'/f'{i}.png')[None]).to('mps');ll=m.get_intermediate_layers(x,n=[5,8,11],return_class_token=True,norm=True);v=torch.cat([torch.cat([c,p.mean(1),p.std(1,correction=0)],1) for p,c in ll],1).cpu().numpy()[0].astype(float);np.testing.assert_array_equal(v,d[arm][j])
     if arm=='B':np.testing.assert_array_equal(extract.spatial_descriptor(ll[-1][0].cpu().numpy()[0]),d['spatial'][j])
   del m;torch.mps.empty_cache()
  raw={'IDs':rawids,'all_six_banks_exact':True,'fit_guard_active':True}
 if features is not None:
  sys.argv=['infer.py','--features',str(features),'--output',str(out/'inference')];inf.main();v=inf.read(out/'inference/inference-validation.json');assert v['CSV_sha256']==manifest['expected_CSV_sha256']
 result={'passed':True,'package_manifest_sha256':inf.sha(_repo_path(ROOT, 'MANIFEST.json')),'package_files_verified':len(manifest['files']),'dependency_versions_exact':True,'fit_guards_tested':blocked,'raw_smoke':raw,'full1000_inference_from_explicit_fresh176_cache':features is not None,'historical_project_files_blocked':boundary is not None,'seconds':time.perf_counter()-start,'opened_project_files':sorted(touched)};inf.write(out/'selftest.json',result);print(json.dumps({k:v for k,v in result.items() if k!='opened_project_files'}),flush=True)
if __name__=='__main__':main()
