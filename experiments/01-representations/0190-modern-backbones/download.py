from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,hashlib,time,datetime,urllib.request,gc,sys,subprocess,re,resource
import torch,torchvision
from torchvision.models import ConvNeXt_Tiny_Weights,Swin_V2_T_Weights,convnext_tiny,swin_v2_t
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3];assert Path(sys.prefix).resolve()==_repo_path(P, 'experiments/pretrained-cnn-185/.venv');torch.set_num_threads(2);torch.set_num_interop_threads(1)
plan=json.loads((_repo_path(R, 'plan.json')).read_text());allowed={q['url'] for q in plan['weights'].values()};records={}
def write(p,z):
 with p.open('x') as f:json.dump(z,f,indent=2)
def swap():
 q=subprocess.check_output(['/usr/sbin/sysctl','vm.swapusage'],text=True);v,u=re.search(r'used = ([0-9.]+)([MG])',q).groups();return int(float(v)*(1024**2 if u=='M' else 1024**3))
initial_swap=swap()
for name,w,build in [('convnext_tiny',ConvNeXt_Tiny_Weights.IMAGENET1K_V1,convnext_tiny),('swin_v2_t',Swin_V2_T_Weights.IMAGENET1K_V1,swin_v2_t)]:
 url=w.url;assert url==plan['weights'][name]['url'] and url in allowed;dest=_repo_path(R, 'weights', url.rsplit('/',1)[1]);partial=dest.with_suffix('.pth.partial');assert not dest.exists();start=time.monotonic();h=hashlib.sha256();count=0;milestone=0
 req=urllib.request.Request(url,headers={'User-Agent':'approved-private-local-research/190'})
 with urllib.request.urlopen(req,timeout=90) as response,partial.open('xb') as out:
  assert response.status==200 and response.geturl()==url;length=response.headers.get('Content-Length')
  if length:assert int(length)<=120*1024**2
  while block:=response.read(1048576):
   count+=len(block);assert count<=120*1024**2;h.update(block);out.write(block)
   if count//(32*1024**2)>milestone:milestone=count//(32*1024**2);print(json.dumps({'downloading':name,'bytes':count}),flush=True)
  if length:assert count==int(length)
 digest=h.hexdigest();assert digest.startswith(plan['weights'][name]['hash_prefix']);partial.rename(dest)
 state=torch.load(_repo_io(dest),map_location='cpu',weights_only=True);model=build(weights=None);model.load_state_dict(state,strict=True);assert sum(p.numel() for p in model.parameters())==w.meta['num_params'];transform=w.transforms();assert transform.mean==[.485,.456,.406] and transform.std==[.229,.224,.225];assert max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,0)<6000000000 and swap()<=initial_swap
 record={'url':url,'path':str(dest.relative_to(P)),'enum':plan['weights'][name]['enum'],'bytes':count,'MiB':count/1024**2,'SHA256':digest,'official_filename_hash_prefix_verified':True,'strict_state_dict_verified':True,'tensor_entries':len(state),'parameters':w.meta['num_params'],'official_transform':str(transform),'mean':transform.mean,'std':transform.std,'metadata':{k:v for k,v in w.meta.items() if k!='categories'},'seconds_download_and_validation':time.monotonic()-start,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()};write(_repo_path(R, f'{name}-weight.json'),record);records[name]=record;print(json.dumps(record),flush=True);del model,state;gc.collect()
license=_repo_path(R, 'licenses/torchvision-BSD-3-Clause.txt');license_hash=hashlib.sha256(license.read_bytes()).hexdigest();assert license_hash=='6502f676851cfe25f8af75531dfb32375b7325b73c37e7b43741fa422893e71d'
provenance={'study':190,'weights':records,'official_docs':{'general':'https://docs.pytorch.org/vision/stable/models.html','convnext_tiny':'https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html','swin_v2_t':'https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.swin_v2_t.html'},'license':{'code':'TorchvisionBSD-3-Clause:retaincopyright/license/disclaimeronredistribution;binarynoticeandnonendorsementconditions;existingv0.29.0licensecopiedwithverifiedhash.','path':str(license.relative_to(P)),'SHA256':license_hash,'source':'https://raw.githubusercontent.com/pytorch/vision/v0.29.0/LICENSE','weights':'Officialdocumentationnotespretrainedweightsmayhaveowntermsderivedfromtrainingdatasets. CodeBSDlicenseisnotassertedasacomprehensiveweight/datasetlicense.','use_scope':'User-authorizedprivatelocalcomparison; nodistribution,login,paidservice,ornewtermsacceptanceperformed.'},'torch':torch.__version__,'torchvision':torchvision.__version__,'existing_environment':'experiments/pretrained-cnn-185/.venv','new_packages':[],'only_two_authorized_weight_downloads':True,'total_download_bytes':sum(x['bytes'] for x in records.values()),'peak_RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'swap_growth':swap()-initial_swap,'cost':0};write(_repo_path(R, 'provenance.json'),provenance);print(json.dumps({'download_complete':True,'total_bytes':provenance['total_download_bytes'],'peak_RSS':provenance['peak_RSS'],'swap_growth':provenance['swap_growth'],'cost':0}),flush=True)
