"""Architecture-only local benchmark. No pretrained download or data access."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,time,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import numpy as np,torch
import dino_frozen_037 as base
import frozen_validation as fv
from dinov2.hub.backbones import dinov2_vitb14
# Make accidental Torch Hub downloads fail before any request can be issued.
def forbid(*args,**kwargs):raise RuntimeError('No downloads authorized in this preflight')
torch.hub.load_state_dict_from_url=forbid
import urllib.request
urllib.request.urlopen=forbid
urllib.request.urlretrieve=forbid
torch.set_num_threads(2);torch.manual_seed(20270102);t=time.perf_counter();m=dinov2_vitb14(pretrained=False).requires_grad_(False).eval().to('mps');x=torch.zeros((1,3,504,504),device='mps');times=[]
with torch.inference_mode():
 for j in range(4):
  torch.mps.synchronize();start=time.perf_counter();out=m.get_intermediate_layers(x,n=[5,8,11],return_class_token=True,norm=True);features=torch.cat([torch.cat((c,p.mean(1),p.std(1,correction=0)),1) for p,c in out],1);assert features.shape==(1,6912) and torch.isfinite(features).all();torch.mps.synchronize()
  if j:times.append(time.perf_counter()-start)
num=sum(p.numel() for p in m.parameters());card=(base.SOURCE/'MODEL_CARD.md').read_text();license_lines=[line for line in card.splitlines() if 'license' in line.lower() or 'apache' in line.lower()]
report={'stage':'Random-initialized architecture resource-only preflight,nottrainedperformance','model':'OriginalnonregisterDINOv2ViT-B14','pretrained':False,'images_or_labels_read':False,'network_downloads':False,'packages_installed':False,'cost':0,'parameters':num,'parameter_float32_bytes':num*4,'three_layer_descriptor_width':6912,'batch':1,'resolution':504,'median_image_seconds':float(np.median(times)),'projected400_forward_seconds':float(np.median(times)*400),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory(),'total_seconds':time.perf_counter()-t,'license_lines_from_pinned_model_card':license_lines,'official_source_dir':str(base.SOURCE.relative_to(ROOT)),'source_hash':fv.sha(Path(__file__)),'limitations':'Random weights andzeroimage only establish execution compatibility/costestimate,notqualityoractualpretrainedruntime. Exactdownloadfilebytes unverified. No model orpredictioncandidate saved.','next_action_requiring_authorization':'Download officialoriginaldinov2_vitb14_pretrain.pth into projectvendor/cache andvalidate provenance/hash/license, thenfrozendevelopment400comparison. Prior explicitinstallation/downloadapproval namedViT-S14; largernewcheckpoint not assumedauthorized.'};fv.write(_repo_path(ROOT, 'reports/dinob-062-resource-preflight.json'),report);print(json.dumps(report),flush=True)
