"""Independent CPU checks of projection, permutation and local receptive support."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import spatial_token_cnn_161 as q
import numpy as np,torch
from torch.nn import functional as F
torch.set_num_threads(2);q.check();out=q.OUT;cr=q.read(out/'cache.json');assert q.sha(out/'projected.npy')==cr['cache_sha256'];cache=np.load(_repo_io(out/'projected.npy'),mmap_mode='r')
with np.load(_repo_io(out/'feature-meta.npz'),allow_pickle=False) as z:P=z['projection'];perm=z['permutation'];ids=z['ids'].tolist()
source=_repo_path(ROOT, 'artifacts/patch-attention-092/patch_tokens.npy');sr=q.read(_repo_path(ROOT, 'artifacts/patch-attention-092/cache.json'));assert q.sha(source)==sr['token_sha256'];tokens=np.load(_repo_io(source),mmap_mode='r');assert ids==sr['ids'];inverse=np.argsort(perm);errors=[];momerr=[]
for i in [0,199,399]:
 manual=F.linear(torch.from_numpy(np.array(tokens[i],dtype=np.float64)),torch.from_numpy(P.T.copy())).numpy().reshape(36,36,32).transpose(2,0,1);np.testing.assert_array_equal(manual.astype(np.float32),cache[i]);errors.append(float(abs(manual-np.asarray(cache[i],dtype=float)).max()));A=np.asarray(cache[i],dtype=float).reshape(32,1296);B=A[:,perm];np.testing.assert_array_equal(B[:,inverse],A);momerr.append(max(float(abs(A.mean(1)-B.mean(1)).max()),float(abs(A.std(1)-B.std(1)).max())))
 pp=perm.reshape(36,36);retained=0;total=0
 for aa,bb in [(pp[:-1],pp[1:]),(pp[:,:-1],pp[:,1:])]:retained+=int(((abs(aa//36-bb//36)+abs(aa%36-bb%36))==1).sum());total+=aa.size
# One saved spatial model, CPU only; inspect local graph before global pooling.
path=out/'original'/f'fold-0-spatial-seed-{q.SEEDS[0]}';rec=q.read(path/'report.json');assert q.sha(path/'model.pt')==rec['model_sha256'];obj=torch.load(_repo_io(path/'model.pt'),weights_only=True,map_location='cpu');state=obj['state'];torch.manual_seed(160);x=torch.randn(1,32,36,36,requires_grad=True)
h=F.gelu(F.conv2d(x,state['stem.weight'],state['stem.bias']));h=F.gelu(F.conv2d(h,state['conv1.weight'],state['conv1.bias'],padding=1));h=F.gelu(F.conv2d(h,state['conv2.weight'],state['conv2.bias'],padding=1));h[0,:,18,18].sum().backward();g=x.grad.abs().sum((0,1));outside=torch.ones((36,36),dtype=torch.bool);outside[16:21,16:21]=False;assert float(g[outside].max())==0 and torch.all(g[16:21,16:21]>0)
r={'passed':True,'projection_float32_cache_matches_independent_torch_float64_linear_exactly':True,'projection_float32_rounding_max_error':max(errors),'permutation_inverse_exact':True,'perimage_channel_mean_std_max_rounding_error':max(momerr),'same_patch_token_multiset':True,'control_true_neighbor_edges_retained':retained,'control_neighbor_edges_total':total,'control_true_neighbor_fraction':retained/total,'CNN_local_prepool_gradient_support_exactly5x5':True,'frozen_DINO_tokens_already_global_context':True,'no_new_fits_or_target_access':True,'source_sha256':q.sha(Path(__file__)),'cache_sha256':cr['cache_sha256']};q.write(out/'feature-audit.json',r);q.check();q.emit(r)
