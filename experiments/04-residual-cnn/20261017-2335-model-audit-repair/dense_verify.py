"""Operator/control checks and one complete training replay, without new selection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import cnn_repair as cr
import dense_mps as dm
import sys,json,time,gc
import numpy as np,torch
from torch import nn
from threadpoolctl import threadpool_limits
def main():
 start=time.monotonic();torch.manual_seed(2026101804);a=cr.DenseFirst('pick');torch.manual_seed(2026101804);b=cr.DenseFirst('pool');assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict());x=torch.randn(2,1,256,256);conv=a.blocks[0][0];dense=conv(x);strided=nn.functional.conv2d(x,conv.weight,conv.bias,stride=4,padding=3);ce=float(abs(strided-dense[:,:,::4,::4]).max());assert ce<1e-5;h=a.blocks[0](x);pool=nn.functional.avg_pool2d(h,4,4);me=float(abs(pool.mean((2,3))-h.mean((2,3))).max());assert me<1e-6;assert torch.all(pool.var((2,3),correction=0)<=h.var((2,3),correction=0)+1e-6)
 shapes=[]
 for m in [a,b]:
  with torch.no_grad():z,aux=m(x,torch.zeros(2,6),'aux')
  assert z.shape==aux.shape==(2,);shapes.append(sum(p.numel() for p in m.parameters()))
 assert shapes[0]==shapes[1]==68788;assert not any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in a.modules());del a,b,x,h,pool,dense,strided;gc.collect()
 d,y=cr.load();z=cr.data(d,y,0,'selector','073');outside=np.setdiff1d(np.arange(500),z['train']);yy=y.copy();yy[outside]=999999.;dd={**d,'images':d['images'].copy()};dd['images'][outside]=.123;poison=cr.data(dd,yy,0,'selector','073');assert z['meta']==poison['meta'];assert all(np.array_equal(z['targets'][k],poison['targets'][k]) for k in z['targets']);assert np.array_equal(z['images'][z['train']],poison['images'][poison['train']]);assert np.array_equal(z['context'],poison['context']);path=c.HERE/'verification/dense-poison-repeat';seconds=dm.fit(poison,'pool',2026101501,path);oldpath=c.HERE/'cnn-dense-pool/seed0/fold0-selector';old=torch.load(_repo_io(oldpath/'model.pt'),map_location='cpu',weights_only=True)['state_dict'];new=torch.load(_repo_io(path/'model.pt'),map_location='cpu',weights_only=True)['state_dict'];wd=max(float(abs(new[k]-old[k]).max()) for k in old);model=cr.DenseFirst('pool').to('mps');model.load_state_dict(new);im=torch.from_numpy(z['images']).to('mps');ctx=torch.from_numpy(z['context']).to('mps');p=dm.infer(model,im,ctx,z['valid'],z['meta']['target_std']);ref=np.load(_repo_io(oldpath/'predictions.npz'))['validation_prediction'];pd=float(abs(p-ref).max());del model,im,ctx;gc.collect();torch.mps.empty_cache()
 result=dict(operator_checks_passed=True,strided_conv_vs_dense_subsample_maxdiff=ce,dense_average_preserves_first_moment_maxdiff=me,pool_variance_not_greater_than_dense=True,identical_initial_weights=True,matched_parameter_count=shapes[0],batchnorm_layers=0,poisoned_outside_sources=len(outside),full40epoch_replay=True,full_training_weight_maxdiff=wd,full_training_original_query_maxdiff=pd,full_training_bit_exact=wd==0 and pd==0,training_replay_tolerance_HV=.001,within_tolerance=pd<.001,seconds=time.monotonic()-start,fit_seconds=seconds,scope='Same seeded40epoch fit with all outside-A labels and pixels changed. Frozen-weight inference is separately checked for all40models. This is not a whole incumbent-training reproduction.')
 if not result['full_training_bit_exact']:
  # Distinguish device repeatability from an excluded-input dependency.
  cleanpath=c.HERE/'verification/dense-original-repeat';cleansec=dm.fit(z,'pool',2026101501,cleanpath);clean=torch.load(_repo_io(cleanpath/'model.pt'),map_location='cpu',weights_only=True)['state_dict'];result['unchanged_input_repeat_weight_maxdiff']=max(float(abs(clean[k]-old[k]).max()) for k in old);result['clean_vs_poison_weight_maxdiff']=max(float(abs(clean[k]-new[k]).max()) for k in old);result['additional_clean_repeat_fit_seconds']=cleansec;result['seconds']=time.monotonic()-start
 c.save(_repo_io('dense-training-verification.json'),result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.27);sys.addaudithook(c.guard)
 with threadpool_limits(limits=2):main()
