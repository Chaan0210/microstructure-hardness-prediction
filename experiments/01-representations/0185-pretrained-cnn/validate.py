"""Independent saved-model reconstruction and NumPy metric/statistic checks."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
from common import R,P,np,torch,nn
import argparse,gc,time

def validate(name,split):
 c.check();root=R/name/split;d=c.data(split);X=c.load(R/name/'features.npz')['X'].astype(float);cache=np.load(_repo_io(R/name/'prefix.npy'),mmap_mode='r');extract=c.read(R/name/'extraction.json');assert c.sha(R/name/'prefix.npy')==extract['prefix_sha256'];assert c.sha(R/name/'features.npz')==extract['feature_sha256'];checks=[];maxerr=0.;started=time.monotonic()
 with c.Monitor(name+'-'+split+'-validation',900):
  prefix,originaltail=c.model(name);prefixhash=c.statehash(prefix);tailhash=c.statehash(originaltail);base=originaltail.state_dict();bnkeys=[]
  for module_name,module in originaltail.named_modules():
   if isinstance(module,nn.modules.batchnorm._BatchNorm):bnkeys.extend(module_name+'.'+k for k in module.state_dict())
  featurechecks=[]
  for i in c.plan['smoke_IDs']:
   j=c.plan['IDs'].index(i)
   with torch.inference_mode():a=prefix(c.image_input(i));b=originaltail(a).numpy()[0]
   err=float(abs(a.numpy()[0]-cache[j]).max());ferr=float(abs(b-X[j]).max());assert max(err,ferr)<.001;featurechecks.append({'ID':i,'CPU_prefix_maxabs':err,'CPU_feature_maxabs':ferr})
  predictions={arm:np.full(400,np.nan) for arm in ['frozen100','frozen112','finetune112']}
  for k in range(5):
   T=np.flatnonzero(d['fold']!=k);V=np.flatnonzero(d['fold']==k);objs={}
   for arm in predictions:
    path=root/f'fold-{k}';o=torch.load(_repo_io(path/f'{arm}.pt'),map_location='cpu',weights_only=True);objs[arm]=o;rec=c.read(path/f'{arm}-report.json');assert c.sha(path/f'{arm}.pt')==rec['checkpoint_sha256'];assert o['train']==T.tolist() and o['validation']==V.tolist();assert not set(d['g'][T])&set(d['g'][V]);np.testing.assert_array_equal(o['feature_mean'].numpy(),X[T].mean(0));np.testing.assert_array_equal(o['feature_var'].numpy(),X[T].var(0));scale=np.sqrt(X[T].var(0));scale[scale==0]=1.;np.testing.assert_array_equal(o['feature_scale'].numpy(),scale);assert o['target_mean']==float(d['y'][T].mean()) and o['target_std']==float(d['y'][T].std())
    h=nn.Sequential(nn.Linear(X.shape[1],64),nn.GELU(),nn.Dropout(.25),nn.Linear(64,1));h.load_state_dict(o['head']);h.eval();out=[]
    if arm=='finetune112':
     _,tail=c.model(name);tail.load_state_dict(o['tail']);tail.eval();assert o['prefix_initial_sha256']==prefixhash and o['tail_initial_sha256']==tailhash and c.statehash(tail)!=tailhash
     for key in bnkeys:assert torch.equal(o['tail'][key],base[key]),key
    else:tail=None
    with torch.inference_mode():
     for j in range(0,len(V),2):
      I=V[j:j+2];a=tail(torch.tensor(np.array(cache[I]))) if tail is not None else torch.tensor(X[I],dtype=torch.float32);a=(a-o['feature_mean'].float())/o['feature_scale'].float();out.extend(h(a).flatten().numpy().astype(float)*o['target_std']+o['target_mean'])
    pp=np.array(out);saved=c.load(path/f'{arm}-prediction.npz');np.testing.assert_array_equal(saved['indices'],V);np.testing.assert_array_equal(saved['ids'],d['ids'][V]);error=float(abs(pp-saved['prediction']).max());assert error<.001,(name,k,arm,error);maxerr=max(maxerr,error);predictions[arm][V]=saved['prediction'];checks.append({'fold':k,'arm':arm,'CPU_reload_maxabs':error,'trainlocal_stats_exact':True,'groups_disjoint':True});del tail,h;gc.collect()
   assert objs['frozen112']['head_initial_for_adaptation_sha256']==objs['finetune112']['head_initial_for_adaptation_sha256']==objs['frozen100']['final_head_sha256'];assert objs['frozen112']['order_adaptation_sha256']==objs['finetune112']['order_adaptation_sha256'];assert objs['frozen112']['tail_final_sha256']==tailhash
   c.emit({'validation':name,'split':split,'fold':k,'max_CPU_error':maxerr})
  for arm,pred in predictions.items():
   o=c.load(root/f'{arm}-OOF.npz');rec=c.read(root/f'{arm}-report.json');assert c.sha(root/f'{arm}-OOF.npz')==rec['OOF_sha256'];np.testing.assert_array_equal(o['ids'],d['ids']);np.testing.assert_array_equal(o['y'],d['y']);np.testing.assert_array_equal(o['fold'],d['fold']);np.testing.assert_array_equal(o['prediction'],pred);np.testing.assert_array_equal(o['blend'],.9*d['blend_L_MLP']+.1*pred);assert np.isfinite(pred).all()
   for key,pp in [('best021',d['blend_L_MLP']),('standalone',pred),('blend10',o['blend'])]:assert abs(rec['RMSE'][key]-float(np.sqrt(np.mean((d['y']-pp)**2))))<1e-12
   fg=[float(np.sqrt(np.mean((d['y'][d['fold']==k]-d['blend_L_MLP'][d['fold']==k])**2))-np.sqrt(np.mean((d['y'][d['fold']==k]-o['blend'][d['fold']==k])**2))) for k in range(5)];np.testing.assert_array_equal(fg,rec['fold_blend_gains'])
  assert not c.network_attempts
 c.check();report={'passed':True,'model':name,'split':split,'checkpoints':15,'checks':checks,'raw_CPU_feature_checks':featurechecks,'max_CPU_reload_error':maxerr,'all_BN_affine_runningstats_unchanged':True,'prefix_unchanged':True,'matched_control_same_initialhead_orders':True,'trainlocal_stats_exact':True,'OOF_ID_finite_order_metric_checks':True,'network_attempts':0,'seconds':time.monotonic()-started};c.write(root/'validation.json',report);c.emit({k:v for k,v in report.items() if k not in ['checks','raw_CPU_feature_checks']})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('model',choices=['resnet50','efficientnet_b0']);p.add_argument('split',choices=['original','20261031','20261101']);a=p.parse_args();validate(a.model,a.split)
