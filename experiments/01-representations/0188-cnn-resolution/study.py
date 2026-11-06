from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
from common import R,P,B,np,torch
import importlib.util,time,gc,hashlib
D=P/'experiments/cnn-scale-floor-187'
spec=importlib.util.spec_from_file_location('engine187',D/'run.py');e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e);e.CURRENT_MODEL='resnet50'

def boundary_diagnostics():
 from scipy.ndimage import gaussian_laplace,gaussian_gradient_magnitude
 from PIL import Image
 rows=[]
 for ident in c.plan['IDs']:
  with Image.open(_repo_io(P/'data/train'/f'{ident}.png')) as im:
   assert im.mode=='L' and im.size==(256,256);a=np.asarray(im,dtype=float)/255;rec=np.asarray(im.resize((224,224),Image.Resampling.BICUBIC).resize((256,256),Image.Resampling.BICUBIC),dtype=float)/255
  g=gaussian_gradient_magnitude(a,1)[3:-3,3:-3];gr=gaussian_gradient_magnitude(rec,1)[3:-3,3:-3];anchor=g>=np.quantile(g,.9);contrast=float(np.mean(abs(gaussian_laplace(a,1)[3:-3,3:-3]))/max(a.std(),1e-12));retention=float(gr[anchor].mean()/max(g[anchor].mean(),1e-12));nmse=float(np.mean((a-rec)**2)/max(a.var(),1e-12));rows.append([contrast,retention,nmse])
 c.arrays(R/'boundary-proxies.npz',ids=c.plan['IDs'],values=np.array(rows),columns=['fine_contrast_sigma1','pseudo_boundary_gradient_roundtrip_retention','roundtrip_normalized_MSE']);c.write(R/'boundary-proxy-definition.json',{'definition':c.plan['boundary_proxy'],'label_free':True,'selected_before_model_predictions':True,'sha256':c.sha(R/'boundary-proxies.npz'),'mean_roundtrip_gradient_retention':float(np.mean(np.array(rows)[:,1])),'mean_roundtrip_normalized_MSE':float(np.mean(np.array(rows)[:,2]))})

def main():
 c.check();assert c.read(R/'smoke.json')['passed'];extract=c.read(R/'resnet50/extraction.json');assert extract['passed'];assert c.sha(R/'resnet50/features.npz')==extract['feature_sha256'];assert c.sha(R/'resnet50/prefix.npy')==extract['prefix_sha256'];assert extract['prefix_shape']==[400,1024,16,16];d=c.data('original');X=c.load(R/'resnet50/features.npz')['X'].astype(float);cache=np.load(_repo_io(R/'resnet50/prefix.npy'),mmap_mode='r');z224=c.load(D/'resnet50/OOF.npz');np.testing.assert_array_equal(z224['ids'],d['ids']);root=R/'resnet50';pred=np.full(400,np.nan);start=time.monotonic();proof=[]
 with c.Monitor('resnet50-256-run',900) as mon:
  boundary_diagnostics()
  for k in range(5):
   fold=c.plan['folds'][str(k)];T=np.array(fold['train']);V=np.array(fold['validation']);out=root/f'fold-{k}';out.mkdir();o=e.save_head(out/'initial-head.pt',X,d['y'],T,V,202710185+k);old=e.ck(D/'resnet50'/f'fold-{k}/initial-head.pt');assert o['initial_head_sha256']==old['initial_head_sha256'] and o['order_sha256']==old['order_sha256'];assert o['target_mean']==old['target_mean'] and o['target_std']==old['target_std'];recipe=c.plan['locked_outer_recipes'][str(k)];seed=202710186+100*k+99;pp,rec=e.train(o,cache,X,d['y'][T].copy(),recipe,seed,out,V);pred[V]=pp;oldfit=e.ck(D/'resnet50'/f'fold-{k}'/('outer-'+recipe['optimizer'])/'model.pt');assert oldfit['order_adaptation_sha256']==e.ck(out/'model.pt')['order_adaptation_sha256'];assert oldfit['adaptation_seed']==seed and oldfit['recipe']==recipe and oldfit['update_count']==rec['updates']==240;proof.append({'fold':k,'recipe':recipe,'same187_untrained_head_seed_and_weights':True,'same_head100_batch_orders':True,'same_adaptation12_orders_seed_recipe_240updates':True,'no256_reselection':True})
   if k==0:
    projected=5*rec['seconds']+60;gate=projected<900;c.write(R/'training-smoke.json',{'passed':gate,'first_full_fold_seconds':rec['seconds'],'projected5fold_seconds':projected,'CPU_threads':2,'microbatch':8,'effective_batch':16,'resources':mon.samples[-1],'first_fit_reused_not_extra':True});assert gate,'256trainingexceedsfixedbudget'
   c.emit({'completed_outerfold':k,'native256':True,'seconds':time.monotonic()-start});del o,old,oldfit;gc.collect();torch.mps.empty_cache()
  assert np.isfinite(pred).all();c.arrays(root/'OOF.npz',ids=d['ids'],y=d['y'],groups=d['g'],fold=d['fold'],best021=d['blend_L_MLP'],control224=z224['selected'],native256=pred);report={'study':188,'model':'resnet50','scores256':e.old.evaluate(d,pred),'scores224':c.read(D/'resnet50/report.json')['scores']['selected'],'pairing_proof':proof,'seconds':time.monotonic()-start,'OOF_sha256':c.sha(root/'OOF.npz'),'no_new_optimizer_or_resolution_selection':True};c.write(root/'report.json',report);c.emit(report)
 c.check()
if __name__=='__main__':main()
