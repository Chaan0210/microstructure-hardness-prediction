from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
from common import R,P,B,np,torch
import engine as e
import time,gc,shutil
D=P/'experiments/cnn-scale-floor-187';A=P/'experiments/cnn-resolution-188';e.CURRENT_MODEL='resnet50'
def main():
 c.check();assert c.read(R/'smoke.json')['passed'];extract=c.read(R/'resnet50/extraction.json');assert extract['passed'] and extract['all400_frozen_features_exact185'];assert extract['prefix_shape']==[400,512,28,28];assert c.sha(R/'resnet50/features.npz')==extract['feature_sha256'] and c.sha(R/'resnet50/prefix.npy')==extract['prefix_sha256'];d=c.data('original');X=c.load(R/'resnet50/features.npz')['X'].astype(float);np.testing.assert_array_equal(X,c.load(B/'resnet50/features.npz')['X']);cache=np.load(_repo_io(R/'resnet50/prefix.npy'),mmap_mode='r');control=c.load(D/'resnet50/OOF.npz');root=R/'resnet50';pred=np.full(400,np.nan);start=time.monotonic();proof=[]
 with c.Monitor('resnet50-two-stage-run',1800) as mon:
  for k in range(5):
   f=c.plan['folds'][str(k)];T=np.array(f['train']);V=np.array(f['validation']);out=root/f'fold-{k}';out.mkdir();src=D/'resnet50'/f'fold-{k}/initial-head.pt';shutil.copyfile(_repo_io(src),_repo_io(out/'initial-head.pt'));assert c.sha(src)==c.sha(out/'initial-head.pt');o=e.ck(out/'initial-head.pt');expected=e.floor_stats(X,d['y'],T,V,202710185+k)
   for n in ['feature_mean','feature_var','feature_scale']:assert torch.equal(o[n],expected[n])
   assert o['feature_RMS_floor']==expected['feature_RMS_floor'];recipe=c.plan['locked_outer_recipes'][str(k)];seed=202710186+100*k+99;pp,rec=e.train(o,cache,X,d['y'][T].copy(),recipe,seed,out,V);pred[V]=pp;old=e.ck(D/'resnet50'/f'fold-{k}'/('outer-'+recipe['optimizer'])/'model.pt');now=e.ck(out/'model.pt');assert old['adaptation_seed']==now['adaptation_seed']==seed and old['recipe']==now['recipe']==recipe and old['order_adaptation_sha256']==now['order_adaptation_sha256'];assert old['update_count']==now['update_count']==240;controlrec=c.read(D/'resnet50'/f'fold-{k}'/('outer-'+recipe['optimizer'])/'report.json');assert abs(rec['initial_train_RMSE']-controlrec['initial_train_RMSE'])<.002;proof.append({'fold':k,'head_checkpoint_byte_exact187':True,'scaler_and_frozen_features_exact187':True,'same_optimizerLR_seed_orders240updates':True,'initial_train_RMSE_difference':rec['initial_train_RMSE']-controlrec['initial_train_RMSE'],'only_layer3_added_to_existing_tail_LR_group':True})
   if k==0:
    estimate=5*rec['seconds']+60;gate=estimate<1800;c.write(R/'training-smoke.json',{'passed':gate,'first_fit_seconds':rec['seconds'],'projected5fit_seconds':estimate,'same_micro8_effective16':True,'resources':mon.samples[-1],'first_fit_reused':True});assert gate,'189timebudget'
   c.emit({'completed_outerfold':k,'two_stages':True,'seconds':time.monotonic()-start});del o,old,now;gc.collect();torch.mps.empty_cache()
  assert np.isfinite(pred).all();c.arrays(root/'OOF.npz',ids=d['ids'],y=d['y'],groups=d['g'],fold=d['fold'],best021=d['blend_L_MLP'],control_layer4=control['selected'],two_stage=pred);r={'study':189,'score_two_stage':e.old.evaluate(d,pred),'score_layer4':c.read(D/'resnet50/report.json')['scores']['selected'],'pairing_proof':proof,'seconds':time.monotonic()-start,'OOF_SHA256':c.sha(root/'OOF.npz')};c.write(root/'report.json',r);c.emit(r)
 c.check()
if __name__=='__main__':main()
