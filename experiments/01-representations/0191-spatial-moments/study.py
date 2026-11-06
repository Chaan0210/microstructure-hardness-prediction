from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
from common import R,np,torch
import pipeline as e
import argparse,time,gc

def run(name):
 c.check();d=c.data('original');root=R/name;preds={a:np.full(400,np.nan) for a in c.plan['arms']};records=[];start=time.monotonic();width=2048 if name=='resnet50' else 768
 with c.Monitor(name+'-train',900):
  for k in range(5):
   f=c.plan['folds'][str(k)];T=np.array(f['train']);V=np.array(f['validation']);assert not set(d['g'][T])&set(d['g'][V]);path=root/f'fold-{k}';path.mkdir();pair=[];meanpre=None
   for arm in c.plan['arms']:
    out=path/arm;out.mkdir();X=e.raw_features(name,arm);Z,s=e.fit_pre(X,T,width);c.arrays(out/'preprocess.npz',**s)
    if arm=='mean':meanpre=s
    else:
     for key in ['mean','var','scale','RMS']:np.testing.assert_array_equal(s['bank0_'+key],meanpre['bank0_'+key])
    h,o,u=e.fit_head(Z,d['y'],T,V,202710185+k);torch.save(o,_repo_io(out/'head.pt'));pp=e.predict(h,Z,V,o);tr=e.predict(h,Z,T,o);q=e.ck(out/'head.pt');h2=e.old.head(128);h2.load_state_dict(q['head']);np.testing.assert_array_equal(e.predict(h2,Z,V,q),pp);h2.to('mps');mp=e.predict(h2,Z,V,q,'mps');err=float(abs(mp-pp).max());assert err<.002
    c.arrays(out/'predictions.npz',indices=V,ids=d['ids'][V],prediction=pp);c.arrays(out/'updates.npz',values=u);preds[arm][V]=pp
    rec={'model':name,'arm':arm,'fold':k,'train_RMSE':float(np.sqrt(np.mean((d['y'][T]-tr)**2))),'validation_RMSE':float(np.sqrt(np.mean((d['y'][V]-pp)**2))),'CPU_reload_exact':True,'CPU_MPS_maxabs':err,'parameters':o['parameter_count'],'initial_head_SHA256':o['initial_head_sha256'],'order_SHA256':o['order_sha256'],'updates':len(u),'mean_clip_fraction':float(np.mean(u[:,5]>1)),'max_preclip_gradient':float(u[:,5].max()),'scale':e.diagnose(X,Z,s,o,T,V),'head_SHA256':c.sha(out/'head.pt'),'preprocess_SHA256':c.sha(out/'preprocess.npz'),'prediction_SHA256':c.sha(out/'predictions.npz'),'updates_SHA256':c.sha(out/'updates.npz')};c.write(out/'report.json',rec);pair.append(rec);records.append(rec);c.emit({'model':name,'arm':arm,'fold':k,'train_RMSE':rec['train_RMSE'],'CPU_MPS_maxabs':err,'PCA_variance_retained':rec['scale']['PCA_explained_variance_ratio_sum'],'seconds':time.monotonic()-start});del h,h2;gc.collect();torch.mps.empty_cache()
   assert pair[0]['initial_head_SHA256']==pair[1]['initial_head_SHA256'] and pair[0]['order_SHA256']==pair[1]['order_SHA256'];u=c.load(path/'mean/updates.npz')['values'];v=c.load(path/'mean_std/updates.npz')['values'];np.testing.assert_array_equal(u[:,:4],v[:,:4])
  assert all(np.isfinite(p).all() for p in preds.values());c.arrays(root/'OOF.npz',ids=d['ids'],y=d['y'],fold=d['fold'],groups=d['g'],best021=d['blend_L_MLP'],**preds);report={'model':name,'scores':{a:e.old.evaluate(d,p) for a,p in preds.items()},'fold_records':records,'OOF_SHA256':c.sha(root/'OOF.npz'),'seconds':time.monotonic()-start};c.write(root/'report.json',report);c.emit({'completed':name,'scores':report['scores']})
 c.check()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('model',choices=c.plan['models']);run(p.parse_args().model)
