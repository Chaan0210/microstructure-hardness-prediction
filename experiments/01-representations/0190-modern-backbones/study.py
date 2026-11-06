from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
from common import R,P,B,np,torch
import engine as e
import argparse,time,gc
RECIPE={'name':'fixed_adamw_low','optimizer':'AdamW','head_lr':.0001,'tail_lr':.00001}
def check():
 c.check()
 for n,h in c.read(R/'training-code-lock.json').items():assert c.sha(R/n)==h,n

def inputs(name):
 z=c.read(R/name/'extraction.json');assert z['passed'];assert c.sha(R/name/'features.npz')==z['feature_SHA256'];assert c.sha(R/name/'prefix.npy')==z['prefix_SHA256'];q=c.load(R/name/'features.npz');assert q['ids'].tolist()==c.plan['IDs'];return q['X'].astype(float),c.data('original')

def scale_diagnostic(o,X,T,V):
 sd=np.sqrt(o['feature_var'].numpy());old=sd.copy();old[old==0]=1.;mu=o['feature_mean'].numpy();floor=o['feature_RMS_floor'];scale=o['feature_scale'].numpy();return {'train_n':len(T),'validation_n':len(V),'raw_std_min':float(sd.min()),'raw_std_quantiles':np.quantile(sd,[0,.01,.1,.5,.9,1]).tolist(),'exact_zero_count':int((sd==0).sum()),'std_below_001_RMS_count':int((sd<.001*floor).sum()),'train_RMS_floor':floor,'actual_scale_min':float(scale.min()),'max_validation_absZ_raw_std':float(np.max(abs((X[V]-mu)/old))),'max_validation_absZ_RMS_floor':float(np.max(abs((X[V]-mu)/scale))),'scale_fit_train_only':True,'no_validation_tuning':True}

def frozen(name):
 check();e.CURRENT_MODEL=name;X,d=inputs(name);root=R/name;pred=np.full(400,np.nan);records=[];start=time.monotonic()
 with c.Monitor(name+'-frozen100',600):
  for k in range(5):
   f=c.plan['folds'][str(k)];T=np.array(f['train']);V=np.array(f['validation']);out=root/f'fold-{k}';out.mkdir();o=e.save_head(out/'initial-head.pt',X,d['y'],T,V,202710185+k);q=e.ck(out/'initial-head.pt');h=e.old.head(768);h.load_state_dict(q['head']);assert c.statehash(h)==o['final_head_sha256'];pp=e.old.predict(h,None,None,X,V,q,'cpu');trainpred=e.old.predict(h,None,None,X,T,q,'cpu');h.to('mps');mp=e.old.predict(h,None,None,X,V,q,'mps');diff=float(abs(pp-mp).max());assert diff<.002;pred[V]=pp;c.arrays(out/'frozen100-predictions.npz',indices=V,prediction=pp);diag=scale_diagnostic(q,X,T,V);c.write(out/'scale-diagnostic.json',diag);rec={'fold':k,'train_RMSE':float(np.sqrt(np.mean((d['y'][T]-trainpred)**2))),'CPU_MPS_maxabs':diff,'checkpoint_SHA256':c.sha(out/'initial-head.pt'),'prediction_minmax':[float(pp.min()),float(pp.max())],'scale':diag};c.write(out/'frozen100-report.json',rec);records.append(rec);c.emit({'frozen100':name,'fold':k,'train_RMSE':rec['train_RMSE'],'CPU_MPS':diff,'seconds':time.monotonic()-start});del h,o,q;gc.collect();torch.mps.empty_cache()
  assert np.isfinite(pred).all();c.arrays(root/'frozen100-OOF.npz',ids=d['ids'],y=d['y'],groups=d['g'],fold=d['fold'],best021=d['blend_L_MLP'],prediction=pred);report={'arm':'frozen100','model':name,'score':e.old.evaluate(d,pred),'fold_records':records,'OOF_SHA256':c.sha(root/'frozen100-OOF.npz'),'seconds':time.monotonic()-start};c.write(root/'frozen100-report.json',report);c.emit({'completed_frozen100':name,'score':report['score']})
 check()

def adapt(name):
 check();assert all((R/n/'frozen100-report.json').exists() for n in c.plan['models']);e.CURRENT_MODEL=name;X,d=inputs(name);cache=np.load(_repo_io(R/name/'prefix.npy'),mmap_mode='r');root=R/name;arms=['head_only112','last_stage112'];OOF={n:np.full(400,np.nan) for n in arms};start=time.monotonic();records=[]
 with c.Monitor(name+'-adapt',1800) as mon:
  for k in range(5):
   f=c.plan['folds'][str(k)];T=np.array(f['train']);V=np.array(f['validation']);path=root/f'fold-{k}';o=e.ck(path/'initial-head.pt');assert o['train']==T.tolist() and o['validation']==V.tolist();pair=[]
   for arm in arms:
    out=path/arm;out.mkdir();pp,rec=e.train(o,cache,X,d['y'][T].copy(),RECIPE,202710186+100*k+99,out,V,train_tail=arm=='last_stage112');OOF[arm][V]=pp;pair.append(rec)
   assert pair[0]['head_initial']==pair[1]['head_initial'] and pair[0]['order_sha256']==pair[1]['order_sha256'];assert pair[0]['updates']==pair[1]['updates']==240;records.append({'fold':k,'paired_same_initial_head_seed_orders_updates':True,'head_control_tail_frozen':True,'fine_tail_trained':True,'optimizer_state_reset_equally':True})
   if k==0:
    estimate=5*sum(q['seconds'] for q in pair)+120;gate=estimate<1800;c.write(root/'training-smoke.json',{'passed':gate,'first_paired_fold_seconds':sum(q['seconds'] for q in pair),'projected5_paired_folds_seconds':estimate,'resources':mon.samples[-1],'first_pair_reused':True});assert gate,'190adaptationtimebudget'
   c.emit({'completed_pair':name,'fold':k,'seconds':time.monotonic()-start});del o;gc.collect();torch.mps.empty_cache()
  assert all(np.isfinite(p).all() for p in OOF.values());c.arrays(root/'adapt-OOF.npz',ids=d['ids'],y=d['y'],groups=d['g'],fold=d['fold'],best021=d['blend_L_MLP'],**OOF);scores={n:e.old.evaluate(d,p) for n,p in OOF.items()};gain=scores['head_only112']['RMSE']['blend10']-scores['last_stage112']['RMSE']['blend10'];report={'model':name,'scores':scores,'laststage_blend_gain_vs_matched_head':gain,'laststage_advance_gate':bool(scores['last_stage112']['advance_gate'] and gain>0),'pairing_proof':records,'OOF_SHA256':c.sha(root/'adapt-OOF.npz'),'seconds':time.monotonic()-start};c.write(root/'adapt-report.json',report);c.emit({'completed_adaptation':name,**report})
 check()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['frozen','adapt']);p.add_argument('model',choices=['convnext_tiny','swin_v2_t']);a=p.parse_args();frozen(a.model) if a.mode=='frozen' else adapt(a.model)
