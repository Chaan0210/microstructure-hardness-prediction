"""Independent nested split, reload, scalar optimization and leakage verification."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import nested_fusion_112 as e
import numpy as np,torch,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize_scalar

def validate_deep(d,path,ti,vi,excluded,offset,ell):
 r=e.read(path/'report.json');assert r['train_indices']==ti.tolist() and r['validation_indices']==vi.tolist() and r['outer_validation_indices']==excluded.tolist()
 assert not set(d['g'][ti])&set(d['g'][vi]) and not (set(ti)|set(vi))&set(excluded)
 for file,h in r['files_sha256'].items():assert e.sha(path/file)==h,file
 with np.load(_repo_io(path/'predictions.npz'),allow_pickle=False) as z:p={a:z[a].copy() for a in z.files}
 np.testing.assert_array_equal(p['train_indices'],ti);np.testing.assert_array_equal(p['validation_indices'],vi);ids=d['ids'];y=d['y'];seed_offset=offset+100*(ell+1)
 a=joblib.load(_repo_io(path/'baseline.joblib'));assert a['train_ids']==[ids[i] for i in ti] and a['weights']==d['weights'];bp=sum(a['weights'][n]*m.predict(d['Xbase'][vi]) for n,m in a['models'].items());np.testing.assert_allclose(bp,p['baseline'],atol=1e-10,rtol=0)
 a=joblib.load(_repo_io(path/'kernel.joblib'));assert a['train_ids']==[ids[i] for i in ti];np.testing.assert_array_equal(a['model'].predict(d['K'][vi]),p['kernel']);kp,km,kr=e.cal.fv.kernel_fold(d['K'],y,d['g'],d['blocks'],ti,vi,seed_offset);assert kr==r['logs']['kernel'];np.testing.assert_array_equal(kp,p['kernel'])
 a=torch.load(_repo_io(path/'cnn.pt'),map_location='cpu',weights_only=True);assert a['train_ids']==[ids[i] for i in ti] and a['validation_ids']==[ids[i] for i in vi];assert a['seed']==e.cal.fv.CNN_SEED+seed_offset;n=a['normalization'];assert n=={'x_mean':float(d['images'][ti].mean()),'x_std':float(d['images'][ti].std()),'y_mean':float(y[ti].mean()),'y_std':float(y[ti].std())}
 m=e.cal.fv.engine.TinyCNN().eval();m.load_state_dict(a['state_dict']);cp=[]
 with torch.no_grad():
  for j in range(0,len(vi),32):cp.extend((m(torch.tensor((d['images'][vi[j:j+32]]-n['x_mean'])/max(n['x_std'],1e-6))).numpy()*n['y_std']+n['y_mean']).tolist())
 cnnerr=float(abs(np.array(cp)-p['cnn']).max());assert cnnerr<1e-3
 for arm in ['S','B']:
  sc=StandardScaler().fit(d[arm][ti]);ym=float(y[ti].mean());ys=float(y[ti].std())
  for seed in e.cal.SEEDS:
   a=torch.load(_repo_io(path/f'{arm}-{seed}.pt'),map_location='cpu',weights_only=True);assert a['train_ids']==[ids[i] for i in ti] and a['validation_ids']==[ids[i] for i in vi] and a['seed']==seed+seed_offset and a['target_mean']==ym and a['target_std']==ys
   for key,value in [('scaler_mean',sc.mean_),('scaler_scale',sc.scale_),('scaler_var',sc.var_)]:np.testing.assert_array_equal(a[key].numpy(),value)
   m=e.cal.h.Head(d[arm].shape[1]);m.load_state_dict(a['model']);pp=e.cal.h.predict(m,sc.transform(d[arm][vi]),ym,ys);np.testing.assert_array_equal(pp,p[f'{arm}_{seed}'])
  np.testing.assert_array_equal(p[arm+'mean'],np.mean([p[f'{arm}_{s}'] for s in e.cal.SEEDS],0))
 exact=.375*p['baseline']+.1875*p['kernel']+.1875*p['cnn']+.125*p['Smean']+.125*p['Bmean'];np.testing.assert_array_equal(exact,p['ensemble008']);return exact,cnnerr

def validate(split):
 plan=e.check();torch.set_num_threads(2);start=time.perf_counter();d=e.data(split);tokens,mom=e.pa.get_cache(d);out=e.OUT/split;r=e.read(out/'report.json');assert e.sha(out/'predictions.npz')==r['prediction_sha256'];maxcnn=0.;maxpool=0.;maxcpu=0.;maxopt=0.;models=0;deepcount=0
 with np.load(_repo_io(out/'predictions.npz'),allow_pickle=False) as z:ps={a:z[a].copy() for a in z.files}
 assert ps['ids'].tolist()==d['ids'];np.testing.assert_array_equal(ps['y'],d['y']);np.testing.assert_array_equal(ps['fold'],d['fid']);np.testing.assert_array_equal(ps['reference011'],d['reference011']);np.testing.assert_array_equal(ps['outerP'],d['outerP'])
 for k in range(5):
  tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);fd=out/f'fold-{k}';fr=e.read(fd/'report.json');assert e.sha(fd/'report.json')==r['fold_report_sha256'][f'fold-{k}/report.json'];assert e.sha(fd/'meta.npz')==fr['meta_sha256']
  with np.load(_repo_io(fd/'meta.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(z['train_indices'],tr);np.testing.assert_array_equal(z['validation_indices'],va);ip=z['innerP'].copy()
  inner=list(GroupKFold(3,shuffle=True,random_state=e.cal.INNER_SEED+k).split(d['S'][tr],d['y'][tr],d['g'][tr]));assembled=np.full_like(ip,np.nan)
  for j,(a,b) in enumerate(inner):
   it,iv=tr[a],tr[b];offset=k+100*(j+1);path=fd/f'inner-{j}';cr=e.read(path/'report.json');assert e.sha(path/'report.json')==fr['cell_report_sha256'][f'inner-{j}/report.json'];assert e.sha(path/'calibration.npz')==cr['calibration_sha256']
   with np.load(_repo_io(path/'calibration.npz'),allow_pickle=False) as z:c={a:z[a].copy() for a in z.files}
   np.testing.assert_array_equal(c['train_indices'],it);np.testing.assert_array_equal(c['validation_indices'],iv);nested=np.full(len(it),np.nan);excluded=np.setdiff1d(np.arange(400),it)
   deep=list(GroupKFold(3,shuffle=True,random_state=e.cal.INNER_SEED+offset).split(d['S'][it],d['y'][it],d['g'][it]))
   for ell,(dt,dv) in enumerate(deep):
    deepdir=path/f'deep-{ell}';assert e.sha(deepdir/'report.json')==cr['deep'][ell]['report_sha256'];pred,err=validate_deep(d,deepdir,it[dt],it[dv],excluded,offset,ell);nested[dv]=pred;maxcnn=max(maxcnn,err);deepcount+=1
   np.testing.assert_array_equal(nested,c['inner_oof008']);center=nested.mean();coef=np.linalg.lstsq(np.column_stack([np.ones(len(it)),nested-center]),d['y'][it],rcond=None)[0];np.testing.assert_allclose(coef,[cr['calibrator']['target_center'],cr['calibrator']['slope']],atol=1e-10,rtol=0);np.testing.assert_allclose(c['p009'],coef[0]+coef[1]*(c['p008']-center),atol=1e-10,rtol=0)
   for arm in ['shortB','mean','attention']:
    file=path/arm/'model.pt';log=e.read(path/arm/'report.json');assert e.sha(path/arm/'report.json')==cr['model_reports'][arm] and e.sha(file)==log['model_sha256'];saved=torch.load(_repo_io(file),map_location='cpu',weights_only=True);assert saved['train_ids']==[d['ids'][i] for i in it] and saved['validation_ids']==[d['ids'][i] for i in iv];assert saved['target_mean']==d['y'][it].mean() and saved['target_std']==d['y'][it].std();assert e.old.e.h.state_hash(saved['state'])==log['state_sha256']
    if arm=='shortB':sc=StandardScaler().fit(d['B'][it]);mean,scale,var=sc.mean_,sc.scale_,sc.var_;m=e.Small(mean,scale,saved['seed']).to('mps')
    else:mean,scale,var=e.pa.normalize(mom,it);m=e.pa.Pool(mean,scale,saved['seed'],arm).to('mps')
    for key,value in [('mean',mean),('scale',scale),('var',var)]:np.testing.assert_array_equal(saved[key].numpy(),value)
    m.load_state_dict(saved['state']);pred=(e.small_predict(m,d['B'],iv) if arm=='shortB' else e.pa.predict(m,tokens,iv)[0])*saved['target_std']+saved['target_mean'];err=float(abs(pred-c[arm]).max());assert err==0;maxpool=max(maxpool,err)
    if k==0 and j==0:
     m=m.cpu().eval()
     with torch.no_grad():cp=m(torch.tensor(d['B'][iv[:2]] if arm=='shortB' else np.array(tokens[iv[:2]],copy=True),dtype=torch.float32)).numpy().astype(float)*saved['target_std']+saved['target_mean']
     ce=float(abs(cp-c[arm][:2]).max());assert ce<.001;maxcpu=max(maxcpu,ce)
    del m;torch.mps.empty_cache();models+=1
   np.testing.assert_array_equal(c['p011'],.9*c['p009']+.1*c['shortB']);assembled[b]=np.column_stack([c['p011'],c['mean'],c['attention']])
  np.testing.assert_array_equal(assembled,ip)
  for arm,col in e.ARMS.items():
   meta=fr['metas'][arm];delta=ip[:,col]-ip[:,0];res=d['y'][tr]-ip[:,0];f=lambda w:float(np.mean((res-w*delta)**2));opt=minimize_scalar(f,bounds=(0,1),method='bounded',options={'xatol':1e-13});choices=[(0.,f(0.)),(1.,f(1.)),(float(opt.x),float(opt.fun))];ind=min(choices,key=lambda z:z[1]);err=abs(f(meta['alpha'])-ind[1]);assert err<1e-9;maxopt=max(maxopt,err);assert f(meta['alpha'])<=f(0)+1e-10
   np.testing.assert_array_equal(e.meta_predict({'alpha':0},d['outerP'][va][:,[0,col]]),d['reference011'][va]);pp=meta['alpha']*d['outerP'][va,col]+(1-meta['alpha'])*d['reference011'][va];np.testing.assert_allclose(pp,ps[arm][va],atol=1e-12,rtol=0)
  e.emit({'validated_split':split,'outer':k,'deep008_cells':deepcount,'new_heads':models})
 for name in ['reference011','mean','attention']:
  assert r['scores'][name]['rmse']==e.rmse(d['y'],ps[name]);assert r['scores'][name]['mse']==float(np.mean((d['y']-ps[name])**2))
 e.check();result={'passed':True,'split':split,'deep008_cells':deepcount,'deep_models':{'CNN':deepcount,'MLP':6*deepcount,'baseline_pipelines':3*deepcount,'KRR':deepcount},'new_heads_reloaded':models,'all_group_disjoint_splits_and_train_normalizers_verified':True,'inner_KRR_selections_refitted':deepcount,'max_CNN_CPU_MPS_error':maxcnn,'max_new_head_MPS_reload_error':maxpool,'max_new_head_CPU_MPS_error':maxcpu,'independent_scalar_optimizer_max_objective_error':maxopt,'alpha0_exactly011':True,'report_sha256':e.sha(out/'report.json'),'seconds':time.perf_counter()-start,'source_sha256':e.sha(Path(__file__))};e.write(e.OUT/f'validation-{split}.json',result);e.emit(result)

def replay():
 e.check();torch.set_num_threads(2);d=e.data('original');tokens,mom=e.pa.get_cache(d);tr=np.flatnonzero(d['fid']!=0);va=np.flatnonzero(d['fid']==0);source=e.cal.OUT/'original/fold-0/inner-0'
 with np.load(_repo_io(source/'predictions.npz'),allow_pickle=False) as z:it=z['train_indices'];iv=z['validation_indices']
 pert=dict(d);pert['y']=d['y'].copy();excluded=np.setdiff1d(np.arange(400),it);pert['y'][excluded]=10000+np.arange(len(excluded));dest=e.OUT/'validation-replay';pp,rec=e.inner_cell(pert,tokens,mom,it,iv,va,0,0,source,dest);old=e.OUT/'original/fold-0/inner-0'
 with np.load(_repo_io(old/'calibration.npz'),allow_pickle=False) as z:expected=np.column_stack([z['p011'],z['mean'],z['attention']])
 err=float(abs(pp-expected).max());assert err<1e-10
 for arm in ['shortB','mean','attention']:assert e.read(dest/arm/'report.json')['state_sha256']==e.read(old/arm/'report.json')['state_sha256']
 for ell in range(3):
  for arm in ['S','B']:
   for seed in e.cal.SEEDS:assert e.read(dest/f'deep-{ell}'/'report.json')['logs'][arm][str(seed)]['final_state_sha256']==e.read(old/f'deep-{ell}'/'report.json')['logs'][arm][str(seed)]['final_state_sha256']
 e.write(e.OUT/'validation-replay.json',{'passed':True,'all_outside_innertrain_labels_perturbed':len(excluded),'max_prediction_error':err,'all18_deep_MLP_and3_head_states_exact':True,'includes_outer80_and_inner107_heldout':True,'source_sha256':e.sha(Path(__file__))});e.emit({'full_nested_cell_label_perturbation_replay_passed':True,'max_error':err})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=[*e.SPLITS,'replay']);a=ap.parse_args();replay() if a.mode=='replay' else validate(a.mode)
