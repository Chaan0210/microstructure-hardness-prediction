from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,csv,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import earlystop_experiment_057 as e
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
fv=e.fv;OUT=e.OUT

def main():
 t=time.perf_counter();plan=e.check();torch.set_num_threads(2);results={};nmodels=0;replay_errors=[]
 for split in ('original','20261031','20261101'):
  out=OUT/split
  if not out.exists():continue
  ids,X,y,g,fid,r003,r007,oldmean=e.inputs(split);report=json.loads((out/'report.json').read_text());assert fv.sha(out/'development_predictions.npz')==report['predictions_sha256']
  with np.load(_repo_io(out/'development_predictions.npz')) as z:assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['y'],y);np.testing.assert_array_equal(z['fold'],fid);allp=z['all_checkpoint_seed_predictions'].copy();primary=z['primary'].copy();selected=z['selected_mlp'].copy()
  errs=[];checkpoint_matches=0;reconstruction=[]
  for k in range(5):
   tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);fd=out/f'fold-{k}';log=json.loads((fd/'report.json').read_text());inner=list(GroupKFold(n_splits=3,shuffle=True,random_state=20261227+k).split(X[tr],y[tr],g[tr]));assert not set(g[tr])&set(g[va]);assert fv.sha(fd/'inner_predictions.npz')==log['inner_predictions_sha256']
   with np.load(_repo_io(fd/'inner_predictions.npz')) as z:np.testing.assert_array_equal(z['train_indices'],tr);ip=z['predictions'].copy()
   assert ip.shape==(4,3,320) and np.isfinite(ip).all();scores=[{'epoch':epoch,'rmse':fv.rmse(y[tr],ip[ei].mean(0))} for ei,epoch in enumerate(e.head.EPOCHS)];assert scores==log['inner_scores'];chosen=min(scores,key=lambda a:(a['rmse'],a['epoch']))['epoch'];assert chosen==log['selected_epoch']==report['selected_epochs'][k]
   for j,((it,iv),ir) in enumerate(zip(inner,log['inner_logs'])):
    np.testing.assert_array_equal(ir['train_indices'],tr[it]);np.testing.assert_array_equal(ir['validation_indices'],tr[iv]);assert not set(g[tr[it]])&set(g[tr[iv]]);sc=StandardScaler().fit(X[tr[it]])
    with np.load(_repo_io(fd/f'inner-{j}-scaler.npz')) as z:np.testing.assert_array_equal(z['mean'],sc.mean_);np.testing.assert_array_equal(z['scale'],sc.scale_);np.testing.assert_array_equal(z['var'],sc.var_)
    assert ir['target_mean']==float(y[tr[it]].mean()) and ir['target_std']==float(y[tr[it]].std())
    # Meaningful independent rerun of one complete inner trajectory.
    if split=='original' and k==0 and j==0:
     _,ym,ys,xt,yt=e.prep(X[tr[it]],y[tr[it]]);states,rr=e.head.fit(xt,yt,e.head.SEEDS[0]+100);assert rr['final_hash']==ir['seed_logs'][str(e.head.SEEDS[0])]['final_hash']
     for ei,epoch in enumerate(e.head.EPOCHS):p=e.head.predict(states[epoch],sc.transform(X[tr[iv]]),ym,ys);np.testing.assert_array_equal(p,ip[ei,0,iv]);replay_errors.append(float(abs(p-ip[ei,0,iv]).max()))
   sc,ym,ys,xt,yt=e.prep(X[tr],y[tr]);oldlog=json.loads((_repo_path(ROOT, 'artifacts/seed-stability-052', split, f'fold-{k}', 'report.json')).read_text());tp=[]
   for si,base in enumerate(e.head.SEEDS):
    path=fd/f'seed-{base}.pt';assert fv.sha(path)==log['models'][str(base)]['checkpoint_sha256'];a=torch.load(_repo_io(path),weights_only=True,map_location='cpu');assert a['train_ids']==[ids[i] for i in tr] and a['validation_ids']==[ids[i] for i in va];assert a['seed']==base+k and a['selected_epoch']==chosen;assert a['target_mean']==ym and a['target_std']==ys;np.testing.assert_array_equal(a['scaler_mean'].numpy(),sc.mean_);np.testing.assert_array_equal(a['scaler_scale'].numpy(),sc.scale_);np.testing.assert_array_equal(a['scaler_var'].numpy(),sc.var_)
    old=oldlog['models'][str(base)];assert fv.sha(_repo_path(ROOT, old['checkpoint']))==old['checkpoint_sha256'];oldstate=torch.load(_repo_path(ROOT, old['checkpoint']),weights_only=True,map_location='cpu');assert e.h.state_hash(a['states']['100'])==e.h.state_hash(oldstate['model']);checkpoint_matches+=1
    for epoch,state in a['states'].items():
     p=e.head.predict(state,sc.transform(X[va]),ym,ys);want=allp[e.head.EPOCHS.index(int(epoch)),si,va];np.testing.assert_array_equal(p,want);errs.append(float(abs(p-want).max()))
    tp.append(e.head.predict(a['states'][str(chosen)],xt,ym,ys));nmodels+=1
   np.testing.assert_array_equal(selected[va],allp[e.head.EPOCHS.index(chosen),:,va].mean(1));reconstruction.extend((np.mean(tp,axis=0)-y[tr]).tolist())
  np.testing.assert_array_equal(primary,.75*r003+.25*selected);np.testing.assert_array_equal(r007,.75*r003+.25*allp[-1].mean(0));assert fv.rmse(y,primary)==report['scores']['primary']['rmse'];assert float(np.sqrt(np.mean(np.array(reconstruction)**2)))==report['selected_train_rmse_pooled'];results[split]={'scores':report['scores'],'comparison':report['comparison'],'selected_epochs':report['selected_epochs'],'selected_train_rmse_pooled':report['selected_train_rmse_pooled'],'each_seed100_state_matches007':checkpoint_matches,'reload_max_error':max(errs)}
 sel=json.loads((OUT/'selection.json').read_text());c=results['original']['comparison'];assert sel['advance']==bool(c['gain']>c['bootstrap_sd'] and c['positive_folds']>=3)
 if not sel['advance']:assert set(results)=={'original'}
 e.check();v={'passed':True,'results':results,'models_verified':nmodels,'independent_inner_trajectory_replay_errors':replay_errors,'protected_files':len(plan['protected']),'seconds':time.perf_counter()-t,'scope':'Development400 only; noholdout/test/candidateCSV/upload/install/cost','audit':'Independent read-only code audit found no blocking issue; all15 individual100epochstates matched verified007 models.'};fv.write(_repo_path(ROOT, 'reports/earlystop-057-validation.json'),v);print(json.dumps(v),flush=True)
if __name__=='__main__':main()
