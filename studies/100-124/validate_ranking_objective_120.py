"""Loss/gradient fixtures, exact paired budget, independent checkpoint/selection verification."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,argparse,hashlib,math
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import ranking_objective_120 as o
import numpy as np,torch
from sklearn.model_selection import GroupKFold
e=o.e

def fixture():
 o.check();pn=np.array([.2,-.7,.9,1.3]);yn=np.array([-1.,-.7,.2,1.8]);pt=torch.tensor(pn,dtype=torch.float64,requires_grad=True);loss,count=o.rank_loss(pt,yn);grad=torch.autograd.grad(loss,pt)[0].numpy();manual=0.;g=np.zeros(4);pairs=[]
 for i in range(4):
  for j in range(i+1,4):
   if abs(yn[i]-yn[j])>=.5:pairs.append((i,j))
 for i,j in pairs:
  sign=np.sign(yn[i]-yn[j]);delta=pn[i]-pn[j];manual+=np.logaddexp(0.,-sign*delta);v=-sign/(1+np.exp(sign*delta));g[i]+=v;g[j]-=v
 manual/=len(pairs);g/=len(pairs);assert count==len(pairs);assert abs(float(loss.detach())-manual)<1e-14;np.testing.assert_allclose(grad,g,atol=1e-14,rtol=0);shift,_=o.rank_loss(pt+7.,yn);assert abs(float(shift.detach())-manual)<1e-14
 z=torch.tensor([1.,2.,3.],requires_grad=True);null,c=o.rank_loss(z,np.zeros(3));assert c==0 and float(null.detach())==0.;np.testing.assert_array_equal(torch.autograd.grad(null,z)[0].numpy(),np.zeros(3));mse=.5*((pt-torch.tensor(yn))**2).mean();mg=torch.autograd.grad(mse,pt)[0].numpy();np.testing.assert_allclose(mg,(pn-yn)/4,atol=1e-14,rtol=0)
 result={'passed':True,'rank_value_and_gradient_independent':True,'rank_shift_invariant':True,'no_eligible_pair_zero_gradient':True,'halfMSE_gradient_exact':True,'pairs':count};e.write(o.OUT/'loss-fixture.json',result);e.emit(result)

def checkpoint(d,tr,va,path,seed,weight):
 rec=e.read(path/'report.json');assert e.sha(path/'model.pt')==rec['model_sha256'] and e.sha(path/'predictions.npz')==rec['predictions_sha256'];a=torch.load(_repo_io(path/'model.pt'),map_location='cpu',weights_only=True);assert a['train_indices']==tr.tolist() and a['validation_indices']==va.tolist();assert a['train_ids']==[d['ids'][i] for i in tr] and a['validation_ids']==[d['ids'][i] for i in va];assert a['seed']==seed and a['rank_weight']==weight and a['loss_kind']=='mse' and a['epochs']==12;assert not set(d['g'][tr])&set(d['g'][va])
 mean=d['B'][tr].mean(0);var=d['B'][tr].var(0);scale=np.sqrt(var);scale[scale==0]=1.;np.testing.assert_allclose(a['mean'].numpy(),mean,atol=1e-12,rtol=1e-12);np.testing.assert_allclose(a['var'].numpy(),var,atol=1e-12,rtol=1e-12);np.testing.assert_allclose(a['scale'].numpy(),scale,atol=1e-12,rtol=1e-12);assert a['target_mean']==d['y'][tr].mean() and a['target_std']==d['y'][tr].std();assert e.old.e.h.state_hash(a['state'])==rec['state_sha256']
 m=e.Small(a['mean'].numpy(),a['scale'].numpy(),seed);assert e.old.e.h.state_hash(m.state_dict())==rec['initial_sha256'];m.load_state_dict(a['state']);m=m.to('mps');pred=e.small_predict(m,d['B'],va)*a['target_std']+a['target_mean']
 with np.load(_repo_io(path/'predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(z['train_indices'],tr);np.testing.assert_array_equal(z['validation_indices'],va);np.testing.assert_array_equal(z['predictions'],pred)
 m=m.cpu().eval();cpu=e.small_predict(m,d['B'],va,device='cpu')*a['target_std']+a['target_mean'];err=float(abs(cpu-pred).max());assert err<1e-3;del m;torch.mps.empty_cache()
 orders=np.random.default_rng(seed).permuted(np.tile(np.arange(len(tr)),(12,1)),axis=1);assert hashlib.sha256(orders.tobytes()).hexdigest()==rec['order_sha256'];maskseeds=[];pairs=0
 for ep,order in enumerate(orders):
  for first in range(0,len(tr),16):
   group=tr[order[first:first+16]];y=(d['y'][group]-a['target_mean'])/a['target_std'];ii,jj=np.triu_indices(len(y),1);pairs+=int((abs(y[ii]-y[jj])>=.5).sum());maskseeds.extend(seed+10000*(ep+1)+first+off for off in range(0,len(group),2))
 assert rec['eligible_pair_uses']==pairs and rec['optimizer_steps']==12*math.ceil(len(tr)/16) and rec['example_visits']==len(tr)*12;assert hashlib.sha256(np.array(maskseeds,dtype=np.int64).tobytes()).hexdigest()==rec['dropout_sha256'];assert (rec['first_weighted_rank_gradient_norm'] is not None and rec['first_weighted_rank_gradient_norm']>0) if weight else rec['first_weighted_rank_gradient_norm'] is None
 return pred,rec,err

def validate(split):
 o.check();torch.set_num_threads(2);d=o.load(split);out=o.OUT/split;report=e.read(out/'report.json');assert e.sha(out/'predictions.npz')==report['predictions_sha256'];start=time.perf_counter();raw=np.full((2,400),np.nan);selected=np.full(400,np.nan);count=0;cpuerr=0.
 for k in range(5):
  tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);path=out/f'fold-{k}';rec=e.read(path/'report.json');assert e.sha(path/'report.json')==report['fold_reports_sha256'][f'fold-{k}/report.json'];assert e.sha(path/'selection.npz')==rec['selection_sha256'];inner=list(GroupKFold(3,shuffle=True,random_state=e.cal.INNER_SEED+k).split(d['B'][tr],d['y'][tr],d['g'][tr]));ip=np.full((2,len(tr)),np.nan)
  for j,(it,iv) in enumerate(inner):
   assert rec['inner_splits'][j]=={'train_indices':tr[it].tolist(),'validation_indices':tr[iv].tolist()};logs=[]
   for wi,w in enumerate(o.WEIGHTS):
    modelpath=path/f'inner-{j}/weight-{wi}';assert e.sha(modelpath/'report.json')==rec['model_reports_sha256'][f'inner-{j}/weight-{wi}/report.json'];pp,rr,err=checkpoint(d,tr[it],tr[iv],modelpath,o.SEED+k+100*(j+1),w);ip[wi,iv]=pp;logs.append(rr);count+=1;cpuerr=max(cpuerr,err)
   for key in ['initial_sha256','order_sha256','dropout_sha256','optimizer_steps','example_visits','eligible_pair_uses']:assert logs[0][key]==logs[1][key]
  scores=np.mean((ip-d['y'][tr][None,:])**2,1);np.testing.assert_array_equal(scores,list(rec['inner_MSE'].values()));chosen=int(scores.argmin());assert chosen==rec['chosen_index'] and o.WEIGHTS[chosen]==rec['chosen_weight'];logs=[]
  for wi,w in enumerate(o.WEIGHTS):
   pp,rr,err=checkpoint(d,tr,va,path/f'outer/weight-{wi}',o.SEED+k,w);raw[wi,va]=pp;logs.append(rr);count+=1;cpuerr=max(cpuerr,err)
  for key in ['initial_sha256','order_sha256','dropout_sha256','optimizer_steps','example_visits','eligible_pair_uses']:assert logs[0][key]==logs[1][key]
  selected[va]=raw[chosen,va]
  with np.load(_repo_io(path/'selection.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(z['inner_predictions'],ip);np.testing.assert_array_equal(z['outer_predictions'],raw[:,va]);np.testing.assert_array_equal(z['selected'],selected[va])
 with np.load(_repo_io(out/'predictions.npz'),allow_pickle=False) as z:
  assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['y'],d['y']);np.testing.assert_array_equal(z['head_predictions'],raw);np.testing.assert_array_equal(z['selected_head'],selected);np.testing.assert_array_equal(z['primary'],d['reference013']+.1*(selected-d['basehead']));np.testing.assert_array_equal(z['mse_control'],d['reference013']+.1*(raw[0]-d['basehead']))
  for key in report['scores']:assert float(np.mean((d['y']-z[key])**2))==report['scores'][key]['MSE']
 o.check();result={'passed':True,'split':split,'models_reloaded':count,'inner_choices_recomputed':5,'MPS_reload_exact':True,'CPU_MPS_max_error':cpuerr,'trainlocal_target_and_feature_stats':True,'pairlabels_training_only':True,'paired_initialization_masks_batches_updates_exact':True,'report_sha256':e.sha(out/'report.json'),'seconds':time.perf_counter()-start};e.write(o.OUT/f'validation-{split}.json',result);e.emit(result)

def leakage():
 o.check();torch.set_num_threads(2);d=o.load('original');alter=dict(d);alter['y']=d['y'].copy();outer=np.flatnonzero(d['fid']==0);alter['y'][outer]=10000+np.arange(len(outer));path=o.OUT/'outer-label-replay';pp,selected,rec=o.outer(alter,'original',0,path);old=o.OUT/'original/fold-0';oldrec=e.read(old/'report.json');assert rec['chosen_weight']==oldrec['chosen_weight'] and rec['inner_MSE']==oldrec['inner_MSE']
 with np.load(_repo_io(old/'selection.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(pp,z['outer_predictions']);np.testing.assert_array_equal(selected,z['selected'])
 for name in rec['model_reports_sha256']:assert e.read(path/name)['state_sha256']==e.read(old/name)['state_sha256']
 tr=np.array(oldrec['inner_splits'][0]['train_indices']);va=np.array(oldrec['inner_splits'][0]['validation_indices']);alter2=dict(d);alter2['y']=d['y'].copy();alter2['y'][va]+=10000;pp,recs=o.paired(alter2,tr,va,o.SEED+100,o.OUT/'inner-label-replay')
 for wi in range(2):assert recs[wi]['state_sha256']==e.read(old/f'inner-0/weight-{wi}/report.json')['state_sha256']
 result={'passed':True,'outer80_perturbed_full_nested8models_states_predictions_choices_exact':True,'inner107_perturbed_fixed2candidate_states_exact':True,'old013_components_unchanged':True,'source_sha256':e.sha(Path(__file__))};e.write(o.OUT/'leakage-validation.json',result);o.check();e.emit(result)
if __name__=='__main__':
 torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['fixture',*e.SPLITS,'leakage']);x=p.parse_args();{'fixture':fixture,'leakage':leakage}.get(x.mode,lambda:validate(x.mode))()
