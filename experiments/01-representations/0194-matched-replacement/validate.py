from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import math
import common as c
from common import R,P,np,torch
c.check();d=c.data();r=c.read(R/'report.json');o=c.load(R/'replacement-OOF.npz');norm=c.load(R/'legacy-precision.npz');f=c.load(R/'frozen-components.npz');b=c.load(R/'bootstrap.npz');assert c.sha(R/'replacement-OOF.npz')==r['replacement_OOF_SHA256'] and c.sha(R/'bootstrap.npz')==r['bootstrap_SHA256'];a=d['a'];y=a['y'];fold=a['fold'];ref=a['best021'];manual_rmse=lambda x,p:math.sqrt(math.fsum((float(v)-float(w))**2 for v,w in zip(x,p))/len(x));maxformula=0.;maxboot=0.
with c.Monitor('validation'):
 for key,other in [('ids','ids'),('y','y'),('fold','fold'),('groups','groups')]:np.testing.assert_array_equal(o[key],a[other])
 np.testing.assert_array_equal(o['best021'],ref);np.testing.assert_array_equal(o['detach'],ref)
 for k in range(5):
  V=np.flatnonzero(fold==k);cal=c.plan['calibrators'][str(k)];meta=torch.load(_repo_io(P/f'experiments/auxiliary-cnn-193/original/fold-{k}-detach/model.pt'),weights_only=True,map_location='cpu')['normalization'];ym=meta['y_mean'];ys=meta['y_std'];assert all(abs(o['coefficient'][i]-.9*.9*.3/3*cal['slope'])<1e-15 for i in V)
  for arm in c.ARMS:
   for i in V:
    z=np.float32((float(a[arm][i])-ym)/ys);q=float(np.float32(np.float32(z*np.float32(ys))+np.float32(ym)));assert q==norm[arm][i]
   component=d['old']['components'][V].copy();component[:,2]=(norm[arm][V]+d['cnn']['seeds'][1,V]+d['cnn']['seeds'][2,V])/3;weights=np.array([0.,.1875,.1875,.125,.125]);raw=component@(weights/weights.sum());base=.9*(cal['target_center']+cal['slope']*(raw-cal['prediction_center']))+.1*d['old']['short'][V];independent=.9*(base+f['correction'][V])+.1*d['b']['L_MLP'][V];error=float(abs(independent-o[arm][V]).max());assert error<1e-10;maxformula=max(maxformula,error)
 for arm in c.ARMS:
  assert abs(manual_rmse(y,o[arm])-r['scores'][arm]['RMSE'])<1e-12
  for k in range(5):
   V=fold==k;assert abs(manual_rmse(y[V],o[arm][V])-r['scores'][arm]['fold_RMSE'][k])<1e-12
 groups=np.unique(a['groups']);np.testing.assert_array_equal(groups,b['unique_groups']);draw=np.random.default_rng(202710194).integers(0,len(groups),size=(3000,len(groups)));np.testing.assert_array_equal(draw,b['draws']);rows=np.array([np.flatnonzero(a['groups']==g)[0] for g in groups]);assert len(set(a['groups']))==400;indices=rows[draw]
 pairs=[(f'{new}_vs_{old}',o[old],o[new],r['paired'][f'{new}_vs_{old}']) for new,old in [('true','detach'),('true','permuted'),('permuted','detach')]]
 pairs += [(f'{arm}_vs_best021',ref,o[arm],r['versus_best021'][arm]) for arm in c.ARMS]
 for name,old,new,z in pairs:
  direct=np.sqrt(np.mean((y[indices]-old[indices])**2,axis=1))-np.sqrt(np.mean((y[indices]-new[indices])**2,axis=1));err=float(abs(direct-b[name]).max());assert err<1e-12;maxboot=max(maxboot,err);np.testing.assert_allclose(np.quantile(direct,[.025,.975]),z['conditional_paired_groupbootstrap_CI95'],rtol=0,atol=1e-12);order=sorted(range(400),key=lambda i:((float(y[i])-float(old[i]))**2-(float(y[i])-float(new[i]))**2,i));assert a['ids'][order[-5:]].tolist()==z['removed_IDs'];keep=order[:-5];assert abs(manual_rmse(y[keep],old[keep])-manual_rmse(y[keep],new[keep])-z['top5_removed_gain'])<1e-12;assert abs(manual_rmse(y,old)-manual_rmse(y,new)-z['gain'])<1e-12
 result={'passed':True,'rows':400,'arms':3,'folds':5,'all_ID_order_finite_group_and_fold_checks':True,'float32_scalar_reconstruction_all1200_exact':True,'detach_best021_exact':True,'independent_fullmixture_reconstruction_max_error':maxformula,'independent_fsum_RMSE_and_top5_verified':True,'independent_all3000_paired_bootstrap_max_error':maxboot,'all_fixed_coefficients_verified':True,'preserved_downstream15blocks':True,'old_noop_pipeline_exact':True,'new_model_training_or_inference':False,'OOF_SHA256':c.sha(R/'replacement-OOF.npz')};c.write(R/'validation.json',result);c.emit(result)
c.check()
