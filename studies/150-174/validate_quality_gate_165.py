"""Independent bounded ridge/OLS and all018 residual fits; nested label exclusion audit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import quality_gate_165 as t
import numpy as np,time,json,sys
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
s=t.s;a=t.a;e=t.e;n=t.n;x=t.x;R=t.ROOT;OUT=t.OUT

def manual_meta(P,y,Q,qt,qv,m):
 center=np.median(qt,0);iqr=np.quantile(qt,.75,axis=0)-np.quantile(qt,.25,axis=0);iqr[iqr<1e-12]=1
 np.testing.assert_array_equal(center,m['quality_center']);np.testing.assert_array_equal(iqr,m['quality_iqr'])
 z=np.maximum(-1,np.minimum(1,(qt-center)/(2*iqr)));zq=np.maximum(-1,np.minimum(1,(qv-center)/(2*iqr)))
 weights=np.array([0,.3,.3,.2,.2]);p=P@weights;pq=Q@weights;contrast=P[:,2]-(P[:,1]*.3+P[:,3]*.2+P[:,4]*.2)/.7;cq=Q[:,2]-(Q[:,1]*.3+Q[:,3]*.2+Q[:,4]*.2)/.7
 design=np.column_stack([np.ones(len(p)),p]);coef=np.linalg.lstsq(design,y,rcond=None)[0];res=(y-design@coef)/y.std();D=z*(contrast*coef[1]/y.std())[:,None]
 theta=np.linalg.lstsq(np.vstack([D/np.sqrt(len(y)),np.sqrt(t.LAMBDA)*np.eye(2)]),np.r_[res/np.sqrt(len(y)),np.zeros(2)],rcond=None)[0]
 if abs(theta).sum()>t.LIMIT:
  H=D.T@D/len(y)+t.LAMBDA*np.eye(2);b=D.T@res/len(y);signs=np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
  r=minimize(lambda v:v@H@v-2*b@v,np.zeros(2),jac=lambda v:2*(H@v-b),constraints={'type':'ineq','fun':lambda v:t.LIMIT-signs@v,'jac':lambda v:-signs},method='SLSQP',options={'ftol':1e-14,'maxiter':1000});assert r.success;theta=r.x
 if not m['enabled']:theta=np.zeros(2)
 np.testing.assert_allclose(theta,m['theta'],atol=1e-9,rtol=0);delta=z@theta;dv=zq@theta
 ww=np.column_stack([.3*(1-dv/.7),.3+dv,.2*(1-dv/.7),.2*(1-dv/.7)])
 assert ww.min()>0 and np.max(abs(ww.sum(1)-1))<1e-12 and max(abs(dv))<=t.LIMIT+1e-10
 adjusted=p+delta*contrast;query=pq+dv*cq;coef2=np.linalg.lstsq(np.column_stack([np.ones(len(p)),adjusted]),y,rcond=None)[0]
 result=np.column_stack([np.ones(len(pq)),query])@coef2
 actual,m2=t.meta(P,y,Q,qt,qv,m['enabled']);err=float(abs(actual-result).max());assert err<1e-7
 for name in ['theta','quality_center','quality_iqr','calibrator','pre_calibrator']:assert m2[name]==m[name]
 return result,err

def verify_block(d,B,V,anchor,path):
 r=e.read(path/'report.json');z=a.readnp(path/'predictions.npz',r['predictions_sha256']);np.testing.assert_array_equal(z['B'],B);np.testing.assert_array_equal(z['V'],V);np.testing.assert_allclose(z['anchor'],anchor,atol=1e-7,rtol=0)
 res=d['y'][B]-z['anchor'];np.testing.assert_array_equal(res,z['residual']);count=0;err=0.;ip=np.empty_like(z['inner']);folds=list(GroupKFold(3,shuffle=True,random_state=r['foldseed']).split(d['X'][B],res,d['g'][B]))
 for ell,(I,H) in enumerate(folds):
  assert r['splits'][ell]=={'train':B[I].tolist(),'validation':B[H].tolist()}
  assert not set(d['g'][B[I]])&set(d['g'][B[H]])
  for ci,kind in enumerate(s.KINDS):
   file=path/f'inner-{ell}-{kind}.joblib';assert e.sha(file)==r['models_sha256'][file.name];obj=t.joblib.load(_repo_io(file));np.testing.assert_array_equal(obj['train'],B[I]);np.testing.assert_array_equal(obj['validation'],B[H])
   pp=n.predict(obj['model'],d['X'][B[H]]);manual=s.q.vn.manual(obj['model'],d['X'][B[I]],res[I],d['X'][B[H]],d['blocks']);err=max(err,float(abs(pp-manual).max()));assert err<2e-9;np.testing.assert_array_equal(pp,z['inner'][ci,H]);ip[ci,H]=pp;count+=1
 w,wins=s.independent_weights((ip-res)**2,d['g'][B],z['draws']);np.testing.assert_array_equal(w,z['weights']);np.testing.assert_array_equal(wins,z['wins']);_,draw,_=s.weights((ip-res)**2,d['g'][B],r['bootseed']);np.testing.assert_array_equal(draw,z['draws']);p=[]
 for ci,kind in enumerate(s.KINDS):
  file=path/f'full-{kind}.joblib';assert e.sha(file)==r['models_sha256'][file.name];obj=t.joblib.load(_repo_io(file));np.testing.assert_array_equal(obj['train'],B);pp=n.predict(obj['model'],d['X'][V]);manual=s.q.vn.manual(obj['model'],d['X'][B],res,d['X'][V],d['blocks']);err=max(err,float(abs(pp-manual).max()));assert err<2e-9;np.testing.assert_array_equal(pp,z['full'][ci]);p.append(pp);count+=1
 np.testing.assert_allclose(sum(wi*pi for wi,pi in zip(w,p)),z['primary'],atol=1e-12,rtol=0);return z['primary'],count,err

def validate(split):
 start=time.perf_counter();plan=e.read(OUT/'plan.json')
 for f,h in {**plan['hashes'],**plan['protected']}.items():assert e.sha(R/f)==h
 rr=e.read(OUT/split/'report.json')
 for f,h in rr['read_input_sha256'].items():assert e.sha(R/f)==h
 d=t.load(split);saved=a.readnp(OUT/split/'predictions.npz',rr['predictions_sha256']);assert saved['ids'].tolist()==d['ids'];np.testing.assert_array_equal(saved['y'],d['y']);np.testing.assert_array_equal(saved['reference018'],d['reference018']);count=0;error=0.;meta_error=0.;model_records=[]
 for k in range(5):
  dd,_=x.replace(d,split,k,x.OUT/split/f'cnn-fold-{k}',False);T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);q=d['quality'];path=OUT/split/f'fold-{k}';r=e.read(path/'report.json');assert e.sha(path/'report.json')==rr['fold_reports'][f'fold-{k}/report.json'];z=a.readnp(path/'predictions.npz',r['predictions_sha256'])
  np.testing.assert_array_equal(r['outer_meta']['train'],T);np.testing.assert_array_equal(r['outer_meta']['query'],V)
  pred,er=manual_meta(dd['middleP'][k],d['y'][T],dd['P'][V],q[T],q[V],r['outer_meta']);meta_error=max(meta_error,er);base=.9*pred+.1*d['short'][V];np.testing.assert_allclose(base,z['base'],atol=1e-7,rtol=0);cor=[];model_records.append(r['outer_meta'])
  for j in range(3):
   A,B,P,short,_=dd['cells'][k,j];assert set(A)|set(B)==set(T) and not set(A)&set(B) and not(set(A)|set(B))&set(V)
   mr=r['anchor_meta'][j];np.testing.assert_array_equal(mr['train'],A);np.testing.assert_array_equal(mr['query'],B)
   pp,er=manual_meta(dd['deepP'][k,j],d['y'][A],P,q[A],q[B],mr);meta_error=max(meta_error,er);model_records.append(mr)
   bp=path/f'block-{j}';assert e.sha(bp/'report.json')==r['block_reports'][f'block-{j}/report.json'];p,c,er=verify_block(dd,B,V,.9*pp+.1*short,bp);count+=c;error=max(error,er);cor.append(p)
  np.testing.assert_array_equal(z['base']+np.mean(cor,0),saved['primary'][V]);print('VERIFIED',split,k,flush=True)
  if k==0:
   # Query quality values must not alter either training fit.
   _,shift=t.meta(dd['middleP'][k],d['y'][T],dd['P'][V],q[T],q[V]+1e6)
   for name in ['theta','quality_center','quality_iqr','calibrator','pre_calibrator']:assert shift[name]==r['outer_meta'][name]
   pert=dict(dd);pert['y']=d['y'].copy();pert['y'][V]+=1e8
   p,_=t.fold(pert,split,0,OUT/f'poison-outer-{split}');np.testing.assert_array_equal(p,saved['primary'][V])
   A,B,Pb,short,_=dd['cells'][0,0];yp=d['y'].copy();yp[B]+=1e8
   p,m=t.meta(dd['deepP'][0,0],yp[A],Pb,q[A],q[B]);orig,om=t.meta(dd['deepP'][0,0],d['y'][A],Pb,q[A],q[B]);np.testing.assert_array_equal(p,orig);assert m==om
   bp=path/'block-0';br=e.read(bp/'report.json');bz=a.readnp(bp/'predictions.npz',br['predictions_sha256']);I=np.array(br['splits'][0]['train']);H=np.array(br['splits'][0]['validation']);yp=d['y'].copy();yp[H]+=1e8;look={int(i):j for j,i in enumerate(B)};ri=yp[I]-bz['anchor'][[look[int(i)] for i in I]]
   for kind in s.KINDS:
    m=n.fit(dd['X'][I],ri,dd['blocks'],kind);old=t.joblib.load(_repo_io(bp/f'inner-0-{kind}.joblib'))['model'];np.testing.assert_array_equal(n.predict(m,dd['X'][H]),n.predict(old,dd['X'][H]))
 # Nontrivial boundary fixture exercises the constrained solver although real fits may be interior.
 H=np.array([[2.,.4],[.4,1.]]);b=np.array([3.,-2.]);theta=t.solve_gate(H,b);signs=np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
 opt=minimize(lambda v:v@H@v-2*b@v,np.zeros(2),jac=lambda v:2*(H@v-b),method='SLSQP',constraints={'type':'ineq','fun':lambda v:t.LIMIT-signs@v,'jac':lambda v:-signs},options={'ftol':1e-13,'maxiter':1000});assert opt.success;np.testing.assert_allclose(theta,opt.x,atol=1e-8,rtol=0)
 quality_diagnostics={}
 for c,name in enumerate(t.NAMES):
  flags=np.zeros(400,bool)
  for k in range(5):
   T=d['fid']!=k;V=~T;flags[V]=d['quality'][V,c]>=np.median(d['quality'][T,c])
  quality_diagnostics[name]={}
  for title,mask in [('below_train_median',~flags),('at_or_above_train_median',flags)]:
   quality_diagnostics[name][title]={'n':int(mask.sum()),'018_RMSE':e.rmse(d['y'][mask],saved['reference018'][mask]),'165_RMSE':e.rmse(d['y'][mask],saved['primary'][mask]),'mean_MSE_gain':float(np.mean((d['y'][mask]-saved['reference018'][mask])**2-(d['y'][mask]-saved['primary'][mask])**2))}
 delta_range=[min(m['delta_query_range'][0] for m in model_records),max(m['delta_query_range'][1] for m in model_records)]
 rec={'passed':True,'split':split,'models_verified':count,'meta_fits_verified':len(model_records),'independent_residual_max_error':error,'independent_meta_max_error':meta_error,'constrained_fixture_passed':True,'whole_outer_label_poison_exact':True,'anchor_B_label_poison_exact':True,'residual_inner_H_label_poison_exact':True,'outer_quality_no_train_stat_leak':True,'all_weight_vectors_positive_sum1_bounded':True,'delta_query_range_all_levels':delta_range,'effective_df_range':[min(m['effective_df_unconstrained'] for m in model_records),max(m['effective_df_unconstrained'] for m in model_records)],'quality_conditional_diagnostics':quality_diagnostics,'diagnostic_only_no_selection':'Fixed2proxies, outertrain median per fold; overlapping subgroups; no tuned cutoffs. Not causal degradation effect.','upstream_nested_training_certificate_sha256':e.sha(R/'reports/nested-selector-146-audit.json'),'report_sha256':e.sha(OUT/split/'report.json'),'seconds':time.perf_counter()-start};write= t.write;write(OUT/f'validation-{split}.json',rec);print('AUDIT_PASS',json.dumps(rec),flush=True)
if __name__=='__main__':validate(sys.argv[1])
