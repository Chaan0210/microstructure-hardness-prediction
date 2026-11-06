from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import heterogeneity_calibration_167 as h
import validate_quality_gate_165 as old
import numpy as np,sys,time
s=h.s;t=h.t;a=h.a;e=h.e;x=h.x;n=t.n;OUT=h.OUT;R=h.R

def manual(P,y,Q,qt,qv,m):
 p=a.mixture(P,x.ARM);pq=a.mixture(Q,x.ARM);pm=p.mean();ps=max(float(p.std()),1e-12);qm=qt.mean();qs=max(float(qt.std()),1e-12)
 for name,value in [('p_mean',pm),('p_std',ps),('q_mean',qm),('q_std',qs)]:assert m[name]==value
 z=(p-pm)/ps;zv=(pq-pm)/ps;q=(qt[:,0]-qm)/qs;qv=(qv[:,0]-qm)/qs
 if m['basis']=='interaction':Z=np.column_stack([np.ones(len(z)),z,q]);ZV=np.column_stack([np.ones(len(zv)),zv,qv]);H=z*q;HV=zv*qv
 else:Z=np.column_stack([np.ones(len(z)),z]);ZV=np.column_stack([np.ones(len(zv)),zv]);H=z*z;HV=zv*zv
 b=np.linalg.lstsq(Z,H,rcond=None)[0];np.testing.assert_array_equal(b,m['projection']);basis=H-Z@b;query=HV-ZV@b;scale=np.sqrt(np.mean(basis*basis))
 if scale<1e-12:basis=np.zeros_like(basis);query=np.zeros_like(query)
 else:basis/=scale;query/=scale
 assert np.max(abs(Z.T@basis/len(y)))<1e-10
 # Joint augmented least squares independently yields unpenalized intercept/slope plus penalized extra term.
 D=np.column_stack([np.ones(len(y)),z,basis]);reg=np.array([[0,0,np.sqrt(len(y)*h.LAMBDA)]]);coef=np.linalg.lstsq(np.vstack([D,reg]),np.r_[y,0.],rcond=None)[0]
 if not m['enabled']:coef[-1]=0
 np.testing.assert_allclose(coef[-1],m['coefficient'],atol=1e-10,rtol=0)
 pred=np.column_stack([np.ones(len(zv)),zv,query])@coef
 h.MODE=m['basis'];actual,re=h.meta(P,y,Q,qt,(qv*qs+qm)[:,None],m['enabled'])
 # Undoing the query normalization above is for predictions only; original stored fits do not depend on query.
 for name in ['p_mean','p_std','q_mean','q_std','projection','basis_scale','coefficient','calibrator']:assert re[name]==m[name]
 err=float(np.max(abs(actual-pred)));assert err<1e-8;return pred,err

def validate(split):
 start=time.perf_counter();plan=e.read(OUT/'plan.json')
 for f,z in {**plan['source_hashes'],**plan['protected']}.items():assert e.sha(R/f)==z
 rr=e.read(OUT/split/'report.json')
 for f,z in rr['read_input_sha256'].items():assert e.sha(R/f)==z
 d=t.load(split);saved=a.readnp(OUT/split/'predictions.npz',rr['predictions_sha256']);assert saved['ids'].tolist()==d['ids'];np.testing.assert_array_equal(saved['reference018'],d['reference018']);results={}
 for arm in ['interaction','quadratic']:
  h.MODE=arm;count=0;metaerr=0.;errmax=0.;records=[]
  for k in range(5):
   dd,_=x.replace(d,split,k,x.OUT/split/f'cnn-fold-{k}',False);T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);q=d['quality'];path=OUT/split/f'{arm}-fold-{k}';r=e.read(path/'report.json');assert e.sha(path/'report.json')==rr['fold_reports'][f'{arm}-fold-{k}/report.json'];z=a.readnp(path/'predictions.npz',r['predictions_sha256'])
   np.testing.assert_array_equal(r['outer_meta']['train'],T);np.testing.assert_array_equal(r['outer_meta']['query'],V)
   pp,err=manual(dd['middleP'][k],d['y'][T],dd['P'][V],q[T],q[V],r['outer_meta']);metaerr=max(metaerr,err);np.testing.assert_allclose(.9*pp+.1*d['short'][V],z['base'],atol=1e-8,rtol=0);records.append(r['outer_meta']);cor=[]
   for j in range(3):
    A,B,P,short,_=dd['cells'][k,j];assert not set(A)&set(B) and set(A)|set(B)==set(T) and not(set(A)|set(B))&set(V)
    m=r['anchor_meta'][j];np.testing.assert_array_equal(m['train'],A);np.testing.assert_array_equal(m['query'],B);pp,err=manual(dd['deepP'][k,j],d['y'][A],P,q[A],q[B],m);metaerr=max(metaerr,err);records.append(m)
    bp=path/f'block-{j}';assert e.sha(bp/'report.json')==r['block_reports'][f'block-{j}/report.json'];pc,c,err=old.verify_block(dd,B,V,.9*pp+.1*short,bp);cor.append(pc);count+=c;errmax=max(errmax,err)
   np.testing.assert_array_equal(z['base']+np.mean(cor,0),saved[arm][V]);print('VERIFIED',split,arm,k,flush=True)
   if k==0:
    _,changed=h.meta(dd['middleP'][k],d['y'][T],dd['P'][V],q[T],q[V]+1000)
    for name in ['coefficient','projection','p_mean','p_std','q_mean','q_std','calibrator']:assert changed[name]==r['outer_meta'][name]
    pert=dict(dd);pert['y']=d['y'].copy();pert['y'][V]+=1e8;pp,_=t.fold(pert,split,k,OUT/f'poison-{split}-{arm}');np.testing.assert_array_equal(pp,saved[arm][V])
    A,B,P,short,_=dd['cells'][0,0];py=d['y'].copy();py[B]+=1e8;p1,m1=h.meta(dd['deepP'][0,0],py[A],P,q[A],q[B]);p2,m2=h.meta(dd['deepP'][0,0],d['y'][A],P,q[A],q[B]);np.testing.assert_array_equal(p1,p2);assert m1==m2
    bp=path/'block-0';br=e.read(bp/'report.json');bz=a.readnp(bp/'predictions.npz',br['predictions_sha256']);I=np.array(br['splits'][0]['train']);H=np.array(br['splits'][0]['validation']);py=d['y'].copy();py[H]+=1e8;lookup={int(v):i for i,v in enumerate(B)};target=py[I]-bz['anchor'][[lookup[int(v)] for v in I]]
    for kind in s.KINDS:
     m=n.fit(dd['X'][I],target,dd['blocks'],kind);om=t.joblib.load(_repo_io(bp/f'inner-0-{kind}.joblib'))['model'];np.testing.assert_array_equal(n.predict(m,dd['X'][H]),n.predict(om,dd['X'][H]))
  results[arm]={'residual_models_verified':count,'meta_models_verified':len(records),'independent_residual_max_error':errmax,'independent_joint_LS_max_error':metaerr,'outer_label_poison_exact':True,'anchor_B_label_poison_exact':True,'inner_H_label_poison_exact':True,'query_quality_cannot_change_fit':True,'coefficient_range':[min(m['coefficient'] for m in records),max(m['coefficient'] for m in records)],'query_correction_range':[min(m['query_correction_range'][0] for m in records),max(m['query_correction_range'][1] for m in records)]}
 r={'passed':True,'split':split,'arms':results,'report_sha256':e.sha(OUT/split/'report.json'),'upstream_nested_certificate_sha256':e.sha(R/'reports/nested-selector-146-audit.json'),'seconds':time.perf_counter()-start};t.write(OUT/f'validation-{split}.json',r);print('AUDIT_PASS',r,flush=True)
if __name__=='__main__':validate(sys.argv[1])
