"""Independent convex-QP, scope, nuisance-LOO and scalar assembly checks."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import engine as e
from fit_fusion import cohorts
import sys,json,time,itertools
from types import SimpleNamespace
import numpy as np
from threadpoolctl import threadpool_limits

def active_set_reference(R,residual,beta,lam):
 n=len(R);theta0=np.array([beta,0,0,0,0.]);eps=1e-8*max(float(np.mean(R**2)),1.);H=(1+lam)*(R.T@R)/n+eps*np.eye(5);b=R.T@residual/n;lower=np.array([beta/2,0,0,0,0.]);G=np.r_[-np.eye(5),[[0,0,0,0,1],[1,1,1,1,0]]];h=np.r_[theta0-lower,.5,1-beta];best=None
 for m in range(6):
  for active in itertools.combinations(range(7),m):
   A=G[list(active)];v=h[list(active)];K=np.block([[H,A.T],[A,np.zeros((m,m))]])
   try:sol=np.linalg.solve(K,np.r_[b,v])
   except np.linalg.LinAlgError:continue
   w=sol[:5];dual=sol[5:]
   if np.max(G@w-h)>1e-7 or np.min(dual,initial=0)<-1e-7:continue
   obj=float(w@H@w-2*b@w)
   if best is None or obj<best[0]:best=(obj,w,active)
 assert best is not None
 return theta0+best[1],best[0],H,b

def main():
 start=time.monotonic();qp=[];replay=[]
 for k in range(5):
  B,V,ix,base,beta=cohorts(k);hi=e.high(B);R=B['R'][hi];residual=(B['y']-B['p76']-beta*B['R'][:,0])[hi]
  for lam in [1,10,100]:
   model=e.fit(B,'global',lam,beta);theta=np.array(model['theta'])[:,0];ref,obj,H,b=active_set_reference(R,residual,beta,lam);w=theta-np.array([beta,0,0,0,0]);gap=float(w@H@w-2*b@w-obj);pd=float(np.max(abs(R@(theta-ref))));qd=float(np.max(abs(V['R']@(theta-ref))));assert abs(gap)<1e-7 and pd<1e-3 and qd<1e-3,(k,lam,gap,pd,qd);qp.append(dict(fold=k,lam=lam,objective_gap=gap,train_prediction_maxdiff=pd,query_prediction_maxdiff=qd))
  saved=json.loads((c.HERE/f'fold{k}-selection.json').read_text());p=np.load(_repo_io(c.HERE/f'fold{k}-predictions.npz'))
  for name,j in saved['selected'].items():
   model=saved['models'][str(j)];pred=e.predict(model,V);assert np.array_equal(pred,p[name]);assert np.array_equal(pred[~e.high(V)],base[~e.high(V)]);replay.append(dict(fold=k,arm=name,exact=True))
  assert abs(e.incumbent_beta(B)-beta)<1e-12
 # Real loader, with all external labels and query features replaced in a separate file.
 k=0;B,V,ix,base,beta=cohorts(k);original=np.load(_repo_io(c.HERE/'inputs/fold0.npz'));poison={name:original[name] for name in original.files};poison['yV']=np.full_like(poison['yV'],999999.);poison['RV']=poison['RV']+1234.;poison['qualityV']=poison['qualityV']-4321.;pdir=c.HERE/'verification/poison';c.arrays('verification/poison/inputs/fold0.npz',**poison)
 loader=c.definitions(c.HERE/'fit_fusion.py',['cohorts'],dict(np=np,c=SimpleNamespace(HERE=pdir)))['cohorts'];Bp,Vp,*_=loader(0);assert 'y' not in Vp;actual=e.select_all(Bp);saved=json.loads((c.HERE/'fold0-selection.json').read_text());assert actual==saved,'Whole selection changed with external data'
 # A held-out B label must be absent from beta, normalizers, coefficients and prediction.
 held=int(np.flatnonzero(e.high(B))[0]);held_id=str(B['ids'][held]);changed={name:arr.copy(order='K') for name,arr in B.items()};changed['y'][held]+=999999.;mask=np.arange(len(B['y']))!=held;train=e.subset(B,mask);trainp=e.subset(changed,mask);query=e.subset(B,np.array([held]),labels=False);loo=[]
 for kind,lam in e.CONFIGS:
  a=e.fit(train,kind,lam);b=e.fit(trainp,kind,lam);assert a==b;assert np.array_equal(e.predict(a,query),e.predict(b,query));loo.append([kind,lam])
 # Independently reconstruct actual query coefficients and every corner constraint.
 corner=[]
 for k in range(5):
  B,V,ix,base,beta=cohorts(k);s=json.loads((c.HERE/f'fold{k}-selection.json').read_text());pred=np.load(_repo_io(c.HERE/f'fold{k}-predictions.npz'));h=e.high(V)
  for key,model in s['models'].items():
   if model['kind'] not in ['shared','global','conditional']:continue
   theta=np.array(model['theta']);u=np.c_[np.ones(h.sum()),np.clip((V['quality'][h]-model['quality_mean'])/model['quality_scale'],-2,2)] if theta.shape[1]==3 else np.ones((h.sum(),1));coef=u@theta.T;assert np.min(coef)>=-1e-7 and np.all(coef[:,0]>=beta/2-1e-7) and np.all(coef[:,:min(len(theta),4)].sum(1)<=1+1e-7)
   if len(theta)==5:assert np.max(coef[:,4])<=.5+1e-7
   p=V['p76'].copy();p[h]+=(V['R'][h,:len(theta)]*coef).sum(1);mine=e.predict(model,V);assert np.max(abs(p-mine))<1e-12;corner.append(dict(fold=k,model=key,query_constraints=True))
 z=np.load(_repo_io(c.HERE/'oof.npz'));r=json.loads((c.HERE/'results.json').read_text());metrics=[]
 for name,rr in r['results'].items():
  p=z[name];y=z['y'];b=z['base082'];mean=sum((float(a)-float(t))**2 for a,t in zip(p,y))/400;assert abs(mean**.5-rr['overall_RMSE'])<1e-12;order=sorted(range(400),key=lambda i:(float(b[i])-float(y[i]))**2-(float(p[i])-float(y[i]))**2,reverse=True)[10:];gb=(sum((float(b[i])-float(y[i]))**2 for i in order)/390)**.5;gp=(sum((float(p[i])-float(y[i]))**2 for i in order)/390)**.5;assert abs(gb-gp-rr['removed_top']['10'])<1e-12;metrics.append(name)
 c.save(_repo_io('verification.json'),dict(passed=True,QP_global_independent_active_set=qp,artifact_replays=replay,whole_nested_selection_external_label_feature_poison_exact=True,held_B_source_id=held_id,LOO_nuisance_label_exclusion_exact=loo,query_constraints=corner,scalar_metrics_checked=metrics,seconds=time.monotonic()-start,engine_sha256=c.sha(c.HERE/'engine.py')));print('ALL VERIFICATION PASSED',len(qp),len(replay),flush=True)
if __name__=='__main__':
 sys.addaudithook(c.guard)
 with threadpool_limits(limits=2):main()
