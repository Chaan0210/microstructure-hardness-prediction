"""Two fixed image-quality proxies, strongly shrunk sample-wise base weights; full018 nesting."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
import sys,time,json,resource,itertools
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import bootstrap_selector_144 as s
import numpy as np,torch,joblib
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
threadpool_limits(2);torch.set_num_threads(2)
a=s.a;e=s.e;n=s.n;x=s.x;OUT=_repo_path(ROOT, 'artifacts/quality-gate-165');NAMES=['noise_contrast_ratio','spectral_low_fraction'];LAMBDA=10.;LIMIT=.03

def write(p,z):
 with p.open('x') as f:json.dump(z,f,indent=2,allow_nan=False)
def quality(ids):
 p=_repo_path(ROOT, 'artifacts/image-diagnostics-094/image_metrics.npz');assert e.sha(p)==e.read(p.with_name('matching-locked.json'))['metrics_sha256']
 with np.load(_repo_io(p),allow_pickle=False) as z:
  names=z['names'].tolist();ix={str(i):j for j,i in enumerate(z['ids'])};q=z['X'][[ix[i] for i in ids]][:,[names.index(n) for n in NAMES]]
 assert q.shape==(400,2) and np.isfinite(q).all();return q

def load(split):
 d=s.data(split);d['quality']=quality(d['ids']);rec=e.read(s.OUT/split/'report.json');z=a.readnp(s.OUT/split/'predictions.npz',rec['predictions_sha256']);d['reference018']=z['primary'];np.testing.assert_array_equal(z['y'],d['y']);return d

def norm(q,center,iqr):return np.clip((q-center)/(2*iqr),-1,1)
def parts(P):
 other=(.3*P[:,1]+.2*P[:,3]+.2*P[:,4])/.7
 return a.mixture(P,x.ARM),P[:,2]-other

def solve_gate(H,b):
 u=np.linalg.solve(H,b);candidates=[]
 if abs(u).sum()<=LIMIT:candidates.append(u)
 vertices=np.array([[LIMIT,0],[0,LIMIT],[-LIMIT,0],[0,-LIMIT]])
 for v,w in zip(vertices,np.roll(vertices,-1,axis=0)):
  D=w-v;t=float(np.clip((D@b-D@H@v)/(D@H@D),0,1));candidates.append(v+t*D)
 return min(candidates,key=lambda t:float(t@H@t-2*b@t))

def meta(P,y,Q,qt,qv,enabled=True):
 p,c=parts(P);pq,cq=parts(Q);center=np.median(qt,axis=0);iqr=np.quantile(qt,.75,axis=0)-np.quantile(qt,.25,axis=0);iqr=np.where(iqr<1e-12,1.,iqr);z=norm(qt,center,iqr);zv=norm(qv,center,iqr)
 old,cal=a.affine(p,y,p);ys=float(y.std());D=z*(c*cal['slope']/ys)[:,None];target=(y-old)/ys;G=D.T@D/len(y);H=G+LAMBDA*np.eye(2);b=D.T@target/len(y);theta=solve_gate(H,b) if enabled else np.zeros(2)
 delta=z@theta;dv=zv@theta;assert np.max(abs(dv))<=LIMIT+1e-12 and abs(theta).sum()<=LIMIT+1e-12
 pred,newcal=a.affine(p+delta*c,y,pq+dv*cq)
 m={'enabled':enabled,'n_train':len(y),'quality_center':center.tolist(),'quality_iqr':iqr.tolist(),'theta':theta.tolist(),'lambda':LAMBDA,'L1_limit':LIMIT,'pre_calibrator':cal,'calibrator':newcal,'target_std':ys,'effective_df_unconstrained':float(np.trace(np.linalg.solve(H,G))),'delta_train_range':[float(delta.min()),float(delta.max())],'delta_query_range':[float(dv.min()),float(dv.max())],'H':H.tolist(),'b':b.tolist()}
 return pred,m

def system(d,B,anchor,V,k,j,split,path):
 path.mkdir();res=d['y'][B]-anchor;seed=a.r.CVSEED+100*k+j;boot=s.SEED+10000*e.SPLITS.index(split)+100*k+j;folds=list(GroupKFold(3,shuffle=True,random_state=seed).split(d['X'][B],res,d['g'][B]));ip=np.full((5,len(B)),np.nan);hashes={}
 for t,(it,iv) in enumerate(folds):
  assert not set(d['g'][B[it]])&set(d['g'][B[iv]])
  for ci,kind in enumerate(s.KINDS):
   m=n.fit(d['X'][B[it]],res[it],d['blocks'],kind);ip[ci,iv]=n.predict(m,d['X'][B[iv]]);file=path/f'inner-{t}-{kind}.joblib';joblib.dump({'model':m,'train':B[it],'validation':B[iv]},_repo_io(file));hashes[file.name]=e.sha(file)
 w,draw,wins=s.weights((ip-res)**2,d['g'][B],boot);pp=[]
 for kind in s.KINDS:
  m=n.fit(d['X'][B],res,d['blocks'],kind);pp.append(n.predict(m,d['X'][V]));file=path/f'full-{kind}.joblib';joblib.dump({'model':m,'train':B},_repo_io(file));hashes[file.name]=e.sha(file)
 pp=np.stack(pp);pred=w@pp
 np.savez_compressed(_repo_io(path/'predictions.npz'),B=B,V=V,anchor=anchor,residual=res,inner=ip,weights=w,draws=draw,wins=wins,full=pp,primary=pred)
 write(path/'report.json',{'foldseed':seed,'bootseed':boot,'splits':[{'train':B[it].tolist(),'validation':B[iv].tolist()} for it,iv in folds],'models_sha256':hashes,'predictions_sha256':e.sha(path/'predictions.npz')});return pred

def fold(d,split,k,path,enabled=True):
 path.mkdir(parents=True,exist_ok=False);start=time.perf_counter();cpu=time.process_time();T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);Q=d['quality']
 cp,om=meta(d['middleP'][k],d['y'][T],d['P'][V],Q[T],Q[V],enabled);base=.9*cp+.1*d['short'][V];cor=[];models=[]
 for j in range(3):
  A,B,P,short,_=d['cells'][k,j];assert not set(A)&set(B) and not(set(A)|set(B))&set(V)
  cp,m=meta(d['deepP'][k,j],d['y'][A],P,Q[A],Q[B],enabled);anchor=.9*cp+.1*short;cor.append(system(d,B,anchor,V,k,j,split,path/f'block-{j}'));m['train']=A.tolist();m['query']=B.tolist();models.append(m)
 pred=base+np.mean(cor,0);assert np.isfinite(pred).all();np.savez_compressed(_repo_io(path/'predictions.npz'),V=V,base=base,corrections=np.stack(cor),primary=pred)
 om['train']=T.tolist();om['query']=V.tolist();r={'split':split,'fold':k,'enabled':enabled,'outer_meta':om,'anchor_meta':models,'predictions_sha256':e.sha(path/'predictions.npz'),'block_reports':{f'block-{j}/report.json':e.sha(path/f'block-{j}/report.json') for j in range(3)},'seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'RMSE':e.rmse(d['y'][V],pred),'reference018_RMSE':e.rmse(d['y'][V],d['reference018'][V])};write(path/'report.json',r);print('FOLD_DONE',split,k,r['RMSE'],r['reference018_RMSE'],r['seconds'],r['peak_rss_bytes'],flush=True);return pred,r

def prepare():
 s.check();OUT.mkdir(exist_ok=False)
 summary=e.read(_repo_path(ROOT, 'reports/candidate018-summary.json'));fit=e.read(_repo_path(ROOT, 'artifacts/final-model-145/fit.json'));assert fit==summary['full500_fit'] and fit['rows']==len(set(fit['train_ids']))==500
 for block in fit['blocks']:
  for name,h in block['models_sha256'].items():assert e.sha(_repo_path(ROOT, 'artifacts/final-model-145', f"block-{block['block']}", name))==h
 keep=[_repo_path(ROOT, f'artifacts/submission-candidate-{v}.csv') for v in ['018','019','020']]+[_repo_path(ROOT, 'deliverables/reproduction-018-v155.zip'),_repo_path(ROOT, 'deliverables/reproduction-011-014-015-v136.zip'),_repo_path(ROOT, 'reports/candidate018-summary.json'),_repo_path(ROOT, 'artifacts/final-model-145/fit.json')]
 names=['quality_gate_165.py','bootstrap_selector_144.py','cnn_seed_mean_133.py','component_ablation_121.py','nonlinear_residual_115.py','artifacts/image-diagnostics-094/image_metrics.npz','artifacts/image-diagnostics-094/matching-locked.json','reports/nested-selector-146-audit.json','reports/post009-research-076-078-summary.json','reports/baseline124-weights125-summary.json','reports/domain-risk-audit-126-summary.json']
 plan={'study':165,'novelty':{'077':'Unregularized global legacy-vsDINO contrast, no image-quality interaction. Failed resplits.','078':'Training-size reliability diagnostic; no quality gating fit.','125':'Sample-independent simplex base weights with fixedridge penalty; failed pooledMSE.','126':'Image-condition risk diagnostic of013→014 removal, no predictions or weights changed; no adverse-condition transfer.','165':'Two predefined image-quality interactions change CNN-vsremaining base mixture samplewise with strong shrinkage; complete018 downstream refit. Not a claim that image risk diagnostics showed a positive effect.'},'quality':NAMES,'quality_meaning':'Noise/contrast proxy and lowfrequency power fraction below1/32 cycles/pixel. Latter is blur/coarse-structure proxy, not identified blur or pure noise. No physical sign assumptions.','normalization':'Current meta training-row median/IQR only. z=clip((q-median)/(2*IQR),-1,1); IQR<1e-12→1. No feature/threshold sweep.','gate':'delta=z@theta; ||theta||1<=.03. weightsCNN=.3+delta, KRR=.3*(1-delta/.7), S=B=.2*(1-delta/.7). Sum1, positive. ShortB.1 unchanged. No constant/global weight-shift term.','objective':'First fixedbase affine fit on currentOOF train. LetD=z*(CNN-other)*affine_slope/std(y), r=(y-affinebase)/std(y). Min mean((r-Dtheta)^2)+10||theta||² withL1ball .03. Exact2D convex quadratic solve. Then refit usual affine to adjusted rawmixture.','lambda':LAMBDA,'limit':LIMIT,'nested':'Outer gate/affine trained only on320 middleOOF rows from213/214 models; eachA gate/affine trainedonly on213/214 deepOOF from142/143 models. A-only anchors predict honestB106/107. Refit all018 residual585 candidates inB inner3fold,512bootstrap thenfullB; mean3 corrections. Global400OOF never used as fit input/target. All source lineage verified by146 and133 caches, no new neural fits.','control':'theta0 whole firstfold must reproduce018 within1e-9 including bootstrap residuals.','gate_to_resplits':'OriginalRMSE strictlybelow018 and>=3of5fold improvements after independent validation. Frozen165 only then run20261031/20261101.','final_assessment':'If extended, pooleduntrimmedMSE overexisting3partitions below018 with honest audits. Reportall partitions and paired ID-cluster conditionalCI; no sweep or automatic resubmission.','resource':'CPU2 existing. Firstwholefold<120s,peakRSS<4e9. Existingpackage environment. No rawimages/holdout/test/download/API/cost.','full500_already_completed':{'rows':500,'train_ids_unique':500,'018_residual_model_hashes_verified':15,'upstream017_reused':True,'fit_sha256':e.sha(_repo_path(ROOT, 'artifacts/final-model-145/fit.json')),'csv_sha256':e.sha(_repo_path(ROOT, 'artifacts/submission-candidate-018.csv')),'no_duplicate_training':True},'hashes':{v:e.sha(_repo_path(ROOT, v)) for v in names},'protected':{str(p.relative_to(ROOT)):e.sha(p) for p in keep},'limitations':['Same400 adaptive studies, no independent validation','Deep142/213/320 training-size mismatch persists','Quality proxies conflate signal and image degradation','Gate is one fixedCNN-vsremaining axis, not exhaustive component weighting']};write(_repo_path(OUT, 'plan.json'),plan);print('PLAN_LOCKED',e.sha(_repo_path(OUT, 'plan.json')),flush=True)

def run(split,benchmark=False):
 plan=e.read(_repo_path(OUT, 'plan.json'))
 for f,h in {**plan['hashes'],**plan['protected']}.items():assert e.sha(_repo_path(ROOT, f))==h,f
 if split!='original':assert e.read(_repo_path(OUT, 'original/report.json'))['advance'] and e.read(_repo_path(OUT, 'validation-original.json'))['passed']
 d=load(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=True);pred=np.full(400,np.nan)
 for k in ([0] if benchmark else range(5)):
  dd,_=x.replace(d,split,k,x.OUT/split/f'cnn-fold-{k}',False);V=np.flatnonzero(d['fid']==k);path=out/f'fold-{k}'
  if (path/'report.json').exists():r=e.read(path/'report.json');p=a.readnp(path/'predictions.npz',r['predictions_sha256'])['primary']
  else:
   if split=='original' and k==0:
    control,_=fold(dd,split,k,_repo_path(OUT, 'control018-fold-0'),False);np.testing.assert_allclose(control,d['reference018'][V],atol=1e-9,rtol=0)
   print('START',split,k,flush=True);p,r=fold(dd,split,k,path)
  pred[V]=p;assert r['seconds']<120 and r['peak_rss_bytes']<4e9,'Resource stop'
 if benchmark:return
 gain=[e.rmse(d['y'][d['fid']==k],d['reference018'][d['fid']==k])-e.rmse(d['y'][d['fid']==k],pred[d['fid']==k]) for k in range(5)];score=e.rmse(d['y'],pred);ref=e.rmse(d['y'],d['reference018']);np.savez_compressed(_repo_io(out/'predictions.npz'),ids=np.array(d['ids']),groups=d['g'],y=d['y'],fold=d['fid'],quality=d['quality'],reference018=d['reference018'],primary=pred)
 rec={'split':split,'RMSE':{'reference018':ref,'primary':score},'fold_gains':gain,'advance':score<ref and sum(g>0 for g in gain)>=3,'predictions_sha256':e.sha(out/'predictions.npz'),'fold_reports':{f'fold-{k}/report.json':e.sha(out/f'fold-{k}/report.json') for k in range(5)},'read_input_sha256':dict(a.SOURCES)};write(out/'report.json',rec);print('RESULT',json.dumps({k:v for k,v in rec.items() if k not in ['read_input_sha256','fold_reports']}),flush=True)
if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='prepare':prepare()
 elif mode=='benchmark':run('original',True)
 else:run(mode)
