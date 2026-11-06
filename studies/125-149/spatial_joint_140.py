"""Fixed projected ViT-B graph energy versus unordered covariance on honest015 anchors."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
import sys,time,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import quadratic_calibration_137 as prior
import validate_nonlinear_residual_115 as vn
import numpy as np,torch,joblib
from sklearn.model_selection import GroupKFold
x=prior.x;a=x.a;e=x.e;OUT=_repo_path(ROOT, 'artifacts/spatial-joint-140');ARMS=('unordered','adjacent');SEED=20271201

def stats(z):
 c=z.reshape(-1,32);c=c-c.mean(0);g=c.T@c/len(c)
 h=np.diff(z,axis=0).reshape(-1,32);v=np.diff(z,axis=1).reshape(-1,32);s=(h.T@h+v.T@v)/(2*(len(h)+len(v)))
 ix=np.triu_indices(32);return g[ix],s[ix]

def check():
 p=e.read(_repo_path(OUT, 'plan.json'))
 for f,h in p['hashes'].items():assert e.sha(_repo_path(ROOT, f))==h,f
 return p

def prepare():
 x.check();OUT.mkdir(exist_ok=False)
 names=['spatial_joint_140.py','quadratic_calibration_137.py','cnn_seed_mean_133.py','component_ablation_121.py','nonlinear_residual_115.py','validate_nonlinear_residual_115.py','artifacts/patch-attention-092/cache.json']
 names += [f'artifacts/submission-candidate-{c}.csv' for c in ['011','014','015','016']]
 for split in e.SPLITS:names += [f'artifacts/cnn-seed-mean-133/{split}/{f}' for f in ['report.json','predictions.npz']]
 p={'study':'140 projected joint-channel spatial graph energy','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hypothesis':'Cross-channel co-change across neighboring frozenB tokens may complement015 physics features beyond unordered within-image covariance. Not evidence of true physical grain boundaries.','representation':'Existing float32 normalized finalB tokens36x36x768; fixed Gaussian QR768x32 seed20271201, no labels/pooled fit. Unordered control population covariance; primary half mean outerproduct of all horizontal/vertical nearest-neighbor differences,2520 edges, no wrap. Uppertriangle528 including offdiagonal. Add bank tophysics57 =>585; one new block normalized1/sqrt528.','duplicate_audit':{'042':'S channelwise variograms lag1/4, no offdiagonal jointchannels','056':'Unordered covariance, no adjacency','070':'Coarse regional variance','092':'Learned attention, no explicit local jointdifference','047':'DINO-morphology productkernel, not this fixedspatialbank'},'nested':'Exactly015 affine frommiddleOOF and honestB anchors fromdeepA-only OOF; replace physics57 bank by physics57+newbank. Each arm separately select unchanged none/ridge1/ridge01/rbf1/rbf01 inside eachB3fold; train-onlyMinMaxclip,StandardScaler,blockweight,kernelcenter/residualRMS. No fitting on015globalOOF residuals; no secondstage residual stacking.','primary':'Run all3fixed partitions. Candidate onlyif adjacent pooleduntrimmedMSE strictlybelow BOTH unchanged015 and unordered control AND independentvalidation passes. Unordered is control only, no fallback. No foldwin/CI gate, no tuningseed/rank/lag/weights.','validation':'Feature explicit edge enumeration onfixtures and3 cachedimages; token-grid rotation/reflection invariance only, not entireDINO pipeline. Reload every residualfit; independent rawscaling/eigensolve; OLS; index/groupchecks; outer-target poison replay andinnerheldoutlabel exclusion. Exact unchanged015 reconstruction.','uncertainty':'10000 pairedIDcluster bootstrapseed20271202; threepartitionsreuse400IDs, conditionalonfixedfits andhistoricaladaptivework.','resource':'ExistingCPU2,noneuralfits,90honestblocks1440models; featurecache1.6GBread sequentially; no holdout/test/full500 beforepositivelockeddecision. Noinstall/cost/upload.','hashes':{f:e.sha(_repo_path(ROOT, f)) for f in names},'cost':0};e.write(_repo_path(OUT, 'plan.json'),p);e.emit({'plan_locked':True,'sha256':e.sha(_repo_path(OUT, 'plan.json'))})

def extract():
 check();start=time.perf_counter();meta=e.read(_repo_path(ROOT, 'artifacts/patch-attention-092/cache.json'));f=_repo_path(ROOT, 'artifacts/patch-attention-092/patch_tokens.npy');assert e.sha(f)==meta['token_sha256'];tok=np.load(_repo_io(f),mmap_mode='r');assert list(tok.shape)==meta['shape']==[400,1296,768]
 P=np.linalg.qr(np.random.default_rng(SEED).normal(size=(768,32)),mode='reduced')[0];features={arm:[] for arm in ARMS};error=0.
 for i,t in enumerate(tok):
  z=(np.asarray(t,dtype=float)@P).reshape(36,36,32);g,s=stats(z)
  if i in [0,199,399]:
   dif=np.array([z[r,c]-z[rr,cc] for r in range(36) for c in range(36) for rr,cc in [(r+1,c),(r,c+1)] if rr<36 and cc<36]);manual=np.einsum('ni,nj->ij',dif,dif)/(2*len(dif));error=max(error,float(abs(manual[np.triu_indices(32)]-s).max()));assert error<1e-12
   for zz in [np.rot90(z),z[:,::-1]]:
    gg,ss=stats(zz);np.testing.assert_allclose(gg,g,atol=1e-12,rtol=0);np.testing.assert_allclose(ss,s,atol=1e-12,rtol=0)
  features['unordered'].append(g);features['adjacent'].append(s)
  if i%100==0:e.emit({'features_complete':i})
 with (_repo_path(OUT, 'features.npz')).open('xb') as h:np.savez_compressed(_repo_io(h),ids=np.array(meta['ids']),projection=P,**{k:np.array(v) for k,v in features.items()})
 r={'passed':True,'features_sha256':e.sha(_repo_path(OUT, 'features.npz')),'token_sha256':meta['token_sha256'],'shape':[400,528],'explicit_edge_max_error':error,'seconds':time.perf_counter()-start};e.write(_repo_path(OUT, 'extraction.json'),r);e.emit(r)

def load(split):
 d=prior.data(split);r=e.read(_repo_path(OUT, 'extraction.json'));z=a.readnp(_repo_path(OUT, 'features.npz'),r['features_sha256']);assert z['ids'].tolist()==d['ids'];return d,z

def augmented(d,z,arm):
 dd=dict(d);dd['X']=np.column_stack([d['X'],z[arm]]);dd['blocks']=list(d['blocks'])+['joint_token']*528;return dd

def run(split):
 check();d,z=load(split);out=_repo_path(OUT, split);out.mkdir();start=time.perf_counter();pred={arm:np.full(400,np.nan) for arm in ARMS}
 for k in range(5):
  dd,_=x.replace(d,split,k,x.OUT/split/f'cnn-fold-{k}',False);V=np.flatnonzero(d['fid']==k)
  if k==0:
   control,_,_=a.fold(dd,k,x.ARM,out/'control015-fold-0');np.testing.assert_allclose(control,d['reference015'][V],atol=1e-10,rtol=0)
  for arm in ARMS:pred[arm][V],_,_=a.fold(augmented(dd,z,arm),k,x.ARM,out/f'{arm}-fold-{k}')
  e.emit({'split':split,'fold_complete':k,'seconds':time.perf_counter()-start})
 with (out/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],reference015=d['reference015'],**pred)
 rec={'split':split,'RMSE':{'reference015':e.rmse(d['y'],d['reference015']),**{arm:e.rmse(d['y'],p) for arm,p in pred.items()}},'predictions_sha256':e.sha(out/'predictions.npz'),'fold_hashes':{f'{arm}-fold-{k}/report.json':e.sha(out/f'{arm}-fold-{k}/report.json') for arm in ARMS for k in range(5)},'seconds':time.perf_counter()-start};e.write(out/'report.json',rec);check();e.emit(rec)

def independent_affine(p,y,v):
 return np.column_stack([np.ones(len(v)),v])@np.linalg.lstsq(np.column_stack([np.ones(len(p)),p]),y,rcond=None)[0]

def validate(split):
 check();d,z=load(split);out=_repo_path(OUT, split);rr=e.read(out/'report.json');saved=a.readnp(out/'predictions.npz',rr['predictions_sha256']);assert saved['ids'].tolist()==d['ids'];np.testing.assert_array_equal(saved['y'],d['y']);maxerr=0.;count=0
 for k in range(5):
  dd,_=x.replace(d,split,k,x.OUT/split/f'cnn-fold-{k}',False);T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);base=.9*independent_affine(a.mixture(dd['middleP'][k],x.ARM),d['y'][T],a.mixture(dd['P'][V],x.ARM))+.1*dd['short'][V]
  for arm in ARMS:
   D=augmented(dd,z,arm);root=out/f'{arm}-fold-{k}';fr=e.read(root/'report.json');assert e.sha(root/'report.json')==rr['fold_hashes'][f'{arm}-fold-{k}/report.json'];fz=a.readnp(root/'predictions.npz',fr['predictions_sha256']);np.testing.assert_allclose(base,fz['base'],atol=1e-9,rtol=0);corr=[]
   for j in range(3):
    A,B,P,short,_=dd['cells'][k,j];assert not set(A)&set(B) and not (set(A)|set(B))&set(V);assert not set(d['g'][T])&set(d['g'][V]);anchor=.9*independent_affine(a.mixture(dd['deepP'][k,j],x.ARM),d['y'][A],a.mixture(P,x.ARM))+.1*short;bp=root/f'block-{j}';br=e.read(bp/'report.json');assert e.sha(bp/'report.json')==fr['block_reports_sha256'][f'block-{j}/report.json'];bz=a.readnp(bp/'predictions.npz',br['predictions_sha256']);np.testing.assert_allclose(anchor,bz['anchor_predictions'],atol=1e-9,rtol=0);res=d['y'][B]-bz['anchor_predictions'];np.testing.assert_array_equal(res,bz['residual']);ip=np.empty_like(bz['inner_predictions'])
    folds=list(GroupKFold(3,shuffle=True,random_state=a.r.CVSEED+100*k+j).split(D['X'][B],res,d['g'][B]))
    for t,(it,iv) in enumerate(folds):
     assert br['splits'][t]=={'train_indices':B[it].tolist(),'validation_indices':B[iv].tolist()};assert not set(d['g'][B[it]])&set(d['g'][B[iv]])
     for ci,kind in enumerate(a.n.KINDS):
      f=bp/f'inner-{t}-{kind}.joblib';assert e.sha(f)==br['models_sha256'][f.name];m=joblib.load(_repo_io(f));np.testing.assert_array_equal(m['train_indices'],B[it]);np.testing.assert_array_equal(m['validation_indices'],B[iv]);actual=a.n.predict(m['model'],D['X'][B[iv]]);manual=vn.manual(m['model'],D['X'][B[it]],res[it],D['X'][B[iv]],D['blocks']);err=float(abs(actual-manual).max());assert err<2e-9;maxerr=max(maxerr,err);np.testing.assert_array_equal(actual,bz['inner_predictions'][ci,iv]);ip[ci,iv]=manual;count+=1
    scores=np.mean((ip-res)**2,axis=1);np.testing.assert_allclose(scores,list(br['inner_MSE'].values()),atol=2e-9,rtol=0);assert a.n.KINDS[int(scores.argmin())]==br['chosen'];f=bp/'selected.joblib';assert e.sha(f)==br['models_sha256'][f.name];m=joblib.load(_repo_io(f));np.testing.assert_array_equal(m['train_indices'],B);actual=a.n.predict(m['model'],D['X'][V]);manual=vn.manual(m['model'],D['X'][B],res,D['X'][V],D['blocks']);err=float(abs(actual-manual).max());assert err<2e-9;maxerr=max(maxerr,err);np.testing.assert_array_equal(actual,bz['correction']);corr.append(actual);count+=1
   np.testing.assert_array_equal(fz['base']+np.mean(corr,0),saved[arm][V])
 # Replay new stages with outer labels poisoned; same upstream exclusion certificates reused.
 dd,_=x.replace(d,split,0,x.OUT/split/'cnn-fold-0',False);V=np.flatnonzero(d['fid']==0);dd['y']=dd['y'].copy();dd['y'][V]+=10000
 for arm in ARMS:
  p,_,_=a.fold(augmented(dd,z,arm),0,x.ARM,_repo_path(OUT, f'poison-{split}-{arm}'));np.testing.assert_array_equal(p,saved[arm][V])
 rec={'passed':True,'split':split,'models_verified':count,'independent_max_error':maxerr,'outer_target_poison_exact':True,'inner_fit_indices_and_groups_verified':True,'report_sha256':e.sha(out/'report.json')};e.write(_repo_path(OUT, f'validation-{split}.json'),rec);check();e.emit(rec)

def summarize():
 check();errors={k:[] for k in ['reference015',*ARMS]};scores={}
 for split in e.SPLITS:
  r=e.read(_repo_path(OUT, split, 'report.json'));v=e.read(_repo_path(OUT, f'validation-{split}.json'));assert v['passed'] and v['report_sha256']==e.sha(_repo_path(OUT, split, 'report.json'));z=a.readnp(_repo_path(OUT, split, 'predictions.npz'),r['predictions_sha256']);scores[split]=r['RMSE']
  for arm in errors:errors[arm].append((z[arm]-z['y'])**2)
 errors={k:np.array(v) for k,v in errors.items()};mse={k:float(v.mean()) for k,v in errors.items()};advance=mse['adjacent']<min(mse['reference015'],mse['unordered']);rng=np.random.default_rng(20271202);boot={k:[] for k in ['reference015','unordered']}
 for _ in range(10000):
  ids=rng.integers(0,400,400)
  for k in boot:boot[k].append(float(np.sqrt(errors[k][:,ids].mean())-np.sqrt(errors['adjacent'][:,ids].mean())))
 r={'passed_validation':True,'advance':advance,'RMSE_by_partition':scores,'pooled_MSE':mse,'pooled_RMSE':{k:float(np.sqrt(v)) for k,v in mse.items()},'conditional_400IDcluster_gain_CI95':{k:np.quantile(v,[.025,.975]).tolist() for k,v in boot.items()},'current_best':'015','Public016':11.5828840997,'public_not_used_to_tune_features':True,'new_neural_fits':0,'cost':0,'test_or_holdout_read':False,'new_CSV':False,'closed_if_negative':True,'plan_sha256':e.sha(_repo_path(OUT, 'plan.json'))};e.write(_repo_path(OUT, 'decision.json'),r);e.write(_repo_path(ROOT, 'reports/spatial-joint-140-summary.json'),r);e.emit(r)
if __name__=='__main__':
 torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','extract','summary',*e.SPLITS,*['validate-'+s for s in e.SPLITS]]);s=p.parse_args().mode
 if s=='prepare':prepare()
 elif s=='extract':extract()
 elif s=='summary':summarize()
 elif s.startswith('validate-'):validate(s[9:])
 else:run(s)
