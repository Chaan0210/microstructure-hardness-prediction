"""One fixed counterfactual: preserve spatial amplitude outside residual train extrema."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,datetime,argparse,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import bootstrap_selector_144 as s
import numpy as np,torch,joblib
from scipy.spatial.distance import cdist
OUT=_repo_path(ROOT, 'artifacts/spatial-support-154');e=s.e;a=s.a;n=s.n

def predict(m,X):
 if m['kind']=='none':return np.zeros(len(X))
 assert X.shape[1]==585;b=m['bound'];U=X*b.scale_+b.min_;U[:,:57]=np.clip(U[:,:57],0,1);Z=m['scale'].transform(U)*m['weights']
 if m['kind'].startswith('ridge'):return m['regressor'].predict(Z)*m['residual_scale']
 K=np.exp(-m['gamma']*cdist(Z,m['Z'],'sqeuclidean'));C=(K-K.mean(1)[:,None]-m['column'][None,:]+m['grand'])/m['kernel_scale'];return C@m['coef']*m['residual_scale']

def manual(m,X,y,V,blocks):
 if m['kind']=='none':return np.zeros(len(V))
 low=X.min(0);span=X.max(0)-low;span[span<10*np.finfo(float).eps]=1.;U=np.clip((X-low)/span,0,1);UV=(V-low)/span;UV[:,:57]=np.clip(UV[:,:57],0,1);mu=U.mean(0);var=U.var(0);sd=np.sqrt(var);sd[sd==0]=1.;w=np.array([1/np.sqrt(blocks.count(b)) for b in blocks]);Z=(U-mu)/sd*w;ZV=(UV-mu)/sd*w;np.testing.assert_allclose(m['bound'].data_min_,low,atol=0,rtol=0);np.testing.assert_allclose(m['scale'].mean_,mu,atol=1e-12,rtol=0);np.testing.assert_allclose(m['scale'].var_,var,atol=1e-12,rtol=0);np.testing.assert_array_equal(w,m['weights']);rs=float(np.sqrt(np.mean(y*y)));rs=rs if rs>1e-12 else 1.;assert rs==m['residual_scale'];assert m['alpha']==({'ridge1':1.,'ridge01':.1,'rbf1':1.,'rbf01':.1}[m['kind']])*len(y)
 if m['kind'].startswith('ridge'):
  dual=np.linalg.solve(Z@Z.T+m['alpha']*np.eye(len(X)),y/rs);coef=Z.T@dual;np.testing.assert_allclose(coef,m['regressor'].coef_,atol=2e-10,rtol=1e-10);return ZV@coef*rs
 dist=np.sum((Z[:,None,:]-Z[None,:,:])**2,2);tri=dist[np.triu_indices(len(X),1)];pos=tri[tri>1e-12];gamma=1/float(np.median(pos)) if len(pos) else 1.;assert abs(gamma-m['gamma'])<1e-12;K=np.exp(-gamma*dist);H=np.eye(len(X))-np.ones((len(X),len(X)))/len(X);C=H@K@H;scale=float(np.trace(C)/len(X));scale=1. if scale<=1e-12 else scale;assert abs(scale-m['kernel_scale'])<1e-12;C/=scale;evals,Q=np.linalg.eigh(C);dual=Q@((Q.T@(y/rs))/(evals+m['alpha']));np.testing.assert_allclose(dual,m['coef'],atol=2e-11,rtol=1e-10);VK=np.exp(-gamma*np.sum((ZV[:,None,:]-Z[None,:,:])**2,2));return (VK-VK.mean(1)[:,None]-K.mean(0)[None,:]+K.mean())/scale@dual*rs

def prepare():
 s.check();OUT.mkdir(exist_ok=False);names=['spatial_support_154.py','bootstrap_selector_144.py','nonlinear_residual_115.py','residual_physics_113.py','reports/support-diagnostic-153.json','reports/candidate019-public-submission.json','reports/nested-selector-146-audit.json','reports/spatial-joint-142-audit.json']
 for split in e.SPLITS:names += [f'artifacts/bootstrap-selector-144/{split}/report.json',f'artifacts/bootstrap-selector-144/validation-{split}.json',f'artifacts/spatial-joint-140/validation-{split}.json']
 r={'study':'154 spatial support saturation control','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hypothesis':'Training-extrema coordinate clipping removes spatial graph-energy amplitudes differently at inner70/72 and full106/107 residual training sizes. Fixed standardized spatial amplitudes without hard saturation may transfer better. Diagnosis shows broad exposure, not positive error association or causal proof.','diagnostic':e.read(_repo_path(ROOT, 'reports/support-diagnostic-153.json'))['feature_support_summary'],'primary':'Only residual features57:585 bypass MinMax clipping at prediction. Physics0:57 keeps original clipping. Same train-only MinMax and StandardScaler, block weights, Ridge/RBF train models, regularization, 512groupbootstrap and3honest blocks. Recompute all innerOOF predictions, bootstrap weights and outer corrections. No hard/soft cutoff grid or variants.','equivalence':'Every training coordinate lies inside own extrema; unchanged fitted models can be reused. Verify train-query predictions agree to floating precision, independently solve all900 model objects, fresh refit representative inner and full models. No new backbone or neural training.','duplicate_audit':'135/142 documented clipping, but did not compare spatial-only amplitude preservation. Different from148 directional statistic,149 local crossmoment,137 quadratic output calibration,118 subsampling,119 pooled residuals,128 update budget,125 base-weight fitting. No rejected feature or ensemble axis reopened.','design':'All three existing partitions. Primary pooled untrimmed MSE must be below unchanged018 and all validation pass. No fold/CI minimum, no rescue blend or physics-unclipping variant. Fail closes this transform.','validation':'Exact018 control from saved models; train-scope/index/group checks; independent raw transforms and linear/kernel solves; independent groupbootstrap weights; representative fresh refits and outer-label poison. Feature and target fits remain inner training only.','risk':'Removing spatial caps restores linear Ridge extrapolation and alters RBF distances outside train box; no guarantee of bounded wholeprediction or Public/private benefit. No test-based thresholding.','uncertainty':'10000paired400IDcluster bootstrap seed20271212, conditional on fits and historical adaptive reuse. No independent validation claim.','resources':'Existing CPU2 only, model reuse and several validation refits; no installs/cost/holdout/test. Positive result requires locked full500 downstream recipe and inference validation. No submission today or by this worker.','hashes':{f:e.sha(_repo_path(ROOT, f)) for f in names},'protected':{f'artifacts/submission-candidate-{i}.csv':e.sha(_repo_path(ROOT, f'artifacts/submission-candidate-{i}.csv')) for i in ['018','019']},'cost':0};e.write(_repo_path(OUT, 'plan.json'),r);e.emit({'locked154':True,'plan_sha256':e.sha(_repo_path(OUT, 'plan.json'))})

def check():
 s.check();r=e.read(_repo_path(OUT, 'plan.json'))
 for f,h in {**r['hashes'],**r['protected']}.items():assert e.sha(_repo_path(ROOT, f))==h,f
 return r

def block(d,split,k,j,path):
 path.mkdir(exist_ok=False);src=s.q.OUT/split/f'adjacent-fold-{k}/block-{j}';old=e.read(src/'report.json');z=a.readnp(src/'predictions.npz',old['predictions_sha256']);B=z['block_indices'];V=z['outer_indices'];res=d['y'][B]-z['anchor_predictions'];np.testing.assert_array_equal(res,z['residual']);assert not set(B)&set(V);ip=np.empty((5,len(B)));lookup={int(ii):ix for ix,ii in enumerate(B)};refs={}
 for t,fold in enumerate(old['splits']):
  I=np.array(fold['train_indices']);H=np.array(fold['validation_indices']);assert set(I)|set(H)==set(B) and not set(I)&set(H) and not set(d['g'][I])&set(d['g'][H]);loc=[lookup[int(ii)] for ii in H]
  for ci,kind in enumerate(s.KINDS):
   f=src/f'inner-{t}-{kind}.joblib';assert e.sha(f)==old['models_sha256'][f.name];obj=joblib.load(_repo_io(f));np.testing.assert_array_equal(obj['train_indices'],I);np.testing.assert_array_equal(obj['validation_indices'],H);ip[ci,loc]=predict(obj['model'],d['X'][H]);refs[f'inner-{t}-{kind}']={'path':str(f.relative_to(ROOT)),'sha256':e.sha(f)}
 seed=s.SEED+10000*e.SPLITS.index(split)+100*k+j;w,draw,wins=s.weights((ip-res)**2,d['g'][B],seed);base=s.OUT/split/f'fold-{k}-block-{j}';br=e.read(base/'report.json');bz=a.readnp(base/'predictions.npz',br['predictions_sha256']);pp=[];control=[]
 for ci,kind in enumerate(s.KINDS):
  f=base/f'{kind}.joblib';assert e.sha(f)==br['models_sha256'][f.name];obj=joblib.load(_repo_io(f));np.testing.assert_array_equal(obj['train_indices'],B);pp.append(predict(obj['model'],d['X'][V]));control.append(n.predict(obj['model'],d['X'][V]));refs[f'full-{kind}']={'path':str(f.relative_to(ROOT)),'sha256':e.sha(f)}
 pp=np.stack(pp);np.testing.assert_array_equal(np.stack(control),bz['full_candidate_predictions']);np.testing.assert_array_equal(bz['weights']@np.stack(control),bz['primary']);primary=w@pp
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),block_indices=B,outer_indices=V,anchor=z['anchor_predictions'],residual=res,inner_predictions=ip,weights=w,draws=draw,winners=wins,full_predictions=pp,primary=primary,control=bz['primary'])
 r={'weights':dict(zip(s.KINDS,w.tolist())),'seed':seed,'inner_MSE':dict(zip(s.KINDS,np.mean((ip-res)**2,1).tolist())),'splits':old['splits'],'source_models':refs,'source_anchor_report_sha256':e.sha(src/'report.json'),'predictions_sha256':e.sha(path/'predictions.npz')};e.write(path/'report.json',r);return primary,bz['primary'],r

def run(split):
 check();d=s.data(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);primary=np.full(400,np.nan);control=primary.copy();hashes={}
 for k in range(5):
  source=s.q.OUT/split/f'adjacent-fold-{k}';fr=e.read(source/'report.json');fz=a.readnp(source/'predictions.npz',fr['predictions_sha256']);V=fz['outer_indices'];p=[];c=[]
  for j in range(3):
   pp,cc,rec=block(d,split,k,j,out/f'fold-{k}-block-{j}');p.append(pp);c.append(cc);hashes[f'fold-{k}-block-{j}/report.json']=e.sha(out/f'fold-{k}-block-{j}/report.json')
  primary[V]=fz['base']+np.mean(p,0);control[V]=fz['base']+np.mean(c,0);e.emit({'split':split,'fold_complete':k})
 old=e.read(s.OUT/split/'report.json');oz=a.readnp(s.OUT/split/'predictions.npz',old['predictions_sha256']);np.testing.assert_array_equal(control,oz['primary'])
 with (out/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],reference018=control,primary=primary)
 r={'split':split,'RMSE':{'reference018':e.rmse(d['y'],control),'primary':e.rmse(d['y'],primary)},'block_reports_sha256':hashes,'predictions_sha256':e.sha(out/'predictions.npz'),'new_fits':0};e.write(out/'report.json',r);check();e.emit(r['RMSE'])

def validate(split):
 check();d=s.data(split);rr=e.read(_repo_path(OUT, split, 'report.json'));saved=a.readnp(_repo_path(OUT, split, 'predictions.npz'),rr['predictions_sha256']);assert saved['ids'].tolist()==d['ids'];np.testing.assert_array_equal(saved['y'],d['y']);count=0;maxerr=0.;trainerr=0.;fresh=0
 for k in range(5):
  source=s.q.OUT/split/f'adjacent-fold-{k}';fr=e.read(source/'report.json');fz=a.readnp(source/'predictions.npz',fr['predictions_sha256']);cor=[]
  for j in range(3):
   path=_repo_path(OUT, split, f'fold-{k}-block-{j}');r=e.read(path/'report.json');assert e.sha(path/'report.json')==rr['block_reports_sha256'][f'fold-{k}-block-{j}/report.json'];z=a.readnp(path/'predictions.npz',r['predictions_sha256']);B=z['block_indices'];V=z['outer_indices'];res=d['y'][B]-z['anchor'];np.testing.assert_array_equal(res,z['residual']);lookup={int(ii):ix for ix,ii in enumerate(B)};ip=np.empty_like(z['inner_predictions']);pp=[]
   for name,ref in r['source_models'].items():
    file=_repo_path(ROOT, ref['path']);assert e.sha(file)==ref['sha256'];obj=joblib.load(_repo_io(file));I=obj['train_indices'];m=obj['model'];kind=m['kind'];target=res[[lookup[int(ii)] for ii in I]];H=obj['validation_indices'] if name.startswith('inner') else V;actual=predict(m,d['X'][H]);man=manual(m,d['X'][I],target,d['X'][H],d['blocks']);err=float(abs(actual-man).max());assert err<2e-9;maxerr=max(maxerr,err);te=float(abs(predict(m,d['X'][I])-n.predict(m,d['X'][I])).max());assert te<1e-10;trainerr=max(trainerr,te);count+=1
    if name.startswith('inner'):ip[s.KINDS.index(kind),[lookup[int(ii)] for ii in H]]=actual
    else:pp.append(actual)
    if k==j==0 and (name.startswith('inner-0-') or name.startswith('full-')):
     freshmodel=n.fit(d['X'][I],target,d['blocks'],kind);np.testing.assert_array_equal(predict(freshmodel,d['X'][H]),actual);fresh+=1
   np.testing.assert_array_equal(ip,z['inner_predictions']);np.testing.assert_array_equal(np.stack(pp),z['full_predictions']);w,wins=s.independent_weights((ip-res)**2,d['g'][B],z['draws']);np.testing.assert_array_equal(w,z['weights']);np.testing.assert_array_equal(wins,z['winners']);_,draw,_=s.weights((ip-res)**2,d['g'][B],r['seed']);np.testing.assert_array_equal(draw,z['draws']);np.testing.assert_array_equal(w@np.stack(pp),z['primary']);cor.append(z['primary'])
  np.testing.assert_array_equal(fz['base']+np.mean(cor,0),saved['primary'][fz['outer_indices']])
 pert=dict(d);pert['y']=d['y'].copy();pert['y'][d['fid']==0]+=100000;pp,_,_=block(pert,split,0,0,_repo_path(OUT, f'poison-{split}'));r=e.read(_repo_path(OUT, split, 'fold-0-block-0/report.json'));z=a.readnp(_repo_path(OUT, split, 'fold-0-block-0/predictions.npz'),r['predictions_sha256']);np.testing.assert_array_equal(pp,z['primary']);r={'passed':True,'split':split,'models_verified':count,'independent_max_error':maxerr,'train_support_equivalence_max_error':trainerr,'fresh_refits_exact':fresh,'outer_target_poison_exact':True,'upstream_anchor_exclusion_certificate':e.sha(_repo_path(ROOT, 'reports/nested-selector-146-audit.json')),'report_sha256':e.sha(_repo_path(OUT, split, 'report.json'))};e.write(_repo_path(OUT, f'validation-{split}.json'),r);check();e.emit(r)

def summary():
 check();E={k:[] for k in ['reference018','primary']};scores={}
 for split in e.SPLITS:
  r=e.read(_repo_path(OUT, split, 'report.json'));v=e.read(_repo_path(OUT, f'validation-{split}.json'));assert v['passed'] and v['report_sha256']==e.sha(_repo_path(OUT, split, 'report.json'));z=a.readnp(_repo_path(OUT, split, 'predictions.npz'),r['predictions_sha256']);scores[split]=r['RMSE']
  for k in E:E[k].append((z[k]-z['y'])**2)
 E={k:np.stack(v) for k,v in E.items()};M={k:float(v.mean()) for k,v in E.items()};rng=np.random.default_rng(20271212);g=[]
 for _ in range(10000):
  ii=rng.integers(0,400,400);g.append(float(np.sqrt(E['reference018'][:,ii].mean())-np.sqrt(E['primary'][:,ii].mean())))
 r={'study':'154 spatial-only support preservation','advance':M['primary']<M['reference018'],'validation_passed':True,'RMSE_by_partition':scores,'pooled_MSE':M,'pooled_RMSE':{k:float(np.sqrt(v)) for k,v in M.items()},'conditional_gain_CI95':np.quantile(g,[.025,.975]).tolist(),'plan_sha256':e.sha(_repo_path(OUT, 'plan.json')),'same400_adaptive_history':True,'current_best':'018','new_neural_fits':0,'test_or_holdout_read':False,'cost':0};e.write(_repo_path(OUT, 'decision.json'),r);e.write(_repo_path(ROOT, 'reports/spatial-support-154-summary.json'),r);e.emit(r)
if __name__=='__main__':
 torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','summary',*e.SPLITS,*['validate-'+v for v in e.SPLITS]]);v=p.parse_args().mode
 if v=='prepare':prepare()
 elif v=='summary':summary()
 elif v.startswith('validate-'):validate(v[9:])
 else:run(v)
