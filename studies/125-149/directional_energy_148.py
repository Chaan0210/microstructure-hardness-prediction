"""One bounded directional joint-token energy scalar added to018's fixed procedure."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import bootstrap_selector_144 as s
import numpy as np,torch,joblib
from sklearn.model_selection import GroupKFold
e=s.e;a=s.a;n=s.n;OUT=_repo_path(ROOT, 'artifacts/directional-energy-148')

def anisotropy(z):
 H=np.diff(z,axis=1).reshape(-1,32);V=np.diff(z,axis=0).reshape(-1,32);H=H.T@H/len(H);V=V.T@V/len(V);total=float(np.trace(H+V));value=float(np.sum((H-V)**2)/(total*total)) if total>1e-12 else 0.;assert np.isfinite(value) and 0<=value<=1+1e-12;return value

def prepare():
 s.check();OUT.mkdir(exist_ok=False);names=['directional_energy_148.py','bootstrap_selector_144.py','reports/complementary-bank-147-diagnostic.json','reports/candidate018-public-submission.json','reports/candidate018-verification.json','artifacts/spatial-joint-140/features.npz','artifacts/spatial-joint-140/extraction.json','artifacts/patch-attention-092/cache.json']
 for split in e.SPLITS:names += [f'artifacts/bootstrap-selector-144/{split}/report.json',f'artifacts/bootstrap-selector-144/{split}/predictions.npz',f'artifacts/bootstrap-selector-144/validation-{split}.json']
 r={'study':'148 one directional joint-token anisotropy scalar','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'motivation':'147twooldbankalternatives have negative pooledalignment withremaining018error andinconsistentpartitions; nooldbankblend pursued.140/018 graphenergykeepsH+V only; H-V encodesdirection-dependentchannelchanges thatsumcannotrecover. Thisisrepresentationalmissinginformation,notprovenresidualsignal.', 'formula':'Samefixed140B768->QR32 tokenmap36x36. H=meanouterproduct(horizontalneighbor differences); V=meanouterproduct(verticalneighbor differences),1260each. scalar=FrobeniusNormSquared(H-V)/trace(H+V)^2,zeroiftrace<=1e-12. PSDmatrices imply0<=scalar<=1. No directionlabel,lag/rank/epsilon/seedsearch. Appendonefixedblock directional_energy ofwidth1 to585=>586.', 'distinction':'042averagesmarginalH/V and070coarseregionalvariance;140hascrosschannelH+V,notdifference.092learnedXYpooling. Rawphysics57alignmentproxiesalreadyexist; thissemanticjoint-tokenanisotropy isnotclaimedindependentof them orphysicalcausality.', 'control':'Exact018bootstrap144 with585features; addonecomputedscalaronly. Same5none/ridge1/ridge01/rbf1/rbf01,train-localMinMaxclip/StandardScaler/blockweights,alphalambda*n,innerGroupKFold3,512groupbootstrapselectionfrequencies,sameseedsandmean3honestblocks. No137curve,changedbaseweightsornewneuralfit.', 'nested':'Useexactverified015/017honestanchorA213/214onB106/107,excludingV80; recomputeallnewinnercandidates onlyBtrainandfullcandidatesonlyB. WeightsfromBinnerOOFonly,notglobalOOF. A-onlydeepbase/calibration/shortB retained. Full500usesA333/334,B166/167 onlyaftervalidatedlock.', 'criterion':'Runall3existingpartitions; advanceonlyifpooleduntrimmedMSE strictlybelow018 andallindependentchecks pass. Noeachpartition/fold/CIgate. Ifnegativeclose thisscalar withouttryingsigneddirection,scale,lag,projectionorblendchanges.', 'validation':'Syntheticconstant,one-directionanddiagonalfixtures,sameH+Vdifferentanisotropy;D4oncomputedmaps; independentlyexplicitneighborouterproducts on3cachedimages;all900modelsindependentrawscaling/kernel/linearvalidation,weightsgrouploop,outerlabelpoison. Exact018controlfromcachedmodelsandpredictions; upstream146lineageauditreused.', 'resources':'ExistingCPU2,400tokencache reused; noimages/holdout/testduringdevelopment. CPU20fitsperblock*45=900includingzero; noinstall/cost. Positivecandidate:full500additional100featureextractandtest1000afterlock usingofficialexistingBweightsMPS.','limits':'Boundedstatisticneednotbeuseful. D4aggregationinvarianceisnotwholeDINOimageinvariance. Same400adaptivehistory; report10000IDclusterconditionalCIseed20271207,notsignificanceproof.','hashes':{f:e.sha(_repo_path(ROOT, f)) for f in names},'protected':{f'artifacts/submission-candidate-{c}.csv':e.sha(_repo_path(ROOT, f'artifacts/submission-candidate-{c}.csv')) for c in ['017','018']},'cost':0};e.write(_repo_path(OUT, 'plan.json'),r);e.emit({'locked148':True,'new_features':1,'sha256':e.sha(_repo_path(OUT, 'plan.json'))})

def check():
 r=e.read(_repo_path(OUT, 'plan.json'))
 for f,h in {**r['hashes'],**r['protected']}.items():assert e.sha(_repo_path(ROOT, f))==h,f
 return r

def extract():
 check();start=time.perf_counter();meta=e.read(_repo_path(ROOT, 'artifacts/patch-attention-092/cache.json'));f=_repo_path(ROOT, 'artifacts/patch-attention-092/patch_tokens.npy');assert e.sha(f)==meta['token_sha256'];T=np.load(_repo_io(f),mmap_mode='r');z=a.readnp(s.q.OUT/'features.npz',e.read(s.q.OUT/'extraction.json')['features_sha256']);assert meta['ids']==z['ids'].tolist();P=z['projection'];values=[];err=0.
 for i,t in enumerate(T):
  zz=(np.asarray(t,dtype=float)@P).reshape(36,36,32);v=anisotropy(zz);values.append(v)
  if i in [0,199,399]:
   H=sum(np.outer(zz[r,c+1]-zz[r,c],zz[r,c+1]-zz[r,c]) for r in range(36) for c in range(35))/1260;V=sum(np.outer(zz[r+1,c]-zz[r,c],zz[r+1,c]-zz[r,c]) for r in range(35) for c in range(36))/1260;manual=float(np.trace((H-V)@(H-V))/(np.trace(H+V)**2));err=max(err,abs(v-manual));assert err<1e-12
   for rot in range(4):
    np.testing.assert_allclose(anisotropy(np.rot90(zz,rot)),v,atol=1e-12,rtol=0);np.testing.assert_allclose(anisotropy(np.rot90(zz,rot)[:,::-1]),v,atol=1e-12,rtol=0)
 X=np.array(values)[:,None];constant=np.zeros((5,5,32));assert anisotropy(constant)==0;horizontal=constant.copy();horizontal[:,:,0]=np.arange(5)[None,:];diag=constant.copy();diag[:,:,0]=(np.arange(5)[None,:]+np.arange(5)[:,None])/np.sqrt(2);np.testing.assert_allclose(s.q.stats(horizontal)[1],s.q.stats(diag)[1],atol=1e-12,rtol=0);assert abs(anisotropy(horizontal)-1)<1e-12 and anisotropy(diag)<1e-28
 with (_repo_path(OUT, 'features.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=z['ids'],X=X)
 r={'passed':True,'rows':400,'width':1,'cache_sha256':e.sha(_repo_path(OUT, 'features.npz')),'explicit_edge_max_error':err,'D4_tokenmap_fixture':True,'same_sum_different_direction_fixture':True,'bounded_range':[float(X.min()),float(X.max())],'seconds':time.perf_counter()-start};e.write(_repo_path(OUT, 'extraction.json'),r);e.emit(r)

def data(split):
 d=s.data(split);rr=e.read(s.OUT/split/'report.json');zz=a.readnp(s.OUT/split/'predictions.npz',rr['predictions_sha256']);assert zz['ids'].tolist()==d['ids'];d['reference018']=zz['primary'];r=e.read(_repo_path(OUT, 'extraction.json'));z=a.readnp(_repo_path(OUT, 'features.npz'),r['cache_sha256']);assert z['ids'].tolist()==d['ids'];d['X']=np.column_stack([d['X'],z['X']]);d['blocks']=list(d['blocks'])+['directional_energy'];return d

def system(d,B,anchor,V,foldseed,bootseed,path):
 path.mkdir(exist_ok=False);res=d['y'][B]-anchor;folds=list(GroupKFold(3,shuffle=True,random_state=foldseed).split(d['X'][B],res,d['g'][B]));ip=np.full((5,len(B)),np.nan);hashes={}
 for t,(it,iv) in enumerate(folds):
  assert not set(d['g'][B[it]])&set(d['g'][B[iv]])
  for ci,kind in enumerate(s.KINDS):
   model=n.fit(d['X'][B[it]],res[it],d['blocks'],kind);ip[ci,iv]=n.predict(model,d['X'][B[iv]]);file=path/f'inner-{t}-{kind}.joblib';joblib.dump({'model':model,'train_indices':B[it],'validation_indices':B[iv]},_repo_io(file));hashes[file.name]=e.sha(file)
 w,draw,wins=s.weights((ip-res)**2,d['g'][B],bootseed);pp=[]
 for kind in s.KINDS:
  model=n.fit(d['X'][B],res,d['blocks'],kind);pp.append(n.predict(model,d['X'][V]));file=path/f'full-{kind}.joblib';joblib.dump({'model':model,'train_indices':B},_repo_io(file));hashes[file.name]=e.sha(file)
 pp=np.stack(pp);primary=w@pp
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),block_indices=B,outer_indices=V,anchor=anchor,residual=res,inner_predictions=ip,weights=w,draws=draw,winners=wins,full_predictions=pp,primary=primary)
 r={'weights':dict(zip(s.KINDS,w.tolist())),'inner_MSE':dict(zip(s.KINDS,np.mean((ip-res)**2,axis=1).tolist())),'foldseed':foldseed,'bootseed':bootseed,'splits':[{'train_indices':B[it].tolist(),'validation_indices':B[iv].tolist()} for it,iv in folds],'models_sha256':hashes,'predictions_sha256':e.sha(path/'predictions.npz')};e.write(path/'report.json',r);return primary,r

def run(split):
 check();d=data(split);out=_repo_path(OUT, split);out.mkdir();primary=np.full(400,np.nan);hashes={};start=time.perf_counter()
 for k in range(5):
  src=s.q.OUT/split/f'adjacent-fold-{k}';fr=e.read(src/'report.json');fz=a.readnp(src/'predictions.npz',fr['predictions_sha256']);V=fz['outer_indices'];cor=[];control=[]
  for j in range(3):
   br=e.read(src/f'block-{j}/report.json');bz=a.readnp(src/f'block-{j}/predictions.npz',br['predictions_sha256']);B=bz['block_indices'];assert not set(B)&set(V);pp,r=system(d,B,bz['anchor_predictions'],V,a.r.CVSEED+100*k+j,s.SEED+10000*e.SPLITS.index(split)+100*k+j,out/f'fold-{k}-block-{j}');cor.append(pp);hashes[f'fold-{k}-block-{j}/report.json']=e.sha(out/f'fold-{k}-block-{j}/report.json');old=e.read(s.OUT/split/f'fold-{k}-block-{j}/report.json');oz=a.readnp(s.OUT/split/f'fold-{k}-block-{j}/predictions.npz',old['predictions_sha256']);control.append(oz['primary'])
  np.testing.assert_array_equal(fz['base']+np.mean(control,0),d['reference018'][V]);primary[V]=fz['base']+np.mean(cor,0);e.emit({'split':split,'fold_complete':k})
 with (out/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],reference018=d['reference018'],primary=primary)
 r={'split':split,'RMSE':{'reference018':e.rmse(d['y'],d['reference018']),'primary':e.rmse(d['y'],primary)},'models_including_none':300,'block_reports_sha256':hashes,'predictions_sha256':e.sha(out/'predictions.npz'),'seconds':time.perf_counter()-start};e.write(out/'report.json',r);check();e.emit(r)

def verify_system(d,B,anchor,V,path):
 r=e.read(path/'report.json');z=a.readnp(path/'predictions.npz',r['predictions_sha256']);np.testing.assert_array_equal(z['block_indices'],B);np.testing.assert_array_equal(z['outer_indices'],V);np.testing.assert_array_equal(z['anchor'],anchor);res=d['y'][B]-anchor;np.testing.assert_array_equal(z['residual'],res);ip=np.empty_like(z['inner_predictions']);maxerr=0.;count=0;folds=list(GroupKFold(3,shuffle=True,random_state=r['foldseed']).split(d['X'][B],res,d['g'][B]))
 for t,(it,iv) in enumerate(folds):
  assert r['splits'][t]=={'train_indices':B[it].tolist(),'validation_indices':B[iv].tolist()};assert not set(d['g'][B[it]])&set(d['g'][B[iv]])
  for ci,kind in enumerate(s.KINDS):
   file=path/f'inner-{t}-{kind}.joblib';assert e.sha(file)==r['models_sha256'][file.name];obj=joblib.load(_repo_io(file));np.testing.assert_array_equal(obj['train_indices'],B[it]);np.testing.assert_array_equal(obj['validation_indices'],B[iv]);manual=s.q.vn.manual(obj['model'],d['X'][B[it]],res[it],d['X'][B[iv]],d['blocks']);actual=n.predict(obj['model'],d['X'][B[iv]]);err=float(abs(actual-manual).max());assert err<2e-9;maxerr=max(maxerr,err);np.testing.assert_array_equal(actual,z['inner_predictions'][ci,iv]);ip[ci,iv]=actual;count+=1
 np.testing.assert_allclose(np.mean((ip-res)**2,1),list(r['inner_MSE'].values()),atol=1e-12,rtol=0);w,wins=s.independent_weights((ip-res)**2,d['g'][B],z['draws']);np.testing.assert_array_equal(w,z['weights']);np.testing.assert_array_equal(wins,z['winners']);np.testing.assert_array_equal(w,list(r['weights'].values()));_,draw,_=s.weights((ip-res)**2,d['g'][B],r['bootseed']);np.testing.assert_array_equal(draw,z['draws']);pp=[]
 for ci,kind in enumerate(s.KINDS):
  file=path/f'full-{kind}.joblib';assert e.sha(file)==r['models_sha256'][file.name];obj=joblib.load(_repo_io(file));np.testing.assert_array_equal(obj['train_indices'],B);manual=s.q.vn.manual(obj['model'],d['X'][B],res,d['X'][V],d['blocks']);actual=n.predict(obj['model'],d['X'][V]);err=float(abs(actual-manual).max());assert err<2e-9;maxerr=max(maxerr,err);np.testing.assert_array_equal(actual,z['full_predictions'][ci]);pp.append(actual);count+=1
 np.testing.assert_allclose(sum(wi*pi for wi,pi in zip(w,pp)),z['primary'],atol=1e-12,rtol=0);return z['primary'],count,maxerr

def validate(split):
 check();d=data(split);out=_repo_path(OUT, split);rr=e.read(out/'report.json');saved=a.readnp(out/'predictions.npz',rr['predictions_sha256']);assert saved['ids'].tolist()==d['ids'];np.testing.assert_array_equal(saved['y'],d['y']);count=0;maxerr=0.
 for k in range(5):
  src=s.q.OUT/split/f'adjacent-fold-{k}';fr=e.read(src/'report.json');fz=a.readnp(src/'predictions.npz',fr['predictions_sha256']);V=fz['outer_indices'];cor=[]
  for j in range(3):
   path=out/f'fold-{k}-block-{j}';assert e.sha(path/'report.json')==rr['block_reports_sha256'][f'fold-{k}-block-{j}/report.json'];old=e.read(src/f'block-{j}/report.json');oz=a.readnp(src/f'block-{j}/predictions.npz',old['predictions_sha256']);B=oz['block_indices'];pp,c,err=verify_system(d,B,oz['anchor_predictions'],V,path);cor.append(pp);count+=c;maxerr=max(maxerr,err)
   if k==j==0:
    pert=dict(d);pert['y']=d['y'].copy();pert['y'][V]+=100000;rp,rec=system(pert,B,oz['anchor_predictions'],V,a.r.CVSEED,s.SEED+10000*e.SPLITS.index(split),_repo_path(OUT, f'poison-{split}'));np.testing.assert_array_equal(rp,pp)
  np.testing.assert_array_equal(fz['base']+np.mean(cor,0),saved['primary'][V])
 r={'passed':True,'split':split,'models_verified':count,'independent_max_error':maxerr,'independent_bootstrap_weights_exact':True,'outer_target_poison_exact':True,'upstream_lineage_audit':e.sha(_repo_path(ROOT, 'reports/nested-selector-146-audit.json')),'report_sha256':e.sha(out/'report.json')};e.write(_repo_path(OUT, f'validation-{split}.json'),r);check();e.emit(r)

def summary():
 check();errors={k:[] for k in ['reference018','primary']};scores={}
 for split in e.SPLITS:
  r=e.read(_repo_path(OUT, split, 'report.json'));v=e.read(_repo_path(OUT, f'validation-{split}.json'));assert v['passed'] and v['report_sha256']==e.sha(_repo_path(OUT, split, 'report.json'));z=a.readnp(_repo_path(OUT, split, 'predictions.npz'),r['predictions_sha256']);scores[split]=r['RMSE']
  for key in errors:errors[key].append((z[key]-z['y'])**2)
 errors={k:np.stack(v) for k,v in errors.items()};M={k:float(v.mean()) for k,v in errors.items()};rng=np.random.default_rng(20271207);g=[]
 for _ in range(10000):
  ids=rng.integers(0,400,400);g.append(float(np.sqrt(errors['reference018'][:,ids].mean())-np.sqrt(errors['primary'][:,ids].mean())))
 r={'study':'148 one directional energy scalar','advance':M['primary']<M['reference018'],'validation_passed':True,'RMSE_by_partition':scores,'pooled_MSE':M,'pooled_RMSE':{k:float(np.sqrt(v)) for k,v in M.items()},'conditional_gain_CI95':np.quantile(g,[.025,.975]).tolist(),'same400_adaptive_history':True,'no_parameter_variants_if_failure':True,'current_best':'018','test_or_holdout_read':False,'new_CSV':False,'cost':0,'plan_sha256':e.sha(_repo_path(OUT, 'plan.json'))};e.write(_repo_path(OUT, 'decision.json'),r);e.write(_repo_path(ROOT, 'reports/directional-energy-148-summary.json'),r);e.emit(r)
if __name__=='__main__':
 torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','extract','summary',*e.SPLITS,*['validate-'+v for v in e.SPLITS]]);v=p.parse_args().mode
 if v=='prepare':prepare()
 elif v=='extract':extract()
 elif v=='summary':summary()
 elif v.startswith('validate-'):validate(v[9:])
 else:run(v)
