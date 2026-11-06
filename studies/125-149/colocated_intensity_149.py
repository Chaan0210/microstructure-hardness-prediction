"""Fixed aligned grayscale/token cross-moment versus permuted-location control."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import directional_energy_148 as core
import numpy as np,torch
from PIL import Image
s=core.s;a=core.a;e=core.e;OUT=_repo_path(ROOT, 'artifacts/colocated-intensity-149');ARMS=('permuted','aligned');PERMSEED=20271208

def image_panel(path):
 with Image.open(_repo_io(path)) as im:
  assert im.mode=='L' and im.size==(256,256);u=np.asarray(im.resize((504,504),Image.Resampling.BICUBIC),dtype=np.float64)/255
 u=u.reshape(36,14,36,14).mean((1,3)).ravel();u=u-u.mean();return u/max(float(u.std()),1e-6)
def describe(z,u):return (u[:,None]*(z-z.mean(0))).mean(0)

def prepare():
 s.check();core.check();assert not e.read(core.OUT/'decision.json')['advance'];OUT.mkdir(exist_ok=False);names=['colocated_intensity_149.py','directional_energy_148.py','bootstrap_selector_144.py','reports/directional-energy-148-summary.json','reports/candidate018-public-submission.json','artifacts/patch-attention-092/cache.json','artifacts/spatial-joint-140/features.npz','artifacts/spatial-joint-140/extraction.json','artifacts/nested-calibration-074/extraction.json']
 for split in e.SPLITS:names += [f'artifacts/bootstrap-selector-144/{split}/report.json',f'artifacts/bootstrap-selector-144/{split}/predictions.npz']
 r={'study':'149 fixed local grayscale-token crossmoment','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hypothesis':'ExistingDINO pooled/graphmoments andglobalphysicsstatistics do notexplicitlyassociatebrightnesswithsemanticpatchfeatures atmatchinglocations. Thislocalcrossmodalassociation mayretaincomplementaryinformation. Notprovenphase/grainidentification orremainingerrorcause.', 'formula':'SameexistingB504finaltokens andfixed140QR768x32. ResizeoriginalL256to504 withidenticalPILbicubic; nonoverlap14x14patchbrightnessmeansrowmajor36x36. Withinimage centeruanddividebymax(populationSD,1e-6); C=mean_i u_i*(Z_i-meanZ),32features. Appendasanewblockwidth32 to018585=>617.', 'control':'Matched32-dimensionalpermutedlocationcrossmoment: sameuandZ butu[fixedpermutation1296seed20271208],sameforallimagesandpartitions. Preservesrawbrightnesshistogram andtokenmarginals; destroysmostexactcolocation. Onlyalignedcanadvance,permutednotfallback. Onepermutationnotuniversalnull.', 'duplicate_audit':'047globalDINO/morphologykernelinteraction,056token-tokencovariance,042/070token-onlyspatialmoments,140jointtokenedgeenergy,148directionalenergy allomitpixel-intensity/tokencolocatedcrossmoment. 148closedandnotincluded. Priorrawcontrast/morphologyfeatures remain in018; no claimnewindependentmeasure.', 'procedure':'Unchanged0185candidates,512groupbootstrapselectionfrequencies,seeds,3honestresidualblocks,trainlocalMinMaxclip/StandardScaler/kernelsettings; recomputeallinner/fullresidualmodelsforeachbank. No addingparameters tobackbone/MLPs,curve,brightnessnormalizationofmodelinput,fitPCAorhyperparametergrid.', 'nested':'Frozenlabelblindfeaturescanextract400once; allregressorpreprocessingonlyBinnertrain/fullB. A-onlynestedanchorswithB/Vexcluded as146audit. No globalOOF fittedblend.', 'criterion':'Runall3existingpartitions. AlignedpooleduntrimmedMSE muststrictlybeatBOTH unchanged018 andpermutedcontrol,withindependentvalidation. Noeveryfold/CIgate. Ifnegativeclosecolocationwithoutalternatebrightnesschannels/pooling/permutations/weights.', 'validation':'Sourceimagehashes andIDs;3explicitpatchmeanloopandcrossmomentchecks;constant/associationfixtures;core148independentall1800models andexplicitbootstrapweights; outertargetpoisononeblockeachsplit/arm; exact018base/control.','resources':'ExistingCPU2/PIL/Numpy,400rawdevelopmentimages andBtokencache only,roughlyseconds extraction andminutesverification. Noholdout/test/full500 untilpositiveverifiedlock. Noinstall/cost/upload.','limits':'DINOtokensglobalcontextmeanslocalassociationisnotlocalreceptivefieldcausalclaim. Same400adaptivehistory,onefixedpermutedcontrol,conditional10000IDclusterbootstrapseed20271209doesnotaccountforsearch. Two smallsequentialhypotheses after018;148failed andpreserved.', 'hashes':{f:e.sha(_repo_path(ROOT, f)) for f in names},'protected':core.check()['protected'],'cost':0};e.write(_repo_path(OUT, 'plan.json'),r);e.emit({'locked149':True,'new_features':32,'control':'fixedpermutedlocations','sha256':e.sha(_repo_path(OUT, 'plan.json'))})

def check():
 r=e.read(_repo_path(OUT, 'plan.json'))
 for f,h in {**r['hashes'],**r['protected']}.items():assert e.sha(_repo_path(ROOT, f))==h,f
 return r

def extract():
 check();start=time.perf_counter();meta=e.read(_repo_path(ROOT, 'artifacts/patch-attention-092/cache.json'));f=_repo_path(ROOT, 'artifacts/patch-attention-092/patch_tokens.npy');assert e.sha(f)==meta['token_sha256'];T=np.load(_repo_io(f),mmap_mode='r');z=a.readnp(s.q.OUT/'features.npz',e.read(s.q.OUT/'extraction.json')['features_sha256']);assert meta['ids']==z['ids'].tolist();P=z['projection'];perm=np.random.default_rng(PERMSEED).permutation(1296);expected=e.read(_repo_path(ROOT, 'artifacts/nested-calibration-074/extraction.json'))['image_sha256'];values={arm:[] for arm in ARMS};maxerr=0.
 for i,(ident,t) in enumerate(zip(meta['ids'],T)):
  path=_repo_path(ROOT, 'data/train', f'{ident}.png');assert e.sha(path)==expected[ident];u=image_panel(path);Z=np.asarray(t,dtype=float)@P;values['aligned'].append(describe(Z,u));values['permuted'].append(describe(Z,u[perm]))
  if i in [0,199,399]:
   with Image.open(_repo_io(path)) as im:arr=np.array(im.resize((504,504),Image.Resampling.BICUBIC),dtype=float)/255
   mu=np.array([arr[r:r+14,c:c+14].mean() for r in range(0,504,14) for c in range(0,504,14)]);mu-=mu.mean();mu/=max(float(mu.std()),1e-6);np.testing.assert_allclose(u,mu,atol=1e-12,rtol=0)
   for arm,uu in [('aligned',mu),('permuted',mu[perm])]:
    manual=sum(float(uu[j])*(Z[j]-Z.mean(0)) for j in range(1296))/1296;err=float(abs(manual-values[arm][-1]).max());assert err<1e-12;maxerr=max(maxerr,err)
 fixture=np.arange(1296,dtype=float);fixture=(fixture-fixture.mean())/fixture.std();ZZ=np.tile(fixture[:,None],(1,32));np.testing.assert_allclose(describe(ZZ,fixture),np.ones(32),atol=1e-12,rtol=0);assert abs(describe(ZZ,fixture[perm])[0])<.2;np.testing.assert_array_equal(describe(ZZ,np.zeros(1296)),np.zeros(32))
 with (_repo_path(OUT, 'features.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=z['ids'],permutation=perm,**{arm:np.array(v) for arm,v in values.items()})
 r={'passed':True,'rows':400,'width_each':32,'source_image_hashes_verified':400,'cache_sha256':e.sha(_repo_path(OUT, 'features.npz')),'explicit_patch_and_crossmoment_max_error':maxerr,'constant_and_colocation_fixture':True,'seconds':time.perf_counter()-start};e.write(_repo_path(OUT, 'extraction.json'),r);e.emit(r)

def data(split,arm):
 d=s.data(split);rr=e.read(s.OUT/split/'report.json');zz=a.readnp(s.OUT/split/'predictions.npz',rr['predictions_sha256']);assert zz['ids'].tolist()==d['ids'];d['reference018']=zz['primary'];z=a.readnp(_repo_path(OUT, 'features.npz'),e.read(_repo_path(OUT, 'extraction.json'))['cache_sha256']);assert z['ids'].tolist()==d['ids'];d['X']=np.column_stack([d['X'],z[arm]]);d['blocks']=list(d['blocks'])+['intensity_token']*32;return d

def run(split):
 check();D={arm:data(split,arm) for arm in ARMS};d=D['aligned'];out=_repo_path(OUT, split);out.mkdir();P={arm:np.full(400,np.nan) for arm in ARMS};hashes={};start=time.perf_counter()
 for k in range(5):
  src=s.q.OUT/split/f'adjacent-fold-{k}';fr=e.read(src/'report.json');fz=a.readnp(src/'predictions.npz',fr['predictions_sha256']);V=fz['outer_indices'];cor={arm:[] for arm in ARMS};control=[]
  for j in range(3):
   br=e.read(src/f'block-{j}/report.json');bz=a.readnp(src/f'block-{j}/predictions.npz',br['predictions_sha256']);B=bz['block_indices'];assert not set(B)&set(V)
   for arm in ARMS:
    path=out/f'{arm}-fold-{k}-block-{j}';pp,r=core.system(D[arm],B,bz['anchor_predictions'],V,a.r.CVSEED+100*k+j,s.SEED+10000*e.SPLITS.index(split)+100*k+j,path);cor[arm].append(pp);hashes[path.name+'/report.json']=e.sha(path/'report.json')
   old=e.read(s.OUT/split/f'fold-{k}-block-{j}/report.json');oz=a.readnp(s.OUT/split/f'fold-{k}-block-{j}/predictions.npz',old['predictions_sha256']);control.append(oz['primary'])
  np.testing.assert_array_equal(fz['base']+np.mean(control,0),d['reference018'][V])
  for arm in ARMS:P[arm][V]=fz['base']+np.mean(cor[arm],0)
  e.emit({'split':split,'fold_complete':k})
 with (out/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],reference018=d['reference018'],**P)
 r={'split':split,'RMSE':{'reference018':e.rmse(d['y'],d['reference018']),**{arm:e.rmse(d['y'],v) for arm,v in P.items()}},'models_including_none':600,'block_reports_sha256':hashes,'predictions_sha256':e.sha(out/'predictions.npz'),'seconds':time.perf_counter()-start};e.write(out/'report.json',r);check();e.emit(r)

def validate(split):
 check();D={arm:data(split,arm) for arm in ARMS};d=D['aligned'];out=_repo_path(OUT, split);rr=e.read(out/'report.json');saved=a.readnp(out/'predictions.npz',rr['predictions_sha256']);assert saved['ids'].tolist()==d['ids'];np.testing.assert_array_equal(saved['y'],d['y']);count=0;maxerr=0.
 for k in range(5):
  src=s.q.OUT/split/f'adjacent-fold-{k}';fr=e.read(src/'report.json');fz=a.readnp(src/'predictions.npz',fr['predictions_sha256']);V=fz['outer_indices'];cor={arm:[] for arm in ARMS}
  for j in range(3):
   old=e.read(src/f'block-{j}/report.json');oz=a.readnp(src/f'block-{j}/predictions.npz',old['predictions_sha256']);B=oz['block_indices']
   for arm in ARMS:
    path=out/f'{arm}-fold-{k}-block-{j}';assert e.sha(path/'report.json')==rr['block_reports_sha256'][path.name+'/report.json'];pp,c,err=core.verify_system(D[arm],B,oz['anchor_predictions'],V,path);cor[arm].append(pp);count+=c;maxerr=max(maxerr,err)
    if k==j==0:
     pert=dict(D[arm]);pert['y']=d['y'].copy();pert['y'][V]+=100000;rp,rec=core.system(pert,B,oz['anchor_predictions'],V,a.r.CVSEED,s.SEED+10000*e.SPLITS.index(split),_repo_path(OUT, f'poison-{split}-{arm}'));np.testing.assert_array_equal(rp,pp)
  for arm in ARMS:np.testing.assert_array_equal(fz['base']+np.mean(cor[arm],0),saved[arm][V])
 r={'passed':True,'split':split,'models_verified':count,'independent_max_error':maxerr,'bootstrap_independent_exact':True,'outer_target_poison_exact':True,'upstream146_certificate':e.sha(_repo_path(ROOT, 'reports/nested-selector-146-audit.json')),'report_sha256':e.sha(out/'report.json')};e.write(_repo_path(OUT, f'validation-{split}.json'),r);check();e.emit(r)

def summary():
 check();errors={k:[] for k in ['reference018',*ARMS]};scores={}
 for split in e.SPLITS:
  r=e.read(_repo_path(OUT, split, 'report.json'));v=e.read(_repo_path(OUT, f'validation-{split}.json'));assert v['passed'] and v['report_sha256']==e.sha(_repo_path(OUT, split, 'report.json'));z=a.readnp(_repo_path(OUT, split, 'predictions.npz'),r['predictions_sha256']);scores[split]=r['RMSE']
  for k in errors:errors[k].append((z[k]-z['y'])**2)
 errors={k:np.stack(v) for k,v in errors.items()};M={k:float(v.mean()) for k,v in errors.items()};rng=np.random.default_rng(20271209);g={k:[] for k in ['reference018','permuted']}
 for _ in range(10000):
  ids=rng.integers(0,400,400)
  for k in g:g[k].append(float(np.sqrt(errors[k][:,ids].mean())-np.sqrt(errors['aligned'][:,ids].mean())))
 r={'study':'149 fixedalignedintensitytokencrossmoment','advance':M['aligned']<min(M['reference018'],M['permuted']),'validation_passed':True,'RMSE_by_partition':scores,'pooled_MSE':M,'pooled_RMSE':{k:float(np.sqrt(v)) for k,v in M.items()},'conditional_gain_CI95':{k:np.quantile(v,[.025,.975]).tolist() for k,v in g.items()},'same400_adaptive_history':True,'no_variants_if_failure':True,'current_best':'018','test_or_holdout_read':False,'new_CSV':False,'cost':0,'plan_sha256':e.sha(_repo_path(OUT, 'plan.json'))};e.write(_repo_path(OUT, 'decision.json'),r);e.write(_repo_path(ROOT, 'reports/colocated-intensity-149-summary.json'),r);e.emit(r)
if __name__=='__main__':
 torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','extract','summary',*e.SPLITS,*['validate-'+v for v in e.SPLITS]]);v=p.parse_args().mode
 if v=='prepare':prepare()
 elif v=='extract':extract()
 elif v=='summary':summary()
 elif v.startswith('validate-'):validate(v[9:])
 else:run(v)
