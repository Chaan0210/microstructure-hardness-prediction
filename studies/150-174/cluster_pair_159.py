"""Single predeclared C2 versus matched S2 addition to the nested018 procedure."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,datetime,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import directional_energy_148 as core
import cluster_pair_features_159 as f
import numpy as np,torch,joblib
from PIL import Image
s=core.s;a=core.a;e=core.e;OUT=_repo_path(ROOT, 'artifacts/cluster-pair-159');ARMS=('s2','c2')

def prepare():
 s.check();OUT.mkdir(exist_ok=False)
 names=['cluster_pair_features_159.py','cluster_pair_159.py','directional_energy_148.py','bootstrap_selector_144.py','connected_grain_features.py','lineal_features_054.py','betti_features_055.py','binary_threepoint_features_130.py','src/microstructure_v2/features.py','src/microstructure_v2/extra.py','reports/nested-selector-146-audit.json','reports/aligned-crossfit-158-summary.json','reports/candidate020-summary.json','artifacts/run-001/split_manifest.json','artifacts/nested-calibration-074/extraction.json']
 for split in e.SPLITS:names += [f'artifacts/bootstrap-selector-144/{split}/report.json',f'artifacts/bootstrap-selector-144/{split}/predictions.npz',f'artifacts/bootstrap-selector-144/validation-{split}.json']
 plan={'study':'159 C2 same-component probability versus matched S2','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'definition':'For inferred binary mask M, label4-connected components within observed image. C2(delta)=sum_x 1[label(x)=label(x+delta)>0]/Nvalid(delta); S2(delta)=sum_x M(x)M(x+delta)/sameNvalid. NO conditional normalization byS2, no wrapping, no smoothing/filtering/closing/denoising. Paths may bend anywhere INSIDE image; outside connectivity is unobserved.','features':{'quantiles':list(f.QUANTILES),'threshold_method':'Each rawuint8image linearquantile, low<=threshold, complement>threshold;tiesretained. No dataset/label fit.','sides':['low','complement'],'connectivity':4,'distances_native_pixels':list(f.RADII),'directions':[[1,0],[0,1]],'summary':'axis mean and populationstd,36features per arm. No diagonals/interpolation. D4invariant; orientation relative toload discarded.','rationale':'Three fixed inherited20/50/80quantiles and sparse4/16/64pixel lags span short/intermediate/quarter-image separations. No image/target outcome used tochoose them.','interpretation':'Estimated binary-region connectivity statistics; NOT identified physical phases, grains, pores, true material topology or mechanicalcausality.'},'audit':{'028':'TV+ridge linking+complement component22summaries. Label operation only for counts/areas/shapes/network; never same-component separated-pair curve.','054':'All pixels on sampled straight segment occupied, versus any4-neighbor path forC2. For axial integerlag same mask/domain: L2<=C2<=S2; inequality not claimed for054interpolateddiagonals.','055':'Counts ofcomponents/holes underdistance excursions; not distance-conditioned within-component paircounts.','130':'Centered endpoint pair/triple products; never asks whether endpoints sharea connectedcomponent. Current matchedS2 is rawprobability onexactC2mask/lags, NOT retry130triplebank.','other_source':'src/microstructure_v2/features.py andextra.py inspected:autocorrelations,componentmarginalshapes/areas,not C2. Source searches foundno directC2implementation.','proof':'Deterministic homometric masks have identical all-vectorS2 and same componentarea multiset/zero-levelBetti butdifferentC2. U-shapefixture path withoutstraightline. BFSindependentverification. No claim jointly independentofallpast descriptors or useful hardnesssignal.'},'model':'Exact018nested procedure, existing585 plus ONE36wide block; same dimensions621 forS2andC2. C2isprimary; S2isdiagnosticmatchedcontrol and neverfallback. No concatenation, residualpooling, newneuraltraining or previousfailedbankretuning.','nesting':'OuterT320/V80 unchangedbase. A213/214 disjointfromB106/107andV supplies honesttargets viaA-onlydeepOOFcalibration. WithinB: residualfit/scalers onlyI70-72; heldoutH35-36 usedonlyOOFloss/bootstrapweights. Fiveexistingkinds none,ridge1,ridge01,rbf1,rbf01,512draws/sameseeds. FullresidualfitonlyB. No globalOOFtarget/weight fit.','criterion':'Runall3existingpartitions. PrimaryC2 pooled1200row untrimmedMSE mustbeatBOTH018 andS2 plusindependentverification. No foldcount/minimumgain/CIgate, no posthocquantile/lag/connectivity/weight/family exploration. Ifnegativeclosethisbank. Positive result initially remainsdevelopmentcandidate; no uploadtoday.','uncertainty':'10000pairedIDgroupbootstrap seed20271220; threeerrors perIDkepttogether. Same400historicallyreused,overlappingpartitions,conditionalfixedfits,notfreshvalidation.','resource':'ExistinguvCPU2/scipy/PIL/numpy/sklearn;zero installs/cost. Benchmarkfirst10imageextraction projected400<300s,RSS<2GB. Benchmarkfirstarmblock includedin1800objects projected90blocks<300s,RSS<2GB.1800objects include360none,1440nonnullfits, plus6outerpoisonblocks and30innerpoisonfits. Existing45upstreamanchors reused; no upstreamretraining.','scope':'Only400developmentimages/labels. No100holdout/testread,full500fit,CSV,uploadtoday. Keep018best and020weaksecondary.','hashes':{z:e.sha(_repo_path(ROOT, z)) for z in names},'protected':{str(p.relative_to(ROOT)):e.sha(p) for p in [_repo_path(ROOT, 'artifacts/submission-candidate-018.csv'),_repo_path(ROOT, 'artifacts/submission-candidate-019.csv'),_repo_path(ROOT, 'artifacts/submission-candidate-020.csv'),_repo_path(ROOT, 'deliverables/reproduction-018-v155.zip'),_repo_path(ROOT, 'deliverables/reproduction-011-014-015-v136.zip')]},'cost':0}
 e.write(_repo_path(OUT, 'plan.json'),plan);e.write(_repo_path(ROOT, 'reports/cluster-pair-159-design.json'),plan);e.write(_repo_path(OUT, 'fixtures.json'),f.fixtures());e.emit({'locked159':True,'fixture_passed':True,'features_per_arm':36,'new_neural_fits':0,'plan_sha256':e.sha(_repo_path(OUT, 'plan.json'))})

def check():
 p=e.read(_repo_path(OUT, 'plan.json'))
 for file,h in {**p['hashes'],**p['protected']}.items():assert e.sha(_repo_path(ROOT, file))==h,file
 return p

def extract():
 check();ids=sorted(e.read(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json'))['development_folds']);assert len(ids)==400
 expected=e.read(_repo_path(ROOT, 'artifacts/nested-calibration-074/extraction.json'))['image_sha256'];S=[];C=[];meta=[];maxerr=0.;start=time.perf_counter()
 for i,ident in enumerate(ids):
  path=_repo_path(ROOT, 'data/train', f'{ident}.png');assert e.sha(path)==expected[ident]
  with Image.open(_repo_io(path)) as im:assert im.size==(256,256) and im.mode=='L';image=np.array(im,dtype=np.uint8)
  sp,cp,m=f.extract(image);S.append(sp);C.append(cp);meta.append(m)
  if i==9:
   report={'first10_seconds':time.perf_counter()-start,'projected400_seconds':40*(time.perf_counter()-start),'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};report['passed']=report['projected400_seconds']<300 and report['RSS_bytes']<2e9;e.write(_repo_path(OUT, 'feature-benchmark.json'),report);e.emit(report);assert report['passed']
  if i in [0,199,399]:
   for t in m['thresholds']:
    mask=image<=t
    for mm in [mask,~mask]:
     ss,cc,_=f.panel(mm);ii,jj,_=f.independent(mm);err=max(float(abs(ss-ii).max()),float(abs(cc-jj).max()));assert err<1e-15;maxerr=max(maxerr,err)
   for k in range(4):
    for mm in [np.rot90(image,k),np.rot90(image,k)[:,::-1]]:
     ss,cc,_=f.extract(mm);np.testing.assert_array_equal(ss,sp);np.testing.assert_array_equal(cc,cp)
  if (i+1)%80==0:e.emit({'features_complete':i+1,'seconds':time.perf_counter()-start})
 S=np.array(S);C=np.array(C);assert S.shape==C.shape==(400,36);assert np.max(abs(S-C))>0
 with (_repo_path(OUT, 'features.npz')).open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(ids),s2=S,c2=C,names=np.array(f.NAMES))
 report={'passed':True,'rows':400,'width_each':36,'image_hashes_verified':400,'cache_sha256':e.sha(_repo_path(OUT, 'features.npz')),'independent_real_BFS_max_error':maxerr,'real_D4_images':3,'seconds':time.perf_counter()-start,'constant_columns':{'s2':int((S.std(0)==0).sum()),'c2':int((C.std(0)==0).sum())},'rms_C2_S2_difference':float(np.sqrt(np.mean((C-S)**2))),'metadata':meta,'label_free':True};e.write(_repo_path(OUT, 'extraction.json'),report);check();e.emit({k:v for k,v in report.items() if k!='metadata'})

def data(split,arm):
 d=s.data(split);rr=e.read(s.OUT/split/'report.json');zz=a.readnp(s.OUT/split/'predictions.npz',rr['predictions_sha256']);assert zz['ids'].tolist()==d['ids'];d['reference018']=zz['primary'];z=a.readnp(_repo_path(OUT, 'features.npz'),e.read(_repo_path(OUT, 'extraction.json'))['cache_sha256']);assert z['ids'].tolist()==d['ids'];d['X']=np.column_stack([d['X'],z[arm]]);d['blocks']=list(d['blocks'])+['binary_pair']*36;return d

def run(split):
 check();D={arm:data(split,arm) for arm in ARMS};d=D['c2'];out=_repo_path(OUT, split);out.mkdir();P={arm:np.full(400,np.nan) for arm in ARMS};hashes={};start=time.perf_counter()
 for k in range(5):
  src=s.q.OUT/split/f'adjacent-fold-{k}';fr=e.read(src/'report.json');fz=a.readnp(src/'predictions.npz',fr['predictions_sha256']);V=fz['outer_indices'];cor={arm:[] for arm in ARMS};control=[]
  for j in range(3):
   br=e.read(src/f'block-{j}/report.json');bz=a.readnp(src/f'block-{j}/predictions.npz',br['predictions_sha256']);B=bz['block_indices'];assert not set(B)&set(V)
   for arm in ARMS:
    blockstart=time.perf_counter();path=out/f'{arm}-fold-{k}-block-{j}';pp,r=core.system(D[arm],B,bz['anchor_predictions'],V,a.r.CVSEED+100*k+j,s.SEED+10000*e.SPLITS.index(split)+100*k+j,path);cor[arm].append(pp);hashes[path.name+'/report.json']=e.sha(path/'report.json')
    if split=='original' and k==j==0 and arm==ARMS[0]:
     b={'first_armblock_seconds':time.perf_counter()-blockstart,'projected90_blocks_seconds':90*(time.perf_counter()-blockstart),'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};b['passed']=b['projected90_blocks_seconds']<300 and b['RSS_bytes']<2e9;e.write(_repo_path(OUT, 'model-benchmark.json'),b);e.emit(b);assert b['passed']
   old=e.read(s.OUT/split/f'fold-{k}-block-{j}/report.json');oz=a.readnp(s.OUT/split/f'fold-{k}-block-{j}/predictions.npz',old['predictions_sha256']);control.append(oz['primary'])
  np.testing.assert_array_equal(fz['base']+np.mean(control,0),d['reference018'][V])
  for arm in ARMS:P[arm][V]=fz['base']+np.mean(cor[arm],0)
  e.emit({'split':split,'fold_complete':k})
 with (out/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],reference018=d['reference018'],**P)
 r={'split':split,'RMSE':{'reference018':e.rmse(d['y'],d['reference018']),**{arm:e.rmse(d['y'],v) for arm,v in P.items()}},'models_including_none':600,'block_reports_sha256':hashes,'predictions_sha256':e.sha(out/'predictions.npz'),'seconds':time.perf_counter()-start};e.write(out/'report.json',r);check();e.emit(r)

def validate_main(split):
 check();D={arm:data(split,arm) for arm in ARMS};d=D['c2'];out=_repo_path(OUT, split);rr=e.read(out/'report.json');saved=a.readnp(out/'predictions.npz',rr['predictions_sha256']);assert saved['ids'].tolist()==d['ids'];np.testing.assert_array_equal(saved['y'],d['y']);count=0;maxerr=0.
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

def validate(split):
 validate_main(split);count=0
 for arm in ARMS:
  d=data(split,arm);path=_repo_path(OUT, split, f'{arm}-fold-0-block-0');r=e.read(path/'report.json');z=a.readnp(path/'predictions.npz',r['predictions_sha256']);it=np.array(r['splits'][0]['train_indices']);iv=np.array(r['splits'][0]['validation_indices']);B=z['block_indices'];lookup={int(v):i for i,v in enumerate(B)};res=d['y'].copy();res[B]=z['residual'];res[iv]+=23456
  for ci,kind in enumerate(s.KINDS):
   m=core.n.fit(d['X'][it],res[it],d['blocks'],kind);p=core.n.predict(m,d['X'][iv]);expected=z['inner_predictions'][ci,[lookup[int(v)] for v in iv]];np.testing.assert_array_equal(p,expected);count+=1
 e.write(_repo_path(OUT, f'inner-poison-{split}.json'),{'passed':True,'fresh_fits':count,'H_labels_excluded_from_innerfit':True,'scope':'CachedA anchors excludeallB; perturbHresiduals andrefitI. Selectorweights intentionallydependonH OOF loss.'});check();e.emit({'inner_poison_passed':True,'split':split,'fresh_fits':count})

def summary():
 check();errors={k:[] for k in ['reference018',*ARMS]};scores={}
 for split in e.SPLITS:
  r=e.read(_repo_path(OUT, split, 'report.json'));assert e.read(_repo_path(OUT, f'inner-poison-{split}.json'))['passed'];v=e.read(_repo_path(OUT, f'validation-{split}.json'));assert v['passed'] and v['report_sha256']==e.sha(_repo_path(OUT, split, 'report.json'));z=a.readnp(_repo_path(OUT, split, 'predictions.npz'),r['predictions_sha256']);scores[split]=r['RMSE']
  for k in errors:errors[k].append((z[k]-z['y'])**2)
 errors={k:np.stack(v) for k,v in errors.items()};M={k:float(v.mean()) for k,v in errors.items()};assert len(np.unique(s.data('original')['g']))==400;rng=np.random.default_rng(20271220);g={k:[] for k in ['reference018','s2']}
 for _ in range(10000):
  ids=rng.integers(0,400,400)
  for k in g:g[k].append(float(np.sqrt(errors[k][:,ids].mean())-np.sqrt(errors['c2'][:,ids].mean())))
 r={'study':'159 C2 versus same-mask S2','advance':M['c2']<min(M['reference018'],M['s2']),'validation_passed':True,'RMSE_by_partition':scores,'pooled_MSE':M,'pooled_RMSE':{k:float(np.sqrt(v)) for k,v in M.items()},'conditional_gain_CI95':{k:np.quantile(v,[.025,.975]).tolist() for k,v in g.items()},'same400_adaptive_history':True,'no_variants_if_failure':True,'current_best':'018','test_or_holdout_read':False,'new_CSV':False,'cost':0,'plan_sha256':e.sha(_repo_path(OUT, 'plan.json'))};e.write(_repo_path(OUT, 'decision.json'),r);e.write(_repo_path(ROOT, 'reports/cluster-pair-159-summary.json'),r);e.emit(r)
if __name__=='__main__':
 torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','extract','summary',*e.SPLITS,*['validate-'+v for v in e.SPLITS]]);v=p.parse_args().mode
 if v=='prepare':prepare()
 elif v=='extract':extract()
 elif v=='summary':summary()
 elif v.startswith('validate-'):validate(v[9:])
 else:run(v)
