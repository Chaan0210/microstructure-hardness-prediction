"""One fixed group-bootstrap aggregation of017's five existing residual candidates."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import spatial_joint_140 as q
import numpy as np,joblib,torch
x=q.x;a=q.a;e=q.e;n=a.n;OUT=_repo_path(ROOT, 'artifacts/bootstrap-selector-144');DRAWS=512;SEED=20271204;KINDS=tuple(n.KINDS)

def weights(loss,groups,seed):
 groups=np.asarray(groups);unique,inverse=np.unique(groups,return_inverse=True);g=len(unique);draw=np.random.default_rng(seed).integers(0,g,(DRAWS,g));counts=np.stack([np.bincount(row,minlength=g) for row in draw]);rowcounts=counts[:,inverse];totals=rowcounts@np.asarray(loss,float).T;wins=np.argmin(totals,axis=1);w=np.bincount(wins,minlength=5)/DRAWS;assert w.min()>=0 and w.sum()==1;return w,draw,wins

def independent_weights(loss,groups,draw):
 unique=np.unique(groups);totals=[]
 for sample in draw:
  sums=np.zeros(5)
  for gid in sample:sums+=loss[:,groups==unique[gid]].sum(1)
  totals.append(sums)
 wins=np.argmin(np.array(totals),axis=1);return np.array([(wins==i).mean() for i in range(5)]),wins

def prepare():
 q.check();OUT.mkdir(exist_ok=False);names=['bootstrap_selector_144.py','spatial_joint_140.py','component_ablation_121.py','nonlinear_residual_115.py','validate_nonlinear_residual_115.py','reports/residual-stability-143-diagnostic.json','reports/candidate017-public-submission.json','reports/candidate017-verification.json','artifacts/spatial-joint-140/features.npz','artifacts/spatial-joint-140/extraction.json']
 for split in e.SPLITS:
  names += [f'artifacts/spatial-joint-140/{split}/report.json',f'artifacts/spatial-joint-140/{split}/predictions.npz',f'artifacts/spatial-joint-140/validation-{split}.json']
 plan={'study':'144 bootstrap averaging of017 residual model selection','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'diagnostic':e.read(_repo_path(ROOT, 'reports/residual-stability-143-diagnostic.json'))['summary'],'hypothesis':'Argmin selection amongclose,correlated residualmodels on106/107honestrows mayamplifyvalidationnoise. Bootstrapselectionfrequencies maystabilize linear/nonlinear/zero complementarity withoutnewfeatures orhyperparameters. No guaranteedgain.','procedure':'For eachhonestB, reuse5xN innerOOF squaredloss from140. Sample G uniquegroups Gtimes withreplacement512draws; eachgroupdrawincludesallitsrows. For eachdraw chooseargminweightedloss, tieorder none,ridge1,ridge01,rbf1,rbf01. Normalizewinnercounts/512. Fit eachexistingcandidateonfullB, thenconvexweightedmean of5predictions. Same3honestblockmeans andunchanged015affinebase usedin017.','draws':DRAWS,'seed':'20271204+10000*splitindex+100*outerfold+block; full500 seed20271204+block. Seed/drawcount fixed,notselected.', 'nested':'WeightslearnedonlyfromBinnerOOF andBlabels. ModelsfitB; outerV excluded. HonestanchorA excludesallB; innerfit excludesitsownheldoutsubset. Selectorweights mayintentionallydependonallBvalidationlabels; do notclaiminnervalidationlabels excludedfromselectoritself. No global400OOFweightfit, no pooled320residualfit, no newneuralfit.', 'duplicate_audit':{'112':'Convexweight between011 andnewattentionbranch failed; differentobjects/lineage','117':'Physicalblock-additive RBF andhardselector, no acrosscandidatebootstrapaggregation','118':'Newhalfgrouptrainingdraws,diagnosticcompositionvariance;144doesnotresampletrainingfits','119':'Pooled320 versusblockresidualfit;144retainsB106/107','125':'Learned4basecomponentweights;144leavesallbaseweightsfixed','137':'Noquadraticcalibrationused'},'criterion':'Run all3preexistingpartitions beforedecision. Advance onlyifpooleduntrimmedMSE strictlybelow017 andindependentvalidationpasses. No foldwin/CIgate, fallbackuniformblend,temperature/shrinkage/grid/seedsearch.','validation':'Exact017hardwinnerreconstruction; reloadall225fullBmodels, independentlyderive trainlocalscaling/kernel/ridge; independentlyloopgroupbootstrapweights; allsourcehashes,IDgroups/exclusions; outertargetpoisononeblockeachsplit. Upstream1440innerfitsandlabelpoisoncertificates from140reused.','uncertainty':'10000paired400IDcluster bootstrapseed20271205; fixedfitconditional,overlappingpartitions,adaptivehistorynotaccountedfor.','resources':'ExistingCPU2, atmost180nonnullfullBfits lessreuse; noimages/holdout/test/newpackages/cost. Full500/testonlyafterpositiveverifiedlock; uploadparentonly.','hashes':{f:e.sha(_repo_path(ROOT, f)) for f in names},'protected':{f'artifacts/submission-candidate-{c}.csv':e.sha(_repo_path(ROOT, f'artifacts/submission-candidate-{c}.csv')) for c in ['015','016','017']},'cost':0};e.write(_repo_path(OUT, 'plan.json'),plan);e.emit({'locked144':True,'sha256':e.sha(_repo_path(OUT, 'plan.json')),'draws':DRAWS,'candidates':KINDS})

def check():
 p=e.read(_repo_path(OUT, 'plan.json'))
 for f,h in {**p['hashes'],**p['protected']}.items():assert e.sha(_repo_path(ROOT, f))==h,f
 return p

def data(split):
 d,z=q.load(split);d=q.augmented(d,z,'adjacent');r=e.read(q.OUT/split/'report.json');zz=a.readnp(q.OUT/split/'predictions.npz',r['predictions_sha256']);assert zz['ids'].tolist()==d['ids'];np.testing.assert_array_equal(zz['y'],d['y']);d['reference017']=zz['adjacent'];return d

def block(d,split,k,j,path):
 path.mkdir(exist_ok=False);src=q.OUT/split/f'adjacent-fold-{k}/block-{j}';rec=e.read(src/'report.json');bz=a.readnp(src/'predictions.npz',rec['predictions_sha256']);B=bz['block_indices'];V=bz['outer_indices'];res=d['y'][B]-bz['anchor_predictions'];np.testing.assert_array_equal(res,bz['residual']);assert not set(B)&set(V) and not set(d['g'][B])&set(d['g'][V]);loss=(bz['inner_predictions']-res)**2;seed=SEED+10000*e.SPLITS.index(split)+100*k+j;w,draw,wins=weights(loss,d['g'][B],seed);models=[];pred=[];hashes={};fits=0
 for ci,kind in enumerate(KINDS):
  if kind==rec['chosen']:
   f=src/'selected.joblib';assert e.sha(f)==rec['models_sha256'][f.name];obj=joblib.load(_repo_io(f));np.testing.assert_array_equal(obj['train_indices'],B);model=obj['model']
  else:model=n.fit(d['X'][B],res,d['blocks'],kind);fits+=kind!='none'
  file=path/f'{kind}.joblib';joblib.dump({'model':model,'train_indices':B},_repo_io(file));hashes[file.name]=e.sha(file);pred.append(n.predict(model,d['X'][V]))
 pred=np.stack(pred);chosen=int(np.argmin(loss.mean(1)));assert KINDS[chosen]==rec['chosen'];np.testing.assert_array_equal(pred[chosen],bz['correction']);primary=w@pred
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),block_indices=B,outer_indices=V,weights=w,draws=draw,winners=wins,full_candidate_predictions=pred,primary=primary,control=pred[chosen])
 report={'split':split,'fold':k,'block':j,'weights':dict(zip(KINDS,w.tolist())),'hardwinner':rec['chosen'],'seed':seed,'new_nonnull_fits':int(fits),'upstream_report_sha256':e.sha(src/'report.json'),'models_sha256':hashes,'predictions_sha256':e.sha(path/'predictions.npz')};e.write(path/'report.json',report);return primary,pred[chosen],report

def run(split):
 check();d=data(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);start=time.perf_counter();primary=np.full(400,np.nan);control=primary.copy();hashes={};records=[]
 for k in range(5):
  root=q.OUT/split/f'adjacent-fold-{k}';fr=e.read(root/'report.json');fz=a.readnp(root/'predictions.npz',fr['predictions_sha256']);V=fz['outer_indices'];cor=[];old=[]
  for j in range(3):
   pp,cc,br=block(d,split,k,j,out/f'fold-{k}-block-{j}');cor.append(pp);old.append(cc);records.append(br);hashes[f'fold-{k}-block-{j}/report.json']=e.sha(out/f'fold-{k}-block-{j}/report.json')
  primary[V]=fz['base']+np.mean(cor,0);control[V]=fz['base']+np.mean(old,0);e.emit({'split':split,'fold_complete':k})
 np.testing.assert_array_equal(control,d['reference017'])
 with (out/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],reference017=control,primary=primary)
 r={'split':split,'RMSE':{'reference017':e.rmse(d['y'],control),'primary':e.rmse(d['y'],primary)},'new_nonnull_fits':sum(v['new_nonnull_fits'] for v in records),'block_report_sha256':hashes,'predictions_sha256':e.sha(out/'predictions.npz'),'seconds':time.perf_counter()-start};e.write(out/'report.json',r);check();e.emit(r)

def validate(split):
 check();d=data(split);out=_repo_path(OUT, split);rr=e.read(out/'report.json');saved=a.readnp(out/'predictions.npz',rr['predictions_sha256']);assert saved['ids'].tolist()==d['ids'];np.testing.assert_array_equal(saved['y'],d['y']);count=0;maxerr=0.;agg=np.full(400,np.nan)
 for k in range(5):
  root=q.OUT/split/f'adjacent-fold-{k}';fr=e.read(root/'report.json');fz=a.readnp(root/'predictions.npz',fr['predictions_sha256']);cor=[]
  for j in range(3):
   path=out/f'fold-{k}-block-{j}';br=e.read(path/'report.json');assert e.sha(path/'report.json')==rr['block_report_sha256'][f'fold-{k}-block-{j}/report.json'];z=a.readnp(path/'predictions.npz',br['predictions_sha256']);src=root/f'block-{j}';old=e.read(src/'report.json');assert e.sha(src/'report.json')==br['upstream_report_sha256'];bz=a.readnp(src/'predictions.npz',old['predictions_sha256']);B=bz['block_indices'];V=bz['outer_indices'];res=bz['residual'];loss=(bz['inner_predictions']-res)**2;np.testing.assert_array_equal(z['block_indices'],B);np.testing.assert_array_equal(z['outer_indices'],V);w,wins=independent_weights(loss,d['g'][B],z['draws']);np.testing.assert_array_equal(w,z['weights']);np.testing.assert_array_equal(wins,z['winners']);np.testing.assert_array_equal(w,list(br['weights'].values()));_,draw,_=weights(loss,d['g'][B],br['seed']);np.testing.assert_array_equal(draw,z['draws']);pp=[]
   for ci,kind in enumerate(KINDS):
    f=path/f'{kind}.joblib';assert e.sha(f)==br['models_sha256'][f.name];obj=joblib.load(_repo_io(f));np.testing.assert_array_equal(obj['train_indices'],B);assert obj['model']['kind']==kind;manual=q.vn.manual(obj['model'],d['X'][B],res,d['X'][V],d['blocks']);actual=n.predict(obj['model'],d['X'][V]);err=float(abs(manual-actual).max());assert err<2e-9;maxerr=max(maxerr,err);np.testing.assert_array_equal(actual,z['full_candidate_predictions'][ci]);pp.append(actual);count+=1
   reconstructed=sum(wi*pi for wi,pi in zip(w,pp));np.testing.assert_allclose(reconstructed,z['primary'],atol=1e-12,rtol=0);np.testing.assert_array_equal(z['full_candidate_predictions'][KINDS.index(br['hardwinner'])],bz['correction']);cor.append(z['primary'])
  agg[fz['outer_indices']]=fz['base']+np.mean(cor,0)
 np.testing.assert_array_equal(agg,saved['primary']);pert=dict(d);pert['y']=d['y'].copy();pert['y'][d['fid']==0]+=100000;pp,cc,br=block(pert,split,0,0,_repo_path(OUT, f'poison-{split}'));old=e.read(out/'fold-0-block-0/report.json');assert br['weights']==old['weights'];z=a.readnp(out/'fold-0-block-0/predictions.npz',old['predictions_sha256']);np.testing.assert_array_equal(pp,z['primary']);check();r={'passed':True,'split':split,'models_verified':count,'independent_max_error':maxerr,'independent_group_bootstrap_exact':True,'outer_target_poison_exact':True,'017_control_exact':True,'upstream_innerfit_certificate':e.sha(q.OUT/f'validation-{split}.json'),'report_sha256':e.sha(out/'report.json')};e.write(_repo_path(OUT, f'validation-{split}.json'),r);e.emit(r)

def summary():
 check();errors={k:[] for k in ['reference017','primary']};part={}
 for split in e.SPLITS:
  r=e.read(_repo_path(OUT, split, 'report.json'));v=e.read(_repo_path(OUT, f'validation-{split}.json'));assert v['passed'] and v['report_sha256']==e.sha(_repo_path(OUT, split, 'report.json'));z=a.readnp(_repo_path(OUT, split, 'predictions.npz'),r['predictions_sha256']);part[split]=r['RMSE']
  for k in errors:errors[k].append((z[k]-z['y'])**2)
 errors={k:np.stack(v) for k,v in errors.items()};M={k:float(v.mean()) for k,v in errors.items()};rng=np.random.default_rng(20271205);g=[]
 for _ in range(10000):
  ids=rng.integers(0,400,400);g.append(float(np.sqrt(errors['reference017'][:,ids].mean())-np.sqrt(errors['primary'][:,ids].mean())))
 r={'study':'144 bootstrap selector aggregation','advance':M['primary']<M['reference017'],'validation_passed':True,'RMSE_by_partition':part,'pooled_MSE':M,'pooled_RMSE':{k:float(np.sqrt(v)) for k,v in M.items()},'conditional_gain_CI95':np.quantile(g,[.025,.975]).tolist(),'bootstrap_weights_are_not_posterior_probabilities':True,'same400_adaptive_overlap':True,'current_best':'017','test_or_holdout_read':False,'new_CSV':False,'cost':0,'plan_sha256':e.sha(_repo_path(OUT, 'plan.json'))};e.write(_repo_path(OUT, 'decision.json'),r);e.write(_repo_path(ROOT, 'reports/bootstrap-selector-144-summary.json'),r);e.emit(r)
if __name__=='__main__':
 torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','summary',*e.SPLITS,*['validate-'+v for v in e.SPLITS]]);mode=p.parse_args().mode
 if mode=='prepare':prepare()
 elif mode=='summary':summary()
 elif mode.startswith('validate-'):validate(mode[9:])
 else:run(mode)
