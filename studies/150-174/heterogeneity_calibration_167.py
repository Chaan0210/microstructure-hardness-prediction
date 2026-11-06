"""Case-motivated one-term conditional common-mode calibration; full018 nesting."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import quality_gate_165 as t
import numpy as np,json,sys,time
from pathlib import Path
R=t.ROOT;e=t.e;a=t.a;s=t.s;x=t.x;OUT=R/'artifacts/heterogeneity-calibration-167';LAMBDA=1.;MODE='interaction'
t.OUT=OUT

def quality(ids):
 p=R/'artifacts/image-diagnostics-094/image_metrics.npz';assert e.sha(p)==e.read(p.with_name('matching-locked.json'))['metrics_sha256']
 with np.load(_repo_io(p),allow_pickle=False) as z:
  ix={str(v):i for i,v in enumerate(z['ids'])};q=z['X'][[ix[i] for i in ids],z['names'].tolist().index('tile_mean_std')]
 return q[:,None]
t.quality=quality

def meta(P,y,Q,qt,qv,enabled=True):
 p=a.mixture(P,x.ARM);pq=a.mixture(Q,x.ARM);pm=float(p.mean());ps=float(p.std());qm=float(qt.mean());qs=float(qt.std());ps=max(ps,1e-12);qs=max(qs,1e-12);zp=(p-pm)/ps;zv=(pq-pm)/ps;zq=(qt[:,0]-qm)/qs;zqv=(qv[:,0]-qm)/qs
 base,cal=a.affine(p,y,pq);train,_=a.affine(p,y,p)
 if MODE=='interaction':Z=np.column_stack([np.ones(len(p)),zp,zq]);ZV=np.column_stack([np.ones(len(pq)),zv,zqv]);h=zp*zq;hv=zv*zqv
 elif MODE=='quadratic':Z=np.column_stack([np.ones(len(p)),zp]);ZV=np.column_stack([np.ones(len(pq)),zv]);h=zp*zp;hv=zv*zv
 else:raise ValueError(MODE)
 projection=np.linalg.lstsq(Z,h,rcond=None)[0];b=h-Z@projection;bv=hv-ZV@projection;scale=float(np.sqrt(np.mean(b*b)));degenerate=scale<1e-12
 if degenerate:b=np.zeros_like(b);bv=np.zeros_like(bv);scale=1.
 else:b=b/scale;bv=bv/scale
 coef=float(np.mean((y-train)*b)/(np.mean(b*b)+LAMBDA)) if enabled else 0.;pred=base+coef*bv
 m={'basis':MODE,'enabled':enabled,'n_train':len(y),'p_mean':pm,'p_std':ps,'q_mean':qm,'q_std':qs,'projection':projection.tolist(),'basis_scale':scale,'degenerate':degenerate,'coefficient':coef,'lambda':LAMBDA,'extra_effective_df':float(np.mean(b*b)/(np.mean(b*b)+LAMBDA)),'calibrator':cal,'query_correction_range':[float((coef*bv).min()),float((coef*bv).max())]};assert np.isfinite(pred).all();return pred,m
t.meta=meta

def prepare():
 OUT.mkdir(exist_ok=False);review=e.read(R/'artifacts/failure-review-166/report.json');assert review['n_unique']==400 and review['no_training'];old=e.read(R/'artifacts/quality-gate-165/plan.json')
 files=['heterogeneity_calibration_167.py','quality_gate_165.py','artifacts/failure-review-166/report.json','artifacts/failure-review-166/analysis-plan.json','artifacts/image-diagnostics-094/image_metrics.npz','artifacts/image-diagnostics-094/matching-locked.json','reports/problem-definition-111-summary.json','reports/quality-gate-165-summary.json','reports/nested-selector-146-audit.json']
 plan={'study':167,'hypothesis':'Image-scale brightness heterogeneity marks variation in common prediction-range compression. Correct one score×heterogeneity interaction rather than choose among similarly biased branches.','evidence':'166:33persistentlarge cases all targetoutside KRR/CNN/DINOmean hull on same side inall3splits; only3have persistenttop10%branchdisagreement. tile_mean_std highestquartile n100 RMSE15.38 and residual-on-score slopes .246,.266,.349; two sided errors cancel mean.','selected_single_feature':'tile_mean_std: std of64native32x32 tilemean intensities, cachedimage-local. Not truegrain variance or pureblur.','primary':'One standardized, train-orthogonalized product of centeredraw018mixture score and standardizedtile_mean_std. Orthogonalize product to[1,score,quality] oncurrentmetaOOFtrain; normalizeRMS; fitonlyone extra coefficient using mean squaredloss +1*coefficient². Existingaffine retained;1/2OLS coefficient when nondegenerate. No sign constraint.','control':'Matched global score-squared term, orthogonalized against[1,score],same oneextra coefficient+lambda1 and same downstream018refits. Controlonly; never fallbackcandidate. This repeats known137basis only to distinguish qualityinteraction from generic curvature on current018.','duplicate_review':{'073_074':'Globalaffine, no spatialquality interaction','137':'Score-only quadratic, represented by matchedcontrol','165':'Quality×CNN-other disagreement; cannot move common-mode when branchcontrast zero','113_140_149':'Physics/adjacency/intensity residualbanks; no this trainlocal scalar score×spatialheterogeneity calibration. Not claiming information independent of image/deepfeatures.'},'nesting':'Reuse165 full018 honestnesting: outergate uses320middleOOF; eachA213/214 anchor usesdeepOOF from142/143, fits stats/projection/coefficient onlyA. RetainshortB.1. RefitB106/107 residual585 with5candidates/512bootstrap, no pooledglobalOOF residuals. Both arms same scopes/seeds.','resplit_gate':'OriginalinteractionRMSE below BOTH unchanged018 and matchedquadratic; >=3/5folds improve versus018; auditrequired. Only sameinteraction and matchedcontrol then evaluatedonboth existingresplits.','final_gate':'If extended pooleduntrimmedMSE belowbothcontrols, allsource/lineage/reload/poisonchecks passed. Same400posthoc exploration; no independent generalization claim.','failure':'Ifgatefails closeboth arms, no alternativetile size,feature,lambda,sign,scoretransformation orblend sweep; nofull500/test/newCSV.','resource':'ExistinguvCPU2, noinstall/download/externaldata, noimageextraction forfit. First pairedfold<120s each,RSS<4GB.','source_hashes':{f:e.sha(R/f) for f in files},'protected':old['protected'],'visual_review':'27outcome-selected developmentimages viewed at originalpixelintensities. Visuallysimilarlynoisy controls hadsmallerrors; neither denoising nor predetermined phase/pore sign justified.','bias':'Hypothesis chosen after checking outcomes ofsame400/repeatedsplits and many prior experiments; any new gain explicitly exploratory.'};t.write(OUT/'plan.json',plan);print('PLAN_LOCKED',e.sha(OUT/'plan.json'),flush=True)

def run(split,benchmark=False):
 global MODE
 plan=e.read(OUT/'plan.json')
 for f,h in {**plan['source_hashes'],**plan['protected']}.items():assert e.sha(R/f)==h,f
 if split!='original':assert e.read(OUT/'original/report.json')['advance'] and e.read(OUT/'validation-original.json')['passed']
 d=t.load(split);out=OUT/split;out.mkdir(exist_ok=True);pred={m:np.full(400,np.nan) for m in ['interaction','quadratic']}
 for k in ([0] if benchmark else range(5)):
  dd,_=x.replace(d,split,k,x.OUT/split/f'cnn-fold-{k}',False);V=np.flatnonzero(d['fid']==k)
  if split=='original' and k==0 and not (OUT/'control018-fold-0/report.json').exists():
   MODE='interaction';control,_=t.fold(dd,split,k,OUT/'control018-fold-0',False);np.testing.assert_allclose(control,d['reference018'][V],atol=1e-9,rtol=0)
  for MODE in ['interaction','quadratic']:
   path=out/f'{MODE}-fold-{k}'
   if (path/'report.json').exists():r=e.read(path/'report.json');p=a.readnp(path/'predictions.npz',r['predictions_sha256'])['primary']
   else:print('START',MODE,split,k,flush=True);p,r=t.fold(dd,split,k,path)
   assert r['seconds']<120 and r['peak_rss_bytes']<4e9;pred[MODE][V]=p
 if benchmark:return
 scores={'reference018':e.rmse(d['y'],d['reference018']),**{m:e.rmse(d['y'],p) for m,p in pred.items()}};gains=[e.rmse(d['y'][d['fid']==k],d['reference018'][d['fid']==k])-e.rmse(d['y'][d['fid']==k],pred['interaction'][d['fid']==k]) for k in range(5)]
 np.savez_compressed(_repo_io(out/'predictions.npz'),ids=np.array(d['ids']),groups=d['g'],y=d['y'],fold=d['fid'],quality=d['quality'],reference018=d['reference018'],**pred)
 r={'split':split,'RMSE':scores,'fold_gains_vs018':gains,'advance':scores['interaction']<min(scores['reference018'],scores['quadratic']) and sum(g>0 for g in gains)>=3,'predictions_sha256':e.sha(out/'predictions.npz'),'fold_reports':{f'{m}-fold-{k}/report.json':e.sha(out/f'{m}-fold-{k}/report.json') for m in pred for k in range(5)},'read_input_sha256':dict(a.SOURCES)};t.write(out/'report.json',r);print('RESULT',json.dumps({k:v for k,v in r.items() if k not in ['read_input_sha256','fold_reports']}),flush=True)
if __name__=='__main__':
 m=sys.argv[1]
 if m=='prepare':prepare()
 elif m=='benchmark':run('original',True)
 else:run(m)
