"""Label-blind finite sliced-W2 descriptor diagnostics; fixed B-final tokens."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,time,hashlib,datetime,csv,resource,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];assert Path(sys.prefix).resolve()==_repo_path(ROOT, '.venv')
import numpy as np
from scipy.special import ndtri
from scipy.spatial.distance import cdist,pdist
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
OUT=_repo_path(ROOT, 'artifacts/sliced-distribution-099');CACHE=_repo_path(ROOT, 'artifacts/patch-attention-092');SEED=20270409;N_DIR=32;P=(np.arange(17)+.5)/17

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,o):
 with Path(p).open('x') as f:json.dump(o,f,indent=2);f.write('\n')
def emit(o):print(json.dumps(o),flush=True)
def directions():
 d=np.random.default_rng(SEED).normal(size=(768,32));return d/np.linalg.norm(d,axis=0,keepdims=True)
def stats(proj):
 mu=proj.mean(0);std=proj.std(0);q=np.quantile(proj,P,axis=0,method='linear').T;shape=(q-mu[:,None])/np.maximum(std[:,None],1e-12)-ndtri(P)[None,:];return q,shape,mu,std

def prepare():
 OUT.mkdir(exist_ok=False);meta=read(_repo_path(CACHE, 'cache.json'));split=read(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json'));ids=sorted(split['development_folds']);assert ids==meta['ids'] and len(ids)==400 and not set(ids)&set(split['holdout']);assert sha(_repo_path(CACHE, 'patch_tokens.npy'))==meta['token_sha256'] and sha(_repo_path(CACHE, 'token_moments.npz'))==meta['moments_sha256']
 pub=read(_repo_path(ROOT, 'reports/candidate011-public-submission.json'))
 for p,h in pub['protected_sha256'].items():assert sha(_repo_path(ROOT, p))==h,p
 p={'study':'099 label-blind sliced distribution diagnostic','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Existing Mac-local development400 frozenB-final normalized1296x768 tokens only. No token upload, image decoding, hardness labels, holdout/test access, newinstallation/cost.','source':'https://proceedings.mlr.press/v162/meunier22b.html','reference_scope':'SW kernel distribution regression theory; no direct hardness or DINO performance evidence. Finite32directions/17quantiles is an approximation, not the exact integral or universal guarantee. Tokens share spatial/backbone context and are not independent samples.',
 'fixed_summary':{'seed':SEED,'directions':32,'direction_distribution':'768-dimensional standardGaussian independently normalized to unitL2; data-independent','probabilities':P.tolist(),'quantile_method':'numpy linear','dimension':544,'layer':'Existing092 final normalized ViT-B504 patch tokens only','distance':'Squared distance=mean over32x17 of squared rawquantile differences. Corresponds to fixed equal-weight quadrature of slicedW2 squared. No PCA, learned dictionary, channelstandardization or directionselection.'},
 'moment_controls':'Full original channelmean/populationstd1536 and matched projectedGaussian quantiles G=projectedmean+projectedstd*normal_inverse_CDF(P). Projectedstd contains covariance information, so a gain over channelmoments alone cannot be attributed purely to tails/multimodality.',
 'shape_diagnostic':'H=(Q-projectedmean)/projectedstd - normal_inverse_CDF(P), denominatorfloor1e-12. Per-image diagnostic normalization only; kernel candidate remains rawQ. H differences indicate non-Gaussian shape differences, not a validated mode count.',
 'fixed_regions':['checkerboard-even','checkerboard-odd','left-half','right-half','top-left','top-right','bottom-left','bottom-right'],
 'redundancy_diagnostic':'Use existing originaldevelopment folds, neverhardness. Reconstruct H and rawQ from originalM1536 with foldtrain StandardScaler+Ridge(alpha1536fixed). Report pooled multivariate R2 with foldtrain targetmean as null. This tests only thisfixedlinear mapping, not information-theoretic independence.',
 'nuisance_diagnostic':'From existing097 label-free CSV noiseHH and q95-q05contrast only. Foldtrain StandardScaler2+RBF KernelRidge(alpha1,gamma.5fixed) predicts H centered at foldtrainHmean. No targethardness or hyperparamsearch. OOF residualfraction is unaccounted for by thismodel, not proof of independence from all imagingnuisance.',
 'pair_diagnostic':'For eachimage nearestotherimage in rawchannelmean/std1536 distance, no global fittedscaler; save400pair distances inM, rawQ andH. These are approximatelymoment-matched, not exactlyequal.',
 'stability':'Compare medianRMS H difference in checkerboard halves with H difference of fixedcrossimage IDrotation by1. Compare medianRMS rawQ left/right difference with samecrossimage baseline. Quadrant heterogeneity descriptive. Global contextualtokens make subregions dependent.',
 'diagnostic_gate':{'synthetic_same_moments_shape_difference_min':.001,'moment_reconstruction_H_residual_fraction_min':.10,'nuisance_reconstruction_H_residual_fraction_min':.25,'checker_H_to_between_H_median_ratio_max':.50,'left_right_Q_to_between_Q_median_ratio_max':1.25,'nearestmoment_H_to_checker_H_median_ratio_min':2.0,'rule':'All required, plus finite/source/ID verification. Practical screening only; no postresult changes. Failure ends thisfixeddirection withoutCV.'},
 'synthetic_fixture':'Equalmean/variance rank1 token distributions: half-1/half+1 versus75%zero,12.5%-2,12.5%+2; commonfixed direction. Demonstrates mechanism inknownsyntheticcase only.',
 'CV_if_pass':'Separate locked2arm SWrawQ versus matchedGaussianQ kernel regression. Each kernel exp(-D2/(2*medianpositive_trainpairD2)), bandwidth recomputed strictly insideinnertrain/outertrain. Targetmean/std trainonly. Alpha [.01,.1,1.] inner3GroupKFold seed20270410+outerfold. Singleblend=.9*samefold011+.1*arm; SW mustbeatcontrolstandaloneandblend,improve011>=.02 with>=3folds,andimprovefixed395. Onlythenbothknownresplits unchanged. No rawmeanstd thirdcandidate, layer/seed/dimension/quantile/ratiosearch or first-orderfallback.',
 'resources':'CPU2; stream cachedtokens; RSS<3GB and extraction<300seconds. No additionaltokens/cachebackbone. No MPS needed.',
 'sources':{'sliced_distribution_099.py':sha(Path(__file__)),'validate_sliced_distribution_099.py':sha(_repo_path(ROOT, 'validate_sliced_distribution_099.py'))},'inputs':{str((_repo_path(CACHE, 'cache.json')).relative_to(ROOT)):sha(_repo_path(CACHE, 'cache.json')),str((_repo_path(CACHE, 'token_moments.npz')).relative_to(ROOT)):meta['moments_sha256'],'artifacts/run-001/split_manifest.json':sha(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')),'artifacts/grain-visual-audit-097/all400_noise_contrast.csv':sha(_repo_path(ROOT, 'artifacts/grain-visual-audit-097/all400_noise_contrast.csv'))},'token_sha256':meta['token_sha256'],'protected011':pub['protected_sha256'],'cost':0}
 write(_repo_path(OUT, 'plan.json'),p);np.save(_repo_path(OUT, 'directions.npy'),directions());emit({'prepared':True,'plan_sha256':sha(_repo_path(OUT, 'plan.json')),'gate':p['diagnostic_gate']})

def check():
 p=read(_repo_path(OUT, 'plan.json'))
 for n,h in {**p['sources'],**p['inputs'],**p['protected011']}.items():assert sha(_repo_path(ROOT, n))==h,n
 np.testing.assert_array_equal(np.load(_repo_path(OUT, 'directions.npy')),directions());return p

def extract():
 p=check();assert sha(_repo_path(CACHE, 'patch_tokens.npy'))==p['token_sha256'];T=np.load(_repo_path(CACHE, 'patch_tokens.npy'),mmap_mode='r');assert T.shape==(400,1296,768) and T.dtype==np.float32;meta=read(_repo_path(CACHE, 'cache.json'));D=directions();yy,xx=np.indices((36,36));masks=[(yy+xx)%2==0,(yy+xx)%2==1,xx<18,xx>=18,(yy<18)&(xx<18),(yy<18)&(xx>=18),(yy>=18)&(xx<18),(yy>=18)&(xx>=18)];masks=[m.ravel() for m in masks]
 with np.load(_repo_path(CACHE, 'token_moments.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==meta['ids'];old=z['moments'].copy()
 Q=[];H=[];G=[];M=[];RQ=[];RH=[];start=time.perf_counter();momenterr=0.
 for i in range(400):
  a=np.asarray(T[i],dtype=np.float64);mu=a.mean(0);second=(a*a).mean(0);momenterr=max(momenterr,float(abs(np.stack([mu,second])-old[i]).max()));np.testing.assert_array_equal(np.stack([mu,second]),old[i]);M.append(np.r_[mu,a.std(0)])
  projected=a@D;q,h,pm,ps=stats(projected);Q.append(q);H.append(h);G.append(pm[:,None]+ps[:,None]*ndtri(P)[None,:]);regional=[stats(projected[mask]) for mask in masks];RQ.append(np.stack([z[0] for z in regional]));RH.append(np.stack([z[1] for z in regional]));assert np.isfinite(projected).all()
  assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<3e9 and time.perf_counter()-start<300
  if (i+1)%50==0:emit({'extracted':i+1,'seconds':time.perf_counter()-start})
 with (_repo_path(OUT, 'summaries.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(meta['ids']),Q=np.stack(Q),H=np.stack(H),G=np.stack(G),M=np.stack(M),regionQ=np.stack(RQ),regionH=np.stack(RH),probabilities=P,directions=D)
 r={'seconds':time.perf_counter()-start,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'source_moments_max_error':momenterr,'summary_sha256':sha(_repo_path(OUT, 'summaries.npz')),'directions_sha256':sha(_repo_path(OUT, 'directions.npy')),'cache_shape':list(T.shape),'no_labels_accessed':True};write(_repo_path(OUT, 'extraction.json'),r);emit(r)

def reconstruct(X,Y,fid,kind):
 pred=np.empty_like(Y);null=np.empty_like(Y);details=[]
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);sc=StandardScaler().fit(X[tr]);mean=Y[tr].mean(0)
  model=Ridge(alpha=1536.) if kind=='moments' else KernelRidge(alpha=1.,kernel='rbf',gamma=.5)
  model.fit(sc.transform(X[tr]),Y[tr]-mean);pred[va]=model.predict(sc.transform(X[va]))+mean;null[va]=mean;details.append({'fold':k,'train_indices':tr.tolist(),'validation_indices':va.tolist(),'scaler_mean':sc.mean_.tolist(),'scaler_scale':sc.scale_.tolist()})
 err=float(np.sum((Y-pred)**2));den=float(np.sum((Y-null)**2));assert den>0
 return {'OOF_R2':1-err/den,'residual_fraction':err/den,'SSE':err,'null_SSE':den},pred,details

def diagnose():
 plan=check();ex=read(_repo_path(OUT, 'extraction.json'));assert sha(_repo_path(OUT, 'summaries.npz'))==ex['summary_sha256']
 with np.load(_repo_path(OUT, 'summaries.npz'),allow_pickle=False) as z:ids=z['ids'].tolist();Q=z['Q'].reshape(400,-1);H=z['H'].reshape(400,-1);G=z['G'].reshape(400,-1);M=z['M'].copy();rq=z['regionQ'];rh=z['regionH']
 sp=read(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json'));assert ids==sorted(sp['development_folds']);fid=np.array([sp['development_folds'][i] for i in ids]);assert not set(ids)&set(sp['holdout'])
 with (_repo_path(ROOT, 'artifacts/grain-visual-audit-097/all400_noise_contrast.csv')).open() as f:nuis={r['ID']:[float(r['noise']),float(r['contrast_q95_q05'])] for r in csv.DictReader(f)}
 N=np.array([nuis[i] for i in ids]);start=time.perf_counter();mr,mp,md=reconstruct(M,H,fid,'moments');qr,qp,qd=reconstruct(M,Q,fid,'moments');nr,npred,nd=reconstruct(N,H,fid,'nuisance')
 rms=lambda x:np.sqrt(np.mean(x*x,axis=tuple(range(1,x.ndim))))
 betweenH=rms(H-np.roll(H,1,axis=0));betweenQ=rms(Q-np.roll(Q,1,axis=0));checker=rms(rh[:,0]-rh[:,1]);leftright=rms(rq[:,2]-rq[:,3]);quadrant=rms(rq[:,4]-rq[:,7]);dm=cdist(M,M,'sqeuclidean')/M.shape[1];np.fill_diagonal(dm,np.inf);near=dm.argmin(1);matchedH=rms(H-H[near]);matchedQ=rms(Q-Q[near]);ratios={'checker_H_to_between_H':float(np.median(checker)/np.median(betweenH)),'left_right_Q_to_between_Q':float(np.median(leftright)/np.median(betweenQ)),'nearestmoment_H_to_checker_H':float(np.median(matchedH)/np.median(checker)),'diagonalquadrant_Q_to_between_Q':float(np.median(quadrant)/np.median(betweenQ))}
 # Exact first/second moment agreement with different synthetic distribution shapes.
 aa=np.r_[np.full(648,-1.),np.full(648,1.)];bb=np.r_[np.zeros(972),np.full(162,-2.),np.full(162,2.)];v=directions()[:,0];ta=aa[:,None]*v[None,:];tb=bb[:,None]*v[None,:];np.testing.assert_allclose(ta.mean(0),tb.mean(0),rtol=0,atol=1e-12);np.testing.assert_allclose(ta.std(0),tb.std(0),rtol=0,atol=1e-12);qa,ha,ma,sa=stats(ta@directions());qb,hb,mb,sb=stats(tb@directions());synthetic={'channel_mean_max_error':float(abs(ta.mean(0)-tb.mean(0)).max()),'channel_std_max_error':float(abs(ta.std(0)-tb.std(0)).max()),'quantile_RMS_difference':float(np.sqrt(np.mean((qa-qb)**2))),'standardized_shape_RMS_difference':float(np.sqrt(np.mean((ha-hb)**2)))}
 gate=plan['diagnostic_gate'];checks={'synthetic_same_moments_different_shape':synthetic['quantile_RMS_difference']>=gate['synthetic_same_moments_shape_difference_min'],'not_reconstructed_by_fixed_moment_model':mr['residual_fraction']>=gate['moment_reconstruction_H_residual_fraction_min'],'not_reconstructed_by_fixed_nuisance_model':nr['residual_fraction']>=gate['nuisance_reconstruction_H_residual_fraction_min'],'checker_stability':ratios['checker_H_to_between_H']<=gate['checker_H_to_between_H_median_ratio_max'],'geographic_stability':ratios['left_right_Q_to_between_Q']<=gate['left_right_Q_to_between_Q_median_ratio_max'],'momentmatched_shape_above_sampling_difference':ratios['nearestmoment_H_to_checker_H']>=gate['nearestmoment_H_to_checker_H_median_ratio_min']}
 with (_repo_path(OUT, 'moment-neighbor-diagnostics.csv')).open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','nearest_moment_ID','moment_RMS','rawQ_RMS','shapeH_RMS','own_checkerH_RMS']);w.writerows((ids[i],ids[near[i]],float(np.sqrt(dm[i,near[i]])),float(matchedQ[i]),float(matchedH[i]),float(checker[i])) for i in range(400))
 with (_repo_path(OUT, 'reconstruction_predictions.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),fold=fid,M_to_H=mp,M_to_Q=qp,N_to_H=npred)
 write(_repo_path(OUT, 'reconstruction-audit.json'),{'M_to_H':md,'M_to_Q':qd,'N_to_H':nd})
 shapeamp=rms(H);assoc={'shape_amplitude_noise_spearman':float(spearmanr(shapeamp,N[:,0]).statistic),'shape_amplitude_contrast_spearman':float(spearmanr(shapeamp,N[:,1]).statistic)}
 r={'labelblind_gate_pass':all(checks.values()),'gate_checks':checks,'M_to_H':mr,'M_to_Q':qr,'noise_contrast_to_H':nr,'region_ratios':ratios,'shape_amplitude_associations':assoc,'synthetic':synthetic,'median_distances':{'betweenH':float(np.median(betweenH)),'betweenQ':float(np.median(betweenQ)),'checkerH':float(np.median(checker)),'left_rightQ':float(np.median(leftright)),'nearestmomentH':float(np.median(matchedH))},'rawQ_vs_matchedGaussian_RMS_median':float(np.median(rms(Q-G))),'seconds':time.perf_counter()-start,'summary_sha256':ex['summary_sha256'],'reconstruction_predictions_sha256':sha(_repo_path(OUT, 'reconstruction_predictions.npz')),'plan_sha256':sha(_repo_path(OUT, 'plan.json')),'interpretation':'Syntheticfixture proves the representation can distinguish equalmoments. Actualnearestmomentpairs are approximate. Crossfitremainingvariance only rejects the specified reconstruction models, not arbitrary nonlinear dependence or all imagingnuisance. Geographical differences may reflect genuine heterogeneity, not just samplingnoise.','no_hardness_labels_used':True,'CV_started':False};write(_repo_path(OUT, 'diagnostic-report.json'),r);check();emit(r)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','extract','diagnose']);a=ap.parse_args();{'prepare':prepare,'extract':extract,'diagnose':diagnose}[a.mode]()
