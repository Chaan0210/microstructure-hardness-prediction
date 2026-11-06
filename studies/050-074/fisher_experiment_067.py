"""Single DINO-S final504 Fisher512 pilot; dictionary strictly outertrain-only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,time,resource,argparse,hashlib
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import numpy as np,torch,joblib
import dino_spatial_039 as ds
import dinob_complement_064 as prior
import fisher_pooling_067 as f
import test_fisher_pooling_067 as fixtures
fv=prior.fv;OUT=_repo_path(ROOT, 'artifacts/fisher-067')

def streamsha(path):
 h=hashlib.sha256()
 with path.open('rb') as q:
  for a in iter(lambda:q.read(2**20),b''):h.update(a)
 return h.hexdigest()

def prepare():
 old=prior.check();pub=json.loads((_repo_path(ROOT, 'reports/candidate008-public-submission.json')).read_text());OUT.mkdir(exist_ok=False);p={'hypothesis':'Mixture-conditioned means/variances of pretrained local tokens can represent motif-distribution changes missed by single globalmean/std/projectedcovariance. This is an unproven transfer toDINO hardness.','reference':{'requested':'https://openaccess.thecvf.com/content_cvpr_2015/html/Cimpoi_Deep_Filter_Banks_2015_CVPR_paper.html','access':'CVFhtml/PDF tools failed403; read author arXiv1411.6836v2PDF method3.1 and descriptor details, plusOxford authorproject.','read_source':'https://arxiv.org/pdf/1411.6836','author_project':'https://www.robots.ox.ac.uk/~vgg/research/deeptex/','scope':'Cimpoi,Maji,Vedaldi CVPR2015 FV-CNN pools pretrained convolutional local descriptors for texture recognition. Not evidence forDINO,syntheticmicrostructuresorhardness; not a reproduction of their CNN/K64/descriptor settings.'},'duplicate_audit':'NoFisher/VLAD found.048supervisedPLS4 compresses image-levelglobal3456 descriptors with labels, notlocalmotifs.056 uses unconditional projectedtoken covariance. phase_features GMM2 fits scalarintensities separately perimage for phaseproxy; no sharedpatchdictionary/Fisher.042semivariograms and066multiscaleconcatenation also distinct.','fixed_features':'Existingofficialfrozen DINO-S14 finalnormalizedpatch tokens full504:1296x384. NoCLS/multilayer/backbone/resolution alternatives.','dictionary_sampling':'Exactly64 uniformwithoutreplacementpatchpositions perimage. Positions are deterministicID-SHA256seed, sorted, same acrossfolds; everytrainingimage contributes64. All1296patches perimage usedonlyfor itsencoding.','dictionary':'Outertrain64*320=20480tokens only -> PCA32(no whitening,randomizedsvd,iterated_power4,seed20270107) -> diagonalGMM8(reg_covar1e-4,tol1e-3,max_iter200,n_init1,kmeansinit,seed20270107). Require convergence; no changes onfailure. No validation/test patches enter dictionaryfit.','Fisher':'ForN PCA tokens z, posteriorq, weightsw,meansmu,diagvariancev: U_k=sum(q*(z-mu)/sqrt(v))/(N*sqrt(w)); V_k=sum(q*((z-mu)^2/v-1))/(N*sqrt(2w)). Concatenate U thenV=512d, nozero-order term, signed-square-root thenwholevectorL2. No spatialcoordinates.','head':'OutertrainStandardScaler512 + targetmean/std + Ridge(alpha512 fixed=1*dimension), fitonceperouterfold. Noinnergrid,sweep,MLPorresidualtraining.','primary':'0.9*samefold008 +0.1*FisherRidge prediction. Auxiliary10% weight fixed before anyscores; directtargetregression, notresidualcorrection.','selection':'Original primary pooledOOF RMSE strictlylower008 -> unchangedboth existingresplits; otherwise reject. All3 pooledRMSEstrictlylower008 plus reproducibility required forpromising candidate consideration. Foldcount/CI/standalone scoresdiagnostic, not gates/fallbacks.','preflight':'FeatureCPU/MPS3images projected400<300s,RSS<3GB,MPSdriver<3GB. Fulloriginalfold0dictionary fit ontrainonly plusencoding32trainimages to project5*(dictionaryfit+400encodings)<300s andRSS<3GB; no outerlabels scored duringpreflight. Stop onnonconvergence/resourcefailure.','cache':'One development400 float32memmap400x1296x384 (~796MB) forfrozenperimagefeatures. Extraction islabel-free but fit dictionaries separately eachfold. Disk budget<1GBnewcache; nofinaltrain/testcachesdecoded.','sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['fisher_pooling_067.py','test_fisher_pooling_067.py','fisher_experiment_067.py','dino_spatial_039.py','dino_frozen_037.py','dinob_complement_064.py']},'inputs':{**old['inputs'],'reports/dinob-complement-064-validation.json':fv.sha(_repo_path(ROOT, 'reports/dinob-complement-064-validation.json'))},'protected':{**old['protected'],**pub['protected_sha256'],'reports/candidate008-public-submission.json':fv.sha(_repo_path(ROOT, 'reports/candidate008-public-submission.json'))},'scope':'Development400only; holdout/testselectionforbidden. ExistinguvNumPy/sklearn/scipy/torch only; noinstall/download/cost/upload.','bias':fv.CAVEAT};fv.write(_repo_path(OUT, 'plan.json'),p);fv.write(_repo_path(OUT, 'fixtures.json'),fixtures.run());print(json.dumps({'prepared':True,'primary':p['primary'],'dictionary':p['dictionary'],'selection':p['selection']}),flush=True)

def check():
 prior.check();p=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,h in {**p['sources'],**p['inputs'],**p['protected']}.items():assert streamsha(_repo_path(ROOT, n))==h,n
 official=json.loads((_repo_path(ROOT, 'artifacts/dino-frozen-037/smoke.json')).read_text())
 for n,h in official['official_source_sha256'].items():assert streamsha(ds.base.SOURCE/n)==h,n
 return p

def tokens(m,ident,device):
 assert ident in ds.base.scope()[0] and not m.training and not any(p.requires_grad for p in m.parameters())
 with torch.inference_mode():
  z=m.forward_features(torch.from_numpy(ds.inputs(ident,'full504')).to(device));P=z['x_norm_patchtokens'];assert P.shape==(1,1296,384);desc=torch.cat([z['x_norm_clstoken'],P.mean(1),P.std(1,correction=0)],1).cpu().numpy()[0];return P.cpu().numpy()[0],desc

def benchmark_tokens():
 check();torch.set_num_threads(2);ids,_=ds.base.scope();m=ds.base.model();cpu=tokens(m,ids[0],'cpu')[0];m=m.to('mps');t=time.perf_counter();a=[tokens(m,i,'mps')[0] for i in ids[:3]];seconds=time.perf_counter()-t;np.testing.assert_allclose(cpu,a[0],atol=2e-3,rtol=2e-3);np.testing.assert_array_equal(a[0],tokens(m,ids[0],'mps')[0]);np.savez_compressed(_repo_path(OUT, 'benchmark_tokens.npz'),ids=np.array(ids[:3]),tokens=np.stack(a));r={'seconds3':seconds,'projected400_seconds':seconds*400/3,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory(),'cpu_mps_max_error':float(abs(cpu-a[0]).max()),'MPS_repeat_exact':True};r['resource_pass']=bool(r['projected400_seconds']<300 and r['rss_peak_bytes']<3e9 and r['mps_driver_bytes']<3e9);fv.write(_repo_path(OUT, 'token-benchmark.json'),r);print(json.dumps(r),flush=True)

def extract():
 check();assert json.loads((_repo_path(OUT, 'token-benchmark.json')).read_text())['resource_pass'];torch.set_num_threads(2);ids,_=ds.base.scope();path=_repo_path(OUT, 'development_tokens.npy');assert not path.exists();A=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(400,1296,384));m=ds.base.model().to('mps');before=prior.h.state_hash(m.state_dict());t=time.perf_counter();desc=[]
 for j,ident in enumerate(ids):
  P,D=tokens(m,ident,'mps');assert np.isfinite(P).all();A[j]=P;desc.append(D)
  if (j+1)%100==0:print(json.dumps({'tokens_extracted':j+1,'seconds':time.perf_counter()-t}),flush=True)
 assert before==prior.h.state_hash(m.state_dict()) and all(p.grad is None for p in m.parameters());A.flush();del A
 with np.load(_repo_path(OUT, 'benchmark_tokens.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(np.load(_repo_io(path),mmap_mode='r')[:3],z['tokens'])
 with np.load(_repo_path(ROOT, 'artifacts/dino-multilayer-044/development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;err=float(abs(np.asarray(desc)-z['X'][:,2]).max());np.testing.assert_allclose(desc,z['X'][:,2],atol=2e-4,rtol=0)
 ix=np.stack([f.sample_indices(i) for i in ids]);np.save(_repo_path(OUT, 'sample_indices.npy'),ix);r={'ids':ids,'shape':[400,1296,384],'dtype':'float32','tokens_sha256':streamsha(path),'sample_indices_sha256':streamsha(_repo_path(OUT, 'sample_indices.npy')),'image_sha256':{i:fv.sha(_repo_path(ROOT, 'data/train', f'{i}.png')) for i in ids},'weights_unchanged':True,'meanstd044_max_error':err,'seconds':time.perf_counter()-t,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'cache_bytes':path.stat().st_size};fv.write(_repo_path(OUT, 'extraction.json'),r);print(json.dumps({k:v for k,v in r.items() if k not in ('ids','image_sha256')}),flush=True)

def inputs(split):
 ids,S,B,y,g,fid,r003,r007,Smean=prior.old.inputs(split);ev=json.loads((_repo_path(ROOT, 'reports/dinob-complement-064-validation.json')).read_text());src=prior.OUT/split/'development_predictions.npz';assert fv.sha(src)==ev['results'][split]['predictions_sha256']
 with np.load(_repo_io(src),allow_pickle=False) as z:assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['y'],y);np.testing.assert_array_equal(z['fold'],fid);ref=z['mixed50'].copy()
 ex=json.loads((_repo_path(OUT, 'extraction.json')).read_text());assert ex['ids']==ids and streamsha(_repo_path(OUT, 'development_tokens.npy'))==ex['tokens_sha256'] and streamsha(_repo_path(OUT, 'sample_indices.npy'))==ex['sample_indices_sha256'];T=np.load(_repo_path(OUT, 'development_tokens.npy'),mmap_mode='r');ix=np.load(_repo_path(OUT, 'sample_indices.npy'));np.testing.assert_array_equal(ix,np.stack([f.sample_indices(i) for i in ids]));return ids,y,g,fid,ref,T,ix

def sample(T,ix,tr):return np.concatenate([np.asarray(T[j][ix[j]],dtype=float) for j in tr],axis=0)
def encode(dictionary,T,indices):return np.stack([dictionary.encode(T[j]) for j in indices])

def benchmark_dictionary():
 check();torch.set_num_threads(2);ids,y,g,fid,ref,T,ix=inputs('original');tr=np.flatnonzero(fid!=0);sampled=sample(T,ix,tr);start=time.perf_counter();dic=f.FisherDictionary().fit(sampled,[ids[j] for j in tr]);fitseconds=time.perf_counter()-start;t=time.perf_counter();A=encode(dic,T,tr[:32]);seconds=time.perf_counter()-t;assert np.isfinite(A).all();r={'dictionary_train_images':320,'patches_per_image':64,'fit_patch_count':len(sampled),'dictionary_seconds':fitseconds,'encode32_seconds':seconds,'projected5fold_seconds':5*(fitseconds+seconds*400/32),'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'gmm_converged':bool(dic.gmm.converged_),'gmm_iterations':int(dic.gmm.n_iter_),'pca_retained_variance':float(dic.pca.explained_variance_ratio_.sum()),'no_outerlabels_scored':True};r['resource_pass']=bool(r['projected5fold_seconds']<300 and r['rss_peak_bytes']<3e9);joblib.dump(dic,_repo_path(OUT, 'benchmark_dictionary.joblib'));fv.write(_repo_path(OUT, 'dictionary-benchmark.json'),r);print(json.dumps(r),flush=True)

def run(split):
 check();assert json.loads((_repo_path(OUT, 'dictionary-benchmark.json')).read_text())['resource_pass']
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 torch.set_num_threads(2);ids,y,g,fid,ref,T,ix=inputs(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);pred=np.full(400,np.nan);records=[];t=time.perf_counter()
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(g[tr])&set(g[va]);start=time.perf_counter();samples=sample(T,ix,tr);dic=f.FisherDictionary().fit(samples,[ids[j] for j in tr]);A=encode(dic,T,np.arange(400));assert A.shape==(400,512) and np.isfinite(A).all();head=f.FisherHead().fit(A[tr],y[tr]);pred[va]=head.predict(A[va]);fd=out/f'fold-{k}';fd.mkdir();path=fd/'model.joblib';joblib.dump({'dictionary':dic,'head':head,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va],'train_patch_indices':ix[tr],'config':{'PCA':32,'GMM':8,'alpha':512.,'mix':.1}},_repo_io(path))
  with (fd/'features.npz').open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(ids),X=A)
  rec={'fold':k,'model_sha256':fv.sha(path),'features_sha256':fv.sha(fd/'features.npz'),'sample_sha256':dic.sample_sha256,'dictionary_train_ids':dic.train_ids,'patch_counts_per_train_image':64,'fit_patch_count':len(samples),'PCA_mean_sha256':hashlib.sha256(dic.pca.mean_.tobytes()).hexdigest(),'PCA_retained_variance':float(dic.pca.explained_variance_ratio_.sum()),'GMM_converged':bool(dic.gmm.converged_),'GMM_iterations':int(dic.gmm.n_iter_),'GMM_weights':dic.gmm.weights_.tolist(),'GMM_min_variance':float(dic.gmm.covariances_.min()),'train_rmse':fv.rmse(y[tr],head.predict(A[tr])),'validation_rmse':fv.rmse(y[va],pred[va]),'seconds':time.perf_counter()-start};fv.write(fd/'report.json',rec);records.append(rec);print(json.dumps({'split':split,'fold':k,'gmm_iterations':rec['GMM_iterations'],'seconds':rec['seconds']}),flush=True)
 new=.9*ref+.1*pred;ps={'reference008':ref,'fisher512':pred,'primary':new};scores={n:{'rmse':fv.rmse(y,v),'fold_rmse':[fv.rmse(y[fid==k],v[fid==k]) for k in range(5)]} for n,v in ps.items()};comp=prior.old.comparison(y,ref,new,fid)
 with (out/'development_predictions.npz').open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(ids),y=y,fold=fid,**ps)
 r={'split':split,'scores':scores,'comparison':comp,'fisher008_residual_correlation':float(np.corrcoef(y-ref,y-pred)[0,1]),'folds':records,'seconds':time.perf_counter()-t,'predictions_sha256':fv.sha(out/'development_predictions.npz'),'new_candidate':None};fv.write(out/'report.json',r)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':bool(comp['gain']>0),'primary':'primary','comparison':comp})
 check();print(json.dumps({k:v for k,v in r.items() if k!='folds'}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','benchmark_tokens','extract','benchmark_dictionary','original','20261031','20261101']);a=ap.parse_args();{'prepare':prepare,'benchmark_tokens':benchmark_tokens,'extract':extract,'benchmark_dictionary':benchmark_dictionary}.get(a.mode,lambda:run(a.mode))()
