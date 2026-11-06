"""Image-local conservative TV denoising before unchanged frozen ViT-B."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,time,resource,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import numpy as np,torch
import dinob_062 as b
import dinob_complement_064 as prior
fv=prior.fv;h=prior.h;SEEDS=prior.SEEDS;OUT=_repo_path(ROOT, 'artifacts/dinob-denoised-071')

def check():
 prior.check();b.check();p=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for name,digest in {**p['sources'],**p['inputs'],**p['protected']}.items():assert fv.sha(_repo_path(ROOT, name))==digest,name
 return p

def image_input(ident):
 from PIL import Image
 from denoised_grain_features import _noise_sigma
 from skimage.restoration import denoise_tv_chambolle
 with Image.open(_repo_path(ROOT, 'data/train', f'{ident}.png')) as im:
  assert im.mode=='L' and im.size==(256,256);raw=np.asarray(im,dtype=np.float64)/255
 sigma=_noise_sigma(raw)
 den=denoise_tv_chambolle(raw,weight=.5*sigma,eps=1e-4,max_num_iter=50,channel_axis=None) if sigma>1e-12 else raw.copy()
 im=Image.fromarray(np.rint(np.clip(den,0,1)*255).astype(np.uint8)).resize((504,504),Image.Resampling.BICUBIC)
 return np.ascontiguousarray((np.repeat((np.asarray(im,dtype=np.float32)/255)[None],3,axis=0)-b.ds.base.MEAN)/b.ds.base.STD)[None]

def feature(m,x):
 assert not m.training and not any(p.requires_grad for p in m.parameters())
 with torch.inference_mode():
  layers=m.get_intermediate_layers(x,n=[5,8,11],return_class_token=True,norm=True)
  return torch.cat([torch.cat((c,p.mean(1),p.std(1,correction=0)),1) for p,c in layers],1)

def prepare():
 prior.check();b.check();pub=json.loads((_repo_path(ROOT, 'reports/candidate008-public-submission.json')).read_text());old=prior.check();assert not json.loads((_repo_path(ROOT, 'artifacts/dinos-coarse-070/selection.json')).read_text())['advance'];OUT.mkdir(exist_ok=False)
 p={'hypothesis':'068noiseproxyweaklycorrelateswithabsoluteOOFerror(~0.10),while069rawtokenamplitudesdidnothelp. Noise-awareedge-preservingTV preprocessing mayreduce acquisitiontexturewhichDINOencodeswhilepreservingcoarsemorphology.',
 'evidence_limits':'Noiseproxycancontainrealboundarytexture;weakdescriptiveassociationisnotcausal. Earlier022/023TVwatershedfeatures improveddevelopment but004Publicregressed;useonlyasfeasibilityevidence,notsuccessclaim. NoDINOdenoisingexperimentfound.',
 'single_primary':'SameB504normTrue multilayer6912 descriptor andsameMLP as008. Onlychange grayscale input: native256 image-localHH-MAD sigma;TVweight0.5*sigma,eps1e-4,max50iters;rounduint8 thenoriginalPILbicubic504pipeline. NoD4, no extra normalization, no featureconcat.',
 'primary_formula':'.75*r003+.125*Smean3+.125*Bdenoisedmean3',
 'fixed_strength_reason':'0.5 is existing022conservativeTVfactor,chosenbeforethisexperiment,notselectedfromdenoisedDINOresults. No strengthgridor fallback.',
 'seeds':list(SEEDS),'config':h.CONFIG,'effective_aux_weight':0.,'preprocessing':'Image-localdeterministicdenoising,no learneddatafit. StandardScaler+targetscale outertrainonly.',
 'screen':'OriginalpooledOOF strictlyimproves008 -> bothknownresplits; otherwise reject.',
 'confirmation':'All3pooledOOF improve008 plusstrictverification. Fold/bootstrapdiagnosticsonly.',
 'resource':'ExistingCPU2/MPS;3featurebenchmark<300sprojected400,<3GB;head<300sprojected15fits,<2GB. No newinstall/download/cost.',
 'sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['dinob_denoised_071.py','validate_dinob_denoised_071.py','denoised_grain_features.py','dinob_mlp_head_063.py','dinob_mlp_063.py','dinob_complement_064.py','dinob_062.py']},
 'inputs':{**old['inputs'],**{n:fv.sha(_repo_path(ROOT, n)) for n in ['reports/dinob-complement-064-validation.json','artifacts/error-diagnostics-068/report.json','artifacts/dinos-coarse-070/selection.json']}},
 'protected':{**old['protected'],**pub['protected_sha256'],'reports/candidate008-public-submission.json':fv.sha(_repo_path(ROOT, 'reports/candidate008-public-submission.json'))},
 'scope':'Development400only,noholdout/testreadforsel;protectedfinalcacheshashedonly.','bias':fv.CAVEAT}
 fv.write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'hypothesis':p['hypothesis']}),flush=True)

def benchmark_features():
 check();torch.set_num_threads(2);ids,_=b.ds.base.scope();m=b.model();x=torch.from_numpy(image_input(ids[0]));cpu=feature(m,x).numpy()[0];m=m.to('mps');values=[];t=time.perf_counter()
 for ident in ids[:3]:values.append(feature(m,torch.from_numpy(image_input(ident)).to('mps')).cpu().numpy()[0])
 seconds=time.perf_counter()-t;X=np.stack(values);np.testing.assert_allclose(cpu,X[0],atol=2e-3,rtol=2e-3);np.testing.assert_array_equal(X[0],feature(m,x.to('mps')).cpu().numpy()[0]);assert all(p.grad is None for p in m.parameters())
 with (_repo_path(OUT, 'benchmark_features.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids[:3]),X=X)
 report={'three_images_seconds':seconds,'projected400_seconds':seconds*400/3,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory(),'cpu_mps_max_error':float(abs(cpu-X[0]).max()),'MPS_repeat_error':0.,'shape':list(X.shape)};report['resource_pass']=bool(report['projected400_seconds']<300 and report['rss_peak_bytes']<3e9 and report['mps_driver_bytes']<3e9);fv.write(_repo_path(OUT, 'feature-benchmark.json'),report);print(json.dumps(report),flush=True)

def extract():
 check();assert json.loads((_repo_path(OUT, 'feature-benchmark.json')).read_text())['resource_pass'];torch.set_num_threads(2);ids,_=b.ds.base.scope();m=b.model().to('mps');before=h.state_hash(m.state_dict());values=[];t=time.perf_counter()
 for j,ident in enumerate(ids):
  values.append(feature(m,torch.from_numpy(image_input(ident)).to('mps')).cpu().numpy()[0])
  if (j+1)%100==0:print(json.dumps({'features':j+1,'seconds':time.perf_counter()-t}),flush=True)
 X=np.stack(values);assert X.shape==(400,6912) and np.isfinite(X).all();assert h.state_hash(m.state_dict())==before and all(p.grad is None for p in m.parameters())
 with np.load(_repo_path(OUT, 'benchmark_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids[:3];np.testing.assert_array_equal(X[:3],z['X'])
 with (_repo_path(OUT, 'development_Bdenoised.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),X=X)
 with np.load(_repo_path(ROOT, 'artifacts/dinob-062/development_features.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;B=z['X'].astype(float)
 a=X.astype(float)-X.astype(float).mean(0);bb=B-B.mean(0);den=np.sqrt((a*a).sum(0)*(bb*bb).sum(0));valid=den>0;corr=np.sum(a[:,valid]*bb[:,valid],axis=0)/den[valid]
 report={'shape':list(X.shape),'seconds':time.perf_counter()-t,'feature_sha256':fv.sha(_repo_path(OUT, 'development_Bdenoised.npz')),'image_sha256':{i:fv.sha(_repo_path(ROOT, 'data/train', f'{i}.png')) for i in ids},'weights_unchanged':True,'benchmark_repeat_exact':True,'resolution':504,'same_channel_prenorm_normalized_correlation_quantiles':np.percentile(corr,[0,10,50,90,100]).tolist(),'valid_correlation_columns':int(valid.sum()),'diagnostic_scope':'All400 featurevalues only, no target-conditioned featureselection. Correlation notproof of incremental targetsignal.','rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory()};check();fv.write(_repo_path(OUT, 'extraction.json'),report);print(json.dumps({k:v for k,v in report.items() if k!='image_sha256'}),flush=True)

def inputs(split):
 ids,S,B,y,g,fid,r003,r007,Smean=prior.old.inputs(split);ex=json.loads((_repo_path(OUT, 'extraction.json')).read_text());assert fv.sha(_repo_path(OUT, 'development_Bdenoised.npz'))==ex['feature_sha256']
 with np.load(_repo_path(OUT, 'development_Bdenoised.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;X=z['X'].astype(float)
 v=json.loads((_repo_path(ROOT, 'reports/dinob-complement-064-validation.json')).read_text());path=prior.OUT/split/'development_predictions.npz';assert fv.sha(path)==v['results'][split]['predictions_sha256']
 with np.load(_repo_io(path),allow_pickle=False) as z:
  assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['y'],y);np.testing.assert_array_equal(z['fold'],fid);np.testing.assert_array_equal(z['reference003'],r003);np.testing.assert_array_equal(z['reference007'],r007);np.testing.assert_array_equal(z['Smean'],Smean);ref=z['mixed50'].copy();Bmean=z['Bmean'].copy()
 np.testing.assert_allclose(ref,.75*r003+.125*Smean+.125*Bmean,atol=1e-12,rtol=0);assert X.shape==(400,6912) and np.isfinite(X).all();return ids,X,y,g,fid,r003,r007,Smean,Bmean,ref

def benchmark_head():
 check();torch.set_num_threads(2);ids,X,y,g,fid,*_=inputs('original');tr=np.flatnonzero(fid!=0);sc,ym,ys,xt,yt,bt,edges=prior.old.prep(X[tr],y[tr]);t=time.perf_counter();m,rec=h.fit(xt,yt,bt,SEEDS[0],0.);seconds=time.perf_counter()-t;old=json.loads((_repo_path(ROOT, 'artifacts/dinob-mlp-063/original/fold-0/report.json')).read_text())['models'][f'baseB-{SEEDS[0]}'];assert rec['batch_order_sha256']==old['batch_order_sha256'];report={'seconds':seconds,'projected15fits_seconds':seconds*15,'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'parameters':sum(v.numel() for v in m.parameters()),'order_matched_B504':True};report['resource_pass']=bool(report['projected15fits_seconds']<300 and report['rss_peak_bytes']<2e9);fv.write(_repo_path(OUT, 'head-benchmark.json'),report);print(json.dumps(report),flush=True)

def run(split):
 check();assert json.loads((_repo_path(OUT, 'head-benchmark.json')).read_text())['resource_pass']
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 torch.set_num_threads(2);ids,X,y,g,fid,r003,r007,Smean,Bmean,ref=inputs(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);pred=np.full((3,400),np.nan);trainerrors=[];t=time.perf_counter()
 for k in range(5):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(g[tr])&set(g[va]);sc,ym,ys,xt,yt,bt,edges=prior.old.prep(X[tr],y[tr]);fd=out/f'fold-{k}';fd.mkdir();rec={'fold':k,'models':{}};tp=[];oldlog=json.loads((_repo_path(ROOT, 'artifacts/seed-stability-052', split, f'fold-{k}/report.json')).read_text())
  for si,seed in enumerate(SEEDS):
   m,log=h.fit(xt,yt,bt,seed+k,0.);assert log['batch_order_sha256']==oldlog['models'][str(seed)]['batch_order_sha256'];path=fd/f'seed-{seed}.pt';torch.save({'model':m.state_dict(),'input_dim':6912,'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'scaler_var':torch.tensor(sc.var_),'target_mean':ym,'target_std':ys,'bin_edges_unused':torch.tensor(edges),'seed':seed+k,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va],'config':h.CONFIG,'effective_aux_weight':0.},_repo_io(path));pred[si,va]=h.predict(m,sc.transform(X[va]),ym,ys);tp.append(h.predict(m,xt,ym,ys));log.update(checkpoint_sha256=fv.sha(path),validation_rmse=fv.rmse(y[va],pred[si,va]));rec['models'][str(seed)]=log
  trainerrors.extend((np.mean(tp,axis=0)-y[tr]).tolist());fv.write(fd/'report.json',rec);print(json.dumps({'split':split,'fold':k,'seconds':time.perf_counter()-t}),flush=True)
 mean=pred.mean(0);new=.75*r003+.125*Smean+.125*mean;ps={'reference008':ref,'reference003':r003,'Smean':Smean,'B504mean':Bmean,'Bdenoised':mean,'primary':new};scores={n:{'rmse':fv.rmse(y,v),'fold_rmse':[fv.rmse(y[fid==k],v[fid==k]) for k in range(5)]} for n,v in ps.items()};comp=prior.old.comparison(y,ref,new,fid)
 with (out/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),y=y,fold=fid,seed_predictions=pred,**ps)
 report={'split':split,'scores':scores,'comparison':comp,'standalone_Bdenoised_B504_residual_correlation':float(np.corrcoef(y-mean,y-Bmean)[0,1]),'mean3_train_rmse_pooled':float(np.sqrt(np.mean(np.array(trainerrors)**2))),'seconds':time.perf_counter()-t,'predictions_sha256':fv.sha(out/'development_predictions.npz'),'new_candidate':None};fv.write(out/'report.json',report)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':bool(comp['gain']>0),'primary':'primary','comparison':comp,'diagnostic_only':'foldcounts and bootstrap; no hard threshold'})
 check();print(json.dumps(report),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','benchmark_features','extract','benchmark_head','original','20261031','20261101']);a=ap.parse_args();{'prepare':prepare,'benchmark_features':benchmark_features,'extract':extract,'benchmark_head':benchmark_head}.get(a.mode,lambda:run(a.mode))()
