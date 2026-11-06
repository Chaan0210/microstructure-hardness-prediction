"""Development-only observable structure/error associations and preprocessing audit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import spearmanr
import dinob_complement_064 as prior
fv=prior.fv;OUT=_repo_path(ROOT, 'artifacts/error-diagnostics-068')

def metrics(raw):
 x=raw.astype(float)/255.;sm=ndi.gaussian_filter(x,1.);gy,gx=np.gradient(sm);a=(gx*gx).mean();b=(gy*gy).mean();c=(gx*gy).mean();den=a+b+1e-12;z=x-x.mean();var=x.var()+1e-12;lap=ndi.convolve(x,np.array([[1,-2,1],[-2,4,-2],[1,-2,1]]),mode='reflect');sigma=np.median(abs(lap))/4.0469385;tiles=x.reshape(8,32,8,32).transpose(0,2,1,3).reshape(64,32,32);f={'image_mean':x.mean(),'image_std':x.std(),'q05':np.quantile(x,.05),'q95':np.quantile(x,.95),'dark_absolute_fraction':(x<.25).mean(),'clipped_fraction':((raw==0)|(raw==255)).mean(),'noise_MAD_proxy':sigma,'noise_contrast_ratio':sigma/(x.std()+1e-12),'highpass1_std':(x-sm).std(),'gradient1_mean':np.hypot(gx,gy).mean(),'anisotropy1':np.sqrt((a-b)**2+4*c*c)/den,'orientation_cos2_weighted':(a-b)/den,'orientation_sin2_weighted':2*c/den,'tile_mean_std':tiles.mean((1,2)).std(),'tile_contrast_std':tiles.std((1,2)).std(),'intensity_skew':np.mean(z**3)/var**1.5,'intensity_kurtosis':np.mean(z**4)/var**2}
 for lag in (1,4,16):f[f'autocorr{lag}']=(np.mean(z[:-lag]*z[lag:])+np.mean(z[:,:-lag]*z[:,lag:]))/(2*var)
 return {k:float(v) for k,v in f.items()}

def corr(a,b):
 return None if np.ptp(a)==0 or np.ptp(b)==0 else float(spearmanr(a,b).statistic)

def main():
 start=time.perf_counter();prior.check();OUT.mkdir(exist_ok=False);ids,_,_,y,g,fid,r003,r007,oldmean=prior.old.inputs('original');ev=json.loads((_repo_path(ROOT, 'reports/dinob-complement-064-validation.json')).read_text());pred={};folds={}
 for split in ('original','20261031','20261101'):
  path=prior.OUT/split/'development_predictions.npz';assert fv.sha(path)==ev['results'][split]['predictions_sha256']
  with np.load(_repo_io(path),allow_pickle=False) as z:assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['y'],y);pred[split]=z['mixed50'].copy();folds[split]=z['fold'].copy()
 meta=[];hashes={};affine=[];resize128=[];resize504=[]
 for ident in ids:
  path=_repo_path(ROOT, 'data/train', f'{ident}.png')
  with Image.open(_repo_io(path)) as im:
   raw=np.asarray(im);meta.append(metrics(raw));x=raw.astype(np.float32)/255.;rgb=np.repeat(x[None],3,axis=0);mean=np.array([.485,.456,.406],np.float32)[:,None,None];std=np.array([.229,.224,.225],np.float32)[:,None,None];affine.append(float(abs(((rgb-mean)/std)*std+mean-rgb).max()))
   small=im.resize((128,128),Image.Resampling.BILINEAR).resize((256,256),Image.Resampling.BILINEAR)
   large=im.resize((504,504),Image.Resampling.BICUBIC).resize((256,256),Image.Resampling.BICUBIC)
   resize128.append(float(np.sqrt(np.mean((np.asarray(small,dtype=float)/255.-x)**2))))
   resize504.append(float(np.sqrt(np.mean((np.asarray(large,dtype=float)/255.-x)**2))))
  hashes[ident]=fv.sha(path)
 names=list(meta[0]);X=np.array([[r[n] for n in names] for r in meta]);assert np.isfinite(X).all();Y=np.stack([y-p for p in pred.values()]);signed=Y.mean(0);absolute=abs(Y).mean(0);records=[]
 for j,name in enumerate(names):
  corrs=[corr(X[:,j],err) for err in Y];abs_corrs=[corr(X[:,j],abs(err)) for err in Y];bins={}
  for split,prediction in pred.items():
   groups=np.empty(400,int);fi=folds[split]
   for k in range(5):groups[fi==k]=np.searchsorted(np.quantile(X[fi!=k,j],[.25,.75]),X[fi==k,j])
   bins[split]=[{'n':int((groups==k).sum()),'rmse':fv.rmse(y[groups==k],prediction[groups==k]) if (groups==k).any() else None,'mean_target_minus_prediction':float(np.mean(y[groups==k]-prediction[groups==k])) if (groups==k).any() else None} for k in range(3)]
  records.append({'feature':name,'signed_error_spearman_by_split':corrs,'absolute_error_spearman_by_split':abs_corrs,'mean_signed_error_spearman':corr(X[:,j],signed),'mean_absolute_error_spearman':corr(X[:,j],absolute),'target_spearman':corr(X[:,j],y),'foldtrain_quartile_bins':bins})
 with (_repo_path(OUT, 'development_diagnostics.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),names=np.array(names),X=X,y=y,signed_errors=Y,**{f'pred_{k}':v for k,v in pred.items()})
 with (_repo_path(OUT, 'per_image_errors.csv')).open('x',newline='') as f:
  w=csv.writer(f);w.writerow(['ID','target','mean_signed_error','mean_abs_error',*names]);w.writerows([ident,y[j],signed[j],absolute[j],*X[j]] for j,ident in enumerate(ids))
 audit={'DINO_input_affine':'FixedImageNetmean/std,notperimagenormalization; grayscaleRGBreplication addsnoinfo; affineinvertible, maxroundtriperror recorded. NoDINOtrainingaugmentation in008.','DINO_token_LayerNorm':'Existingget_intermediate_layers(norm=True) normalizes eachtoken acrosschannels. Removes pertokenpre-LN channelmean/scale; this differs from inputaffine. Poolmean/std then discards explicitpatcharrangement. Pre-LN descriptors not yet compared.','CNN':'Foldtrainingglobalpixelmean/std isaffine, preservesinterimage brightness/contrast;128downsampling suppresseshighfrequency. BatchD4augmentation encouragesabsoluteorientation invariance but doesnotremoveimageintensity. Only.1875 of008 comesfromthisCNN.','morphology':'Some thresholdfeatures useperimage2/98percentileclip, discardingglobaloffset/contrastandextremes inthosebranches; fullbaseline explicitlyretainsintensity/histogramfeatures andcompact26 retainsimage_mean/std. BalancedKernel also clips features tofoldtrainminmax, potentiallylosingextrapolation.','no_false_claim':'008doesnotstandardizeeveryimageindividually and doesnotapplyphotometricjitter. It isincorrect toclaimbrightness is globallyerased bytheinputpipeline.','affine_max_roundtrip_error':max(affine),'128_roundtrip_RMSE_quantiles':np.percentile(resize128,[0,50,90,100]).tolist(),'504_roundtrip_RMSE_quantiles':np.percentile(resize504,[0,50,90,100]).tolist(),'resize_measurement':'Diagnostic PILdown/up roundtrip estimatesinformationloss; notthecompletelearnedpipeline or its predictioneffect.'}
 report={'preprocessing_audit':audit,'feature_names':names,'associations':records,'top_signed':sorted(records,key=lambda r:abs(r['mean_signed_error_spearman'] or 0),reverse=True)[:8],'top_absolute':sorted(records,key=lambda r:abs(r['mean_absolute_error_spearman'] or 0),reverse=True)[:8],'image_sha256':hashes,'diagnostics_sha256':fv.sha(_repo_path(OUT, 'development_diagnostics.npz')),'seconds':time.perf_counter()-start,'limits':'Descriptive associations on historicallyreused400 andsame3partitions, notcausalorindependentvalidation. Multiplemetricsinspected; nopvalue/significanceclaim. Structures areimageproxies,notgeneratortruth. Noholdout/test data.','source_sha256':fv.sha(Path(__file__))};fv.write(_repo_path(OUT, 'report.json'),report);print(json.dumps({'audit':audit,'top_signed':[{k:r[k] for k in ('feature','signed_error_spearman_by_split','mean_signed_error_spearman','target_spearman')} for r in report['top_signed']],'top_absolute':[{k:r[k] for k in ('feature','absolute_error_spearman_by_split','mean_absolute_error_spearman')} for r in report['top_absolute']],'seconds':report['seconds']}),flush=True)
if __name__=='__main__':main()
