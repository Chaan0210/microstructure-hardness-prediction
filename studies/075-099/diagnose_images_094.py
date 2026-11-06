"""094: image-only matching locked before descriptive outcome comparison; no learner."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,hashlib,time,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage as ndi
from scipy.stats import rankdata,spearmanr
from scipy.optimize import linear_sum_assignment
from diagnose_errors_068 import metrics as prior_metrics
OUT=_repo_path(ROOT, 'artifacts/image-diagnostics-094')
TOP=['TRAIN_000438','TRAIN_000331','TRAIN_000227','TRAIN_000071','TRAIN_000168']
GROUPS={
 'photometry':['image_mean','image_std','intensity_skew','intensity_kurtosis','dark_absolute_fraction','clipped_fraction'],
 'noise':['noise_contrast_ratio','highpass1_std'],
 'texture':['anisotropy1','autocorr4','autocorr16','spectral_low_fraction','spectral_mid_fraction'],
 'spatial':['tile_mean_std','tile_contrast_std','border_center_absdiff','quadrant_mean_range'],
 'dark_regions':['dark_component_density','dark_largest_fraction','dark_area_cv','dark_distance_p95','dark_boundary_density']}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)
def ids400():
 ids=sorted(json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text())['development_folds']);assert len(ids)==400 and set(TOP)<=set(ids);return ids

def additional(a):
 x=a.astype(float)/255.;sm=ndi.gaussian_filter(x,1.);z=(sm-sm.mean())/(sm.std()+1e-12);dark=z < -1
 lab,n=ndi.label(dark,np.ones((3,3)));sizes=np.bincount(lab.ravel())[1:];keep=sizes>=4;areas=sizes[keep];clean=np.isin(lab,np.flatnonzero(keep)+1);dist=ndi.distance_transform_edt(clean)
 border=np.ones(x.shape,bool);border[16:-16,16:-16]=False
 zz=sm-sm.mean();win=np.outer(np.hanning(256),np.hanning(256));power=abs(np.fft.fft2(zz*win))**2;f=np.fft.fftfreq(256);radius=np.hypot(f[:,None],f[None,:]);total=power[radius>0].sum()+1e-12
 return {'dark_component_density':float(keep.sum()/65536.),'dark_largest_fraction':float(areas.max()/65536.) if len(areas) else 0.,'dark_area_cv':float(areas.std()/(areas.mean()+1e-12)) if len(areas) else 0.,'dark_distance_p95':float(np.quantile(dist[clean],.95)) if clean.any() else 0.,'dark_boundary_density':float((clean^ndi.binary_erosion(clean)).mean()),'dark_relative_fraction':float(clean.mean()),'border_center_absdiff':float(abs(x[border].mean()-x[~border].mean())),'quadrant_mean_range':float(np.ptp(x.reshape(2,128,2,128).mean((1,3)))),'spectral_low_fraction':float(power[(radius>0)&(radius<1/32)].sum()/total),'spectral_mid_fraction':float(power[(radius>=1/32)&(radius<1/8)].sum()/total)}

def extract():
 start=time.perf_counter();OUT.mkdir(exist_ok=False);ids=ids400();plan={'study':'094 image-only observable matching followed by descriptive outcome analysis','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fixed_top5':TOP,'selection_bias':'Anchors were selected by original092 outcome; this entire investigation is posthoc. Controls and metrics use images only. Same400 and repeatedCV are not independent evidence.','metrics':'Reuse all20 diagnostics068, plus10 fixed image-local metrics. Relative dark proxy: Gaussian sigma1, below mean minus1SD,8-connected,remove components<4px. Not true pore/grain labels. Spectral bands fixed at wavelengths32px and8px;border16px;quadrants128px.','matching':{'groups':GROUPS,'transform':'Per-metric midranks across development400,scaled to[0,1]. Descriptive matching only, not predictive normalization.','distance':'Equal mean squared percentile distance pergroup; equal5group mean then sqrt.','controls_per_anchor':3,'selection':'Global minimum sum squared distance assignment with3slots peranchor,all395nonanchors;15uniquecontrols. No y,prediction,error used.','tie_break':'Stable sorted ID order; scipy linear_sum_assignment deterministic.'},'analysis':'Freeze control IDs/features and hashes before reading outcomes. Report all400 metric distributions,anchor percentiles, matched outcome signs/RMSE, full400/fixed395 metric-gain Spearman and fixedquartile summaries ineachpartition. No p-values,learnedgate,newlabel-defined subgroup or threshold search.','scope':'Only allowlisted400trainimages; noholdout/test, no training or submission. CPU2, existinglibraries,cost0.','protected011_sha256':sha(_repo_path(ROOT, 'artifacts/submission-candidate-011.csv')),'source_sha256':sha(Path(__file__)),'prior_metrics_source_sha256':sha(_repo_path(ROOT, 'diagnose_errors_068.py'))};write(_repo_path(OUT, 'plan.json'),plan)
 rows=[];hashes={};metadata=[]
 for ident in ids:
  p=_repo_path(ROOT, 'data/train', f'{ident}.png')
  with Image.open(_repo_io(p)) as im:
   metadata.append({'ID':ident,'mode':im.mode,'size':list(im.size)});a=np.asarray(im).copy();assert a.shape==(256,256) and a.dtype==np.uint8
  row=prior_metrics(a);row.update(additional(a));rows.append(row);hashes[ident]=sha(p)
 names=list(rows[0]);X=np.array([[r[n] for n in names] for r in rows]);assert np.isfinite(X).all();R=np.column_stack([(rankdata(X[:,j],method='average')-.5)/400 for j in range(len(names))]);D2=np.zeros((400,400))
 for ns in GROUPS.values():ix=[names.index(n) for n in ns];D2+=np.mean((R[:,None,ix]-R[None,:,ix])**2,axis=2)/len(GROUPS)
 topix=[ids.index(i) for i in TOP];rest=np.array([i for i in range(400) if i not in topix]);slots=np.repeat(topix,3);rr,cc=linear_sum_assignment(D2[slots][:,rest]);assert np.array_equal(rr,np.arange(15));controls=rest[cc];pairs=[]
 for n,i in enumerate(topix):
  js=controls[n*3:n*3+3];ds=sorted([(float(np.sqrt(D2[i,j])),ids[j]) for j in js]);pairs.append({'anchor':ids[i],'controls':[{'ID':s,'distance':d,'rank_among395':int(1+np.sum(D2[i,rest]<d*d-1e-14))} for d,s in ds]})
 with (_repo_path(OUT, 'image_metrics.csv')).open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID',*names]);w.writerows([ident,*X[i]] for i,ident in enumerate(ids))
 np.savez_compressed(_repo_path(OUT, 'image_metrics.npz'),ids=np.array(ids),names=np.array(names),X=X,rank=R,distance=np.sqrt(D2))
 with (_repo_path(OUT, 'metadata.json')).open('x') as f:json.dump(metadata,f)
 allpairs=np.sqrt(D2[np.triu_indices(400,1)]);anchor_dist=np.sqrt(D2[np.ix_(topix,topix)])[np.triu_indices(5,1)];rng=np.random.default_rng(20270324);random_disp=[]
 for _ in range(5000):ix=rng.choice(400,5,replace=False);random_disp.append(np.sqrt(D2[np.ix_(ix,ix)])[np.triu_indices(5,1)].mean())
 info={'pairs':pairs,'controls_are_unique':len(set(controls.tolist()))==15,'outcomes_read_for_selection':False,'metrics_sha256':sha(_repo_path(OUT, 'image_metrics.npz')),'csv_sha256':sha(_repo_path(OUT, 'image_metrics.csv')),'plan_sha256':sha(_repo_path(OUT, 'plan.json')),'image_sha256':hashes,'anchor_mean_pairdistance':float(anchor_dist.mean()),'random5_mean_pairdistance_quantiles':np.quantile(random_disp,[0,.05,.25,.5,.75,.95,1]).tolist(),'anchor_dispersion_percentile_descriptive_only':float(np.mean(np.array(random_disp)<=anchor_dist.mean())*100),'allpair_distance_quantiles':np.quantile(allpairs,[0,.05,.25,.5,.75,.95,1]).tolist(),'metric_summaries':[{'metric':name,'quantiles':np.quantile(X[:,j],[0,.05,.25,.5,.75,.95,1]).tolist(),'anchor_percentiles':{ident:float(R[ids.index(ident),j]*100) for ident in TOP}} for j,name in enumerate(names)],'seconds':time.perf_counter()-start}
 write(_repo_path(OUT, 'matching-locked.json'),info)
 # Unmodified native256px grayscale, identicaldisplay scale, no per-image contrast adjustment.
 canvas=Image.new('RGB',(4*276,5*292),'white');draw=ImageDraw.Draw(canvas)
 for row,pair in enumerate(pairs):
  for col,ident in enumerate([pair['anchor']]+[c['ID'] for c in pair['controls']]):
   with Image.open(_repo_path(ROOT, 'data/train', f'{ident}.png')) as im:canvas.paste(im.convert('RGB'),(col*276+10,row*292+28))
   draw.text((col*276+10,row*292+7),('ANCHOR ' if col==0 else 'CONTROL ')+ident,fill='black')
 canvas.save(_repo_path(OUT, 'matched-images.png'));print(json.dumps({k:v for k,v in info.items() if k not in ('image_sha256','metric_summaries')},ensure_ascii=False),flush=True)

def group_stats(y,a,b,ix):
 ea=y[ix]-a[ix];eb=y[ix]-b[ix];return {'n':len(ix),'RMSE011':float(np.sqrt(np.mean(ea*ea))),'RMSEattentionblend':float(np.sqrt(np.mean(eb*eb))),'RMSE_gain':float(np.sqrt(np.mean(ea*ea))-np.sqrt(np.mean(eb*eb))),'mean_squared_error_gain':float(np.mean(ea*ea-eb*eb)),'improved':int(np.sum(ea*ea>eb*eb)),'mean_prediction_change':float(np.mean(b[ix]-a[ix])),'mean_target_minus011':float(ea.mean())}

def analyze():
 lock=json.loads((_repo_path(OUT, 'matching-locked.json')).read_text());plan=json.loads((_repo_path(OUT, 'plan.json')).read_text());assert sha(_repo_path(OUT, 'plan.json'))==lock['plan_sha256'] and sha(_repo_path(OUT, 'image_metrics.npz'))==lock['metrics_sha256'];assert sha(Path(__file__))==plan['source_sha256'];ids=ids400()
 with np.load(_repo_path(OUT, 'image_metrics.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;X=z['X'];R=z['rank'];names=z['names'].tolist();D=z['distance']
 ti=np.array([ids.index(i) for i in TOP]);ci=np.array([ids.index(c['ID']) for p in lock['pairs'] for c in p['controls']]);rest=np.array([i for i in range(400) if i not in ti]);results={};outputs=[]
 for split in ('original','20261031','20261101'):
  path=_repo_path(ROOT, 'artifacts/patch-attention-092/original' if split=='original' else f'artifacts/patch-replication-093/{split}', 'development_predictions.npz')
  with np.load(_repo_io(path),allow_pickle=False) as z:
   assert z['ids'].tolist()==ids;y=z['y'];a=z['reference011'];b=z['attention_blend'];mean=z['mean_blend'];gain=(y-a)**2-(y-b)**2
  records=[]
  for j,name in enumerate(names):
   groups=np.minimum((R[:,j]*4).astype(int),3);rec={'metric':name,'Spearman_metric_squared_gain_all400':float(spearmanr(X[:,j],gain).statistic) if np.ptp(X[:,j]) else None,'Spearman_metric_squared_gain_fixed395':float(spearmanr(X[rest,j],gain[rest]).statistic) if np.ptp(X[rest,j]) else None,'quartiles_image_only':[]}
   for q in range(4):
    ix=np.flatnonzero(groups==q)
    if len(ix):rec['quartiles_image_only'].append({'quartile':q+1,'anchor_count':int(np.isin(ix,ti).sum()),**group_stats(y,a,b,ix)})
   records.append(rec)
  pairs=[]
  for pair in lock['pairs']:
   js=[ids.index(pair['anchor'])]+[ids.index(c['ID']) for c in pair['controls']];rows=[]
   for i in js:
    r={'ID':ids[i],'role':'anchor' if ids[i]==pair['anchor'] else 'image_matched_control','target':float(y[i]),'prediction011':float(a[i]),'prediction_attentionblend':float(b[i]),'prediction_meanblend':float(mean[i]),'target_minus011':float(y[i]-a[i]),'prediction_change':float(b[i]-a[i]),'squared_error_gain':float(gain[i]),'attention_over_mean_squared_gain':float((y[i]-mean[i])**2-(y[i]-b[i])**2)};rows.append(r);outputs.append({'partition':split,'anchor_group':pair['anchor'],**r})
   pairs.append(rows)
  results[split]={'top5':group_stats(y,a,b,ti),'matched15':group_stats(y,a,b,ci),'other395':group_stats(y,a,b,rest),'pairs':pairs,'all_metrics':records,'outcome_source_sha256':sha(path)}
 with (_repo_path(OUT, 'matched-outcomes.csv')).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(outputs[0]));w.writeheader();w.writerows(outputs)
 assert sha(_repo_path(ROOT, 'artifacts/submission-candidate-011.csv'))==plan['protected011_sha256']
 report={'study':'094 image-observable posthoc diagnosis','matching_locked_sha256':sha(_repo_path(OUT, 'matching-locked.json')),'results':results,'source_sha256':sha(Path(__file__)),'limits':plan['selection_bias']+' Metrics and descriptive matching ranks use400images; no predictivefit oroutcomeselection.30metrics inspected, no causality/significance claims. No joint1200sample inference. Image proxies lack physicaltruth.','scope':{'holdout_test_read':False,'new_model_fit':False,'cost':0,'upload':False,'best011_unchanged':True}}
 write(_repo_path(ROOT, 'reports/image-diagnostics-094.json'),report);print(json.dumps({s:{k:v for k,v in r.items() if k in ('top5','matched15','other395','pairs')} for s,r in results.items()}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['extract','analyze']);a=p.parse_args();extract() if a.mode=='extract' else analyze()
