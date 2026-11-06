from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,time,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from skimage.measure import regionprops
from PIL import Image
P=Path(__file__).resolve().parents[4];R=Path(__file__).resolve().parent;sys.path[:0]=[str(R),str(_repo_path(P, 'studies'))]
import diagnose as d
import grain_features as gf
d.protect();d.guard();start=time.monotonic();old=d.load(_repo_path(P, 'artifacts/run-006/development_features.npz'));base=d.data('original');np.testing.assert_array_equal(old['ids'],base['ids']);names=d.read(_repo_path(P, 'artifacts/run-006/report.json'))['feature_names'];prov=d.read(_repo_path(P, 'artifacts/run-012/input-provenance.json'));assert d.sha(_repo_path(P, 'artifacts/run-006/development_features.npz'))==prov['grain_feature_cache_sha256']
plan={'reason':'Active compact57 contains both oldGaussian26 and TV31. InitialL4 plan measured TV31; source audit identified Gaussian sigma2 empty0 so add complete all400/all2scale coverage, without changing previous reports or selecting rows by errors.','cache_counts_seen_before_extension':{'s1_empty0':0,'s2_empty0':3},'scope':'diagnostic only; fixed existing sigma1/2 and area16; independent regioncounts/border areas and cachedsize statistics verification; no tuning','source':{p:d.sha(_repo_path(P, p)) for p in ['grain_features.py','compact_regression.py','artifacts/run-006/development_features.npz','artifacts/run-012/input-provenance.json']},'script_sha256':d.sha(__file__)};d.write(_repo_path(R, 'L4-legacy-extension-plan-v2.json'),plan)
rows=[];maxerr=0
for i,ident in enumerate(old['ids']):
 path=_repo_path(P, 'data/train', f'{ident}.png');assert d.sha(path)==prov['dev_image_sha256'][str(ident)]
 with Image.open(_repo_io(path)) as im:a=np.asarray(im)
 rr=[]
 for sigma in [1,2]:
  x,ridge,edge,lab=gf.segment(a,float(sigma));allr=[r for r in regionprops(lab) if r.area>=16];inside=[r for r in allr if r.bbox[0]>0 and r.bbox[1]>0 and r.bbox[2]<256 and r.bbox[3]<256];border=[r for r in allr if r.bbox[0]==0 or r.bbox[1]==0 or r.bbox[2]==256 or r.bbox[3]==256];di=np.array([r.equivalent_diameter_area for r in inside]);da=np.array([r.equivalent_diameter_area for r in allr]);prefix=f'grain_s{sigma}'
  checks={'count':len(allr),'interior_count':len(inside),'interior_diam_mean':float(di.mean()) if di.size else 0.,'interior_hall_petch':float(np.mean(1/np.sqrt(di))) if di.size else 0.,'all_diam_mean':float(da.mean())}
  for key,v in checks.items():e=abs(v-old['X'][i,names.index(prefix+'_'+key)]);assert e<1e-12;maxerr=max(maxerr,e)
  rr.append({'sigma':sigma,**checks,'empty0':len(inside)==0,'border_count':len(border),'border_area_fraction':sum(r.area for r in border)/65536,'interior_area_fraction':sum(r.area for r in inside)/65536})
 rows.append({'ID':str(ident),'configs':rr})
 if (i+1)%100==0:d.emit({'legacy_grain_images':i+1,'seconds':time.monotonic()-start})
d.write(_repo_path(R, 'L4-legacy-measurements.json'),{'rows':rows,'cached_descriptor_maxerror':maxerr,'seconds':time.monotonic()-start})
y=base['y'];res=y-base['blend_L_MLP'];se=np.mean([(d.data(s)['y']-d.data(s)['blend_L_MLP'])**2 for s in ['original','20261031','20261101']],axis=0);idx=np.argsort(y,kind='stable');out={}
for j,s in enumerate([1,2]):
 empty=np.array([r['configs'][j]['empty0'] for r in rows]);features={key:np.array([r['configs'][j][key] for r in rows]) for key in ['border_area_fraction','interior_area_fraction','interior_diam_mean','all_diam_mean','count','interior_count']};groups={}
 masks={'empty0':empty,'nonempty':~empty,'target_low40':np.isin(np.arange(400),idx[:40]),'target_high40':np.isin(np.arange(400),idx[-40:])}
 for key in ['border_area_fraction','interior_diam_mean']:
  for k,ix in enumerate(np.array_split(np.argsort(features[key],kind='stable'),4)):masks[key+f'_quartile{k+1}']=np.isin(np.arange(400),ix)
 for name,m in masks.items():groups[name]={'n':int(m.sum()),'IDs':base['ids'][m].tolist(),'pooled_RMSE':float(np.sqrt(se[m].mean())) if m.any() else None,'mean_residual_original':float(res[m].mean()) if m.any() else None,'median_features':{k:float(np.median(v[m])) if m.any() else None for k,v in features.items()}}
 out['sigma'+str(s)]={'empty0_n':int(empty.sum()),'empty0_fraction':float(empty.mean()),'feature_quantiles':{k:np.quantile(v,[0,.1,.5,.9,1]).tolist() for k,v in features.items()},'spearman':{k:{n:float(spearmanr(v,t).statistic) for n,t in [('target',y),('signed_residual_original',res),('pooled_SE',se)]} for k,v in features.items()},'groups':groups}
# Check actual compact descriptor encoding, without modifying it.
comp=d.load(_repo_path(P, 'artifacts/run-012/development_features.npz'));ix1=names.index('grain_s1_interior_hall_petch');ix2=names.index('grain_s2_interior_hall_petch');np.testing.assert_array_equal(comp['X'][:,0],.5*(old['X'][:,ix1]+old['X'][:,ix2]));out['active_zero_encoding']='At three sigma2 empty images the compact inverse_sqrt descriptor averages the valid sigma1 value with sigma2 sentinel0; this halves the sigma1 value. No model impact is inferred without controlled validation.'
out['seconds']=time.monotonic()-start;out['cache_reconstruction_maxerror']=maxerr;out['best_preserved']=True;d.protect();d.write(_repo_path(R, 'L4-legacy.json'),out);d.emit({k:{'empty0_n':v['empty0_n'],'border_quantiles':v['feature_quantiles']['border_area_fraction'],'emptygroup':v['groups']['empty0']} for k,v in out.items() if k.startswith('sigma')})
