"""Independent regionprops checks for saved-geometry measurement decomposition."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import frame_censoring_100 as s
import numpy as np
from skimage.measure import regionprops
from scipy.spatial.distance import cdist

def independent(lbl,kind='all'):
 rs=regionprops(lbl)
 if kind!='all':rs=[r for r in rs if r.area>=16]
 if kind=='interior':rs=[r for r in rs if min(r.bbox[:2])>0 and max(r.bbox[2:])<256]
 if kind=='border':rs=[r for r in rs if 0 in r.bbox[:2] or 256 in r.bbox[2:]]
 ar=np.array([r.area for r in rs]);d=np.array([r.equivalent_diameter_area for r in rs]);centers=np.array([r.centroid for r in rs]);nn=None
 if len(rs)>1:
  dd=cdist(centers,centers);np.fill_diagonal(dd,np.inf);nn=float(np.median(dd.min(1)))
 return {'count':len(rs),'covered_area_fraction':float(ar.sum()/lbl.size),'mean_area_px':float(ar.mean()) if len(rs) else None,'mean_diameter_px':float(d.mean()) if len(rs) else None,'area_weighted_diameter_px':float(np.average(d,weights=ar)) if len(rs) else None,'mean_inverse_sqrt_diameter':float(np.mean(d**-.5)) if len(rs) else None,'centroid_rms_radius_px':float(np.sqrt(np.mean(np.sum((centers-127.5)**2,axis=1)))) if len(rs) else None,'nearest_centroid_median_px':nn},rs

def run():
 start=time.perf_counter();p=s.check();report=s.read(s.OUT/'report.json');assert report['fixed_TV_measurements']==12;maxerr=0.;losschecks=0;stageschecked=0;trueareas={}
 for row in report['results']:
  ci=row['case'];hi=0 if row['h']==.1 else 1
  with np.load(_repo_io(s.OLD/'verified-fixtures'/f'case-{ci}.npz'),allow_pickle=False) as z:truth=z['canonical_truth'];pred=z['seg'][hi]
  truthbase,truthregions=independent(truth);trueareas[ci]=sorted([int(r.area) for r in truthregions]);assert truthbase['covered_area_fraction']==1.
  sources={'A0_truth_all':(truth,'all'),'A1_truth_same_foreground':(np.where(pred>0,truth,0),'all'),'A2_pred_all_positive':(pred,'all'),'A3_pred_area_ge16':(pred,'eligible'),'A4_pred_interior':(pred,'interior')}
  for stage,(lab,kind) in sources.items():
   metrics,_=independent(lab,kind)
   for name,value in metrics.items():
    other=row['stages'][stage][name]
    if value is None:assert other is None
    else:maxerr=max(maxerr,abs(value-other));assert abs(value-other)<1e-9,(ci,stage,name)
   stageschecked+=1
  assert row['stages']['A1_truth_same_foreground']['covered_area_fraction']==row['stages']['A2_pred_all_positive']['covered_area_fraction']
  for metric,d in row['decomposition'].items():
   if d is not None:assert abs(sum(d[k] for k in ['boundarymask','segmentation_partition','small_region_filter','frame_filter'])-d['total_interior_minus_truthall'])<1e-9
  _,inside=independent(pred,'interior');_,border=independent(pred,'border');_,tin=independent(truth,'interior');_,tb=independent(truth,'border');keep=np.isin(pred,[r.label for r in inside]);tedge=np.isin(truth,[r.label for r in tb]);lost=np.isin(truth,[r.label for r in tin])&np.isin(pred,[r.label for r in border]);c=row['correspondence'];assert int(keep.sum())==c['retained_interior_pixels'] and int((keep&tedge).sum())==c['retained_pixels_from_trueborder'] and int(lost.sum())==c['trueinterior_pixels_discarded_as_predborder'];losschecks+=1
  truthprops={r.label:r for r in truthregions};parent_counts={r.label:0 for r in truthregions};correct=0;validpixels=0;falsecomplete=0
  for region in regionprops(pred):
   yy,xx=region.coords.T;labels,n=np.unique(truth[yy,xx],return_counts=True);parent=int(labels[np.argmax(n)]);parent_counts[parent]+=1;correct+=int(n.max());validpixels+=int(n.sum())
   if region in []:raise AssertionError()
   is_inside=region.area>=16 and min(region.bbox[:2])>0 and max(region.bbox[2:])<256;t=truthprops[parent]
   if is_inside and (0 in t.bbox[:2] or 256 in t.bbox[2:]):falsecomplete+=1
  assert correct/validpixels==c['foreground_pixel_purity'];assert {str(k):v for k,v in parent_counts.items()}==c['majority_fragments_per_true'];assert falsecomplete==c['interior_regions_majority_trueborder']
 for lo,hi in [(0,1),(2,3),(4,5)]:assert trueareas[lo]==trueareas[hi]
 s.check();v={'passed':True,'stage_summaries_independently_verified':stageschecked,'measurements':12,'metric_max_abs_error':maxerr,'telescoping_decomposition_verified':True,'frame_contamination_checks':losschecks,'region_majority_correspondence_verified':True,'same_geometry_across_noise_verified':True,'reference_limited_to_observed_window':True,'source_sha256':s.sha(Path(__file__)),'report_sha256':s.sha(s.OUT/'report.json'),'seconds':time.perf_counter()-start,'new_segmentation':False,'new_synthetic_data':False,'hardness_labels_accessed':False,'holdout_test_accessed':False,'protected011_files_verified':len(p['protected011']),'cost':0};s.write(_repo_path(ROOT, 'reports/frame-censoring-100-validation.json'),v);print(json.dumps(v),flush=True)
if __name__=='__main__':run()
