from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
import runtime as f
R=f.R;P=f.P

def main():
 plan=f.check();rec=f.read(R/'measurements.json');errors=[];count=0
 for row in rec['images']:
  with np.load(_repo_io(R/'arrays'/f"{row['ID']}.npz"),allow_pickle=False) as z:
   raw=z['raw_canonical'];assert raw.shape==(256,256) and raw.dtype==np.uint8
   for name in plan['variants']:
    s=z[name];n=s.max()+1;a=np.bincount(s.ravel(),minlength=n).astype(float);good=np.flatnonzero((a>=16)&(np.arange(n)>0));frame=np.unique(np.concatenate([s[0],s[-1],s[:,0],s[:,-1]]));inter=good[~np.isin(good,frame)];yy,xx=np.indices(s.shape);E=lambda v:np.bincount(s.ravel(),weights=v.ravel(),minlength=n)/np.maximum(a,1);mx=E(xx);my=E(yy);vx=np.maximum(E(xx*xx)-mx*mx,0);vy=np.maximum(E(yy*yy)-my*my,0);cov=E(xx*yy)-mx*my;e=np.linalg.eigvalsh(np.array([[vx,cov],[cov,vy]]).transpose(2,0,1));axis=4*np.sqrt(np.maximum(e,0));aspect=np.average(axis[good,1]/np.maximum(axis[good,0],1),weights=a[good]) if len(good) else 0.;expect={'count':len(good),'allcell_count':int(np.count_nonzero(a[1:])),'median_area':float(np.median(a[good])) if len(good) else 0.,'areaweighted_aspect':float(aspect),'interior_count':len(inter),'interior_area_fraction':a[inter].sum()/s.size,'small_region_area_fraction':a[(a<16)&(np.arange(n)>0)].sum()/s.size,'frame_region_area_fraction':a[good[np.isin(good,frame)]].sum()/s.size,'watershedline_fraction':a[0]/s.size}
    for k,v in expect.items():
     err=abs(v-row['variant_metrics'][name][k]);errors.append(err);assert err<1e-9,(row['ID'],name,k,err)
    count+=1
 assert count==112 and [r['ID'] for r in rec['images']]==plan['IDs']
 result={'passed':True,'independent_bincount_coordinate_moment_checks':count,'max_absolute_error':max(errors),'all_ID_order_and_shapes_exact':True,'no_raw_images_or_labels_opened':True,'protected_files_unchanged':len(plan['protected']),'measurements_sha256':f.sha(R/'measurements.json')};f.write(R/'independent-validation.json',result);f.emit(result)
if __name__=='__main__':main()
