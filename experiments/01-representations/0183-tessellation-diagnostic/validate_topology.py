from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
import runtime as f
R=f.R

def edges(s):
 result=set()
 for a,b in [(s[:-1].ravel(),s[1:].ravel()),(s[:,:-1].ravel(),s[:,1:].ravel())]:
  for u,v in zip(a,b):
   if u!=v:result.add((min(int(u),int(v)),max(int(u),int(v))))
 return result

def main():
 p=f.check();m=f.read(R/'measurements.json');errors=[];n=0;passed=0
 for r in m['images']:
  with np.load(_repo_io(R/'arrays'/f"{r['ID']}.npz"),allow_pickle=False) as z:
   fs={};gs={}
   for name in p['variants']:
    s=z[name];ii=ndi.distance_transform_edt(s==0,return_distances=False,return_indices=True);s=s[tuple(ii)];fs[name]=s;gs[name]=edges(s)
   base=fs['base'];al,ai=np.unique(base,return_inverse=True);ac=np.bincount(ai.ravel());probe_ok=True
   for name in list(p['variants'])[1:]:
    b=fs[name];bl,bi=np.unique(b,return_inverse=True);bc=np.bincount(bi.ravel());counts=np.zeros((len(al),len(bl)),np.int64);np.add.at(counts,(ai.ravel(),bi.ravel()),1);aa,bb=linear_sum_assignment(counts,maximize=True);iou=np.array([counts[i,j]/(ac[i]+bc[j]-counts[i,j]) for i,j in zip(aa,bb)]);good=iou>=.5;mapping={int(bl[j]):int(al[i]) for i,j in zip(aa[good],bb[good])};es=set()
    for u,v in gs[name]:
     u=mapping.get(u,-u-1);v=mapping.get(v,-v-1);es.add((min(u,v),max(u,v)))
    union=gs['base']|es;jaccard=len(gs['base']&es)/len(union) if union else 1.;matched=sum(int(ac[i]) for i in aa[good])/base.size;ref=r['probe_stability'][name];errors.extend([abs(jaccard-ref['adjacency_Jaccard']),abs(matched-ref['matched_sourcepixel_fraction'])]);n+=1;probe_ok &= jaccard>=.9 and matched>=.9 and max(ref['relative_changes'].values())<=.1
   assert bool(probe_ok)==r['stability_pass'];passed+=probe_ok
 assert n==96 and max(errors)==0 and passed==m['aggregate']['stability_pass_images']
 result={'passed':True,'independent_pixel_loop_graph_and_overlap_table_pairs':n,'max_error':max(errors),'stability_pass_images':int(passed),'source_coordinate_moment_validation_passed':f.read(R/'independent-validation.json')['passed'],'no_raw_images_or_target_arrays_decoded':True,'protected_hashes_unchanged':len(p['protected'])};f.write(R/'topology-validation.json',result);f.emit(result)
if __name__=='__main__':main()
