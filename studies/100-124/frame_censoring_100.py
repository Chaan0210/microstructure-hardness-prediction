"""Saved six known-geometry fixtures only: segmentation versus frame censoring."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,csv,time,hashlib,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];assert Path(sys.prefix).resolve()==_repo_path(ROOT, '.venv')
import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image,ImageDraw,ImageFont
OUT=_repo_path(ROOT, 'artifacts/frame-censoring-100');OLD=_repo_path(ROOT, 'artifacts/nlm-measurement-098')
METRICS=['count','covered_area_fraction','mean_area_px','mean_diameter_px','area_weighted_diameter_px','mean_inverse_sqrt_diameter','centroid_rms_radius_px','nearest_centroid_median_px']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 with Path(p).open('x') as f:json.dump(d,f,indent=2);f.write('\n')
def prepare():
 OUT.mkdir(exist_ok=False);pub=read(_repo_path(ROOT, 'reports/candidate011-public-submission.json'));old=read(_repo_path(OLD, 'report.json'));v=read(_repo_path(ROOT, 'reports/grain-audit-097-nlm098-validation.json'));assert v['passed'] and v['NLM_fixture_cases_rerun']==6
 p={'study':'100 fixed frame-censoring decomposition on existing6 fixtures','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Read only098 savedfixture truth,raw and existingTVsegmentations0/1; no newimages,no newsegmentation,no NLMretrial,no hardness model/no realhardnesslabels/no holdout/test. Twoexistingh=.1/.2 atTVweight1*noise. Twelvefixedmeasurements.','duplicate_audit':'097 independentlyvalidated112overlays and supportpixeltransfers, not knowngeometry sizebias decomposition.098 measured count/PR on sixfixtures, not allvsinterior size/arrangement. Existinglineintercepts account endpointcensoring but do not answer thissamewatershed decomposition.',
 'reference':'Known finite256 rasterVoronoiregions. Fulluntruncated cells outsideimage are NOT available; sizes are observedin-window geometry, not unbiasedphysicalgrain diameters. Trueboundary pixels are assigned to knownregions incanonical_truth.',
 'ordered_decomposition':['A0: all known rastertruth regions, positiveIDs','A1: same truthlabels restricted to pixels where savedsegmentation>0; isolates area lost aswatershedline','A2: all positive predictedregions, no minimumarea orframefilter; isolates partition/split/merge discrepancy on identicalforeground','A3: predictedregions witharea>=16, noframefilter; isolates existingminimumarea exclusion','A4: A3 minus anyregionlabel on anyframeedge; isolates frame exclusion'],
 'metrics':METRICS,'size_definition':'Equivalentdiameter2sqrt(area/pi) inpx. Reportnumbermean andareaweightedmean separately. Count andarea are populationsummaries, not unbiasedestimators ofwhole-image numberaftercensoring. CentroidRMS aroundimagecenter andmedian nearestcentroiddistance are arrangementproxies. Zeroeligiblepopulation=>nullsizestats, not fabricatedzero.',
 'error_accounting':'For eachmetric, A4-A0=sum(A1-A0,A2-A1,A3-A2,A4-A3). Reportboundarymask,segmentation,minimumarea,frame contributions separately; accountingdepends onthisfixedorder. Also compare A4 with trueinterior-only target forlike-population errors.',
 'recoverability':'Assess whether keepingA3 insteadA4 improves absoluteerror versus finitewindow A0. Thisdoesnot validate clippedborderregions asfullgrains. Retaincomplete-only targetcomparison. Nofeature/threshold/parameterselection fromthese results.',
 'geometry_correspondence':'Contingencytable onpredictedforeground; majoritytruthassignment perpredictedregion withoutnewpuritythreshold. Reportextra majorityfragments,unrepresentedtruthregions,pixelpurity,conditional split/merge entropies. Falsecomplete pixels: retainedA4 pixels whose trueparenttouchesframe. Track trueinteriorpixels discarded byframefilter separately.',
 'noise_pairing':'Eachlayout has samegeometry atnoise.025/.075; compare fixedh separately, no averagingtochooseh. Keepall12 rows.',
 'proposal_policy':'Only propose one fixedmeasurementcontrol if evidence supportsrecoverableinformation; reportdiagnostic first. Do not implementregressionorchangeexistingfeatures in100.',
 'sources':{n:sha(_repo_path(ROOT, n)) for n in ['frame_censoring_100.py','validate_frame_censoring_100.py']},'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [_repo_path(OLD, 'plan.json'),_repo_path(OLD, 'report.json'),_repo_path(ROOT, 'reports/grain-audit-097-nlm098-validation.json'),*[_repo_path(OLD, 'verified-fixtures', f'case-{i}.npz') for i in range(6)]]},'protected011':pub['protected_sha256'],'cost':0}
 for n,h in pub['protected_sha256'].items():assert sha(_repo_path(ROOT, n))==h,n
 write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'plan_sha256':sha(_repo_path(OUT, 'plan.json'))}),flush=True)

def check():
 p=read(_repo_path(OUT, 'plan.json'))
 for n,h in {**p['sources'],**p['inputs'],**p['protected011']}.items():assert sha(_repo_path(ROOT, n))==h,n
 return p

def inventory(labels):
 sizes=np.bincount(labels.ravel());idx=np.flatnonzero(sizes);idx=idx[idx>0];edges=np.unique(np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]]);yy,xx=np.indices(labels.shape);sx=np.bincount(labels.ravel(),weights=xx.ravel(),minlength=len(sizes));sy=np.bincount(labels.ravel(),weights=yy.ravel(),minlength=len(sizes));return sizes,idx,edges,sx,sy

def summarize(labels,selection='all'):
 sizes,ids,edge,sx,sy=inventory(labels)
 if selection!='all':ids=ids[sizes[ids]>=16]
 if selection=='interior':ids=ids[~np.isin(ids,edge)]
 if selection=='border':ids=ids[np.isin(ids,edge)]
 area=sizes[ids].astype(float);total=float(area.sum());centers=np.column_stack([sy[ids]/area,sx[ids]/area]) if len(ids) else np.empty((0,2));diam=2*np.sqrt(area/np.pi)
 nn=None
 if len(ids)>1:
  dist=cdist(centers,centers);np.fill_diagonal(dist,np.inf);nn=float(np.median(dist.min(1)))
 return {'count':len(ids),'covered_area_fraction':total/labels.size,'mean_area_px':float(area.mean()) if len(ids) else None,'mean_diameter_px':float(diam.mean()) if len(ids) else None,'area_weighted_diameter_px':float(area@diam/total) if total else None,'mean_inverse_sqrt_diameter':float(np.mean(1/np.sqrt(diam))) if len(ids) else None,'centroid_rms_radius_px':float(np.sqrt(np.mean(np.sum((centers-127.5)**2,axis=1)))) if len(ids) else None,'nearest_centroid_median_px':nn,'ids':ids.tolist(),'areas':area.tolist(),'centroids_yx':centers.tolist()}

def correspondence(truth,pred):
 valid=pred>0;nt=int(truth.max()+1);ns=int(pred.max()+1);tab=np.bincount((truth[valid]*ns+pred[valid]).ravel(),minlength=nt*ns).reshape(nt,ns);sizes=tab.sum(0);pids=np.flatnonzero(sizes);pids=pids[pids>0];tids=np.unique(truth);tids=tids[tids>0];parent=tab.argmax(0);counts=np.bincount(parent[pids],minlength=nt);joint=tab.astype(float)/tab.sum();ti=joint.sum(1);sj=joint.sum(0);ii,jj=np.nonzero(joint);split=float(np.sum(joint[ii,jj]*np.log2(ti[ii]/joint[ii,jj])));merge=float(np.sum(joint[ii,jj]*np.log2(sj[jj]/joint[ii,jj])))
 ts,tall,tedge,_,_=inventory(truth);ss,sall,sedge,_,_=inventory(pred);eligible=sall[ss[sall]>=16];inside=eligible[~np.isin(eligible,sedge)];border=eligible[np.isin(eligible,sedge)];trueinside=tall[~np.isin(tall,tedge)];trueedge=np.isin(truth,tedge);retain=np.isin(pred,inside);losttrueinside=np.isin(truth,trueinside)&np.isin(pred,border);interiorpix=int(retain.sum())
 return {'foreground_pixel_purity':float(tab.max(0).sum()/tab.sum()),'split_entropy_bits':split,'merge_entropy_bits':merge,'majority_extra_fragments_all':int(np.maximum(counts[tids]-1,0).sum()),'truth_regions_without_majority_pred_all':int(np.sum(counts[tids]==0)),'retained_interior_pixels':interiorpix,'retained_pixels_from_trueborder':int((retain&trueedge).sum()),'falsecomplete_area_share_of_retained':float((retain&trueedge).sum()/interiorpix) if interiorpix else None,'interior_regions_majority_trueborder':int(np.isin(parent[inside],tedge).sum()),'trueinterior_pixels_discarded_as_predborder':int(losttrueinside.sum()),'trueinterior_total_pixels':int(np.isin(truth,trueinside).sum()),'trueinterior_area_discarded_by_frame_fraction':float(losttrueinside.sum()/np.isin(truth,trueinside).sum()) if trueinside.size else None,'majority_fragments_per_true':{str(int(t)):int(counts[t]) for t in tids}}

def codes(seg):
 size,ids,edge,_,_=inventory(seg);c=np.zeros(len(size),np.uint8);c[ids[size[ids]<16]]=3;eligible=ids[size[ids]>=16];c[eligible]=1;c[eligible[np.isin(eligible,edge)]]=2;return c[seg]

def run():
 check();start=time.perf_counter();old=read(_repo_path(OLD, 'report.json'));results=[];flat=[];font=ImageFont.load_default(size=14);canvas=Image.new('RGB',(1280,6*300),'white');palette=np.array([[0,0,0],[35,195,85],[245,145,35],[210,35,210]],np.uint8)
 for ci,case in enumerate(old['cases']):
  with np.load(_repo_path(OLD, 'verified-fixtures', f'case-{ci}.npz'),allow_pickle=False) as z:truth=z['canonical_truth'].copy();bound=z['canonical_boundary'].copy();segs=z['seg'][:2].copy();raw=z['raw'].copy()
  # Restore saved canonical orientation by the same lexicographic D4 representative, no newsegmentation.
  choices=[np.ascontiguousarray(z) for k in range(4) for z in [np.rot90(raw,k),np.fliplr(np.rot90(raw,k))]];canonical=min(choices,key=lambda z:z.tobytes());oracle=summarize(truth);oracleinterior=summarize(truth,'interior');oracleborder=summarize(truth,'border')
  for hi,h in enumerate([.1,.2]):
   pred=segs[hi];stages={'A0_truth_all':oracle,'A1_truth_same_foreground':summarize(np.where(pred>0,truth,0)),'A2_pred_all_positive':summarize(pred),'A3_pred_area_ge16':summarize(pred,'eligible'),'A4_pred_interior':summarize(pred,'interior')};keys=list(stages);decomp={}
   for metric in METRICS:
    vals=[stages[k][metric] for k in keys]
    if any(v is None for v in vals):decomp[metric]=None;continue
    diffs=np.diff(vals);assert abs(float(diffs.sum())-(vals[-1]-vals[0]))<1e-9
    decomp[metric]={'boundarymask':float(diffs[0]),'segmentation_partition':float(diffs[1]),'small_region_filter':float(diffs[2]),'frame_filter':float(diffs[3]),'total_interior_minus_truthall':float(vals[-1]-vals[0])}
   corr=correspondence(truth,pred);errors={}
   for metric in ['mean_diameter_px','area_weighted_diameter_px','mean_area_px','nearest_centroid_median_px']:
    a0=oracle[metric];a3=stages['A3_pred_area_ge16'][metric];a4=stages['A4_pred_interior'][metric];oi=oracleinterior[metric]
    errors[metric]={'all_pred_relative_error_vs_truthall':(a3-a0)/a0 if a3 is not None and a0 else None,'interior_pred_relative_error_vs_truthall':(a4-a0)/a0 if a4 is not None and a0 else None,'interior_pred_relative_error_vs_truthinterior':(a4-oi)/oi if a4 is not None and oi else None,'oracle_interior_relative_shift_vs_truthall':(oi-a0)/a0 if oi is not None and a0 else None,'all_population_has_smaller_abs_error_vs_truthall':bool(abs(a3-a0)<abs(a4-a0)) if a3 is not None and a4 is not None else None}
   row={'case':ci,'layout':case['layout'],'noise_sigma':case['noise_sigma'],'TV_noise_multiplier':1.,'h':h,'stages':stages,'oracle_interior':oracleinterior,'oracle_border':oracleborder,'predicted_border':summarize(pred,'border'),'decomposition':decomp,'correspondence':corr,'errors':errors};results.append(row)
   f={'case':ci,'layout':case['layout'],'noise':case['noise_sigma'],'h':h}
   for stage,z in stages.items():
    for metric in METRICS:f[stage+'__'+metric]=z[metric]
   f.update({k:v for k,v in corr.items() if k!='majority_fragments_per_true'});flat.append(f)
  # Fixed h.2 visualization only; numericalreport retainsbothh.
  pred=segs[1];c=codes(pred);truthsupport=codes(truth);rawrgb=np.repeat(canonical[:,:,None],3,2);oraclepic=rawrgb.copy();oraclepic[bound]=[30,220,70];overlay=rawrgb.copy();overlay[pred==0]=[255,35,30];edgeids=inventory(truth)[2];false=(c==1)&np.isin(truth,edgeids);fp=rawrgb.copy();fp[false]=[190,35,225]
  pics=[oraclepic,palette[truthsupport],overlay,palette[c],fp];titles=['Known boundary','Oracle interior/frame','TV h.2 boundary','TV h.2 support','Retained from trueframe']
  for j,(pic,title) in enumerate(zip(pics,titles)):
   ImageDraw.Draw(canvas).text((j*256+3,ci*300+2),f'{title}\nN{case["true_count"]} sigma{case["noise_sigma"]}',fill='black',font=font);canvas.paste(Image.fromarray(pic),(j*256,ci*300+42))
 canvas.save(_repo_path(OUT, 'geometry-decomposition.png'))
 with (_repo_path(OUT, 'metrics.csv')).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 report={'cases':6,'fixed_TV_measurements':12,'results':results,'elapsed_seconds':time.perf_counter()-start,'plan_sha256':sha(_repo_path(OUT, 'plan.json')),'no_new_segmentation':True,'no_hardness_training':True,'reference_is_finitewindow_geometry':True,'cost':0};write(_repo_path(OUT, 'report.json'),report);check();print(json.dumps({'complete':True,'cases':6,'measurements':12,'seconds':report['elapsed_seconds'],'report_sha256':sha(_repo_path(OUT, 'report.json'))}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);a=ap.parse_args();{'prepare':prepare,'run':run}[a.mode]()
