"""Independent pixel bookkeeping, artifact QA, and fixed fixture rerun."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,time,hashlib,csv,ast,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import disk
import nlm_measurement_pilot_098 as nlm
start=time.perf_counter();o=_repo_path(ROOT, 'artifacts/grain-visual-audit-097');plan=json.loads((_repo_path(o, 'plan.json')).read_text());sel=json.loads((_repo_path(o, 'selection-locked.json')).read_text());rec=json.loads((_repo_path(o, 'segmentation-records.json')).read_text());sp=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,h in {**plan['source_sha256'],**plan['protected011']}.items():assert sha(_repo_path(ROOT, name))==h,name
with (_repo_path(o, 'all400_noise_contrast.csv')).open() as f:rows=list(csv.DictReader(f))
assert {r['ID'] for r in rows}==set(sp['development_folds']) and not set(sp['holdout'])&set(rec)
noise=np.array([float(r['noise']) for r in rows]);contrast=np.array([float(r['contrast_q95_q05']) for r in rows]);cuts=np.quantile(noise,[1/3,2/3]);ct=float(np.median(contrast));np.testing.assert_array_equal(cuts,sel['noise_cuts']);assert ct==sel['contrast_cut'];selected=[]
for i in range(3):
 for j in range(2):
  pool=[r['ID'] for r in rows if np.searchsorted(cuts,float(r['noise']),side='right')==i and int(float(r['contrast_q95_q05'])>=ct)==j];pool.sort(key=lambda ident:hashlib.sha256(('097-label-blind-v1:'+ident).encode()).hexdigest());selected.extend(pool[:2])
assert selected==[r['ID'] for r in sel['selected']]
verified=0;loss=0;borderloss=0
for ident,r in rec.items():
 assert sha(_repo_path(ROOT, 'data/train', f'{ident}.png'))==r['image_sha256']
 with np.load(_repo_path(o, 'arrays', f'{ident}.npz'),allow_pickle=False) as z:
  a=z['raw'];c=z['canonical'];expected=np.rot90(a,r['canonical_rot90']);expected=np.fliplr(expected) if r['canonical_flip'] else expected;np.testing.assert_array_equal(c,expected)
  with Image.open(_repo_path(o, 'panels', f'{ident}.png')) as im:pixels=np.asarray(im);assert im.size==(1280,916)
  np.testing.assert_array_equal(pixels[76:332,:256],np.repeat(c[:,:,None],3,2))
  for idx,(seg,support,stat) in enumerate(zip(z['seg'],z['support'],r['configs'])):
   sizes=np.bincount(seg.ravel());edge=np.unique(np.r_[seg[0,:],seg[-1,:],seg[:,0],seg[:,-1]]);codes=np.where(sizes>=16,1,3).astype(np.uint8);codes[edge[sizes[edge]>=16]]=2;codes[0]=0;np.testing.assert_array_equal(support,codes[seg]);labels=np.arange(len(sizes));count=int(np.sum((sizes>=16)&(labels>0)));inside=int(np.sum((codes==1)&(labels>0)));assert count==stat['grain_count'] and inside==stat['interior_count']
   for key,val in [('boundary_fraction',0),('interior_area_fraction',1),('border_area_fraction',2),('small_area_fraction',3)]:assert float(np.mean(support==val))==stat[key]
   row=1 if idx in [0,1,4,5] else 2;cols=[0,1,4,5] if row==1 else [2,3,6,7];col=cols.index(idx);y0=40+row*292+36;x0=col*256;expected=np.repeat(c[:,:,None],3,2);expected[seg==0]=[255,50,35];np.testing.assert_array_equal(pixels[y0:y0+256,x0:x0+256],expected);verified+=1
  if ident in selected:
   for wi,si in [(0,2),(1,3),(4,6),(5,7)]:
    lost=(z['support'][wi]==1)&(z['support'][si]!=1);loss+=int(lost.sum());borderloss+=int((lost&(z['support'][si]==2)).sum())
a=json.loads((_repo_path(o, 'assessment.json')).read_text());assert loss==a['matched48_config_support_transitions']['lost_interior_pixels_total'] and borderloss==a['matched48_config_support_transitions']['lost_to_border_pixels']
# Recreate frozen six fixtures and recompute count/precision/recall independently.
no=_repo_path(ROOT, 'artifacts/nlm-measurement-098');npplan=json.loads((_repo_path(no, 'plan.json')).read_text());nr=json.loads((_repo_path(no, 'report.json')).read_text());assert sha(_repo_path(no, 'plan.json'))==nr['plan_sha256']
for name,h in npplan['source_sha256'].items():assert sha(_repo_path(ROOT, name))==h,name
(_repo_path(no, 'verified-fixtures')).mkdir(exist_ok=False);metric_error=0.;count_errors={'tv':[],'nlm':[]};boundary_pass=True;noninferior=True
for i,case in enumerate(nr['cases']):
 layout=case['layout'];sigma=case['noise_sigma'];severity=0 if sigma==.025 else 1;truth,bound,blur=nlm.fixture(layout);rng=np.random.default_rng(20270401+layout*10+severity);raw=np.clip(np.rint(255*(blur+rng.normal(0,sigma,blur.shape))),0,255).astype(np.uint8);ct,cb=nlm.transform_truth(raw,truth,bound);segs=[]
 for method in ['tv','nlm']:
  _,_,outs,_=nlm.segment(raw,method);counts=[]
  for j,(seg,_) in enumerate(outs):
   sizes=np.bincount(seg.ravel());count=int(np.sum(sizes[1:]>=16));counts.append(count);stored=case['arms'][method]['configs'][j];assert count==stored['count'];roi=np.zeros((256,256),bool);roi[2:-2,2:-2]=True;p=(seg==0)&roi;t=cb&roi;prec=float(ndi.binary_dilation(t,structure=disk(2))[p].mean());recall=float(ndi.binary_dilation(p,structure=disk(2))[t].mean());metric_error=max(metric_error,abs(prec-stored['boundary_precision']),abs(recall-stored['boundary_recall']));assert abs(prec-stored['boundary_precision'])<1e-12 and abs(recall-stored['boundary_recall'])<1e-12
   if method=='nlm':boundary_pass=boundary_pass and prec>=.6 and recall>=.8
   segs.append(seg)
  error=abs(float(np.median(counts))-len(np.unique(truth)))/len(np.unique(truth));assert error==case['arms'][method]['relative_count_error'];count_errors[method].append(error)
 noninferior=noninferior and count_errors['nlm'][-1]<=count_errors['tv'][-1]+.1
 np.savez_compressed(_repo_path(no, 'verified-fixtures', f'case-{i}.npz'),raw=raw,canonical_truth=ct,canonical_boundary=cb,seg=np.stack(segs))
means={m:float(np.mean(z)) for m,z in count_errors.items()};assert means==nr['mean_relative_count_error'];gate=means['nlm']<=.75*means['tv'] and noninferior and boundary_pass;assert bool(gate)==nr['fidelity_gate_pass']
for name,h in plan['protected011'].items():assert sha(_repo_path(ROOT, name))==h,name
r={'passed':True,'selected12_reproduced_without_labels':True,'images_verified':len(rec),'segmentations_and_pixel_overlays_verified':verified,'canonical_alignment_exact':True,'support_partition_and_counts_independently_verified':True,'interior_to_border_loss_fraction':borderloss/loss,'NLM_fixture_cases_rerun':len(nr['cases']),'independent_dilation_PR_max_error':metric_error,'NLM_gate_reproduced':bool(gate),'protected011_files_verified':len(plan['protected011']),'real_hardness_labels_accessed':False,'holdout_test_images_accessed':False,'seconds':time.perf_counter()-start,'source_sha256':sha(Path(__file__)),'audit_assessment_sha256':sha(_repo_path(o, 'assessment.json')),'NLM_report_sha256':sha(_repo_path(no, 'report.json')),'cost':0}
with (_repo_path(ROOT, 'reports/grain-audit-097-nlm098-validation.json')).open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r),flush=True)
