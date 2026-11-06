"""Bounded, target-blind boundary-layout audit; no predictor or label imports."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[name]='2'
import sys,json,time,hashlib,datetime,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];assert Path(sys.prefix).resolve()==_repo_path(ROOT, '.venv');OUT=_repo_path(ROOT, 'artifacts/layout-dependency-160')
import numpy as np
from scipy import ndimage as ndi
from PIL import Image,ImageDraw,ImageFont

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,z):
 with Path(p).open('x') as f:json.dump(z,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def emit(z):print(json.dumps(z,ensure_ascii=False),flush=True)
def transform(a,k):
 z=np.rot90(a,k//2);return np.fliplr(z) if k%2 else z

def edge(a):
 a=np.asarray(a,dtype=float)/255.;gx=ndi.gaussian_filter(a,2.,order=(0,1),mode='reflect');gy=ndi.gaussian_filter(a,2.,order=(1,0),mode='reflect');return np.hypot(gx,gy)
def descriptor(a):
 E=edge(a);F=E.reshape(64,4,64,4).mean((1,3));Z=F-F.mean();norm=np.linalg.norm(Z);return E,F,Z/max(float(norm),1e-15)
def raw(ident):
 ids=read(_repo_path(OUT, 'plan.json'))['development_ids'];assert ident in ids
 with Image.open(_repo_path(ROOT, 'data/train', f'{ident}.png')) as im:assert im.mode=='L' and im.size==(256,256);return np.array(im,dtype=np.uint8)

def prepare():
 OUT.mkdir(exist_ok=False);(_repo_path(OUT, 'panels')).mkdir();m=read(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json'));ids=sorted(m['development_folds']);assert len(ids)==400
 names=['layout_dependency_audit_160.py','reports/train-similarity-audit.json','grain_visual_audit_097.py','artifacts/grain-visual-audit-097/pair-comparison.json','bootstrap_selector_144.py','disagreement_study_156.py','candidate020_pipeline.py','resume_candidate020_fit_v2.py','artifacts/bootstrap-selector-144/plan.json','artifacts/disagreement-156/plan.json','artifacts/final-model-157/fit.json','reports/cluster-pair-159-design.json','reports/cluster-pair-159-summary.json']
 p={'study':'160 bounded image dependency audit','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'development_ids':ids,'scope':'Onlyraw400allowlisteddevelopmentimages. No targetCSV,OOF,predictionarrays,holdout/testpixels,modelinference/training. Existingpublicscores andperformance histories are irrelevant to pairselection/review. ProtectedCSV/ZIP onlybytehashed,neverparsed.','prior_evidence':{'exact_D4':'Existing canonical-hashgroups alreadydone;notrepeated.','train_similarity':read(_repo_path(ROOT, 'reports/train-similarity-audit.json')),'097_pair_only':read(_repo_path(ROOT, 'artifacts/grain-visual-audit-097/pair-comparison.json'))['best_bounded_geometry_match'],'gap':'Noexistingall400boundary-layout/top-pairjunctioninspection found.097testedoneexternallysuppliedpair,notall400.094descriptormatchingwasmorphology/textureandoutcome-selectedanchors,notgeometryreuseevidence.'},'fixed_descriptor':'Gaussian derivative sigma2nativepx,reflectpadding,magnitude sqrt(gx^2+gy^2). Meanpool4x4 ->64x64edgefield. Perimage centerandL2normalize. Fullspatialcoordinatesretained; no histogram/DINO/CNN feature. Positiveaffineintensity invariance absentclipping; mildblur tolerance heuristic,notinvariance.','fixed_alignment':'Exactly8D4transforms ofsecondedgefield; no translations,scale,crop,elastic registration oralignmentparametersearch. This revisits approximate geometryunderphotometricchange,not exactpixelduplicatehashing.','ranking':'Pearson/cosineofcenterededgefields; max8D4score perunorderedpair. All79800pairs,rankdescendingthenlexicographicIDtie. Freeze exacttop8pairs beforevisualreview;allowIDrepetition,noreplacement. No feature/threshold/rankexpansion afterviewing.','review':'View all8samepanels:rawA/rawalignedB/edgeoverlay pluscontrast-normalizedA/B andsmoothedimageblend. Recognize >=5distinctjunctions withsamebranchlayout across>=3quadrants (<=3nativepx afterfixedD4), AND>=3connectedcurvedboundarysegments each>=20px following samepath within3px. Overallnetworkmustbeconsistent;similaraveragegrainshape/noise/texture alonefails. Recordexplicitlandmarkcoordinatesifaccepted. Numericalscorealonecannotcreateagroup. Ifuncertain,notconfirmed;do not forcegroup.','group_policy':'Onlyvisuallyconfirmedcorrespondences defineedges;reviewedconfirmedconnectedcomponents arecandidate groups,withdirectevidenceperedge. Freezegroupsbeforelookingatfoldmaps;reportwhetherexistingfoldssplitconfirmedgroups. Ifnoneconfirmed, no newgroups/foldchanges andcloseboundednegativeaudit.','fixtures':'KnownsamerectangularjitteredVoronoilayout underpositiveaffinebrightness/contrast andextraGaussianblur1.2; exactunclippedaffine andD4descriptorcommutation checks; independentdotproducts. Syntheticfixturesonlyengineering not materialtruth/detectionrecall estimate.','resources':'ExistingCPU2/PIL/numpy/scipy; noinstall/GPU/payment. 400edgefields+8matrixproducts; oneexistingcostfreeprocessor. Stop iffirst10projected400>120sec orRSS>2GB; scoringsoftlimit120sec.','limits':'Top8underonescorecannotexcludeweak/shifted/cropped/rescaled/partial/geometricallychangedorothergeneratorfamilies. Sharedgeneratingparametersmaynotreuseidenticalboundaries. Noise/blurcanhidejunctions. No performance/noiselowerbound guaranteed.','159_interpretation':'Quantilebinarymasksnearlyfixareafraction andprimarilymeasure rank-basedconnectivity.159negative resultdoesNOT implytruephasefraction useless oridentifiedphaseconnectivity useless.','hashes':{n:sha(_repo_path(ROOT, n)) for n in names},'protected':{str(p.relative_to(ROOT)):sha(p) for p in [_repo_path(ROOT, 'artifacts/submission-candidate-018.csv'),_repo_path(ROOT, 'artifacts/submission-candidate-019.csv'),_repo_path(ROOT, 'artifacts/submission-candidate-020.csv'),_repo_path(ROOT, 'deliverables/reproduction-018-v155.zip'),_repo_path(ROOT, 'deliverables/reproduction-011-014-015-v136.zip')]},'split_manifest_sha256':sha(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')),'cost':0}
 write(_repo_path(OUT, 'plan.json'),p);write(_repo_path(ROOT, 'reports/layout-dependency-160-design.json'),p);emit({'plan_locked':True,'images':400,'pairs_to_review':8,'label_access':False,'plan_sha256':sha(_repo_path(OUT, 'plan.json'))})
def check():
 p=read(_repo_path(OUT, 'plan.json'))
 for name,h in {**p['hashes'],**p['protected']}.items():assert sha(_repo_path(ROOT, name))==h,name
 return p

def fixtures():
 rng=np.random.default_rng(20271221);yy,xx=np.indices((256,256));seeds=np.array([(y,x) for y in (24,72,120,168,216) for x in (24,72,120,168,216)],float)+rng.uniform(-14,14,(25,2));D=(yy[:,:,None]-seeds[:,0])**2+(xx[:,:,None]-seeds[:,1])**2;L=D.argmin(2);boundary=np.zeros_like(L,bool);boundary[1:]|=L[1:]!=L[:-1];boundary[:,1:]|=L[:,1:]!=L[:,:-1];a=rng.uniform(130,210,25)[L];a[boundary]=35;a=ndi.gaussian_filter(a,.6)+rng.normal(0,3,a.shape);a=np.clip(a,0,255);_,_,Z=descriptor(a);_,_,Za=descriptor(.65*a+30);affineerr=float(abs(Z-Za).max());assert affineerr<1e-14
 b=ndi.gaussian_filter(.65*a+30,1.2);_,_,Zb=descriptor(b);score=float(np.sum(Z*Zb));assert score>.9
 errs=[]
 for k in range(8):
  _,_,Zr=descriptor(transform(a,k));errs.append(float(abs(Zr-transform(Z,k)).max()));assert errs[-1]<1e-14
 return {'passed':True,'affine_max_error':affineerr,'extra_blur1_2_similarity':score,'D4_max_error':max(errs),'no_label_access':True,'fixture_only_no_material_or_recall_claim':True}

def run():
 p=check();write(_repo_path(OUT, 'fixtures.json'),fixtures());ids=p['development_ids'];F=[];imagehash={};start=time.perf_counter()
 for i,ident in enumerate(ids):
  a=raw(ident);_,_,z=descriptor(a);F.append(z.astype(np.float64));imagehash[ident]=sha(_repo_path(ROOT, 'data/train', f'{ident}.png'))
  if i==9:
   b={'first10_seconds':time.perf_counter()-start,'projected400_seconds':40*(time.perf_counter()-start),'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};b['passed']=b['projected400_seconds']<120 and b['RSS_bytes']<2e9;write(_repo_path(OUT, 'benchmark.json'),b);assert b['passed'];emit(b)
 F=np.array(F);best=np.full((400,400),-np.inf);orient=np.zeros((400,400),np.int8);scstart=time.perf_counter();A=F.reshape(400,-1)
 for k in range(8):
  B=np.stack([transform(f,k) for f in F]).reshape(400,-1);scores=A@B.T;better=scores>best;best[better]=scores[better];orient[better]=k;emit({'D4_scoring_completed':k+1})
 scoreseconds=time.perf_counter()-scstart;assert scoreseconds<120;rows=[{'ID_A':ids[i],'ID_B':ids[j],'similarity':float(best[i,j]),'D4_B':int(orient[i,j]),'indices':[i,j]} for i in range(400) for j in range(i+1,400)];rows.sort(key=lambda r:(-r['similarity'],r['ID_A'],r['ID_B']));top=rows[:8]
 for rank,r in enumerate(top,1):
  i,j=r['indices'];manual=float(sum((F[i]*transform(F[j],r['D4_B'])).ravel()));assert abs(manual-r['similarity'])<1e-12;r['rank']=rank;r['independent_dot_max_error']=abs(manual-r['similarity'])
 with (_repo_path(OUT, 'ranking.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),edge_descriptors=F,pair_scores=best,orientations=orient)
 rec={'selection_locked_before_review':True,'top8':top,'all_unordered_pairs':len(rows),'pair_score_quantiles':dict(zip(['min','median','q90','q99','max'],np.quantile([r['similarity'] for r in rows],[0,.5,.9,.99,1]).tolist())),'feature_and_ranking_seconds':time.perf_counter()-start,'score_seconds':scoreseconds,'image_sha256':imagehash,'ranking_sha256':sha(_repo_path(OUT, 'ranking.npz')),'plan_sha256':sha(_repo_path(OUT, 'plan.json')),'labels_accessed':False};write(_repo_path(OUT, 'selection.json'),rec);check();emit({k:v for k,v in rec.items() if k not in ['image_sha256']})

def contrast(a):
 a=np.asarray(a,float);lo,hi=np.quantile(a,[.02,.98]);return np.uint8(np.clip((a-lo)/max(hi-lo,1e-9),0,1)*255)
def panel():
 check();selection=read(_repo_path(OUT, 'selection.json'));font=ImageFont.load_default(size=14);files={}
 for r in selection['top8']:
  A=raw(r['ID_A']);B=transform(raw(r['ID_B']),r['D4_B']);ea=edge(A);eb=edge(B);ea=np.clip(ea/max(float(np.quantile(ea,.95)),1e-12),0,1);eb=np.clip(eb/max(float(np.quantile(eb,.95)),1e-12),0,1);over=np.uint8(np.stack([eb,ea,ea],2)*255);ac=contrast(A);bc=contrast(B);blend=np.uint8(np.stack([bc,ac,ac],2));canvas=Image.new('RGB',(768,602),'white');draw=ImageDraw.Draw(canvas);draw.text((5,4),f"Rank {r['rank']} | {r['ID_A']} vs {r['ID_B']} | score {r['similarity']:.4f} | B rot={90*(r['D4_B']//2)},flip={r['D4_B']%2}",font=font,fill='black')
  tiles=[(A,'A raw native256'),(B,'B raw aligned native256'),(over,'Edges: A cyan / B red / overlap white'),(ac,'A image-local contrast q2..q98'),(bc,'B image-local contrast q2..q98'),(blend,'Intensity overlay A cyan / B red')]
  for i,(arr,title) in enumerate(tiles):
   x=(i%3)*256;y=28+(i//3)*286;draw.text((x+3,y+3),title,font=font,fill='black');canvas.paste(Image.fromarray(arr).convert('RGB'),(x,y+26))
  path=_repo_path(OUT, 'panels', f'pair-{r["rank"]:02d}.png');canvas.save(_repo_io(path));files[str(path.relative_to(ROOT))]=sha(path)
 write(_repo_path(OUT, 'panels.json'),{'files':files,'selection_sha256':sha(_repo_path(OUT, 'selection.json')),'view':'Native256pixels;onlyD4fixedbyranking. Contrastnormalizationdisplayonlynotclassification.'});check();emit({'panels_complete':8,'labels_accessed':False})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run','panels']);args=ap.parse_args();{'prepare':prepare,'run':run,'panels':panel}[args.mode]()
