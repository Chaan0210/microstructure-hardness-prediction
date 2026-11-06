from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import json,hashlib
from pathlib import Path
from PIL import Image
R=Path(__file__).resolve().parent;P=R.parents[3]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):
 with (_repo_path(R, n)).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2)
p=read(_repo_path(R, 'selection.json'))
notes={
'TRAIN_000008':'Clear oblique polygon boundaries; broad regions of differing grayscale; dark round and thin spots.',
'TRAIN_000220':'Dense small polygon mesh, mostly vertical elongation and dark spots; visible fine-grained texture.',
'TRAIN_000244':'Strong fine-grained texture; underlying boundaries weak, some dark spots visible. No inference about hardness.',
'TRAIN_000254':'Clear elongated oblique polygons; moderate gray variation, sparse dark spots.',
'TRAIN_000256':'Blurred broad oblique brightness regions with fine texture and dark spots.',
'TRAIN_000280':'Broad oblique polygons with clear outlines and scattered darker polygons/spots.',
'TRAIN_000416':'Weak blurred oblique region structure under fine texture; dark spots remain visible.',
'TRAIN_000423':'Dense narrow oblique polygons, sharp borders, heterogeneous region shades and dark spots.',
'TRAIN_000427':'Medium-scale mixed polygon sizes; softened boundaries and variable region grayscale.',
'TRAIN_000439':'Large vertically elongated polygons with clear borders and dark filled regions/spots.',
'TRAIN_000466':'Wide oblique elongated regions, distinct grayscale patches, fairly clear boundaries.',
'TRAIN_000493':'Narrow oblique polygons with clear outlines and dark spots.',
'TRAIN_000150':'Broad vertical elongated brightness regions, strong fine texture and weak outlines.',
'TRAIN_000331':'Broad vertically elongated gray regions under fine texture; local dark filled regions visible.',
'TRAIN_000491':'Clear polygon outlines and dark spots, relatively weak fine texture. Historical high error includes visually clear cases.',
'TRAIN_000175':'Large oblique brightness domains under strong fine texture; outlines weak.',
'TRAIN_000227':'Broad vertical elongated regions, visibly varying gray levels, weak/soft outlines and dark spots.',
'TRAIN_000467':'Very broad weak brightness variations under fine texture; few dark spots.',
'TRAIN_000206':'Broad faint oblique region outlines and smooth brightness differences; few isolated dark spots.',
'TRAIN_000120':'Broad oblique region differences with mixed sharp/soft outlines and fine texture.',
'TRAIN_000475':'Oblique elongated regions with moderate grayscale variation and multiple dark spots.'}
for f,h in p['sources_sha256'].items():assert sha(_repo_path(P, f))==h,f
assert sha(_repo_path(R, 'audit.py'))==p['code_sha256']
prov=read(_repo_path(R, 'image-provenance.json'));seen=[]
for name,ids,cols in [('representatives',p['representatives'],4),('historical-cases',p['historical_large_residual_cases'],3)]:
 with Image.open(_repo_path(R, f'{name}.png')) as panel:
  for i,ident in enumerate(ids):
   src=_repo_path(P, 'data/train', f'{ident}.png');entry=next(r for r in prov if r['ID']==ident and r['group']==name);assert sha(src)==entry['source_sha256']
   x=(i%cols)*272+8;y=(i//cols)*294+30
   with Image.open(_repo_io(src)) as im:assert panel.crop((x,y,x+256,y+256)).tobytes()==im.convert('RGB').tobytes()
   seen.append(ident)
assert len(seen)==len(set(seen))==21 and set(notes)==set(seen)
write('visual-review.json',{'review_completed':True,'presentation':'Original native pixels; no perimage contrast changes, resize, segmentation or label/error overlay. Not blinded: prior written histories read first.','observations':[{'ID':i,'sample_group':'label_blind12' if i in p['representatives'] else 'historical_outcome_selected9','observation':notes[i]} for i in seen],'inference':'Visible breadth of quality/region size/elongation exists in both sets; no causal/quantitative enrichment claim from21. Historical9 are018 error cases, not freshly ranked021 failure set. No label-error or generator inference.'})
write('verification.json',{'passed':True,'development_unique_images':21,'representative_count':12,'historical_count':9,'overlap':0,'all_panel_pixel_crops_exact':True,'all_source_hashes_unchanged':True,'image_hashes_unchanged':True,'new_model_fits':0,'new_model_inferences':0,'new_segmentations':0,'external_data_or_network':False,'test_holdout_full500_data_read':False,'write_scope':str(R),'note':'199 read only its historical summary as requested; no underlying full500/test files or statistics recomputed.'})
write('conclusion.json',{'decision':'No newly justified executable structure/residual hypothesis in this bounded audit. Preserve best021.','new_minimal_diagnostics_proposed':[],'hypothesis_count':0,'evidence':['024 noise proxy correlates with segmentation sensitivity but cannot identify causal/irreducible noise.','068 signed residual correlations were weak across20 proxies; raw brightness is not globally erased in existing inputs.','094 matched15 and coarse-pattern95 did not reproduce benefit of outcome-selected5.','126 outertrain-selected image condition did not transfer reliable relative risk.','166/167 similar appearances can have opposite residuals and heterogeneity interaction did not beat matched quadratic control.','199 tiny gains concentrate in few observations/label regions; its published historical test/full500 discussion is not reused as current diagnosis data.','L2/L3 magnitude clusters but signed neighbor residual correlations ~-.10; no smooth correction mechanism established.','L4 measurement validity concern is real, but follow-up218 explicitly failed all3 partitions.','Previous HH6 fixed comparison failed all3 partitions.','Current native21-panel review reveals no distinct new morphology defect separate from these examined families.'],'not_concluded':['No label error diagnosis.','No Voronoi/power-diagram generator identification.','No irreducible noise floor or information-theoretic limit.','No evidence that all future residual models must fail.','No population morphology enrichment from small historical case panel.'],'evidence_needed_to_reopen':'Verified image generation/target construction details, original latent/measurement truth, or permitted independent labeled evidence that discriminates a specific mechanism. Not requested or accessed here.','selection_caveat':'Development400 historically reused; case selection outcome based; reviewer knew histories; hypothesis generation retrospective. Any new feature mined from cases would require prospective validation and multiplicity/selection accounting.','outputs':['selection.json','representatives.png','historical-cases.png','visual-review.json','verification.json','REPORT.txt'],'blockers':'No execution blocker. Identifiability/evidence gap remains; compute/connection alone does not resolve it.'})
print('21 native pixel crops exact; sources unchanged; zero new fits/inferences/segmentations.')
