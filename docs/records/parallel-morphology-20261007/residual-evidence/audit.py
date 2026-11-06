"""No-training visual evidence audit: preselected dev images only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import json,hashlib,sys,time
from pathlib import Path
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent;P=R.parents[3]
assert Path(sys.prefix).resolve()==(_repo_path(P, '.venv')).resolve()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):
 with (_repo_path(R, n)).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2)
sources=['reports/denoised-error-analysis-024.json','reports/diagnostic-research-068-072-summary.json','reports/image-diagnostics-094-assessment.json','reports/image-diagnostics-094.json','reports/domain-risk-audit-126-summary.json','reports/failure-review-166-167-summary.json','reports/submission022-diagnostic-199-summary.json','docs/records/diagnostics/l1-l4-20261007/summary.json','docs/records/diagnostics/l1-l4-20261007/REPORT.txt','experiments/grain-validity-218/REPORT.txt','parallel-morphology-20261007/summary.json']
m=read(_repo_path(P, 'artifacts/run-001/split_manifest.json'));ids=sorted(m['development_folds']);assert len(ids)==400
reps=sorted(sorted(ids,key=lambda i:hashlib.sha256(('residual-evidence-20261007-labelblind:'+i).encode()).hexdigest())[:12])
# All nine existing166 cases tagged persistent large categories, no new error ranking.
old=read(_repo_path(P, 'reports/failure-review-166-167-summary.json'))
cases=[r['ID'] for r in old['visual_case_changes'] if r['category'] in ['large_stable_disagreement','large_stable_without_disagreement']]
assert len(cases)==9 and len(set(cases))==9 and set(cases)<=set(ids)
p={'selection_locked_before_image_reads':True,'representative_rule':'12 smallest SHA256(residual-evidence-20261007-labelblind:+ID) among development400; lexical display. No target/errors/features used. Convenience audit sample, not statistical representativeness guarantee.','representatives':reps,'historical_large_residual_cases':cases,'case_rule':'All9 preexisting166 records with large_stable_disagreement/large_stable_without_disagreement category; includes old018 cases, not newly ranked021 tail. Outcome-selected and never mixed with label-blind sample.','overlap':sorted(set(reps)&set(cases)),'visual_protocol':'Original native256 grayscale, fixed0..255 appearance; ID and sampling group only, no target/error overlay, no enhancement, no segmentation. Record visible differences without mapping brightness to true phases or diagnosing label errors. Reviewer has read prior case histories, so this is not blinded visual grading.','question':'Any concrete previously untested visible failure mechanism that can justify <=2 minimal future diagnostics? Zero admissible. No new residual statistics, image matching, segmentation, inverse rendering, fitting or inference.','sources_sha256':{n:sha(_repo_path(P, n)) for n in sources},'code_sha256':sha(Path(__file__)),'scope':'All writes within residual-evidence; existingfiles read-only; only development source pixels. No external network/install/transfers or test/holdout/full500 data.'}
write('selection.json',p)
meta=[]
for name,group,cols in [('representatives',reps,4),('historical-cases',cases,3)]:
 rows=(len(group)+cols-1)//cols;canvas=Image.new('RGB',(cols*272,rows*294),(245,245,245));draw=ImageDraw.Draw(canvas)
 for k,ident in enumerate(group):
  path=_repo_path(P, 'data/train', f'{ident}.png')
  with Image.open(_repo_io(path)) as im:
   assert im.mode=='L' and im.size==(256,256)
   x=(k%cols)*272+8;y=(k//cols)*294+30;canvas.paste(im.convert('RGB'),(x,y));draw.text((x,y-21),ident,fill=(0,0,0))
  meta.append({'group':name,'ID':ident,'source_sha256':sha(path),'size':[256,256],'mode':'L'})
 canvas.save(_repo_path(R, f'{name}.png'))
write('image-provenance.json',meta)
print(json.dumps(p,ensure_ascii=False,indent=2))
