from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import ast,json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
P=_RepoPath(__file__).resolve().parents[3];R=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2)
old=read(_repo_path(P, 'experiments/dinol-177/plan.json'));prior=read(_repo_path(P, 'experiments/candidate021-178/plan.json'))
inputs=['artifacts/run-022/development_features.npz','artifacts/run-022/report.json','artifacts/spatial-joint-140/features.npz','artifacts/spatial-joint-140/extraction.json','artifacts/patch-attention-092/cache.json','experiments/dinol-177/features.npz','experiments/dinol-177/extraction.json','experiments/dinol-177/plan.json','experiments/dinol-177/provenance.json','reports/nested-selector-146-audit.json','reports/spatial-joint-142-audit.json','reports/dinol-177-summary.json','reports/spatial-support-154-summary.json','reports/spatial-token-cnn-161-summary.json','bootstrap_selector_144.py','spatial_joint_140.py','experiments/reproduction-018-176/feature_helpers.py']
for split in ['original','20261031','20261101']:
 for root in ['artifacts/spatial-joint-140','artifacts/bootstrap-selector-144']:
  inputs += [f'{root}/validation-{split}.json',f'{root}/{split}/report.json',f'{root}/{split}/predictions.npz']
 inputs += [f'experiments/dinol-177/{split}/{n}' for n in ['predictions.npz','report.json','validation.json']]
 for k in range(5):
  inputs += [f'artifacts/spatial-joint-140/{split}/adjacent-fold-{k}/{n}' for n in ['report.json','predictions.npz']]
  for j in range(3):
   for root in [f'artifacts/spatial-joint-140/{split}/adjacent-fold-{k}/block-{j}',f'artifacts/bootstrap-selector-144/{split}/fold-{k}-block-{j}']:
    inputs += [f'{root}/{n}' for n in ['report.json','predictions.npz']]
protected={**prior['protected'],'artifacts/submission-candidate-021.csv':sha(_repo_path(P, 'artifacts/submission-candidate-021.csv')),'artifacts/submission-candidate-021.manifest.json':sha(_repo_path(P, 'artifacts/submission-candidate-021.manifest.json')),'reports/candidate021-178-summary.json':sha(_repo_path(P, 'reports/candidate021-178-summary.json'))}
for n,h in protected.items():assert sha(_repo_path(P, n))==h,n
plan={'study':179,'UTC':datetime.now(timezone.utc).isoformat(),'IDs':old['IDs'],'groups':old['groups'],'smoke_IDs':old['smoke_IDs'],'scope':'development400 only; raw400image whitelist; noholdout/test/full500/CSV/upload/install/download/cost','hypothesis':'Replace B spatial neighbor co-change statistics with L statistics, holding all dimensionality, physics, upstream anchors, residual training and globalL021 term fixed. No concatenation arm.','duplicate_audit':'140 introduced B neighbor graph energy;154 changed spatial clipping;161 learned B spatial CNN.177/178 stored L CLS/mean/std only, no ordered tokens. No existing L spatial extraction found. Global summaries discard explicit neighboring arrangement, though contextual tokens can implicitly encode spatial content.','projection':'Gaussian QR with seed20271201, B768x32/L1024x32, no fitted PCA or labels. Same procedure/rank, different coordinate bases due different channel widths.','statistic':'Normalized final layer tokens36x36, projected32. Half mean outerproduct over2520 horizontal/vertical nearest-neighbor differences, no wrap; uppertriangle528. Same definition as018.','recipe':'Replace only B528 in physics57+B528 by L528. No concatenation/addition, layer/rank/seed/grid changes. MinMaxclip andStandardScaler trainonly; block1/sqrt(dim). Existing none/ridge1/ridge01/rbf1/rbf01, lambda*n, trainlocal residualRMS andRBF distance/kernel centering. Each honest block inner3fold;512groupbootstrap selectionfrequency seed20271204+10000*splitindex+100*fold+block,mean3honestblocks.','primary':'021_swap = .9*(unchanged015base + newly fitted L residual mean3) + .1*fixed177 LglobalMLP. Control021=.9*018+.1*sameLglobal. Also report018-equivalent swap as attribution only, never fallback candidate.','anchors':'Reuse verified140 A-only honest anchors and015base; no newglobalOOFresidual fitting. Refresh ALL B andL inner/full residualfits and bootstrap weights; B must exactly reproduce018 and021. No need refit unchanged upstream neural learners.','gate':'After original5fold, expand both existingresplits only if primary strictly improves BOTH fixed021 and018 byRMSE, at least3/5folds improve021, allvalidationpasses. If gatefails stop. Afterresplits proposeonlyif primary improvesbothreferences inall3partitions; no rescue blend.','diagnostics':'OOF residual correlations primary/control021/018/globalL and correctiondifference vs globalLdelta; centered linearR2 descriptive only, no fitted fusion. Pairedgroupbootstrap3000 fixedseed202710179, conditional fixedfits andadaptive reused400; not independentvalidation.','validation':'Bcontrol exact allinner/fullpredictions/weights; all600 modelobjects perpartition reload+independent rawscaling/dualridge/eigenkernel prediction; exactgroupbootstrap independent aggregation; index/group scopes; outerfold0 all3Lblock poisonreplays; oneinner heldout poisonperblock; upstream exclusioncertificates reused.','resources':old['resources'],'tolerances':{'independent_prediction':2e-9,'CPU_MPS_spatial_feature':.002,'global_feature_exact':True},'inputs':{n:sha(_repo_path(P, n)) for n in inputs},'protected':protected,'official_code':old['official_code'],'no_new_model_selection_axes':True}
write(_repo_path(R, 'plan.json'),plan)
# Reuse178 runtime functions; replace only study check with new locked plan.
src=(_repo_path(P, 'experiments/candidate021-178/runtime.py')).read_text();tree=ast.parse(src);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check');lines=src.splitlines(True);lines[node.lineno-1:node.end_lineno]=['def check():\n p=read(R/"plan.json")\n for name,h in read(R/"code-lock.json")["sources"].items():assert sha(R/name)==h,name\n for name,h in {**p["inputs"],**p["protected"]}.items():assert sha(P/name)==h,name\n for name,h in p["official_code"].items():assert sha(V/name)==h,name\n return p\n'];(_repo_path(R, 'runtime.py')).write_text(''.join(lines))
for n in ['residual_models.py','residual_linear.py']:shutil.copyfile(_repo_path(P, 'experiments/reproduction-018-176', n),_repo_path(R, n))
write(_repo_path(R, 'setup-provenance.json'),{'runtime_from':sha(_repo_path(P, 'experiments/candidate021-178/runtime.py')),'residual_modules':{n:sha(_repo_path(R, n)) for n in ['residual_models.py','residual_linear.py']}})
print(json.dumps({'plan_locked':True,'plan_SHA256':sha(_repo_path(R, 'plan.json')),'inputs':len(inputs),'protected':len(protected)}))
