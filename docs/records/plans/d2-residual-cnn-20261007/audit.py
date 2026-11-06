"""Read-only model/cache membership analysis. No torch import, inference or fitting."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,hashlib,datetime,math
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.model_selection import GroupKFold
P=Path(__file__).resolve().parents[4];R=Path(__file__).resolve().parent;assert Path(sys.prefix).resolve()==_repo_path(P, '.venv')
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,q):
 with p.open('x') as f:json.dump(q,f,ensure_ascii=False,indent=2,allow_nan=False)
def load(p,keys):
 with np.load(_repo_io(p),allow_pickle=False) as z:return {k:z[k].copy() for k in keys}
p=read(_repo_path(P, 'experiments/cnn-nested-204/plan.json'));inputs=p['inputs_sha256'];checks={};protected={}
def verify(path,expected=None):
 q=sha(_repo_path(P, path));expected=inputs.get(path) if expected is None else expected
 if expected is not None:assert q==expected,path
 checks[path]={'sha256':q,'matches_prior_hash':expected is not None};return _repo_path(P, path)
for f in ['artifacts/submission-candidate-021.csv','experiments/candidate021-178/models/manifest.json','experiments/grain-validity-218/completion.json']:
 protected[f]=sha(_repo_path(P, f))
for n,h in read(_repo_path(P, 'experiments/candidate021-178/models/manifest.json'))['files'].items():f='experiments/candidate021-178/models/'+n;assert sha(_repo_path(P, f))==h;protected[f]=h
z=load(verify('experiments/cnn-nested-204/cache/data.npz',p['data_sha256']),['ids','groups','fold']);assert len(z['ids'])==len(set(z['ids']))==400
cells=[];newnodes=[];oldsets={tuple(n['train']) for n in p['nodes']};originalnodes=[n for n in p['nodes'] if n['split']=='original']
for k in range(5):
 T=np.array(p['folds']['original'][str(k)]['train']);V=np.array(p['folds']['original'][str(k)]['validation']);node=next(n for n in originalnodes if n['outer']==k and n['middle']==0 and n['type']=='middle');A=np.array(node['train']);B=np.array(node['validation']);assert set(A)|set(B)==set(T) and not set(A)&set(B) and not (set(A)|set(B))&set(V)
 for n in [x for x in originalnodes if x['outer']==k and x['middle']==0]:
  q=load(verify('experiments/cnn-nested-204/cache/'+n['name']+'.npz',n['cache_sha256']),['train','validation']);np.testing.assert_array_equal(q['train'],n['train']);np.testing.assert_array_equal(q['validation'],n['validation']);assert set(q['validation']).isdisjoint(V)
  f=n['source']+'/report.json';rep=read(verify(f))
  for name,h in rep['files_sha256'].items():
   if name in ['kernel.joblib','predictions.npz'] or name.startswith(('S-','B-')):verify(n['source']+'/'+name,h)
  rr=read(verify(n['old3_source']+'/report.json'))
  for m in rr['models']:verify(n['old3_source']+f"/seed-{m['base_seed']}.pt",m['checkpoint_sha256'])
  if n['type']=='deep':
   C=np.array(n['train']);D=np.array(n['validation']);assert set(C)|set(D)==set(A) and not(set(C)|set(D))&(set(B)|set(V));deep_splits=GroupKFold(3,shuffle=True,random_state=20270110+n['offset']).split(np.zeros((len(C),1)),groups=z['groups'][C])
   for j,(ei,fi) in enumerate(deep_splits):
    E,F=C[ei],C[fi];assert not set(E)&set(F) and set(E)|set(F)==set(C) and not (set(E)|set(F))&(set(B)|set(D)|set(V));assert tuple(E) not in oldsets
    newnodes.append({'outer':k,'anchor_cell':0,'middle_inside_anchor':n['deep'],'new_deep':j,'train':E.tolist(),'query':F.tolist(),'train_n':len(E),'query_n':len(F),'offset':n['offset']+100*(j+1),'cached_identical_training_set_in204':False})
 short=f'artifacts/nested-fusion-112/original/fold-{k}/inner-0/shortB';sr=read(verify(short+'/report.json'));verify(short+'/model.pt',sr['model_sha256']);verify(short+'/predictions.npz',sr['predictions_sha256'])
 lr=read(verify(f'experiments/dinol-177/original/fold-{k}/report.json'));assert lr['T']==T.tolist();assert set(lr['T'])!=set(A)
 cells.append({'outer':k,'T':T.tolist(),'V':V.tolist(),'A':A.tolist(),'B':B.tolist(),'root_offset':node['offset'],'cached_root_and3middle_component_nodes':4,'full021_anchor_exists_in204':False,'rootA_prediction_cache_covers_B_not_V':True,'L_cached_head_train_n':len(lr['T']),'needed_L_head_train_n':len(A),'models_need_requery_on_V':True})
train=read(_repo_path(P, 'experiments/cnn-nested-204/training-complete.json'));recs=[x for x in train['reports'] if x['phase']=='main'];rate=np.array([r['seconds']/r['updates'] for r in recs]);deep=[r for r in recs if r['type']=='deep'];tq=np.quantile([r['seconds']/r['updates'] for r in deep],[.1,.5,.9]);newanchorcnn=3*len(newnodes);rescnn=2*len(cells);mainupdates=3*sum(80*math.ceil(n['train_n']/32) for n in newnodes)+2*sum(80*math.ceil(len(c['B'])/32) for c in cells);first=[n for n in newnodes if n['outer']==0];verifyupdates=3*sum(80*math.ceil(n['train_n']/32) for n in first)+2*80*math.ceil(len(cells[0]['B'])/32)+3*80*math.ceil(first[0]['train_n']/32)+2*80*math.ceil(len(cells[0]['B'])/32)
sourcefiles=['experiments/cnn-nested-204/train.py','experiments/cnn-nested-204/pipeline.py','experiments/cnn-nested-204/prepare.py','experiments/reproduction-018-176/train.py','experiments/reproduction-018-176/infer.py','experiments/candidate021-178/infer.py','artifacts/residual-physics-113/plan.json','artifacts/nonlinear-residual-115/plan.json','docs/records/diagnostics/l1-l4-20261007/summary.json']
for f in sourcefiles:verify(f)
result={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Read/hash/cachemembership/GroupKFoldindex construction only. No model inference, training, newrawimages, holdout/test or external calls.','D2_is_duplicate_of204':False,'duplicate_evidence':'204train.py passes original z[y] to fv.cnn_fold; adds3seed CNNsatalllevels. Residualtargets in204pipeline.py feed five classical residual_models, not TinyCNN.113/115are linear/RBF residualfeatures,not rawimage residualCNN. Boundedcodeaudit,not universalnoveltyclaim.','best021_formula':'.9*(.9*affine(raw)+.1*shortB+mean3residualcorrection)+.1*Lmean3;raw=.3KRR+.3CNNmean3+.2Smean3+.2Bmean3','cached_middle_anchor_formula':'.9*affine_A(raw_A)+.1*shortB_A; no full018residualcorrection_A and noL_A','exact_unchanged_production_anchor_limitation':'Already fitted500 best021 contains labels ofall400. Cannot use it to generate honest outerheldoutpredictions or anchor-excluded residualtraininglabels. Fullrecipe anchors fittedonA are different weights/sample sizes from021_T or existingproduction021500.','safe_paired_alternative':'Use complete021-family G_A trained onlyA, residualCNN_B trainedyB-G_A(B), evaluateG_A(V)+CNN_B(V). Zerobranch andfixed-permutationbranch sameG_A. Futuredeployment repeats sameA/Bsplitpolicy; cannot substitute existing021500 forG_A. This changesanchortrainingallocation and isnot exactdrop-in correction tobest021.','deployment_selection_detail':'176train.node applies specialKRR selection400/500 onlyatfull500 root. Allsubnodes useallU. ProposedpairedG_A hasn213 inCV andfuture333/334, usesallU nestedselection consistently. Labelled021-family,not aclaimthatlegacyfull500 trainingprocedureisidentical.','minimal_plan_cells':cells,'new_deep_nodes':newnodes,'cached204_train_sizes':dict(Counter(len(n['train']) for n in p['nodes'])),'missing_cache_types':['95-ish-rowdeeperCNN/S/B/KRRforcomplete018_A','shortB_C at142-ishnodes','L_A heads trained213','full018_A residualmodels andcalibrators','cachedAtrainedcomponent predictions onouterV; savedmodelspermitfutureinference but thissessiondoesnotrunit'],'counts_original5fold':{'new_anchor_CNN':newanchorcnn,'direct_residual_CNN':len(cells),'permuted_residual_control_CNN':len(cells),'total_main_CNN':newanchorcnn+rescnn,'new_long_S_B_MLP':6*len(newnodes),'new_L_MLP':3*len(cells),'new_shortB_MLP':3*len(cells),'new_KRR_final':len(newnodes),'new_KRR_inner':27*len(newnodes),'affine_OLS':4*len(cells),'nonnull_residual_classical_fits':48*len(cells),'zeroresidual_objects_no_fit':12*len(cells),'CNN_verification':34,'total_CNN_main_plus_verification':newanchorcnn+rescnn+34,'discarded_probe_updates':8,'main_CNN_updates':mainupdates,'verification_CNN_updates':verifyupdates,'reused_base_component_nodes':4*len(cells)},'verification_definition':'Firstanchor complete complement(B+V)poison replay27newCNN; bothdirectresidualheadsouterVpoison2; representative deepest3seedcleanrepeat3; bothresidualheadscleanrepeat2=34CNN. CPUheads/KRR/residuals firstanchorpoison replay too; cacheonlypaths useexistingverified hashes and membership certificates. Freshvalidation no invisibleextra fits.','timing_evidence':{'204_main_MPS_fits':len(recs),'204_main_plus48validation_elapsed_seconds':train['elapsed_seconds'],'204_deep_sec_per_update_p10_p50_p90':tq.tolist(),'extrapolated_CNN_main_and_verification_seconds_p10_p50_p90':((mainupdates+verifyupdates)*tq).tolist(),'recommended_total_budget_minutes':[15,25],'forecast_not_benchmark':'Smaller94-95trainingcount and changedtargets not yetbenchmarked; CPUheads,oldmodelrequery/I/O andverification overheadaddedroughly. No assurance Maccurrentloadmatches204.','historical204_peak_RSS_plus_MPS_bytes':read(_repo_path(P, 'reports/cnn-nested-204-summary.json'))['resources']['study-resources.json']['max_RSS_plus_MPS_driver_bytes'],'proposed_cap_bytes':6000000000,'probe_only_after_training_authorization':True},'cache_evidence':checks,'protected':protected,'neural_fits_this_audit':0,'classical_fits_this_audit':0,'inference_this_audit':0,'remaining_work_before_anyfit':['ImplementgeneralizedsafeG_Atrainer withfrozenmembership/seedmappingandallfeaturesfromdevcaches; noexistingtrainingmoduleimportsideeffects','VerifyfrozenS/B/Lfeaturehashes andexactpreprocessing/dimensions; no needextractbackbonefeatures','Freeze fullchosenplan/code, audit complete label-flow and cachetrainIDs; implement guards/poison tests','Explicit go/no-godecision onnewpairedanchorpolicy and145+34CNNcost; currentturndoesnotauthorizeexecution'],'evidence_strength':'L1-L4showerrorconcentration but signedlocalresidualcorrnegative~-.10 and218failed. No verified rawimage residualpredictability gain. D2 ismechanisticallydifferent but speculative; nolargegainorprizeclaim.'}
write(_repo_path(R, 'feasibility.json'),result)
for f,h in protected.items():assert sha(_repo_path(P, f))==h
print(json.dumps({'cache_files_verified':len(checks),'new_deep_nodes':len(newnodes),'counts':result['counts_original5fold'],'time':result['timing_evidence'],'newfits':0},ensure_ascii=False))
