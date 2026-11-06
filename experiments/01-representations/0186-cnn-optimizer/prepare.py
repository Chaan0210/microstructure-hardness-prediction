from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json,hashlib
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3];B=_repo_path(P, 'experiments/pretrained-cnn-185')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((_repo_path(B, 'plan.json')).read_text());protection=json.loads((_repo_path(B, 'preservation.json')).read_text())
for n,h in protection.items():assert sha(_repo_path(P, n))==h,n
for name in ['plan.json','summary.json','completion.json','train.py','common.py','provenance.json','requirements.lock']:
 p=_repo_path(B, name);protection[str(p.relative_to(P))]=sha(p)
for p in B.glob('*/original/**/*.pt'):protection[str(p.relative_to(P))]=sha(p)
for n in old['models']:
 for name in ['prefix.npy','features.npz','extraction.json','original/validation.json']:
  f=_repo_path(B, n, name);protection[str(f.relative_to(P))]=sha(f)
with np.load(_repo_path(P, 'experiments/dinol-177/original/predictions.npz')) as z:d={k:z[k].copy() for k in z.files}
folds={}
for k in range(5):
 T=np.flatnonzero(d['fold']!=k);V=np.flatnonzero(d['fold']==k);inner=[]
 for j,(a,b) in enumerate(GroupKFold(3,shuffle=True,random_state=202710186+k).split(T,groups=d['g'][T])):
  I,H=T[a],T[b];assert not set(d['g'][I])&set(d['g'][H]);assert not (set(I)|set(H))&set(V);inner.append({'train':I.tolist(),'validation':H.tolist(),'updates':12*((len(I)+15)//16)})
 folds[str(k)]={'train':T.tolist(),'validation':V.tolist(),'inner':inner,'outer_updates':240}
recipes=[{'name':'adamw_low','optimizer':'AdamW','tail_lr':1e-5,'head_lr':1e-4},{'name':'adamw_high','optimizer':'AdamW','tail_lr':1e-4,'head_lr':1e-3},{'name':'sgd_low','optimizer':'SGD','tail_lr':1e-3,'head_lr':1e-2},{'name':'sgd_high','optimizer':'SGD','tail_lr':1e-2,'head_lr':1e-1}]
plan={'study':186,'authorization':'Parent explicitly authorizes bounded optimizer/LR tuning bothCNN, train-onlyinnerselectionandouterOOF; existingenvweights only, cost0, noholdout/test/download.','IDs':old['IDs'],'models':old['models'],'input_sha256':old['input_sha256'],'folds':folds,'recipes':recipes,'fairness':'Exactlysame4recipes,3innerfolds,epochs,updates,seed,stage,batch,BNfixed,scheduler,WD,activation and data forbothmodels. ResNet185standaloneadaptationgainwaslarger; this motivated investigation but NO extraResNetbudget orposthocarm pruning. Architecturehasdifferentparams,FLOPsandruntime; equalupdatesnotidenticalcompute.','selection':'For each outertrain320: train4recipeswithin3groupinnerfolds; pooledinnerOOF MSE minimized, deterministicfirstrecipebreaktie; fixedfinal12epoch, nocheckpointselection. Alsoreportoptimizer-familybestLR selectedinternally. Retrainselectedconfigurationonouter320 fromthatfold185frozen100head. Evaluateouter80onlyafterselection/refit. Noouter-labeltuning.','warmstart':'Everyinnerfoldgetsnew100epochfrozenhead+scalersfitoninnertrainonly; neverreuseouterheadinsideinnerCV. Sharedinnerheadinitacross4recipes. Outerfull320frozen100headfrom185 legallyreused unchanged.','epochs':12,'effective_batch':16,'microbatch':8,'batch_note':'Microbatch8 reducesdispatch overhead versus185micro2; sameeffective16andmatchingnewrandomstreams acrossall4recipes. Require6GBmonitor andfinitefirstfit beforefullschedule.','seed':'head202710185+outer; adaptation202710186+100*outer+inner; outerinnercode99. IdenticalRNGorders/dropoutseeds acrossrecipes.','optimizer':'AdamW betas(.9,.999),eps1e-8; SGDmomentum.9,dampening0,nesterovFalse. Bothoptimizerinstancesweight_decay0; explicitshareddecoupleddecay .01 appliedwith eachgroupcurrentLR immediatelybeforeoptimizerstep. ThismatchesAdamWdecay semantics andgivesSGDsameWDmechanism ratherthanconfoundingSGDL2. UnitcompareexplicitdecayAdamWvsnativeAdamW.','weight_decay':.01,'scheduler':'same12epochcosinefactor .1+.9*(1+cos(pi*epoch/12))/2; actualcurrentLR perupdate recorded; epoch0factor1. No schedulersearch.','loss':'SmoothL1beta1 normalizedtarget; globalgradnormclip1','model_policy':old['image_policy'],'trainable':old['fine_tune']['trainable'],'BN':'allBNaffine/runningstatsfrozen; entirebackboneeval; headtrain dropout.25/GELU64. Prefixcachedfrozen officialweights.','diagnostics':'Logeachupdateunclippedgradnorm,total/head/tailnorms,LR,BNmode/nontrainablegrads. TrainRMSE/normalizedlossat0/3/6/9/12. Innerheldoutpredictionsatfixedendonly; allsavedmodelsreload. Diagnoseundertrainingfromtraincurves/gradients,notarchitectureclaims.','audit':'For eachmodel repeatoneinnerfitwithitsvalidationlabels altered (head/scalersfittrainonly) andoneentireouterselectedrefitwithouterlabels altered. Innerselectionmustnotchangeunderouterlabelpoison; states/predictions exactreplay; selectionmay legitimately change whenINNERlabelschange. IndependentCPUreloadallinner/outercheckpoints andtrainstats.','evaluation':'SelectedprocedureOOF standaloneandfixed.9best021+.1selectedCNN; reportfourrecipeinnerresults andfamilyselection, noouterwinnerselection. Same185advancegate:gain>=.02,>=3positivefolds,conditionalclusterbootstraplower>0,leave5largestgainpositive. Noresplit/full500/testinthis186untilresultreview.','resources':{'CPU_threads':2,'max_RSS_plus_MPS':6000000000,'swap_growth':0,'max_basic_seconds_per_model':2700,'max_validation_and_poison_seconds_per_model':900,'firstfit_projected_gate':'firstinnerfit compute seconds *60 plus5estimatedouterfits +headwarmstarts <2700/model; otherwise stopwithoutchangingepochs/LR torescue. Equal4x3x5trials+5outerrefits bothmodels.'},'history_undertraining':{},'no_download':True,'cost':0}
for n in old['models']:
 plan['history_undertraining'][n]=[{a:json.loads((_repo_path(B, n, 'original', f'fold-{k}', f'{a}-report.json')).read_text())['train_RMSE'] for a in ['frozen100','finetune112']} for k in range(5)]
for name,obj in [('plan.json',plan),('preservation.json',protection)]:
 with (_repo_path(R, name)).open('x') as f:json.dump(obj,f,indent=2)
for name in ['cache','config','tmp']:(_repo_path(R, name)).mkdir()
print(json.dumps({'plan_sha256':sha(_repo_path(R, 'plan.json')),'protected_count':len(protection),'inner_rows':[(len(x['train']),len(x['validation']),x['updates']) for x in folds['0']['inner']]}))
