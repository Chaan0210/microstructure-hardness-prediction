from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='2'
import sys,json,hashlib,datetime,time
from pathlib import Path
import numpy as np
R=Path.cwd();O=_repo_path(R, 'experiments/cnn-nested-204');assert R==Path(__file__).resolve().parents[2] and Path(sys.prefix)==_repo_path(R, '.venv')
inputs={}; protected={};checks=[]
def H(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def track(rel,expected=None):
 p=_repo_path(R, rel);h=H(p)
 if expected is not None:assert h==expected,rel
 inputs[rel]=h;return p
def J(rel):return json.loads(track(rel).read_text())
def Z(rel,expected=None):
 with np.load(_repo_io(track(rel,expected)),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def W(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def arr(p,**kw):
 with p.open('xb') as f:np.savez_compressed(_repo_io(f),**kw)
def reported(folder,split):
 r=J(f'{folder}/{split}/report.json');return Z(f'{folder}/{split}/predictions.npz',r['predictions_sha256']),r
start=time.perf_counter();oldplan=J('experiments/cnn-seed-expand-203/plan.json')
for p,h in oldplan['protected_sha256'].items():
 # The registry/README were legitimately updated at completion203; preserve current snapshots.
 protected[p]=H(_repo_path(R, p))
for p in ['README.md','reports/strategy-experiment-register.csv','docs/records/reviews/structure-20261006/response.md','experiments/cnn-seed-expand-203/completion.json','reports/cnn-seed-expand-203-summary.json']:protected[p]=H(_repo_path(R, p))
for p in sorted(R.glob('*.py')):inputs[str(p.relative_to(R))]=H(p)
splits=['original','20261031','20261101'];p202=Z('experiments/simplification-202/predictions.npz');v203=J('experiments/cnn-seed-expand-203/validation.json');assert v203['passed'];p203=Z('experiments/cnn-seed-expand-203/ensemble-OOF.npz',v203['ensemble_sha256'])
for key in ['ids','y','fold','groups']:np.testing.assert_array_equal(p202[key],p203[key])
np.testing.assert_array_equal(p202['R'],p203['reference']);ids=p202['ids'];y=p202['y'];groups=p202['groups'];fold=p202['fold'];assert len(ids)==len(set(ids))==len(set(groups))==400
ex=J('artifacts/nested-calibration-074/extraction.json')
with np.load(_repo_io(track('artifacts/nested-calibration-074/development_features.npz',ex['feature_sha256'])),allow_pickle=False) as z:np.testing.assert_array_equal(z['ids'],ids);images=z['images'].copy()
assert images.shape==(400,1,128,128) and images.dtype==np.float32
phys=Z('artifacts/run-022/development_features.npz');np.testing.assert_array_equal(phys['ids'],ids);np.testing.assert_array_equal(phys['groups'],groups);blocks=J('artifacts/run-022/report.json')['blocks']
fx=J('artifacts/spatial-joint-140/extraction.json');sp=Z('artifacts/spatial-joint-140/features.npz',fx['features_sha256']);np.testing.assert_array_equal(sp['ids'],ids);X=np.column_stack([phys['X'].astype(float),sp['adjacent']]);blocks=blocks+['joint_token']*528;assert X.shape==(400,585)
outerP=[];nodes=[];folds={};original_data=[]
for si,split in enumerate(splits):
 cp,cr=reported('artifacts/component-ablation-121',split);lp,lr=reported('experiments/dinol-177',split);rp,rr=reported('artifacts/bootstrap-selector-144',split)
 for z in [cp,lp,rp]:
  for key,ref in [('ids',ids),('y',y),('fold',fold[si])]:np.testing.assert_array_equal(z[key],ref)
 np.testing.assert_array_equal(lp['blend_L_MLP'],p202['R'][si]);np.testing.assert_array_equal(lp['L_MLP'],p202['L_seeds'][si].mean(0));np.testing.assert_array_equal(lp['reference018'],rp['primary']);np.testing.assert_array_equal(cp['short'],p202['short'][si])
 outerP.append(cp['components']); folds[split]={}
 for folder in ['artifacts/cnn-seed-mean-133','artifacts/spatial-joint-140','artifacts/bootstrap-selector-144','artifacts/nested-fusion-112','artifacts/component-ablation-121']:
  vr=J(f'{folder}/validation-{split}.json');assert vr['passed']
  if 'report_sha256' in vr:track(f'{folder}/{split}/report.json',vr['report_sha256'])
 for k in range(5):
  T=np.flatnonzero(fold[si]!=k);V=np.flatnonzero(fold[si]==k);assert len(T)==320 and len(V)==80;folds[split][str(k)]={'train':T.tolist(),'validation':V.tolist()};covered=[]
  fr=J(f'artifacts/nested-fusion-112/{split}/fold-{k}/report.json')
  for j in range(3):
   cell=f'artifacts/nested-fusion-112/{split}/fold-{k}/inner-{j}';br=J(cell+'/report.json');assert inputs[cell+'/report.json']==fr['cell_report_sha256'][f'inner-{j}/report.json'];b=Z(cell+'/calibration.npz',br['calibration_sha256']);A=b['train_indices'];B=b['validation_indices'];assert set(A).isdisjoint(B) and set(A)|set(B)==set(T) and set(A).isdisjoint(V);covered.extend(B.tolist())
   sr=J(cell+'/shortB/report.json');assert inputs[cell+'/shortB/report.json']==br['model_reports']['shortB'];track(cell+'/shortB/model.pt',sr['model_sha256']);track(cell+'/shortB/predictions.npz',sr['predictions_sha256'])
   specs=[('middle',-1,f'artifacts/nested-calibration-074/{split}/fold-{k}/inner-{j}',f'artifacts/cnn-seed-mean-133/{split}/cnn-fold-{k}/middle-{j}',A,B,b['shortB'])]
   for ell in range(3):
    src=cell+f'/deep-{ell}';dr=J(src+'/report.json');assert inputs[src+'/report.json']==br['deep'][ell]['report_sha256'];C=np.array(dr['train_indices']);D=np.array(dr['validation_indices']);assert set(C).isdisjoint(D) and set(C)|set(D)==set(A) and (set(C)|set(D)).isdisjoint(set(B)|set(V));specs.append(('deep',ell,src,f'artifacts/cnn-seed-mean-133/{split}/cnn-fold-{k}/deep-{j}-{ell}',C,D,np.array([],float)))
   for typ,ell,src,old,Tn,Vn,short in specs:
    rec=J(src+'/report.json')
    for name,h in rec['files_sha256'].items():track(src+'/'+name,h)
    z=Z(src+'/predictions.npz',rec['files_sha256']['predictions.npz']);q=J(old+'/report.json');oz=Z(old+'/predictions.npz',q['predictions_sha256'])
    for m in q['models']:track(old+f"/seed-{m['base_seed']}.pt",m['checkpoint_sha256'])
    np.testing.assert_array_equal(z['train_indices'],Tn);np.testing.assert_array_equal(z['validation_indices'],Vn);np.testing.assert_array_equal(oz['train'],Tn);np.testing.assert_array_equal(oz['validation'],Vn);np.testing.assert_array_equal(oz['seeds'][0],z['cnn']);np.testing.assert_array_equal(oz['mean3'],oz['seeds'].mean(0));assert not set(groups[Tn])&set(groups[Vn])
    P=np.column_stack([z[n] for n in ['baseline','kernel','cnn','Smean','Bmean']]);idx=len(nodes);name=f'node-{idx:03d}';arr(_repo_path(O, 'cache', f'{name}.npz'),train=Tn,validation=Vn,P=P,oldseeds=oz['seeds'],short=short)
    node={'index':idx,'name':name,'split':split,'si':si,'outer':k,'middle':j,'deep':ell,'type':typ,'offset':q['offset'],'train':Tn.tolist(),'validation':Vn.tolist(),'source':src,'old3_source':old,'cache_sha256':H(_repo_path(O, 'cache', f'{name}.npz'))};nodes.append(node)
   deep_validation=[n for n in nodes if n['si']==si and n['outer']==k and n['middle']==j and n['type']=='deep'];assert sorted(i for n in deep_validation for i in n['validation'])==sorted(A.tolist())
  assert sorted(covered)==T.tolist()
  for seed,path in oldplan['checkpoints'][split][str(k)].items():track(path,oldplan['inputs_sha256'][path])
  for seed in oldplan['new_seeds']:
   path=f'experiments/cnn-seed-expand-203/{split}/seed-{seed}-fold-{k}';r=J(path+'/report.json');track(path+'/model.pt',r['model_sha256'])
print(json.dumps({'prepared_nodes':len(nodes),'provenance_inputs':len(inputs),'stage':'inputs_audit'}),flush=True)
for p in ['artifacts/nested-fusion-112/validation-replay.json','artifacts/nested-fusion-112/incumbent-head-control-validation.json','reports/nested-calibration-074-validation.json','reports/nested-calibration-074-validation-v2.json']:
 if (_repo_path(R, p)).exists():J(p)
arr(_repo_path(O, 'cache/data.npz'),ids=ids,y=y,groups=groups,fold=fold,R=p202['R'],F=p203['primary'],images=images,X=X,outerP=np.stack(outerP),oldouter=p203['old_seeds'],newouter=p203['new_seeds'],short=p202['short'],L=p202['L_seeds'].mean(1),oldslopes=p202['slopes'],oldcorrection=p202['correction'])
plan={'study':204,'locked_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':'Parent explicitly authorized complete nested comparison540newCNN+48validation,one additionalsafeClaude review,no othertraining expansion/test/holdout/full500/CSV/submission/install/download.','splits':splits,'old_seeds':oldplan['old_seeds'],'new_seeds':oldplan['new_seeds'],'nodes':nodes,'folds':folds,'blocks':blocks,'recipe':oldplan['recipe'],'recipe_update':'Per-node updates80*ceil(n/32),alllabels outside currenttrain masked;oldoffsets unchanged;no other recipechange.','arms':{'R':'old3alllevels,refit same downstream;must reproduce021 before newCNN starts','F':'203cached6outer withold3postprocess,diagnostic only','N':'all6alllevels,fullrefit same downstream policy;onlycandidate'},'main_CNN_fits':540,'main_updates':237600,'poison_CNN_fits':48,'poison_scope':'originalfold0 all12nodes*3seeds outerV perturbation plusmiddle0andits3deepnodes*3seeds wholeB perturbation;allnewtraincalls masknontrainlabels.','downstream':'60affine+720nonnullresidualfits perarm;all15outerlabelpoison perarm;fivecandidateorderexistingnone,ridge1,ridge01,rbf1,rbf01;CVseed20271013+100k+j;512bootstrapseed20271204+10000si+100k+j;samepaireddraws;585existingfeatures.','weights':{'raw':[0,.3,.3,.2,.2],'short':[.9,.1],'L':[.9,.1]},'gate':['technical_validation','pooled_gain_positive','all3_gains_nonnegative','conditional_CI_lower_positive','pooled_top5_removal_positive','acrosspartition_variance_not_increased'],'evaluation':'all3beforecomparativescores;10000paired400IDclusterseed202610204;R/F/N contrastsandtop1/5/10stress;conditionalnotselectionadjusted.','review':'One authorizedClaudeadditionalreview;clearance disposition required beforeCNN training;advisorynotautomatic','resources':{'CPU_threads':2,'memory_limit_bytes':6000000000,'main_time_limit_seconds':7200,'poison_time_limit_seconds':1800,'forecast_total_minutes':[75,95],'network':False,'cost':0},'inputs_sha256':inputs,'protected_sha256':protected,'data_sha256':H(_repo_path(O, 'cache/data.npz')),'preparation_seconds':time.perf_counter()-start,'same400_adaptive_history':True,'no_full500_test_holdout_CSV_submission':True}
W(_repo_path(O, 'plan.json'),plan);W(_repo_path(O, 'preparation.json'),{'passed':True,'nodes':180,'old_nested_CNNs':540,'outerCNNs_reused':90,'input_files':len(inputs),'all_sources_hash_and_indices_checked':True,'neural_training':0,'seconds':time.perf_counter()-start,'plan_sha256':H(_repo_path(O, 'plan.json'))});print(json.dumps({'preparation_passed':True,'inputs':len(inputs),'nodes':len(nodes),'seconds':time.perf_counter()-start}),flush=True)
