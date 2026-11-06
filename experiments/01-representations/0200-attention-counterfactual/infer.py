from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import importlib,gc
import common as c,intervention as it
from common import R,P,np,torch
c.check();ref=c.load(P/'experiments/query-normalization-197/OOF.npz',['ids','fold','groups']);assert ref['ids'].tolist()==c.plan['IDs'] and len(set(ref['ids']))==len(set(ref['groups']))==400;moment_ids=c.load(P/'artifacts/patch-attention-092/token_moments.npz',['ids']);np.testing.assert_array_equal(moment_ids['ids'],ref['ids']);records=[];out={'ids':ref['ids'],'fold':ref['fold']};atts={};assert torch.backends.mps.is_available()
with c.Monitor('inference',600),c.TokenReader() as reader,torch.inference_mode():
 for study,folder in [('196','multihead-decoder-196'),('197','query-normalization-197')]:
  module=importlib.import_module('model'+study);normal_all=np.full(400,np.nan);uniform_all=np.full(400,np.nan);cpu_all={arm:np.full(400,np.nan) for arm in ['normal','uniform']};atlas={arm:{} for arm in ['normal','uniform']}
  for k in range(5):
   V=np.flatnonzero(ref['fold']==k);T=np.flatnonzero(ref['fold']!=k);base=P/f'experiments/{folder}/fold-{k}/mha';ck=torch.load(_repo_io(base/'model.pt'),map_location='cpu',weights_only=True);norm=c.load(base/'normalization.npz');saved=c.load(base/'predictions.npz',['indices','ids','prediction']);assert ck['validation_indices']==V.tolist() and ck['train_indices']==T.tolist() and ck['seed']==202710196+k and ck['updates']==400;np.testing.assert_array_equal(saved['ids'],ref['ids'][V]);np.testing.assert_array_equal(saved['indices'],V);assert ck['target_mean']==float(norm['target_mean']) and ck['target_std']==float(norm['target_std']);before=c.statehash(ck['state']);details={};metrics={arm:{} for arm in ['normal','uniform']}
   for device in ['mps','cpu']:
    model=module.initialize(norm,k,device);model.load_state_dict(ck['state'],strict=True);model.eval();model.requires_grad_(False);assert not any(p.requires_grad for p in model.parameters()) and all(not q.training for q in model.modules());assert c.statehash(model.state_dict())==before;self_func=model.self_mixer.mix.__func__;rngCPU=torch.get_rng_state().clone();rngMPS=torch.mps.get_rng_state().clone();pred={arm:[] for arm in ['normal','uniform']};max_prob_error=0.;manualerr=0.;double_prediction_error=0.
    for j in range(0,80,2):
     x=torch.from_numpy(reader.read(V[j:j+2])).to(device);p,z=it.call(model,x,False);q,w=it.call(model,x,True);again,again_z=it.call(model,x,False);assert torch.equal(p,again) and torch.equal(z['cross_weights'],again_z['cross_weights']) and torch.equal(z['self_weights'],again_z['self_weights']);q2,_=it.call(model,x,True);assert torch.equal(q,q2);assert model.self_mixer.mix.__func__==self_func
     for arm,value,dist in [('normal',p,z),('uniform',q,w)]:
      pp=value.cpu().numpy().astype(float)*ck['target_std']+ck['target_mean'];pred[arm].extend(pp.tolist())
      if device=='mps':
       measures,error=it.attention(dist);max_prob_error=max(max_prob_error,error)
       for n,a in measures.items():metrics[arm].setdefault(n,[]).append(a)
     if device=='cpu' and j==0:
      for arm,value in [('normal',p),('uniform',q)]:
       manual,er=it.independent_double(model,x,arm=='uniform');manualerr=max(manualerr,er);double_prediction_error=max(double_prediction_error,float(abs(manual-value.cpu().numpy()).max())*ck['target_std'])
    assert c.statehash(model.state_dict())==before and all(p.grad is None for p in model.parameters());assert torch.equal(rngCPU,torch.get_rng_state()) and torch.equal(rngMPS,torch.mps.get_rng_state());pred={a:np.array(p) for a,p in pred.items()};assert all(np.isfinite(p).all() for p in pred.values());details[device]={'pred':pred,'original_model_state_and_RNG_unchanged':True,'normal_after_intervention_exact':True,'same_condition_repeat_exact':True,'probability_sum_max_error':max_prob_error,'double_SDPA_or_dense_uniform_max_error':manualerr,'double_vs_float32_prediction_max_error':double_prediction_error};assert double_prediction_error<.002
    if device=='mps':np.testing.assert_array_equal(pred['normal'],saved['prediction']);normal_all[V]=pred['normal'];uniform_all[V]=pred['uniform']
    else:
     for arm in pred:cpu_all[arm][V]=pred[arm]
    del model;gc.collect();torch.mps.empty_cache()
   errs={arm:float(abs(details['mps']['pred'][arm]-details['cpu']['pred'][arm]).max()) for arm in ['normal','uniform']};assert max(errs.values())<.002
   for arm in metrics:
    for n,parts in metrics[arm].items():
     a=np.concatenate(parts);assert len(a)==80
     if n not in atlas[arm]:atlas[arm][n]=np.empty((400,*a.shape[1:]),dtype=float)
     atlas[arm][n][V]=a
   rec={'study':int(study),'fold':k,'rows':80,'original_MPS_saved_prediction_exact':True,'state_and_RNG_both_backends_unchanged':True,'self_algorithm_parameters_unchanged':True,'MPS_CPU_maxabs':errs,'double_reference_maxabs':details['cpu']['double_SDPA_or_dense_uniform_max_error'],'double_vs_CPU_float32_prediction_maxabs':details['cpu']['double_vs_float32_prediction_max_error'],'original_state_SHA256':before,'normal_after_intervention_repeat_exact':True,'uniform_repeat_exact':True};records.append(rec);c.emit(rec)
  out.update({study+'_normal':normal_all,study+'_uniform':uniform_all,study+'_CPU_normal':cpu_all['normal'],study+'_CPU_uniform':cpu_all['uniform']})
  for arm in atlas:
   for n,a in atlas[arm].items():atts[study+'_'+arm+'_'+n]=a
 # These outputs and statistics have been computed without loading any heldout y array.
 c.arrays(R/'predictions.npz',**out);c.arrays(R/'attention.npz',ids=ref['ids'],fold=ref['fold'],**atts);cert={'passed':True,'models':10,'rows_per_study':400,'folds':5,'conditions':['originalMHA','cross_uniform_only'],'new_training_optimizer_steps':0,'groundtruth_y_loaded_in_inference':False,'all_records':records,'max_CPU_MPS_difference':max(max(q['MPS_CPU_maxabs'].values()) for q in records),'max_double_reference_difference':max(q['double_reference_maxabs'] for q in records),'predictions_SHA256':c.sha(R/'predictions.npz'),'attention_SHA256':c.sha(R/'attention.npz'),'scope':'Selfparameters andalgorithm remainunchanged; selfprobabilities maychange as crossoutputschange. Alltransformationbuffers remainfrozen.','no_test_full500_raw_data':True}
c.check();c.write(R/'inference-validation.json',cert);c.emit({k:v for k,v in cert.items() if k!='all_records'})
