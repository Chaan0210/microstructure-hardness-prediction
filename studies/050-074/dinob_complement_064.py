"""User-authorized single 50:50 average, no mixture-weight search."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,time,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import dinob_mlp_063 as old
import numpy as np,torch
fv=old.fv;h=old.h;SEEDS=old.SEEDS;OUT=_repo_path(ROOT, 'artifacts/dinob-complement-064')

def check():
 old.check();p=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for name,digest in {**p['sources'],**p['inputs'],**p['protected']}.items():assert fv.sha(_repo_path(ROOT, name))==digest,name
 return p

def prepare():
 prior=old.check();OUT.mkdir(exist_ok=False);p={'authorization':'User follow-up explicitly requests one fixed50:50 prediction average after063 and replaces the previous standalone bootstrap-SD gate for this distinct complementarity comparison.','recipe':{'candidate007':.5,'ViTB_primary063':.5},'equivalent_recipe':{'samefold003':.75,'S_MLP_mean3':.125,'B_MLP_mean3':.125},'config':h.CONFIG,'effective_aux_weight':0.,'seeds':list(SEEDS),'original_gate':'Strictly lower original pooled OOF RMSE than007; no bootstrap threshold. If equal or worse reject direction.','confirmation_gate':'Unchanged50:50 average must strictly improve007 pooled RMSE in BOTH existing resplits20261031/20261101, and improve >=3/5 folds in each of all three partitions. All45 B checkpoints reload exactly, fold-only scalers/targets verified, B100epoch replay per split, samefold007 provenance verified. No fallback recipe.','final_fit_gate':'Only after confirmation and validation, lock the sole average recipe then full500 fit3S+3B identical heads (existing003 alreadyfull500 retained), test1000 inference once, ID/finite/order/reload/CSV hash verification. No holdout separate scoring and no test-driven selection.','no_search':'Exactly one ratio50:50. No alternateweights, optimizer, width, epochs, seeds, layers or variants; no bootstrap/pvalue as a significance claim.','previous_gate_explanation':{'value':0.09300178059368959,'quantity':'Population standard deviation (numpy.std ddof=0) across3000 paired row-bootstrap RMSE gains: RMSE(y,007)-RMSE(y,Bprimary).','algorithm':'np.random.default_rng(20270104).integers(0,400,(3000,400)); each bootstrap uses same400 sampled row indices for both fixed OOF residual arrays, with replacement. No refitting.','previous_rule':'063 originally required gain>bootstrapSD and>=3/5 improving folds before resplits, inherited from057-062 as a conservative compute-allocation heuristic.','basis_and_limit':'No theoretical or empirical calibration made1SD a valid hard model-selection threshold. It is not a standard error for the whole procedure, not cross-foldSD, not a significance test, and can reject useful ensemble diversity. It omits fitting, historical selection, dependency and repeated400-row exploration uncertainty. This user-authorized study supersedes that gate for the specified mixture only.','previous_outcome_preserved':'063 remains unchanged and correctly reports failure of its own declared rule;064 tests a different fixed predictor.'},'sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['dinob_complement_064.py','dinob_mlp_063.py','dinob_mlp_head_063.py','auxiliary_head_051.py','earlystop_experiment_057.py','auxiliary_experiment_051.py']},'inputs':{**prior['inputs'],**{n:fv.sha(_repo_path(ROOT, n)) for n in ['reports/dinob-mlp-063-validation.json','artifacts/dinob-mlp-063/original/development_predictions.npz','artifacts/dinob-mlp-063/original/report.json','artifacts/dinob-mlp-063/plan.json']}},'protected':prior['protected'],'scope':'Project-only existinguvCPU/MPS;cost0; no install/download/upload; development400 only before lock.','limitations':fv.CAVEAT};fv.write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'single_ratio':[.5,.5],'original_gate':p['original_gate'],'confirmation_gate':p['confirmation_gate']}),flush=True)

def evaluate(split,ids,y,fid,r003,r007,Smean,Bseeds,source,seconds):
 Bmean=Bseeds.mean(0);Bprimary=.75*r003+.25*Bmean;mix=.5*r007+.5*Bprimary;np.testing.assert_allclose(mix,.75*r003+.125*Smean+.125*Bmean,atol=1e-12,rtol=0);ps={'reference003':r003,'reference007':r007,'Smean':Smean,'Bmean':Bmean,'Bprimary':Bprimary,'mixed50':mix};scores={k:{'rmse':fv.rmse(y,v),'fold_rmse':[fv.rmse(y[fid==j],v[fid==j]) for j in range(5)]} for k,v in ps.items()};comp=old.comparison(y,r007,mix,fid)
 errS=y-r007;errB=y-Bprimary;avg_mse=float(np.mean((errS**2+errB**2)/2));diversity=float(np.mean((r007-Bprimary)**2)/4);actual_mse=float(np.mean((y-mix)**2));np.testing.assert_allclose(actual_mse,avg_mse-diversity,atol=1e-10,rtol=0)
 diag={'prediction_residual_correlation':float(np.corrcoef(errS,errB)[0,1]),'standalone_head_residual_correlation':float(np.corrcoef(y-Smean,y-Bmean)[0,1]),'prediction_difference_RMS':float(np.sqrt(np.mean((r007-Bprimary)**2))),'Bprimary_lower_absolute_error_rows':int((abs(errB)<abs(errS)).sum()),'mean_component_MSE':avg_mse,'disagreement_MSE_reduction':diversity,'mixture_MSE':actual_mse,'MSE_identity_error':float(abs(actual_mse-(avg_mse-diversity))),'interpretation':'For fixed equal average, MSE(mix)=mean(MSE(007),MSE(Bprimary)) - mean((007-Bprimary)^2)/4. Identity explains observed complementarity, not independent validation.'}
 out=_repo_path(OUT, split)
 with (out/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(ids),y=y,fold=fid,Bseeds=Bseeds,**ps)
 report={'split':split,'scores':scores,'comparison':comp,'complementarity':diag,'B_model_source':str(source.relative_to(ROOT)),'predictions_sha256':fv.sha(out/'development_predictions.npz'),'seconds':seconds,'new_candidate':None};fv.write(out/'report.json',report)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':bool(comp['gain']>0),'fixed_ratio':[.5,.5],'comparison':comp})
 check();print(json.dumps(report),flush=True)

def run(split):
 check();torch.set_num_threads(2);ids,S,B,y,g,fid,r003,r007,oldmean=old.inputs(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);start=time.perf_counter()
 if split=='original':
  v=json.loads((_repo_path(ROOT, 'reports/dinob-mlp-063-validation.json')).read_text());assert v['passed'] and v['models_verified']==30;source=old.OUT/split
  with np.load(_repo_io(source/'development_predictions.npz'),allow_pickle=False) as z:
   assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['y'],y);np.testing.assert_array_equal(z['fold'],fid);np.testing.assert_array_equal(z['reference007'],r007);np.testing.assert_array_equal(z['controlS'],oldmean);pred=z['baseB_seeds'].copy()
 else:
  assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance'];source=out;pred=np.full((3,400),np.nan)
  for k in range(5):
   tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(g[tr])&set(g[va]);fd=out/f'fold-{k}';fd.mkdir();sc,ym,ys,xt,yt,bt,edges=old.prep(B[tr],y[tr]);log={'fold':k,'models':{}};oldlog=json.loads((_repo_path(ROOT, 'artifacts/seed-stability-052', split, f'fold-{k}/report.json')).read_text())
   for si,seed in enumerate(SEEDS):
    m,r=h.fit(xt,yt,bt,seed+k,0.);assert r['batch_order_sha256']==oldlog['models'][str(seed)]['batch_order_sha256'];path=fd/f'baseB-seed-{seed}.pt';torch.save({'model':m.state_dict(),'input_dim':6912,'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'scaler_var':torch.tensor(sc.var_),'target_mean':ym,'target_std':ys,'bin_edges_unused':torch.tensor(edges),'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va],'seed':seed+k,'config':h.CONFIG,'effective_aux_weight':0.},_repo_io(path));pred[si,va]=h.predict(m,sc.transform(B[va]),ym,ys);r.update(checkpoint_sha256=fv.sha(path),validation_rmse=fv.rmse(y[va],pred[si,va]));log['models'][f'baseB-{seed}']=r
   fv.write(fd/'report.json',log);print(json.dumps({'split':split,'fold':k,'seconds':time.perf_counter()-start}),flush=True)
 evaluate(split,ids,y,fid,r003,r007,oldmean,pred,source,time.perf_counter()-start)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','original','20261031','20261101']);a=ap.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
