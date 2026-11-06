"""Paired400-ID size curves and fixed-mixture contribution accounting for127."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,datetime
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import resume_base_learning_curve_127 as q
import numpy as np
e=q.e

def main():
    plan=q.check();valid=e.read(q.OUT/'validation.json');assert valid['passed'];allP=[];ys=[];records=[];evidence={}
    for split in e.SPLITS:
        path=q.OUT/split;rec=e.read(path/'report.json');assert e.sha(path/'report.json')==valid['report_sha256'][split];assert e.sha(path/'predictions.npz')==rec['predictions_sha256'];records+=rec['records']
        with np.load(_repo_io(path/'predictions.npz')) as z:
            if ys:np.testing.assert_array_equal(z['y'],ys[0]);assert z['ids'].tolist()==ids
            ids=z['ids'].tolist();ys.append(z['y'].copy());allP.append(z['components'].copy())
        evidence[str((path/'report.json').relative_to(ROOT))]=e.sha(path/'report.json');evidence[str((path/'predictions.npz').relative_to(ROOT))]=rec['predictions_sha256']
    P=np.stack(allP);Y=np.stack(ys);err=P-Y[:,None,None,None,:];E=err**2;perid=E.mean(axis=(0,2));R=np.sqrt(perid.mean(-1));mix=np.einsum('slpcn,c->slpn',P,q.W);mixE=(mix-Y[:,None,None,:])**2;mr=np.sqrt(mixE.mean(axis=(0,2,3)));draw=np.random.default_rng(20271107).integers(0,400,(10000,400));boot=np.sqrt(perid[:,:,draw].mean(-1));mb=np.sqrt(mixE.mean(axis=(0,2))[:,draw].mean(-1));rows=[]
    for l,level in enumerate(q.LEVELS):
        rs=[r for r in records if r['level']==level];den=sum(r['n_train'] for r in rs);train=np.sqrt(np.sum([r['train_SSE'] for r in rs],axis=0)/den);permodel=[];er=err[:,l].transpose(2,0,1,3).reshape(4,-1);mu=er.mean(1);second=er@er.T/er.shape[1];cov=(er-mu[:,None])@(er-mu[:,None]).T/er.shape[1];assert abs(q.W@second@q.W-mr[l]**2)<1e-10;assert abs(q.W@cov@q.W+(q.W@mu)**2-mr[l]**2)<1e-10
        individual=float(q.W@(R[l]**2));diversity=float(np.mean(np.einsum('c,spcn->spn',q.W,(P[:,l]-mix[:,l,:,None,:])**2)));assert abs(individual-diversity-mr[l]**2)<1e-10
        for c,name in enumerate(q.NAMES):
            w=q.W.copy();w[c]=0;w/=w.sum();ab=np.einsum('spcn,c->spn',P[:,l],w);abl=float(np.mean((ab-Y[:,None,:])**2));permodel.append({'name':name,'train_RMSE':float(train[c]),'same_outer80_validation_RMSE':float(R[l,c]),'validation_minus_train_RMSE':float(R[l,c]-train[c]),'raw_mix_leaveout_RMSE':float(np.sqrt(abl)),'raw_mix_leaveout_minus_full_MSE':abl-mr[l]**2})
        rows.append({'train_size':level,'bundles':len(rs),'training_row_evaluations_overlapping':den,'components':permodel,'raw_fixed_mix_RMSE':float(mr[l]),'weighted_mean_individual_MSE':individual,'weighted_disagreement_MSE_reduction':diversity,'error_mean':mu.tolist(),'error_second_moment':second.tolist(),'centered_error_covariance':cov.tolist(),'error_correlation':np.corrcoef(er).tolist(),'kernel_choice_counts':{str(key):sum((r['kernel_bandwidth'],r['kernel_alpha'])==key for r in rs) for key in sorted(set((r['kernel_bandwidth'],r['kernel_alpha']) for r in rs))},'CNN_last10_aug_train_RMSE_mean':float(np.mean([r['CNN_aug_train_RMSE_last10'] for r in rs if r['CNN_aug_train_RMSE_last10'] is not None])),'CNN_previous10_aug_train_RMSE_mean':float(np.mean([r['CNN_aug_train_RMSE_prev10'] for r in rs if r['CNN_aug_train_RMSE_prev10'] is not None]))})
    changes=[]
    for lo,hi in [(0,1),(1,2),(0,2)]:
        values=[]
        for c,name in enumerate(q.NAMES):values.append({'name':name,'validation_RMSE_change_more_minus_less':float(R[hi,c]-R[lo,c]),'conditional_ID_cluster95_CI':np.quantile(boot[hi,c]-boot[lo,c],[.025,.975]).tolist(),'per_partition_RMSE_changes':[float(np.sqrt(E[s,hi,:,c].mean())-np.sqrt(E[s,lo,:,c].mean())) for s in range(3)]})
        changes.append({'from':q.LEVELS[lo],'to':q.LEVELS[hi],'components':values,'raw_mix_RMSE_change':float(mr[hi]-mr[lo]),'raw_mix_conditional_ID_cluster95_CI':np.quantile(mb[hi]-mb[lo],[.025,.975]).tolist()})
    # Independent bootstrap count arithmetic for200 draw fixtures, allsize/componentcells.
    maxerr=0.
    for b,ix in enumerate(draw[:200]):
        counts=np.bincount(ix,minlength=400);manual=np.sqrt(np.einsum('n,lcn->lc',counts,perid)/400);maxerr=max(maxerr,float(abs(manual-boot[:,:,b]).max()))
    assert maxerr<1e-12
    report={'study':'127 saved current base-model size and complementarity diagnostic','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'complete; interpretation required; no model selection','component_order':list(q.NAMES),'fixed_raw_weights':q.W.tolist(),'curve':rows,'size_changes':changes,'observations':{'unique_images':400,'partitions':3,'nested_paths_per_outer':3,'same_outer80_at_all_sizes':True,'mean_individual_submodel_MSE_not_subset_ensemble':True,'largest_size_duplicated_path_axis_not_extra_models':True},'validation':valid,'bootstrap_independent_count_fixture_max_error':maxerr,'old_failures_preserved':{'initial_batch_arithmetic_assertion':'reports/base-learning-curve-127-original.log','legacy_CNN_metadata_assertion':'reports/base-learning-curve-v2-127-original.log','numeric_explanation':plan['numeric_repair'],'metadata_compatibility':e.read(q.OUT/'metadata-compatibility.json')},'interpretation_limits':[plan['scope_bias'],plan['training_protocol'],plan['contribution'],'KRRalpha uses summedloss and selectedalpha/n changes withtrainingn; not isolatedeffective-regularization control. CNNsameepochs has moreoptimizersteps atlargern. Seed offsets differ. No causal attribution amongdata,regularization,optimization.','TrainingMSE and augmented-online trainingcurves are different measurements, not directly interchangeable. Curves cannot establish the benefit of training past80epochs.','Raw4way contributions omit train-onlyaffine,shortB andphysics residual; use121/122 forwholepipeline removal evidence. Do not selectnewweights or dropbranches fromthis table.'],'scope':{'new_fits':0,'leaf_models_replayed':840,'new_backbone_extraction':0,'new_images_decoded':0,'holdout_test_decoded':False,'new_packages':0,'cost':0,'new_candidate':False,'uploaded':False},'evidence_sha256':{**evidence,**{str(f.relative_to(ROOT)):e.sha(f) for f in [q.OUT/'plan.json',q.OUT/'validation.json',q.OUT/'metadata-compatibility.json']}},'source_sha256':e.sha(Path(__file__))}
    e.write(_repo_path(ROOT, 'reports/base-learning-curve-127-analysis.json'),report);q.check();e.emit({'curve':rows,'changes':changes,'bootstrap_check':maxerr})
if __name__=='__main__':main()
