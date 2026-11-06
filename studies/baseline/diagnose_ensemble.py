"""One post-hoc diagnosis using saved development OOF/checkpoints only; no image access."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,csv,hashlib,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, '.venv')).resolve():raise SystemExit('Use project .venv')
import numpy as np
import torch

def read_csv(p):
    with p.open() as f:return {r['ID']:r for r in csv.DictReader(f)}
def write(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=_repo_path(ROOT, 'artifacts/diagnostic-008');out.mkdir(exist_ok=False);start=time.perf_counter()
    recipes={'equal_three':{'baseline':1/3,'grain':1/3,'cnn':1/3},'base50_grain25_cnn25':{'baseline':.5,'grain':.25,'cnn':.25},'base75_grain125_cnn125':{'baseline':.75,'grain':.125,'cnn':.125}}
    plan={'purpose':'Post-hoc bounded diagnostic, no training or promotion by arbitrary 0.15 threshold','recipes':recipes,'weights_chosen_before_evaluation':True,'caveat':'Recipes fixed now but components previously selected on these same OOF rows. Not a prospectively independent experiment. No optimized weights.','data_scope':'Existing development OOF, metadata, logs, source and saved CNN checkpoints only. No train/holdout/test image reads.','seed_bootstrap':20261004,'bootstrap_replicates':5000}
    write(out/'plan.json',plan)
    a=read_csv(_repo_path(ROOT, 'artifacts/run-006/development_oof.csv'));b=read_csv(_repo_path(ROOT, 'artifacts/run-007/development_oof.csv'));sp=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text());ids=sorted(a)
    assert set(a)==set(b)==set(sp['development_folds']) and len(ids)==400
    y=np.array([float(a[s]['hardness']) for s in ids]);assert np.array_equal(y,[float(b[s]['hardness']) for s in ids])
    p={'baseline':np.array([float(a[s]['baseline']) for s in ids]),'grain':np.array([float(a[s]['grain_ridge100']) for s in ids]),'cnn':np.array([float(b[s]['cnn']) for s in ids])}
    np.testing.assert_allclose(p['baseline'],[float(b[s]['baseline']) for s in ids],rtol=0,atol=1e-10)
    fid=np.array([sp['development_folds'][s] for s in ids]);cnn=json.loads((_repo_path(ROOT, 'artifacts/run-007/report.json')).read_text());benchmark=json.loads((_repo_path(ROOT, 'artifacts/cnn-benchmark-001/benchmark.json')).read_text())
    # Audit normalization from development targets and saved checkpoint scalars.
    audits=[];curves=[]
    for k in range(5):
        ck=torch.load(_repo_path(ROOT, f'artifacts/run-007/fold-{k}.pt'),map_location='cpu',weights_only=True);tr=np.flatnonzero(fid!=k);n=ck['normalization']
        assert set(ck['train_ids'])=={ids[i] for i in tr}
        np.testing.assert_allclose([n['y_mean'],n['y_std']],[y[tr].mean(),y[tr].std()],rtol=0,atol=1e-12)
        assert n['x_std']>0 and all(np.isfinite(v) for v in n.values())
        audits.append({'fold':k,'train_rows':len(tr),'normalization':n,'target_normalization_matches_train_only':True,'image_stats_recomputed':False})
        c=cnn['curves'][k];loss=np.array([z['augmented_train_rmse'] for z in c]);epochs=np.arange(1,len(loss)+1)
        curves.append({'fold':k,'epoch_1':float(loss[0]),'epoch_10':float(loss[9]),'epoch_20':float(loss[19]),'epoch_30':float(loss[29]),'epoch_40':float(loss[39]),'last10_slope_rmse_per_epoch':float(np.polyfit(epochs[-10:],loss[-10:],1)[0]),'mean_epochs21_30':float(loss[20:30].mean()),'mean_epochs31_40':float(loss[30:40].mean()),'averaged_model_train_rmse':cnn['folds'][k]['train_rmse'],'averaged_model_validation_rmse':cnn['folds'][k]['validation_rmse'],'final_generalization_gap':cnn['folds'][k]['validation_rmse']-cnn['folds'][k]['train_rmse']})
    basecorr=np.corrcoef(np.stack([y-p[n] for n in ('baseline','grain','cnn')]))
    correction_corr=float(np.corrcoef(p['grain']-p['baseline'],p['cnn']-p['baseline'])[0,1])
    p['prior_grain25']=.75*p['baseline']+.25*p['grain'];p['prior_cnn25']=.75*p['baseline']+.25*p['cnn']
    priorcorr=float(np.corrcoef(y-p['prior_grain25'],y-p['prior_cnn25'])[0,1])
    for name,weights in recipes.items():p[name]=sum(w*p[n] for n,w in weights.items())
    scores={};rng=np.random.default_rng(plan['seed_bootstrap']);draws=rng.integers(0,len(y),(5000,len(y)))
    baseerr=(y-p['baseline'])**2
    q=np.quantile(y,[0,.25,.5,.75,1])
    for name,pred in p.items():
        err=(y-pred)**2;diff=np.sqrt(baseerr[draws].mean(1))-np.sqrt(err[draws].mean(1))
        strata=[]
        for j in range(4):
            mask=(y>=q[j])&(y<=q[j+1] if j==3 else y<q[j+1]);strata.append({'target_range':q[j:j+2].tolist(),'rmse':rmse(y[mask],pred[mask]),'mean_prediction_minus_target':float((pred[mask]-y[mask]).mean())})
        scores[name]={'oof_rmse':rmse(y,pred),'fold_rmse':[rmse(y[fid==k],pred[fid==k]) for k in range(5)],'improvement_vs_baseline':rmse(y,p['baseline'])-rmse(y,pred),'conditional_paired_bootstrap_improvement_95':np.percentile(diff,[2.5,97.5]).tolist(),'prediction_std':float(pred.std()),'target_std':float(y.std()),'prediction_min':float(pred.min()),'prediction_max':float(pred.max()),'bias':float((pred-y).mean()),'calibration_slope_diagnostic_only':float(np.cov(y,pred,ddof=0)[0,1]/np.var(pred)),'target_strata':strata}
    report={**plan,'n_development':400,'target_min':float(y.min()),'target_max':float(y.max()),'normalization_audit':audits,'learning_curves':curves,'validation_curve_available':False,'validation_curve_limitation':'Only final averaged-model validation RMSE was logged. Cannot locate validation optimum or establish overfitting onset. Training losses use augmentation/dropout and evolving weights; not directly comparable to final eval losses.','residual_correlation_order':['baseline','grain','cnn'],'residual_correlation_matrix':basecorr.tolist(),'prior_improved_blend_residual_correlation':priorcorr,'grain_cnn_correction_correlation':correction_corr,'scores':scores,'resizing_diagnosis':{'source':'cnn_experiment.py: PIL BOX resize 256 to 128; four stride-2 convs; final 8x8 feature map','verified_by_pixel_comparison':False,'finding':'2x2 averaging irreversibly removes some pixel-scale detail; likely relevant to thin boundaries, but grain information loss/accuracy impact was not measured. No images read in this diagnosis. Spatial pooling discards arrangement; multiscale resolution hypothesis remains untested.'},'capacity_conclusion':'Only one small scratch architecture/seed/schedule/resolution was tested. No technical performance ceiling established. Falling train loss supports an under-training hypothesis; train/validation gaps also allow generalization limitations. Both require controlled experiments.','benchmark_based_next_experiments':{'frozen_128px_80epoch':{'hypothesis':'Additional optimization may help because train loss is still falling; validation may also worsen.','estimated_5fold_training_seconds':round(cnn['elapsed_seconds']*2,1),'protocol':'One fixed 80-epoch run; log diagnostics at40/80 but choose primary80 endpoint beforehand, compare the same predeclared blends; no outer-fold checkpoint tuning.'},'frozen_256px_40epoch':{'hypothesis':'Retaining thin boundary details may improve morphology information.','estimated_5fold_training_seconds_range':[round(cnn['elapsed_seconds']*3),round(cnn['elapsed_seconds']*6)],'protocol':'Same architecture/seed/40epochs and fixed blends; benchmark256 first, reduce batch only for memory; time/memory projections are unmeasured.'},'verification':{'protocol':'After choosing one hypothesis, evaluate baseline and fixed ensemble on two predeclared new development-only grouped fold seeds. Reusing400 rows diagnoses split sensitivity but is not new independent data.','cnn_extra_time':'Approximately twice chosen 5-fold runtime plus feature-model refits. No extra packages or pretrained weights needed.'}},'unchanged_run005_model_sha256':digest(_repo_path(ROOT, 'artifacts/run-005/best_model.joblib')),'elapsed_seconds':time.perf_counter()-start}
    assert report['unchanged_run005_model_sha256']==cnn['baseline_model_sha256']
    write(out/'report.json',report)
    with (out/'development_ensemble_oof.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','hardness',*p]);w.writerows([s,y[i],*[pred[i] for pred in p.values()]] for i,s in enumerate(ids))
    print(json.dumps({'curves':curves,'correlation':basecorr.tolist(),'prior_blends_correlation':priorcorr,'correction_correlation':correction_corr,'scores':{n:{k:v for k,v in z.items() if k not in ('target_strata',)} for n,z in scores.items()}},indent=2))
if __name__=='__main__':main()
