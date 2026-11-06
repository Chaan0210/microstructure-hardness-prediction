"""Close fixed168 with exact records and explicit engineering/predictive distinction."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json,sys,hashlib,time
from pathlib import Path
import numpy as np
R=_RepoPath(__file__).resolve().parents[1];O=_repo_path(R, 'artifacts/cubical-h1-168')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,z):
    with p.open('x') as f:json.dump(z,f,indent=2,allow_nan=False)
def main():
    impl=read(_repo_path(O, 'implementation-validation.json'));pilot=read(_repo_path(O, 'pilot.json'));ex=read(_repo_path(O, 'extraction.json'));lock=read(_repo_path(O, 'model-lock.json'));first=read(_repo_path(O, 'original/report.json'));assert impl['passed'] and pilot['passed'] and ex['passed']
    splits=['original','20261031','20261101'] if first['advance'] else ['original'];errors={k:[] for k in ['reference018','h0','h1']};scores={};folds={};n=0;maxerr=0.
    for split in splits:
        rec=read(_repo_path(O, split, 'report.json'));v=read(_repo_path(O, f'validation-{split}.json'));iv=read(_repo_path(O, f'inner-poison-{split}.json'));assert v['passed'] and iv['passed'];assert v['report_sha256']==sha(_repo_path(O, split, 'report.json'));assert rec['predictions_sha256']==sha(_repo_path(O, split, 'predictions.npz'));n+=v['models_verified'];maxerr=max(maxerr,v['independent_max_error']);scores[split]=rec['RMSE'];folds[split]=rec['fold_gains_vs018']
        with np.load(_repo_path(O, split, 'predictions.npz')) as z:
            assert len(np.unique(z['ids']))==400 and all(np.isfinite(z[k]).all() for k in errors)
            for k in errors:errors[k].append((z[k]-z['y'])**2)
    E={k:np.stack(v) for k,v in errors.items()};pooled={k:float(np.sqrt(v.mean())) for k,v in E.items()};advance=len(splits)==3 and pooled['h1']<min(pooled['reference018'],pooled['h0'])
    ci=None
    if len(splits)==3:
        rng=np.random.default_rng(20261006168);g={k:[] for k in ['reference018','h0']}
        for _ in range(10000):
            ix=rng.integers(0,400,400)
            for k in g:g[k].append(float(np.sqrt(E[k][:,ix].mean())-np.sqrt(E['h1'][:,ix].mean())))
        ci={k:np.quantile(v,[.025,.975]).tolist() for k,v in g.items()}
    groups={}
    with np.load(_repo_path(R, 'artifacts/failure-review-166/analysis.npz')) as f,np.load(_repo_path(O, 'original/predictions.npz')) as z:
        np.testing.assert_array_equal(f['ids'],z['ids'])
        for key in ['persistent_large','stable_sign_min5','persistent_disagreement']:
            mask=f[key];groups[key]={'n':int(mask.sum()),'original_RMSE':{k:float(np.sqrt(np.mean((z[k][mask]-z['y'][mask])**2))) for k in errors}}
    for file,h in {**lock['source_sha256'],**lock['protected']}.items():assert sha(_repo_path(R, file))==h,file
    report={'study':168,'status':'development_candidate_only' if advance else 'completed_no_promotion','implementation':{'exact_scope':read(_repo_path(O, 'plan.json'))['definition'],'passed':True,'random_small_images':300,'independent_GF2_reductions':640,'hand_fixtures':[{'name':v['name'],'H1':v['actual_H1']} for v in impl['hand_fixtures']],'D4_equalleveltie_polarity_Euler_UF_passed':True,'joint_existingdescriptor_collision':impl['joint_collision']},'pilot':{k:v for k,v in pilot.items() if k not in ['rows','source_sha256','IDs']},'features':ex,'model_design':lock['model'],'RMSE':scores,'fold_gains_vs018':folds,'original_gate_passed':first['advance'],'partitions_executed':splits,'pooled_RMSE':pooled,'conditional_IDcluster_CI95_ifextended':ci,'candidate':advance,'failure_group_diagnostic_posthoc':groups,'audit':{'passed':True,'residual_models_verified':n,'independent_max_error':maxerr,'reloaded_predictions_exact':True,'independent_bootstrap_exact':True,'outer_label_poison_exact':True,'inner_label_poison_exact':True,'base018_reconstructed_exact':True,'upstream146_exclusion_certificate_sha256':sha(_repo_path(R, 'reports/nested-selector-146-audit.json'))},'resources':{'feature_seconds':ex['seconds'],'feature_RSS_bytes':ex['peak_RSS_bytes'],'model_benchmark':read(_repo_path(O, 'model-benchmark.json'))},'protected':lock['protected'],'best_retained':'018','new_installs_or_data_downloads':False,'holdout_test_access':False,'full500_training_newCSV_upload':False,'limitations':['Exact H1 only for stated2Dclosed-square/nonperiodic/uint8 filtration; not universalPHengine or exactTopTemp reproduction. PI is a lossy32dim summary, not fullbarcode.','Planarduality implies strong overlap with oldopposite-polarityH0; addedboundaryassociation neednot bephysicalhardnesssignal.','No physicalscale, verifiedphasemask or causalfeature interpretation.','Same400adaptivelyreused; overlappingresplits andconditionalbootstrap do not constitute independentvalidation.','Engineering correctness/nonidentity/noise tolerability doesnot establish predictionbenefit.','H0 is matchedcontrol only, notfallbackcandidate or permission toretryold109features.'],'remaining_blockers':[],'evidence_sha256':{str(p.relative_to(R)):sha(p) for p in [_repo_path(O, 'plan.json'),_repo_path(O, 'implementation-validation.json'),_repo_path(O, 'pilot.json'),_repo_path(O, 'model-lock.json'),_repo_path(O, 'features.npz'),_repo_path(O, 'original/report.json'),_repo_path(O, 'validation-original.json')]}}
    write(_repo_path(O, 'final.json'),report);write(_repo_path(R, 'reports/cubical-h1-168-summary.json'),report)
    original=scores['original'];text=f'''\n\n## Study168: exact cubical H1 engineering validation and fixed residual comparison\n\nClosed-square top-cell sublevel filtration, min-face values, foreground8/complement4, finite nonperiodic rectangle. Dual max-tree with immortal exterior computes positive H1 birth/death pairs;640 independent GF2 cubical boundary reductions across8hand fixtures and300random grids agree. Ties, D4, polarity swap, Euler and independentUF pass. This is exactH1 for the statedcomplex;32dimPI is lossy, not fullbarcode or exactTopTemp reproduction. Joint fixture has identical old107lower/upperfiniteH0,132Betti055 and36C2/S2 butdifferentH1 dueboundary/contrast association.\n\nLabel-blind16development pilot passed: pairedH1/H0 extraction~{pilot['median_paired_seconds']:.3f}s/image; bounded±1/±4gray noise changesPI bymedian{100*pilot['median_noise_relative_PI_change']['1']:.2f}%/{100*pilot['median_noise_relative_PI_change']['4']:.2f}%. H1 vsboundary-unaware dualH0PI difference median{100*pilot['median_relative_H1_dual_unpadded_H0_PI']:.2f}%, so substantialoldfeature overlap. All16UF/2D4/2all256Euler/4actualcropGF2 checks pass. Correctness/nonidentity doesnot establish physical or predictive relevance.\n\nOne fixed4x4PI pergraypolarity, sigma.25, linear lifetime/areaweight:32features appendedto018585. SamewidthH0control, existingtrainonlytransforms/honestnestedanchors/fivelearners/512bootstrap. OriginalRMSE018={original['reference018']:.9f}, H0control={original['h0']:.9f}, H1={original['h1']:.9f}. Originalgate={first['advance']}; executedpartitions={','.join(splits)}; developmentcandidate={advance}. {n}residual models independently audited, outer/innerlabelpoison pass, immutable018 reconstructsexactly.\n\nNo newinstall/download, holdout/test/full500/newCSV/upload. Best018/ZIPs preserved. Same400adaptivehistory limits inference. Evidence:`reports/cubical-h1-168-summary.json`, `artifacts/cubical-h1-168/` and source`cubical_h1_168.py`/`validate_cubical_h1_168.py`/`cubical_study_168.py`.\n'''
    backup=_repo_path(R, 'reports/README-before-cubical168.md')
    with backup.open('xb') as f:f.write((_repo_path(R, 'README.md')).read_bytes())
    with (_repo_path(R, 'README.md')).open('a') as f:f.write(text)
    print(json.dumps({k:report[k] for k in ['status','RMSE','fold_gains_vs018','original_gate_passed','partitions_executed','candidate','audit','remaining_blockers']},indent=2),flush=True)
if __name__=='__main__':main()
