"""Archive completed120 without replacing incumbent artifacts."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import ranking_objective_120 as o
e=o.e
p=o.check();report=_repo_path(ROOT, 'reports/ranking-objective-120-summary.json');r=e.read(report);assert r['status']=='completed_not_promoted' and r['decision']['retain013']
old=e.read(_repo_path(ROOT, 'artifacts/problem-definition-111/plan.json'))['existing_report_sha256']
for name,h in old.items():assert e.sha(_repo_path(ROOT, name))==h,name
protected=e.read(_repo_path(ROOT, 'reports/candidate011-public-submission.json'))['protected_sha256']
for name,h in protected.items():assert e.sha(_repo_path(ROOT, name))==h,name
readme=_repo_path(ROOT, 'README.md');before=e.sha(readme);assert before==e.read(_repo_path(ROOT, 'reports/readme-update-119.json'))['after_sha256']
backup=_repo_path(ROOT, 'reports/README-before-120.md')
with backup.open('xb') as f:f.write(readme.read_bytes())
d=r['decision'];scores=d['pooled_RMSE'];text=f"""

## Study 120: MSE and pairwise ranking objective (completed)

The unchanged cached ViT-B short head was trained for the same 12 epochs with half-MSE alone or half-MSE plus a fixed 0.1 pairwise logistic ranking term. Pairs were formed within each training batch, using a fixed gap of 0.5 training-target standard deviations. Three inner folds selected the objective using head MSE inside every outer training fold. Only the existing 10% short-head contribution was replaced; candidate013's remaining components stayed fixed.

Across the three existing development partitions, pooled RMSE was {scores['reference013']:.9f} for candidate013, {scores['mse_control']:.9f} for the matched MSE replacement, {scores['rank_fixed_diagnostic']:.9f} for fixed ranking (diagnostic), and {scores['primary']:.9f} for the nested primary. Ranking was selected in {d['chosen_rank_folds']}/15 outer folds. The primary failed the prespecified improvement criterion; candidate013 is retained. No new final500 fit, test inference, CSV, or upload was performed.

Validation reproduced the original SmoothL1 incumbent's parameters and predictions exactly, checked the ranking loss and gradient independently, reloaded all 120 comparison checkpoints, and replayed the full nested fold on deterministic CPU after perturbing outer labels. Both candidate fits were also replayed on CPU after perturbing inner validation labels. The original bit-exact MPS retraining assertion failed only for ranking models and is preserved; unchanged-label MPS repetition confirms small numerical variability within the recorded tolerance, while within-CPU label intervention is exact. An incorrect cross-device training equivalence assertion was also preserved and corrected: CPU and MPS have different dropout RNG streams. No bit-exact MPS retraining claim is made. Same initialization, dropout seeds, batches, update counts, and sample exposures were verified. The three partitions reuse the same400 examples, so their pooled results and conditional ID bootstrap do not remove historical selection bias.

The ranking term was active, but its effect did not reliably improve absolute prediction error. This is evidence against this fixed objective replacement, not all ordinal or distributional methods. No rank-weight, kernel, epoch, or seed sweep follows this result. Detailed metrics, head-only diagnostics, hashes, and limits are in `reports/ranking-objective-120-summary.json`; the frozen protocol is `artifacts/ranking-objective-120/plan.json`.
"""
with readme.open('a') as f:f.write(text)
e.write(_repo_path(ROOT, 'reports/readme-update-120.json'),{'before_sha256':before,'after_sha256':e.sha(readme),'backup':str(backup.relative_to(ROOT)),'reason':'Appendcompleted120 matchedobjectivecomparison andretain013'})
o.check();verification={'passed':True,'013_011_012_preserved':True,'prior_reports':len(old),'packages_unchanged':len(p['environment']),'summary_sha256':e.sha(report),'candidate013_CSV_SHA256':e.sha(_repo_path(ROOT, 'artifacts/submission-candidate-013.csv')),'candidate011_CSV_SHA256':e.sha(_repo_path(ROOT, 'artifacts/submission-candidate-011.csv')),'source_sha256':{name:e.sha(_repo_path(ROOT, name)) for name in ['ranking_objective_120.py','validate_ranking_objective_120.py','summarize_ranking_objective_120.py','finalize_ranking_objective_120.py','validate_ranking_replay_120.py','validate_ranking_replay_v2_120.py']},'new_CSV':False,'full500_test_executed':False,'cost':0,'uploaded':False};e.write(_repo_path(ROOT, 'reports/ranking-objective-120-preservation.json'),verification);e.emit(verification)
