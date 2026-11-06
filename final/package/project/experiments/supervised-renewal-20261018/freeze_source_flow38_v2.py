"""Add the newly reported Public best comparator without changing old gates."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import freeze_source_flow38 as original

HERE = Path(__file__).resolve().parent


def main(a):
    assert a.seeds == [1811, 1823], 'Both fixed encoder seeds are required.'
    assert datetime.datetime.now(datetime.timezone.utc) <= datetime.datetime.fromisoformat('2026-10-18T19:30:00+00:00')
    comparison_path = HERE/'source-h300/current-best-comparison-s1811-1823.json'
    comparison = json.loads(comparison_path.read_text())
    summary = json.loads((HERE/'source-h300/proper-summary-s1811-1823.json').read_text())
    assert comparison['encoder_seed_suffixes'] == a.seeds and comparison['rows'] == 400
    assert comparison['source_candidate_rmse'] == summary['flow38']['candidate_rmse']
    checks = dict(
        every_encoder_seed_improves=all(v < comparison['current_best_rmse'] for v in summary['flow38']['per_encoder_seed_rmse']),
        positive_gain=comparison['gain'] > 0,
        four_of_five_folds=comparison['positive_folds'] >= 4,
        positive_interval=comparison['ci95'][0] > 0,
        positive_influence_sensitivity=comparison['top10_positive_removed_gain'] > 0,
    )
    assert all(checks.values()), f'Current-best final-fit resource checks did not pass: {checks}'
    original.main(a)  # Original historical083 conditional gates remain necessary.
    out = Path(a.candidate).resolve()
    path = out/'frozen-recipe.json'
    recipe = json.loads(path.read_text())
    recipe.update(current_best_comparison=comparison, current_best_final_fit_resource_checks=checks,
                  current_best_comparison_sha256=original.sha(comparison_path),
                  final_fit_rule_sha256=original.sha(HERE/'source-final-fit-resource-rule.json'),
                  freeze_factory_sha256=original.sha(Path(__file__)),
                  recommendation_scope='Separate exploratory candidate. These resource checks do not pass the original all500 complete end-to-end confirmation or establish a reliable Private improvement over the user-submitted globalF38 model.')
    path.write_text(json.dumps(recipe,indent=2)+'\n')
    print('CURRENT_BEST_CHECKS_RECORDED',json.dumps(checks),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[1811,1823])
    main(p.parse_args())
