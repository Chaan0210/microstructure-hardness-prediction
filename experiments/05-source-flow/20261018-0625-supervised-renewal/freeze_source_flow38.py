"""Freeze a source-encoder candidate only after its conditional checks finish."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import argparse, datetime, hashlib, json
HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[3]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def main(a):
    assert a.seeds in ([1811], [1811, 1823])
    summary_path = _repo_path(HERE, f"source-h300/proper-summary-s{'-'.join(map(str,a.seeds))}.json")
    summary = json.loads(summary_path.read_text()); result = summary['flow38']
    checks = dict(gain_at_least_point_one=result['gain'] >= .10, four_of_five_folds=result['positive_folds'] >= 4,
                  positive_interval=result['ci95'][0] > 0, positive_influence_sensitivity=result['top10_positive_removed_gain'] > 0,
                  every_encoder_seed_improves=all(v < result['baseline_rmse'] for v in result['per_encoder_seed_rmse']))
    assert all(checks.values()), f'Conditional promotion checks did not pass: {checks}'
    out = Path(a.candidate).resolve(); assert out.is_relative_to(HERE) and out != HERE
    out.mkdir(parents=True, exist_ok=False)
    prior = json.loads((_repo_path(HERE, 'candidate-flow38-v1/frozen-recipe.json')).read_text())
    sources = ['source_h300_v2.py','source_flow38_infer.py','source_flow38_heads.py','flow38_fast.py','domain_semantics.py']
    training_inputs = ['data.npz','train-pixels.npy','source-h300/all-original-flow-targets.npz']
    recipe = dict(status='frozen_conditional_source_candidate_not_selected', name=out.name,
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), field_count=38,
        formula='p083 + eligible*(.384*(mean_encoder(mean3Tab(P104+sourceF38))-mean3Tab(P104)) + .128*(mean_encoder(Ridge100(P104+sourceF38))-Ridge100(P104)) + .128*(mean_encoder(exp(logRidge100(P104+sourceF38)))-exp(logRidge100(P104))))',
        encoder_prediction_weights=[1/len(a.seeds)]*len(a.seeds), encoder_training_seeds=[2026100000+s for s in a.seeds],
        encoder_tags=[f'full500-sourceflow38-s{s}' for s in a.seeds], flow_keys=prior['flow_keys'],
        auxiliary_recipe='Original eligible CPSAM initialization; lower-noise half of the500provided training originals, selected only within training;300updates,effectivebatch4/microbatch1,lr2e-5,warm15,cosinefloor.05,AdamWwd.1,ordinaryoriginalblur/contrast/brightness/noise andcovariantD4weakannotations. No hardness labels in auxiliary fit.',
        initial_weights='experiments/grain-cellpose-220/weights/cpsam',
        initial_weights_sha256='e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2',
        auxiliary_source_sha256=sha(_repo_path(HERE, 'source_h300_v2.py')), descriptor_source_sha256=sha(_repo_path(HERE, 'flow38_fast.py')),
        inference_source_sha256=sha(_repo_path(HERE, 'source_flow38_infer.py')), head_source_sha256=sha(_repo_path(HERE, 'source_flow38_heads.py')),
        source_hashes={name:sha(_repo_path(HERE, name)) for name in sources}, training_input_hashes={name:sha(_repo_path(HERE, name)) for name in training_inputs},
        noise_training_file='experiments/push-300/noise_train.npy', noise_training_sha256=sha(_repo_path(ROOT, 'experiments/push-300/noise_train.npy')),
        control_training_context=prior['training_context'], control_training_sha256=prior['training_context_sha256'],
        control_predictions=prior['control_predictions'], control_predictions_sha256=prior['control_predictions_sha256'],
        physical_test=prior['physical_test'], physical_keys=prior['physical_keys'], baseline_csv=prior['baseline_csv'],
        baseline_sha256=prior['baseline_sha256'], failure_cache=prior['failure_cache'], failure_cache_sha256=prior['failure_cache_sha256'],
        tabpfn=prior['tabpfn'], linear=prior['linear'],
        inference='MPSfloat32C4two-view batches for the auxiliary encoder; bit-equivalent38-field descriptor; all learned preprocessing/readouts fit on500training originals only. CPU nativeTabPFN n8 withseeds0/100/200. Equal prediction averaging across encoders, not feature or parameter averaging.',
        conditional_validation=result, validation_summary_sha256=sha(summary_path), conditional_promotion_checks=checks,
        full_system_protocol_passed=False,
        limitations='All labels have been reused. New encoder/eligibility ranking excludes outer-query originals in validation, but inherited P104/083 representations are historically global. These conditional checks do not complete the predefined all500 matched end-to-end incumbent protocol and are not a Private probability. No new Public result.',
        final_fit_start_cutoff_utc='2026-10-18T19:30:00+00:00', recovery_target_kst='2026-10-19 08:00', official_deadline_kst='2026-10-19 10:00',
        cost=0, no_submission_or_selection_change=True)
    (out/'frozen-recipe.json').write_text(json.dumps(recipe, indent=2))
    print('FROZEN', out.name, json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--candidate', required=True); p.add_argument('--seeds', type=int, nargs='+', default=[1811,1823])
    main(p.parse_args())
