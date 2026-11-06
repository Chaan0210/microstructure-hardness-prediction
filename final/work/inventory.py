"""Enumerate the selected submission's local source/asset closure, without copying it."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACK = HERE.parent / 'package'
H = 'experiments/supervised-renewal-20261018-0625'
F = H + '/final-nonlinear-readouts-20261019-0945'
entries = {}

def add(pattern, role, required=True):
    found = sorted(p for p in ROOT.glob(pattern) if p.is_file())
    if required and not found:
        raise FileNotFoundError(pattern)
    for p in found:
        key = p.relative_to(ROOT).as_posix()
        assert not p.is_symlink(), key
        entries.setdefault(key, {'path': key, 'bytes': p.stat().st_size, 'roles': []})['roles'].append(role)

def files(folder, names, role):
    for name in names.split():
        add(folder + '/' + name, role)

# Existing portable G018 and G021 training/inference implementations.
for d in ['experiments/reproduction-018-176', 'experiments/candidate021-178']:
    add(d + '/*.py', 'upstream_source')
    add(d + '/licenses/*', 'license')
    add(d + '/models/*', 'trained_model')
    add(d + '/config/*', 'configuration', required=False)
    add(d + '/vendor/**/*.py', 'vendor_source', required=False)
    add(d + '/README*.md', 'historical_reproduction')
add('experiments/reproduction-018-176/requirements-current.lock', 'environment')
add('experiments/reproduction-018-176/weights/*', 'pretrained_weight')
add('experiments/dinol-177/dinov2_vitl14_pretrain.pth', 'pretrained_weight')
add('experiments/reproduction-018-176/test-run/test-features.npz', 'inference_feature_cache')
add('experiments/candidate021-178/*-features.npz', 'feature_cache')
add('experiments/candidate021-178/*validation.json', 'historical_reproduction')
add('experiments/candidate021-178/*-provenance.json', 'weight_provenance')
add('experiments/candidate021-178/*.json', 'upstream_configuration')
add('experiments/reproduction-018-176/run/*comparison*.json', 'historical_reproduction')
add('experiments/reproduction-018-176/run/*validation*.json', 'historical_reproduction')

# Original-image segmentation and auxiliary feature learning.
files('experiments/grain-cellpose-220', 'features.py features_v3.py provenance.json', 'upstream_source')
files('experiments/grain-cellpose-220', 'masks-v3-train.npz masks-v3-test.npz features-v3-train.npz features-v3-test.npz', 'frozen_per_image_annotation_cache')
add('experiments/grain-cellpose-220/licenses/*', 'license')
add('experiments/grain-cellpose-220/weights/cpsam', 'pretrained_weight')
files('experiments/push-300', 'ft_cellpose.py seg_ft.py seglib.py degrade.py build_feat.py stcoh.py harness.py pipeline_grain.py finalize_multi.py REPRO-030.md', 'upstream_source')
add('experiments/push-300/feat-a1s250n4vstd-stcoh-*.npz', 'upstream_feature_cache')
files('experiments/push-300', 'ft-a1s250.pt ft-H300.pt', 'trained_auxiliary')
files('experiments/push-300', 'noise_train.npy', 'train_statistic_cache')
files('experiments/push-400', 'phase_unet.py phase_unet2.py phase_unet4.py unet_feats.py unet2_feats.py unet4_feats.py pipeline_final.py make_synth.py make_synth2.py make_synth3.py REPRO-035.md', 'upstream_source')
files('experiments/push-400', 'unet-u1.pt unet2-m1.pt unet2-m2.pt unet2-h1.pt unet2-m3.pt unet4-z1.pt', 'trained_auxiliary')
files('experiments/push-500', 'unet_het.py make_het_sets.py add_z.py REPRO-038.md', 'upstream_source')
files('experiments/push-700', 'angle_feats.py build_sets.py', 'upstream_source')
files('experiments/push-800', 'shade_feat.py add_cols.py REPRO-045.md', 'upstream_source')
files('experiments/push-800', 'make_045.py', 'historical_construction')
files('experiments/push-900', 'make_050.py make_051.py make_053.py', 'historical_construction_with_public_selection_caveat')
files('experiments/push-900', 'realdeg2.py calib_own.py gmm_partner.py', 'upstream_source')
files('experiments/push-1200', 'make_synth4.py synth_unet.py synth_anghcor.py build_rows_g.py synth_feats_g.py unet_feats_sg.py final_aug.py make_061.py', 'upstream_source')
files('experiments/push-1300', 'cumul_ext.py final_h9s2_tab.py final_srcwB.py srcw.py final_nb_partners.py nb_partners.py make_073.py make_074.py make_076.py make_078.py make_079.py', 'upstream_source')
files('experiments/push-1300', 'abl/groups.json', 'configuration')
add('experiments/push-1300/cumul-*.npz', 'measurement_cache')
for v in ['u1dm1sz','u1dm2sz','u1dmavg','u1dmavg4']:
    add(f'experiments/push-500/feat-{v}H2Z-*.npz', 'teacher_feature_cache')
    add(f'experiments/push-700/feat-{v}H2Z-angh-*.npz', 'teacher_feature_cache')
    add(f'experiments/push-800/feat-{v}H2Z-anghcor-*.npz', 'feature_cache')
    for tag in ['deg','deg2','deg3','deg4']:
        add(f'experiments/push-1200/synthrows-{tag}-{v}H2Z-anghcor.npz', 'original_image_transformation_feature_cache')
    for slot in ['038','039','044']:
        for gate in ['nb','full']:
            add(f'experiments/push-1300/abl/feat-h9s2-{gate}-{slot}-{v}-*.npz', 'feature_cache')
add('experiments/tabpfn-v3/weights/tabpfn-v3.5-20260909.safetensors', 'pretrained_weight')

# Grain object set and fitted distribution regression.
files('experiments/probabilistic-grains-900', 'common.py extract_objects.py object-bank-train.npz object-bank-test.npz object-bank-train-provenance.json object-bank-test-provenance.json', 'object_features')
add('experiments/probabilistic-grains-900/base.npz', 'training_context')
add('experiments/probabilistic-grains-900/probabilistic/contrast-adaptive/grain_measurement.py', 'upstream_source')
files('experiments/probabilistic-grains-900/set_model', 'run.py finalize500.py README.md', 'upstream_source')
add('experiments/probabilistic-grains-900/set_model/full500/*', 'trained_set_model')
add('experiments/plateau-break-20261016-2035/incumbent-*.joblib', 'trained_readout')
files('experiments/plateau-break-20261016-2035', 'reproduce.py linear-replay.json neural-replay.json gmm-replay.json', 'historical_reproduction')

# Residual CNN: keep complete original training code, but not unrelated rounds.
add('experiments/rmse-research-20261015/round4-full073/*.py', 'upstream_source')
files('experiments/rmse-research-20261015', 'research.py inputs.npz provenance.json', 'upstream_source')
files('experiments/rmse-research-20261015/round2', 'gmm-features500.npz cumulants.py cumulants-train.npz', 'upstream_source')
add('experiments/rmse-research-20261015/round4-full073/noise-calibration/summary.json','train_only_calibration_evidence')
add('experiments/rmse-research-20261015/round4-full073/final500/teachers/*/predictions.npz', 'cross_fitted_teacher_cache')
add('experiments/rmse-research-20261015/round4-full073/final500/teachers/*/contract.json', 'cross_fitted_teacher_contract')
add('experiments/rmse-research-20261015/round4-full073/final500/teachers/*/*', 'cross_fitted_teacher_training_evidence')
add('experiments/rmse-research-20261015/round4-full073/FINAL500.md', 'configuration')
add('experiments/rmse-research-20261015/round4-full073/final500/*.json', 'configuration')
add('experiments/rmse-research-20261015/round4-full073/final500/models/*/model.pt', 'trained_residual_cnn')
add('experiments/rmse-research-20261015/round4-full073/final500/test-predictions.npz', 'inference_cache')
files('experiments/082-fusion-20261018-0155', 'reproduce.py reproduction/082-inference-verification.json reproduction/incumbent-components.npz', 'historical_reproduction')

# Selected new source encoders, exact readouts and their contracts.
files(H, 'source_h300_v2.py source_flow38_infer.py source_flow38_infer_v2.py source_flow38_heads.py flow38_fast.py domain_semantics.py reproduce_source_flow38.py freeze_source_flow38_v2.py flow_candidate_finalize.py', 'upstream_source')
files(H, 'data.npz source-h300/all-original-flow-targets.npz source-h300/all-original-flow-targets.json source-h300/flow-targets.npz', 'train_annotation_cache')
for tag in ['full500-sourceflow38-s1811','full500-sourceflow38-s1823']:
    files(H+'/source-h300/'+tag, 'model.pt recipe.json trained.json curve.json', 'trained_source_encoder')
add(H+'/candidate-source-flow38-v1/features/*', 'feature_cache')
add(H+'/candidate-source-flow38-v1/models/*/*', 'trained_source_readout')
files(H+'/candidate-source-flow38-v1', 'frozen-recipe.json test-predictions.npz candidate-source-flow38.csv metadata-repair.json csv-verification.json', 'source_candidate_contract')
files(H+'/candidate-flow46-v1', 'training-context.npz test-predictions.npz frozen-recipe.json', 'control_context')
add(H+'/candidate-flow46-v1/models/*control*', 'trained_control_readout', required=False)
add(H+'/environment-*.json', 'environment')
files(F, 'build.py frozen-recipe.json fit-complete.json boostedreadout-verification.json fresh-refit-reproduction.json public-results.json candidate-source-flow38-boostedreadout.csv', 'final_selected_contract')
add(F+'/models/boostedreadout-*.joblib', 'trained_boosted_readout')

# Numeric ancestry evidence; these files are not used to fit any new coefficients.
for n in ['018','021','038','039','044','053','061','073','074','076','078','079','082','083']:
    add(f'artifacts/submission-candidate-{n}.csv', 'historical_arithmetic_reference')
    add(f'artifacts/submission-candidate-{n}.manifest.json', 'historical_construction_manifest',required=False)
for p in ['push-1200/fin-none.npz','push-1200/fin-d1234n12.npz','push-1300/fin-nb-partners.npz','push-1300/fin-srcwB.npz','push-1300/fin-nb-tab-cpu.npz','push-1300/fin-h9-tab-cpu.npz','push-1300/fin-h9s2-tab-cpu.npz','probabilistic-grains-900/ridge/final500/predictions.npz']:
    add('experiments/'+p,'historical_component_cache')

dest=HERE/'dependency-inventory.json'
dest.write_text(json.dumps({'source_root':str(ROOT),'entries':list(entries.values()),'total_bytes':sum(x['bytes'] for x in entries.values())},indent=2)+'\n')
print(json.dumps({'files':len(entries),'GiB':sum(x['bytes'] for x in entries.values())/2**30,'largest':sorted(entries.values(),key=lambda x:x['bytes'],reverse=True)[:12]},indent=2))
