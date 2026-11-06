# Study182: fixed phase-harmonic covariance versus amplitude control

Read plan.json and duplicate-audit.json first. Locked custom 4-scale, 8-axial-direction filterbank produces15 phase and15 amplitude covariances. Same residual models, alpha rules, groupfolds and bootstrap seeds. Existing baseline replay is exact. Only development400 image IDs are allowed; model-fitting processes prohibit rawdata opens.

Source order: prepare.py; features.py/synthetic.py; extract.py; readout.py original; validate.py original. Code hashes were recorded before feature extraction and training. Run each stage in a fresh directory when reproducing; outputs refuse overwrite. build_readout.py is generation provenance, not an execution step to repeat over the final readout source.

The initial SciPy multi-worker FFT run failed before producing a cache or fitting models. attempt1-scipy-threadpool retains original sources/locks/logs. backend-amendment.json records the switch to existing NumPy FFT and SciPy worker1 verification, without feature/head/gate changes. Synthetic checks were rerun before development extraction.

See ../../reports/wph-182-summary.json for scores, decisions, validation and resource evidence. No model or CSV is promoted by this folder. The papers concern texture/randomprocess synthesis, not hardness prediction; this small custom subset is not full WPH/Portilla-Simoncelli replication.
