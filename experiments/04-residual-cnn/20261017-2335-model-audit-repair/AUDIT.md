# Before fitting: audit findings and exclusions

082 is verified recorded Public best8.9343098075,0768.9646534604.083 has not
been submitted: the last upload attempt stopped before attachment. No current
Python training process was found; unrelated browser processes were untouched.
Initial154570-file checksum verification succeeded. GitHEAD and the existing
dirty/untracked state are recorded without modifying Git.

## Trace of the deployed learning problem

082 =076 + gated old residual CNN.076's normal-row assembly is
.16G021+.384H9Tab+.128Ridge+.128logRidge+.05set+.15GMM. Failure rows returnG021.
The CNN target is y−T073, whereT073 omitsG021 and renormalizes by.84; it is
not the residual of the final deployed082/076 ensemble. `round4-full073/
full_branch.py` creates source-excluded teachers and `cnn_experiment.py`
constructs6context features. `cnn_fitter.py` uses source-train image/target/
context normalization, D4,40epochs and final5state average. No obvious ID/order,
inverse-scaling or train/query label leak was found in this constructor.

The teacher target changes across training stages:142-row teachers provide
selector213-source targets;213/214 teachers provide320-source targets; deployed
teachers use larger contexts. This is an actual train/inference distribution
difference, not proof of material harm. Teacher073 vs deployed076 is a second,
separate discrepancy. The old T076 pilot passed its declared2fold gate, but
remaining teachers were halted at70min;27/30nodes and36members ofnode28 were
complete. Completing it is appropriate; it cannot be described as independent
confirmation of the already inspected2fold pilot.

Large grain-set errors do exist (e.g.TRAIN_000019,644.99HV in a142-source teacher),
but the weighted final teacher has a segmentation-failure fallback. The new
audit explicitly reconstructs that fallback before attributing downstream
harm. A large raw-component error is not automatically a large CNN-target error.
The previous infogain report also measured negligible final082 effect from
removing/clipping the set component. Do not repeat a global clip sweep.

The log partner is fitted on log labels and exponentiated. Geometric averaging
has a different statistical target from an arithmetic conditional mean. This
is intentional historical behavior, not a coding typo. A single nested
smearing test and matched offset control can falsify its practical importance.
Its OOF calibration is fitted with smaller source contexts than final refit;
that limitation must remain explicit. No oracle calibration of outer residuals.

The residual CNN starts with7x7 stride4, then strides2,2,2. Only means/SDs are
retained from feature maps. Our dense-first-block control has exactly the same
trainable tensors and scalar loss; comparison of dense subsampling vs dense
average pooling controls for the changed normalization/computation. It tests
sampling behavior, not correctness of physical grain annotations.

## Prior experiments checked, and why they are not repeated

- `078-080-update-20261016/REPORT.md`: high-noise-only residual loss
  (all-source hardness auxiliary) lost.0164 against the matched1seed control;
  deleting teacher/context inputs lost.1118. Do not rebrand either as new.
- `076-complement-20261016/REPORT.md`: Gaussian input channel did not beat
  a duplicated raw channel; calibration ordering gains failed role reassignment.
- `infogain-20261017/REPORT.md`: image mean added to TabPFN gained only
  .0067 and failed influence robustness; explicit spectral linear residuals
  failed. No full mean-column block expansion is being run.
- `plateau-break-20261016-2035/REPORT.md`: class-weight posterior repair improves
  weak-mask accuracy, not hardness; physical/spatial kernels fail. MPS automatic
  precision mismatch is real but too small to explain the plateau.
- `highnoise-relational-*`, `highnoise-ssl-*`, `structure-context-*` and
  `boundary-connectivity-*`: neighborhood loss, same-image VICReg, masked
  intensity recovery and weak-mask connectivity did not earn advancement.
  The last connectivity implementation also had duplicate-gather backward
  nonrepeatability; its separate deterministic fix has not received newOOF.
  It is not part of the incumbent CNN, so fixing it is not credited to082.
- `degradation-pilot-audit-*`: new copies did not beat source duplication;
  the shuffled-label control leak was repaired in that separate audit.

Some older reports interpret shared errors, stable correlations or low grain
counts as a sampling floor. Those observations cannot identify an irreducible
error floor without repeated truth measurements or stronger independent evidence.
We do not use them to stop investigation or claim no remaining image signal.

All labels have prior exposure; no untouched holdout is available. Every new
method is evaluated as a conditional improvement over the preserved pipeline,
with new fitted stages restricted to their assigned source partitions.
