# Supervised renewal — 2026-10-18

Authorized scope: this directory only, local CPU/MPS, existing packages, no spending, no external mutations. Protect 083, 082, 076 and every shared dependency. Existing complete preservation/reproduction manifests are referenced in baseline-protection.json; do not repeat whole-project hashes.

Verified incumbent: 083, Public 8.9179249297, submission 57 at 2026-10-18 06:11 KST. Exact formula: 076 + (round4 aux-noise addition − 073) + (079 − 078). 082 is the first two terms (Public 8.9343098075); 076 Public 8.9646534604. Record source: experiments/submissions/20261018-060706-candidate-083/result-state-final.txt. Numbers are observations, never coefficient-fitting inputs.

Official page: https://daker.ai/public/hackathons/synthetic-microstructure-property-prediction-ai . Submission closes 2026-10-19 10:00 KST (page clock/countdown checked). Q&A dated Oct 16 permits original-image selected-region brightness/contrast and weakening selected boundaries by mixing/replacing with same-image neighbor pixels. Separate rendered synthetic images and test pseudo-labels forbidden. Initial experiments use only original train images and D4. No external pretrained models in the initial branch.

Execution window: exploratory work until Oct 18 22:00 KST; confirmation/recovery until Oct 19 04:00; final fitting/inference/verification reserved 04:00–08:00, remaining two hours for recovery and user review. Stop launching work whose benchmarked completion violates these reserves. Disk reserve >=15 GiB; one MPS trainer, CPU 2 threads, monitor combined process/MPS memory and stop only this job if it exceeds 6 GiB. Earlier completion is allowed when evidence resolves the decision; no arbitrary hypothesis count.

## Diagnosis and first experiment

Prior SSL and relational experiments used 40–48-epoch small CNNs; MSE control training RMSE remained about 13.17. The masked Transformer used a frozen reconstruction representation and PCA/PLS. Fine-tuned EfficientNetV2-S used crops, photometric jitter and auxiliary physical measurements, so its failure does not isolate direct supervised full-image training. Previous spectral Ridge replacement tested a linear function of 23 power statistics, not learned nonlinear texture interactions. None proves a ceiling.

Test a substantially stronger direct hardness learner: residual multiscale image encoder, full 256 images, mean/std and learned second-order feature pooling; compare a convolution-plus-attention variant. Random initialization, no pseudo masks or teachers. Preserve brightness information with explicit raw moments. Train sufficiently long, save learning curves and checkpoints; inspect training/validation curves on screening folds to diagnose optimization versus overfitting. Screening may be adaptive and is not confirmation.

Initial screen uses historical original fold0 (T320/V80); fold1 is the next screen for promising settings. Assess replacement prediction and a fixed 0.25 candidate / 0.75 incumbent blend as a complementarity diagnostic. This fixed blend is not a fitted stack. Do not present screening-selected epochs as unbiased OOF. Full five-fold conditional development comparison only after choosing a recipe; use 083 (and separately 082/076) on identical images.

## Confirmation protocol (fixed before finalist results)

For an advanced recipe, freeze architecture, preprocessing, epochs, optimizer, seeds and combination rule before confirmation. Use two repetitions of 5-fold splits of all 500 training images (KFold seeds 2026101801 and 2026101802), two fixed initialization seeds 2026101811 and 2026101823, equal averaging. Original and any derivatives share a split. Entire new image representation is learned only inside the training fold. Any stacking uses inner cross-fitting with preprocessing and representations fit inside inner training, never repartitioned OOF from models exposed to outer validation. No test data read until final freeze.

A credible improvement requires >=0.10 overall RMSE gain over the complete matched incumbent, improvement in both repetitions and >=7/10 folds, a positive lower 95% paired image-ID bootstrap bound, and positive gain after excluding the 10 largest positive SSE contributors as sensitivity only. Difficult images remain in the primary score. Also report noise bands <9.5/9.5–12/>=12, target tertiles, morphology tertiles, seed stability and errors. Bootstrap groups repeated observations by original ID, and remains conditional on reused data; it is no proof against adaptive selection or a Private probability.

Rebuild the complete incumbent at matching training sizes/eligibility for decisive comparisons. Existing shared-representation OOF is useful only as explicitly conditional screening evidence. If full incumbent reconstruction cannot be completed in time, do not relabel conditional evidence as end-to-end confirmation. A standalone challenger may still be fitted and delivered with this limitation and no replacement recommendation.

Deliver the strongest supported frozen challenger separately, even if exploratory, with full500 weights, inference-only test CSV, exact recipe and reproduction command. No upload, submission attempt or final-selection change. Do not claim a breakthrough unless measured.
