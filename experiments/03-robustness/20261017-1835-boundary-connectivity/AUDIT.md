# Boundary and connectivity audit before fitting

Recorded incumbent082 Public8.9343098075;076 rollback8.9646534604. All previous studies immutable. No unrelated Python/GPU jobs were running at09:25 UTC on17 October2026.

- push-400/phase_unet.py learns four semantic classes from Cellpose clean-image masks, including boundary, using local corruptions. Its output statistics are mostly class fractions. It does not optimize connectivity between arbitrary same-grain pixel pairs. push-1300/build_u1b.py substitutes three phase columns; NOTES §26 records high-noise12.31->12.62, rejected.
- push-900/setZ/graphset.py uses one K6 centroid-neighbour message-passing round on a frozen object bank, then area-weighted pooling and hardness regression. It never learns raw-pixel affinities or a partition-critical edge loss.
- connected_grain_features.py links fixed dark ridges, counts enclosed components/skeleton endpoints. It is an image-local engineered descriptor, not an image-trained affinity network.
- denoised_grain_features.py derives TV/gradient-watershed statistics. Prior tessellation-diagnostic-183 explicitly audits matched-cell/adjacency instability; these masks are hypotheses, not truth.
- topology-H1-166 is a review, with older H0/Betti/C2 descriptors already failed. It is not a completed maximin-affinity training experiment. We do not repeat threshold-count columns alone.
- structure-context-20261017-1415 predicts coarse grayscale fields. It learned DC43.1%,centered2.5%,gradient0.9% improvements on high-noise proxy targets but no incremental hardness signal. PCA/PLS selection also failed. Its intensity restoration and pooled representation are the immediate contrast.
- Source search in non-vendor project Python/Markdown/text found no affinity/maximin/MALIS/clDice grain-training implementation. This is a scoped audit, not a claim that all topology methods are new.

Chosen mechanism: supervise dense boundaries AND same-grain pixel-pair connectivity. Compare ordinary local affinity loss with a sampled maximin path loss, which routes gradients to the edge that actually merges/separates two graph components. The connection objective differs from predicting every edge independently and from forcing hardness-neighbour images together.

Concept reference: Turaga et al., Maximin affinity learning of image segmentation, NeurIPS2009, https://arxiv.org/abs/0911.5372 . We implement and test a small sampled variant with weak mask confidence, not the paper's original data/model or a claimed exact reproduction. Only the conceptual paper was read; no code, images, or pretrained weights were downloaded. clDice was reviewed but is not included: adding skeleton and maximin objectives together would obscure attribution.

Reliability concern: existing Cellpose base/TTA masks are auxiliary predictions, not true grains, and can share hallucinated boundaries. First inspect source-local consensus and raw boundary evidence without hardness labels. Store unknown targets as unknown; do not turn missing masks into different-grain labels. High-noise and low-noise coverage, disagreement, withheld-view agreement, and physical edge support must be reported. A prior-confidence audit cannot prove physical correctness.
