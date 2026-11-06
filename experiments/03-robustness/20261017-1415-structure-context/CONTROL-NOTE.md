# Interpretation of the shuffled-layout control

Recorded before any hardness readout/outer score. S and P have identical original D4 image views, initialization, sample order and masks on the encoder's16x16 token grid. P permutes macro tokens and their local reconstruction targets before applying that grid mask. Consequently, the physical source patches hidden by the mask differ between S and P. P retains every16x16 local patch and its within-patch target structure, but destroys their arrangement and changes which physical neighbors remain visible.

This is a comparison of two complete structural pretraining procedures, not a pure causal intervention on topology with every visible source pixel held identical. Geometry diagnostics must be reported relative to each arm's own constant-field reference as well as in absolute error units. The full-image readout is unmasked for both arms. No protocol, input, weight or promotion criterion changes in this note.
