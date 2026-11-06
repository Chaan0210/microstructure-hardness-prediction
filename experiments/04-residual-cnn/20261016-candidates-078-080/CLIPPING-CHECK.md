# Additional diagnostic, registered before refitting

The source audit found a concrete mismatch: 078's dev H9 partner predictions
did not clip the two added features, while its full500 inference clips test
features to the train range. Quantify this with exactly the original three
400-image partitions and original partner training/alpha rules. Refit on
each outer training subset and predict the same outer rows both unmodified
and clipped to that subset's feature range. No thresholds are tuned.

First reproduce the saved unmodified H9 predictions to verify alignment and
recipe equivalence. Then report clipping's incremental score and number of
affected rows. This is a diagnostic, not permission to choose a clipping
threshold from outer scores or to alter numbered078. Existing076 OOF remains
historical as-produced; do not call this a fully refitted076 benchmark.

Run sequentially after the current single-process CNN training. Test inputs
and Public metrics are not used in these fits. No corrected078 candidate is
promoted merely because clipping changes its local score.
