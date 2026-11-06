# Mechanism controls, registered after the first screen before their scores

The image-only residual arm passed the compute-allocation screen. Before
attributing its gain to learned spatial CNN features, compare two controls:

1. Stats-only: the same initialized correction MLP and optimizer sees only the
   input image's mean/std; all convolution-derived slots and context slots are
   exactly zero. It shares all normalization, residual targets, seed, D4 order,
   40 epochs and .25 shrinkage. No learned spatial image features can contribute.
2. Constant-only: .25 times the outer train's mean honest inner-OOF residual,
   added to the outer tabular prediction. No validation labels select it.

These are mechanism/complexity controls, not preselected improved candidates.
Run every control on every executed partition. They must not change the existing
CNN fits or the confirmation decision. Preserve their outputs separately.
