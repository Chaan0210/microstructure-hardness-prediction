# Independent resource ownership

The degradation-copy pilot belongs to a detached AI session. Its current handover was read from `experiments/03-robustness/20261016-infomatched-pilot/HANDOVER.md`; it says its parent turn stopped at a usage limit but two owned feature processes continue. AI chat discovery found no corresponding active pilot chat, so no unrelated task was messaged.

At23:12 KST, PID92042 was running synth_feats_g.py train_v2a and PID97377 synth_anghcor.py seg v1a. They use the pilot's own rendered copies, Cellpose/U-Net feature pipeline, and output directory. This SSL experiment uses neither those copies nor those outputs and does not resume any pilot stage. It reserves only one sequential root-.venv CPU process with two threads and a1GiB RSS cap. GPU/MPS and pilot files are left untouched. Neither process is terminated or reprioritized.

This record is intentionally in the new experiment directory; no coordination marker is written into another experiment. Pilot progress may continue independently and is excluded from the immutable incumbent inventory. Its results are not read for SSL hyperparameter selection. No claim of bidirectional messaging or pilot completion is made.
