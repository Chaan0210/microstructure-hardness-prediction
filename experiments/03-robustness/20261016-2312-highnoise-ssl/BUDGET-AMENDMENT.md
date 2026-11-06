# A1: timing-only budget amendment, before outer scores

Recorded at approximately23:26 KST on16 October, while fold0 refit controls were running. No outer-fold score, error, blend coefficient or candidate comparison has been read. First selector phase times were37–39 seconds; the320-source refit phases take56–58 seconds. The fixed60-phase matrix therefore projects to about48 minutes, plus a possible80-second deterministic replay.

The one-time revised neural-training cap is **55 minutes including replay**, with the existing90-minute active-work cap, one CPU process/two threads and1GiB RSS cap unchanged. No variants, seeds, strengths, losses, epochs, fitted parameters, success criteria or other scientific choices are added or changed. If55 minutes is insufficient, report the incomplete matrix without promotion; no further extension.

The already-running study.py retains its original45-minute check. If it reaches that limit, it exits between arms. `resume.py` then completes only the remaining preregistered arms, accounting for the actual seconds of all previously completed fits. It refuses partial/ambiguous output directories. It uses the exact frozen training function with only completion skipping and the budget constant changed. This avoids replacing the code while it is running or overwriting any checkpoint. If the original run finishes, no resumption occurs.

This is a disclosed execution-budget change, not adherence to the initial45-minute estimate. The revised limit is fixed before model comparison and cannot be extended in response to accuracy.
