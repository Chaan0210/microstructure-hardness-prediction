# Decision — 2026-10-16

**User approval (chat):** "Approve 082 as the recommended final candidate. I will submit it myself; keep 076 as the rollback."

| role | file | SHA-256 | Public |
|---|---|---|---|
| recommended final candidate | `artifacts/submission-candidate-082.csv` | `0fa34cec1626b0887d3d7971197207c135a447a8f4805096b258e77b912f0e9f` | not yet submitted; arithmetic reading **8.9343** |
| rollback (protected incumbent) | `artifacts/submission-candidate-076.csv` | `b1be09b6c060de1a92191bbf31f76f1c501a126493976771f335d7ff3242e3f0` | 8.9646534604 (#53) |

- **Composition.** 082 = 076 + round4 aux-noise residual-CNN correction, applied only on the 318 gated test rows (noise ≥ 12 and not failed).
  - It is identical to the AI file `experiments/04-residual-cnn/20261016-candidates-078-080/candidate-076-plus-frozen-cnn.csv` (max diff 0.0).
- **Pre-submission checks** (`record/validate-082.json`):
  - 1000 rows, IDs in exact sample order, all values finite (153.7–272.7).
  - SHA matches `artifacts/submission-candidate-082.manifest.json`.
- **Submission sanity check.**
  - The Public score after upload should read **8.9343** (to about ±0.0001), because the correction rows do not overlap the 078 change: √(8.9647² + 8.9998² − 9.0299²).
  - Any other value means a different file was uploaded.
- **Final selection.** The page is in auto mode (final = Public best), so once 082 shows 8.9343 it becomes the final selection automatically.
  - To roll back, select 076 manually as the final file.
  - The dev gain reverses if its top 20 images are removed (REPORT.md §5).
- **At approval time** nothing had been submitted (the submission is recorded below). No file was renamed or overwritten by this session. 076 and its full reproduction pipeline remain protected and verified (INCUMBENT.md).

## Submission (2026-10-16 21:25 KST)
- The user then instructed: "go ahead and submit 082 yourself". Submitted as **#56** through the built-in browser.
- The bytes were rebuilt in-page and verified before upload: SHA-256 `0fa34cec…` matched for both the rebuilt CSV and the attached file.
- **Public 8.9343098075.** This equals the arithmetic prediction 8.93430980744, so the correct file was uploaded.
- Rank 5 (#4 8.8662, leader 8.5843). Quota 3/5 used today.
- Auto mode now selects 082 as the final file. No manual selection was made. To roll back: select #53 (076) as the final file.
- Record: `reports/submissions/candidate082-public-submission.json`.

## Re-test of the correction (2026-10-17 01:50 KST; `experiments/04-residual-cnn/20261017-rescnn-retest/REPORT.md`)
- Pre-registered verdict: **SUPPORTS 082**. On the two other official partitions with fresh nested teachers and the frozen recipe, the combined gated dMSE is −5.7 ± 3.0 (P 0.97). RMSE gains are +0.116 and +0.070, vs +0.209 on the original partition.
- Caveats:
  - The gain is concentrated in about 10 images.
  - On the holdout-100 (35 gated rows) the correction did not help: dMSE +6.2 ± 7.0, P(better) 0.18.
  - Fresh-row evidence (Public + holdout) roughly cancels. The expected private difference 082 vs 076 is small (about −0.03 to +0.02 RMSE).
- **Final stays 082 (rule-based). 076 (#53) remains the rollback.** No selection change was made.

## User decision (2026-10-17)
- "추가 제출은 하지 말고 082를 최종으로 유지해." No further submissions. 082 stays the final file. The page is in auto mode (final = Public best = 082, #56), so no selection change is needed or made. 076 (#53) remains the rollback.
- 083 (082 + σ2/σ4 cumulant columns) was considered and not submitted. Dev evidence: +0.05 on all 3 partitions. Holdout-100 contrary (+3.4 ± 4.1 dMSE). A Public reading against 082 would have SE ≈ 0.028 RMSE.
