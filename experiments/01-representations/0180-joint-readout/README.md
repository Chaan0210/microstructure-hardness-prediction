# Study180: fixed-capacity SB/SBL joint readout

rejected_no_new_candidate; current Public best021 remains unchanged.

Duplicate audit: no prior identical frozen S/B/L joint low-dimensional MLP found. Negative controls060 S-PCA and066 B multiscale concat are retained explicitly in duplicate-audit.json. Prior084 joint means backbone/head finetuning, not feature fusion.177 demonstrated developmentL complementarity, which motivates this one limited comparison.

For each current outertrainT, StandardScaler each originalglobal bank; divideeachbank bysqrt(originalwidth), concatenate inS,B,L order as applicable. Fit PCA64fullSVD whitenFalse onTonly. StandardScaler on64trainingPCA scores, then transformheldout. Botharms input64 with no inactive padding. No target-dependent featureselection or globalPCA. No innerhyperselection because allsettings fixed.

Same2113active parameters, initial tensors, sampleorders, dropoutstream, optimizer and100epochs. Only additionofLbank beforefixed64PCA differs; sharedS/Bscalers exact. No moreseeds/ranks/width/epoch/loss variants.

SBLblend=.9*fixed018OOF+.1*SBLjointmean3; matchedSBblend same. Compare both tofixed021=.9*018+.1*Lglobalmean3. Thisreplacesonly021 tenpercentbranch, notentire018 ensemble; existing018 independentS/Bbranches remain unchanged. No .9*021 blend, ratiosearch or reweightingexisting021.

Runoriginal5fold first. Require SBLblend strictlybetterthan021ANDmatchedSBblend, >=3/5foldgainsvs021, allvalidationpass. Onlythenbothknownresplits unchanged. Proposeonlyifbothcomparisonsimproveonall3partitions. No fallbackpromotion ofSB/control, no posthocblend.

Architecture:64 ->32 GELU ->Dropout0.25 ->1,2113 active trainable parameters;3fixedseeds,100epochs,AdamW0.0003cosine0.00001,wd0.01,batch64,SmoothL1beta1,target trainmean/std. No rank/width/ratio/epoch search. PCA and allscalers fitoutertrain only. No hyperparameter selection loop exists.

Results:

```json
{
  "original": {
    "018": 12.811945280526919,
    "021": 12.786200775750695,
    "globalL": 13.897309666752351,
    "SB": 15.017018560975407,
    "SB_blend": 12.890309020944908,
    "SBL": 15.02316579437874,
    "SBL_blend": 12.880848911580447
  }
}
```

All saved heads were replayed in a second process. PCA eigensystem/manual transforms, saved checkpoint/manual GELU/CPU-MPS outputs, sameinit/order controls andfold0 botharms3seed wholepipeline replays after outside-train labels/features perturbation passed. Failure33 are fixed historicaldiagnostic IDs, never trainingweights/features or selectiongate.

The firstattempt stopped beforeanyfit becauseSfeatures were stored400x3x1152; fixed rowmajorreshape matcheslegacy3456inputs. Originalsource/lock/error/log are retained. Query-relative numeric tolerance appliesonly to artificialverylargeheldoutfeaturepoison, ordinarytransform absolute1e-10 unchanged. No modelrecipe changes.

Executed commands from project root:

```sh
.venv/bin/python -B experiments/01-representations/0180-joint-readout/prepare.py
.venv/bin/python -B experiments/01-representations/0180-joint-readout/study.py original
.venv/bin/python -B experiments/01-representations/0180-joint-readout/finalize.py
```

Completedoutputs useexclusivepaths. Originalpreparation, plan, code hashes, repairhistory, checkpoints and foldreports are retained forreview; do not rerunovercompletedartifacts. All inputs resideintheexistingproject; noportableexternalpackagecreated. Summary:reports/joint-readout-180-summary.json.
