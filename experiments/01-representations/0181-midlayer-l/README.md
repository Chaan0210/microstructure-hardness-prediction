#181 L midpoint versus final readout

rejected_close_L_representation_variants; currentbest021.

044already comparedSblock6/12 withsameRidge; midblend13.5572628869 wasworsefinal13.5224522318.177usedL12/18/24combined,notmidpoint-onlyMLP. L12 informationalreadyexistsin021. Thisoneexperiment isolatesL12 fromL24 withsame3072CLS/patchmean/stdfeatures andsameexisting64hidden3seedMLP. No layer sweep.

Exactly177 existing64hidden/GELU/dropout.25/SmoothL1beta1/100epochs/batch64/AdamWlr.0003cosine.00001/wd.01/clip1/aux0. Same3seeds20261128,20261205,20261219+fold. Sameinit/dropout/sampleorders betweenMID/LAST. TrainlocalStandardScalerandtargetmean/std; noPCA/selection/earlystop. Legacyunusedclassifierhead isidenticalinboth, effectivepredictiveparametercount196737.

MIDblend=.9*fixed018+.1*MIDmean3; LASTblend=.9*same018+.1*LASTmean3.021=.9*018+.1*existingjointL12/18/24mean3. Replacetenpercentbranch only; no additionalblendwith021 orweighttuning.

Original5fold: MIDblend mustimprove021ANDLASTblend by>0.01RMSE each, and>=3/5folds improveeach, allvalidationpasses. Onlythenbothknownresplits unchanged. Proposeonlyif MIDbeatsbothreferences inall3partitions. OtherwisecloseLrepresentationvariants; no LASTfallbackpromotion.

Fresh400rawimagefeaturesmatched177slicesbitexact. CPU/MPS3smoke andbackbonestate/hashchecks passed. Everyhead saved/reloaded,independentGELUforward/CPU-MPS/train-scaler statistics validated. Fold0botharms3seedtrainingreplayedwithoutside-trainfeatures/labelsperturbed; evaluateoriginalqueriesandrequireexactweights/statistics/predictions. Pre-fitvalidationreview sourceandlocks retained. Finalizer separatelyreplayedall30mainheads.

Results:

```json
{
  "original": {
    "018": 12.811945280526919,
    "021": 12.786200775750695,
    "Lmulti": 13.897309666752351,
    "MID": 15.139965837885304,
    "MID_blend": 12.92675058111654,
    "LAST": 15.218594479290314,
    "LAST_blend": 12.808443322487186
  }
}
```

Commands fromprojectroot (completedpaths areexclusive;preserveoutputs):

```sh
.venv/bin/python -B experiments/01-representations/0181-midlayer-l/prepare.py
.venv/bin/python -B experiments/01-representations/0181-midlayer-l/extract.py
.venv/bin/python -B experiments/01-representations/0181-midlayer-l/readout.py original
.venv/bin/python -B experiments/01-representations/0181-midlayer-l/finalize.py
```

No newCSV/full500/holdout/test/install/download/cost/upload. Featurecache,checkpoints,plan/codehashes,validation andresourcetraces retained. Summary:reports/midlayer-L-181-summary.json.
