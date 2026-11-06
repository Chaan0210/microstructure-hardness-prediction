"""Complete169 validation logging without rerunning already-audited models.
Original locked driver omitted json import only in final print AFTER report save.
For a not-yet-validated split, inject only json into its namespace before run.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json,sys
import numpy as np
import validate_density_cnn_169 as v
s=v.s;e=s.e

def main(split):
    target=s.OUT/f'validation-{split}.json'
    if not target.exists():v.json=json;v.main(split);return
    s.check();r=e.read(target);assert r['passed'] and r['report_sha256']==e.sha(s.OUT/split/'report.json');assert r['source_sha256']==e.sha(s.R/'validate_density_cnn_169.py');assert r['models_reloaded']==10
    if split=='original':
        a=s.OUT/split/'fold-0-density';b=s.OUT/'heldout-label-replay';ra=e.read(a/'report.json');rb=e.read(b/'report.json');assert ra['average_state_sha256']==rb['average_state_sha256'];assert ra['initial_state_sha256']==rb['initial_state_sha256'] and ra['rng_start']==rb['rng_start'] and ra['rng_end']==rb['rng_end']
        for root,rec in [(a,ra),(b,rb)]:
            for file,key in [('model.pt','model_sha256'),('predictions.npz','predictions_sha256'),('streams.npz','streams_sha256'),('weights.npz','weights_sha256')]:assert e.sha(root/file)==rec[key]
        for file,keys in [('predictions.npz',['train_predictions','validation_predictions']),('weights.npz',['T','y','weights','density','raw_weights']),('streams.npz',['order','rot','flip','lr','loss','unweighted_loss'])]:
            with np.load(_repo_io(a/file)) as aa,np.load(_repo_io(b/file)) as bb:
                for key in keys:np.testing.assert_array_equal(aa[key],bb[key])
    correction={'passed':True,'scope':'Final printing only: lockedvalidator callsjson.dumps withoutimportjson, afterall assertions andcompletedvalidation.jsonwrite. No model/density/loss/metric changes.','original_error_log_sha256':e.sha(s.R/'reports/label-density-169-validation-original.log'),'original_validation_sha256':e.sha(target),'outside_label_replay_saved_states_predictions_weights_streams_exact_rechecked':True,'additional_training':False,'next_rerun_entrypoint':'resume_validation_169.py <split> injectsjsononlyifvalidationnotyetwritten','source_sha256':e.sha(s.R/'resume_validation_169.py')}
    p=s.OUT/'validation-execution-amendment.json'
    if not p.exists():s.write(p,correction)
    print('AUDIT_PASS',json.dumps(r),flush=True);print('LOGGING_RECOVERY_PASS',json.dumps(correction),flush=True)
if __name__=='__main__':main(sys.argv[1])
