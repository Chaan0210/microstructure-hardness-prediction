"""Prediction/code/model lineage checks only; no test fitting or new submission."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys
sys.dont_write_bytecode=True
import csv
import json
import numpy as np
import study as s


def main():
    R=s.R4;ROOT=s.ROOT
    recipe=json.loads((R/'final500/recipe.json').read_text())
    fit=json.loads((R/'final500/fit-complete.json').read_text())
    inference=json.loads((R/'final500/inference-verification.json').read_text())
    proof=json.loads((R/'final500/independent-verification.json').read_text())
    assert proof['all_passed'] and fit['complete'] and inference['complete']
    assert s.r.digest(R/'final_model.py')==recipe['code_sha256']
    assert s.r.digest(R/'full500_fitter.py')==recipe['fitter_sha256']
    assert s.r.digest(R/'final500/calibration-frozen.json')==recipe['calibration_frozen_sha256']
    paths=[R/'final_model.py',R/'full500_fitter.py',R/'final500/recipe.json',R/'final500/fit-complete.json',R/'final500/independent-verification.json']
    for item in fit['models']:
        path=R/item['model_file'];assert s.r.digest(path)==item['sha256'];paths.append(path)
    for name,digest in inference['input_sha256'].items():assert s.r.digest(ROOT/name)==digest,name
    for name in ('021','073','078'):
        paths.append(_repo_path(ROOT, f'artifacts/submission-candidate-{name}.csv'))
    s.protect(paths)
    def read(name):
        rows=list(csv.DictReader((_repo_path(ROOT, f'artifacts/submission-candidate-{name}.csv')).open()))
        return [x['ID'] for x in rows],np.array([float(x['hardness']) for x in rows])
    ids,p21=read('021');i73,p73=read('073');i78,p78=read('078');assert ids==i73==i78
    z=np.load(_repo_io(R/'final500/test-predictions.npz'));assert z['ids'].tolist()==ids
    T73=(p73-.16*p21)/.84;T73[z['failed']]=p73[z['failed']]
    T78=(p78-.16*p21)/.84;T78[z['failed']]=p78[z['failed']]
    error=float(np.max(np.abs(z['context_raw'][:,0]-T73)));assert error==0.
    high=(z['noise']>=12)&~z['failed'];std=recipe['normalization']['context_std'][0]
    result=dict(all_lineage_hashes_match=True,n_original_inference_inputs_checked=len(inference['input_sha256']),
        model_seeds=recipe['seeds'],coefficient_vectors=recipe['production_weights'],
        learned_target=recipe['normalization']['training_objective'],actual_CNN_context_base='T073',applied_additive_baseline='078',
        context073_exact_error=error,context078_minus_context073_high_rms=float(np.sqrt(np.mean((T78[high]-T73[high])**2))),
        standardized_context_shift_high_rms=float(np.sqrt(np.mean(((T78[high]-T73[high])/std)**2))),
        interpretation='Confirmed deliberate transport of a073-conditioned correction. This is not a proven assembly bug or a demonstrated cause of the Public result.',
        test_fitting=False,test_counterfactual_predictions_generated=False,public_inverse_used=False)
    s.save(_repo_io('version-chain.json'),result);s.check_protected();print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
