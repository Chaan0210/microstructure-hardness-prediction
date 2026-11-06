"""Frozen arithmetic only: replace the complete historical TabPFN block."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import csv
import json
import hashlib
import datetime
import numpy as np

OUT=Path(__file__).resolve().parent
HERE=OUT.parent
ROOT=_RepoPath(__file__).resolve().parents[4]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path,ids):
    rows=list(csv.DictReader(path.open()))
    assert list(rows[0])==['ID','hardness'] and [r['ID'] for r in rows]==list(ids)
    return np.array([float(r['hardness']) for r in rows])


def main():
    recipe=json.loads((_repo_path(OUT, 'frozen-recipe.json')).read_text())
    assert recipe['coefficient']==.384 and not recipe['promotion_gate_passed']
    ids=np.array([r['ID'] for r in csv.DictReader((_repo_path(ROOT, 'data/sample_submission.csv')).open())])
    assert len(ids)==len(set(ids))==1000
    source_path=_repo_path(HERE, recipe['source_candidate']);assert sha(source_path)==recipe['source_candidate_sha256']
    current=read(source_path,ids)
    control_path=_repo_path(HERE, recipe['control_predictions']);assert sha(control_path)==recipe['control_predictions_sha256']
    controls=np.load(_repo_io(control_path));assert np.array_equal(controls['ids'],ids)
    old_path=_repo_path(ROOT, recipe['incumbent_components']);assert sha(old_path)==recipe['incumbent_components_sha256']
    old=np.load(_repo_io(old_path));assert np.array_equal(old['ids'],ids)
    source=np.load(_repo_path(HERE, recipe['source_predictions']));assert np.array_equal(source['ids'],ids)
    eligible=~old['failed'];assert np.array_equal(eligible,source['eligible']) and np.array_equal(eligible,controls['eligible'])
    assert np.array_equal(current,source['prediction'])
    p79=read(_repo_path(ROOT, recipe['legacy_difference_csvs'][0]),ids);p78=read(_repo_path(ROOT, recipe['legacy_difference_csvs'][1]),ids)
    legacy_tab=old['Tab']+(p79-p78)/.384
    # Independent legacy-block reconstruction from its original deployed bank.
    tabfile=_repo_path(ROOT, 'experiments/push-1300/fin-h9s2-tab-cpu.npz');bank=np.load(_repo_io(tabfile))
    assert np.array_equal(bank['ids'],ids)
    direct_tab=np.where(old['low'],np.mean([bank[f'nb_{c}'] for c in ['038','039','044']],0),
                                   np.mean([bank[f'full_{c}'] for c in ['038','039','044']],0))
    legacy_error=float(abs(legacy_tab[eligible]-direct_tab[eligible]).max());assert legacy_error<1e-10
    pred=current+eligible*.384*(controls['control_tab']-legacy_tab)
    tags=['full500-sourceflow38-s1811','full500-sourceflow38-s1823']
    avg=lambda kind:np.mean([source[tag+'_'+kind] for tag in tags],0)
    independent=source['base083']+eligible*(.384*(avg('tab')-direct_tab)+.128*(avg('raw')-controls['control_linear'])+.128*(avg('log')-controls['control_log']))
    arithmetic_error=float(abs(pred-independent).max());assert arithmetic_error<1e-10
    assert np.isfinite(pred).all() and np.array_equal(pred[~eligible],current[~eligible])
    hashes=[]
    for name in [recipe['candidate'],'candidate-source-flow38-tabreplace-replay.csv']:
        path=_repo_path(OUT, name);assert not path.exists()
        with path.open('x',newline='') as f:
            w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,[repr(float(v)) for v in pred]))
        recovered=read(path,ids);assert np.array_equal(recovered,pred);hashes.append(sha(path))
    assert hashes[0]==hashes[1]
    np.savez(_repo_path(OUT, 'frozen-predictions.npz'),ids=ids,prediction=pred,current_source=current,eligible=eligible,control_tab=controls['control_tab'],legacy083_tab=legacy_tab)
    inputs=[source_path,control_path,old_path,_repo_path(HERE, recipe['source_predictions']),tabfile,*[_repo_path(ROOT, p) for p in recipe['legacy_difference_csvs']]]
    result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),csv_sha256=hashes[0],
        recipe_sha256=sha(_repo_path(OUT, 'frozen-recipe.json')),code_sha256=sha(Path(__file__)),rows=1000,
        schema=['ID','hardness'],ordered_unique_ids=True,finite=True,failure_fallback_unchanged=True,
        legacy_block_reconstruction_max_difference=legacy_error,independent_arithmetic_max_difference=arithmetic_error,
        csv_roundtrip_max_difference=0.,replay_byte_identical=True,
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        current_source_sha256_after=sha(source_path),no_fitting=True,no_test_based_selection=True,
        promotion_gate_passed=False,full_confirmation_passed=False,uploaded=False)
    (_repo_path(OUT, 'verification.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
