"""Independent CSV reconstruction and small label-free neural replay."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from threadpoolctl import threadpool_limits
import final_model as f


def main():
    z=np.load(_repo_io(f.FINAL/'test-predictions.npz'));recipe=json.loads((f.FINAL/'recipe.json').read_text())
    fit=json.loads((f.FINAL/'fit-complete.json').read_text());inference=json.loads((f.FINAL/'inference-verification.json').read_text())
    assert inference['complete'] and not inference['uploaded']
    ids=z['ids'].tolist();assert len(ids)==len(set(ids))==1000
    assert np.array_equal(z['mean_correction'],z['seed_corrections'].mean(0))
    band=(z['noise']>=12).astype(int);reports={}
    for arm,base in [('addition',z['p073']),('replacement',z['non021'])]:
        pred=base+np.asarray(recipe['production_weights'][arm])[band]*z['mean_correction']
        pred[z['failed']]=z['p073'][z['failed']]
        path=f.FINAL/f'candidate-aux-noise-{arm}.csv';rows=list(csv.DictReader(path.open()))
        assert [r['ID'] for r in rows]==ids
        saved=np.array([float(r['hardness']) for r in rows]);assert np.array_equal(saved,pred)
        assert np.isfinite(saved).all() and f.r.digest(path)==inference['outputs'][arm]['sha256']
        reports[arm]=dict(csv_formula_exact=True,row_count=1000,finite=True,sha256=f.r.digest(path))
    probe=z['active_indices'][:8]
    images=np.stack([np.asarray(Image.open(_repo_path(f.r.ROOT, f'data/test/{ids[i]}.png')),np.float32)/255 for i in probe])[:,None]
    replay=[]
    for j,record in enumerate(fit['models']):
        checkpoint=torch.load(_repo_io(f.HERE/record['model_file']),map_location='cpu',weights_only=True)
        model=f.e.c.ComplementCNN();model.load_state_dict(checkpoint['state_dict']);meta=checkpoint['normalization']
        image_data=((images-meta['image_mean'])/meta['image_std']).astype(np.float32)
        raw=z['context_raw'][probe];raw=np.where(np.isfinite(raw),raw,np.asarray(meta['context_median']))
        context=((raw-np.asarray(meta['context_mean']))/np.asarray(meta['context_std'])).astype(np.float32)
        prediction=f.e.c.predict(model,image_data,context,np.arange(len(probe)),'aux')*meta['target_std']
        error=float(np.max(np.abs(prediction-z['seed_corrections'][j,probe])))
        assert error<1e-4,error
        replay.append(dict(seed=checkpoint['seed'],n_test_probes=len(probe),max_abs_error=error))
    assert f.protected()==recipe['protected']
    f.r.save_json(f.FINAL/'independent-verification.json',dict(all_passed=True,csv=reports,neural_replay=replay,
        original_artifacts_unchanged=True,test_targets_read=False,test_fitting=False))
    print(json.dumps(dict(csv=reports,neural_replay=replay),indent=2),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
