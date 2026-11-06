"""Artifact checks and fixed train-image feature replay; no model selection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os
import sys
HERE=Path(__file__).resolve().parent
ROOT=_RepoPath(__file__).resolve().parents[3]
sys.dont_write_bytecode=True
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='2'
for key in ['TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME','CELLPOSE_LOCAL_MODELS_PATH']:
    os.environ[key]=str(_repo_path(HERE, 'tmp'))
os.environ['HF_HUB_OFFLINE']='1'
import csv
import datetime
import fcntl
import gc
import json
import numpy as np
import torch
from cellpose.vit import CPSAM
from domain_semantics import descriptors,normalize,sha


def main():
    out=_repo_path(HERE, 'candidate-source-flow38-v1')
    assert (out/'replay-verification.json').exists()
    recipe=json.loads((out/'frozen-recipe.json').read_text())
    repair=json.loads((out/'metadata-repair.json').read_text())
    assert repair['frozen_recipe_sha256']==sha(out/'frozen-recipe.json')
    assert repair['new_inference_sha256']==sha(_repo_path(HERE, 'source_flow38_infer_v2.py'))
    rows=list(csv.DictReader((out/'candidate-source-flow38.csv').open()))
    sample=list(csv.DictReader((_repo_path(ROOT, 'data/sample_submission.csv')).open()))
    assert len(rows)==1000 and list(rows[0])==['ID','hardness']
    ids=np.array([r['ID'] for r in rows]);p=np.array([float(r['hardness']) for r in rows])
    assert len(set(ids))==1000 and np.array_equal(ids,[r['ID'] for r in sample]) and np.isfinite(p).all()
    saved=np.load(_repo_io(out/'test-predictions.npz'));assert np.array_equal(saved['ids'],ids) and np.array_equal(saved['prediction'],p)
    controls=np.load(_repo_path(HERE, recipe['control_predictions']))
    assert sha(_repo_path(HERE, recipe['control_predictions']))==recipe['control_predictions_sha256']
    assert np.array_equal(p[~controls['eligible']],controls['base083'][~controls['eligible']])
    assert sha(out/'candidate-source-flow38.csv')==sha(out/'candidate-source-flow38-replay.csv')
    px=np.load(_repo_path(HERE, 'train-pixels.npy'),mmap_mode='r');data=np.load(_repo_path(HERE, 'data.npz'))
    checks=[]
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert torch.backends.mps.is_available();torch.mps.set_per_process_memory_fraction(.3)
    for tag in recipe['encoder_tags']:
        source=_repo_path(HERE, 'source-h300', tag)
        ck=torch.load(_repo_io(source/'model.pt'),map_location='cpu',weights_only=False,mmap=True)
        assert ck['recipe']['train']==list(range(500)) and ck['recipe']['query']==[]
        assert len(ck['recipe']['source_indices'])==250
        feature=np.load(_repo_io(out/f'features/{tag}-train.npz'))
        assert np.array_equal(feature['ids'],data['ids'])
        model=CPSAM();state={k:v for k,v in ck['state'].items() if k not in ['diam_labels','diam_mean']}
        status=model.load_state_dict(state,strict=False)
        assert set(status.missing_keys)<={'diam_labels','diam_mean'} and not status.unexpected_keys
        del state,ck;gc.collect();model.eval().requires_grad_(False).to('mps')
        capture={};handle=model.encoder.neck.register_forward_hook(lambda module,args,value:capture.update(neck=value))
        with torch.no_grad():
            for j in [0,249,499]:
                values=[]
                for first in [0,2]:
                    raw=px[j,0]
                    batch=np.stack([np.ascontiguousarray(np.rot90(normalize(raw),rot)) for rot in [first,first+1]])[:,None]
                    flow,_=model(torch.tensor(batch,device='mps'))
                    neck=capture.pop('neck').cpu().numpy();flow=flow.cpu().numpy()
                    for k,rot in enumerate([first,first+1]):
                        _,fields,keys=descriptors(neck[k],flow[k],np.rot90(raw,rot))
                        assert keys[:38]==recipe['flow_keys'];values.append(fields[:38])
                difference=float(abs(np.mean(values,0)-feature['flow'][j]).max())
                assert difference<1e-5
                checks.append(dict(encoder_tag=tag,training_id=str(data['ids'][j]),feature_max_difference=difference))
                torch.mps.empty_cache()
        handle.remove();del model,feature;gc.collect();torch.mps.empty_cache()
    result=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete=True,
        rows=1000,unique_ordered_ids=True,schema=['ID','hardness'],finite=True,
        csv_sha256=sha(out/'candidate-source-flow38.csv'),recipe_sha256=sha(out/'frozen-recipe.json'),
        cache_to_csv_exact=True,replay_csv_byte_identical=True,fallback_unchanged=True,
        fixed_train_image_feature_checks=checks,parameters_updated=0,quality_metrics_or_selection=False,
        protected_boundary_record='../final-protected-boundary-verification.json',full_system_protocol_passed=False)
    (out/'final-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    with (_repo_path(HERE, 'tmp/heavy-learning.lock')).open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX);main()
