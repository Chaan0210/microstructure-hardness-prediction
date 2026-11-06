"""Same training tensors, one normalization buffer instead of temporary copies."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import gc
import inspect
import json
import shutil
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import stream_cnn as s
p=s.p


def main():
    p.protect();original=s.e.cnn_data;source=inspect.getsource(original)
    needle="images=((d['images']-xm)/xs).astype(np.float32)"
    assert source.count(needle)==1
    replacement="""assert d['images'].dtype==np.float32
    images=d['images'].copy()
    images-=xm
    images/=xs
    for first in range(0,len(images),16):
        reference=((d['images'][first:first+16]-xm)/xs).astype(np.float32)
        assert np.array_equal(reference,images[first:first+16])
    normalization_checks.append(dict(fold=fold,stage=stage,n_images=len(images),pixelwise_exact=True))"""
    derived=source.replace(needle,replacement)
    saved=p.HERE/'lowmem_cnn_data.py'
    text='import numpy as np\nnormalization_checks=[]\n\n'+derived
    if saved.exists():assert saved.read_text()==text
    else:saved.write_text(text)
    namespace={'np':np,'normalization_checks':[]};exec(compile(derived,str(saved),'exec'),namespace)
    revised=namespace['cnn_data']
    rng=np.random.default_rng(2026101635)
    sample=dict(ids=np.array([f'TRAIN_{j:06d}' for j in range(8)]),images=rng.random((8,1,16,16),dtype=np.float32))
    tr=np.arange(5);va=np.arange(5,8);labels=rng.normal(200,15,5);teacher=rng.normal(200,15,8);context=rng.normal(size=(8,6))
    context[1,2]=np.nan
    a=original(sample,tr,va,labels,teacher,[],-1,'fixture',context)
    b=revised(sample,tr,va,labels,teacher,[],-1,'fixture',context)
    for key in ('images','context','train','valid'):assert np.array_equal(a[key],b[key]),key
    for key in a['targets']:assert np.array_equal(a['targets'][key],b['targets'][key])
    assert a['meta']==b['meta'];del a,b,sample;gc.collect()
    backup=p.HERE/'pre-lowmem-verification';backup.mkdir(exist_ok=True)
    for file in (p.HERE/'stream/duplicate-seed1').glob('*/verification.json'):
        dest=backup/(file.parent.name+'.json')
        if not dest.exists():shutil.copyfile(_repo_io(file),_repo_io(dest))
    p.save(_repo_io('lowmem-derivation.json'),dict(source_file=str((p.R4/'cnn_experiment.py').relative_to(p.ROOT)),
        source_sha256=p.r.digest(p.R4/'cnn_experiment.py'),derived_sha256=p.r.digest(saved),
        only_change='In-place image normalization; chunkwise exact equality check on every prepared500-image array.',
        fixture_all_arrays_targets_and_metadata_exact=True,learning_algorithm_changed=False))
    s.e.cnn_data=revised;s.ARMS=('duplicate',)
    s.train(1)
    p.save(_repo_io('lowmem-reproduction.json'),dict(complete=True,normalization_checks=namespace['normalization_checks'],
        existing_fold0_checkpoints_replayed_exactly=True,source_models_preserved=True))


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
