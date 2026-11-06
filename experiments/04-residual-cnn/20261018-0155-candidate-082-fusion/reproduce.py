"""Fresh076 assembly and082 saved-CNN inference, with isolated destinations."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import os,sys,ast,json,csv,hashlib
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3];EXP=_RepoPath(__file__).resolve().parents[2];OUT=_repo_path(HERE, 'reproduction')
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
import numpy as np,torch
from torch import nn
from PIL import Image
from threadpoolctl import threadpool_limits

def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).absolute();mode,flags=args[1:]
        writing=(mode is not None and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing and not p.is_relative_to(HERE):raise RuntimeError('Outside experiment write: '+str(p))
        if not writing and (p==_repo_path(ROOT, 'data/train.csv') or p.is_relative_to(_repo_path(ROOT, 'data/train'))):raise RuntimeError('No training labels/pixels during incumbent inference')

def main():
    OUT.mkdir(exist_ok=False);sys.addaudithook(guard)
    source=_repo_path(EXP, 'plateau-break-20261016-2035/reproduce.py');tree=ast.parse(source.read_text())
    names={'sha','track','save','pack','readcsv','nz','csvout','cached'};nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns=dict(Path=Path,HERE=OUT,ROOT=ROOT,EXP=EXP,USED={},np=np,json=json,csv=csv,hashlib=hashlib)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns);ns['track'](source);ns['cached']()
    C=ns['nz'](_repo_path(OUT, 'incumbent-components.npz'));ids=C['ids'];base76=C['p76'];high=C['high'];failed=C['failed']
    r4=_repo_path(EXP, 'rmse-research-20261015/round4-full073');source=r4/'cnn_fitter.py';tree=ast.parse(ns['track'](source).read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['ComplementCNN','predict']]
    cs=dict(torch=torch,nn=nn,np=np);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),cs)
    saved=ns['nz'](r4/'final500/test-predictions.npz');assert np.array_equal(saved['ids'],ids);raw=saved['context_raw']
    recipe=json.loads(ns['track'](r4/'final500/recipe.json').read_text());weights=np.array(recipe['production_weights']['addition']);assert weights[0]==0 and abs(weights[1]-.5467904360508202)<1e-15
    active=np.flatnonzero(high);images=[]
    for j in active:
        with Image.open(_repo_io(ns['track'](_repo_path(ROOT, f'data/test/{ids[j]}.png')))) as im:
            assert im.mode=='L' and im.size==(256,256);images.append(np.asarray(im,np.float32)/255.)
    images=np.stack(images)[:,None];corr=np.zeros((2,1000))
    for seed in [0,1]:
        ck=torch.load(_repo_io(ns['track'](r4/f'final500/models/aux-seed{seed}/model.pt')),map_location='cpu',weights_only=True);meta=ck['normalization'];assert meta==recipe['normalization']
        model=cs['ComplementCNN']();model.load_state_dict(ck['state_dict']);im=((images-meta['image_mean'])/meta['image_std']).astype(np.float32)
        clean=np.where(np.isfinite(raw),raw,np.array(meta['context_median']));ctx=((clean-np.array(meta['context_mean']))/np.array(meta['context_std'])).astype(np.float32)[active]
        corr[seed,active]=cs['predict'](model,im,ctx,np.arange(len(active)),'aux')*meta['target_std']
    err=float(np.max(abs(corr-saved['seed_corrections'])));assert err<1e-5
    _,p73=ns['readcsv']('073');corrected=p73+high*weights[1]*corr.mean(0);corrected[failed]=p73[failed]
    p82=base76+(corrected-p73);rid,reference=ns['readcsv']('082');assert np.array_equal(ids,rid)
    error=float(np.max(abs(p82-reference)));assert error<1e-5
    ns['csvout']('incumbent-082-replay.csv',ids,p82)
    byte=ns['sha'](_repo_path(OUT, 'incumbent-082-replay.csv'))==ns['sha'](_repo_path(ROOT, 'artifacts/submission-candidate-082.csv'));assert byte
    ns['save']('082-inference-verification.json',dict(all_passed=True,CNN_replay_max_difference=err,incumbent082_max_difference=error,CSV_byte_identical=byte,
        incumbent082_sha256=ns['sha'](_repo_path(OUT, 'incumbent-082-replay.csv')),incumbent076_sha256=ns['sha'](_repo_path(OUT, 'incumbent-076-cache-replay.csv')),
        all1000_rows=True,raw_test_images_inferred=len(active),training_labels_pixels_read=False,fitting_during_replay=False,source_sha256=ns['USED'],
        scope='076 reassembled from frozen component caches;082 residual CNNs inferred from saved weights and318 original deployed images. Earlier complete076 head replay remains protected. No full historical training reproduction.'))
    print('082/076 REPRODUCED',err,error,'byte_exact',byte,flush=True)

if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
