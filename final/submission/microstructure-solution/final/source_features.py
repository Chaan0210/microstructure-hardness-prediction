from pathlib import Path
import argparse,csv,gc,hashlib,importlib.util,json,os,sys,time
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
os.environ['HF_HUB_OFFLINE']='1'
if '--out' in sys.argv and sys.argv.index('--out')+1<len(sys.argv):
    runtime_tmp=Path(sys.argv[sys.argv.index('--out')+1]).resolve()/'tmp'
    runtime_tmp.mkdir(parents=True,exist_ok=True)
    for k in ['TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME','CELLPOSE_LOCAL_MODELS_PATH']:
        os.environ[k]=str(runtime_tmp)
import numpy as np
import torch
from PIL import Image
from cellpose.vit import CPSAM
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
H=HERE/'project/experiments/supervised-renewal-20261018'
spec=importlib.util.spec_from_file_location('frozen_flow38',H/'flow38_fast.py')
flow38=importlib.util.module_from_spec(spec);spec.loader.exec_module(flow38)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def normalize(a):
    lo,hi=np.percentile(a,[1,99]);return ((a.astype(np.float32)-float(lo))/max(float(hi-lo),1.)).astype(np.float32)

def main(a):
    global H
    H=Path(a.project).resolve()/'experiments/supervised-renewal-20261018'
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    for k in ['TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME','CELLPOSE_LOCAL_MODELS_PATH']:os.environ[k]=str(out/'tmp')
    (out/'tmp').mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert torch.backends.mps.is_available();torch.mps.set_per_process_memory_fraction(.3)
    recipe_path=H/'candidate-source-flow38-v1/frozen-recipe.json';recipe=json.loads(recipe_path.read_text())
    assert sha(H/'flow38_fast.py')==recipe['descriptor_source_sha256']
    tag=f'full500-sourceflow38-s{a.seed}';modelpath=H/f'source-h300/{tag}/model.pt'
    trained=json.loads(modelpath.with_name('trained.json').read_text());assert trained['complete'] and sha(modelpath)==trained['model_sha256']
    data=Path(a.data).resolve();rows=list(csv.DictReader((data/('train.csv' if a.split=='train' else 'sample_submission.csv')).open()))
    ids=np.array([r['ID'] for r in rows]);assert len(ids)==(500 if a.split=='train' else 1000) and len(set(ids))==len(ids)
    name=f'{tag}-{a.split}';final=out/f'{name}.npz'
    if final.exists():raise SystemExit('Completed feature file exists; use a new destination.')
    ck=torch.load(modelpath,map_location='cpu',weights_only=False,mmap=True)
    assert ck['recipe']['protocol']=='full500' and ck['recipe']['seed']==2026100000+a.seed
    net=CPSAM();state={k:v for k,v in ck['state'].items() if k not in ['diam_labels','diam_mean']}
    status=net.load_state_dict(state,strict=False);assert set(status.missing_keys)<={'diam_labels','diam_mean'} and not status.unexpected_keys
    del ck,state;gc.collect();net.eval().requires_grad_(False).to('mps')
    contract=dict(model_sha256=trained['model_sha256'],recipe_sha256=sha(recipe_path),descriptor_sha256=sha(H/'flow38_fast.py'),split=a.split)
    partial=out/f'{name}-partial.npz';values=[];elapsed=0.
    if partial.exists():
        z=np.load(partial);assert json.loads(str(z['contract']))==contract
        values=list(z['flow']);elapsed=float(z['elapsed']);assert np.array_equal(z['ids'],ids[:len(values)])
    start=time.monotonic();limit=min(a.limit or len(ids),len(ids));keys=recipe['flow_keys']
    with torch.no_grad():
        for j in range(len(values),limit):
            with Image.open(data/a.split/f'{ids[j]}.png') as image:
                assert image.mode=='L' and image.size==(256,256);raw=np.asarray(image)
            features=[]
            for first in [0,2]:
                x=np.stack([np.ascontiguousarray(np.rot90(normalize(raw),rot)) for rot in [first,first+1]])[:,None]
                flow,_=net(torch.tensor(x,device='mps'));flow=flow.cpu().numpy()
                for k,rot in enumerate([first,first+1]):
                    f,cols=flow38.descriptors(flow[k],np.rot90(raw,rot));assert cols==keys;features.append(f)
            value=np.mean(features,0);assert value.shape==(38,) and np.isfinite(value).all();values.append(value);torch.mps.empty_cache()
            if len(values)%10==0 or len(values)==limit:
                np.savez(partial.with_name(name+'-partial-new.npz'),ids=ids[:len(values)],flow=np.array(values),elapsed=elapsed+time.monotonic()-start,contract=json.dumps(contract))
                partial.with_name(name+'-partial-new.npz').replace(partial)
                print(name,len(values),'/',len(ids),'seconds',round(elapsed+time.monotonic()-start,2),flush=True)
    values=np.array(values);error=None;exact=None
    if not a.retrained:
        ref=np.load(H/f'candidate-source-flow38-v1/features/{name}.npz')
        assert np.array_equal(ref['ids'][:len(values)],ids[:len(values)])
        error=float(abs(values-ref['flow'][:len(values)]).max());exact=np.array_equal(values,ref['flow'][:len(values)])
    report=dict(rows=len(values),all_rows=len(values)==len(ids),max_abs_feature_difference=error,exact_array=exact,seconds=elapsed+time.monotonic()-start,parameters_updated=0,hardness_values_read=False,contract=contract)
    if len(values)==len(ids):np.savez(final,ids=ids,flow=values,flow_keys=np.array(keys))
    (out/f'{name}-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--project',type=Path,default=HERE/'project')
    p.add_argument('--retrained',action='store_true')
    p.add_argument('--seed',choices=[1811,1823],type=int,required=True);p.add_argument('--split',choices=['train','test'],required=True);p.add_argument('--limit',type=int,default=0)
    with threadpool_limits(2):main(p.parse_args())
