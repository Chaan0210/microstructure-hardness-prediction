"""Reload all source-restricted encoders and representative saved feature batches."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import encoder as e
import numpy as np,json,sys,gc,time
import torch
def main():
    start=time.monotonic();complete=json.loads((c.HERE/'encoders-complete.json').read_text());assert complete['complete'] and complete['source_sha256']==c.sha(e.__file__);z,d,pos=e.load();yd=c.inputs(labels=True);checks=[];random=[];mx=0.
    for k,g in enumerate(d['graphs']):
        for stage in ['selector','refit']:
            tr=np.array(g['A' if stage=='selector' else 'T']);q=np.r_[g['B'],g['V']] if stage=='selector' else np.array(g['V']);ti=np.array([pos[int(j)] for j in tr]);qi=np.array([pos[int(j)] for j in q]);norm=e.meta(z,ti,d['ids'][tr]);assert set(tr).isdisjoint(q)
            for repeat,seed0 in enumerate(e.SEEDS):
                base=c.HERE/f'encoders/fold{k}-{stage}-seed{repeat}-topology';initial=torch.load(_repo_io(base/'initial.pt'),map_location='cpu',weights_only=True);ih=e.state_hash(initial['state_dict']);paired=[]
                for arm in e.ARMS:
                    path=c.HERE/f'encoders/fold{k}-{stage}-seed{repeat}-{arm}';record=json.loads((path/'training.json').read_text());ck=torch.load(_repo_io(path/'model.pt'),map_location='cpu',weights_only=True);f=np.load(_repo_io(path/'features.npz'));assert e.state_hash(ck['state_dict'])==record['state_sha256'];assert record['initial_sha256']==ih and record['normalization']==norm==ck['normalization'];assert record['source_sha256']==c.sha(e.__file__);assert record['training_pixels_sha256']==c.ah(z['pixels'][ti]) and record['training_nodes_sha256']==c.ah(z['nodes'][ti]);assert record['hardness_used']==(arm=='hardness')
                    if arm=='hardness':assert record['hardness_sha256']==c.ah(yd['y'][tr]) and ck['y_mean']==float(yd['y'][tr].mean()) and ck['y_std']==float(yd['y'][tr].std())
                    else:assert record['hardness_sha256'] is None and ck['y_mean'] is None
                    assert np.array_equal(f['train'],tr) and np.array_equal(f['query'],q);assert f['train_latent'].shape==(len(tr),320) and f['query_signature'].shape==(len(q),28);assert all(np.isfinite(f[key]).all() for key in f.files if f[key].dtype.kind=='f');paired.append((record['order_sha256'],record['view_sha256']))
                    m=e.Net().to('mps');m.load_state_dict(ck['state_dict']);r=e.features(m,z['pixels'][qi[:e.BATCH]],norm);a=float(np.max(abs(r['latent']-f['query_latent'][:e.BATCH])));b=float(np.max(abs(r['signature']-f['query_signature'][:e.BATCH])));mx=max(mx,a,b);assert a<1e-5 and b<1e-8,(path,a,b);checks.append(dict(path=str(path.relative_to(c.HERE)),state_exact=True,normalization_training_only=True,latent_replay_error=a,signature_replay_error=b));del m,ck,f,r;gc.collect();torch.mps.empty_cache()
                assert len(set(paired))==1,'Input/order streams differ among matched objectives'
                f=np.load(_repo_io(base/'random-features.npz'));m=e.Net().to('mps');m.load_state_dict(initial['state_dict']);r=e.features(m,z['pixels'][qi[:e.BATCH]],norm);a=float(np.max(abs(r['latent']-f['query_latent'][:e.BATCH])));b=float(np.max(abs(r['signature']-f['query_signature'][:e.BATCH])));assert a<1e-5 and b<1e-8;random.append(dict(fold=k,stage=stage,seed=repeat,latent_replay_error=a,signature_replay_error=b));del m,f,r,initial;gc.collect();torch.mps.empty_cache()
                print('ENCODER AUDIT',k,stage,repeat,round(time.monotonic()-start,1),flush=True)
    c.save(_repo_io('encoder-verification.json'),dict(all_passed=True,trained=checks,random=random,all_initializations_and_source_streams_paired=True,maximum_replay_error=mx,seconds=time.monotonic()-start,source_sha256=c.sha(__file__),scope='Every saved state/source scope plus first original-size query batch from60 trained and20 random encoders. No new encoder fitting.'))
if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.27);sys.addaudithook(c.guard)
    with c.threadpool_limits(limits=2):main()
