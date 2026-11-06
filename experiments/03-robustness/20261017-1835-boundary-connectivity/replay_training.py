"""Full same-seed topology and ordinary-hardness refits with outside data poisoned."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import encoder as e
import numpy as np,json,sys,gc,time
import torch
def main():
    z,d,pos=e.load();yd=c.inputs(labels=True);tr=np.array(d['graphs'][0]['A']);ix=np.array([pos[int(j)] for j in tr]);outside=np.setdiff1d(np.arange(len(z['dev'])),ix);original=e.subset(z,ix);norm=e.meta(z,ix,d['ids'][tr]);poison={k:v.copy() for k,v in z.items()};poison['pixels'][outside]=255-poison['pixels'][outside];poison['nodes'][outside]=0;poison['boundary'][outside]=1-poison['boundary'][outside];poison['boundary_weight'][outside]=0;s=e.subset(poison,ix)
    assert all(np.array_equal(s[k],original[k]) for k in s);labels=yd['y'].copy();excluded=np.setdiff1d(np.arange(500),tr);labels[excluded]+=10000;y=(labels[tr]-labels[tr].mean())/labels[tr].std();assert np.array_equal(labels[tr],yd['y'][tr]);out=[];start=time.monotonic()
    for arm in ['topology','hardness']:
        folder=c.HERE/f'encoders/fold0-selector-seed0-{arm}';old=json.loads((folder/'training.json').read_text());saved=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);m=e.init(e.SEEDS[0]);assert e.state_hash(m.state_dict())==old['initial_sha256'];rec=e.fit(m,s,norm,arm,e.SEEDS[0],y if arm=='hardness' else None);assert rec['view_sha256']==old['view_sha256'] and rec['order_sha256']==old['order_sha256'];error=max(float((v.detach().cpu()-saved['state_dict'][k]).abs().max()) for k,v in m.state_dict().items());exact=e.state_hash(m.state_dict())==old['state_sha256']
        # If accelerator training is not bit deterministic, report the difference;
        # the direct capability check still proves excluded values were not passed.
        q=np.array(d['graphs'][0]['B'][:32]);qi=np.array([pos[int(j)] for j in q]);r=e.features(m,z['pixels'][qi],norm);f=np.load(_repo_io(folder/'features.npz'));latent=float(np.max(abs(r['latent']-f['query_latent'][:32])));signature=float(np.max(abs(r['signature']-f['query_signature'][:32])));out.append(dict(arm=arm,training_sources=len(tr),outside_cached_images_changed=len(outside),outside_labels_changed=len(excluded),training_inputs_and_streams_exact=True,parameter_states_exact=exact,maximum_parameter_difference=error,query_latent_max_difference=latent,query_signature_max_difference=signature,seconds=rec['seconds'],peak_memory=rec['peak_memory'],hardness_received=arm=='hardness'));del m,saved,r;gc.collect();torch.mps.empty_cache();print('FULL REPLAY',arm,exact,error,latent,signature,flush=True)
    c.save(_repo_io('training-repeatability.json'),dict(complete=True,records=out,all_states_exact=all(x['parameter_states_exact'] for x in out),seconds=time.monotonic()-start,source_sha256=c.sha(__file__),scope='Two complete40epochA213 refits; outside data changed before selecting model inputs. Not a retraining of all60 models or the incumbent.'))
if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.27);sys.addaudithook(c.guard)
    with c.threadpool_limits(limits=2):main()
