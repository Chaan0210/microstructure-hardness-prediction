"""Verify the equivalent aggregated loss; NOT a new evaluated candidate."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import encoder as e
import deterministic_loss as dl
import numpy as np,json,sys,gc,time
import torch
ORIGINAL_OBJECTIVE=e.objective
def objective(outputs,pack,arm,y=None):
    if arm!='topology':return ORIGINAL_OBJECTIVE(outputs,pack,arm,y)
    b,edges,_,_=outputs;_,by,bw,ey,ew,pairs,py,pw=pack;convert=lambda x:torch.from_numpy(x).to('mps');connection,critical=dl.connection_loss(edges,pairs,py,pw);loss=e.balanced(b,convert(by),convert(bw))+e.balanced(edges,convert(ey),convert(ew))+connection;return loss.mean(),critical
def main(mode):
    assert mode in ['original','poison'];e.objective=objective;z,d,pos=e.load();g=d['graphs'][0];tr=np.array(g['A']);q=np.r_[g['B'],g['V']];ti=np.array([pos[int(i)] for i in tr]);qi=np.array([pos[int(i)] for i in q]);s=e.subset(z,ti);norm=e.meta(z,ti,d['ids'][tr]);path=c.HERE/f'deterministic-replay/{mode}';path.mkdir(parents=True,exist_ok=False);old=json.loads((c.HERE/'encoders/fold0-selector-seed0-topology/training.json').read_text());m=e.init(e.SEEDS[0]);initial=e.state_hash(m.state_dict());assert initial==old['initial_sha256'];start=time.monotonic();torch.save(dict(state_dict={k:v.cpu() for k,v in m.state_dict().items()},normalization=norm),_repo_io(path/'initial.pt'));warmq=z['pixels'][qi].copy()
    if mode=='poison':warmq=255-warmq
    a=e.features(m,z['pixels'][ti],norm);b=e.features(m,warmq,norm);assert e.state_hash(m.state_dict())==initial;c.arrays(str((path/'random-features.npz').relative_to(c.HERE)),train=tr,query=q,train_latent=a['latent'],query_latent=b['latent'],train_signature=a['signature'],query_signature=b['signature']);record=e.fit(m,s,norm,'topology',e.SEEDS[0]);assert record['order_sha256']==old['order_sha256'] and record['view_sha256']==old['view_sha256'];a=e.features(m,z['pixels'][ti],norm);b=e.features(m,z['pixels'][qi],norm);state={k:v.cpu() for k,v in m.state_dict().items()};torch.save(dict(state_dict=state,normalization=norm,objective_implementation='deterministic_loss.py'),_repo_io(path/'model.pt'));c.arrays(str((path/'features.npz').relative_to(c.HERE)),train=tr,query=q,train_latent=a['latent'],query_latent=b['latent'],train_signature=a['signature'],query_signature=b['signature']);comparison=None
    if mode=='poison':
        previous=c.HERE/'deterministic-replay/original';ck=torch.load(_repo_io(previous/'model.pt'),map_location='cpu',weights_only=True);f=np.load(_repo_io(previous/'features.npz'));error=max(float((v-ck['state_dict'][k]).abs().max()) for k,v in state.items());latent=max(float(np.max(abs(a['latent']-f['train_latent']))),float(np.max(abs(b['latent']-f['query_latent']))));signature=max(float(np.max(abs(a['signature']-f['train_signature']))),float(np.max(abs(b['signature']-f['query_signature']))));comparison=dict(parameter_states_exact=e.state_hash(state)==e.state_hash(ck['state_dict']),maximum_parameter_difference=error,maximum_latent_difference=latent,maximum_signature_difference=signature)
    c.save(_repo_io(str((path/'verification.json').relative_to(c.HERE))),dict(mode=mode,complete=True,training=record,comparison=comparison,hardness_labels_received=False,original_OOF_not_modified=True,new_variant_accuracy_not_evaluated=True,excluded_warmup_images_changed=len(q) if mode=='poison' else 0,warmup_has_no_parameter_effect=True,training_pixels_sha256=c.ah(s['pixels']),warmup_query_sha256=c.ah(warmq),state_sha256=e.state_hash(state),seconds=time.monotonic()-start,source_sha256=c.sha(__file__),objective_sha256=c.sha(dl.__file__),scope='Only gradient aggregation changed, on the sameA213sources/40epochs/seed. Two complete numerical/source-isolation replays, no replacement of the60-model nested result.'));print('AGGREGATED REPLAY',mode,comparison,flush=True)
if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.27);sys.addaudithook(c.guard)
    with c.threadpool_limits(limits=2):main(sys.argv[1])
