"""Match original startup exactly and vary only excluded warm-up images."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import encoder as e
import numpy as np,json,sys,gc,time
import torch
def main(mode):
    assert mode in ['original','poison'];z,d,pos=e.load();g=d['graphs'][0];tr=np.array(g['A']);q=np.r_[g['B'],g['V']];ti=np.array([pos[int(i)] for i in tr]);qi=np.array([pos[int(i)] for i in q]);s=e.subset(z,ti);norm=e.meta(z,ti,d['ids'][tr]);path=c.HERE/f'warmup-replay/{mode}';path.mkdir(parents=True,exist_ok=False);folder=c.HERE/'encoders/fold0-selector-seed0-topology';old=json.loads((folder/'training.json').read_text());saved=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);reference=np.load(_repo_io(folder/'features.npz'));m=e.init(e.SEEDS[0]);initial=e.state_hash(m.state_dict());assert initial==old['initial_sha256'];start=time.monotonic()
    torch.save(dict(state_dict={k:v.cpu() for k,v in m.state_dict().items()},normalization=norm),_repo_io(path/'initial.pt'))
    # Main training computed both random-control feature sets before fitting.
    # Match this startup; only the excluded query values differ in poison mode.
    warmq=z['pixels'][qi].copy()
    if mode=='poison':warmq=255-warmq
    a=e.features(m,z['pixels'][ti],norm);b=e.features(m,warmq,norm);assert e.state_hash(m.state_dict())==initial
    c.arrays(str((path/'random-features.npz').relative_to(c.HERE)),train=tr,query=q,train_latent=a['latent'],train_signature=a['signature'],query_latent=b['latent'],query_signature=b['signature'])
    record=e.fit(m,s,norm,'topology',e.SEEDS[0]);assert record['order_sha256']==old['order_sha256'] and record['view_sha256']==old['view_sha256'];maximum=max(float((v.detach().cpu()-saved['state_dict'][k]).abs().max()) for k,v in m.state_dict().items());exact=e.state_hash(m.state_dict())==old['state_sha256'];a=e.features(m,z['pixels'][ti],norm);b=e.features(m,z['pixels'][qi],norm);latent=max(float(np.max(abs(a['latent']-reference['train_latent']))),float(np.max(abs(b['latent']-reference['query_latent']))));signature=max(float(np.max(abs(a['signature']-reference['train_signature']))),float(np.max(abs(b['signature']-reference['query_signature']))))
    torch.save(dict(state_dict={k:v.cpu() for k,v in m.state_dict().items()},normalization=norm),_repo_io(path/'model.pt'));c.arrays(str((path/'features.npz').relative_to(c.HERE)),train=tr,query=q,train_latent=a['latent'],query_latent=b['latent'],train_signature=a['signature'],query_signature=b['signature']);diff=[abs(x['loss']-y['loss']) for x,y in zip(record['curve'],old['curve'])];c.save(_repo_io(str((path/'verification.json').relative_to(c.HERE))),dict(mode=mode,complete=True,exact_original_startup_sequence=True,training_inputs_and_streams_exact=True,query_warmup_cannot_change_any_saved_state=True,warmup_query_sha256=c.ah(warmq),training_pixels_sha256=c.ah(s['pixels']),excluded_query_images_changed=len(q) if mode=='poison' else 0,hardness_labels_received=False,parameter_states_exact=exact,maximum_parameter_difference=maximum,maximum_latent_difference=latent,maximum_signature_difference=signature,maximum_epoch_training_loss_difference=max(diff),first_different_epoch=next((i+1 for i,v in enumerate(diff) if v!=0),None),state_sha256=e.state_hash(m.state_dict()),seconds=time.monotonic()-start,training=record,source_sha256=c.sha(__file__)));print('WARMUP REPLAY',mode,exact,maximum,latent,signature,flush=True)
if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.27);sys.addaudithook(c.guard)
    with c.threadpool_limits(limits=2):main(sys.argv[1])
