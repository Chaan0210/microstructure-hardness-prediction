"""Measure complete100-epoch MPS repeatability with excluded pixels poisoned."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import encoder as e
import numpy as np,json,sys,torch,gc
def main():
    assert json.loads((e.HERE/'encoders-complete.json').read_text())['complete'];d=e.data();g=d['graphs'][0];tr=np.array(g['A']);q=np.r_[g['B'],g['V']];outside=np.setdiff1d(np.arange(500),tr);changed=d['pixels'].copy();changed[outside]=255-changed[outside]
    original=e.HERE/'encoders/fold0-selector-seed0-spatial';r=json.loads((original/'training.json').read_text());old=torch.load(_repo_io(original/'model.pt'),map_location='cpu',weights_only=True);meta=old['normalization'];m=e.init(e.SEEDS[0]);initial=e.sh(m.state_dict());assert initial==r['initial_sha256']
    # Recreate the original random-feature warmup, but poison excluded query pixels.
    e.features(m,changed[tr],tr,meta,'random');e.features(m,changed[q],q,meta,'random');assert e.sh(m.state_dict())==initial
    record=e.fit_encoder(m,changed[tr].copy(),tr,meta,'spatial',e.SEEDS[0]);assert e.ah(changed[tr])==r['training_pixels_sha256']
    for key in ['order_sha256','view_sha256','mask_sha256','steps','source_visits']:assert record[key]==r[key]
    diff={k:float((v.detach().cpu()-old['state_dict'][k]).abs().max()) for k,v in m.state_dict().items()};pred=e.features(m,d['pixels'][q],q,meta,'spatial');ref=np.load(_repo_io(original/'features.npz'))['query_features'];ferr=float(np.max(abs(pred-ref)))
    folder=e.HERE/'encoder-repeat';folder.mkdir(exist_ok=False);torch.save(dict(state_dict={k:v.cpu() for k,v in m.state_dict().items()},normalization=meta),_repo_io(folder/'model.pt'));e.arrays('encoder-repeat/features.npz',query=q,features=pred)
    e.save(_repo_io('encoder-repeatability.json'),dict(complete=True,epochs=100,training_sources=len(tr),outside_array_slots_changed=len(outside),changed_query_images=len(q),unread_holdout_slots=100,hardness_labels_received=False,query_warmup_does_not_change_parameters=True,training_pixels_and_view_streams_exact=True,dataset_reads_blocked=record['dataset_reads_blocked'],all_tensor_states_exact=all(v==0 for v in diff.values()),maximum_tensor_difference=max(diff.values()),maximum_query_feature_difference=ferr,seconds=record['seconds'],peak_memory=record['peak_memory'],scope='One complete same-seed MPS refit. Report numerical differences rather than claiming all historical training was reproduced.'))
    print('ENCODER REPLAY',max(diff.values()),ferr,record['seconds'],flush=True)
if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.27);sys.addaudithook(e.guard)
    with e.threadpool_limits(limits=2):main()
