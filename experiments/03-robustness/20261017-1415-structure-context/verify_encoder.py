"""Checkpoint/source/view checks; no hardness labels used."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import encoder as e
import numpy as np,json,sys,gc,torch
@torch.no_grad()
def cpu_features(model,px,indices,meta,arm):
    model=model.cpu().eval();raw=e.normalize(px,meta);perm=torch.from_numpy(e.permutation(indices,arm));out=[]
    for rot,flip in [(0,False),(1,False),(0,True),(1,True)]:
        x=np.rot90(raw,rot,axes=(2,3));x=x[:,:,:,::-1] if flip else x
        _,v=model(torch.from_numpy(np.ascontiguousarray(x)),perm,decode=False);out.append(v.numpy())
    return np.mean(out,0)
def main():
    complete=json.loads((e.HERE/'encoders-complete.json').read_text());assert complete['complete'];d=e.data();rows=[];matched=[]
    for k,g in enumerate(d['graphs']):
        for stage in ['selector','refit']:
            tr=np.array(g['A' if stage=='selector' else 'T']);q=np.r_[g['B'],g['V']] if stage=='selector' else np.array(g['V']);meta=dict(image_mean=float(d['pixels'][tr].mean(dtype=np.float64)/255),image_std=float(d['pixels'][tr].std(dtype=np.float64)/255),train_ids=d['ids'][tr].tolist())
            for repeat in range(2):
                pair={}
                for arm in ['spatial','permuted']:
                    path=e.HERE/f'encoders/fold{k}-{stage}-seed{repeat}-{arm}';r=json.loads((path/'training.json').read_text());ck=torch.load(_repo_io(path/'model.pt'),map_location='cpu',weights_only=True);assert ck['normalization']==meta and r['normalization']==meta;assert r['training_pixels_sha256']==e.ah(d['pixels'][tr]);assert r['source_visits']==100*len(tr) and r['labels_used'] is False and r['dataset_reads_blocked'];assert set(tr).isdisjoint(q)
                    m=e.ContextNet().to('mps');m.load_state_dict(ck['state_dict']);f=np.load(_repo_io(path/'features.npz'));assert np.array_equal(f['train'],tr) and np.array_equal(f['query'],q);actual=e.features(m,d['pixels'][q],q,meta,arm);queryerr=float(np.max(abs(actual-f['query_features'])));assert queryerr<2e-5
                    spot=np.r_[np.arange(8),np.arange(len(tr)-8,len(tr))];act=e.features(m,d['pixels'][tr[spot]],tr[spot],meta,arm);terr=float(np.max(abs(act-f['train_features'][spot])));assert terr<2e-5
                    ce=cpu_features(m,d['pixels'][q[:16]],q[:16],meta,arm);cerr=float(np.max(abs(ce-f['query_features'][:16])));assert cerr<3e-4
                    rows.append(dict(fold=k,stage=stage,repeat=repeat,arm=arm,query_reload_max_error=queryerr,train_spot_max_error=terr,CPU_vs_MPS_feature_max_error=cerr,source_ids_isolated=True,normalization_training_only=True));pair[arm]=r;del m;gc.collect();torch.mps.empty_cache()
                for key in ['initial_sha256','order_sha256','view_sha256','mask_sha256','source_visits','steps']:assert pair['spatial'][key]==pair['permuted'][key],key
                matched.append(dict(fold=k,stage=stage,repeat=repeat,initialization_order_views_masks_exact=True))
    e.save(_repo_io('encoder-verification.json'),dict(all_passed=True,models=rows,matched_controls=matched,maximum_MPS_reload_error=max(x['query_reload_max_error'] for x in rows),maximum_CPU_feature_error=max(x['CPU_vs_MPS_feature_max_error'] for x in rows),feature_tolerance_CPU=3e-4,feature_tolerance_MPS=2e-5,scope='All query descriptors,16 training-descriptor spots/model, all metadata and paired streams. CPU feature tolerance is dimensionless; downstream hardness tolerance is checked separately.'))
    print('ENCODERS VERIFIED',len(rows),flush=True)
if __name__=='__main__':
    assert torch.backends.mps.is_available();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.27);sys.addaudithook(e.guard)
    with e.threadpool_limits(limits=2):main()
