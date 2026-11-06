"""Validate readout inputs and aggregation before opening hardness scores."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import readout as r
import encoder as e
import numpy as np,json,sys
from sklearn.preprocessing import StandardScaler
def main():
    assert json.loads((e.HERE/'readouts-complete.json').read_text())['complete'];d=r.physical();graphs=json.loads((e.OLD/'graphs.json').read_text());contracts=0;pca_count=0
    for k,g in enumerate(graphs):
        for stage in ['selector','refit']:
            tr=np.array(g['A' if stage=='selector' else 'T']);q=np.r_[g['B'],g['V']] if stage=='selector' else np.array(g['V']);folder=e.HERE/f'readouts/fold{k}-{stage}';agg=np.load(_repo_io(folder/'predictions.npz'));assert np.array_equal(agg['train'],tr) and np.array_equal(agg['query'],q) and set(tr).isdisjoint(q)
            for arm in ['base',*r.ARMS]:
                for repeat in ([None] if arm=='base' else [0,1]):
                    parent=folder if arm=='base' else folder/f'{arm}-seed{repeat}';ctx=np.load(_repo_io(parent/('physical-context.npz' if arm=='base' else 'context.npz')));assert np.array_equal(ctx['X'][:,:100],d['X'][tr],equal_nan=True);assert np.array_equal(ctx['Q'][:,:100],d['X'][q],equal_nan=True);assert np.array_equal(ctx['y'],d['y'][tr])
                    if arm!='base':
                        src=e.HERE/(f'encoders/fold{k}-{stage}-seed{repeat}-'+('spatial' if arm=='random' else arm));f=np.load(_repo_io(src/('random-features.npz' if arm=='random' else 'features.npz')));assert np.array_equal(f['train'],tr) and np.array_equal(f['query'],q);sc=StandardScaler().fit(f['train_features']);assert np.array_equal(sc.mean_,ctx['scaler_mean']) and np.array_equal(sc.scale_,ctx['scaler_scale']);assert np.isfinite(ctx['X'][:,-16:]).all() and np.isfinite(ctx['Q'][:,-16:]).all();pca_count+=1
                    pp=[]
                    for j in range(3):
                        sub=folder/f'base-tab{j}' if arm=='base' else parent/f'tab{j}';record=json.loads((sub/'contract.json').read_text());z=np.load(_repo_io(sub/'prediction.npz'));assert record['train']==tr.tolist() and record['query']==q.tolist();assert record['train_X_sha256']==e.ah(ctx['X']) and record['query_X_sha256']==e.ah(ctx['Q']) and record['train_y_sha256']==e.ah(ctx['y']);assert np.array_equal(z['train'],tr) and np.array_equal(z['query'],q) and np.isfinite(z['prediction']).all();pp.append(z['prediction']);contracts+=1
                    assert np.array_equal(np.mean(pp,0),agg['base'] if arm=='base' else agg[arm][repeat])
    e.save(_repo_io('pre-score-verification.json'),dict(all_passed=True,contexts=contracts,training_PCA_scalers=pca_count,source_query_ids_disjoint=True,physical_features_original=True,training_labels_original=True,input_context_hashes_match=True,seed_aggregates_exact=True,outer_accuracy_not_computed=True,source_sha256=e.sha(__file__)));print('PRE-SCORE CHECKS PASSED',contracts,pca_count)
if __name__=='__main__':
    sys.addaudithook(e.guard)
    with e.threadpool_limits(limits=2):main()
