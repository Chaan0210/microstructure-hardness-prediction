"""Audit every new readout context and projection before outer hardness scoring."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import readout as r
import numpy as np,json,sys,time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def main():
    assert json.loads((c.HERE/'readouts-complete.json').read_text())['complete'];assert json.loads((c.HERE/'encoder-verification.json').read_text())['all_passed'];d=r.physical();rawz=np.load(_repo_io(c.HERE/'raw-signatures.npz'));raw=dict(zip(map(int,rawz['indices']),rawz['signature']));contexts=0;projection=[];baseline=[];start=time.monotonic()
    for k,g in enumerate(d['graphs']):
        for stage in ['selector','refit']:
            tr=np.array(g['A' if stage=='selector' else 'T']);q=np.r_[g['B'],g['V']] if stage=='selector' else np.array(g['V']);folder=c.HERE/f'readouts/fold{k}-{stage}';agg=np.load(_repo_io(folder/'predictions.npz'));ref=json.loads((folder/'baseline-reference.json').read_text());old=_repo_path(c.ROOT, ref['path']);assert c.sha(old/'physical-context.npz')==ref['context_sha256'] and c.sha(old/'predictions.npz')==ref['predictions_sha256'];bc=np.load(_repo_io(old/'physical-context.npz'));bp=np.load(_repo_io(old/'predictions.npz'));assert np.array_equal(bc['X'],d['X'][tr],equal_nan=True) and np.array_equal(bc['Q'],d['X'][q],equal_nan=True) and np.array_equal(bc['y'],d['y'][tr]);assert np.array_equal(agg['base'],bp['base']) and set(tr).isdisjoint(q);base=[]
            for j in range(3):
                cr=json.loads((old/f'base-tab{j}/contract.json').read_text());assert cr['seed']==100*j+k and cr['train_X_sha256']==c.ah(bc['X']) and cr['query_X_sha256']==c.ah(bc['Q']) and cr['train_y_sha256']==c.ah(bc['y']);base.append(np.load(_repo_io(old/f'base-tab{j}/prediction.npz'))['prediction'])
            assert np.array_equal(np.mean(base,0),agg['base']);baseline.append(dict(fold=k,stage=stage,physical_reference_contexts=3,exact=True))
            for arm in r.ARMS:
                for repeat in ([0] if arm=='raw' else [0,1]):
                    path=folder/f'{arm}-seed{repeat}';ctx=np.load(_repo_io(path/'context.npz'));assert np.array_equal(ctx['train'],tr) and np.array_equal(ctx['query'],q);assert np.array_equal(ctx['X'][:,:100],d['X'][tr],equal_nan=True) and np.array_equal(ctx['Q'][:,:100],d['X'][q],equal_nan=True) and np.array_equal(ctx['y'],d['y'][tr])
                    if arm=='raw':
                        assert np.array_equal(ctx['X'][:,100:128],np.array([raw[int(i)] for i in tr])) and np.array_equal(ctx['Q'][:,100:128],np.array([raw[int(i)] for i in q]));assert np.count_nonzero(ctx['X'][:,-16:])==0 and np.count_nonzero(ctx['Q'][:,-16:])==0
                    else:
                        src=c.HERE/(f'encoders/fold{k}-{stage}-seed{repeat}-'+('topology' if arm=='random' else arm));f=np.load(_repo_io(src/('random-features.npz' if arm=='random' else 'features.npz')));sc=StandardScaler().fit(f['train_latent']);pc=PCA(n_components=16,svd_solver='full').fit(sc.transform(f['train_latent']));assert np.array_equal(sc.mean_,ctx['scaler_mean']) and np.array_equal(sc.scale_,ctx['scaler_scale']);assert np.array_equal(pc.components_,ctx['pca_components']) and np.array_equal(pc.mean_,ctx['pca_mean']);assert np.array_equal(ctx['X'][:,100:128],f['train_signature']) and np.array_equal(ctx['Q'][:,100:128],f['query_signature']);assert np.array_equal(ctx['X'][:,-16:],pc.transform(sc.transform(f['train_latent']))) and np.array_equal(ctx['Q'][:,-16:],pc.transform(sc.transform(f['query_latent'])));manual=((f['query_latent'].astype(float)-ctx['scaler_mean'])/ctx['scaler_scale']-ctx['pca_mean'])@ctx['pca_components'].T;err=float(np.max(abs(manual-ctx['Q'][:,-16:])));assert err<5e-4;projection.append(dict(fold=k,stage=stage,arm=arm,seed=repeat,independent_transform_error=err))
                    pp=[]
                    for j in range(3):
                        sub=path/f'tab{j}';cc=json.loads((sub/'contract.json').read_text());pred=np.load(_repo_io(sub/'prediction.npz'));assert cc['seed']==100*j+k and cc['train']==tr.tolist() and cc['query']==q.tolist();assert cc['train_X_sha256']==c.ah(ctx['X']) and cc['train_y_sha256']==c.ah(ctx['y']) and cc['query_X_sha256']==c.ah(ctx['Q']);assert np.array_equal(pred['train'],tr) and np.array_equal(pred['query'],q) and np.isfinite(pred['prediction']).all();pp.append(pred['prediction']);contexts+=1
                    assert np.array_equal(np.mean(pp,0),agg[arm][repeat])
                    if arm=='raw':assert np.array_equal(agg[arm][0],agg[arm][1])
    c.save(_repo_io('pre-score-verification.json'),dict(all_passed=True,new_contexts=contexts,reused_baselines=baseline,PCA=projection,source_labels_and_physical_columns_exact=True,all_seed_aggregates_exact=True,outer_accuracy_not_read=True,seconds=time.monotonic()-start,source_sha256=c.sha(__file__)));print('PRE SCORE PASSED',contexts,len(projection),flush=True)
if __name__=='__main__':
    sys.addaudithook(c.guard)
    with c.threadpool_limits(limits=2):main()
