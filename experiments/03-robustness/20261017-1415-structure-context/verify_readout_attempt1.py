"""Train-scope/PCA/context/aggregation audit and selected numerical replays."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import readout as r
import encoder as e
import numpy as np,json,math,sys,gc,time,torch
from sklearn.preprocessing import StandardScaler
def predict(X,y,Q,seed,device):
    m=r.TabPFNRegressor.create_default_for_version(r.ModelVersion('v3.5'),model_path=str(r.WEIGHT),device=device,inference_precision=torch.float32,memory_saving_mode=True,random_state=seed,n_preprocessing_jobs=1,show_progress_bar=False);m.fit(X,y.copy());p=np.asarray(m.predict(Q),float);small=np.asarray(m.predict(Q[:8]),float);del m;gc.collect()
    if device=='mps':torch.mps.empty_cache()
    return p,small
def main():
    d=r.physical();graphs=json.loads((e.OLD/'graphs.json').read_text());rows=[];pca_checks=[];max_pc_error=0.;source_hash=e.sha(r.WEIGHT);lock=json.loads((e.HERE/'readout-code-lock.json').read_text());assert source_hash==lock['foundation_weight_sha256']
    for k,g in enumerate(graphs):
        for stage in ['selector','refit']:
            tr=np.array(g['A' if stage=='selector' else 'T']);q=np.r_[g['B'],g['V']] if stage=='selector' else np.array(g['V']);folder=e.HERE/f'readouts/fold{k}-{stage}';p=np.load(_repo_io(folder/'predictions.npz'));ctx=np.load(_repo_io(folder/'physical-context.npz'));assert np.array_equal(ctx['X'],d['X'][tr],equal_nan=True) and np.array_equal(ctx['Q'],d['X'][q],equal_nan=True) and np.array_equal(ctx['y'],d['y'][tr]);base=[]
            for arm in ['base','spatial','permuted','random']:
                repeats=[None] if arm=='base' else [0,1]
                for repeat in repeats:
                    if arm=='base':c=ctx;parent=folder
                    else:
                        parent=folder/f'{arm}-seed{repeat}';c=np.load(_repo_io(parent/'context.npz'));src=e.HERE/(f'encoders/fold{k}-{stage}-seed{repeat}-'+('spatial' if arm=='random' else arm));f=np.load(_repo_io(src/('random-features.npz' if arm=='random' else 'features.npz')));sc=StandardScaler().fit(f['train_features']);assert np.array_equal(sc.mean_,c['scaler_mean']) and np.array_equal(sc.scale_,c['scaler_scale'])
                        # Independent float64 application of the saved training PCA.
                        manual=((f['query_features'].astype(float)-c['scaler_mean'])/c['scaler_scale']-c['pca_mean'])@c['pca_components'].T;err=float(np.max(abs(manual-c['Q'][:,-16:])));max_pc_error=max(max_pc_error,err);assert err<5e-4
                        assert np.array_equal(c['X'][:,:100],d['X'][tr],equal_nan=True) and np.array_equal(c['Q'][:,:100],d['X'][q],equal_nan=True) and np.array_equal(c['y'],d['y'][tr]);pca_checks.append(dict(fold=k,stage=stage,arm=arm,repeat=repeat,training_statistics_exact=True,manual_PCA_max_error=err))
                    assert np.array_equal(c['train'],tr) and np.array_equal(c['query'],q) and set(tr).isdisjoint(q);pp=[]
                    for j in range(3):
                        sub=folder/f'base-tab{j}' if arm=='base' else parent/f'tab{j}';contract=json.loads((sub/'contract.json').read_text());old=np.load(_repo_io(sub/'prediction.npz'));assert contract['seed']==100*j+k and contract['train']==tr.tolist() and contract['query']==q.tolist();assert contract['train_X_sha256']==e.ah(c['X']) and contract['query_X_sha256']==e.ah(c['Q']) and contract['train_y_sha256']==e.ah(c['y']);assert np.array_equal(old['train'],tr) and np.array_equal(old['query'],q);pp.append(old['prediction']);rows.append(dict(fold=k,stage=stage,arm=arm,repeat=repeat,tab_seed=j,context_hashes_passed=True))
                    average=np.mean(pp,0);reference=p['base'] if arm=='base' else p[arm][repeat];assert np.array_equal(average,reference)
    # Refit one representative paired context with all outside-A labels poisoned.
    folder=e.HERE/'readouts/fold0-selector';tr=np.array(graphs[0]['A']);outside=np.setdiff1d(np.arange(500),tr);poison=d['y'].copy();poison[outside]+=10000;replays=[]
    for arm in ['base','spatial-seed0']:
        c=np.load(_repo_io(folder/('physical-context.npz' if arm=='base' else arm+'/context.npz')));sub=folder/'base-tab0' if arm=='base' else folder/arm/'tab0';old=np.load(_repo_io(sub/'prediction.npz'))['prediction'];assert np.array_equal(poison[tr],c['y']);mps,small=predict(c['X'],poison[tr],c['Q'],0,'mps');err=float(np.max(abs(mps-old)));subset=float(np.max(abs(small-mps[:8])));assert err<1e-5 and subset<1e-4
        cpu,_=predict(c['X'],poison[tr],c['Q'][:8],0,'cpu');cpuerr=float(np.max(abs(cpu-mps[:8])));assert cpuerr<1e-3;replays.append(dict(arm=arm,MPS_replay_error=err,query_subset_error=subset,CPU_vs_MPS_max_hardness_error=cpuerr,outside_labels_poisoned=len(outside)))
    out=np.load(_repo_io(e.HERE/'oof.npz'));result=json.loads((e.HERE/'results.json').read_text());arith={};coefficients=[]
    for arm in r.ARMS:
        value=math.sqrt(math.fsum((float(p)-float(y))**2 for p,y in zip(out[arm],out['y']))/400);assert abs(value-result['results'][arm]['candidate_rmse'])<1e-12;arith[arm]=value
    for k,g in enumerate(graphs):
        an=np.load(_repo_io(e.OLD/f'anchors/fold{k}.npz'));B=an['B'];V=an['V'];s=np.load(_repo_io(e.HERE/f'readouts/fold{k}-selector/predictions.npz'));v=np.load(_repo_io(e.HERE/f'readouts/fold{k}-refit/predictions.npz'));good=np.flatnonzero(an['highB']);changed=d['y'].copy();changed[V]+=10000
        for arm in r.ARMS:
            delta=(s[arm].mean(0)-s['base'])[:len(B)];num=math.fsum(float(delta[i])*float(changed[B[i]]-an['p82B'][i]) for i in good);den=math.fsum(float(delta[i])**2 for i in good)+len(good);w=max(0.,min(.5,num/den));expected=result['selection'][k]['arms'][arm]['beta'];assert abs(w-expected)<1e-12;pred=an['p82V']+an['highV']*expected*(v[arm].mean(0)-v['base']);assert np.array_equal(pred,out[arm][an['ix']]);coefficients.append(dict(fold=k,arm=arm,beta=expected,outer_label_poison_unchanged=True))
    e.save(_repo_io('readout-verification.json'),dict(all_passed=True,contexts=rows,PCA=pca_checks,maximum_manual_PCA_error=max_pc_error,selected_actual_replays=replays,independent_scalar_RMSE=arith,coefficient_checks=coefficients,scope='All210 contexts/seed aggregations and60 training PCA transforms; two actual poisoned paired Tab replays, query-subset checks and CPU/MPS spots. Not a complete retraining of every Tab context or the incumbent.'))
    print('READOUT VERIFIED',len(rows),'contexts',len(pca_checks),'PCA fits',flush=True)
if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.27);sys.addaudithook(e.guard)
    with e.threadpool_limits(limits=2):main()
