"""Saved-model, nested selection and complete deterministic training checks."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import experiment as e
import evaluate as a
import numpy as np,json,math,sys,gc,os

def main(repeats):
    tag=''.join(map(str,repeats));result=json.loads((e.HERE/f'results-{tag}.json').read_text());z=np.load(_repo_io(e.HERE/f'oof-{tag}.npz'));d=e.load();models=[]
    for repeat in repeats:
        for k,g in enumerate(d['graphs']):
            for stage in ['selector','refit']:
                tr=np.array(g['A'] if stage=='selector' else g['T']);q=np.r_[g['B'],g['V']] if stage=='selector' else np.array(g['V']);reports=[]
                for arm in e.ARMS:
                    path=e.HERE/f'models/seed{repeat}/fold{k}-{stage}-{arm}';m,meta=e.checkpoint(path/'model.pt');p,emb=e.predict(m,d['pixels'][q],meta);old=np.load(_repo_io(path/'predictions.npz'))
                    assert np.array_equal(p,old['prediction']) and np.array_equal(emb,old['embedding'])
                    assert set(meta['train_ids'])==set(d['ids'][tr]) and set(meta['train_ids']).isdisjoint(d['ids'][q])
                    assert meta['target_mean']==float(d['y'][tr].mean()) and meta['target_std']==float(d['y'][tr].std())
                    assert meta['image_mean']==float(d['pixels'][tr].mean(dtype=np.float64)/255.) and meta['image_std']==float(d['pixels'][tr].std(dtype=np.float64)/255.)
                    rec=json.loads((path/'training.json').read_text());assert rec['updates']==e.EPOCHS*int(np.ceil(len(tr)/32));assert rec['gradients']['relational_encoder_norm']>1e-9
                    reports.append(rec);models.append(dict(repeat=repeat,fold=k,stage=stage,arm=arm,prediction_reload_max_error=0.,model_sha256=e.sha(path/'model.pt')));del m;gc.collect()
                for key in ['initial_sha256','order_sha256','views_sha256','updates','source_visits','view_visits']:assert len({r[key] for r in reports})==1,key
    # Refit a full40-epoch selector after poisoning every outside label. During
    # the fit, no dataset/model file may be read at all; training receives only A.
    g=d['graphs'][0];A=np.array(g['A']);outside=np.setdiff1d(np.arange(500),A);changed=d['y'].copy();changed[outside]+=10000
    scope={'training':False}
    def no_data_reads(event,args):
        if scope['training'] and event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=e.Path(os.fsdecode(args[0]));mode,flags=args[1:];writing=(mode is not None and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if not writing and p.suffix.lower() in ['.npz','.npy','.csv','.png','.jpg','.jpeg','.pt','.joblib']:raise RuntimeError('Dataset read during isolated fit: '+str(p))
    sys.addaudithook(no_data_reads);folder=e.HERE/'label-poison-replay';scope['training']=True
    model,meta,rec=e.fit_model(d['pixels'][A],changed[A].copy(),d['ids'][A].tolist(),'relational',e.SEED,folder)
    scope['training']=False
    original=e.torch.load(_repo_io(e.HERE/'models/seed0/fold0-selector-relational/model.pt'),map_location='cpu',weights_only=True)
    actual=e.torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True)
    assert original['normalization']==actual['normalization'];assert all(e.torch.equal(v,actual['state_dict'][k]) for k,v in original['state_dict'].items())
    q=np.r_[g['B'],g['V']];p,_=e.predict(model,d['pixels'][q],meta);ref=np.load(_repo_io(e.HERE/'models/seed0/fold0-selector-relational/predictions.npz'));assert np.array_equal(p,ref['prediction'])
    independent={}
    for arm in e.ARMS:
        score=math.sqrt(math.fsum((float(p)-float(y))**2 for p,y in zip(z[arm],z['y']))/len(z['y']))
        assert abs(score-result['results'][arm]['candidate_rmse'])<1e-12;independent[arm]=score
    coefficient_checks=[]
    for k,g in enumerate(d['graphs']):
        anchor=np.load(_repo_io(e.HERE/f'anchors/fold{k}.npz'));B=anchor['B'];V=anchor['V'];mask=anchor['highB'];ix=anchor['ix']
        for arm in e.ARMS:
            aa=np.mean([np.load(_repo_io(e.HERE/f'models/seed{r}/fold{k}-selector-{arm}/predictions.npz'))['prediction'][:len(B)] for r in repeats],axis=0)
            dd=aa-anchor['p82B'];numerator=math.fsum(float(dd[j])*float(d['y'][B[j]]-anchor['p82B'][j]) for j in np.flatnonzero(mask));den=math.fsum(float(dd[j])**2 for j in np.flatnonzero(mask));w=max(0.,min(.25,numerator/den)) if den>1e-12 else 0.
            saved=result['selection'][k]['arms'][arm]['weight'];assert abs(w-saved)<1e-14
            pp=np.mean([np.load(_repo_io(e.HERE/f'models/seed{r}/fold{k}-refit-{arm}/predictions.npz'))['prediction'] for r in repeats],axis=0);assembled=anchor['p82V']+anchor['highV']*saved*(pp-anchor['p82V'])
            assert np.array_equal(assembled,z[arm][ix]);coefficient_checks.append(dict(fold=k,arm=arm,weight=saved))
    e.save(_repo_io(f'verification-{tag}.json'),dict(all_passed=True,models=models,same_initialization_orders_views_visits=True,normalization_training_only=True,
        complete_label_poison_training_replay=dict(epochs=e.EPOCHS,outside_labels_changed=len(outside),all_tensor_states_exact=True,predictions_exact=True,dataset_reads_blocked_during_training=True),
        independent_scalar_RMSE=independent,coefficient_rechecks=coefficient_checks,limitations='Verifies new image learner and selection. Frozen incumbent auxiliary representations retain their historical validation limits.'))
    print('VERIFIED',len(models),'models and full40-epoch label-poison replay',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,nargs='+',default=[0]);args=p.parse_args();assert args.repeats in [[0],[0,1]]
    sys.addaudithook(e.guard);e.torch.set_num_threads(2);e.torch.set_num_interop_threads(1);e.torch.use_deterministic_algorithms(True)
    with e.threadpool_limits(limits=2):main(args.repeats)
