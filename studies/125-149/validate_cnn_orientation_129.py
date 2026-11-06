"""Independent stream/schedule/averaging, model reload and label intervention."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,math,time,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cnn_orientation_129 as s
import numpy as np,torch
e=s.e

def main(split):
    s.check();start=time.perf_counter();d=s.cal.inputs(split);path=s.OUT/split;r=e.read(path/'report.json');assert e.sha(path/'predictions.npz')==r['predictions_sha256']
    with np.load(_repo_io(path/'predictions.npz')) as z:
        assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['y'],d['y']);np.testing.assert_array_equal(z['fold'],d['fid']);P=z['cnn'].copy();raw=z['raw'].copy();C=z['old_components'].copy()
    maxcpu=maxmps=0.;records=[]
    for k in range(5):
        T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);assert not set(d['g'][T])&set(d['g'][V]);streams=[]
        for ai,arm in enumerate(s.ARMS):
            folder=path/f'fold-{k}-{arm}';rec=e.read(folder/'report.json');assert rec==r['folds'][k]['arms'][arm]
            for file,key in [('model.pt','model_sha256'),('predictions.npz','predictions_sha256'),('streams.npz','streams_sha256')]:assert e.sha(folder/file)==rec[key]
            c=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);assert c['train_indices']==T.tolist() and c['validation_indices']==V.tolist();assert c['train_ids']==[d['ids'][i] for i in T] and c['validation_ids']==[d['ids'][i] for i in V];assert c['arm']==arm and c['seed']==s.SEED+k and c['epochs']==80 and c['average_last_epochs']==5
            norm=c['normalization'];assert norm=={'x_mean':float(d['images'][T].mean()),'x_std':float(d['images'][T].std()),'y_mean':float(d['y'][T].mean()),'y_std':float(d['y'][T].std())}
            with np.load(_repo_io(folder/'streams.npz')) as z:O=z['order'].copy();R=z['rot'].copy();F=z['flip'].copy();L=z['lr'].copy();assert np.isfinite(z['loss']).all()
            assert O.shape==(800,32);rng=np.random.default_rng(s.SEED+k);step=0
            for epoch in range(80):
                perm=rng.permutation(320);assert sorted(O[epoch*10:epoch*10+10].ravel())==list(range(320))
                for first in range(0,320,32):
                    np.testing.assert_array_equal(O[step],perm[first:first+32]);assert R[step]==int(rng.integers(4)) and F[step]==bool(rng.random()<.5);expected=.0001+.5*(.002-.0001)*(1+math.cos(math.pi*epoch/80));assert abs(L[step]-expected)<1e-17
                    if step in (0,99,399,799):
                        X=(d['images'][T[O[step]]]-norm['x_mean'])/max(norm['x_std'],1e-6)
                        if arm=='D4':
                            # Independent tensor transform, compared to actual NumPy training input.
                            X=torch.rot90(torch.from_numpy(X),int(R[step]),dims=(2,3))
                            if F[step]:X=X.flip(3)
                            X=X.numpy()
                        assert s.prior.ah(X)==rec['actual_input_sha256'][str(step)]
                    step+=1
            avg=None
            for epoch in range(76,81):
                state=torch.load(_repo_io(folder/f'epoch-{epoch}.pt'),map_location='cpu',weights_only=True);assert s.cal.h.state_hash(state)==rec['checkpoint_state_sha256'][str(epoch)]
                if avg is None:avg={n:v/5 for n,v in state.items()}
                else:
                    for n,v in state.items():avg[n]+=v/5
            assert all(torch.equal(v,c['state_dict'][n]) for n,v in avg.items());assert s.cal.h.state_hash(avg)==rec['average_state_sha256']
            model=s.engine.TinyCNN().to('mps');model.load_state_dict(c['state_dict']);mp=s.prior.predict(model,d['images'],V,norm,'mps');err=float(abs(mp-P[ai,V]).max());assert err<1e-3;maxmps=max(maxmps,err);model=model.cpu();cp=s.prior.predict(model,d['images'],V,norm,'cpu');err=float(abs(cp-P[ai,V]).max());assert err<1e-3;maxcpu=max(maxcpu,err)
            with np.load(_repo_io(folder/'predictions.npz')) as z:np.testing.assert_array_equal(z['validation_predictions'],P[ai,V]);assert e.rmse(d['y'][T],z['train_predictions'])==rec['train_RMSE']
            streams.append((O,R,F,L));del model
        for j in range(4):np.testing.assert_array_equal(streams[0][j],streams[1][j])
        a=r['folds'][k]['arms'];assert all(a['D4'][n]==a['identity'][n] for n in ['initial_state_sha256','rng_start','rng_end']);records.append({'fold':k,'all800paired_batches_LR_random_draws':True,'actual_input_sentinels_independently_verified':True,'no_group_or_ID_overlap':True,'five_states_average_exact':True,'legacy_reproduction':e.read(path/f'fold-{k}-legacy.json')})
    for ai,arm in enumerate(s.ARMS):
        np.testing.assert_allclose(raw[ai],.3*C[:,0]+.3*P[ai]+.2*C[:,2]+.2*C[:,3],rtol=0,atol=1e-12);assert e.rmse(d['y'],P[ai])==r['scores'][arm]['CNN_RMSE'] and e.rmse(d['y'],raw[ai])==r['scores'][arm]['raw4way_RMSE']
    stress=None
    if split=='original':
        T=np.flatnonzero(d['fid']!=0);V=np.flatnonzero(d['fid']==0);alt=dict(d);alt['y']=d['y'].copy();alt['y'][V]+=10000;pred,rec=s.fit(alt,T,V,0,'identity',s.OUT/'heldout-label-replay');old=r['folds'][0]['arms']['identity'];err=float(abs(pred-P[1,V]).max());assert err<1e-3 and rec['average_state_sha256']==old['average_state_sha256'];stress={'outside_training_targets_changed':80,'prediction_max_error':err,'state_sha256_exact':True}
    result={'passed':True,'split':split,'models_reloaded':10,'MPS_reload_max_error':maxmps,'CPU_MPS_max_error':maxcpu,'fold_checks':records,'heldout_label_refit':stress,'report_sha256':e.sha(path/'report.json'),'source_sha256':e.sha(Path(__file__)),'seconds':time.perf_counter()-start};s.check();e.write(s.OUT/f'validation-{split}.json',result);e.emit(result)
if __name__=='__main__':
    torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('split',choices=s.e.SPLITS);main(p.parse_args().split)
