"""Independent protocol, prediction reload and heldout-label intervention checks."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,math,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cnn_update_budget_128 as q
import numpy as np,torch
e=q.e

def main(split):
    q.check();start=time.perf_counter();d=q.cal.inputs(split);path=q.OUT/split;report=e.read(path/'report.json');assert e.sha(path/'predictions.npz')==report['predictions_sha256']
    with np.load(_repo_io(path/'predictions.npz')) as z:
        assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['y'],d['y']);P=z['predictions'].copy()
    maxmps=0.;maxcpu=0.;models=0;streamrecords=[]
    for k in range(5):
        small,large,V=q.indices(d,split,k);streams=[];inits=[]
        for ai,arm in enumerate(q.ARMS):
            folder=path/f'fold-{k}-{arm}';rec=e.read(folder/'report.json');assert rec==report['folds'][k]['arms'][arm]
            for name,key in [('model.pt','model_sha256'),('predictions.npz','predictions_sha256'),('streams.npz','streams_sha256')]:assert e.sha(folder/name)==rec[key]
            c=torch.load(_repo_io(folder/'model.pt'),weights_only=True,map_location='cpu');T=np.array(c['train_indices']);np.testing.assert_array_equal(T,large if arm=='large800' else small);assert c['validation_indices']==V.tolist() and c['train_ids']==[d['ids'][i] for i in T] and c['validation_ids']==[d['ids'][i] for i in V];assert not set(d['g'][T])&set(d['g'][V]);norm=c['normalization'];assert norm=={'x_mean':float(d['images'][T].mean()),'x_std':float(d['images'][T].std()),'y_mean':float(d['y'][T].mean()),'y_std':float(d['y'][T].std())}
            with np.load(_repo_io(folder/'streams.npz')) as z:order=z['order'].copy();rot=z['rot'].copy();flip=z['flip'].copy();lr=z['lr'].copy();assert np.isfinite(z['loss_standardized']).all()
            U=400 if arm=='small400' else 800;assert order.shape==(U,32) and len(lr)==U+1 and c['updates']==U and c['seed']==q.SEED+k
            expected=np.array([.0001+(.002-.0001)*(1+math.cos(math.pi*t/U))/2 for t in range(U+1)]);np.testing.assert_allclose(lr,expected,atol=1e-18,rtol=0);assert abs(lr[-1]-.0001)<1e-18
            flat=order.ravel();n=len(T);rng=np.random.default_rng(q.SEED+k+4100)
            for st in range(0,len(flat),n):
                perm=rng.permutation(n);take=min(n,len(flat)-st);np.testing.assert_array_equal(flat[st:st+take],perm[:take]);assert len(set(perm.tolist()))==n
            assert np.ptp(np.bincount(flat,minlength=n))<=1
            rng=np.random.default_rng(q.SEED+k+4200);er=rng.integers(0,4,800);ef=rng.random(800)<.5;np.testing.assert_array_equal(rot,er[:U]);np.testing.assert_array_equal(flip,ef[:U]);streams.append((order,rot,flip,lr));inits.append(rec['initial_state_sha256'])
            points=[380,385,390,395,400] if U==400 else [760,770,780,790,800];assert c['average_steps']==rec['average_steps']==points;avg=None
            for step in points:
                state=torch.load(_repo_io(folder/f'average-step-{step}.pt'),map_location='cpu',weights_only=True);assert q.cal.h.state_hash(state)==rec['checkpoint_state_sha256'][str(step)]
                if avg is None:avg={n:v/5 for n,v in state.items()}
                else:
                    for n,v in state.items():avg[n]+=v/5
            for n,v in avg.items():assert torch.equal(v,c['state_dict'][n])
            assert q.cal.h.state_hash(avg)==rec['average_state_sha256']
            model=q.engine.TinyCNN().to('mps');model.load_state_dict(c['state_dict']);mp=q.predict(model,d['images'],V,norm,'mps');err=float(abs(mp-P[ai,V]).max());assert err<1e-3;maxmps=max(maxmps,err);model=model.cpu();cpu=q.predict(model,d['images'],V,norm,'cpu');err=float(abs(cpu-P[ai,V]).max());assert err<1e-3;maxcpu=max(maxcpu,err);models+=1
        assert len(set(inits))==1
        np.testing.assert_array_equal(streams[0][0],streams[1][0][:400]);np.testing.assert_array_equal(streams[0][1],streams[1][1][:400]);np.testing.assert_array_equal(streams[0][2],streams[1][2][:400]);np.testing.assert_array_equal(streams[1][1],streams[2][1]);np.testing.assert_array_equal(streams[1][2],streams[2][2]);np.testing.assert_array_equal(streams[1][3],streams[2][3]);np.testing.assert_array_equal(streams[0][3],streams[1][3][::2]);rs=[report['folds'][k]['arms'][arm]['rng_state_sha256'] for arm in q.ARMS];assert len(set(r['0'] for r in rs))==1 and len(set(r['400'] for r in rs))==1 and rs[1]['800']==rs[2]['800'];streamrecords.append({'fold':k,'all32_batches_and_no_drop_last':True,'small_prefix_and_common_D4':True,'normalized_schedule_endpoints_and_weight_average':True,'MPS_dropout_RNG_consumption_prefix_matched':True})
    for ai,arm in enumerate(q.ARMS):assert e.rmse(d['y'],P[ai])==report['RMSE'][arm]
    stress=None
    if split=='original':
        T,_,V=q.indices(d,split,0);alt=dict(d);alt['y']=d['y'].copy();outside=np.setdiff1d(np.arange(400),T);alt['y'][outside]+=10000;pred,rec=q.fit(alt,T,V,400,q.SEED,q.OUT/'heldout-label-replay');err=float(abs(pred-P[0,V]).max());assert err<1e-3;old=e.read(path/'fold-0-small400/report.json');assert rec['initial_state_sha256']==old['initial_state_sha256'] and rec['sample_order_sha256']==old['sample_order_sha256'] and rec['augmentation_sha256']==old['augmentation_sha256'];stress={'all258_outside_training_labels_changed':len(outside),'prediction_max_error':err,'initialization_order_augmentation_unchanged':True,'state_sha256_exact':rec['average_state_sha256']==old['average_state_sha256']}
    result={'passed':True,'split':split,'models_reloaded':models,'MPS_reload_max_error':maxmps,'CPU_MPS_max_error':maxcpu,'protocol_checks':streamrecords,'heldout_label_refit':stress,'report_sha256':e.sha(path/'report.json'),'source_sha256':e.sha(Path(__file__)),'seconds':time.perf_counter()-start};q.check();e.write(q.OUT/f'validation-{split}.json',result);e.emit(result)
if __name__=='__main__':
    torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('split',choices=q.e.SPLITS);main(p.parse_args().split)
