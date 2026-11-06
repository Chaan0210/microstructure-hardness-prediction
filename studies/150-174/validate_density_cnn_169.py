"""Independent density/gradient, matched budget, checkpoint and leakage validation."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,math,time
import numpy as np,torch
import density_cnn_169 as s
e=s.e

def independent_weights(y):
    t=[float(v) for v in y];n=len(t);mean=sum(t)/n;sd=math.sqrt(sum((v-mean)**2 for v in t)/n);h=max(.4*sd,1e-6)
    rho=np.array([sum(math.exp(-.5*((v-u)/h)**2) for j,u in enumerate(t) if i!=j)/((n-1)*h*math.sqrt(2*math.pi)) for i,v in enumerate(t)])
    med=float(np.median(rho));raw=np.array([min(1.25,max(.8,(med/max(float(r),.1*med,1e-30))**.25)) for r in rho]);return rho,raw,raw/raw.mean()

def main(split):
    s.check();start=time.perf_counter();d=s.load(split);path=s.OUT/split;r=e.read(path/'report.json');assert e.sha(path/'predictions.npz')==r['predictions_sha256']
    with np.load(_repo_io(path/'predictions.npz')) as z:assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['y'],d['y']);np.testing.assert_array_equal(z['fold'],d['fid']);P=z['cnn'].copy();blend=z['blend'].copy()
    # Hand loss gradient fixture independent of actual training code.
    p=torch.tensor([-.4,1.2,2.1],dtype=torch.float64,requires_grad=True);y=torch.tensor([.7,-.3,.8],dtype=torch.float64);w=torch.tensor([.9,1.,1.1],dtype=torch.float64);loss=torch.mean(w*(p-y)**2);grad=torch.autograd.grad(loss,p)[0];np.testing.assert_allclose(grad.numpy(),2*w.numpy()*(p.detach().numpy()-y.numpy())/3,rtol=0,atol=1e-15)
    maxcpu=maxmps=maxweight=maxdensity=maxloss=0.;records=[]
    for k in range(5):
        T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);assert not set(d['g'][T])&set(d['g'][V]);rho,raw,weight=independent_weights(d['y'][T]);streams=[]
        for ai,arm in enumerate(s.ARMS):
            folder=path/f'fold-{k}-{arm}';rec=e.read(folder/'report.json');assert rec==r['folds'][k]['arms'][arm]
            for file,key in [('model.pt','model_sha256'),('predictions.npz','predictions_sha256'),('streams.npz','streams_sha256'),('weights.npz','weights_sha256')]:assert e.sha(folder/file)==rec[key]
            obj=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);assert obj['train_indices']==T.tolist() and obj['validation_indices']==V.tolist();assert obj['train_ids']==[d['ids'][i] for i in T] and obj['validation_ids']==[d['ids'][i] for i in V];assert obj['arm']==arm and obj['seed']==s.SEED+k
            norm=obj['normalization'];assert norm=={'x_mean':float(d['images'][T].mean()),'x_std':float(d['images'][T].std()),'y_mean':float(d['y'][T].mean()),'y_std':float(d['y'][T].std())}
            with np.load(_repo_io(folder/'weights.npz')) as z:
                np.testing.assert_array_equal(z['T'],T);np.testing.assert_array_equal(z['y'],d['y'][T]);actual=z['weights'].copy();expected=weight if arm=='density' else np.ones(len(T));er=float(abs(actual-expected).max());ed=float(abs(z['density']-rho).max());assert er<2e-14 and ed<1e-15;maxweight=max(maxweight,er);maxdensity=max(maxdensity,ed);assert abs(actual.mean()-1)<1e-12 and actual.max()/actual.min()<=1.5625+1e-12
            with np.load(_repo_io(folder/'streams.npz')) as z:O=z['order'].copy();R=z['rot'].copy();F=z['flip'].copy();L=z['lr'].copy();assert np.isfinite(z['loss']).all() and np.isfinite(z['unweighted_loss']).all()
            assert O.shape==(800,32);rng=np.random.default_rng(s.SEED+k);step=0
            for epoch in range(80):
                perm=rng.permutation(320);assert sorted(O[epoch*10:epoch*10+10].ravel())==list(range(320))
                for first in range(0,320,32):
                    np.testing.assert_array_equal(O[step],perm[first:first+32]);assert R[step]==int(rng.integers(4)) and F[step]==bool(rng.random()<.5);lr=.0001+.5*(.002-.0001)*(1+math.cos(math.pi*epoch/80));assert abs(L[step]-lr)<1e-17
                    if step in (0,99,399,799):
                        X=(d['images'][T[O[step]]]-norm['x_mean'])/max(norm['x_std'],1e-6);X=torch.rot90(torch.from_numpy(X),int(R[step]),dims=(2,3));X=X.flip(3) if F[step] else X;assert s.prior.ah(X.numpy())==rec['actual_input_sha256'][str(step)]
                    step+=1
            for key,z in rec['loss_sentinels'].items():
                st=int(key);ww=np.array(z['weight']);np.testing.assert_array_equal(ww,actual[O[st]].astype(np.float32));tar=((d['y'][T[O[st]]]-norm['y_mean'])/norm['y_std']).astype(np.float32);np.testing.assert_array_equal(tar,z['target']);manual=float(np.mean(ww*(np.array(z['prediction'])-np.array(z['target']))**2));err=abs(manual-z['loss']);assert err<max(1e-6,abs(manual)*2e-6);maxloss=max(maxloss,err)
            avg=None
            for epoch in range(76,81):
                state=torch.load(_repo_io(folder/f'epoch-{epoch}.pt'),map_location='cpu',weights_only=True);assert s.cal.h.state_hash(state)==rec['checkpoint_state_sha256'][str(epoch)]
                if avg is None:avg={n:v/5 for n,v in state.items()}
                else:
                    for n,v in state.items():avg[n]+=v/5
            assert all(torch.equal(v,obj['state_dict'][n]) for n,v in avg.items());assert s.cal.h.state_hash(avg)==rec['average_state_sha256']
            model=s.engine.TinyCNN().to('mps');model.load_state_dict(obj['state_dict']);mp=s.prior.predict(model,d['images'],V,norm,'mps');err=float(abs(mp-P[ai,V]).max());assert err<1e-3;maxmps=max(maxmps,err);model=model.cpu();cp=s.prior.predict(model,d['images'],V,norm,'cpu');err=float(abs(cp-P[ai,V]).max());assert err<1e-3;maxcpu=max(maxcpu,err)
            with np.load(_repo_io(folder/'predictions.npz')) as z:
                np.testing.assert_array_equal(z['validation_predictions'],P[ai,V]);assert e.rmse(d['y'][T],z['train_predictions'])==rec['train_RMSE'];traincpu=s.prior.predict(model,d['images'],T,norm,'cpu');assert abs(traincpu-z['train_predictions']).max()<1e-3
            streams.append((O,R,F,L));del model
        for j in range(4):np.testing.assert_array_equal(streams[0][j],streams[1][j])
        pair=r['folds'][k]['arms'];assert all(pair['mse'][n]==pair['density'][n] for n in ['initial_state_sha256','rng_start','rng_end','actual_input_sha256']);records.append({'fold':k,'all800paired_streams_LR_D4_dropout':True,'train_only_density_stats':True,'five_state_average_exact':True,'legacy_MSE_reproduction':e.read(path/f'legacy-{k}.json')})
    np.testing.assert_array_equal(blend,.8*d['reference018'][None,:]+.2*P)
    for ai,arm in enumerate(s.ARMS):assert e.rmse(d['y'],P[ai])==r['RMSE'][arm+'_CNN'] and e.rmse(d['y'],blend[ai])==r['RMSE'][arm+'_blend']
    stress=None
    if split=='original':
        T=np.flatnonzero(d['fid']!=0);V=np.flatnonzero(d['fid']==0);alt=dict(d);alt['y']=d['y'].copy();alt['y'][V]+=10000
        pred,rec=s.fit(alt,T,V,0,'density',s.OUT/'heldout-label-replay');ref=r['folds'][0]['arms']['density'];err=float(abs(pred-P[1,V]).max());assert err<1e-3 and rec['average_state_sha256']==ref['average_state_sha256'];stress={'outside_labels_changed':80,'prediction_max_error':err,'state_sha256_exact':True}
        with np.load(_repo_io(s.OUT/'heldout-label-replay/weights.npz')) as a,np.load(_repo_io(path/'fold-0-density/weights.npz')) as b:np.testing.assert_array_equal(a['weights'],b['weights']);np.testing.assert_array_equal(a['density'],b['density'])
    result={'passed':True,'split':split,'models_reloaded':10,'MPS_reload_max_error':maxmps,'CPU_MPS_max_error':maxcpu,'independent_density_max_error':maxdensity,'independent_weights_max_error':maxweight,'independent_training_loss_max_error':maxloss,'loss_gradient_fixture':True,'fold_checks':records,'heldout_label_refit':stress,'report_sha256':e.sha(path/'report.json'),'source_sha256':e.sha(s.R/'validate_density_cnn_169.py'),'seconds':time.perf_counter()-start};s.check();s.write(s.OUT/f'validation-{split}.json',result);print('AUDIT_PASS',json.dumps(result),flush=True)
if __name__=='__main__':main(sys.argv[1])
