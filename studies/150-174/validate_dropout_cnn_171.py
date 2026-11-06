"""Independent validation ofone dropout-only CNN fit and its fixedcontrol."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,math,time
import numpy as np,torch
import dropout_cnn_171 as s
e=s.e

def main(k):
    s.check();start=time.perf_counter();d=s.load();T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);path=s.OUT/f'fold-{k}';r=e.read(path/'report.json');control=e.read(s.ref(k)/'report.json');assert r['dropout']==0 and r['updates']==800 and r['seed']==s.SEED+k
    for file,key in [('model.pt','model_sha256'),('predictions.npz','predictions_sha256'),('streams.npz','streams_sha256')]:assert e.sha(path/file)==r[key]
    obj=torch.load(_repo_io(path/'model.pt'),map_location='cpu',weights_only=True);assert obj['dropout']==0 and obj['arm']=='D4' and obj['epochs']==80 and obj['average_last_epochs']==5;assert obj['train_indices']==T.tolist() and obj['validation_indices']==V.tolist();assert obj['train_ids']==[d['ids'][i] for i in T] and obj['validation_ids']==[d['ids'][i] for i in V];assert not set(d['g'][T])&set(d['g'][V])
    norm=obj['normalization'];assert norm=={'x_mean':float(d['images'][T].mean()),'x_std':float(d['images'][T].std()),'y_mean':float(d['y'][T].mean()),'y_std':float(d['y'][T].std())}
    old=torch.load(_repo_io(s.ref(k)/'model.pt'),map_location='cpu',weights_only=True);assert norm==old['normalization'];assert obj['train_ids']==old['train_ids'];assert obj['seed']==old['seed'];assert r['initial_state_sha256']==control['initial_state_sha256'] and r['rng_start']==control['rng_start'];assert r['rng_end']==r['rng_start'] and control['rng_end']!=control['rng_start']
    with np.load(_repo_io(path/'streams.npz')) as z:O=z['order'].copy();R=z['rot'].copy();F=z['flip'].copy();L=z['lr'].copy();assert np.isfinite(z['loss']).all()
    with np.load(_repo_io(s.ref(k)/'streams.npz')) as z:
        for name,x in [('order',O),('rot',R),('flip',F),('lr',L)]:np.testing.assert_array_equal(x,z[name])
    assert O.shape==(800,32);rng=np.random.default_rng(s.SEED+k);step=0
    for epoch in range(80):
        perm=rng.permutation(320);assert sorted(O[epoch*10:epoch*10+10].ravel())==list(range(320))
        for first in range(0,320,32):
            np.testing.assert_array_equal(O[step],perm[first:first+32]);assert R[step]==int(rng.integers(4)) and F[step]==bool(rng.random()<.5);lr=.0001+.5*(.002-.0001)*(1+math.cos(math.pi*epoch/80));assert abs(L[step]-lr)<1e-17
            if step in (0,99,399,799):
                X=(d['images'][T[O[step]]]-norm['x_mean'])/max(norm['x_std'],1e-6);X=torch.rot90(torch.from_numpy(X),int(R[step]),dims=(2,3));X=X.flip(3) if F[step] else X;assert s.prior.ah(X.numpy())==r['actual_input_sha256'][str(step)]==control['actual_input_sha256'][str(step)]
            step+=1
    avg=None
    for epoch in range(76,81):
        state=torch.load(_repo_io(path/f'epoch-{epoch}.pt'),map_location='cpu',weights_only=True);assert s.cal.h.state_hash(state)==r['checkpoint_state_sha256'][str(epoch)]
        if avg is None:avg={n:v/5 for n,v in state.items()}
        else:
            for n,v in state.items():avg[n]+=v/5
    assert all(torch.equal(v,obj['state_dict'][n]) for n,v in avg.items());assert s.cal.h.state_hash(avg)==r['average_state_sha256']
    with np.load(_repo_io(path/'predictions.npz')) as z:pp=z['validation_predictions'].copy();tp=z['train_predictions'].copy();np.testing.assert_array_equal(z['train_indices'],T);np.testing.assert_array_equal(z['validation_indices'],V)
    errors={}
    for name,sv,drop,expected in [('control',old,.15,d['legacy'][V]),('dropout0',obj,0.,pp)]:
        model=s.engine.TinyCNN().to('mps');model.head[2].p=drop;model.load_state_dict(sv['state_dict']);mp=s.audit.predict(model,d['images'][V],norm);em=float(abs(mp-expected).max());assert em<1e-3;model=model.cpu();cp=s.audit.predict(model,d['images'][V],norm,device='cpu');ec=float(abs(cp-expected).max());assert ec<1e-3
        xx=torch.tensor((d['images'][V[:32]]-norm['x_mean'])/max(norm['x_std'],1e-6))
        with torch.inference_mode():fp=s.audit.functional(sv['state_dict'],xx).numpy()*norm['y_std']+norm['y_mean']
        ef=float(abs(fp-cp[:32]).max());assert ef<1e-3;errors[name]={'MPS_reload':em,'CPU_reload':ec,'functional_CPU':ef}
        if drop==0:
            ct=s.audit.predict(model,d['images'][T],norm,device='cpu');assert abs(ct-tp).max()<1e-3;assert e.rmse(d['y'][T],tp)==r['train_RMSE'] and e.rmse(d['y'][V],pp)==r['validation_RMSE'];model.train()
            with torch.inference_mode():p_train=model(xx);model.eval();p_eval=model(xx);assert torch.equal(p_train,p_eval)
        del model
    poison=None
    if k==0:
        alt=dict(d);alt['y']=d['y'].copy();alt['y'][V]+=10000;rp,rr=s.fit(alt,T,V,0,'D4',s.OUT/'outside-label-replay');err=float(abs(rp-pp).max());assert err<1e-3 and rr['average_state_sha256']==r['average_state_sha256'];assert rr['initial_state_sha256']==r['initial_state_sha256'];poison={'changed_labels':80,'prediction_max_error':err,'state_exact':True}
    result={'passed':True,'fold':k,'model_sha256':r['model_sha256'],'report_sha256':e.sha(path/'report.json'),'same_initial_weights_seed':True,'same800sample_D4_LR_streams':True,'same_train_only_normalization_IDs_labels':True,'four_actual_pixel_hashes_match':True,'five_checkpoint_average_exact':True,'dropout0_train_eval_identical':True,'p0_consumes_no_dropout_MPS_RNG':True,'control_reloaded':True,'reload_errors':errors,'outside_label_refit':poison,'source_sha256':e.sha(_repo_path(s.ROOT, 'validate_dropout_cnn_171.py')),'seconds':time.perf_counter()-start};s.check();s.write(path/'validation.json',result);print('AUDIT_PASS',json.dumps(result),flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
