"""Pre-data engineering tests, independent slow convolution and noiseless toy fit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
import json,time,sys
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
import d4_cnn_172 as m
import cnn_controls as old
R=_RepoPath(__file__).resolve().parents[1];OUT=_repo_path(R, 'artifacts/d4-cnn-172')
torch.set_num_threads(2)

def slow(layer,x):
    outputs=[]
    for g in range(8):
        terms=[]
        for h in range(1 if layer.lift else 8):
            # Independent inverse: find r such that g*r=h directly in table.
            r=0 if layer.lift else next(r for r in range(8) if m.MULT[g][r]==h)
            w=m.action(layer.weight[:,:,r],g);xx=x if layer.lift else x[:,:,h]
            terms.append(F.conv2d(xx,w,None,padding=layer.k//2))
        outputs.append(sum(terms)+layer.bias[None,:,None,None])
    return torch.stack(outputs,2)

def unit():
    torch.manual_seed(172);rec={'group_order':8,'group_compositions_tested':64,'devices':{}}
    x=torch.randn(1,2,8,7,7,dtype=torch.double)
    for a in range(8):
        for b in range(8):assert torch.equal(m.feature_action(m.feature_action(x,b),a),m.feature_action(x,m.MULT[a][b]))
    rec['slow_forward_gradient']=[]
    for lift in [True,False]:
        layer=m.D4Conv(2,3,3,lift).double();xx=torch.randn(1,2,7,7,dtype=torch.double) if lift else x
        xx=xx.requires_grad_();fast=layer(xx);ref=slow(layer,xx);torch.testing.assert_close(fast,ref,rtol=1e-12,atol=1e-12)
        probe=torch.randn_like(fast);a=torch.autograd.grad((fast*probe).sum(),(xx,layer.weight,layer.bias),retain_graph=True);b=torch.autograd.grad((ref*probe).sum(),(xx,layer.weight,layer.bias))
        for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,rtol=1e-11,atol=1e-11)
        eq=[]
        for g in range(8):
            got=layer(m.action(xx,g) if lift else m.feature_action(xx,g));expected=m.feature_action(fast,g);err=float((got-expected).abs().max().detach());assert err<1e-12;eq.append(err)
        rec['slow_forward_gradient'].append({'lift':lift,'forward_max':float((fast-ref).abs().max().detach()),'gradient_max':max(float((aa-bb).abs().max()) for aa,bb in zip(a,b)),'equivariance_max':max(eq)})
    # Every orientation branch must affect other orientations: dense group mixing, not 8 independent CNNs.
    layer=m.D4Conv(1,1,3).double();xx=torch.randn(1,1,8,7,7,dtype=torch.double,requires_grad=True);grad=torch.autograd.grad(layer(xx)[:,:,0].square().sum(),xx)[0];assert all(grad[:,:,h].abs().sum()>0 for h in range(8));rec['all8_input_orientations_influence_one_output_orientation']=True
    for device in ['cpu','mps']:
        assert device!='mps' or torch.backends.mps.is_available();torch.manual_seed(172);model=m.D4CNN().to(device).eval();xx=torch.randn(2,1,128,128).to(device)
        with torch.inference_mode():
            stages=model.stages(xx);p=model(xx);fe=[];pe=[]
            for g in range(8):
                other=model.stages(m.action(xx,g));fe.append([float((o-m.feature_action(v,g)).abs().max().cpu()) for o,v in zip(other,stages)]);pe.append(float((model(m.action(xx,g))-p).abs().max().cpu()))
            assert max(max(v) for v in fe)<3e-4 and max(pe)<2e-5,(device,fe,pe)
        rec['devices'][device]={'stage_max_abs_error':np.max(fe,axis=0).tolist(),'invariant_prediction_max_error':max(pe),'stage_shapes':[list(s.shape) for s in stages]}
        del model,xx,stages
    torch.manual_seed(20261003);legacy=old.TinyCNN();torch.manual_seed(20261003);control=m.PoolCNN();assert all(torch.equal(v,control.state_dict()[k]) for k,v in legacy.state_dict().items());counts={n:sum(p.numel() for p in model.parameters()) for n,model in [('legacy',legacy),('pool_control',control),('group',m.D4CNN())]};assert abs(counts['group']/counts['legacy']-1)<.01;rec['parameters']=counts;rec['pool_control_initial_state_identical_to_legacy']=True;rec['passed']=True
    return rec

def synthetic():
    # 16 deterministic noiseless targets; no development labels or images.
    rng=np.random.default_rng(172);a=rng.uniform(-1,1,(16,1,1,1));b=rng.uniform(.1,1,(16,1,1,1));raw=rng.normal(size=(16,1,32,32));raw=np.repeat(np.repeat(raw[:,:,::4,::4],4,axis=2),4,axis=3);xx=torch.tensor(a+b*raw,dtype=torch.float32)
    def target(x):
        edge=((x[:,:,:,1:]-x[:,:,:,:-1]).square().mean((1,2,3))+(x[:,:,1:,:]-x[:,:,:-1,:]).square().mean((1,2,3)))/2
        return .8*x.mean((1,2,3))+.6*x.std((1,2,3),correction=0)+.3*edge
    y=target(xx)
    for g in range(8):torch.testing.assert_close(target(m.action(xx,g)),y,atol=1e-6,rtol=1e-6)
    y=(y-y.mean())/y.std(correction=0);torch.manual_seed(172);model=m.D4CNN().to('mps');model.head[2].p=0.;opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=0);x=xx.to('mps');y=y.to('mps');start=time.perf_counter();curve=[]
    with torch.no_grad():initial=float(F.mse_loss(model(x),y).cpu())
    for i in range(300):
        model.train();opt.zero_grad(set_to_none=True);loss=F.mse_loss(model(m.action(x,i%8)),y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if (i+1)%50==0:curve.append({'step':i+1,'online_mse':float(loss.detach().cpu())})
    model.eval()
    with torch.no_grad():
        p=model(x);final=float(F.mse_loss(p,y).cpu());all8=max(float(F.mse_loss(model(m.action(x,g)),y).cpu()) for g in range(8))
    result={'n':16,'size':32,'steps':300,'noise_free_analytic_D4_invariant_targets':True,'dropout':0,'weight_decay':0,'initial_standardized_MSE':initial,'final_standardized_MSE':final,'max8view_MSE':all8,'curve':curve,'seconds':time.perf_counter()-start,'passed':final<.001 and all8<.001};assert result['passed'],result
    return result

def benchmark():
    records={}
    for name,cls in [('legacy',old.TinyCNN),*m.MODELS.items()]:
        torch.manual_seed(172);model=cls().to('mps');opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01);x=torch.randn(32,1,128,128).to('mps');y=torch.randn(32).to('mps');times=[]
        for i in range(16):
            torch.mps.synchronize();s=time.perf_counter();opt.zero_grad(set_to_none=True);loss=F.mse_loss(model(x),y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();torch.mps.synchronize()
            if i>=4:times.append(time.perf_counter()-s)
        r={'median_update_seconds':float(np.median(times)),'projected800update_seconds':float(np.median(times)*800),'memory':old.memory('mps'),'finite_loss':bool(torch.isfinite(loss))};assert r['projected800update_seconds']<600 and r['memory']['process_peak_rss_bytes']<3e9 and r['memory']['driver_allocated_memory']<3e9;records[name]=r
        del model,opt,x,y;torch.mps.empty_cache()
    records['passed']=True;return records

if __name__=='__main__':
    import study_d4_172 as s
    s.check();mode=sys.argv[1];result={'unit':unit,'synthetic':synthetic,'benchmark':benchmark}[mode]();s.write(_repo_path(OUT, f'{mode}.json'),result);print(mode.upper(),'PASS',json.dumps(result),flush=True)
