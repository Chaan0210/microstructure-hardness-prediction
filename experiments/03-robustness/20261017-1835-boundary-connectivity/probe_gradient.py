"""Isolate repeated-index maximin gather backward, with no optimizer update."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import encoder as e
import graph as g
import deterministic_loss as dl
import numpy as np,json,sys,time,gc
import torch
def test(name,values,critical,y,w,repeats=40):
    py=torch.from_numpy(y).to('mps');pw=torch.from_numpy(w).to('mps');idx=torch.from_numpy(critical).to('mps');wp,wn=dl.weights(critical,y,w,values.shape[1]);x=values.astype(float);reference_loss=float(np.mean((np.logaddexp(0,-x)*wp+np.logaddexp(0,x)*wn).sum(1)));reference_gradient=((wp+wn)/(1+np.exp(-x))-wp)/len(x);result={}
    for mode in ['gather','aggregated']:
        hashes=[];gradients=[];losses=[]
        for _ in range(repeats):
            logits=torch.tensor(values,device='mps',requires_grad=True)
            loss=e.balanced(torch.gather(logits,1,idx),py,pw).mean() if mode=='gather' else dl.from_critical(logits,critical,y,w).mean();loss.backward();grad=logits.grad.detach().cpu().numpy().copy();hashes.append(c.ah(grad));gradients.append(grad);losses.append(float(loss.detach().cpu()));del logits,loss
        diff=max(float(np.max(abs(x-gradients[0]))) for x in gradients);gold=max(float(np.max(abs(x-reference_gradient))) for x in gradients);lf=max(abs(x-reference_loss) for x in losses);assert gold<1e-6 and lf<1e-5;result[mode]=dict(repetitions=repeats,unique_gradient_hashes=len(set(hashes)),maximum_repeat_gradient_difference=diff,maximum_reference_gradient_error=gold,maximum_reference_loss_error=lf)
    return dict(case=name,shape=list(values.shape),pairs_per_source=critical.shape[1],duplicate_fraction=float(1-sum(len(np.unique(v)) for v in critical)/critical.size),modes=result)
def main():
    start=time.monotonic();z,d,pos=e.load();tr=np.array(d['graphs'][0]['A']);ti=np.array([pos[int(i)] for i in tr]);s=e.subset(z,ti);norm=e.meta(z,ti,d['ids'][tr]);m=e.init(e.SEEDS[0]);e.seed_all(e.SEEDS[0]+41);rng=np.random.default_rng(e.SEEDS[0]+41);order=rng.permutation(len(tr));ix=np.array_split(order,int(np.ceil(len(tr)/e.BATCH)))[0];pack=e.batch(s,ix,norm,rng);m.train();outputs=m(torch.from_numpy(pack[0]).to('mps'));values=outputs[1].detach().cpu().numpy().copy();pair,py,pw=pack[5:];critical=np.stack([g.critical_edges(v,p) for v,p in zip(values,pair)]);initial=e.state_hash(m.state_dict());del outputs,m;gc.collect();torch.mps.empty_cache();records=[test('actual_initial_training_batch',values,critical,py,pw)]
    # Numeric stress fixture for the operator, not a training image or model.
    rng=np.random.default_rng(882);values=rng.normal(size=(16,1984)).astype(np.float32);critical=np.zeros((16,64),np.int64);y=rng.integers(0,2,(16,64)).astype(np.float32);w=rng.uniform(.1,1,(16,64)).astype(np.float32);records.append(test('duplicate_edge_numeric_fixture',values,critical,y,w))
    c.save(_repo_io('gradient-repeatability.json'),dict(complete=True,no_optimizer_steps=True,no_new_training_images=True,hardness_read=False,training_source_ids=d['ids'][tr[ix]].tolist(),initial_model_state_sha256=initial,cases=records,seconds=time.monotonic()-start,prototype_used_for_original_OOF=False,source_sha256=c.sha(__file__),prototype_sha256=c.sha(dl.__file__)));print(json.dumps(records,indent=2),flush=True)
if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.27);sys.addaudithook(c.guard)
    with c.threadpool_limits(limits=2):main()
