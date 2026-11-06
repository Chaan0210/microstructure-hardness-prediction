"""Check that hidden pixels cannot leak into context prediction."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import encoder as e
import numpy as np,torch,json,sys
def main():
    torch.set_num_threads(2);torch.manual_seed(17);model=e.ContextNet().eval()
    with torch.no_grad():model.decoder[-1].weight.normal_(0,.05)
    rng=np.random.default_rng(31);x=torch.from_numpy(rng.normal(size=(2,1,256,256)).astype(np.float32));indices=np.array([0,9]);mask=e.mask_batch(2,rng);checks=[]
    for arm in ['spatial','permuted']:
        perm=e.permutation(indices,arm);m=torch.from_numpy(mask);p=torch.from_numpy(perm);changed=x.clone();source_hidden=np.zeros((2,256),bool)
        for i in range(2):
            source_hidden[i,perm[i,mask[i]]]=True
            for j in np.flatnonzero(source_hidden[i]):changed[i,0,j//16*16:j//16*16+16,j%16*16:j%16*16+16]=123.456
        with torch.no_grad():a,ad=model(x,p,m);b,bd=model(changed,p,m)
        assert torch.equal(a,b) and torch.equal(ad,bd)
        xx=x.clone().requires_grad_(True);pred,_=model(xx,p,m);pred[m].square().mean().backward();grad=xx.grad.numpy().reshape(2,1,16,16,16,16).transpose(0,2,4,1,3,5).reshape(2,256,256);assert np.max(abs(grad[source_hidden]))==0 and np.max(abs(grad[~source_hidden]))>0
        checks.append(dict(arm=arm,hidden_pixel_change_prediction_exact=True,hidden_gradient_max=0.,visible_gradient_max=float(np.max(abs(grad[~source_hidden])))))
    # The decoder is image-context dependent, not a tokenwise independent map.
    xx=x[:1].clone().requires_grad_(True);perm=torch.arange(256)[None];mask=torch.zeros((1,256),dtype=torch.bool);mask[0,0]=True;pred,_=model(xx,perm,mask);pred[0,0].square().sum().backward();far=float(xx.grad[0,0,240:256,240:256].abs().sum());assert far>1e-12
    e.save(_repo_io('mathematical-checks.json'),dict(all_passed=True,checks=checks,far_corner_to_masked_corner_gradient=far,encoder_sha256=e.sha(e.HERE/'encoder.py'),source_sha256=e.sha(__file__)))
    print('HIDDEN PIXEL ISOLATION AND GLOBAL CONTEXT VERIFIED',far)
if __name__=='__main__':
    sys.addaudithook(e.guard)
    with e.threadpool_limits(limits=2):main()
