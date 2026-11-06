"""Reproduce descriptive auxiliary uncertainty and OOF calibration summaries."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import study as s
import numpy as np,json,sys
def main():
    m=json.loads((s.HERE/'measurement-screen.json').read_text());rng=np.random.default_rng(2026101637)
    rr=np.array([[x[k]['phase_error2'] for x in m['per_image']] for k in ['raw','corrected']]);draw=rng.integers(0,30,(10000,30));gain=np.sqrt(rr[0,draw].mean(1))-np.sqrt(rr[1,draw].mean(1))
    z=np.load(_repo_io(s.HERE/'all-predictions.npz'));b=z['baseline'];y=z['y'];noise=z['noise'];records={}
    for key,ix in [('all',np.arange(400)),('high',np.where(noise>=12)[0])]:
        pred=b[ix];target=y[ix];D=np.c_[np.ones(len(ix)),pred-pred.mean()];coef=np.linalg.lstsq(D,target,rcond=None)[0]
        dr=rng.integers(0,len(ix),(10000,len(ix)));xx=pred[dr];yy=target[dr];xx=xx-xx.mean(1,keepdims=True);yy=yy-yy.mean(1,keepdims=True);sl=(xx*yy).sum(1)/(xx*xx).sum(1)
        cuts=np.quantile(pred,[.2,.4,.6,.8]);q=np.digitize(pred,cuts);groups=[]
        for k in range(5):
            sel=q==k;groups.append(dict(quintile=k,n=int(sel.sum()),mean_prediction=float(pred[sel].mean()),mean_target=float(target[sel].mean()),mean_residual=float((pred-target)[sel].mean())))
        records[key]=dict(slope=float(coef[1]),conditional_slope_ci95=np.percentile(sl,[2.5,97.5]).tolist(),prediction_quintiles=groups)
    obj=dict(auxiliary_phase_rmse_raw=float(np.sqrt(rr[0].mean())),auxiliary_phase_rmse_corrected=float(np.sqrt(rr[1].mean())),conditional_phase_rmse_gain_ci95=np.percentile(gain,[2.5,97.5]).tolist(),aux_images_improved=int((rr[1]<rr[0]).sum()),calibration_diagnostic=records,warning='Exploratory descriptions; target-dependent inspection never used to fit or choose a new calibrator. Intervals condition on already selected predictions.')
    path=s.HERE/'measurement-and-calibration-uncertainty.json'
    if path.exists():assert json.loads(path.read_text())==obj
    else:s.save(_repo_io(path.name),obj)
    print('DESCRIPTIVE UNCERTAINTY REPRODUCED',flush=True)
if __name__=='__main__':
    sys.addaudithook(s.guard)
    with s.threadpool_limits(limits=1):main()
