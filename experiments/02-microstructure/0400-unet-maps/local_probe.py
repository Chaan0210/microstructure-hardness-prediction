from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, csv, sys
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import spearmanr
P='../../'
r=list(csv.DictReader(open(_repo_io(P+'data/train.csv')))); allids=[x['ID'] for x in r]; yd=dict((x['ID'],float(x['hardness'])) for x in r)
F=np.load(_repo_io('../push-300/feat-a1s250n4vstd-stcoh-train.npz'),allow_pickle=True); ids=[str(i) for i in F['ids']]; keys=[str(k) for k in F['keys']]; X=F['X']
Y=np.array([yd[i] for i in ids]); nz=np.load(_repo_io('../push-300/noise_train.npy')); nzd=dict(zip(allids,nz)); Z=np.array([nzd[i] for i in ids]); t=np.digitize(Z,np.quantile(Z,[1/3,2/3]))
base=np.stack([np.load(_repo_io(f'../push-300/oof_none_v2_feat-a1s250n4vstd-stcoh-train_s{s}.npy')) for s in range(3)]).mean(0).mean(0); e=Y-base
M=np.load(_repo_io('../grain-cellpose-220/masks-v3-train.npz'))['masks']
yy,xx=np.mgrid[0:256,0:256]
centers={'center':(127.5,127.5),'tl':(48,48),'br':(207,207),'tr':(48,207)}
radii=[16,32,48,64,96]
out={c:{R:[] for R in radii} for c in centers}; outsz={c:{R:[] for R in radii} for c in centers}
for k,i in enumerate(ids):
    a=np.array(Image.open(_repo_io(P+f'data/train/{i}.png'))).astype(float)/255; m=M[k].astype(int)
    sm=ndi.gaussian_filter(a,1.0)
    inner=m*(ndi.minimum_filter(m,3)==ndi.maximum_filter(m,3)); lab=np.unique(m); lab=lab[lab>0]
    cnt=ndi.sum(np.ones_like(a),inner,lab); I=ndi.sum(sm,inner,lab)/np.maximum(cnt,1); A=ndi.sum(np.ones_like(a),m,lab)
    okg=cnt>=3
    if okg.sum()<5:
        for c in centers:
            for R in radii: out[c][R].append(np.nan); outsz[c][R].append(np.nan)
        continue
    o=np.argsort(I[okg]); w=A[okg][o]; cc=np.cumsum(w)/w.sum(); hi=I[okg][o][np.searchsorted(cc,0.75)]
    lo=np.percentile(sm,0.1); z=(hi-I)/max(hi-lo,1e-3)
    lut_sp=np.zeros(m.max()+1); lut_sp[lab]=((z>0.3)&(z<=0.75)&okg).astype(float)
    lut_la=np.zeros(m.max()+1); lut_la[lab]=np.log(A)
    spmap=lut_sp[m]; lamap=lut_la[m]; fg=(m>0)
    for c,(cy,cx) in centers.items():
        d2=(yy-cy)**2+(xx-cx)**2
        for R in radii:
            disk=(d2<=R*R)&fg
            out[c][R].append(spmap[disk].mean() if disk.sum() else np.nan); outsz[c][R].append(lamap[disk].mean() if disk.sum() else np.nan)
g=X[:,keys.index('grain_eqd_mean')]
sel={'clean+mid':t<=1,'clean+mid coarse40%':(t<=1)&(g>np.quantile(g[t<=1],0.6)),'clean+mid fine40%':(t<=1)&(g<np.quantile(g[t<=1],0.4))}
for c in centers:
    for R in radii:
        v=np.array(out[c][R]); s=np.array(outsz[c][R])
        print(c.ljust(6),'R',R, ' '.join(f'{n}: sp~res {spearmanr(v[q],e[q],nan_policy="omit")[0]:+.3f} sz~res {spearmanr(s[q],e[q],nan_policy="omit")[0]:+.3f} |' for n,q in sel.items()),flush=True)
np.savez(_repo_io('local_probe.npz'),**{f'{c}_{R}_sp':np.array(out[c][R]) for c in centers for R in radii},**{f'{c}_{R}_la':np.array(outsz[c][R]) for c in centers for R in radii})
