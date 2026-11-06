"""A matched raw/denoised-input residual-CNN test with one CPU thread."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import argparse
import gc
import json
import resource
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
import probe as p
import noise_calibration as nc
HERE=p.HERE;e=p.old.e;r=p.r;Original=e.c.ComplementCNN
ARMS=('duplicate','gaussian')


class TwoStream(Original):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        self.blocks[0][0]=nn.Conv2d(2,16,7,stride=4,padding=3)
        x=torch.arange(-2,3,dtype=torch.float32);g=torch.exp(-x*x/2);g/=g.sum()
        self.register_buffer('smooth_kernel',(g[:,None]*g[None,:])[None,None])

    def forward(self,x,context,arm):
        smooth=F.conv2d(F.pad(x,(2,2,2,2),mode='reflect'),self.smooth_kernel) if self.kind=='gaussian' else x
        stats=[x.mean((2,3)),(x.var((2,3),correction=0)+1e-6).sqrt()]
        h=torch.cat([x,smooth],1)
        for block in self.blocks:
            h=block(h);stats.extend([h.mean((2,3)),(h.var((2,3),correction=0)+1e-6).sqrt()])
        image=torch.cat(stats,1)
        return self.main(torch.cat([image,context],1)).squeeze(1),self.aux(image).squeeze(1)


def train(repeat):
    p.protect();p.sys.addaudithook(p.old.no_test)
    torch.manual_seed(2026101501);a=TwoStream('duplicate')
    torch.manual_seed(2026101501);b=TwoStream('gaussian')
    assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
    parameters=sum(v.numel() for v in a.parameters());assert parameters==69572
    assert torch.allclose(b.smooth_kernel.sum(),torch.tensor(1.))
    # Smoothing preserves a constant, including at the image boundary.
    ones=torch.ones(1,1,16,16)
    assert torch.max(torch.abs(F.conv2d(F.pad(ones,(2,2,2,2),mode='reflect'),b.smooth_kernel)-1))<1e-6
    del a,b;gc.collect()
    p.save(_repo_io('stream-config.json'),dict(parameters=parameters,matched_initialization=True,sigma=1,kernel=5,
        padding='reflect',source_fitter_sha256=r.digest(p.R4/'cnn_fitter.py'),CPU_threads=1,
        target='same honest full073 non021 residual as the current correction',outer_baseline='strict076'))
    d=e.c.load_data();labels=d.pop('y');graphs=json.loads((p.R4/'label-graph.json').read_text())
    for k,g in enumerate(graphs):
        for stage in ('selector','final'):
            data,contract=p.old.prepare(d,labels,g,stage);base_meta=data['meta']
            for arm in ARMS:
                data['meta']={**base_meta,'input_representation':arm,'input_channels':2,'gaussian_sigma':1,'CPU_threads':1}
                path=HERE/f'stream/{arm}-seed{repeat}/fold{k}-{stage}';path.parent.mkdir(parents=True,exist_ok=True)
                seed=2026101501+100*k+100000*repeat
                e.c.ComplementCNN=lambda:TwoStream(arm)
                if not (path/'training.json').exists():e.c.fit_one(data,'aux',seed,path)
                record=json.loads((path/'training.json').read_text());assert record['meta']==data['meta'] and record['seed']==seed
                assert set(record['meta']['train_ids']).isdisjoint(record['meta']['validation_ids'])
                ck=torch.load(_repo_io(path/'model.pt'),map_location='cpu',weights_only=True)
                model=TwoStream(arm);model.load_state_dict(ck['state_dict'])
                pred=e.c.predict(model,data['images'],data['context'],data['valid'],'aux')*data['meta']['target_std']
                saved=np.load(_repo_io(path/'predictions.npz'));assert np.array_equal(pred,saved['validation_prediction'])
                peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                p.save(_repo_io(str(path.relative_to(HERE)/'verification.json')),dict(reload_exact=True,peakRSS_bytes=peak,
                    script_sha256=r.digest(__file__),plan_sha256=r.digest(HERE/'STREAM-PLAN.md')))
                print('STREAM COMPLETE',arm,repeat,k,stage,'peakMB',round(peak/1048576),flush=True)
                del model,ck;gc.collect()
                assert peak<1024**3,'PeakRSS exceeded1GiB; do not launch more fits'
            del data;gc.collect()
    p.protect();p.save(_repo_io(f'stream-seed{repeat}-complete.json'),dict(complete=True,repeat=repeat,arms=list(ARMS)))


def evaluate(repeats):
    from confidence import compare
    p.protect();d=dict(np.load(_repo_io(p.R4.parent/'inputs.npz')));z=dict(np.load(_repo_io(HERE/'baseline076.npz')))
    y=d['y'][d['dev']];fold=z['fold'];reference=np.load(_repo_io(HERE/'probe-oof.npz'))['old4']
    frozen=json.loads((p.R4/'noise-calibration/summary.json').read_text())['aux']['selection']
    out={'current':reference};records={}
    for family in ('legacy',*ARMS):
        out[family+'_matched']=z['p76'].copy();out[family+'_frozen']=z['p76'].copy();records[family]=[]
        for k in range(5):
            inner=np.load(_repo_io(p.R4/f'aux-mean2/fold{k}-selection.npz'));B=inner['B']
            means=[]
            for stage in ('selector','final'):
                ps=[]
                for j in repeats:
                    path=p.R4/f'aux-seed{j}/fold{k}-{stage}/predictions.npz' if family=='legacy' else HERE/f'stream/{family}-seed{j}/fold{k}-{stage}/predictions.npz'
                    q=np.load(_repo_io(path));ps.append(q['validation_prediction'])
                    if stage=='selector':assert np.array_equal(q['validation_index'],B)
                    else:assert np.array_equal(q['validation_index'],z['dev'][fold==k])
                means.append(np.mean(ps,axis=0))
            rb,rv=means;good=inner['eligible'];ix=np.flatnonzero(fold==k)
            w,meta=nc.select(rb[good],(d['y'][B]-inner['base073'])[good],d['noise'][B][good]>=12)
            poison=d['y'].copy();poison[z['dev'][ix]]+=10000
            ww,mm=nc.select(rb[good],(poison[B]-inner['base073'])[good],d['noise'][B][good]>=12)
            assert np.array_equal(w,ww) and meta==mm
            eligible=(z['noise'][ix]>=12)&~z['failed'][ix]
            out[family+'_matched'][ix[eligible]]+=w[1]*rv[eligible]
            out[family+'_frozen'][ix[eligible]]+=frozen[k]['addition']['weights'][1]*rv[eligible]
            records[family].append(dict(fold=k,weights=w.tolist(),**meta,outer_poison_exact=True))
    if repeats==[0,1]:assert np.max(np.abs(out['legacy_matched']-reference))<1e-10
    vs_current={name:compare(reference,v,y,fold) for name,v in out.items()}
    gaussian_vs_legacy=compare(out['legacy_matched'],out['gaussian_matched'],y,fold)
    gaussian_vs_duplicate=compare(out['duplicate_matched'],out['gaussian_matched'],y,fold)
    frozen_gain=compare(out['legacy_frozen'],out['gaussian_frozen'],y,fold)
    advance=all(q['gain']>=.01 and sum(x>1e-10 for x in q['fold_gains'])>=3 for q in (gaussian_vs_legacy,gaussian_vs_duplicate)) and frozen_gain['gain']>0
    if repeats==[0,1]:
        q=vs_current['gaussian_matched'];advance=bool(advance and q['ci95'][0]>0 and sum(x>1e-10 for x in q['fold_gains'])>=3)
    scores={name:dict(rmse=r.rmse(v,y),folds=[r.rmse(v[fold==k],y[fold==k]) for k in range(5)]) for name,v in out.items()}
    tag=''.join(map(str,repeats))
    result=dict(repeats=repeats,scores=scores,vs_current=vs_current,gaussian_vs_legacy=gaussian_vs_legacy,
        gaussian_vs_duplicate=gaussian_vs_duplicate,frozen_weight_effect=frozen_gain,advance=bool(advance),selections=records,
        limitation='Same400 previously explored images; strict076 base, but CNN targets and inner calibration match the existing073 correction protocol.')
    p.save(_repo_io(f'stream-results-{tag}.json'),result)
    np.savez_compressed(_repo_io(HERE/f'stream-oof-{tag}.npz'),ids=z['ids'],fold=fold,y=y,**out)
    print(json.dumps(dict(scores=scores,gaussian_vs_legacy=gaussian_vs_legacy,gaussian_vs_duplicate=gaussian_vs_duplicate,
                         frozen_weight_effect=frozen_gain,advance=bool(advance)),indent=2),flush=True)
    p.protect()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['train','evaluate']);parser.add_argument('--repeat',type=int,choices=[0,1],default=0)
    parser.add_argument('--repeats',type=int,nargs='+',default=[0]);args=parser.parse_args()
    assert args.repeats in ([0],[0,1])
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):train(args.repeat) if args.mode=='train' else evaluate(args.repeats)
