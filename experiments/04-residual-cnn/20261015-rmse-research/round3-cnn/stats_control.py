"""Same residual head with convolution slots masked; plus analytic mean residual."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import gc
import json
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import complement_cnn as c

Original=c.ComplementCNN


class StatsOnly(Original):
    def forward(self,x,context,arm):
        stats=torch.cat([x.mean((2,3)),(x.var((2,3),correction=0)+1e-6).sqrt()],1)
        image=torch.cat([stats,torch.zeros((len(x),320),dtype=x.dtype,device=x.device)],1)
        ctx=torch.zeros_like(context)
        return self.main(torch.cat([image,ctx],1)).squeeze(1),self.aux(image).squeeze(1)


def main():
    # Process-local class substitution preserves the original training source and
    # exactly reuses its initializer, target scaling, optimizer and augmentation.
    c.ComplementCNN=StatsOnly
    d=c.load_data();out={name:np.zeros((3,500))for name in ('base','stats','constant')};folds=np.zeros((3,500),int)
    for part in range(3):
        root=c.HERE/f'partition-{part}'
        for k in range(5):
            data=c.fold_data(d,part,k);output=root/f'fold{k}-stats_control'
            if(output/'training.json').exists():p=np.load(_repo_io(output/'predictions.npz'))['validation_prediction']
            else:p=c.fit_one(data,'residual',c.SEEDS[part]+k,output)
            va=data['valid'];base=data['base'][va];fail=data['failed'][va]
            stats=base+c.SHRINK*p;constant=base+c.SHRINK*float(np.mean(data['targets']['residual']))*data['meta']['target_std']
            stats[fail]=base[fail];constant[fail]=base[fail]
            out['base'][part,va]=base;out['stats'][part,va]=stats;out['constant'][part,va]=constant;folds[part,va]=k
            c.r.save_json(output/'control.json',dict(architecture='Same instantiated CNN with all convolution-derived feature slots masked to zero',
                active_image_summary_dimensions=2,context_used=False,forward_source_sha256=c.r.digest(__file__),training_source_sha256=c.r.digest(c.HERE/'complement_cnn.py')))
            print('stats control completed',part,k,flush=True);del data;gc.collect()
    ref=np.load(_repo_io(c.HERE/'confirmation-summary.npz'))
    assert np.array_equal(ref['fold'],folds)and np.array_equal(ref['base'],out['base'])
    result={key:c.r.compare(out['base'],value,d['y'],folds)for key,value in out.items()if key!='base'}
    result['image_CNN_vs_stats']=c.r.compare(out['stats'],ref['residual'],d['y'],folds)
    result['image_CNN_vs_constant']=c.r.compare(out['constant'],ref['residual'],d['y'],folds)
    np.savez_compressed(_repo_io(c.HERE/'mechanism-controls.npz'),ids=d['ids'],fold=folds,**out)
    c.r.save_json(c.HERE/'mechanism-controls.json',result)
    for key in result:
        q=result[key];print(key,q['candidate']['pooled'],q['gain'],q['partition_gains'],q['conditional_id_bootstrap_gain_ci95'],flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
