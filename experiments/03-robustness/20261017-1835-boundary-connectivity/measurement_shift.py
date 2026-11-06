"""Same-image, same-training-size descriptor inclusion diagnostic; no fitting."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import numpy as np,json,sys,time
NAMES=['boundary_mean','boundary_std','boundary_q10','boundary_q50','boundary_q90','boundary_q95','boundary_fraction_gt05','boundary_score_entropy','affinity_mean','affinity_std','affinity_q10','affinity_q50','affinity_q90','affinity_score_entropy',*[f'connection_r{r}_{s}' for r in [1,2,4,8] for s in ['mean','std']],*[f'{s}_threshold{t}' for t in ['03','05','07'] for s in ['component_fraction','largest_fraction']]]
def main():
    start=time.monotonic();d=c.inputs();pos={int(v):i for i,v in enumerate(d['dev'])};high=(d['noise'][d['dev']]>=12)&~d['failed'];result={};saved={};draw=np.random.default_rng(2026101798).integers(0,int(high.sum()),(10000,int(high.sum())))
    for arm in ['topology','local','hardness','random']:
        train=np.zeros((400,28));query=np.zeros_like(train);nt=np.zeros(400,int);nq=np.zeros(400,int)
        for k in range(5):
            for repeat in [0,1]:
                path=c.HERE/(f'encoders/fold{k}-refit-seed{repeat}-'+('topology' if arm=='random' else arm));f=np.load(_repo_io(path/('random-features.npz' if arm=='random' else 'features.npz')));tr=np.array([pos[int(i)] for i in f['train']]);q=np.array([pos[int(i)] for i in f['query']]);assert len(tr)==320 and len(q)==80;train[tr]+=f['train_signature'];query[q]+=f['query_signature'];nt[tr]+=1;nq[q]+=1
        assert np.all(nt==8) and np.all(nq==2);train/=nt[:,None];query/=nq[:,None];delta=train-query;columns=[]
        for j,name in enumerate(NAMES):
            values=delta[high,j];columns.append(dict(feature=name,high_in_sample_mean=float(train[high,j].mean()),high_out_of_sample_mean=float(query[high,j].mean()),high_paired_difference=float(values.mean()),conditional_CI95=np.quantile(values[draw].mean(1),[.025,.975]).tolist()))
        result[arm]=columns;saved[arm+'_in_sample']=train;saved[arm+'_out_of_sample']=query
    c.save(_repo_io('measurement-shift.json'),dict(complete=True,results=result,training_size_all_models=320,in_sample_model_count_per_source=8,out_of_sample_model_count_per_source=2,source_images_identical_in_each_comparison=True,no_new_fitting=True,no_protocol_changes=True,seconds=time.monotonic()-start,limits='Descriptive same-source comparison. Different overlapping320-source folds and seeds also differ; conditional intervals are not independent training replications. Scores/entropy are not calibrated probabilities. A shift is consistent with measurement generalization, not proof of the sole hardness-error cause.'))
    c.arrays('measurement-shift.npz',ids=d['ids'][d['dev']],high=high,**saved)
    for arm,rows in result.items():print(arm,[(r['feature'],round(r['high_paired_difference'],5),r['conditional_CI95']) for r in rows if r['feature'] in ['boundary_score_entropy','affinity_score_entropy','component_fraction_threshold05','largest_fraction_threshold05']])
if __name__=='__main__':
    sys.addaudithook(c.guard)
    with c.threadpool_limits(limits=2):main()
