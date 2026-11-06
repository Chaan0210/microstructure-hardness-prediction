"""Bounded080 audit; all new artifacts local, all old artifacts read-only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
import sys
sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
R4=_repo_path(ROOT, 'experiments/rmse-research-20261015/round4-full073')
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name,'2')
os.environ['MPLCONFIGDIR']=str(_repo_path(HERE, 'mpl-cache'))
os.environ.setdefault('TABPFN_DISABLE_TELEMETRY','1');os.environ.setdefault('HF_HUB_OFFLINE','1')
sys.path.insert(0,str(R4));sys.path.insert(0,str(R4.parent))
import argparse
import csv
import hashlib
import json
import signal
import time
import numpy as np
from threadpoolctl import threadpool_limits
import research as r

ANCHORS=(0,1)
SPLIT_SEED=2026101608
PERMUTATIONS=64


def save(name,value):
    path=_repo_path(HERE, name);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def load_inputs():return dict(np.load(_repo_path(R4.parent, 'inputs.npz')))


def protect(paths):
    p=_repo_path(HERE, 'protected.json');old=json.loads(p.read_text()) if p.exists() else {}
    for path in paths:
        path=Path(path);key=str(path.relative_to(ROOT));sha=r.digest(path)
        if key in old:assert old[key]==sha,('protected source changed',key)
        else:old[key]=sha
    save('protected.json',old)


def check_protected():
    for path,digest in json.loads((_repo_path(HERE, 'protected.json')).read_text()).items():
        assert r.digest(_repo_path(ROOT, path))==digest,path


def metric(base,new,y):
    a=(base-y)**2;b=(new-y)**2;gain=a-b;order=np.argsort(gain)[::-1]
    result=dict(base_rmse=float(np.sqrt(a.mean())),new_rmse=float(np.sqrt(b.mean())),
                gain=float(np.sqrt(a.mean())-np.sqrt(b.mean())),delta_rms=float(np.sqrt(np.mean((new-base)**2))),
                delta_mse=float((b-a).mean()),net_sse_gain=float(gain.sum()))
    for count in (5,10):
        keep=order[count:]
        result[f'top{count}_net_gain_share']=float(gain[order[:count]].sum()/gain.sum()) if gain.sum()!=0 else None
        result[f'gain_without_top{count}']=float(np.sqrt(a[keep].mean())-np.sqrt(b[keep].mean()))
    return result


def audit():
    submissions=_repo_path(ROOT, 'artifacts');report_dir=_repo_path(ROOT, 'reports');h13=_repo_path(ROOT, 'experiments/push-1300')
    paths=[submissions/f'submission-candidate-{n}.{s}' for n in ('061','073','076','078','080') for s in ('csv','manifest.json')]
    paths += [report_dir/'candidate080-public-submission.json',report_dir/'candidate076-public-submission.json',
              h13/'public_scores.csv',h13/'make_080.py',h13/'make_078.py',h13/'make_076.py',h13/'audit_rescnn.py',
              _repo_path(R4, 'final500/candidate-aux-noise-addition.csv'),_repo_path(R4, 'final500/recipe.json'),_repo_path(R4, 'final500/test-predictions.npz'),
              _repo_path(R4, 'final500/inference-verification.json'),_repo_path(R4, 'noise-calibration/aux-oof.npz'),
              _repo_path(R4, 'noise-calibration/context-oof.npz'),_repo_path(R4, 'noise-calibration/summary.json'),
              _repo_path(R4, 'simultaneous-bands.json'),_repo_path(R4.parent, 'inputs.npz'),h13/'dev-078.npy',h13/'dev-nb-candidate.npy']
    protect(paths)
    order=[row['ID'] for row in csv.DictReader((_repo_path(ROOT, 'data/sample_submission.csv')).open())]
    def read(path):
        rows=list(csv.DictReader(path.open()));assert list(rows[0])==['ID','hardness']
        assert [row['ID'] for row in rows]==order and len(set(order))==len(order)==1000
        out=np.array([float(row['hardness']) for row in rows]);assert np.isfinite(out).all();return out
    pred={n:read(submissions/f'submission-candidate-{n}.csv') for n in ('073','076','078','080')}
    corr=read(_repo_path(R4, 'final500/candidate-aux-noise-addition.csv'))-pred['073']
    saved=np.load(_repo_path(R4, 'final500/test-predictions.npz'));assert saved['ids'].tolist()==order
    recipe=json.loads((_repo_path(R4, 'final500/recipe.json')).read_text());w=np.array(recipe['production_weights']['addition'])
    rebuilt=w[(saved['noise']>=12).astype(int)]*saved['mean_correction'];rebuilt[saved['failed']]=0
    assert np.max(np.abs(corr-rebuilt))<1e-10
    composition_error=float(np.max(np.abs(pred['080']-(pred['078']+corr))));assert composition_error<1e-10
    changed=np.abs(corr)>1e-9;gate=(saved['noise']>=12)&~saved['failed']
    assert not np.any(changed&~gate)
    pub80=json.loads((report_dir/'candidate080-public-submission.json').read_text())
    pub76=json.loads((report_dir/'candidate076-public-submission.json').read_text())
    assert pub80['CSV_sha256']==r.digest(submissions/'submission-candidate-080.csv')
    assert pub76['CSV_sha256']==r.digest(submissions/'submission-candidate-076.csv')
    m80=json.loads((submissions/'submission-candidate-080.manifest.json').read_text())
    assert m80['correction_source_sha256']==r.digest(_repo_path(R4, 'final500/candidate-aux-noise-addition.csv'))
    ledger=list(csv.DictReader((h13/'public_scores.csv').open()));p78=[row for row in ledger if row['cand']=='078']
    assert not p78,'Re-evaluate the public contrast if a real078 score is found'
    shift=pred['078']-pred['076'];assert np.max(np.abs(shift[gate]))<1e-10
    d=load_inputs();ids=d['ids'][d['dev']];y=d['y'][d['dev']]
    oldbase=np.load(_repo_path(ROOT, 'experiments/probabilistic-grains-900/base.npz'),allow_pickle=True)
    assert np.array_equal(oldbase['ids'],ids) and np.array_equal(oldbase['fold'],d['fold']) and np.array_equal(oldbase['y'],y)
    aux=np.load(_repo_path(R4, 'noise-calibration/aux-oof.npz'));ctx=np.load(_repo_path(R4, 'noise-calibration/context-oof.npz'))
    assert np.array_equal(aux['ids'],ids) and np.array_equal(aux['fold'],d['fold'][0])
    delta=aux['addition']-aux['baseline'];p73=np.load(_repo_io(h13/'dev-nb-candidate.npy'))[0];p78=np.load(_repo_io(h13/'dev-078.npy'))[0]
    mask=(d['noise'][d['dev']]>=12)&~d['fail'][0]
    x=aux['baseline'][mask]-aux['baseline'][mask].mean();slope=float(x@delta[mask]/(x@x))
    stripped=delta.copy();stripped[mask]-=slope*x
    oof={name:metric(base,base+delta,y) for name,base in [('strict073',aux['baseline']),('recorded073',p73),('recorded078',p78)]}
    oof['spread_projection_diagnostic']=dict(slope=slope,**metric(aux['baseline'],aux['baseline']+stripped,y),
        limitation='Projection removes one linear direction on these same predictions; not evidence of information on new data.')
    oof['matched_no_image_control']=dict(**metric(ctx['addition'],aux['addition'],y),
        limitation='Passing this specific control does not establish independent generalization or rule out other tabular learners.')
    oof['transport_interaction_mse']=float(2*np.mean(delta*(p78-p73)))
    oof['historical_baseline_limit']='078 NPY is aligned through its generating code and shared base400 IDs, but is not a fresh strict-source078 refit; do not relabel it as such.'
    exposure=[];hold=set(d['ids'])-set(ids)
    for seed in (2026101501,2026101502,2026101503):
        directory=_repo_path(R4.parent, f'round2/full500-{seed}')
        train=set();valid=set()
        for path in sorted(directory.glob('fold[0-4].npz')):
            z=np.load(_repo_io(path));train.update(z['train_ids'].tolist());valid.update(z['validation_ids'].tolist())
        exposure.append(dict(stage=f'round2/round3 full500 partition{seed}',holdout_ids_in_training=len(hold&train),holdout_ids_evaluated=len(hold&valid)))
    production=set(recipe['training_ids'])
    outside=set(d['ids'][d['src']])-set(ids)
    result=dict(public=dict(score080=pub80['public_rmse'],score076=pub76['public_rmse'],
            difference080_minus076=pub80['public_rmse']-pub76['public_rmse'],score078=None,
            conclusion='080 is worse than076; marginal Public CNN effect080-minus078 is unobserved.',
            public_inverse_used=False),
        assembly=dict(all_ids_and_order_exact=True,composition_max_error=composition_error,
            changed_rows=int(changed.sum()),correction_rms=float(np.sqrt(np.mean(corr**2))),
            correction_mean_changed=float(corr[changed].mean()),mask_respected=True,
            updates078_vs076_and080_vs078_disjoint=True,max_cross_product=float(np.max(np.abs(shift*corr))),
            missing_ID_guards_in_make080=True,observed_ID_mismatch=False,
            manifest_uploaded_flag_is_stale=(m80.get('uploaded') is False)),
        oof=oof,holdout=dict(nondev_count=len(hold),earlier_exposure=exposure,
            full500_CNN_training_overlap=len(hold&production),historical_augmented_sources_outside_dev=len(outside),
            independent_holdout_available=False),
        status=dict(confirmed=['CSV composition and identities match','078 standalone Public is absent','100 non-dev images are previously exposed'],
            suspected=['Selection/sparse-error concentration','Refit and coefficient transport','073-trained correction applied to078'],
            unresolved=['Pure CNN Public effect','Independent new-data generalization','Which transport mechanism caused failure']))
    save('audit.json',result)
    np.savez_compressed(_repo_path(HERE, 'oof-audit.npz'),ids=ids,y=y,fold=d['fold'][0],strict073=aux['baseline'],recorded073=p73,recorded078=p78,correction=delta)
    check_protected();print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def extend_teachers():
    import torch
    import full_branch as b
    torch.set_num_threads(2)
    def timeout(*unused):raise TimeoutError('Pre-registered60min fitting limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(3600)
    d,y=b.load();graphs=json.loads((_repo_path(R4, 'label-graph.json')).read_text())
    b.HERE=_repo_path(HERE, 'extensions');b.HERE.mkdir(exist_ok=True)
    records=[]
    for k in ANCHORS:
        g=graphs[k];A=np.array(g['A']);B=np.array(g['B']);V=np.array(g['V']);query=np.r_[V,B[:8]]
        out=b.fit(d,A,query,y[A].copy(),k,'A-query-extension')
        prior=np.load(_repo_path(R4, f'teachers/fold{k}-outer0/predictions.npz'))
        assert np.array_equal(prior['query'],B) and np.array_equal(prior['train'],A)
        errors={key:float(np.max(np.abs(out[key][-8:]-prior[key][:8]))) for key in ('base','tab','ridge','logridge','gmm','set')}
        assert max(errors.values())<1e-4,errors
        records.append(dict(anchor=k,train_ids=d['ids'][A].tolist(),new_query_ids=d['ids'][V].tolist(),overlap8_max_errors=errors))
        save('extension-verification.json',records);print('EXTENSION COMPLETE',k,errors,flush=True)
    signal.alarm(0);check_protected()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['audit','extend']);args=parser.parse_args()
    with threadpool_limits(limits=2):audit() if args.mode=='audit' else extend_teachers()


if __name__=='__main__':main()
