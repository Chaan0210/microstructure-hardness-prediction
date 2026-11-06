"""Safely reproduce preserved 061/073 submissions; never write shared artifacts.

Default: byte-exact reconstruction from frozen component-prediction artifacts.
--refit-linear: actually refit and save the 15 Ridge/log-Ridge models from train
features, verify them against production, then use their predictions. All image,
TabPFN, grain-set and GMM components remain frozen in this mode. This is not an
end-to-end neural retraining claim; original image-pipeline instructions are
linked in REPORT.md. Test features are passed only to predict, after each fit.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
import research as r

VARIANTS=('u1dm1szH2Z','u1dm2szH2Z','u1dmavgH2Z','u1dmavg4H2Z')


def main(refit=False,output_name=None):
    name=output_name or ('retrained-linear' if refit else 'reproduction')
    if Path(name).name!=name or name in ('.','..'):
        raise ValueError('Output name must be a new directory name inside this experiment.')
    outdir=r.HERE/name
    if outdir.exists(): raise FileExistsError(f'Refusing to overwrite {outdir}')
    outdir.mkdir()
    used={}

    def track(path):
        path=Path(path); used[str(path.relative_to(r.ROOT))]=r.digest(path); return path

    order=[v['ID'] for v in csv.DictReader(track(_repo_path(r.ROOT, 'data/sample_submission.csv')).open())]

    def read_csv(name):
        rows=list(csv.DictReader(track(_repo_path(r.ROOT, f'artifacts/submission-candidate-{name}.csv')).open()))
        assert [v['ID'] for v in rows]==order
        return np.array([float(v['hardness']) for v in rows])

    def read_npz(path):
        z=np.load(_repo_io(track(path)),allow_pickle=True)
        if 'ids' in z: assert list(z['ids'].astype(str))==order
        return {k:z[k] for k in z.files}

    c53=read_csv('053'); p021=read_csv('021')
    original_none=read_npz(r.EXP/'push-1200/fin-none.npz')
    original_aug=read_npz(r.EXP/'push-1200/fin-d1234n12.npz')
    pn=read_npz(r.H13/'fin-nb-partners.npz')
    tn=read_npz(r.H13/'fin-nb-tab-cpu.npz')
    rr=read_npz(r.EXP/'probabilistic-grains-900/ridge/final500/predictions.npz')
    ss=read_npz(r.EXP/'probabilistic-grains-900/set_model/full500/predictions.npz')
    failed=rr['failed']|ss['failed']
    plain,aug=dict(original_none),dict(original_aug)
    refit_audit=[]
    if refit:
        labels={v['ID']:float(v['hardness']) for v in csv.DictReader(track(_repo_path(r.ROOT, 'data/train.csv')).open())}
        groups=json.loads(track(r.H13/'abl/groups.json').read_text())
        drop=set((groups['het']+' '+groups['intensity']).split())
        predictions={}
        for condition in ('full','nb','aug'):
            logs=[]
            for variant in VARIANTS:
                fp=r.EXP/f'push-800/feat-{variant}-anghcor-train.npz'
                f=np.load(_repo_io(track(fp)),allow_pickle=True)
                keys=list(f['keys'].astype(str)); ids=f['ids'].astype(str)
                x=f['X'].astype(float); y=np.array([labels[i] for i in ids])
                cols=[j for j,k in enumerate(keys) if condition!='nb' or k not in drop]
                if condition=='aug':
                    xx=[x]; yy=[y]
                    for tag in ('deg','deg2','deg3','deg4'):
                        s=np.load(_repo_io(track(r.EXP/f'push-1200/synthrows-{tag}-{variant}-anghcor.npz')),allow_pickle=True)
                        assert list(s['keys'].astype(str))==keys
                        use=s['X'][:,keys.index('noise')].astype(float)*255>=12
                        xx.append(s['X'].astype(float)[use]); yy.append(np.array([labels[str(i)] for i in s['ids'][use]]))
                    x=np.vstack(xx); y=np.concatenate(yy)
                for kind in (('log','ridge') if variant=='u1dmavgH2Z' else ('log',)):
                    model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=(kind=='ridge')),
                        StandardScaler(),RidgeCV(alphas=[.1,1,10,100,1000]) if kind=='ridge' else Ridge(alpha=2.))
                    model.fit(x[:,cols], y if kind=='ridge' else np.log(y))
                    modelpath=outdir/f'{condition}-{variant}-{kind}.joblib'; joblib.dump(model,_repo_io(modelpath))
                    te=np.load(_repo_io(track(r.EXP/f'push-800/feat-{variant}-anghcor-test.npz')),allow_pickle=True)
                    assert list(te['ids'].astype(str))==order and list(te['keys'].astype(str))==keys
                    z=te['X'].astype(float)[:,cols]
                    prediction=model.predict(z)
                    reload_diff=float(np.max(np.abs(joblib.load(_repo_io(modelpath)).predict(z)-prediction)))
                    assert reload_diff==0.
                    refit_audit.append(dict(condition=condition,variant=variant,kind=kind,n_train_rows=len(y),
                        n_features=len(cols),alpha=float(model[-1].alpha_ if kind=='ridge' else model[-1].alpha),reload_difference=reload_diff,
                        training_sources='provided train only; aug adds deg/deg2/deg3/deg4 source labels',test_fit=False))
                    if kind=='log': logs.append(prediction)
                    else: predictions[condition+'_ridge']=prediction
            predictions[condition+'_logridge']=np.exp(np.mean(logs,axis=0))
        checks={}
        for kind in ('ridge','logridge'):
            for name,reference in [('full',original_none[kind]),('aug',original_aug[kind]),('nb',pn[kind+'_nb'])]:
                diff=float(np.max(np.abs(predictions[name+'_'+kind]-reference)))
                checks[name+'_'+kind]=diff; assert diff<1e-7,(name,kind,diff)
            plain[kind]=predictions['full_'+kind]; aug[kind]=predictions['aug_'+kind]
            pn[kind+'_full']=predictions['full_'+kind]; pn[kind+'_nb']=predictions['nb_'+kind]
        r.save_json(outdir/'fit-validation.json',dict(models=refit_audit,production_max_differences=checks))
    gate=(np.nan_to_num(aug['noise'],nan=0)>=12)&~failed
    c61=c53+gate*(.128*(aug['ridge']-plain['ridge'])+.128*(aug['logridge']-plain['logridge']))
    full={slot:(read_csv(slot)-.25*p021)/.75 for slot in ('038','039','044')}
    tab_delta=np.mean([tn['nb_'+slot]-full[slot] for slot in full],axis=0)
    delta=.384*tab_delta+.128*(pn['ridge_nb']-pn['ridge_full'])+.128*(pn['logridge_nb']-pn['logridge_full'])+.15*(pn['gmm_nb']-pn['gmm_full'])
    ng=(np.nan_to_num(aug['noise'],nan=0)<12)&~failed
    c73=np.where(ng,c61+delta,c61)
    outputs={}
    for name,pred in [('061',c61),('073',c73)]:
        assert len(pred)==len(order)==1000 and len(set(order))==1000 and np.isfinite(pred).all()
        assert np.array_equal(pred[failed],p021[failed])
        path=outdir/f'submission-{name}.csv'
        with path.open('x',newline='') as f:
            writer=csv.writer(f); writer.writerow(['ID','hardness'])
            writer.writerows((i,repr(float(v))) for i,v in zip(order,pred))
        reference=read_csv(name)
        maxdiff=float(np.max(np.abs(pred-reference)))
        same=r.digest(path)==r.digest(_repo_path(r.ROOT, f'artifacts/submission-candidate-{name}.csv'))
        assert maxdiff<1e-7
        if not refit: assert same, f'{name} is not byte-exact'
        outputs[name]=dict(file=str(path.relative_to(r.ROOT)),sha256=r.digest(path),
                           max_absolute_difference=maxdiff,byte_identical=same,n_rows=1000)
    recipe=dict(weights=dict(p021=.16,tabpfn=.384,ridge=.128,logridge=.128,grain_set=.05,gmm=.15),
                gate_noise_gray=12,failed_fallback='p021',tabpfn=dict(version='3.5',local_only=True,slots=['038','039','044'],
                variants=list(VARIANTS),seeds=[0,100,200],aggregation='uniform arithmetic mean'),
                ridge=dict(alphas=[.1,1,10,100,1000],selection='GCV',impute='train median',scaler='train standardization'),
                logridge=dict(alpha=2,variants=list(VARIANTS),aggregation='geometric mean',smearing=False),
                augmentation=dict(sets=['deg','deg2','deg3','deg4'],measured_noise_min=12,source='train only'),
                feature_drop_073='het and intensity: 13 columns, applied only to predictions with noise<12',
                mode='refit only Ridge/logRidge; other components frozen' if refit else 'frozen component prediction reconstruction',
                no_upload=True,no_test_fitting=True,outputs=outputs,input_sha256=used,code_sha256=r.digest(__file__))
    r.save_json(outdir/'manifest.json',recipe)
    print(json.dumps(outputs,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--refit-linear',action='store_true')
    ap.add_argument('--output-name',help='A new child directory name; existing directories are never overwritten.')
    args=ap.parse_args()
    with threadpool_limits(limits=2): main(args.refit_linear,args.output_name)
