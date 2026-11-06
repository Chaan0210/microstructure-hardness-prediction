"""Confirm a preselected denoised panel using matching fresh-fold CNN predictions."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,csv,time,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, '.venv')).resolve():raise SystemExit('Use project .venv')
sys.path.insert(0,str(ROOT))
import numpy as np
import joblib
import frozen_validation as fv
import frozen_multiseed as fm
OUT=_repo_path(ROOT, 'artifacts/denoised-confirmation-023')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','evaluate']);a=parser.parse_args()
    if a.mode=='prepare':
        OUT.mkdir(exist_ok=False)
        fv.write(_repo_path(OUT, 'plan.json'),{'secondary_candidate':'compact26+denoised31 nested KRR, .5baseline+.25threeSeedD4CNN+.25kernel','split_seeds':[20261031,20261101],'same_fold_component_reuse':'Use newly trained run021 CNN/baseline only after asserting exact IDs, targets and fold memberships. Never use old original-fold OOF.','comparison':'candidate003 and compact26 mean3D4','selection_policy':'Prefer simpler compact26 unless both new partitions favor denoised extension; report fold variation and selection bias, no arbitrary numeric cutoff.','original_cv_rmse':13.53511407926948,'feature_report_sha256':fv.sha(_repo_path(ROOT, 'artifacts/run-022/report.json')),'feature_cache_sha256':fv.sha(_repo_path(ROOT, 'artifacts/run-022/development_features.npz')),'source_sha256':{n:fv.sha(_repo_path(ROOT, n)) for n in ['confirm_denoised.py','denoised_grain_features.py','compact_regression.py','frozen_validation.py']},'independent_holdout':False});print('Secondary plan frozen; no new-fold kernel evaluation performed');return
    plan=json.loads((_repo_path(OUT, 'plan.json')).read_text());assert not (_repo_path(OUT, 'report.json')).exists()
    for n,h in plan['source_sha256'].items():assert fv.sha(_repo_path(ROOT, n))==h
    assert fv.sha(_repo_path(ROOT, 'artifacts/run-022/report.json'))==plan['feature_report_sha256'] and fv.sha(_repo_path(ROOT, 'artifacts/run-022/development_features.npz'))==plan['feature_cache_sha256']
    parent=_repo_path(ROOT, 'artifacts/frozen-validation-021');parent_report=json.loads((parent/'report.json').read_text());assert [r['split_seed'] for r in parent_report['seeds']]==plan['split_seeds']
    ids,y,groups,_,_,_,panels,provenance=fv.read_inputs(['run-022']);panel=panels['run-022'];start=time.perf_counter();results=[]
    for seed in plan['split_seeds']:
        d=parent/f'seed-{seed}';s=json.loads((d/'split_manifest.json').read_text());assert set(s['development_folds'])==set(ids)
        with (d/'development_oof.csv').open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==400 and len({r['ID'] for r in rows})==400;by={r['ID']:r for r in rows};assert set(by)==set(ids)
        np.testing.assert_array_equal(y,[float(by[i]['hardness']) for i in ids]);fid=np.array([s['development_folds'][i] for i in ids]);assert all(int(by[i]['fold'])==s['development_folds'][i] for i in ids)
        values={n:np.array([float(by[i][n]) for i in ids]) for n in ['baseline','candidate003','mean3_identity','mean3_d4','cnn_mean3_d4']};new=np.full(400,np.nan);logs=[];dest=_repo_path(OUT, f'seed-{seed}');dest.mkdir()
        for k in range(5):
            tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);meta=json.loads((d/f'fold-{k}-report.json').read_text())
            assert set(meta['train_ids'])=={ids[i] for i in tr} and set(meta['validation_ids'])=={ids[i] for i in va};assert not set(groups[tr])&set(groups[va])
            new[va],model,summary=fv.kernel_fold(panel['X'],y,groups,panel['blocks'],tr,va,k)
            joblib.dump({'model':model,'feature_names':panel['names'],'blocks':panel['blocks'],'train_ids':[ids[i] for i in tr],'validation_ids':[ids[i] for i in va]},_repo_io(dest/f'kernel-fold{k}.joblib'));logs.append({'fold':k,**summary})
        values['denoised_kernel']=new;values['denoised_candidate']=.5*values['baseline']+.25*values['cnn_mean3_d4']+.25*new
        scores=fm.paired_summary(y,values,fid,seed)
        with (dest/'development_oof.csv').open('x',newline='') as f:
            w=csv.writer(f);w.writerow(['ID','hardness','fold',*values]);w.writerows([i,y[j],fid[j],*[v[j] for v in values.values()]] for j,i in enumerate(ids))
        result={'seed':seed,'scores':scores,'kernel_selection':logs};fv.write(dest/'report.json',result);results.append(result)
        print(json.dumps({'seed':seed,'denoised_rmse':scores['denoised_candidate']['oof_rmse'],'vs_compact3D4':scores['denoised_candidate']['paired_vs']['mean3_d4']}),flush=True)
    fv.write(_repo_path(OUT, 'report.json'),{'plan':plan,'seeds':results,'elapsed_seconds':time.perf_counter()-start,'parent_validation_report_sha256':fv.sha(parent/'report.json'),'provenance':provenance,'caveat':fm.CAVEAT})
if __name__=='__main__':main()
