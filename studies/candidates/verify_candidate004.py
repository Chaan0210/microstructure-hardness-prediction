"""Final post-freeze verification only; no model fitting or selection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,csv
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, '.venv')).resolve():raise SystemExit('Use project .venv')
sys.path.insert(0,str(ROOT))
import numpy as np
import joblib
import candidate004_pipeline as cp4
from cnn_ensemble_inference import predict_views

def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def main():
    cp=cp4.cp;folder=_repo_path(ROOT, 'artifacts/final-model-024');output=_repo_path(ROOT, 'artifacts/submission-candidate-004.csv');manifest=json.loads(output.with_suffix('.manifest.json').read_text())
    assert cp.sha(output)==manifest['submission_sha256'];assert cp4.sources()==manifest['source_sha256']
    cp4.protected()
    for name,h in manifest['model_sha256'].items():assert cp.sha(folder/name)==h
    for path,key in [(output.with_suffix('.components.csv'),'components_sha256'),(output.with_suffix('.cnn-views.npz'),'cnn_views_sha256')]:assert cp.sha(path)==manifest[key]
    rows=read(output);c=read(output.with_suffix('.components.csv'));old=read(_repo_path(ROOT, 'artifacts/submission-candidate-003.components.csv'))
    ids=[r['ID'] for r in rows];assert len(ids)==len(set(ids))==1000 and ids==[r['ID'] for r in c]==[r['ID'] for r in old]
    with np.load(_repo_io(output.with_suffix('.cnn-views.npz')),allow_pickle=False) as a:
        assert a['ids'].tolist()==ids and a['seeds'].tolist()==list(cp4.CNN_SEEDS)
        views=a['predictions'];assert views.shape==(3,8,1000)
    cn=views.mean(axis=(0,1));saved_cn=np.array([float(r['cnn_mean3_d4']) for r in c]);np.testing.assert_allclose(cn,saved_cn,atol=1e-10,rtol=0)
    b=np.array([float(r['baseline']) for r in c]);k=np.array([float(r['kernel']) for r in c]);p=np.array([float(r['hardness']) for r in rows]);np.testing.assert_allclose(.5*b+.25*k+.25*cn,p,atol=1e-10,rtol=0)
    oldb=np.array([float(r['baseline']) for r in old]);np.testing.assert_allclose(b,oldb,atol=1e-10,rtol=0)
    oldc=np.array([float(r['cnn']) for r in old]);np.testing.assert_allclose(views[0,0],oldc,atol=1e-3,rtol=0)
    ix=np.linspace(0,999,10,dtype=int);records=[cp4.extract_one(ids[i],'test','run-022') for i in ix];names=json.loads((folder/'report.json').read_text())['feature_names'];assert all(r[1]==names for r in records)
    assert all(r[3]==manifest['image_sha256'][ids[i]] for r,i in zip(records,ix))
    X=np.stack([r[0] for r in records]);images=np.stack([r[2] for r in records])[:,None];kernel=joblib.load(_repo_io(folder/'kernel.joblib'))
    predk=kernel['model'].predict(X);np.testing.assert_allclose(predk,k[ix],atol=1e-9,rtol=0)
    cp.torch.set_num_threads(2);cpu=[]
    for seed in cp4.CNN_SEEDS:cpu.append(predict_views(images,folder/f'cnn-seed{seed}.pt','cpu'))
    cpu=np.stack(cpu);np.testing.assert_allclose(cpu,views[:,:,ix],atol=1e-3,rtol=0)
    result={'valid':True,'sha256':cp.sha(output),'rows':1000,'seed_view_average_max_diff':float(np.max(abs(cn-saved_cn))),'weighted_formula_max_diff':float(np.max(abs(.5*b+.25*k+.25*cn-p))),'baseline_vs003_max_diff':float(np.max(abs(b-oldb))),'first_cnn_identity_vs003_max_diff':float(np.max(abs(views[0,0]-oldc))),'reloaded_kernel10_max_diff':float(np.max(abs(predk-k[ix]))),'cpu_vs_mps_3seed_8view_10image_max_diff':float(np.max(abs(cpu-views[:,:,ix]))),'reload_sample_ids':[ids[i] for i in ix],'old_candidate003_preserved':True,'test_use':'Post-freeze inference integrity only','uploaded':False}
    with (_repo_path(ROOT, 'reports/candidate004-reload-verification.json')).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
