"""Post-freeze numerical verification; never fits or tunes a model."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, csv, hashlib
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, '.venv')).resolve(): raise SystemExit('Use project .venv')
sys.path.insert(0,str(ROOT))
import candidate_pipeline as cp
import numpy as np
import joblib
from PIL import Image

def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def main():
    folder=_repo_path(ROOT, 'artifacts/final-model-017')
    manifest=json.loads((_repo_path(ROOT, 'artifacts/submission-candidate-003.manifest.json')).read_text())
    output=_repo_path(ROOT, 'artifacts/submission-candidate-003.csv')
    assert cp.sha(output)==manifest['submission_sha256']
    assert cp.sha(cp.PREVIOUS)=='778b927586c95d50b4ab94a7b7f74515a055de80f6bf6ba6a8652cb92e6a85f0'
    for name,digest in manifest['model_sha256'].items():assert cp.sha(folder/name)==digest
    assert cp.sources('run-012')==manifest['source_sha256']
    rows=read(output);components=read(output.with_suffix('.components.csv'));old=read(cp.PREVIOUS)
    assert [r['ID'] for r in rows]==[r['ID'] for r in components]==[r['ID'] for r in old]
    formula=np.array([.5*float(r['baseline'])+.25*float(r['cnn'])+.25*float(r['kernel']) for r in components])
    predictions=np.array([float(r['hardness']) for r in rows])
    np.testing.assert_allclose(formula,predictions,rtol=0,atol=1e-10)
    base=np.array([float(r['baseline']) for r in components]);previous=np.array([float(r['hardness']) for r in old])
    np.testing.assert_allclose(base,previous,rtol=0,atol=1e-10)
    # Fixed evenly spaced rows, selected without inspecting their predictions.
    ix=np.linspace(0,999,10,dtype=int);ids=[rows[i]['ID'] for i in ix]
    records=[cp.extract_one(s,'test','run-012') for s in ids]
    X=np.stack([r[0] for r in records]);images=np.stack([r[2] for r in records])[:,None]
    kernel=joblib.load(_repo_io(folder/'kernel.joblib'))
    assert all(r[1]==kernel['feature_names'] for r in records)
    assert all(r[3]==manifest['image_sha256'][s] for r,s in zip(records,ids))
    kp=kernel['model'].predict(X);expected_kernel=np.array([float(components[i]['kernel']) for i in ix])
    np.testing.assert_allclose(kp,expected_kernel,rtol=0,atol=1e-9)
    cp.torch.set_num_threads(2)
    cnn=cp.predict_cnn(images,folder/'cnn.pt','cpu');expected_cnn=np.array([float(components[i]['cnn']) for i in ix])
    np.testing.assert_allclose(cnn,expected_cnn,rtol=0,atol=1e-3)
    result={'valid':True,'submission_sha256':cp.sha(output),'rows':len(rows),
      'full_weighted_formula_max_abs_difference':float(np.max(abs(formula-predictions))),
      'full_baseline_vs_candidate002_max_abs_difference':float(np.max(abs(base-previous))),
      'reloaded_kernel_sample_max_abs_difference':float(np.max(abs(kp-expected_kernel))),
      'cpu_vs_mps_cnn_sample_max_abs_difference':float(np.max(abs(cnn-expected_cnn))),
      'reload_sample_ids':ids,'cpu_mps_tolerance':1e-3,'old_candidate002_unchanged':True,
      'test_usage':'Post-freeze inference and integrity verification only; no fit or selection',
      'uploaded':False}
    (_repo_path(ROOT, 'reports/candidate003-reload-verification.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
