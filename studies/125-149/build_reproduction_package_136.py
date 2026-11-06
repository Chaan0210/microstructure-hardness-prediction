"""Allowlisted local packaging; never includes dataset files or feature caches."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import ast,json,shutil,hashlib,importlib.metadata,platform,sys
R=_RepoPath(__file__).resolve().parents[1];P=_repo_path(R, 'deliverables/reproduction-011-014-015-v136')
P.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(rel,v):
 p=_repo_path(P, rel);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def copy(src,dest):
 p=_repo_path(P, dest);p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(_repo_path(R, src),p)
def subset(src,dest,names,header,replacements={}):
 s=(_repo_path(R, src)).read_text();tree=ast.parse(s);parts=[ast.get_source_segment(s,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names];assert len(parts)==len(names)
 t=header+'\n\n'+'\n\n'.join(parts)+'\n'
 for a,b in replacements.items():t=t.replace(a,b)
 (_repo_path(P, dest)).write_text(t)
subset('compact_regression.py','compact_regression.py',['BalancedKernel','compact_existing'],'import numpy as np\nfrom sklearn.base import BaseEstimator,RegressorMixin\nfrom sklearn.preprocessing import MinMaxScaler,StandardScaler\nfrom scipy.spatial.distance import pdist,cdist,squareform\nfrom scipy.linalg import solve')
subset('cnn_controls.py','cnn_model.py',['TinyCNN'],'import torch\nfrom torch import nn')
subset('candidate004_pipeline.py','cnn_training.py',['fit_cnn'],'import json,numpy as np,torch\nfrom cnn_model import TinyCNN\ndef sync(device):\n if device=="mps":torch.mps.synchronize()',{'cp.engine.TinyCNN()':'TinyCNN()','cp.engine.sync(device)':'sync(device)'})
subset('residual_physics_113.py','residual_linear.py',['fit_model','predict'],'import numpy as np\nfrom sklearn.preprocessing import StandardScaler,MinMaxScaler\nfrom sklearn.linear_model import Ridge\nCANDIDATES=("none","ridge1","ridge01")\nLAMBDAS={"ridge1":1.,"ridge01":.1}')
subset('nonlinear_residual_115.py','residual_models.py',['fit','predict'],'import numpy as np\nfrom scipy.spatial.distance import pdist,cdist,squareform\nfrom sklearn.preprocessing import StandardScaler,MinMaxScaler\nimport residual_linear as r\nKINDS=(*r.CANDIDATES,"rbf1","rbf01")\nLAM={"rbf1":1.,"rbf01":.1}')
for f in ['candidate_features.py','grain_features.py','latent_features.py','denoised_grain_features.py']:copy(f,f)
copy('dinob_mlp_head_063.py','mlp_head.py')
for f in ['__init__.py','features.py','extra.py']:copy('src/microstructure_v2/'+f,'microstructure_v2/'+f)
source=json.loads((_repo_path(R, 'vendor/dinov2-official/provenance.json')).read_text())['source_dir']
for file in (_repo_path(R, source, 'dinov2')).rglob('*.py'):
 if 'cell_dino' not in file.parts:copy(str(file.relative_to(R)),'vendor/'+str(file.relative_to(_repo_path(R, source))))
for f in ['LICENSE','MODEL_CARD.md','README.md']:copy(source+'/'+f,'licenses/DINOv2-'+f)
for arm,file in [('S','vendor/dinov2-official/dinov2_vits14_pretrain.pth'),('B','vendor/dinov2-base-062/dinov2_vitb14_pretrain.pth')]:copy(file,f'weights/DINO-{arm}.pth')
copy('reports/software-licenses-current-135.json','licenses/current-model-provenance.json')
# This old license inventory is included only for dependency notices with its outdated prose removed.
old=json.loads((_repo_path(R, 'reports/software-licenses.json')).read_text());write('licenses/dependency-notices.json',{'dependencies':old['dependencies']})
assets={}
def asset(src,name):copy(src,'models/'+name);assets[name]={'source':src,'sha256':sha(_repo_path(R, src))}
asset('artifacts/final-model-017/baseline.joblib','baseline.joblib');asset('artifacts/final-model-017/kernel.joblib','kernel.joblib')
for seed in [20261003,20261023,20261043]:asset(f'artifacts/final-model-024/cnn-seed{seed}.pt',f'cnn-{seed}.pt')
fit=json.loads((_repo_path(R, 'artifacts/final-model-065/report.json')).read_text())
for rec in fit['models']:asset('artifacts/final-model-065/'+rec['checkpoint'],rec['checkpoint'])
asset('artifacts/final-model-086/head/model.pt','shortB.pt')
for c,folder in [('011','075'),('014','123'),('015','134')]:
 asset(f'artifacts/final-model-{folder}/calibrator.json',f'calibrator-{c}.json')
 if c!='011':
  for j in range(3):asset(f'artifacts/final-model-{folder}/block-{j}/residual/selected.joblib',f'residual-{c}-{j}.joblib')
import joblib
from sklearn.base import clone
b=joblib.load(_repo_path(R, 'artifacts/final-model-017/baseline.joblib'));joblib.dump({'models':{k:clone(v) for k,v in b['models'].items()},'weights':b['weights'],'feature_names':b['feature_names']},_repo_path(P, 'models/baseline-template.joblib'))
sp=json.loads((_repo_path(R, 'artifacts/run-001/split_manifest.json')).read_text());write('config/splits.json',{'development_ids':sorted(sp['development_folds']),'development_folds':sp['development_folds'],'group_by_id':sp['group_by_id'],'meaning':'Synthetic competition IDs and derived group hashes only; no target values or image/feature arrays.'})
panel=json.loads((_repo_path(R, 'artifacts/run-012/report.json')).read_text());physics=json.loads((_repo_path(R, 'artifacts/run-022/report.json')).read_text())
write('config/schema.json',{'compact_names':panel['features'],'compact_blocks':panel['blocks'],'physics_names':physics['features'],'physics_blocks':physics['blocks'],'baseline_names':b['feature_names']})
write('config/assets.json',assets)
write('config/candidates.json',{c:{'candidate':c,'expected_CSV_SHA256':sha(_repo_path(R, f'artifacts/submission-candidate-{c}.csv')),'status':'historical public best (parent-confirmed)' if c=='011' else 'verified; not submitted','formula':'.9*affine(.375baseline+.1875KRR+.1875CNNfirst+.125Smean+.125Bmean)+.1shortB' if c=='011' else '.9*affine(.3KRR+.3CNN'+('mean3' if c=='015' else 'first')+'+.2Smean+.2Bmean)+.1shortB+mean3residual','frozen_models':True} for c in ['011','014','015']})
versions={d.metadata['Name'].lower().replace('_','-'):d.version for d in importlib.metadata.distributions()};write('environment.json',{'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),'packages':versions,'CPU_threads':2,'training_device':'MPS for CNN/shortB; CPU for S/B heads and sklearn','package_install_performed':False,'determinism':'Exact replay validated on existing environment; other OS/devices may differ in floating point.'})
(_repo_path(P, 'requirements-current.lock')).write_text(''.join(f'{n}=={v}\n' for n,v in sorted(versions.items())))
write('config/build-provenance.json',{'source_functions':{f:sha(_repo_path(R, f)) for f in ['compact_regression.py','cnn_controls.py','candidate004_pipeline.py','residual_physics_113.py','nonlinear_residual_115.py','dinob_mlp_head_063.py','candidate_features.py','denoised_grain_features.py']},'pinned_DINO_source_commit':json.loads((_repo_path(R, 'vendor/dinov2-official/provenance.json')).read_text())['commit'],'raw_data_or_feature_caches_included':False,'historical_reports_and_personal_identifiers_included':False})
print(P.name,'base files ready')
