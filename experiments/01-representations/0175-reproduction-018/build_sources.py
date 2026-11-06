from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import ast,shutil,json,hashlib
R=_RepoPath(__file__).resolve().parents[3];D=Path(__file__).resolve().parent;src=_repo_path(R, 'deliverables/reproduction-018-v155');provenance={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['candidate_features.py','grain_features.py','latent_features.py','denoised_grain_features.py','compact_regression.py','residual_linear.py','residual_models.py','cnn_model.py']:
 assert not (_repo_path(D, name)).exists();shutil.copyfile(_repo_path(src, name),_repo_path(D, name));provenance[name]={'source':str((_repo_path(src, name)).relative_to(R)),'sha256':sha(_repo_path(D, name)),'copy':'byte_exact'}
shutil.copytree(_repo_path(src, 'vendor'),_repo_path(D, 'vendor'),ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copyfile(_repo_path(R, 'studies/dinob_mlp_head_063.py'),_repo_path(D, 'mlp_head.py'));provenance['mlp_head.py']={'source':'dinob_mlp_head_063.py','sha256':sha(_repo_path(D, 'mlp_head.py')),'copy':'byte_exact'}
def funcs(file,names):
 text=(_repo_path(R, file)).read_text();a=ast.parse(text);return '\n\n'.join(ast.get_source_segment(text,x) for x in a.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names)
text='''import numpy as np,torch,json\nfrom pathlib import Path\nfrom sklearn.model_selection import GroupKFold\nfrom compact_regression import BalancedKernel\nfrom cnn_model import TinyCNN\nCNN_SEED=20261003\nINNER_SEED=20261005\nWIDTHS=(.25,1.,4.)\nALPHAS=(.01,.1,1.)\ndef rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))\n'''+funcs('frozen_validation.py',['cnn_fold','kernel_fold']).replace('engine.TinyCNN()','TinyCNN()').replace('engine.sync(device)',"torch.mps.synchronize() if device=='mps' else None")
(_repo_path(D, 'base_train.py')).write_text(text);provenance['base_train.py']={'source':'frozen_validation.py','source_sha256':sha(_repo_path(R, 'studies/frozen_validation.py')),'functions':['cnn_fold','kernel_fold'],'edits':'Replace engine.TinyCNN and engine.sync with local class and torch.mps.synchronize only; no training logic changes.'}
text='import numpy as np,torch\nfrom torch import nn\n'+funcs('deliverables/reproduction-018-v155/runtime.py',['ShortHead','Short','spatial_descriptor','dino_input'])+'\nSPATIAL_PROJECTION=None\nfrom PIL import Image\n'
(_repo_path(D, 'feature_helpers.py')).write_text(text);provenance['feature_helpers.py']={'source':'deliverables/reproduction-018-v155/runtime.py','source_sha256':sha(_repo_path(src, 'runtime.py')),'functions':['ShortHead','Short','spatial_descriptor','dino_input']}
with (_repo_path(D, 'source-provenance.json')).open('x') as f:json.dump(provenance,f,indent=2)
print('STANDALONE_SOURCES_CREATED')
