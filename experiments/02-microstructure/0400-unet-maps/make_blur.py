from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, csv, numpy as np
from PIL import Image
sys.path.insert(0, '.')
from blur_feats import matrix, KEYS
split = sys.argv[1]
ids = [r['ID'] for r in csv.DictReader(open(_repo_io('../../data/' + ('train.csv' if split == 'train' else 'sample_submission.csv'))))]
X = matrix([np.array(Image.open(_repo_io(f'../../data/{split}/{i}.png'))) for i in ids])
np.savez(_repo_io(f'blur-{split}.npz'), ids=np.array(ids), X=X, keys=np.array(KEYS)); print(X.shape, np.isnan(X).sum(0))
