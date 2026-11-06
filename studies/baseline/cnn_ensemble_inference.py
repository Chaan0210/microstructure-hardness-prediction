"""Prediction-only D4 averaging for locally trained TinyCNN checkpoints."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
import torch
from cnn_controls import TinyCNN,sync

def predict_views(images, checkpoint, device='cpu'):
    saved=torch.load(_repo_io(checkpoint),map_location='cpu',weights_only=True)
    n=saved['normalization'];x=(images-n['x_mean'])/max(n['x_std'],1e-6)
    model=TinyCNN().to(device);model.load_state_dict(saved['state_dict']);model.eval();views=[]
    with torch.no_grad():
        for k in range(4):
            for flip in (False,True):
                z=np.rot90(x,k,axes=(2,3));z=z[:,:,:,::-1] if flip else z;p=[]
                for first in range(0,len(z),32):
                    xb=torch.from_numpy(np.ascontiguousarray(z[first:first+32])).to(device)
                    p.extend((model(xb).cpu().numpy()*n['y_std']+n['y_mean']).tolist())
                views.append(p)
    sync(device);del model
    if device=='mps':torch.mps.empty_cache()
    values=np.array(views,dtype=np.float64)
    if values.shape!=(8,len(images)) or not np.isfinite(values).all():raise ValueError('Invalid D4 predictions')
    return values
