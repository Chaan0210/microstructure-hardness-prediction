from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,hashlib
import numpy as np
import torch
import phase_joint_v3 as p

H=Path(__file__).resolve().parent;out=_repo_path(H, 'runs/phase-joint-b4-f0-s81');path=_repo_path(out, 'stopped-epoch50.pt')
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.35)
ck=torch.load(_repo_io(path),map_location='cpu',weights_only=False);assert ck['epoch']==50
d=dict(np.load(_repo_path(H, 'data.npz')));px=np.load(_repo_path(H, 'train-pixels.npy'),mmap_mode='r');tr=np.array(ck['recipe']['train']);q=np.array(ck['recipe']['validation']);assert not set(tr)&set(q)
aux=p.prepare(d,px,tr,out);net=p.UNet(ck['recipe']['width']).to('mps');net.load_state_dict({k.removeprefix('net.'):v for k,v in ck['state'].items() if k.startswith('net.')})
frozen=np.load(_repo_path(out, 'frozen-features.npz'));assert np.array_equal(frozen['train'],tr) and np.array_equal(frozen['query'],q)
f=p.extract(net,px,aux,tr,batch=4);v=p.extract(net,px,aux,q,batch=4);target=_repo_path(out, 'joint-features.npz');assert not target.exists()
np.savez(_repo_io(target),train=tr,query=q,train_features=f,query_features=v,train_globals=frozen['train_globals'],query_globals=frozen['query_globals'],encoder_epoch=np.array(50))
(_repo_path(out, 'joint-feature-provenance.json')).write_text(json.dumps(dict(checkpoint='stopped-epoch50.pt',checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),encoder_epoch=50,original_planned_epochs=80,complete80=False,selection='Latest preserved checkpoint after stopping an overfitting configuration; not the best checkpoint.',training_during_extraction=False),indent=2))
print('Extracted epoch50 latent features for planned nonlinear readout; no new training.',flush=True)
