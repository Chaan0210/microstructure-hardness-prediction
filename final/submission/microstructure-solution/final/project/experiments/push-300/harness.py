import socket
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parents[2]
W3 = P / "experiments/tabpfn-v3/weights"


def noise_of(ids, split='train'):
    from PIL import Image
    out = []
    for i in ids:
        f = np.asarray(Image.open(P / f'data/{split}/{i}.png'), np.float32) / 255
        hh = (f[0::2, 0::2] - f[0::2, 1::2] - f[1::2, 0::2] + f[1::2, 1::2]) / 2
        out.append(np.median(np.abs(hh)) / 0.6745)
    return np.array(out)


def tabpfn_new(version='v3.5', seed=0, device='mps', n_est=None):
    if version != 'v3.5':
        raise ValueError('The final model requires TabPFN v3.5')
    def blocked(*a, **k): raise RuntimeError('offline')
    socket.create_connection = blocked; socket.getaddrinfo = blocked
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    kw = dict(model_path=str(W3 / 'tabpfn-v3.5-20260909.safetensors'),
              device=device, random_state=seed, show_progress_bar=False)
    if n_est is not None: kw['n_estimators'] = n_est
    return TabPFNRegressor.create_default_for_version(ModelVersion(version), **kw)
