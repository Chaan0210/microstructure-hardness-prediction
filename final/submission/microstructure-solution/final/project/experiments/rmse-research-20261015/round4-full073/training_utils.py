from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / 'experiments'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def rmse(pred, y):
    return float(np.sqrt(np.mean((pred - y) ** 2)))
