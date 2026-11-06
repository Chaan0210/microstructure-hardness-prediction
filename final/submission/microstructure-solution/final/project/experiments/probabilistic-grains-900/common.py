"""File helpers for original-image object extraction."""
from pathlib import Path
import hashlib, json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def destination(path):
    p = Path(path).resolve()
    if not p.is_relative_to(HERE):
        raise ValueError(f'write outside isolated study: {p}')
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def dump_json(path, data):
    destination(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))

