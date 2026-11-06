from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

PACKAGE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def prepare(download=False):
    records = json.loads((PACKAGE / 'model-provenance.json').read_text())['models']
    for record in records:
        path = PACKAGE / 'project' / record['local']
        if path.exists():
            if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
                raise RuntimeError(f'Checkpoint differs from the original: {path}')
        elif download:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + '.download')
            try:
                with urllib.request.urlopen(record['source'], timeout=120) as source, partial.open('xb') as dest:
                    while block := source.read(2**20):
                        dest.write(block)
                if partial.stat().st_size != record['bytes'] or digest(partial) != record['sha256']:
                    raise RuntimeError(f'Download checksum mismatch: {record["model"]}')
                partial.replace(path)
            finally:
                partial.unlink(missing_ok=True)
        else:
            raise FileNotFoundError(f'{path}; run prepare_pretrained.py --download')
        print('Pretrained checkpoint verified:', record['model'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Download missing weights from the recorded official URLs.')
    prepare(parser.parse_args().download)
