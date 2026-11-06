"""Export only the explicit final training/inference dependency list."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import shutil
import zipfile

BASE = Path(__file__).resolve().parent.parent
PACK = BASE / 'package'
OUT = BASE / 'submission'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def selection():
    selected = json.loads((PACK / 'submission-files.json').read_text())
    names = selected['code'] + selected['config'] + selected['assets']
    assert len(names) == len(set(names)), 'Duplicated submission member'
    for name in names:
        p = Path(name)
        assert not p.is_absolute() and '..' not in p.parts, name
        assert not (PACK / p).is_symlink(), name
    assert set(selected['training_config']) <= set(selected['config'])
    assert set(selected.get('markdown_exclude', [])) <= set(selected['config'])
    return selected


def metadata():
    path = OUT / 'DELIVERY.json'
    if path.exists():
        record = json.loads(path.read_text())
        for name in ('microstructure-solution.md', 'microstructure-solution.pdf', 'microstructure-solution.zip', 'microstructure-assets.zip'):
            p = OUT / name
            if p.exists():
                record['files'][name] = dict(bytes=p.stat().st_size, sha256=sha(p))
        record.pop('final_code_archive_matches_current_package_text_files', None)
        record['final_code_archive_matches_selected_package_files'] = True
        record['code_scope'] = 'Selected final training and raw-image inference, including configuration; historical research excluded.'
        record['missing_files'] = [name for name in record.get('missing_files', []) if not (OUT / name).exists()]
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')


def protect_review():
    if any((OUT / name).is_dir() for name in ('microstructure-solution', 'microstructure-assets')):
        raise FileExistsError('Unpacked review directories exist. Compress the reviewed directories manually; do not overwrite the edited Markdown.')


def markdown(paths, selected):
    intro = (PACK / 'README.md').read_bytes().decode('utf-8')
    blocks = [intro + '\n---\n\n# 제출 코드와 실행 설정\n\n']
    for p in paths:
        if p.relative_to(PACK).as_posix() in selected.get('markdown_exclude', []):
            continue
        source = p.read_bytes().decode('utf-8')
        fence = '`' * max(4, 1 + max((len(line) for line in source.splitlines()
                                     if line and set(line) == {'`'}), default=0))
        kind = 'python' if p.suffix == '.py' else ('json' if p.suffix == '.json' else 'text')
        name = p.relative_to(PACK).as_posix()
        blocks.append(f'## `{name}`\n\nSHA256: `{sha(p)}`\n\n{fence}{kind}\n{source}\n{fence}\n\n')
    (OUT / 'microstructure-solution.md').write_bytes(''.join(blocks).encode('utf-8'))


def code():
    protect_review()
    selected = selection()
    sources = sorted((PACK / n for n in selected['code']),
                     key=lambda p: (len(p.relative_to(PACK).parts) > 1, str(p.relative_to(PACK))))
    records = []
    for p in sources:
        ast.parse(p.read_bytes().decode('utf-8'))
        records.append(dict(path=p.relative_to(PACK).as_posix(), sha256=sha(p), bytes=p.stat().st_size))
    (PACK / 'code-manifest.json').write_text(json.dumps(dict(python_files=len(sources), sources=records), indent=2) + '\n')
    files = []
    for name in sorted(selected['code'] + selected['config'] + selected['assets']):
        p = PACK / name
        if name.startswith('project/'):
            files.append(dict(path=name.removeprefix('project/'), bytes=p.stat().st_size,
                              sha256=sha(p), roles=['selected_final_dependency']))
    (PACK / 'source-manifest.json').write_text(json.dumps(dict(
        scope='Final #062 raw-image inference and original training recipes; saved original weights are optional for retraining.',
        files=files, total_bytes=sum(r['bytes'] for r in files)), indent=2) + '\n')
    paths = sources + [PACK / n for n in sorted(selected['config'])]
    markdown(paths, selected)
    path = OUT / 'microstructure-solution.zip'
    temp = path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in paths:
            z.write(p, Path('final') / p.relative_to(PACK))
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
    temp.replace(path)
    metadata()
    print('CODE', len(sources), 'CONFIG', len(selected['config']),
          'MD bytes', (OUT / 'microstructure-solution.md').stat().st_size, 'ZIP bytes', path.stat().st_size, flush=True)


def assets():
    protect_review()
    paths = [PACK / n for n in selection()['assets']]
    size = sum(p.stat().st_size for p in paths)
    assert shutil.disk_usage(OUT).free > size + 6 * 2**30, 'Preserve 6 GiB free disk'
    path = OUT / 'microstructure-assets.zip'
    temp = path.with_suffix('.zip.tmp')
    try:
        with zipfile.ZipFile(temp, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
            for p in paths:
                assert p.suffix not in ('.png', '.jpg', '.pyc'), p
                z.write(p, Path('final') / p.relative_to(PACK))
        with zipfile.ZipFile(temp) as z:
            assert z.testzip() is None and len(z.infolist()) == len(paths)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    metadata()
    print('ASSETS', len(paths), 'bytes', path.stat().st_size, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('code', 'assets'))
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    code() if a.action == 'code' else assets()
