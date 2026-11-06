"""Check the selected submission directories or archives and Markdown."""
from pathlib import Path
from contextlib import ExitStack
import hashlib
import json
import re
import zipfile

BASE = Path(__file__).resolve().parent.parent
PACK = BASE / 'package'
OUT = BASE / 'submission'
selected = json.loads((PACK / 'submission-files.json').read_text())
expected = {'project/' + r['path']: r['sha256'] for r in json.loads((PACK / 'source-manifest.json').read_text())['files']}
expected.update({r['path']: r['sha256'] for r in json.loads((PACK / 'code-manifest.json').read_text())['sources']})
for name in selected['config']:
    expected[name] = hashlib.sha256((PACK / name).read_bytes()).hexdigest()
text = (OUT / 'microstructure-solution.md').read_bytes().decode('utf-8')
assert text.partition('\n---\n\n# 제출 코드와 실행 설정\n\n')[0].encode('utf-8') == (PACK / 'README.md').read_bytes()
pattern = r'^## `([^`]+)`\n\nSHA256: `([0-9a-f]{64})`\n\n(?P<fence>`{3,})(?:python|json|text)\n(.*?)\n(?P=fence)(?=\n|$)'
blocks = re.findall(pattern, text, re.M | re.S)
markdown_names = set(selected['code'] + selected['config']) - set(selected.get('markdown_exclude', []))
assert len(blocks) == len(markdown_names)
assert {n for n, _, _, _ in blocks} == markdown_names
for name, digest, _, source in blocks:
    assert hashlib.sha256(source.encode('utf-8')).hexdigest() == digest == expected[name], name
for stem, names in [('microstructure-solution', selected['code'] + selected['config']), ('microstructure-assets', selected['assets'])]:
    archive = OUT / (stem + '.zip')
    with ExitStack() as stack:
        if archive.exists():
            z = stack.enter_context(zipfile.ZipFile(archive))
            assert len(z.infolist()) == len(names)
            assert set(z.namelist()) == {'final/' + n for n in names}
            open_member = lambda name: z.open('final/' + name)
            kind = 'archive'
        else:
            directory = OUT / stem / 'final'
            files = [p for p in directory.rglob('*') if p.is_file()]
            assert all(not p.is_symlink() for p in files)
            assert {p.relative_to(directory).as_posix() for p in files} == set(names)
            open_member = lambda name: (directory / name).open('rb')
            kind = 'directory'
        for name in names:
            h = hashlib.sha256()
            with open_member(name) as stream:
                for block in iter(lambda: stream.read(2**20), b''):
                    h.update(block)
            assert h.hexdigest() == expected[name], name
    print(stem, kind, len(names), 'files: SHA256 verified', flush=True)
print('Markdown:', len(blocks), 'exact code/configuration blocks; no historical extra members.', flush=True)
