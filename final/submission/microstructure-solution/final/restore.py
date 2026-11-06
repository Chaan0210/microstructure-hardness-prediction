from pathlib import Path
import argparse
import hashlib
import re


def restore(markdown, output):
    text = Path(markdown).read_bytes().decode('utf-8')
    pattern = r'^## `([^`]+)`\n\nSHA256: `([0-9a-f]{64})`\n\n(?P<fence>`{3,})(?:python|json|text)\n(.*?)\n(?P=fence)(?=\n|$)'
    blocks = re.findall(pattern, text, re.M | re.S)
    if not blocks:
        raise ValueError('No source blocks found')
    files = {}
    for name, expected, _, source in blocks:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or name in files:
            raise ValueError(f'Unsafe or duplicated path: {name}')
        data = source.encode('utf-8')
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Source checksum mismatch: {name}')
        files[name] = data
    introduction, separator, _ = text.partition('\n---\n\n# 제출 코드와 실행 설정\n\n')
    if separator and 'README.md' not in files:
        files['README.md'] = introduction.encode('utf-8')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print('Restored', len(files), 'code/configuration files to', root)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('markdown', type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    restore(a.markdown, a.out)
