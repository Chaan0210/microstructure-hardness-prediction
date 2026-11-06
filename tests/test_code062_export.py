"""Code submitted as Markdown must preserve frozen source-file hashes."""
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class Code062ExportTests(unittest.TestCase):
    def test_markdown_restores_exact_utf8_bytes_and_line_endings(self):
        spec = importlib.util.spec_from_file_location("delivery_export_test", ROOT / "final/work/build_delivery.py")
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        originals = {
            "no_newline.py": b"value = 1",
            "blank_eof.py": b"value = 2\n\n",
            "spaces.py": b"value = 3   \n",
            "nested/crlf.py": 'name = "미세구조"\r\n'.encode("utf-8"),
        }
        with tempfile.TemporaryDirectory() as directory:
            package, output = Path(directory) / "package", Path(directory) / "submission"
            package.mkdir()
            output.mkdir()
            (package / "README.md").write_text("# Reproduction\n")
            (package / "settings.json").write_text('{"coefficient":0.384}\n')
            (package / "unused_experiment.py").write_text('raise RuntimeError("Must not be submitted")\n')
            selection = dict(code=list(originals), config=["README.md", "settings.json", "submission-files.json"],
                             assets=[], training_config=["settings.json"], markdown_exclude=["README.md"])
            (package / "submission-files.json").write_text(json.dumps(selection))
            for name, data in originals.items():
                path = package / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            with patch.multiple(exporter, PACK=package, OUT=output), contextlib.redirect_stdout(io.StringIO()):
                exporter.markdown([package / n for n in selection['code'] + selection['config']], selection)
            text = (output / "microstructure-solution.md").read_bytes().decode("utf-8")
            self.assertNotIn('## `README.md`', text)
            pattern = r"^## `([^`]+\.py)`\n\nSHA256: `([0-9a-f]{64})`\n\n````python\n(.*?)\n````"
            blocks = re.findall(pattern, text, re.M | re.S)
            self.assertEqual(len(blocks), len(originals))
            for name, declared_hash, code in blocks:
                restored = code.encode("utf-8")
                self.assertEqual(restored, originals[name], name)
                self.assertEqual(hashlib.sha256(restored).hexdigest(), declared_hash, name)
            self.assertFalse(list(output.glob('*.zip')))
            self.assertNotIn('## `unused_experiment.py`', text)
            restore_spec = importlib.util.spec_from_file_location("restore062_test", ROOT / "final/package/restore.py")
            restore = importlib.util.module_from_spec(restore_spec)
            restore_spec.loader.exec_module(restore)
            with contextlib.redirect_stdout(io.StringIO()):
                restore.restore(output / "microstructure-solution.md", Path(directory) / "restored")
            for name in selection['code'] + selection['config']:
                self.assertEqual((Path(directory) / "restored" / name).read_bytes(), (package / name).read_bytes())
            three_fences = output / 'three-fences.md'
            three_fences.write_bytes(text.replace('````', '```').encode('utf-8'))
            with contextlib.redirect_stdout(io.StringIO()):
                restore.restore(three_fences, Path(directory) / 'restored-three')
            for name in selection['code'] + selection['config']:
                self.assertEqual((Path(directory) / 'restored-three' / name).read_bytes(), (package / name).read_bytes())
            before = (output / 'microstructure-solution.md').read_bytes()
            (output / 'microstructure-solution').mkdir()
            with patch.multiple(exporter, PACK=package, OUT=output):
                self.assertRaises(FileExistsError, exporter.code)
                self.assertRaises(FileExistsError, exporter.assets)
            self.assertEqual((output / 'microstructure-solution.md').read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
