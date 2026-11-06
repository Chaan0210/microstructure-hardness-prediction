"""Legacy identifiers still locate the organized research files."""
import ast
import importlib.machinery
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import research_paths as paths


class ResearchLayoutTests(unittest.TestCase):
    def test_experiment_identifiers(self):
        for old, relative in paths.EXPERIMENT_PATHS.items():
            expected = ROOT / "experiments" / relative
            with self.subTest(experiment=old):
                self.assertTrue(expected.is_dir())
                self.assertEqual(paths.resolve(ROOT, "experiments", old), expected)
                self.assertEqual(paths.resolve(ROOT, "experiments", relative), expected)

    def test_cross_group_relative_path(self):
        folder = ROOT / "experiments/02-microstructure/0800-shading-features"
        expected = ROOT / "experiments/03-robustness/1200-degradation-augmentation/make_synth4.py"
        self.assertEqual(paths.resolve(folder, "../push-1200/make_synth4.py"), expected)
        self.assertTrue(expected.is_file())

    def test_current_relative_paths_and_future_outputs(self):
        folder = ROOT / "experiments/03-robustness/1300-robust-ensembles"
        expected = ROOT / "experiments/02-microstructure/0300-cellpose-finetuning/dev_eval.py"
        self.assertEqual(
            paths.resolve(folder, "../../02-microstructure/0300-cellpose-finetuning/dev_eval.py"),
            expected,
        )
        self.assertTrue(expected.is_file())
        future = ROOT / "experiments/03-robustness/1200-degradation-augmentation/path-check-new.json"
        self.assertFalse(future.exists())
        self.assertEqual(paths.resolve(folder, "../1200-degradation-augmentation/path-check-new.json"), future)
        self.assertEqual(paths.resolve(folder, "../../../data/train.csv"), ROOT / "data/train.csv")
        self.assertEqual(paths.resolve(folder, "../../data/train.csv"), ROOT / "data/train.csv")

    def test_study_module_names(self):
        sources = paths.study_sources()
        self.assertEqual(len(sources), 417)
        self.assertEqual(len({p.name for p in sources}), 417)
        for source in sources:
            self.assertEqual(paths.resolve(paths.STUDIES, source.name), source)
        self.assertEqual(paths.resolve(paths.STUDIES, ".venv"), ROOT / ".venv")

    def test_report_roles(self):
        examples = {
            "candidate056-public-submission.json": "submissions/candidate056-public-submission.json",
            "dinob-062-validation.json": "validation/dinob-062-validation.json",
            "push-1200-report.md": "notes/push-1200-report.md",
            "strategy-experiment-register.csv": "registers/strategy-experiment-register.csv",
        }
        for old, relative in examples.items():
            with self.subTest(report=old):
                expected = ROOT / "reports" / relative
                self.assertEqual(paths.resolve(ROOT, "reports", old), expected)
                self.assertTrue(expected.is_file())

    def test_grouped_grain_cli_defaults(self):
        source = ROOT / "experiments/02-microstructure/0300-cellpose-finetuning/pipeline_grain.py"
        spec = importlib.util.spec_from_file_location("grain_path_check", source)
        module = importlib.util.module_from_spec(spec)
        defaults = {}

        class Parsed(Exception):
            pass

        def inspect(parser):
            for name in ("best021", "zs_weights", "ckpt"):
                defaults[name] = Path(parser.get_default(name))
            raise Parsed()

        with patch.object(sys, "path", sys.path.copy()):
            spec.loader.exec_module(module)
            with patch("argparse.ArgumentParser.parse_args", inspect), self.assertRaises(Parsed):
                module.main()
        self.assertEqual(defaults["best021"], ROOT / "artifacts/submission-candidate-021.csv")
        self.assertTrue(defaults["best021"].is_file())
        self.assertEqual(defaults["zs_weights"], ROOT / "experiments/02-microstructure/0220-grain-cellpose/weights/cpsam")
        self.assertEqual(defaults["ckpt"], source.parent / "ft-a1s250.pt")

    def test_study_launchers_locate_child_scripts(self):
        checked = 0
        for source in paths.study_sources():
            if not source.name.startswith("run_"):
                continue
            tree = ast.parse(source.read_text())
            scripts = [
                n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and n.value in paths._SOURCE_NAMES
            ]
            for node in ast.walk(tree):
                if not (
                    scripts and isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "subprocess" and node.args
                    and isinstance(node.args[0], ast.List)
                    and any(isinstance(n, ast.Name) and n.id == "script" for n in ast.walk(node.args[0]))
                ):
                    continue
                env = dict(sys=sys, str=str, root=paths.STUDIES, ROOT=paths.STUDIES,
                           PY=ROOT / ".venv/bin/python", args=[], arg="probe",
                           _repo_path=paths.resolve, _repo_io=paths.io_path)
                for script in scripts:
                    command = eval(compile(ast.Expression(node.args[0]), str(source), "eval"),
                                   {**env, "script": script})
                    child = paths.STUDIES / command[3]
                    with self.subTest(launcher=source.name, child=script):
                        self.assertEqual(child, paths.resolve(paths.STUDIES, script))
                        self.assertTrue(child.is_file())
                    checked += 1
        self.assertGreater(checked, 0)

    def test_cross_experiment_import_search_paths(self):
        examples = {
            "experiments/02-microstructure/0300-cellpose-finetuning/seg_ft.py": "features_v3",
            "experiments/03-robustness/1200-degradation-augmentation/synaug_ridge2.py": "blend_dev",
        }
        original_cwd = Path.cwd()
        try:
            for relative, module in examples.items():
                source = ROOT / relative
                os.chdir(source.parent)
                directories = []
                for node in ast.walk(ast.parse(source.read_text())):
                    if isinstance(node, ast.Call) and ast.unparse(node.func) == "sys.path.insert":
                        directories.append(eval(
                            compile(ast.Expression(node.args[1]), str(source), "eval"),
                            {"_repo_io": paths.io_path},
                        ))
                with self.subTest(script=source.name, module=module):
                    self.assertIsNotNone(importlib.machinery.PathFinder.find_spec(module, directories))
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
