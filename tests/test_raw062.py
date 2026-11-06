"""Runtime paths and the unchanged predictors needed by raw #062 inference."""
import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "final/package"


def load(name):
    spec = importlib.util.spec_from_file_location(name, PACKAGE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Raw062Tests(unittest.TestCase):
    def test_missing_data_does_not_create_half_initialized_workspace(self):
        workspace = load("workspace")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(
                workspace=root / "project", data=root / "missing-data",
                python_core=sys.executable, python_cellpose=sys.executable,
                python_tabpfn=sys.executable,
            )
            with self.assertRaises(FileNotFoundError):
                workspace.init(args)
            self.assertFalse((root / "project").exists())

    def test_moved_environments_preserve_venv_paths(self):
        workspace = load("workspace")
        args = argparse.Namespace(python_core=None, python_cellpose=None, python_tabpfn=None)
        candidates = {
            "core": ROOT / ".venv/bin/python",
            "cellpose": ROOT / "experiments/02-microstructure/0220-grain-cellpose/.venv/bin/python",
            "tabpfn": ROOT / "experiments/frameworks/tabpfn-v2/.venv/bin/python",
        }
        if not all(p.is_file() for p in candidates.values()):
            self.skipTest("Local runtime environments are not installed")
        self.assertEqual(workspace.interpreters(args), {k: str(v) for k, v in candidates.items()})

    def test_original_tabpfn_members_are_present_once(self):
        raw = load("raw")
        for family, expected in (("tab", 72), ("context", 36), ("source", 9)):
            members = raw.plans(family)
            self.assertEqual(len(members), expected)
            self.assertEqual(len(set(members)), expected)
            self.assertEqual({seed for _, seed in members}, {0, 100, 200})
        self.assertEqual({name for name, _ in raw.plans("source")}, {"control", *raw.TAGS})

    def test_heterogeneity_keeps_original_float32_mean(self):
        import numpy as np
        workspace = load("workspace")
        values = np.asarray([.15242852, .15114754, .16078259, .15239082], dtype=np.float32)
        expected = values[:3].mean(dtype=np.float32)
        self.assertNotEqual(float(expected), float(values[:3].astype(float).mean()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            maps = root / "experiments/push-500"
            base = root / "experiments/push-400"
            maps.mkdir(parents=True)
            base.mkdir()
            for name, value in zip(("m1", "m2", "h1", "m3"), values):
                np.savez(maps / f"uhet-{name}-test.npz", ids=["TEST_000001"],
                         keys=["het4_u", "het4_u_mx"], X=np.asarray([[value, value]], dtype=float))
            for name in ("u1dm1sz", "u1dm2sz", "u1dmavg", "u1dmavg4"):
                np.savez(base / f"feat-{name}-test.npz", ids=["TEST_000001"],
                         keys=["base"], X=[[1.]])
            workspace.heterogeneity(root, ["test"])
            with np.load(maps / "feat-u1dmavgH2-test.npz") as result:
                np.testing.assert_array_equal(result["X"], [[1., expected, expected]])

    def test_saved_legacy_context_heads_reproduce_original_predictions(self):
        import numpy as np
        import joblib
        exp = PACKAGE / "project/experiments"
        folder = exp / "push-1200/models"
        if not (folder / "manifest.json").is_file() or not (folder / "legacy073-ridge.joblib").is_file():
            self.skipTest("The model asset bundle is not installed")
        reference = np.load(exp / "push-1200/fin-d1234n12.npz")
        logs = []
        for name in ("u1dm1sz", "u1dm2sz", "u1dmavg", "u1dmavg4"):
            z = np.load(exp / f"push-800/feat-{name}H2Z-anghcor-test.npz")
            np.testing.assert_array_equal(z["ids"], reference["ids"])
            model = joblib.load(folder / f"legacy073-{name}H2Z-log.joblib")
            self.assertEqual(model[-1].alpha, 2.)
            logs.append(model.predict(z["X"].astype(float)))
            if name == "u1dmavg":
                ridge = joblib.load(folder / "legacy073-ridge.joblib")
                np.testing.assert_array_equal(ridge.predict(z["X"].astype(float)), reference["ridge"])
        np.testing.assert_array_equal(np.exp(np.mean(logs, 0)), reference["logridge"])


if __name__ == "__main__":
    unittest.main()
