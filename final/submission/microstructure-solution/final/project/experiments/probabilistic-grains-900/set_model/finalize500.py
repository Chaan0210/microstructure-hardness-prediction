from pathlib import Path
import importlib.util
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("grain_set_run",HERE/"run.py")
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
SEEDS=[731503,731509,731527]

class InferenceTensorData:
    """Label-free adapter; batch returns an unused None instead of any targets."""
    def __init__(self, bank, context):
        self.device = "cpu"
        self.offsets = bank["offsets"]
        self.objects = run.torch.as_tensor(run.transform_norm(bank["objects"], context["object_norm"]))
        self.weights = run.torch.as_tensor(bank["weights"].astype(np.float32))
        self.globals = run.torch.as_tensor(run.transform_norm(bank["global_X"], context["global_norm"]))
        predicted = context["ridge"].predict(bank["summary"])
        self.skip = run.torch.as_tensor(((predicted - context["y_mean"]) / context["y_scale"]).astype(np.float32))

    def batch(self, images):
        images = np.asarray(images, int)
        rows = np.concatenate([np.arange(self.offsets[i], self.offsets[i + 1]) for i in images])
        group = np.repeat(np.arange(len(images)), self.offsets[images + 1] - self.offsets[images])
        ri, bi, ii = (run.torch.as_tensor(a) for a in (rows, group, images))
        return (self.objects[ri], self.weights[ri], bi, self.globals[ii], self.skip[ii]), None


def load_test_bank(path):
    with np.load(path, allow_pickle=False) as data:
        assert "y" not in data.files, "Inference bank must be label-free"
        bank = {k: data[k] for k in data.files}
    bank["ids"] = bank["ids"].astype(str)
    assert len(set(bank["ids"])) == len(bank["ids"]) and all(i.startswith("TEST_") for i in bank["ids"])
    bank["offsets"] = bank["offsets"].astype(np.int64)
    bank["objects"] = bank["objects"].astype(np.float64)
    bank["global_X"] = bank["global_X"].astype(np.float64)
    bank["weights"] = bank["weights"].astype(np.float64)
    assert bank["offsets"].shape == (len(bank["ids"]) + 1,)
    assert bank["offsets"][0] == 0 and bank["offsets"][-1] == len(bank["objects"])
    assert np.all(np.diff(bank["offsets"]) > 0)
    assert bank["objects"].shape[1] == len(bank["object_keys"])
    assert bank["global_X"].shape == (len(bank["ids"]), len(bank["global_keys"]))
    assert len(bank["weights"]) == len(bank["objects"]) and np.isfinite(bank["weights"]).all()
    for a, b in zip(bank["offsets"][:-1], bank["offsets"][1:]):
        assert bank["weights"][a:b].sum() > 0 and (bank["weights"][a:b] >= 0).all()
        bank["weights"][a:b] /= bank["weights"][a:b].sum()
    bank["summary"] = run.raw_summaries(bank)
    return bank

