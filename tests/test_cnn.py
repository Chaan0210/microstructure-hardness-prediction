"""CNN engineering fixtures; no competition training or performance claims."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[1]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,unittest,io
from pathlib import Path
sys.path.insert(0,str(_repo_path(Path(__file__).resolve().parents[1], "studies")))
import torch
from cnn_experiment import TinyCNN
class CNNTests(unittest.TestCase):
    def test_forward_backward_finite(self):
        torch.set_num_threads(2);torch.manual_seed(20261003)
        model=TinyCNN();x=torch.linspace(0,1,2*128*128).reshape(2,1,128,128)
        p=model(x);self.assertEqual(tuple(p.shape),(2,));self.assertTrue(torch.isfinite(p).all())
        p.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
    def test_tensor_only_checkpoint_roundtrip(self):
        torch.manual_seed(20261003);model=TinyCNN().eval();x=torch.zeros(2,1,128,128)
        buf=io.BytesIO();torch.save(model.state_dict(),_repo_io(buf));buf.seek(0)
        restored=TinyCNN().eval();restored.load_state_dict(torch.load(_repo_io(buf),weights_only=True,map_location='cpu'))
        with torch.no_grad():torch.testing.assert_close(model(x),restored(x),rtol=0,atol=0)
