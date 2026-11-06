"""Image-local physics proxy engineering checks, never performance evidence."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[1]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(_repo_path(Path(__file__).resolve().parents[1], "studies")))
import numpy as np
from latent_features import extract_latent
from phase_features import extract_phase
from compact_regression import BalancedKernel
class PhysicsTests(unittest.TestCase):
    def test_constant_schema(self):
        for fn,count in [(extract_latent,36),(extract_phase,30)]:
            for value in (0,128,255):
                f,n=fn(np.full((256,256),value,dtype=np.uint8));self.assertEqual(len(f),count);self.assertEqual(len(n),len(set(n)));self.assertTrue(np.isfinite(f).all())
    def test_d4_invariance(self):
        yy,xx=np.indices((256,256));a=np.where((xx%32<2)|(yy%32<2),40,170).astype(np.uint8);a[40:100,40:100]=110
        for fn in (extract_latent,extract_phase):
            f,n=fn(a)
            for k in range(4):
                for b in (np.rot90(a,k),np.fliplr(np.rot90(a,k))):
                    g,m=fn(b);self.assertEqual(n,m);np.testing.assert_allclose(f,g,rtol=1e-9,atol=1e-9)
    def test_kernel_fit_predict_bounds_and_roundtrip(self):
        import joblib,io
        X=np.arange(60,dtype=float).reshape(20,3);y=np.sin(np.arange(20)/4)
        m=BalancedKernel(['a','a','b'],1,.1).fit(X,y)
        limits=m.bound_.data_min_.copy();p=m.predict(np.array([[-1e6]*3,[1e6]*3]));self.assertTrue(np.isfinite(p).all())
        np.testing.assert_array_equal(limits,m.bound_.data_min_)
        buf=io.BytesIO();joblib.dump(m,_repo_io(buf));buf.seek(0);np.testing.assert_allclose(joblib.load(_repo_io(buf)).predict(X),m.predict(X),rtol=0,atol=0)
