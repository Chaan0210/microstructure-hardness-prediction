from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[1]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(_repo_path(Path(__file__).resolve().parents[1], "studies")))
from native_texture_features import extract_native
from denoised_grain_features import extract_denoised

class NativeDenoisedTests(unittest.TestCase):
    def test_native_invariants(self):
        rng=np.random.default_rng(708)
        a=rng.integers(0,256,(256,256),dtype=np.uint8)
        x,n,b=extract_native(a)
        self.assertEqual((len(x),len(set(n)),len(b)),(28,28,28))
        for k in range(4):
            z,nn,bb=extract_native(np.fliplr(np.rot90(a,k)))
            np.testing.assert_allclose(x,z,atol=1e-10,rtol=0)
            self.assertEqual(n,nn);self.assertEqual(b,bb)
        np.testing.assert_array_equal(extract_native(np.full((256,256),127,dtype=np.uint8))[0],np.zeros(28))
    def test_denoised_known_grid(self):
        a=np.full((256,256),180,dtype=np.uint8);a[::32,:]=40;a[:,::32]=40
        x,n,b=extract_denoised(a)
        self.assertTrue(np.isfinite(x).all());self.assertEqual(len(set(n)),31)
        # Ground-truth fixture contains 8x8 cells. Dark line centers must not be counted as extra grains.
        count=x[n.index('denoised_grain_median_grain_count')]
        self.assertEqual(count,64)
        y,nn,bb=extract_denoised(np.rot90(a));np.testing.assert_array_equal(x,y)
        self.assertEqual(n,nn);self.assertEqual(b,bb)
    def test_invalid_input(self):
        for extractor in (extract_native,extract_denoised):
            with self.assertRaises(ValueError):extractor(np.zeros((128,128),dtype=np.uint8))
            with self.assertRaises(ValueError):extractor(np.zeros((256,256),dtype=float))
if __name__=='__main__':unittest.main()
