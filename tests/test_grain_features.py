"""Engineering fixtures; these images are never used for model fitting."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[1]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(_repo_path(ROOT, "studies")))
from grain_features import extract_grains,segment
class GrainTests(unittest.TestCase):
    def test_constant_images_finite_consistent_schema(self):
        names=None
        for value in (0,128,255):
            f,n=extract_grains(np.full((256,256),value,dtype=np.uint8))
            self.assertTrue(np.isfinite(f).all());self.assertEqual(len(f),157)
            self.assertEqual(len(n),len(set(n)))
            if names is not None:self.assertEqual(n,names)
            names=n
    def test_segmentation_deterministic_valid_labels(self):
        yy,xx=np.indices((256,256));a=np.where((xx%32<2)|(yy%32<2),30,180).astype(np.uint8)
        first=segment(a,1.)[-1];second=segment(a,1.)[-1]
        np.testing.assert_array_equal(first,second)
        self.assertEqual(first.shape,a.shape);self.assertGreater(first.max(),10)
        self.assertGreaterEqual(first.min(),0)
