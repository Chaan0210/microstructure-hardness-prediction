"""Engineering fixtures only: never competition training data or performance evidence."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[1]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(_repo_path(ROOT, "studies", "src")))
from microstructure.features import extract, canonical_hash
from microstructure.pipeline import rmse, local_path, candidates
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.base import clone

class PipelineTests(unittest.TestCase):
    def test_rmse_known(self):
        self.assertAlmostEqual(rmse([1,2],[4,6]),np.sqrt(12.5))
        with self.assertRaises(ValueError): rmse([1],[float("nan")])
        with self.assertRaises(ValueError): rmse([1,2],[1])
    def test_path_guard(self):
        with self.assertRaises(ValueError): local_path("../secret")
        with self.assertRaises(ValueError): local_path("/tmp/file")
    def test_descriptors(self):
        # Pixel fixtures test edge cases, never fit any competition estimator.
        yy,xx=np.indices((256,256))
        fixtures=[np.zeros((256,256),dtype=np.uint8),
                  np.full((256,256),255,dtype=np.uint8),
                  ((xx+yy)%256).astype(np.uint8)]
        reference=None
        for img in fixtures:
            x,names=extract(img)
            self.assertTrue(np.isfinite(x).all())
            self.assertEqual(len(names),len(set(names)))
            if reference is not None: self.assertEqual(names,reference)
            reference=names
            self.assertEqual(canonical_hash(img),canonical_hash(np.fliplr(np.rot90(img))))
    def test_group_split_no_overlap(self):
        groups=np.repeat(np.arange(30),2); X=np.zeros((60,2)); y=np.arange(60)
        dev,hold=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=20260930).split(X,y,groups))
        self.assertFalse(set(groups[dev]) & set(groups[hold]))
        for tr,va in GroupKFold(5,shuffle=True,random_state=20260930).split(X[dev],y[dev],groups[dev]):
            self.assertFalse(set(groups[dev[tr]]) & set(groups[dev[va]]))
    def test_train_fitted_input_stabilization(self):
        from sklearn.preprocessing import MinMaxScaler
        train=np.array([[4.,10.],[5.,20.]])
        scaler=MinMaxScaler(clip=True).fit(train)
        np.testing.assert_allclose(scaler.transform([[78.,-10.]]),[[1.,0.]])
        np.testing.assert_allclose(scaler.data_min_,[4.,10.])
        np.testing.assert_allclose(scaler.data_max_,[5.,20.])

    def test_extra_morphology_edge_cases(self):
        from microstructure_v2.extra import extract_extra
        names=None
        for value in (0,255):
            x,n=extract_extra(np.full((256,256),value,dtype=np.uint8))
            self.assertTrue(np.isfinite(x).all())
            self.assertEqual(len(n),len(set(n)))
            if names is not None:self.assertEqual(names,n)
            names=n

    def test_estimator_fit_roundtrip(self):
        # Tiny numeric fixture checks estimator/persistence APIs, not accuracy.
        import tempfile
        import joblib
        X=np.arange(160,dtype=float).reshape(40,4)/160
        y=np.arange(40,dtype=float)
        with tempfile.TemporaryDirectory(dir=_repo_path(ROOT, 'artifacts'), prefix='engineering-test-') as td:
            for name,model in candidates(20260930,2).items():
                model.fit(X,y)
                pred=model.predict(X[:3])
                self.assertTrue(np.isfinite(pred).all())
                path=_repo_path(Path(td), name+'.joblib')
                joblib.dump(model,_repo_io(path))
                np.testing.assert_allclose(joblib.load(_repo_io(path)).predict(X[:3]),pred)

    def test_estimators_clone(self):
        for model in candidates(20260930,2).values():
            clone(model)

if __name__=="__main__":
    unittest.main()
