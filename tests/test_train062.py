"""The submission must train without copying old fitted state or reading test data."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'final/package'
sys.path.insert(0, str(PACKAGE))
import train


class Train062Tests(unittest.TestCase):
    def test_failed_row_noise_is_zero_before_cnn_normalization(self):
        import numpy as np
        z = dict(X=np.array([[np.nan], [12 / 255.]]), keys=['noise'])
        np.testing.assert_array_equal(train.feature_noise(z), [0., 12.])

    def test_selected_shared_gmm_module_imports_and_predicts(self):
        import numpy as np
        path = PACKAGE / 'project/experiments/probabilistic-grains-900/probabilistic/contrast-adaptive/grain_measurement.py'
        gm = train.module('test_final_grain_measurement', path)
        sample = gm.GrainImage('TRAIN_000001', np.array([.1,.9]), np.zeros(2), np.array([.5,.5]),
                               np.array([[1.,0.,0.,0.,0.],[1.,0.,0.,0.,0.]]), np.zeros(10), .01, False)
        means, variances, _ = gm.transform([sample], 'image_gmm')
        result = gm.physical_readout(means, variances)
        self.assertEqual(result.shape, (1, 20))
        self.assertTrue(np.isfinite(result).all())

    def test_every_declared_training_source_exists_and_historical_drivers_are_excluded(self):
        selected = json.loads((PACKAGE / 'submission-files.json').read_text())
        for name in selected['code']:
            self.assertTrue((PACKAGE / name).is_file(), name)
        for name in ('report.py', 'finish_report.py', 'export_anchors.py', 'freeze_source_flow38_v2.py', 'nb_partners.py'):
            self.assertNotIn(name, [Path(n).name for n in selected['code']])
        self.assertLess(len(selected['code']), 153)

    def test_training_children_reject_both_real_and_linked_test_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, project = root / 'data', root / 'project'
            data.mkdir(); project.mkdir()
            (data / 'test').mkdir(); (data / 'train').mkdir()
            (data / 'test/TEST_000001.png').write_bytes(b'test')
            (data / 'train/TRAIN_000001.png').write_bytes(b'train')
            (project / 'data').symlink_to(data, target_is_directory=True)
            (project / 'alias').symlink_to(data / 'test', target_is_directory=True)
            script = '''
import train_guard
from pathlib import Path
import os
data=Path(os.environ['TRAIN062_DATA']); project=Path(os.environ['TRAIN062_PROJECT'])
assert (data/'train/TRAIN_000001.png').read_bytes()==b'train'
for p in (data/'test/TEST_000001.png',project/'data/test/TEST_000001.png',project/'alias/TEST_000001.png',data/'sample_submission.csv',project/'features-test.npz'):
    try: p.read_bytes()
    except RuntimeError: pass
    else: raise AssertionError(p)
'''
            subprocess.run([sys.executable, '-B', '-c', script], check=True,
                env=dict(os.environ, PYTHONPATH=str(PACKAGE), TRAIN062_DATA=str(data), TRAIN062_PROJECT=str(project)))

    def test_init_never_copies_fitted_weights_or_feature_banks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, data = root / 'source', root / 'data'
            source.mkdir(); data.mkdir(); (data / 'train').mkdir()
            (data / 'train.csv').write_text('ID,hardness\nTRAIN_000001,1\n')
            (data / 'train/TRAIN_000001.png').write_bytes(b'image')
            originals = {'original-train-hashes.json': dict(CSV_sha256=train.digest(data / 'train.csv'),
                image_sha256={'TRAIN_000001': train.digest(data / 'train/TRAIN_000001.png')})}
            config = 'project/' + train.G018 + '/config/original-train-hashes.json'
            train.save(source / config, originals['original-train-hashes.json'])
            (source / 'train.py').write_text('# code\n')
            (source / 'README.md').write_text('# instructions\n')
            train.save(source / 'model-provenance.json', {'models': []})
            train.save(source / 'submission-files.json', dict(code=['train.py'], training_config=[config], config=[config], assets=['project/old.pt','project/old-train.npz']))
            (source / 'project/old.pt').write_bytes(b'old trained weight')
            (source / 'project/old-train.npz').write_bytes(b'old feature')
            args = type('Args', (), dict(workspace=root / 'fresh', data=data))()
            with patch.object(train, 'PACKAGE', source), patch('prepare_pretrained.prepare'), patch('workspace.interpreters', return_value=dict(core=sys.executable,cellpose=sys.executable,tabpfn=sys.executable)):
                train.initialize(args)
            self.assertTrue((root / 'fresh/train.py').exists())
            self.assertFalse((root / 'fresh/project/old.pt').exists())
            self.assertFalse((root / 'fresh/project/old-train.npz').exists())
            self.assertFalse(train.read(root / 'fresh/training-state.json')['original_weights_reused'])


if __name__ == '__main__':
    unittest.main()
