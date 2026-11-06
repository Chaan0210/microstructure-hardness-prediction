import os
from pathlib import Path
import sys


def guard(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    mode, flags = args[1:3]
    writing = (mode is not None and any(c in mode for c in 'wax+')) or (
        isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
    if writing:
        return
    data = Path(os.environ['TRAIN062_DATA']).resolve()
    project = Path(os.environ['TRAIN062_PROJECT']).resolve()
    if (path.is_relative_to(data / 'test') or path == data / 'sample_submission.csv'
            or path.is_relative_to(project / 'data/test')
            or path == project / 'data/sample_submission.csv'
            or (path.is_relative_to(project) and path.name.endswith('-test.npz'))):
        raise RuntimeError(f'Test data cannot be read during training: {path}')


if os.environ.get('TRAIN062_DATA'):
    sys.addaudithook(guard)
