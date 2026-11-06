from pathlib import Path
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
STUDY = 'experiments/supervised-renewal-20261018'
G018 = 'experiments/reproduction-018-176'
CNN = 'experiments/rmse-research-20261015/round4-full073'
VARIANTS = ('u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4')
STAGES = ('g018', 'auxiliary', 'physical', 'augment', 'heads', 'source', 'teachers', 'cnn', 'finish')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def arrays(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def labels(project, ids):
    import numpy as np
    lab = {r['ID']: float(r['hardness']) for r in csv.DictReader((project / 'data/train.csv').open())}
    assert len(lab) == 500 and len(ids) == 500 and set(ids) == set(lab)
    return np.array([lab[str(i)] for i in ids])


def feature_noise(z):
    """The final CNN/teacher inputs stored failed-row noise as zero."""
    import numpy as np
    return np.nan_to_num(z['X'][:, list(z['keys']).index('noise')].astype(float) * 255, nan=0)


def initialize(a):
    import workspace
    import prepare_pretrained
    prepare_pretrained.prepare()
    envs = workspace.interpreters(a)
    data = Path(a.data).resolve()
    expected = read(PACKAGE / 'project' / G018 / 'config/original-train-hashes.json')
    assert digest(data / 'train.csv') == expected['CSV_sha256']
    for name, sha in expected['image_sha256'].items():
        assert digest(data / 'train' / f'{name}.png') == sha, name
    root = Path(a.workspace).resolve()
    root.mkdir(parents=True, exist_ok=False)
    selected = read(PACKAGE / 'submission-files.json')
    # No learned .pt/.joblib, feature bank, predictions or completion markers
    # from an older run are copied into this new training workspace.
    paths = set(selected['code'] + selected['training_config'])
    paths.update(('submission-files.json', 'model-provenance.json'))
    paths.update('project/' + row['local'] for row in read(PACKAGE / 'model-provenance.json')['models'])
    for name in sorted(paths):
        workspace.clone(PACKAGE / name, root / name)
    project = root / 'project'
    (project / 'data').symlink_to(data, target_is_directory=True)
    for kind, location in (('core', '.venv'), ('cellpose', 'experiments/grain-cellpose-220/.venv'),
                           ('tabpfn', 'experiments/tabpfn-v2/.venv')):
        target = project / location
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(Path(envs[kind]).parent.parent, target_is_directory=True)
    for name in ('experiments/push-1200/models', 'experiments/push-1300/abl',
                 'experiments/plateau-break-20261016-2035', STUDY + '/tmp',
                 STUDY + '/candidate-source-flow38-v1/features',
                 STUDY + '/final-nonlinear-readouts-20261019-0945/models'):
        (project / name).mkdir(parents=True, exist_ok=True)
    save(project / 'workspace.json', dict(data=str(data), python=envs, source_package=str(root),
                                            retrained=True, model_settings_changed=False))
    runtime = root / 'training-runtime'
    runtime.mkdir()
    (runtime / 'sitecustomize.py').write_text('import train_guard\n')
    save(root / 'training-state.json', dict(completed=[], original_weights_reused=False))
    print('Fresh training workspace:', root, flush=True)


def training_bank(project):
    import numpy as np
    sm = module('train062_set', project / 'experiments/probabilistic-grains-900/set_model/run.py')
    bank = arrays(project / 'experiments/probabilistic-grains-900/object-bank-train.npz')
    bank['ids'] = bank['ids'].astype(str)
    for key in ('objects', 'weights', 'global_X', 'y'):
        bank[key] = bank[key].astype(np.float64)
    bank['offsets'] = bank['offsets'].astype(np.int64)
    np.testing.assert_array_equal(bank['y'], labels(project, bank['ids']))
    for first, last in zip(bank['offsets'][:-1], bank['offsets'][1:]):
        assert last > first and bank['weights'][first:last].sum() > 0
        bank['weights'][first:last] /= bank['weights'][first:last].sum()
    bank['summary'] = sm.raw_summaries(bank)
    return sm, bank


def augmented(project, variant):
    import numpy as np
    path = project / f'experiments/push-800/feat-{variant}H2Z-anghcor-train.npz'
    z = arrays(path)
    X, y, ids = z['X'].astype(float), labels(project, z['ids']), z['ids']
    xx, yy, source = [X], [y], [ids]
    lab = dict(zip(ids, y))
    for tag in ('deg', 'deg2', 'deg3', 'deg4'):
        a = arrays(project / f'experiments/push-1200/synthrows-{tag}-{variant}H2Z-anghcor.npz')
        np.testing.assert_array_equal(a['keys'], z['keys'])
        use = a['X'][:, list(a['keys']).index('noise')].astype(float) * 255 >= 12
        xx.append(a['X'][use].astype(float))
        source.append(a['ids'][use])
        yy.append(np.array([lab[str(i)] for i in a['ids'][use]]))
    return z, np.vstack(xx), np.concatenate(yy), np.concatenate(source)


def fit_heads(project):
    import numpy as np
    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.model_selection import GroupKFold, KFold
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    exp = project / 'experiments'
    dest = exp / 'plateau-break-20261016-2035'
    legacy = exp / 'push-1200/models'
    groups = read(exp / 'push-1300/abl/groups.json')
    drop = set((groups['het'] + ' ' + groups['intensity']).split())
    def pipe(kind, alpha):
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=kind == 'ridge'),
                             StandardScaler(), Ridge(alpha=alpha))
    for variant in VARIANTS:
        z, AX, ay, source = augmented(project, variant)
        X, y = z['X'].astype(float), labels(project, z['ids'])
        noise = X[:, list(z['keys']).index('noise')] * 255
        keep = np.array([str(k) not in drop for k in z['keys']])
        # The CNN's older 073 context uses all augmented rows and alpha=2.
        joblib.dump(pipe('log', 2.).fit(AX, np.log(ay)), legacy / f'legacy073-{variant}H2Z-log.joblib')
        if variant == 'u1dmavg':
            model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                StandardScaler(), RidgeCV(alphas=[.1, 1, 10, 100, 1000])).fit(AX, ay)
            joblib.dump(model, legacy / 'legacy073-ridge.joblib')
        for condition in ('nb', 'B'):
            xx, yy, src = (X[:, keep], y, z['ids']) if condition == 'nb' else (AX, ay, source)
            for kind in (('log', 'ridge') if variant == 'u1dmavg' else ('log',)):
                target = np.log(yy) if kind == 'log' else yy
                if condition == 'B':
                    grid = [.3, 1, 2, 3, 10, 30, 100] if kind == 'log' else [.1, .3, 1, 3, 10, 30, 100, 1000]
                    losses = []
                    for alpha in grid:
                        errors = []
                        for tr, va in GroupKFold(5).split(xx, yy, src):
                            va = va[va < len(y)]
                            va = va[noise[va] >= 12]
                            m = pipe(kind, alpha).fit(xx[tr], target[tr], ridge__sample_weight=np.ones(len(tr)))
                            pred = m.predict(xx[va])
                            errors.extend(((np.exp(pred) if kind == 'log' else pred) - yy[va]) ** 2)
                        losses.append(float(np.mean(errors)))
                    model = pipe(kind, grid[int(np.argmin(losses))]).fit(xx, target, ridge__sample_weight=np.ones(len(yy)))
                elif kind == 'log':
                    model = pipe(kind, 2.).fit(xx, target)
                else:
                    model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                        StandardScaler(), RidgeCV(alphas=[.1, 1, 10, 100, 1000])).fit(xx, target)
                joblib.dump(model, dest / f'incumbent-{condition}-{variant}H2Z-{kind}.joblib')
    save(legacy / 'manifest.json', {'files': {p.name: digest(p) for p in sorted(legacy.glob('*.joblib'))}})
    sm, bank = training_bank(project)
    source_path = exp / 'push-900/gmm_partner.py'
    ns = {'__file__': str(source_path), '__name__': 'train062_gmm'}
    exec(compile(source_path.read_text().split('mode = sys.argv[1]')[0], str(source_path), 'exec'), ns)
    z = arrays(exp / 'push-800/feat-u1dmavgH2Z-anghcor-train.npz')
    y = labels(project, z['ids'])
    gmm = ns['gmm_feats'](bank, z['ids'].tolist())
    for gate in ('nb', 'full'):
        cols = np.array([gate == 'full' or str(k) not in drop for k in z['keys']])
        X = np.c_[z['X'].astype(float)[:, cols], gmm]
        losses = []
        for alpha in ns['ALPHAS']:
            errors = []
            for tr, va in KFold(3, shuffle=True, random_state=19001).split(X):
                errors.extend((ns['ridge'](alpha).fit(X[tr], y[tr]).predict(X[va]) - y[va]) ** 2)
            losses.append(float(np.mean(errors)))
        model = ns['ridge'](ns['ALPHAS'][int(np.argmin(losses))]).fit(X, y)
        joblib.dump(model, dest / f'incumbent-gmm-{gate}.joblib')
    out = exp / 'probabilistic-grains-900/set_model/full500'
    out.mkdir(parents=True, exist_ok=True)
    base_config = copy.deepcopy(sm.CONFIG)
    sm.torch.set_num_threads(2)
    threadpool_limits(2)
    tr = np.arange(len(bank['ids']))
    for seed in (731503, 731509, 731527):
        sm.CONFIG = {**copy.deepcopy(base_config), 'seed': seed}
        stopping = sm.train_with_inner_stop(bank, tr, 'cpu', seed)
        model, context, _, _ = sm.refit(bank, tr, 'cpu', seed, stopping['selected_epochs'])
        sm.save_checkpoint(out / f'seed{seed}.pt', bank, model, context,
                           dict(seed=seed, training_ids=bank['ids'].tolist(), selected_epochs=stopping['selected_epochs']))
    print('All final linear, GMM and GrainSet heads trained.', flush=True)


def fit_boosted(project):
    import numpy as np
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    h = project / STUDY
    z = arrays(project / 'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz')
    y = labels(project, z['ids'])
    settings = read(h / 'final-nonlinear-readouts-20261019-0945/frozen-recipe.json')['boostedreadout']
    kwargs = {k: v for k, v in settings.items() if k not in ('model', 'preprocessing', 'hyperparameters_searched')}
    out = h / 'final-nonlinear-readouts-20261019-0945/models'
    for tag in ('control', 'full500-sourceflow38-s1811', 'full500-sourceflow38-s1823'):
        X = z['X'].astype(float)
        if tag != 'control':
            flow = arrays(h / f'candidate-source-flow38-v1/features/{tag}-train.npz')
            np.testing.assert_array_equal(flow['ids'], z['ids'])
            X = np.c_[X, flow['flow']]
        model = HistGradientBoostingRegressor(**kwargs).fit(X, y)
        joblib.dump({'model': model}, out / f'boostedreadout-{tag}.joblib')
    print('All three final boosted readouts trained.', flush=True)


def fit_teachers(project):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from sklearn.model_selection import KFold
    folder = project / CNN
    sys.path.insert(0, str(folder))
    b = module('train062_branch', folder / 'full_branch.py')
    exp = project / 'experiments'
    z = arrays(exp / 'push-800/feat-u1dmavgH2Z-anghcor-train.npz')
    ids, y = z['ids'], labels(project, z['ids'])
    groups = read(exp / 'push-1300/abl/groups.json')
    drop = set((groups['het'] + ' ' + groups['intensity']).split())
    X = z['X'].astype(float)
    d = dict(ids=ids, X=X, keys=z['keys'], keep=np.array([str(k) not in drop for k in z['keys']]),
             noise=feature_noise(z),
             failed=np.nan_to_num(X[:, list(z['keys']).index('n_obj')], nan=0) < 20,
             banks={}, variants={}, source_hashes={})
    for slot, (directory, suffix) in b.SLOTS.items():
        for v in b.VARIANTS:
            p = exp / f'{directory}/feat-{v}{suffix}-train.npz'
            a = arrays(p)
            np.testing.assert_array_equal(a['ids'], ids)
            d['banks'][slot, v] = (a['X'].astype(float), np.array([str(k) not in drop for k in a['keys']]))
            d['source_hashes'][str(p.relative_to(project))] = digest(p)
    pos = {str(i): j for j, i in enumerate(ids)}
    for v in VARIANTS:
        _, AX, _, sources = augmented(project, v)
        d['variants'][v + 'H2Z'] = (d['banks']['044', v + 'H2Z'][0], AX[500:],
                                   np.array([pos[str(i)] for i in sources[500:]]))
    source_path = exp / 'push-900/gmm_partner.py'
    ns = {'__file__': str(source_path), '__name__': 'train062_gmm'}
    exec(compile(source_path.read_text().split('mode = sys.argv[1]')[0], str(source_path), 'exec'), ns)
    sm, bank = training_bank(project)
    np.testing.assert_array_equal(bank['ids'], ids)
    dependencies = [exp / 'push-1300/abl/groups.json', exp / 'probabilistic-grains-900/object-bank-train.npz',
                    source_path, exp / 'probabilistic-grains-900/probabilistic/contrast-adaptive/grain_measurement.py',
                    exp / 'probabilistic-grains-900/set_model/run.py']
    dependencies.extend(exp / f'push-1200/synthrows-{tag}-{v}H2Z-anghcor.npz'
                        for tag in ('deg', 'deg2', 'deg3', 'deg4') for v in VARIANTS)
    d['source_hashes'].update({str(p.relative_to(project)): digest(p) for p in dependencies})
    d['source_hashes']['training_driver'] = digest(Path(__file__))
    d.update(gmm_X=ns['gmm_feats'](bank, ids.tolist()), gmm=ns, set_module=sm,
             set_config=copy.deepcopy(sm.CONFIG), bank_index=np.arange(500), object_bank=bank)
    bank.pop('y')
    b.HERE = folder / 'final500'
    records = []
    for k, (tr, va) in enumerate(KFold(3, shuffle=True, random_state=2026101607).split(y)):
        b.fit(d, tr, va, y[tr].copy(), k, 'full500')
        records.append(dict(fold=k, train=tr.tolist(), query=va.tolist()))
    save(b.HERE / 'teacher-plan.json', dict(training_ids=ids.tolist(), folds=records, split_seed=2026101607))


def fit_cnn(project):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from PIL import Image
    folder = project / CNN
    sys.path.insert(0, str(folder))
    e = module('train062_cnn_data', folder / 'cnn_experiment.py')
    fitter = module('train062_cnn_fitter', folder / 'full500_fitter.py')
    z = arrays(project / 'experiments/push-800/feat-u1dmavgH2Z-anghcor-train.npz')
    ids, y = z['ids'], labels(project, z['ids'])
    images = np.stack([np.asarray(Image.open(project / f'data/train/{i}.png'), np.float32) / 255 for i in ids])[:, None]
    base, gap, count = np.full(500, np.nan), np.full(500, np.nan), np.zeros(500, int)
    sources = []
    for row in read(folder / 'final500/teacher-plan.json')['folds']:
        p = folder / f"final500/teachers/fold{row['fold']}-full500/predictions.npz"
        t = arrays(p)
        np.testing.assert_array_equal(t['train'], row['train'])
        np.testing.assert_array_equal(t['query'], row['query'])
        va = t['query']
        assert not set(t['train']) & set(va)
        base[va], gap[va] = t['base'], t['tab'] - t['ridge']
        count[va] += 1
        sources.append(dict(train_ids=ids[t['train']].tolist(), validation_ids=ids[va].tolist(),
                            file=str(p.relative_to(folder)), sha256=digest(p)))
    assert np.all(count == 1) and np.isfinite(base).all() and np.isfinite(gap).all()
    X, keys = z['X'].astype(float), list(z['keys'])
    raw = np.c_[base, gap, feature_noise(z), X[:, keys.index('u_ph')],
                np.log1p(np.maximum(np.nan_to_num(X[:, keys.index('n_obj')], nan=0), 0)), X[:, keys.index('coverage')]]
    data = e.cnn_data(dict(ids=ids, images=images), np.arange(500), np.array([], int), y.copy(),
                      base, sources, -1, 'full500_refit', raw)
    data['meta'].update(test_inputs_read=False, teacher_split_seed=2026101607, validation_role='none; full training fit')
    out = folder / 'final500/models'
    out.mkdir(parents=True, exist_ok=True)
    for j, seed in enumerate((2026101501, 2026201501)):
        fitter.fit_one(data, 'aux', seed, out / f'aux-seed{j}')
    path = folder / 'final500/recipe.json'
    recipe = read(path)
    recipe['normalization'] = data['meta']
    # The previously frozen noise threshold and mixture coefficients stay fixed.
    save(path, recipe)


def finish(root):
    selected = read(root / 'submission-files.json')
    paths = set(selected['code'] + selected['config'] + selected['assets'])
    learned = [n for n in selected['assets'] if Path(n).suffix in ('.pt', '.joblib')]
    for name in learned:
        if not (root / name).is_file():
            raise FileNotFoundError(f'Training did not create required model: {name}')
    files = []
    for name in sorted(paths):
        p = root / name
        if name.startswith('project/') and p.is_file():
            files.append(dict(path=name.removeprefix('project/'), bytes=p.stat().st_size,
                              sha256=digest(p), roles=['retrained_model_dependency']))
    save(root / 'source-manifest.json', dict(scope='Fresh train-data fit.', files=files))
    save(root / 'retrained-model.json', dict(complete=True, original_weights_reused=False,
        inference='raw.py --retrained --data DATA --out NEW.csv'))


def run_stage(root, stage):
    import numpy as np
    project = root / 'project'
    config = read(project / 'workspace.json')
    runtime = root / 'training-runtime'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join((str(runtime), str(root))),
               TRAIN062_DATA=config['data'], TRAIN062_PROJECT=str(project), HF_HUB_OFFLINE='1',
               TABPFN_DISABLE_TELEMETRY='1', TABPFN_MODEL_CACHE_SIZE='0')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = '2'
    for key in ('HEAVY', 'SRCQ', 'OFFMAX', 'NOMAPS', 'LA_MU', 'LA_SD', 'ALPHAS', 'LOGALPHAS', 'SHIFTFIX'):
        env.pop(key, None)
    env.pop('REPRO062_SPLITS', None)
    env.pop('REPRO062_RETRAINED', None)
    def execute(name, args=(), kind='core', extra=None):
        script = root / name
        print('TRAIN', stage, name, *args, flush=True)
        subprocess.run([config['python'][kind], '-B', str(script), *map(str, args)],
                       cwd=script.parent, env=dict(env, **(extra or {})), check=True)
    def internal(name, kind='core'):
        execute('train.py', [name, '--workspace', root], kind)
    if stage == 'g018':
        old = 'project/' + G018 + '/'
        execute(old + 'extract.py', ['train', '--data', project / 'data', '--run', project / G018 / 'run'])
        execute(old + 'train.py', ['prepare'])
        execute(old + 'train.py', ['fit'])
        execute(old + 'validate_full500.py')
        execute(old + 'export_models.py')
        execute('dinol.py', ['features', '--project', project, '--data', project / 'data', '--split', 'train',
                               '--out', project / 'dinol-train', '--retrained'])
        execute('dinol.py', ['train', '--project', project, '--data', project / 'data',
                               '--features', project / 'dinol-train/train-features.npz',
                               '--out', project / 'experiments/candidate021-178/models', '--retrained'])
    elif stage == 'auxiliary':
        folder = project / 'zero-shot-train'
        for action in ('zs_segment', 'zs_features'):
            execute('project/experiments/push-300/pipeline_grain.py', [action, 'train', '--data', project / 'data', '--out', folder], 'cellpose')
        for name in ('masks-v3-train.npz', 'features-v3-train.npz'):
            shutil.copy2(folder / name, project / 'experiments/grain-cellpose-220' / name)
        internal('_train_inputs')
        execute('project/experiments/push-300/ft_cellpose.py', ['a1', 800, 1, '2e-5', 4, 0], 'cellpose')
        execute('project/experiments/push-300/ft_cellpose.py', ['H300', 300, 0, '2e-5', 4, 0, 250, 'heavy'], 'cellpose')
        commands = [('phase_unet.py', ['u1', 3000, 0], {}),
                    ('phase_unet2.py', ['m1', 4000, 0, 0, 32], {}),
                    ('phase_unet2.py', ['m2', 4000, 1, 1, 32], {}),
                    ('phase_unet2.py', ['h1', 6000, 3, 0, 32], {'HEAVY': '.5'}),
                    ('phase_unet2.py', ['m3', 4000, 5, 1, 32], {}),
                    ('phase_unet4.py', ['z1', 4000, 0], {})]
        for name, args, extra in commands:
            execute('project/experiments/push-400/' + name, args, extra=extra)
    elif stage in ('physical', 'augment'):
        args = [stage, '--workspace', project]
        if stage == 'physical':
            args.extend(['--splits', 'train', '--retrained'])
        execute('workspace.py', args)
    elif stage == 'heads':
        internal('_heads')
    elif stage == 'source':
        execute('project/' + STUDY + '/source_h300_v2.py', ['prepare'], 'cellpose')
        for seed in (1811, 1823):
            tag = f'full500-sourceflow38-s{seed}'
            execute('project/' + STUDY + '/source_h300_v2.py', ['train', '--protocol', 'full500', '--tag', tag, '--seed', 2026100000 + seed], 'cellpose')
            execute('source_features.py', ['--project', project, '--data', project / 'data', '--split', 'train',
                '--seed', seed, '--out', project / 'trained-flow', '--retrained'], 'cellpose')
            shutil.copy2(project / f'trained-flow/{tag}-train.npz', project / STUDY / f'candidate-source-flow38-v1/features/{tag}-train.npz')
        internal('_hgb')
    elif stage == 'teachers':
        internal('_teachers', 'tabpfn')
    elif stage == 'cnn':
        internal('_cnn')
    elif stage == 'finish':
        finish(root)
    elif stage == '_train_inputs':
        from PIL import Image
        p = project / 'experiments/push-300'
        sys.path.insert(0, str(p))
        harness = module('train062_harness', p / 'harness.py')
        ids = np.array([r['ID'] for r in csv.DictReader((project / 'data/train.csv').open())])
        np.save(p / 'noise_train.npy', harness.noise_of(ids))
        np.savez(project / STUDY / 'data.npz', ids=ids)
        pixels = np.stack([np.asarray(Image.open(project / f'data/train/{i}.png'), np.uint8) for i in ids])[:, None]
        np.save(project / STUDY / 'train-pixels.npy', pixels)
    else:
        {'_heads': fit_heads, '_hgb': fit_boosted, '_teachers': fit_teachers, '_cnn': fit_cnn}[stage](project)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('init', 'all', *STAGES, '_train_inputs', '_heads', '_hgb', '_teachers', '_cnn'))
    p.add_argument('--workspace', required=True, type=Path)
    p.add_argument('--data', type=Path)
    for kind in ('core', 'cellpose', 'tabpfn'):
        p.add_argument('--python-' + kind)
    a = p.parse_args()
    if a.action == 'init':
        if not a.data:
            p.error('--data is required for init')
        initialize(a)
        return
    root = a.workspace.resolve()
    if a.action.startswith('_'):
        run_stage(root, a.action)
        return
    state = read(root / 'training-state.json')
    for stage in (STAGES if a.action == 'all' else (a.action,)):
        if stage in state['completed']:
            print('Already completed in this fresh run:', stage, flush=True)
            continue
        required = STAGES[:STAGES.index(stage)]
        if not set(required) <= set(state['completed']):
            raise RuntimeError(f'Complete preceding stages first: {required}')
        run_stage(root, stage)
        state['completed'].append(stage)
        save(root / 'training-state.json', state)
    if state['completed'] == list(STAGES):
        print('Training complete. Run the workspace raw.py with --retrained for inference.', flush=True)


if __name__ == '__main__':
    main()
