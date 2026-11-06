"""Train-only hierarchical grain measurements and exact distribution RBF kernels."""
from __future__ import annotations
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[6]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT

import hashlib
import json
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_key, "2")

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


HERE = Path(__file__).resolve().parent
MEAN_KEYS = [
    "phase_fraction", "phase_logarea_centered", "phase_logar_centered",
    "phase_cos2", "phase_sin2", "mean_logarea", "std_logarea",
    "mean_area_inverse_quarter", "mean_logar", "mean_cos2", "mean_sin2",
    "log_effective_grain_count", "coverage", "relative_noise", "log_contrast",
]
ARMS = ("threshold", "image_gmm", "hierarchical")
CONFIG = {
    "version": 2, "adaptive_initial_means": [[0.05, 0.35], [0.10, 0.75]],
    "adaptive_warm_iterations": 8, "adaptive_refine_iterations": 12,
    "adaptive_min_gap": 0.10, "adaptive_bright_prior_pseudoimages": 0.5,
    "adaptive_initialization_selection": "balanced_train_laplace_evidence", "scope": "train400_only", "arms": list(ARMS),
    "inner_splits": 3, "alpha_ridge": [0.1, 1., 10., 100., 1000.],
    "blend_new_weight": 0.2, "threshold": 0.5, "pore_exclusion": True,
    "global_em_iterations": 20, "local_em_iterations": 16,
    "quadrature_order": 7, "quadrature_orders": [7, 15, 25, 41, 61, 81, 121, 161], "posterior_starts_eta": ["prior", -4., 4.], "beta_concentration": 2.,
    "ridge_inputs": ["physical", "x98_augmented"],
    "kernel_lengths": [0.5, 1., 2.], "kernel_alpha": [0.003, 0.03, 0.3, 3.],
    "kernel_uncertainty_scale": [0.25, 1., 4.],
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def safe_output(path):
    path = Path(path).resolve()
    if HERE not in path.parents and path != HERE:
        raise ValueError(f"Output must remain below {HERE}: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class GrainImage:
    image_id: str
    d: np.ndarray
    pixel_var: np.ndarray
    area_weights: np.ndarray
    phase_basis: np.ndarray
    physical: np.ndarray
    shade_var: float
    empty: bool


class ExperimentData:
    def __init__(self, bank_path, base_path):
        self.bank_path, self.base_path = Path(bank_path), Path(base_path)
        bank = np.load(_repo_io(self.bank_path), allow_pickle=False)
        base = np.load(_repo_io(self.base_path), allow_pickle=False)
        self.ids = np.asarray(base["ids"]).astype(str)
        if len(self.ids) != 400 or len(np.unique(self.ids)) != 400:
            raise ValueError("Primary experiment requires exactly 400 unique baseline IDs")
        bank_ids = np.asarray(bank["ids"]).astype(str)
        if len(np.unique(bank_ids)) != len(bank_ids):
            raise ValueError("Duplicated object-bank image IDs")
        pos = {i: j for j, i in enumerate(bank_ids)}
        if not set(self.ids).issubset(pos):
            raise ValueError("Baseline IDs missing from object bank")
        bank_rows = np.array([pos[i] for i in self.ids])
        self.y = np.asarray(base["y400"] if "y400" in base else base["y"], dtype=float)
        bank_y = np.asarray(bank["y"], dtype=float)[bank_rows]
        if not np.allclose(self.y, bank_y, rtol=0, atol=1e-6):
            raise ValueError("Labels differ between base and object bank")
        self.folds = np.asarray(base["fold"], dtype=int)
        self.baseline = np.asarray(base["baseline"], dtype=float)
        self.p021 = np.asarray(base["p021"], dtype=float)
        self.grain = np.asarray(base["grain"], dtype=float)
        self.failed = np.asarray(base["failed"], dtype=bool)
        self.X98 = np.asarray(base["X98"], dtype=float)
        self.split_names = np.asarray(base["split_names"]).astype(str)
        if self.folds.shape != (3, 400) or self.baseline.shape != (3, 400):
            raise ValueError("Expected three aligned 400-row partitions")
        if self.failed.shape != self.folds.shape or self.X98.shape != (400, 98):
            raise ValueError("Invalid baseline auxiliary arrays")
        if not np.isfinite(self.y).all() or not np.isfinite(self.baseline).all():
            raise ValueError("Non-finite labels or baseline predictions")
        for fold in self.folds:
            if set(fold) != set(range(5)):
                raise ValueError("Each partition must contain folds 0..4")
        obj = np.asarray(bank["objects"], dtype=float)
        offsets = np.asarray(bank["offsets"], dtype=int)
        weights = np.asarray(bank["weights"], dtype=float)
        if offsets.shape != (len(bank_ids) + 1,) or offsets[0] != 0 or offsets[-1] != len(obj):
            raise ValueError("Malformed ragged object bank")
        if np.any(np.diff(offsets) < 0) or len(weights) != len(obj):
            raise ValueError("Malformed object offsets or weights")
        object_keys = {str(k): j for j, k in enumerate(bank["object_keys"])}
        global_keys = {str(k): j for j, k in enumerate(bank["global_keys"])}
        global_X = np.asarray(bank["global_X"], dtype=float)
        required_objects = {"area", "log_area", "intensity_se2", "shade_darkness", "log_ar", "cos2", "sin2", "pore_flag"}
        required_globals = {"contrast_span", "noise_gray", "coverage"}
        if not required_objects.issubset(object_keys) or not required_globals.issubset(global_keys):
            raise ValueError(f"Missing bank columns: objects={required_objects-set(object_keys)}, globals={required_globals-set(global_keys)}")
        self.images = []
        for image_id, row in zip(self.ids, bank_rows):
            z = obj[offsets[row]:offsets[row+1]]
            g = global_X[row]
            get = lambda key: z[:, object_keys[key]]
            span = max(float(g[global_keys["contrast_span"]]), 1.)
            ok = (get("area") > 0) & (get("pore_flag") < 0.5)
            for key in required_objects:
                ok &= np.isfinite(get(key))
            z = z[ok]
            empty = len(z) == 0 or ("whole_image_fallback" in object_keys and len(z) > 0 and np.any(get("whole_image_fallback") > .5))
            if empty:
                d, v, a = np.array([0.]), np.array([0.01]), np.array([1.])
                basis = np.array([[1., 0., 0., 0., 0.]])
                phys = np.zeros(10)
                shade_var = 0.01
            else:
                get = lambda key: z[:, object_keys[key]]
                area = get("area")
                a = area / area.sum()
                la, ar = get("log_area"), get("log_ar")
                c, s = get("cos2"), get("sin2")
                mean_la, mean_ar = np.dot(a, la), np.dot(a, ar)
                basis = np.column_stack([np.ones(len(z)), la-mean_la, ar-mean_ar, c, s])
                d = np.clip(get("shade_darkness"), -1.5, 2.5)
                v = np.clip(get("intensity_se2") / span**2, 1e-8, 1.)
                shade_var = float(np.dot(a, np.clip(get("shade_se2"), 0, np.inf))/span**2) if "shade_se2" in object_keys else 0.
                shade_var = float(np.clip(shade_var, 0., 0.09))
                noise = max(float(g[global_keys["noise_gray"]]), 0.) / span
                cov = np.clip(float(g[global_keys["coverage"]]), 0., 1.)
                phys = np.array([
                    mean_la, np.sqrt(np.dot(a, (la-mean_la)**2)),
                    np.dot(a, area**-0.25), mean_ar, np.dot(a, c), np.dot(a, s),
                    np.log(1./np.dot(a, a)), cov, noise, np.log(span),
                ])
            self.images.append(GrainImage(image_id, d, v, a, basis, phys, shade_var, empty))
        self.empty = np.array([im.empty for im in self.images])
        self.provenance = {
            "bank_path": str(self.bank_path.resolve()), "bank_sha256": sha256(self.bank_path),
            "base_path": str(self.base_path.resolve()), "base_sha256": sha256(self.base_path),
            "measurement_code_sha256": sha256(__file__), "adaptive_fit_code_sha256": sha256(_repo_path(HERE, "fit_adaptive.py")), "posterior_code_sha256": sha256(_repo_path(HERE, "posterior_multimode.py")), "configuration": CONFIG,
            "primary_ids": self.ids.tolist(), "primary_count": 400,
            "unused_bank_ids": sorted(set(bank_ids)-set(self.ids)),
            "n_grains_used": int(sum(len(i.d) for i in self.images if not i.empty)),
            "empty_ids": self.ids[self.empty].tolist(),
            "feature_mean_keys": MEAN_KEYS,
        }
        self.cache_key = hashlib.sha256(json.dumps(self.provenance, sort_keys=True).encode()).hexdigest()


def _responsibility(d, v, mu, tau2, offset, pi):
    var = v[:, None] + tau2
    logp = -0.5 * (np.log(var) + (d[:, None]-offset-mu)**2/var)
    logp += np.log([1.-pi, pi])
    return np.exp(logp[:, 1]-logsumexp(logp, axis=1))


def phase_moments(image, q):
    coefficients = image.area_weights[:, None] * image.phase_basis
    mean = q @ coefficients
    var = (q*(1.-q)) @ (coefficients**2)
    return mean, var


def threshold(image):
    q = (image.d >= CONFIG["threshold"]).astype(float)
    return (*phase_moments(image, q), {"offset": 0., "offset_sd": 0., "prevalence": float(q.mean()), "ess": 1.})


def image_gmm(image):
    d = image.d
    mu = np.quantile(d, [.2, .8])
    if mu[1]-mu[0] < .1:
        mu = np.array([min(mu[0], .3), max(mu[1], .7)])
    tau2 = np.array([.12**2, .12**2])
    pi = .35
    for _ in range(30):
        q = _responsibility(d, np.zeros_like(d), mu, tau2, 0., pi)
        resp = np.column_stack([1.-q, q])
        count = resp.sum(0) + 1e-8
        mu = (resp*d[:, None]).sum(0)/count
        tau2 = np.maximum((resp*(d[:, None]-mu)**2).sum(0)/count, .04**2)
        pi = float(np.clip(q.mean(), .01, .99))
        if mu[0] > mu[1]:
            mu, tau2, pi = mu[::-1], tau2[::-1], 1.-pi
    q = _responsibility(d, np.zeros_like(d), mu, tau2, 0., pi)
    return (*phase_moments(image, q), {"offset": 0., "offset_sd": 0., "prevalence": pi, "ess": 1.})


@dataclass
class HierarchicalMeasurement:
    mu: np.ndarray
    tau2: np.ndarray
    offset_var: float
    prior_phase_mean: float
    train_ids: list
    fit_history: list

    @classmethod
    def fit(cls, images):
        from fit_adaptive import fit_adaptive
        return fit_adaptive(cls, images, CONFIG)

    def _posterior(self, image, point, hessian=False):
        b, eta = point
        pi = expit(eta)
        var = image.pixel_var[:, None] + self.tau2
        residual = image.d[:, None]-b-self.mu
        logp = -.5*(np.log(var)+residual**2/var)
        logp += np.array([-np.logaddexp(0., eta), -np.logaddexp(0., -eta)])
        ll = logsumexp(logp, axis=1)
        q = np.exp(logp[:, 1]-ll)
        prior_var = self.offset_var + image.shade_var
        a = CONFIG["beta_concentration"]*self.prior_phase_mean
        c = CONFIG["beta_concentration"]*(1.-self.prior_phase_mean)
        # Posterior density in (offset, logit prevalence), including Jacobian.
        lp = ll.sum()-.5*b*b/prior_var-a*np.logaddexp(0., -eta)-c*np.logaddexp(0., eta)
        g = residual/var
        grad = np.array([np.sum((1.-q)*g[:, 0]+q*g[:, 1])-b/prior_var,
                         np.sum(q-pi)+a-(a+c)*pi])
        if not hessian:
            return -float(lp), -grad, q
        qq = q*(1.-q)
        diff = g[:, 1]-g[:, 0]
        hbb = np.sum(-(1.-q)/var[:, 0]-q/var[:, 1]+qq*diff**2)-1./prior_var
        hee = np.sum(qq)-len(q)*pi*(1.-pi)-(a+c)*pi*(1.-pi)
        hbe = np.sum(qq*diff)
        return -float(lp), -grad, q, -np.array([[hbb, hbe], [hbe, hee]])

    def infer(self, image):
        from posterior_multimode import infer_multimode
        return infer_multimode(self, image, CONFIG)


def transform(images, method, model=None):
    means, variances, diagnostics = [], [], []
    for image in images:
        if method == "threshold":
            m, v, diag = threshold(image)
        elif method == "image_gmm":
            m, v, diag = image_gmm(image)
        elif method == "hierarchical":
            m, v, diag = model.infer(image)
        else:
            raise ValueError(method)
        means.append(np.r_[m, image.physical])
        variances.append(np.r_[v, np.zeros(10)])
        diagnostics.append(diag)
    means, variances = np.asarray(means), np.asarray(variances)
    if means.shape[1] != len(MEAN_KEYS) or not np.isfinite(means).all() or not np.isfinite(variances).all():
        raise ValueError("Invalid measurement features")
    return means, variances, diagnostics


def physical_readout(means, variances):
    return np.column_stack([means, np.sqrt(np.maximum(variances[:, :5], 0.))])


def cache_matches(data, path, result):
    """Accept current caches or byte-attested converged caches from cap extension."""
    if str(result["cache_key"]) == data.cache_key:
        return True
    manifest_path = _repo_path(HERE, "cache-compatibility-order-extension.json")
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    if data.cache_key != manifest.get("new_cache_key"):
        return False
    old = manifest["old_provenance"]
    if str(result["cache_key"]) != manifest["old_cache_key"]:
        return False
    for key in ["bank_sha256", "base_sha256", "adaptive_fit_code_sha256", "posterior_code_sha256", "primary_ids"]:
        if old[key] != data.provenance[key]:
            return False
    previous_config, current_config = dict(old["configuration"]), dict(CONFIG)
    old_orders = previous_config.pop("quadrature_orders")
    new_orders = current_config.pop("quadrature_orders")
    if previous_config != current_config or new_orders[:len(old_orders)] != old_orders or new_orders != manifest["new_quadrature_orders"]:
        return False
    relative = str(Path(path).resolve().relative_to(HERE))
    entries = [e for e in manifest["entries"] if e["npz"] == relative]
    if len(entries) != 1:
        return False
    entry = entries[0]
    for label in ["npz", "prior", "diagnostics"]:
        if sha256(_repo_path(HERE, entry[label])) != entry[label+"_sha256"]:
            return False
    if not np.array_equal(result["train_idx"], entry["train_idx"]) or not np.array_equal(result["eval_idx"], entry["eval_idx"]):
        return False
    diagnostics = json.loads((_repo_path(HERE, entry["diagnostics"])).read_text())["diagnostics"]
    for tag in ["train", "eval"]:
        for record in diagnostics["hierarchical_"+tag]:
            if not record["quadrature_converged"] or len(record["convergence_history"]) < 2 or not all(r["stable"] for r in record["convergence_history"][-2:]):
                return False
    return True


def fold_features(data, train_idx, eval_idx, cache_dir, name):
    """Learn priors using train_idx only; cache train/eval feature representations."""
    cache_dir = safe_output(cache_dir)
    path, model_path = cache_dir/f"{name}.npz", cache_dir/f"{name}-measurement.pkl"
    tr, ev = np.asarray(train_idx, int), np.asarray(eval_idx, int)
    if set(tr)&set(ev):
        raise ValueError("Training/validation images overlap")
    if path.exists() and model_path.exists():
        result = dict(np.load(_repo_io(path), allow_pickle=False))
        if not cache_matches(data, path, result) or not np.array_equal(result["train_idx"], tr) or not np.array_equal(result["eval_idx"], ev):
            raise ValueError(f"Stale cache {path}; use a new output folder")
        return result
    train_images, eval_images = [data.images[i] for i in tr], [data.images[i] for i in ev]
    model = HierarchicalMeasurement.fit(train_images)
    assert set(model.train_ids).isdisjoint(data.ids[ev])
    result = {"cache_key": np.array(data.cache_key), "train_idx": tr, "eval_idx": ev}
    diag = {}
    for arm in ARMS:
        for tag, images in (("train", train_images), ("eval", eval_images)):
            mean, var, diagnostics = transform(images, arm, model)
            result[f"{arm}_{tag}_mean"] = mean
            result[f"{arm}_{tag}_var"] = var
            diag[f"{arm}_{tag}"] = diagnostics
    np.savez_compressed(_repo_io(path), **result)
    with model_path.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    json_write(cache_dir/f"{name}-diagnostics.json", {"training_ids": model.train_ids,
               "fit_history": model.fit_history, "diagnostics": diag})
    return result


def expected_rbf(mean_a, var_a, mean_b, var_b, length):
    """Analytic RBF expectation for diagonal Gaussian-distributed inputs."""
    length2 = float(length)**2
    denom = length2 + var_a[:, None, :] + var_b[None, :, :]
    delta2 = (mean_a[:, None, :]-mean_b[None, :, :])**2
    return np.exp(-.5*np.sum(np.log(denom/length2)+delta2/denom, axis=2))


def metrics(data, prediction):
    pred = np.asarray(prediction, dtype=float)
    if pred.shape != data.baseline.shape or not np.isfinite(pred).all():
        raise ValueError("Incomplete/nonfinite predictions")
    fallback = data.failed | data.empty[None, :]
    mixed = .8*data.baseline+.2*pred
    mixed = np.where(fallback, data.baseline, mixed)
    same021 = .2*data.p021+.8*(.8*data.grain+.2*pred)
    same021 = np.where(fallback, data.baseline, same021)
    y = data.y[None, :]
    rmse = lambda z: float(np.sqrt(np.mean((z-y)**2)))
    parts = lambda z: np.sqrt(np.mean((z-y)**2, axis=1)).tolist()
    folds = lambda z: [[float(np.sqrt(np.mean((z[s, data.folds[s] == f]-data.y[data.folds[s] == f])**2))) for f in range(5)] for s in range(3)]
    return {"standalone_rmse": rmse(pred), "blend20_rmse": rmse(mixed),
            "same021_blend20_rmse": rmse(same021), "same021_by_partition": parts(same021),
            "baseline_rmse": rmse(data.baseline), "standalone_by_partition": parts(pred),
            "blend20_by_partition": parts(mixed), "baseline_by_partition": parts(data.baseline),
            "blend20_by_fold": folds(mixed), "baseline_by_fold": folds(data.baseline),
            "fallback_predictions": int(fallback.sum())}
