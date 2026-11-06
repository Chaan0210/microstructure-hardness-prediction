from pathlib import Path
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
os.environ["TABPFN_MODEL_CACHE_SIZE"] = "0"

PACKAGE = Path(__file__).resolve().parent
FROZEN = PACKAGE / "project"
STUDY = "experiments/supervised-renewal-20261018"
TAGS = ("full500-sourceflow38-s1811", "full500-sourceflow38-s1823")
VARIANTS = ("u1dm1sz", "u1dm2sz", "u1dmavg", "u1dmavg4")
SLOTS = {"038": ("push-500", ""), "039": ("push-700", "-angh"), "044": ("push-800", "-anghcor")}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def check(a):
    import workspace
    envs = workspace.interpreters(a)
    manifest = read_json(PACKAGE / "source-manifest.json")
    for row in manifest["files"]:
        path = FROZEN / row["path"]
        if not path.is_file() or digest(path) != row["sha256"]:
            raise RuntimeError(f"Missing or changed packaged dependency: {path}")
    for kind, imports in {
        "core": "import numpy, scipy, sklearn, PIL, torch; assert torch.backends.mps.is_available(), 'MPS is required'",
        "cellpose": "import torch, cellpose; from cellpose.vit import CPSAM; assert torch.backends.mps.is_available()",
        "tabpfn": "import torch, tabpfn; from tabpfn.constants import ModelVersion; ModelVersion('v3.5')",
    }.items():
        recorded = read_json(FROZEN / STUDY / f"environment-{kind}.json")["packages"]
        expected = {name: version for name, version in recorded.items() if version is not None}
        version_check = (
            "; import importlib.metadata as md,json,sys"
            "; expected=json.loads(sys.argv[1])"
            "; changed=[(k,v,md.version(k)) for k,v in expected.items() if md.version(k)!=v]"
            "; assert not changed, ('Reproduction dependency versions differ',changed)"
        )
        subprocess.run([envs[kind], "-B", "-c", imports + version_check, json.dumps(expected)], check=True)
    data = Path(a.data).resolve()
    for split, name, count in (("train", "train.csv", 500), ("test", "sample_submission.csv", 1000)):
        rows = list(csv.DictReader((data / name).open()))
        ids = [row["ID"] for row in rows]
        assert len(ids) == len(set(ids)) == count, name
        assert all((data / split / f"{i}.png").is_file() for i in ids), split
        original = read_json(FROZEN / f"experiments/reproduction-018-176/config/original-{split}-hashes.json")
        assert digest(data / name) == original["CSV_sha256"], f"Official CSV differs: {name}"
        for i in ids:
            assert digest(data / split / f"{i}.png") == original["image_sha256"][i], f"Official image differs: {i}"
    print(f"Dependencies ready: {len(manifest['files'])} files; 500 train / 1000 test images; three local environments.", flush=True)
    return envs


def g018(project, data, verify_reference=True):
    import numpy as np
    import torch
    old = project / "experiments/reproduction-018-176"
    sys.path[:0] = [str(old), str(old / "vendor")]
    extract = module("raw062_extract", old / "extract.py")
    from dinov2.hub.backbones import dinov2_vits14, dinov2_vitb14
    ids = [row["ID"] for row in csv.DictReader((data / "sample_submission.csv").open())]
    image_data = Path(read_json(project / "workspace.json")["data"]).resolve()
    records = [extract.one(image_data, "test", i) for i in ids]
    result = dict(ids=np.array(ids), K=np.stack([r[0] for r in records]),
                  X57=np.stack([r[1] for r in records]), images=np.stack([r[2] for r in records]))
    for arm, factory in (("S", dinov2_vits14), ("B", dinov2_vitb14)):
        net = factory(pretrained=False)
        net.load_state_dict(torch.load(old / f"weights/DINO-{arm}.pth", map_location="cpu", weights_only=True), strict=True)
        net.requires_grad_(False).eval().to("mps")
        values, spatial = [], []
        for index, i in enumerate(ids):
            with torch.inference_mode():
                x = torch.from_numpy(extract.dino_input(image_data / f"test/{i}.png")[None]).to("mps")
                layers = net.get_intermediate_layers(x, n=[5, 8, 11], return_class_token=True, norm=True)
                values.append(torch.cat([torch.cat([c, p.mean(1), p.std(1, correction=0)], 1)
                                         for p, c in layers], 1).cpu().numpy()[0])
                if arm == "B":
                    spatial.append(extract.spatial_descriptor(layers[-1][0].cpu().numpy()[0]))
            if (index + 1) % 100 == 0:
                print("Raw DINO", arm, index + 1, "/", len(ids), flush=True)
        result[arm] = np.stack(values).astype(float)
        if arm == "B":
            result["spatial"] = np.stack(spatial)
            result["X"] = np.c_[result["X57"], result["spatial"]]
        del net
        torch.mps.empty_cache()
    if verify_reference:
        reference = arrays(FROZEN / "experiments/reproduction-018-176/test-run/test-features.npz")
        positions = {str(i): j for j, i in enumerate(reference["ids"])}
        rows = [positions[i] for i in ids]
        for name in ("K", "X57", "images", "S", "B", "spatial", "X"):
            np.testing.assert_array_equal(result[name], reference[name][rows])
    np.savez_compressed(project / "raw-g018.npz", **result)


def standard(project):
    import numpy as np
    import joblib
    import torch
    torch.set_num_threads(2)
    exp = project / "experiments"
    old = exp / "reproduction-018-176"
    saved = exp / "plateau-break-20261016-2035"
    groups = read_json(exp / "push-1300/abl/groups.json")
    drop = set((groups["het"] + " " + groups["intensity"]).split())
    main = arrays(exp / "push-800/feat-u1dmavgH2Z-anghcor-test.npz")
    ids, keys = main["ids"], list(main["keys"])
    bank = arrays(exp / "probabilistic-grains-900/object-bank-test.npz")
    np.testing.assert_array_equal(bank["ids"], ids)
    source = exp / "push-900/gmm_partner.py"
    ns = {"__file__": str(source), "__name__": "raw062_gmm"}
    exec(compile(source.read_text().split("mode = sys.argv[1]")[0], str(source), "exec"), ns)
    measured = ns["gmm_feats"](bank, ids.tolist())
    predictions = {}
    for condition, gate in (("nb", "low"), ("B", "high")):
        logs = []
        for variant in VARIANTS:
            z = arrays(exp / f"push-800/feat-{variant}H2Z-anghcor-test.npz")
            np.testing.assert_array_equal(z["ids"], ids)
            keep = np.array([condition == "B" or str(k) not in drop for k in z["keys"]])
            X = z["X"].astype(float)[:, keep]
            model = joblib.load(saved / f"incumbent-{condition}-{variant}H2Z-log.joblib")
            logs.append(np.log(np.exp(model.predict(X))))
            if variant == "u1dmavg":
                predictions["ridge_" + gate] = joblib.load(saved / f"incumbent-{condition}-{variant}H2Z-ridge.joblib").predict(X)
        predictions["logridge_" + gate] = np.exp(np.mean(logs, 0))
        keep = np.array([gate == "high" or str(k) not in drop for k in keys])
        model = joblib.load(saved / f"incumbent-gmm-{'full' if gate == 'high' else 'nb'}.joblib")
        predictions["gmm_" + gate] = model.predict(np.c_[main["X"].astype(float)[:, keep], measured])
    legacy = exp / "push-1200/models"
    for name, expected in read_json(legacy / "manifest.json")["files"].items():
        assert digest(legacy / name) == expected, name
    predictions["legacy_ridge"] = joblib.load(legacy / "legacy073-ridge.joblib").predict(main["X"].astype(float))
    logs = [joblib.load(legacy / f"legacy073-{v}H2Z-log.joblib").predict(
        arrays(exp / f"push-800/feat-{v}H2Z-anghcor-test.npz")["X"].astype(float)) for v in VARIANTS]
    predictions["legacy_logridge"] = np.exp(np.mean(logs, 0))

    # The guarded G018 inference below disables fitting process-wide.
    sys.path.insert(0, str(old))
    infer = module("raw062_infer018", old / "infer.py")
    g018_features = arrays(project / "raw-g018.npz")
    np.testing.assert_array_equal(g018_features["ids"], ids)
    g18 = infer.predict(g018_features, old / "models", "cpu")["primary"]
    L = arrays(project / "raw-dinol/test-features.npz")
    np.testing.assert_array_equal(L["ids"], ids)
    lvalues = [infer.head(L["X"], torch.load(exp / f"candidate021-178/models/L-MLP-{seed}.pt",
                                            map_location="cpu", weights_only=True))
               for seed in (20261128, 20261205, 20261219)]
    G = .9 * g18 + .1 * np.mean(lvalues, 0)
    sm = module("raw062_set", exp / "probabilistic-grains-900/set_model/finalize500.py")
    test_bank = sm.load_test_bank(exp / "probabilistic-grains-900/object-bank-test.npz")
    sets = []
    for seed in sm.SEEDS:
        ck = torch.load(exp / f"probabilistic-grains-900/set_model/full500/seed{seed}.pt",
                        map_location="cpu", weights_only=False)
        sm.run.CONFIG = ck["config"]
        model = sm.run.make_model(test_bank, "cpu", seed)
        model.load_state_dict(ck["model_state"])
        tensors = sm.InferenceTensorData(test_bank, ck["context"])
        sets.append(sm.run.predict_y(model, tensors, ck["context"], np.arange(len(ids))))
    # Original make_053.py averages raw seed predictions in float32.
    set_raw = np.asarray(sets, dtype=np.float32).mean(0)
    object_column = list(bank["object_keys"]).index("whole_image_fallback")
    failed_objects = np.array([np.any(bank["objects"][a:b, object_column] > .5)
                               for a, b in zip(bank["offsets"][:-1], bank["offsets"][1:])])
    failed = (np.nan_to_num(main["X"][:, keys.index("n_obj")], nan=0) < 20) | failed_objects
    noise = np.nan_to_num(main["X"][:, keys.index("noise")].astype(float) * 255, nan=0)
    high = (noise >= 12) & ~failed
    np.savez(project / "raw-standard.npz", ids=ids, G=G, Set=set_raw, failed=failed,
             high=high, low=~high & ~failed, noise=noise, **predictions)


def plans(family):
    if family == "source":
        return [(tag, seed) for tag in ("control", *TAGS) for seed in (0, 100, 200)]
    gates = ("full",) if family == "context" else ("nb", "full")
    return [(f"{gate}_{slot}_{variant}", seed) for slot in SLOTS
            for gate in gates for variant in VARIANTS for seed in (0, 100, 200)]


def tab_member(project, data, family, member, device="cpu"):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if device == "mps":
        assert torch.backends.mps.is_available()
        torch.mps.set_per_process_memory_fraction(.3)
    exp = project / "experiments"
    sys.path.insert(0, str(exp / "push-300"))
    import harness
    name, seed = plans(family)[member]
    standard = arrays(project / "raw-standard.npz")
    ids = standard["ids"]
    labels = {r["ID"]: float(r["hardness"]) for r in csv.DictReader((data / "train.csv").open())}
    if family == "source":
        tr = arrays(exp / "push-1300/abl/feat-h9s2-full-044-u1dmavg-train.npz")
        te = arrays(exp / "push-1300/abl/feat-h9s2-full-044-u1dmavg-test.npz")
        X, Q = tr["X"].astype(float), te["X"].astype(float)
        if name != "control":
            folder = project / STUDY / "candidate-source-flow38-v1/features"
            a, b = arrays(folder / f"{name}-train.npz"), arrays(folder / f"{name}-test.npz")
            np.testing.assert_array_equal(a["ids"], tr["ids"])
            np.testing.assert_array_equal(b["ids"], te["ids"])
            X, Q = np.c_[X, a["flow"]], np.c_[Q, b["flow"]]
        active = np.ones(len(ids), bool)
    else:
        gate, slot, variant = name.split("_", 2)
        if family == "context":
            folder, suffix = SLOTS[slot]
            pattern = exp / f"{folder}/feat-{variant}H2Z{suffix}-{{split}}.npz"
        else:
            pattern = exp / f"push-1300/abl/feat-h9s2-{gate}-{slot}-{variant}-{{split}}.npz"
        tr, te = (arrays(str(pattern).format(split=s)) for s in ("train", "test"))
        X, Q = tr["X"].astype(float), te["X"].astype(float)
        active = standard["high"] if gate == "full" else standard["low"]
    np.testing.assert_array_equal(te["ids"], ids)
    y = np.array([labels[str(i)] for i in tr["ids"]])
    assert len(y) == 500
    prediction = np.full(len(ids), np.nan)
    if active.any():
        model = harness.tabpfn_new("v3.5", seed, device, n_est=8 if family == "source" else None)
        model.set_params(inference_precision=torch.float32, memory_saving_mode=True)
        model.fit(X, y)
        prediction[active] = model.predict(Q[active])
        assert np.isfinite(prediction[active]).all()
    folder = project / "raw-tab"
    folder.mkdir(exist_ok=True)
    np.savez(folder / f"{family}-{name}-{seed}.npz", ids=ids, prediction=prediction)
    print(f"{family}: {member + 1}/{len(plans(family))} {name} seed={seed}", flush=True)


def combine(project, data, verify_reference=True):
    import numpy as np
    import torch
    from torch import nn
    from PIL import Image
    import joblib
    d = arrays(project / "raw-standard.npz")
    ids, high, failed = d["ids"], d["high"], d["failed"]
    def mean(family, names):
        values = []
        for name in names:
            for seed in (0, 100, 200):
                z = arrays(project / f"raw-tab/{family}-{name}-{seed}.npz")
                np.testing.assert_array_equal(z["ids"], ids)
                values.append(z["prediction"])
        return np.mean(values, 0)
    plain = mean("context", [name for name, seed in plans("context") if seed == 0])
    tab = np.where(high,
                   mean("tab", [name for name, seed in plans("tab") if seed == 0 and name.startswith("full_")]),
                   mean("tab", [name for name, seed in plans("tab") if seed == 0 and name.startswith("nb_")]))
    exp = project / "experiments"
    r4 = exp / "rmse-research-20261015/round4-full073"
    p73 = (.16*d["G"] + .384*plain + .128*d["legacy_ridge"] +
           .128*d["legacy_logridge"] + .05*d["Set"] + .15*d["gmm_high"])
    p73[failed] = d["G"][failed]
    X = arrays(exp / "push-800/feat-u1dmavgH2Z-anghcor-test.npz")
    keys = list(X["keys"])
    raw = np.c_[(p73-.16*d["G"])/.84, plain-d["legacy_ridge"], d["noise"],
                X["X"][:, keys.index("u_ph")],
                np.log1p(np.maximum(np.nan_to_num(X["X"][:, keys.index("n_obj")], nan=0), 0)),
                X["X"][:, keys.index("coverage")]]
    tree = ast.parse((r4 / "cnn_fitter.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in ("ComplementCNN", "predict")]
    ns = dict(torch=torch, nn=nn, np=np)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(r4 / "cnn_fitter.py"), "exec"), ns)
    corrections = np.zeros((2, len(ids)))
    active = np.flatnonzero(high)
    if len(active):
        images = np.stack([np.asarray(Image.open(data / f"test/{ids[j]}.png"), np.float32)/255
                           for j in active])[:, None]
        for seed in (0, 1):
            ck = torch.load(r4 / f"final500/models/aux-seed{seed}/model.pt", map_location="cpu", weights_only=True)
            norm = ck["normalization"]
            model = ns["ComplementCNN"]()
            model.load_state_dict(ck["state_dict"])
            image = ((images-norm["image_mean"])/norm["image_std"]).astype(np.float32)
            clean = np.where(np.isfinite(raw), raw, np.asarray(norm["context_median"]))
            context = ((clean-np.asarray(norm["context_mean"]))/np.asarray(norm["context_std"])).astype(np.float32)[active]
            corrections[seed, active] = ns["predict"](model, image, context, np.arange(len(active)), "aux")*norm["target_std"]
    coefficient = read_json(r4 / "final500/recipe.json")["production_weights"]["addition"][1]
    base = (.16*d["G"] + .384*tab + .128*np.where(high, d["ridge_high"], d["ridge_low"]) +
            .128*np.where(high, d["logridge_high"], d["logridge_low"]) + .05*d["Set"] +
            .15*np.where(high, d["gmm_high"], d["gmm_low"]) + high*coefficient*corrections.mean(0))
    base[failed] = d["G"][failed]
    h = project / STUDY
    modeldir = h / "final-nonlinear-readouts-20261019-0945/models"
    physical = arrays(exp / "push-1300/abl/feat-h9s2-full-044-u1dmavg-test.npz")
    np.testing.assert_array_equal(physical["ids"], ids)
    P = physical["X"].astype(float)
    boosted = {"control": joblib.load(modeldir / "boostedreadout-control.joblib")["model"].predict(P)}
    for tag in TAGS:
        flow = arrays(h / f"candidate-source-flow38-v1/features/{tag}-test.npz")
        np.testing.assert_array_equal(flow["ids"], ids)
        boosted[tag] = joblib.load(modeldir / f"boostedreadout-{tag}.joblib")["model"].predict(np.c_[P, flow["flow"]])
    source = {tag: mean("source", [tag]) for tag in ("control", *TAGS)}
    final = base + (~failed)*(.384*(np.mean([source[tag] for tag in TAGS], 0)-source["control"]) +
                             .256*(np.mean([boosted[tag] for tag in TAGS], 0)-boosted["control"]))
    assert np.isfinite(final).all()
    np.testing.assert_array_equal(final[failed], d["G"][failed])
    if verify_reference:
        reference = {r["ID"]: float(r["hardness"]) for r in csv.DictReader(
            (FROZEN / STUDY / "final-nonlinear-readouts-20261019-0945/candidate-source-flow38-boostedreadout.csv").open())}
        error = float(np.max(abs(final-np.array([reference[str(i)] for i in ids]))))
        print("Raw #062:", len(ids), "rows; maximum difference:", error, flush=True)
        assert error < .002, "Maximum prediction difference must be less than 0.002 HV."
    else:
        print("Retrained recipe:", len(ids), "rows.", flush=True)
    np.savez(project / "raw-result.npz", ids=ids, prediction=final)
    with (project / "raw-result.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ID", "hardness"])
        writer.writerows(zip(ids, [repr(float(v)) for v in final]))


def smoke_inputs(project, data, count):
    """Limit diagnostic query rows; retain the documented frozen train context."""
    target = project / "smoke-data"
    target.mkdir()
    for name in ("train", "train.csv", "test"):
        (target / name).symlink_to(data / name)
    rows = list(csv.DictReader((data / "sample_submission.csv").open()))[:count]
    with (target / "sample_submission.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ID", "hardness"])
        writer.writeheader()
        writer.writerows(rows)
    (project / "data").unlink()
    (project / "data").symlink_to(target, target_is_directory=True)


def run(a):
    import workspace
    if a.retrained and not read_json(PACKAGE / 'retrained-model.json')['complete']:
        raise RuntimeError('Complete train.py before using --retrained')
    if a.retrained and a.smoke:
        raise ValueError('Frozen-context smoke mode applies to the original saved model only')
    envs = check(a)
    data = Path(a.data).resolve()
    if not a.smoke:
        if not a.out:
            raise ValueError("--out is required for a complete reproduction")
        output = Path(a.out).resolve()
        if output.exists():
            raise FileExistsError(output)
        if not output.parent.is_dir():
            raise FileNotFoundError(output.parent)
    with tempfile.TemporaryDirectory(prefix="raw-", dir=a.scratch_dir) as temp:
        project = Path(temp) / "project"
        workspace.init(argparse.Namespace(workspace=str(project), data=str(data),
                                             **{"python_"+k: v for k, v in envs.items()}))
        runtime = project / "runtime-cache"
        runtime.mkdir()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for key in ("TMPDIR", "MPLCONFIGDIR", "XDG_CACHE_HOME", "HF_HOME", "CELLPOSE_LOCAL_MODELS_PATH"):
            env[key] = str(runtime)
        for key in ("NOMAPS", "OUTDIR", "LA_MU", "LA_SD", "ALPHAS", "LOGALPHAS", "SHIFTFIX"):
            env.pop(key, None)
        for key in ('REPRO062_SPLITS', 'REPRO062_RETRAINED'):
            env.pop(key, None)
        def execute(kind, script, *args):
            extra = ['--retrained'] if a.retrained else []
            subprocess.run([envs[kind], "-B", str(PACKAGE / script), *map(str, args), *extra], env=env, check=True)
        if a.smoke:
            smoke_inputs(project, data, a.smoke)
            print("SMOKE ONLY: raw query rows; frozen train feature banks. No full-reproduction CSV will be saved.", flush=True)
        execute("core", "workspace.py", "physical", "--workspace", project,
                "--splits", "test" if a.smoke else "train,test")
        execute("core", "raw.py", "--action", "g018", "--project", project, "--data", project / "data")
        execute("core", "dinol.py", "features", "--project", project, "--data", project / "data",
                "--split", "test", "--device", "mps", "--out", project / "raw-dinol")
        for seed in (1811, 1823):
            for split in (("test",) if a.smoke else ("train", "test")):
                execute("cellpose", "source_features.py", "--project", project, "--data", data,
                        "--split", split, "--seed", seed, "--out", project / "raw-flow",
                        "--limit", a.smoke if split == "test" else 0)
                name = f"full500-sourceflow38-s{seed}-{split}"
                generated = project / "raw-flow" / (name + ("-partial.npz" if a.smoke else ".npz"))
                shutil.copy2(generated, project / STUDY / "candidate-source-flow38-v1/features" / f"{name}.npz")
        execute("core", "raw.py", "--action", "standard", "--project", project)
        for family in ("context", "tab", "source"):
            members = (None,) if a.tab_device == "mps" else range(len(plans(family)))
            for member in members:
                selection = () if member is None else ("--member", member)
                execute("tabpfn", "raw.py", "--action", family, *selection, "--project", project,
                        "--data", project / "data", "--tab-device", a.tab_device)
        execute("core", "raw.py", "--action", "combine", "--project", project, "--data", project / "data")
        if not a.smoke:
            result = project / "raw-result.csv"
            ids = [row["ID"] for row in csv.DictReader(result.open())]
            assert len(ids) == len(set(ids)) == 1000
            assert ids == [row["ID"] for row in csv.DictReader((data / "sample_submission.csv").open())]
            with output.open("xb") as stream:
                stream.write(result.read_bytes())
            print("CSV:", output, "SHA256:", digest(output), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data")
    p.add_argument("--out")
    p.add_argument("--check", action="store_true")
    p.add_argument("--retrained", action="store_true", help="Use the weights from a completed training workspace.")
    p.add_argument("--smoke", type=int, default=0, help="Diagnostic first N queries; frozen train context; no output CSV.")
    p.add_argument("--scratch-dir", type=Path)
    p.add_argument("--tab-device", choices=["cpu", "mps"], default="cpu",
                   help="TabPFN execution backend; CPU preserves the canonical backend, MPS uses the same float32 recipe.")
    for kind in ("core", "cellpose", "tabpfn"):
        p.add_argument("--python-"+kind)
    p.add_argument("--action", choices=["run", "g018", "standard", "context", "tab", "source", "combine"], default="run", help=argparse.SUPPRESS)
    p.add_argument("--project", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--member", type=int, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.smoke < 0 or a.smoke > 1000:
        p.error("--smoke must be between 0 and 1000")
    if a.action != "standard" and not a.data:
        p.error("--data is required")
    if a.check:
        check(a)
    elif a.action == "run":
        run(a)
    elif a.action == "g018":
        g018(a.project.resolve(), Path(a.data).resolve(), not a.retrained)
    elif a.action == "standard":
        standard(a.project.resolve())
    elif a.action == "combine":
        combine(a.project.resolve(), Path(a.data).resolve(), not a.retrained)
    else:
        import gc
        import torch
        members = range(len(plans(a.action))) if a.member is None else [a.member]
        for member in members:
            tab_member(a.project.resolve(), Path(a.data).resolve(), a.action, member, a.tab_device)
            gc.collect()
            if a.tab_device == "mps":
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()
