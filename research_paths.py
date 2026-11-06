"""Resolve historical research IDs within the organized repository.

Existing experiment records keep their original identifiers. These small path
adapters locate the corresponding source/report without editing scientific data.
The standalone final/package tree does not depend on this module.
"""
from pathlib import Path
import os
import re
import sys

ROOT = Path(__file__).resolve().parent
STUDIES = ROOT / "studies"
STUDY_GROUPS = ("baseline", "candidates", "026-049", "050-074", "075-099",
                "100-124", "125-149", "150-174")
REPORT_GROUPS = {"summaries", "validation", "submissions", "provenance",
                 "registers", "notes", "figures", "source-snapshots", "logs"}
EXPERIMENT_PATHS = {'076-complement-20261016': '04-residual-cnn/20261016-candidate-076-complement',
 '078-080-update-20261016': '04-residual-cnn/20261016-candidates-078-080',
 '080-followup-20261016': '04-residual-cnn/20261016-candidate-080-followup',
 '082-fusion-20261018-0155': '04-residual-cnn/20261018-0155-candidate-082-fusion',
 'attention-counterfactual-200': '01-representations/0200-attention-counterfactual',
 'auxiliary-cnn-193': '01-representations/0193-auxiliary-cnn',
 'auxiliary-repartition-201': '01-representations/0201-auxiliary-repartition',
 'blur-augmentation-215': '01-representations/0215-blur-augmentation',
 'boundary-connectivity-20261017-1835': '03-robustness/20261017-1835-boundary-connectivity',
 'brightness-augmentation-213': '01-representations/0213-brightness-augmentation',
 'candidate021-178': '01-representations/0178-candidate021',
 'candidate022-198': '01-representations/0198-candidate022',
 'cellpose-panel-221': '02-microstructure/0221-cellpose-panel',
 'checkpoint-averaging-207': '01-representations/0207-checkpoint-averaging',
 'cnn-nested-204': '01-representations/0204-cnn-nested',
 'cnn-optimizer-186': '01-representations/0186-cnn-optimizer',
 'cnn-resolution-188': '01-representations/0188-cnn-resolution',
 'cnn-scale-floor-187': '01-representations/0187-cnn-scale-floor',
 'cnn-seed-expand-203': '01-representations/0203-cnn-seed-expand',
 'cnn-unfreeze-189': '01-representations/0189-cnn-unfreeze',
 'contrast-augmentation-206': '01-representations/0206-contrast-augmentation',
 'convnext-llrd-210': '01-representations/0210-convnext-llrd',
 'degradation-pilot-audit-20261017-0030': '03-robustness/20261017-0030-degradation-pilot-audit',
 'dinol-177': '01-representations/0177-dinol',
 'ema-repartition-209': '01-representations/0209-ema-repartition',
 'gamma-augmentation-214': '01-representations/0214-gamma-augmentation',
 'grain-cellpose-220': '02-microstructure/0220-grain-cellpose',
 'grain-validity-218': '01-representations/0218-grain-validity',
 'highnoise-relational-20261016': '03-robustness/20261016-highnoise-relational',
 'highnoise-ssl-20261016-2312': '03-robustness/20261016-2312-highnoise-ssl',
 'huber-loss-211': '01-representations/0211-huber-loss',
 'huber-repartition-212': '01-representations/0212-huber-repartition',
 'infogain-20261017': '03-robustness/20261017-infogain',
 'infomatched-copies-20261016': '03-robustness/20261016-infomatched-copies',
 'infomatched-pilot-20261016': '03-robustness/20261016-infomatched-pilot',
 'joint-readout-180': '01-representations/0180-joint-readout',
 'matched-replacement-194': '01-representations/0194-matched-replacement',
 'micronet-frozen-205': '01-representations/0205-micronet-frozen',
 'midlayer-L-181': '01-representations/0181-midlayer-l',
 'model-audit-repair-20261017-2335': '04-residual-cnn/20261017-2335-model-audit-repair',
 'modern-backbones-190': '01-representations/0190-modern-backbones',
 'multihead-decoder-196': '01-representations/0196-multihead-decoder',
 'pca-ridge-192': '01-representations/0192-pca-ridge',
 'plateau-break-20261016-2035': '04-residual-cnn/20261016-2035-plateau-break',
 'pooling-extrema-195': '01-representations/0195-pooling-extrema',
 'pretrained-cnn-185': '01-representations/0185-pretrained-cnn',
 'probabilistic-grains-900': '02-microstructure/0900-probabilistic-grains',
 'push-1000': '03-robustness/1000-structure-diagnostics',
 'push-1100': '03-robustness/1100-noise-diagnostics',
 'push-1200': '03-robustness/1200-degradation-augmentation',
 'push-1300': '03-robustness/1300-robust-ensembles',
 'push-1400': '03-robustness/1400-compliant-rebuild',
 'push-1400v2': '03-robustness/1400-compliant-rebuild-v2',
 'push-1500': '03-robustness/1500-multitask-cnn',
 'push-300': '02-microstructure/0300-cellpose-finetuning',
 'push-400': '02-microstructure/0400-unet-maps',
 'push-500': '02-microstructure/0500-heterogeneity',
 'push-600': '02-microstructure/0600-shape-features',
 'push-700': '02-microstructure/0700-directional-features',
 'push-800': '02-microstructure/0800-shading-features',
 'push-900': '02-microstructure/0900-partner-ensembles',
 'query-normalization-197': '01-representations/0197-query-normalization',
 'reproduction-018-175': '01-representations/0175-reproduction-018',
 'reproduction-018-176': '01-representations/0176-reproduction-018',
 'reproduction-018-176-isolated': '01-representations/0176-reproduction-018-isolated',
 'rescnn-retest-20261017': '04-residual-cnn/20261017-rescnn-retest',
 'residual-cnn-219': '01-representations/0219-residual-cnn',
 'resolution-216': '01-representations/0216-resolution',
 'resolution-repartition-217': '01-representations/0217-resolution-repartition',
 'rmse-research-20261015': '04-residual-cnn/20261015-rmse-research',
 'simplification-202': '01-representations/0202-simplification',
 'spatial-L-179': '01-representations/0179-spatial-l',
 'spatial-moments-191': '01-representations/0191-spatial-moments',
 'stepchange-20261016': '04-residual-cnn/20261016-stepchange',
 'structure-context-20261017-1415': '03-robustness/20261017-1415-structure-context',
 'submission-083-20261017-231722': 'submissions/20261017-231722-candidate-083',
 'submission-083-20261018-060706': 'submissions/20261018-060706-candidate-083',
 'submission022-diagnostic-199': '01-representations/0199-submission022-diagnostic',
 'supervised-renewal-20261018-0625': '05-source-flow/20261018-0625-supervised-renewal',
 'tabicl-v2': 'frameworks/tabicl-v2',
 'tabpfn-v2': 'frameworks/tabpfn-v2',
 'tabpfn-v3': 'frameworks/tabpfn-v3',
 'tabular-fm': 'frameworks/tabular-fm',
 'tessellation-diagnostic-183': '01-representations/0183-tessellation-diagnostic',
 'update-ema-208': '01-representations/0208-update-ema',
 'wph-182': '01-representations/0182-wph'}

def study_group(name):
    if re.search(r"candidate\d{3}", name):
        return "candidates"
    numbers = re.findall(r"(?<!\d)(\d{3})(?!\d)", Path(name).stem)
    if not numbers:
        return "baseline"
    n = int(numbers[-1])
    for low, high in [(26, 49), (50, 74), (75, 99), (100, 124), (125, 149), (150, 174)]:
        if low <= n <= high:
            return f"{low:03d}-{high:03d}"
    return "baseline"

def report_group(name):
    name = Path(name).name
    lower = name.lower()
    suffix = Path(name).suffix
    if suffix == ".csv":
        return "registers"
    if suffix in {".html", ".png", ".svg"}:
        return "figures"
    if suffix == ".py":
        return "source-snapshots"
    if suffix == ".md":
        return "notes"
    if suffix in {".log", ".txt"}:
        if re.search(r"candidate\d{3}|public-submission", lower):
            return "logs/submissions"
        numbers = re.findall(r"(?<!\d)(\d{3})(?!\d)", Path(name).stem)
        if numbers:
            n = int(numbers[-1])
            for low, high in [(0, 49), (50, 99), (100, 149), (150, 174), (175, 221)]:
                if low <= n <= high:
                    return f"logs/{low:03d}-{high:03d}"
        return "logs/general"
    if re.search(r"candidate\d{3}|public-submission", lower):
        return "submissions"
    if re.search(r"preserv|protect|provenance|sources?|licenses?|environment|plan|design|preflight|readme", lower):
        return "provenance"
    if re.search(r"validat|verif|audit|check|reproduc|replay", lower):
        return "validation"
    return "summaries"


def study_sources(root=STUDIES):
    """The original flat study modules, now grouped without changing module names."""
    return sorted(p for group in STUDY_GROUPS for p in (root / group).glob("*.py"))


_SOURCE_NAMES = {p.name for p in study_sources()} | {p.name for p in STUDIES.glob("*.py")}
_EXPERIMENT_ORIGINS = {new: old for old, new in EXPERIMENT_PATHS.items()}
_EXPERIMENT_GROUPS = {new.split("/")[0] for new in EXPERIMENT_PATHS.values()}


def resolve(base, *parts):
    """Join a research path and translate only known old source/report locations."""
    base = Path(os.path.abspath(base))
    physical = Path(os.path.abspath(base.joinpath(*parts)))
    if physical.is_relative_to(ROOT):
        current = physical.relative_to(ROOT).parts
        # Explicit paths in the new layout also arrive as CLI arguments. They
        # must keep their physical meaning, including outputs not yet created.
        if (
            len(current) >= 3
            and current[0] == "experiments"
            and "/".join(current[1:3]) in _EXPERIMENT_ORIGINS
            or len(current) >= 2
            and (
                current[0] == "studies" and current[1] in STUDY_GROUPS
                or current[0] == "reports" and current[1] in REPORT_GROUPS
            )
            or current[:1] in (("data",), ("artifacts",), (".venv",))
        ):
            return physical
    # The legacy scripts may use ../ paths from their newly grouped directory.
    # Interpret those paths in the script's original logical location first.
    if base.is_relative_to(ROOT / "experiments"):
        current = base.relative_to(ROOT / "experiments").parts
        if len(current) == 1 and current[0] in _EXPERIMENT_GROUPS:
            base = ROOT / "experiments"
        elif len(current) >= 2:
            old = _EXPERIMENT_ORIGINS.get("/".join(current[:2]))
            if old:
                base = ROOT / "experiments" / old / Path(*current[2:])
    if base.is_relative_to(STUDIES):
        current = base.relative_to(STUDIES).parts
        if current and current[0] in STUDY_GROUPS:
            base = STUDIES / Path(*current[1:])
    target = Path(os.path.abspath(base.joinpath(*parts)))
    try:
        rel = target.relative_to(ROOT)
    except ValueError:
        return target
    segments = rel.parts
    if not segments:
        return target
    if segments[0] == "experiments" and len(segments) > 1:
        replacement = EXPERIMENT_PATHS.get(segments[1])
        if replacement:
            return ROOT / "experiments" / replacement / Path(*segments[2:])
    if segments[:2] == ("studies", ".venv"):
        return ROOT / ".venv" / Path(*segments[2:])
    if segments[0] == "studies" and len(segments) == 2 and segments[1] in _SOURCE_NAMES:
        return STUDIES / study_group(segments[1]) / segments[1]
    if len(segments) == 1 and segments[0] in _SOURCE_NAMES:
        return STUDIES / study_group(segments[0]) / segments[0]
    offset = 1 if segments[:1] == ("reports",) else 2 if segments[:2] == ("studies", "reports") else None
    if offset is not None and len(segments) > offset and segments[offset] not in REPORT_GROUPS:
        name = segments[offset]
        group = "validation" if name == "resume-readiness-20261007" else report_group(name)
        prefix = ROOT / Path(*segments[:offset])
        return prefix / group / Path(*segments[offset:])
    return target


def io_path(path):
    """Resolve a repository-relative filename passed directly to an I/O function."""
    if isinstance(path, str):
        return str(resolve(Path.cwd(), path))
    return resolve(Path.cwd(), path) if isinstance(path, os.PathLike) else path


# Keep historical bare module imports and serialized estimator module names valid.
for directory in (STUDIES, *(STUDIES / group for group in STUDY_GROUPS), STUDIES / "src"):
    if directory.is_dir() and str(directory) not in sys.path:
        sys.path.append(str(directory))
