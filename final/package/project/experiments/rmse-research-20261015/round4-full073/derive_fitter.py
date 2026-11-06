"""Derive the existing round3 fitter without editing its preserved source."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
old=HERE.parent/'round3-cnn/complement_cnn.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected='974609cba768294ca969fa627978d839b604c8e61b658c8a19f65246b910cb56'
assert sha(old)==expected
source=old.read_text().split('\ndef probe():')[0]
assert source.count('assert len(target)==len(tr)==400')==1
source=source.replace('assert len(target)==len(tr)==400','assert len(target)==len(tr) and len(tr)>0')
assert source.count('shrinkage=SHRINK')==1
source=source.replace('shrinkage=SHRINK','shrinkage=None')
source='''# Derived by derive_fitter.py from the immutable round3 source.
# Only training-size assertion and non-operative checkpoint shrinkage metadata change.
# The experiment runner, not this fitter, selects ensemble weights internally.
'''+source
new=HERE/'cnn_fitter.py'
if new.exists():assert new.read_text()==source
else:new.write_text(source)
(HERE/'fitter-derivation.json').write_text(json.dumps(dict(source=str(old),source_sha256=sha(old),
    derived=str(new),derived_sha256=sha(new),architecture_optimizer_augmentation_unchanged=True),indent=2)+'\n')
