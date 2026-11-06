from pathlib import Path
import shutil,json,hashlib
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent;S=R/'experiments/reproduction-018-175';P=R/'deliverables/reproduction-018-v155'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
prov={}
for name in ['base_train.py','cnn_model.py','compact_regression.py','mlp_head.py','feature_helpers.py','candidate_features.py','grain_features.py','latent_features.py','denoised_grain_features.py','residual_linear.py','residual_models.py','train.py','validate.py']:
 shutil.copyfile(S/name,D/name);prov[name]={'source':str((S/name).relative_to(R)),'original_sha256':sha(S/name)}
for name in ['vendor','licenses']:shutil.copytree(S/name,D/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copytree(P/'weights',D/'weights');(D/'config').mkdir();schema=json.loads((P/'config/schema.json').read_text());schema.pop('baseline_names');(D/'config/schema.json').write_text(json.dumps(schema,indent=2))
old=json.loads((S/'run/feature-plan.json').read_text());(D/'config/development-ids.json').write_text(json.dumps(old['ids'],indent=2));(D/'config/source-origin.json').write_text(json.dumps(prov,indent=2));shutil.copyfile(P/'requirements-current.lock',D/'requirements-current.lock')
protected=json.loads((R/'artifacts/krr-scope-174/plan.json').read_text())['protected'];protected['reports/retraining-175-summary.json']=sha(R/'reports/retraining-175-summary.json');protected['experiments/reproduction-018-175/run/plan.json']=sha(S/'run/plan.json')
(D/'project-protection.json').write_text(json.dumps(protected,indent=2));(D/'run').mkdir()
print('176_SOURCES_WEIGHTS_CONFIG_READY')
