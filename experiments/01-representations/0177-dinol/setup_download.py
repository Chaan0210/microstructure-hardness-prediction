from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,hashlib,time,urllib.request,shutil,datetime
import numpy as np
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3];V=_repo_path(P, 'vendor/dinov2-official/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
 with np.load(_repo_path(P, 'artifacts/run-012/development_features.npz'),allow_pickle=False) as z:ids=z['ids'].tolist();groups=z['groups'].tolist()
 with np.load(_repo_path(P, 'artifacts/bootstrap-selector-144/original/predictions.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==ids;fold=z['fold'];smoke=[i for i,k in zip(ids,fold) if k!=0][:3]
 old=json.loads((_repo_path(P, 'experiments/reproduction-018-176/project-protection.json')).read_text());protected={**old,**{f:sha(_repo_path(P, f)) for f in ['deliverables/reproduction-018-v176.zip','reports/retraining-176-summary.json']}}
 for f,h in protected.items():assert sha(_repo_path(P, f))==h
 inputs=['artifacts/run-012/development_features.npz','artifacts/dinob-062/development_features.npz','experiments/reproduction-018-176/mlp_head.py']
 for s in ['original','20261031','20261101']:inputs += [f'artifacts/bootstrap-selector-144/{s}/predictions.npz',f'artifacts/bootstrap-selector-144/validation-{s}.json',f'artifacts/dinob-complement-064/{s}/development_predictions.npz']
 plan={'study':177,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':'User explicitly authorized officialL1.22GB and comparison,4-6GBmemory,localM1Pro16GB,existingvenv,cost0. NoG/newpackage/full500/test/holdout.','IDs':ids,'groups':groups,'smoke_IDs':smoke,'backbone':'originalnonregister DINOv2ViT-L14,frozenFP32,batch1,504bicubicImageNetnorm,noaugmentation','layers':{'B':[5,8,11],'L':[11,17,23]},'MLP':'CLS/patchmean/populationstd at relative1/2,3/4,finaldepth; B6912,L9216. Samehidden64/dropout.25/100epochs3seeds20261128,20261205,20261219+fold;aux0. Recipecontrol,notparametercountcontrol.','SVR':{'input':'finalnormalizedCLS only;B768,L1024. Paper-inspired PCA/SVR,notliteralpaperreplication.','scaler':'StandardScaler fitinnertrain/outertrain only','PCA_components':[16,34,64],'PCA_whiten':False,'PCA_solver':'full','kernel':'rbf','gamma':'scale','C':[1.,10.,100.],'epsilon':.1,'target':'mean/stdfittrainonly','selection':'innerGroupKFold3shuffle random_state202710177+10000*splitindex+fold;minpooledRMSE,tiefirstgridorder;outerheldoutneverusedforfit/selection'},'mixture':{'reference018':.9,'new_head':.1,'B_matched_control_same_weights':True,'no_weight_search':True},'original_screen':'For each of two Lreadouts: fixedblend pooledOOF RMSE strictly below018 and below matchedBblend;validatedonly. Eligible arms max2 advance to BOTH existingresplits unchanged. No arbitrarybootstrapSD/foldmajority threshold; foldgains/pairedCI diagnostic only.','confirmation':'Require samefixedLarm blend to improve018 and itsmatchedBblend in ALL3partitions. Ifbothconfirm, lowestpooledMSE across3 wins; acknowledge adaptive400 reuse. No mixtureweight, layer,seed,epoch search.','nested':'Existing018 OOFalreadyfullynested;newSVRhyperparameters/scaler/PCA fitnestedinsideoutertrain;fixed.9/.1mixhas no learnedcombiner, so no additionalcombiner fit. Anylater learnedresidualfusion requires separate honestnested study,notallowedhere.','resources':{'threads':2,'max_RSS_bytes':6000000000,'max_MPS_driver_bytes':6000000000,'max_conservative_current_RSS_plus_MPS_bytes':6000000000,'swap_growth_stop_bytes':67108864,'minimum_memory_free_percent':15,'max_feature400_seconds':600,'max_benchmark_seconds':180},'tolerance':{'CPU_MPS_feature_maxabs':.002,'same_device_repeat_abs':0.,'B_control_prediction_abs':1e-8},'protected':protected,'inputs':{f:sha(_repo_path(P, f)) for f in inputs},'official_code':{str(f.relative_to(V)):sha(f) for f in V.rglob('*.py')},'source_commit':'7764ea0f912e53c92e82eb78a2a1631e92725fc8','no_independent_generalization_claim':True};write(_repo_path(R, 'plan.json'),plan)
 shutil.copyfile(_repo_path(P, 'experiments/reproduction-018-176/mlp_head.py'),_repo_path(R, 'mlp_head.py'));(_repo_path(R, 'licenses')).mkdir();shutil.copyfile(_repo_path(V, 'LICENSE'),_repo_path(R, 'licenses/DINOv2-LICENSE'));shutil.copyfile(_repo_path(V, 'MODEL_CARD.md'),_repo_path(R, 'licenses/DINOv2-MODEL_CARD.md'))
 url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth';expected=1217586395;tmp=_repo_path(R, 'dinov2_vitl14_pretrain.pth.part');start=time.perf_counter();h=hashlib.sha256();size=0
 print(json.dumps({'stage':'DOWNLOAD_START','URL':url,'expected_bytes':expected,'plan_sha256':sha(_repo_path(R, 'plan.json'))}),flush=True)
 with urllib.request.urlopen(url,timeout=60) as response,tmp.open('xb') as f:
  assert response.status==200;assert int(response.headers['Content-Length'])==expected
  while True:
   b=response.read(4*1024*1024)
   if not b:break
   f.write(b);h.update(b);size+=len(b)
  assert size==expected
 tmp.rename(_repo_path(R, 'dinov2_vitl14_pretrain.pth'));assert sha(_repo_path(R, 'dinov2_vitl14_pretrain.pth'))==h.hexdigest()
 provenance={'model':'original DINOv2 ViT-L/14 nonregister','official_URL':url,'HTTP_status':200,'bytes':size,'sha256':h.hexdigest(),'seconds':time.perf_counter()-start,'license':'Apache-2.0','license_sha256':sha(_repo_path(R, 'licenses/DINOv2-LICENSE')),'model_card_sha256':sha(_repo_path(R, 'licenses/DINOv2-MODEL_CARD.md')),'source_commit':plan['source_commit'],'publisher':'Meta AI','local_integrity_hash_not_publisher_signature':True,'other_weight_downloads':0,'new_packages':0,'cost':0};write(_repo_path(R, 'provenance.json'),provenance);print(json.dumps(provenance),flush=True)
if __name__=='__main__':main()
