from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,hashlib,datetime
P=Path.cwd();R=_repo_path(P, 'experiments/ema-repartition-209');A=_repo_path(P, 'experiments/update-ema-208');B=_repo_path(P, 'experiments/auxiliary-repartition-201')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,z):
 with p.open('x') as f:json.dump(z,f,ensure_ascii=False,indent=2);f.write('\n')
for x in ['tmp','cache']: (_repo_path(R, x)).mkdir()
a=read(_repo_path(A, 'plan.json'));b=read(_repo_path(B, 'plan.json'));pres=read(_repo_path(A, 'preservation.json'));pres.update(read(_repo_path(A, 'artifact-manifest.json')))
for p in [_repo_path(A, 'completion.json'),_repo_path(A, 'artifact-manifest.json'),_repo_path(B, 'plan.json'),_repo_path(B, 'common.py'),_repo_path(B, 'engine.py'),_repo_path(B, 'baseline-validation.json')]:pres[str(p.relative_to(P))]=sha(p)
inputs=dict(b['inputs_SHA256'])
for split in ['20261031','20261101']:
 for p in (_repo_path(B, split)).glob('fold-*-detach/*'):
  if p.is_file():inputs[str(p.relative_to(P))]=sha(p)
 for n in ['frozen-components.npz','detach-OOF.npz']:inputs[str((_repo_path(B, split, n)).relative_to(P))]=sha(_repo_path(B, split, n))
for n,h in {**pres,**inputs}.items():assert sha(_repo_path(P, n))==h,n
write(_repo_path(R, 'preservation.json'),pres)
plan={'study':209,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'학습 전 정책 고정','authorization':'사용자가 208의 고정 EMA를 두 기존 재분할에서 5fold씩 총 10회 학습하여 검증하도록 명시했다. 전체 nested 재학습과 채택은 이번 범위가 아니다.','IDs':a['IDs'],'splits':['20261031','20261101'],'folds':b['folds'],'image_cache':a['image_cache'],'recipe':dict(a['recipe'],training_trajectories=10),'EMA':a['EMA'],'calibrators':b['calibrators'],'source208_plan_SHA256':sha(_repo_path(A, 'plan.json')),'source201_plan_SHA256':sha(_repo_path(B, 'plan.json')),'tolerances':a['tolerances'],'scope':a['scope'],'resources':dict(a['resources'],training_total_seconds_limit=2400),'inputs_SHA256':inputs,'control':'기존 021 checkpoint 및 201 detach main state와 마지막 5 epoch 평균, legacy FP32 OOF exact. 두 분할 전체 사전 확인 후 학습한다.','training_invariance':{'historical':'각 800 update의 순서·D4·LR·MSE·main gradient norm, 초기 state, epoch 76–80 main state, 평균 및 OOF exact. 201 최종 main optimizer·scheduler도 exact.','RNG_limit':'201은 update별 RNG를 저장하지 않았다. 시작·종료 RNG만 과거와 직접 비교한다. 209는 전체 update RNG를 기록하고 EMA 전후 불변을 검증한다.','optimizer_reference':'208과 같은 별도 AdamW 파라미터 복제본에 같은 clipped gradient를 전달. 추가 forward/backward 없이 매 update state exact 및 EMA 부작용 부재 검증.','normalization':'기존 train 320 이미지·타깃 전용. validation y NaN으로 통계 독립성 검사.'},'fixed_ensemble':{'formula':'Q = P021 + 0.081 * 기존 split/fold calibrator slope * (새 첫 seed CNN - 기존 첫 seed CNN)','meaning':'기존 CNN 3seed 중 첫 seed 출력 치환이다. 새 CNN을 추가한 blend가 아니다. 나머지 두 seed와 calibration·잔차는 고정하며 전체 nested 재학습이 아니다.','null':'Q_uniform = P021 exact, 기존 121/133/140/144/177 계보 재구성 확인'},'statistics':{'draws':10000,'seed':202610209,'unit':'동일한 400개 ID cluster. 세 분할 결과를 항상 함께 재표집. 1200개 독립 관측으로 취급하지 않는다.','aggregate':'ID별 세 분할 squared error 평균 후 전체 ID 평균의 제곱근. 예측 평균 앙상블이 아니다.','reports':['208 원본과 두 재분할 각각의 RMSE·CI·전체 fold·상위 5개 이득 ID 제외','두 재분할 pooled 및 세 분할 pooled의 ID cluster CI·상위 5개 ID 제외','208 원본에서 제외한 다섯 ID를 동일하게 제외한 각 비교의 감도'],'original_CI':'208의 사전 seed 202610208 CI는 보존하고 209 공통 draw CI를 별도 구분한다.'},'stability_gate':{'each_new_split':'208 동일 기준: CNN 개선 >= 0.02, 3/5 fold 이상 개선, CI 하한 > 0, 상위 5개 이득 제외 개선 > 0','aggregate':'두 재분할 pooled 및 세 분할 pooled 모두 개선 > 0, ID cluster CI 하한 > 0, 상위 5개 ID 제외 개선 > 0','technical':'모든 사전·재현 검증 통과','action':'두 분할 모두 실행·검증한 후 비교한다. 보고 후 정지하며 후단 전체 재학습은 별도 판단한다.'}}
assert plan['EMA']==a['EMA'];write(_repo_path(R, 'plan.json'),plan)
s=(_repo_path(A, 'common.py')).read_text().replace('208','209');start=s.index('def data():');end=s.index('def norm(',start)
s=s[:start]+'''SPLITS=plan['splits']
def data(split):
 assert split in SPLITS;ids,x=images();d=load(P/f'experiments/dinol-177/{split}/predictions.npz');z=load(P/'artifacts/run-012/development_features.npz',['ids','y','groups']);old=load(P/f'artifacts/cnn-seed-mean-133/audit/{split}.npz')
 for key in ['ids','y']:np.testing.assert_array_equal(d[key],z[key]);np.testing.assert_array_equal(d[key],old[key])
 np.testing.assert_array_equal(d['g'],z['groups']);np.testing.assert_array_equal(ids,d['ids']);np.testing.assert_array_equal(d['fold'],old['fold']);assert len(set(ids))==400 and len(set(d['g']))==400;d['images']=x;d['split']=split;d['best021']=d['blend_L_MLP'];d['old_CNN']=old['seeds'][0];d['old_CNN_seeds']=old['seeds']
 np.testing.assert_array_equal(d['best021'],.9*d['reference018']+.1*d['L_MLP'])
 for k in range(5):
  T,V=indices(d,k);assert len(T)==320 and len(V)==80 and not set(d['g'][T])&set(d['g'][V]);assert T.tolist()==plan['folds'][split][str(k)]['train'] and V.tolist()==plan['folds'][split][str(k)]['validation']
 return d
def indices(d,k):return np.flatnonzero(d['fold']!=k),np.flatnonzero(d['fold']==k)
def old_checkpoint(split,k):return P/f'artifacts/frozen-validation-021/seed-{split}/fold-{k}/cnn-seed20261003.pt'
def base_state(sd):return {n:v for n,v in sd.items() if not n.startswith('aux.')}
'''+s[end:]
src=(_repo_path(B, 'common.py')).read_text();lineage=src[src.index('def lineage(d):'):src.index('def swap():')];s+='\n'+lineage
(_repo_path(R, 'common.py')).write_text(s)
s=(_repo_path(A, 'engine.py')).read_text();s=s.replace("folder=R/f'fold-{k}'", "folder=R/d['split']/f'fold-{k}'").replace('c.indices(k)','c.indices(d,k)').replace("oldfolder=P/f'experiments/contrast-augmentation-206/fold-{k}-control'", "oldfolder=P/f\"experiments/auxiliary-repartition-201/{d['split']}/fold-{k}-detach\"").replace("oldfolder/'fit.json'","oldfolder/'report.json'")
s=s.replace("oldfit['initial_state_SHA256']","oldfit['initial_base_SHA256']").replace("oldfit['rng_start']","{n.upper():v for n,v in oldfit['rng_start'].items()}").replace("oldfit['rng_end']","{n.upper():v for n,v in oldfit['rng_end'].items()}")
s=s.replace(";assert state['CPU']==oldstreams['rng_CPU'][n] and state.get('MPS','CPU only')==oldstreams['rng_MPS'][n]",'')
s=s.replace("oldstreams['values'][n]", "oldstreams['values'][n,[0,2]]")
s=s.replace("assert eph[str(epoch+1)]==oldfit['epoch_state_SHA256'][str(epoch+1)]", "historic=torch.load(oldfolder/f'epoch-{epoch+1}.pt',map_location='cpu',weights_only=True);assert c.statehash(historic)==oldfit['epoch_state_hashes'][str(epoch+1)];assert eph[str(epoch+1)]==c.statehash(c.base_state(historic))")
s=s.replace(" torch.save({'optimizer':cpu_tree(o.state_dict())", " oldopt=torch.load(oldfolder/'optimizer-final.pt',map_location='cpu',weights_only=True);assert_tree(cpu_tree(o.state_dict()),oldopt['main']);assert sch.state_dict()==oldopt['schedulers'][0]\n torch.save({'optimizer':cpu_tree(o.state_dict())")
s=s.replace("ref=c.load(P/'experiments/contrast-augmentation-206/frozen-reference.npz');np.testing.assert_array_equal(predictions['last5_uniform'],ref['old_first_CNN'][V])", "np.testing.assert_array_equal(predictions['last5_uniform'],d['old_CNN'][V])")
s=s.replace("'initial_state_exact206'","'initial_state_exact201'").replace("'sample_D4_LR_RNG_loss_gradient_norm_exact_all800'","'sample_D4_LR_MSE_main_gradient_norm_exact_all800'").replace("'last5_epoch_states_and_average_exact206'","'last5_epoch_states_and_average_exact201'").replace("'source206_control_fit_SHA256'","'source201_control_fit_SHA256'")
s=s.replace("rec={'fold':k", "rec={'split':d['split'],'historical_RNG_start_end_exact':True,'historical_per_update_RNG_available':False,'historical_final_main_optimizer_scheduler_exact':True,'fold':k")
s=s.replace("'fit_start':k", "'split':d['split'],'fit_start':k").replace("'fold_complete':k", "'split':d['split'],'fold_complete':k")
s=s.replace('def unpack(vector,named):','''def assert_tree(a,b):
 if torch.is_tensor(a):assert torch.equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:assert_tree(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):assert_tree(x,y)
 else:assert a==b

def unpack(vector,named):''')
(_repo_path(R, 'engine.py')).write_text(s)
print('Created fixed plan and shared engine. Protected:',len(pres),'input hashes:',len(inputs))
