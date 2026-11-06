from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,hashlib,datetime
P=Path.cwd();R=_repo_path(P, 'experiments/convnext-llrd-210');C=_repo_path(P, 'experiments/modern-backbones-190')
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def write(p,z):
 with p.open('x') as f:json.dump(z,f,ensure_ascii=False,indent=2);f.write('\n')
v=read(_repo_path(P, 'docs/records/roadmap-audit/llrd-overfit-verification.json'));pres={**v['source_SHA256'],**v['209_preservation']}
for n in ['docs/records/roadmap-audit/next-experiment-210.md','docs/records/roadmap-audit/llrd-overfit-audit.md','docs/records/roadmap-audit/llrd-overfit-verification.json','docs/records/roadmap-audit/209-status-after-review.md','reports/ema-repartition-209-status-after-review.json','experiments/pretrained-cnn-185/train.py','artifacts/run-012/development_features.npz','experiments/dinol-177/original/predictions.npz']:
 pres[n]=sha(_repo_path(P, n))
for p in (_repo_path(C, 'convnext_tiny')).rglob('*'):
 if p.is_file():pres[str(p.relative_to(P))]=sha(p)
for p in C.glob('*.py'):pres[str(p.relative_to(P))]=sha(p)
for n in ['plan.json','provenance.json','code-lock.json','training-code-lock.json']:pres[str((_repo_path(C, n)).relative_to(P))]=sha(_repo_path(C, n))
for n,h in pres.items():assert sha(_repo_path(P, n))==h,n
write(_repo_path(R, 'preservation.json'),pres)
p=read(_repo_path(C, 'plan.json'));plan={'study':210,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':'사용자가 고정된 210 계획 실행을 명시했다. 대조 fold 0 한 번과 후보 원본 5fold만 실행한다.','source_plan':'docs/records/roadmap-audit/next-experiment-210.md','source_plan_SHA256':sha(_repo_path(P, 'docs/records/roadmap-audit/next-experiment-210.md')),'IDs':p['IDs'],'folds':p['folds'],'models':['convnext_tiny'],'stability':p['stability'],'ratio':.8,'multipliers':v['proposed_normalized_block_multipliers'],'initial_LRs':[1e-4,*v['proposed_block_initial_LR'],1e-5],'decay_reference_LRs':[1e-4,1e-5,1e-5,1e-5,1e-5],'optimizer_weight_decay':0.,'explicit_decay_coefficient':.01,'group_names':['head','block0','block1','block2','terminal_norm'],'tail_group_counts':[4763136,4763136,4763136,1536],'tail_parameters':14290944,'recipe':p['adaptation'],'head':p['head'],'input_policy':p['input_policy'],'scaler_policy':p['scaler_policy'],'counts':{'control_fits':1,'candidate_fits':5,'total_updates':1440,'head_refits':0,'inner_fits':0},'gate':{'primary':'uniform standalone RMSE - LLRD standalone RMSE','gain_min':.02,'positive_folds_min':3,'CI_lower_gt':0.,'top5_removed_gt':0.,'bootstrap_draws':10000,'bootstrap_seed':202610210,'ensemble':'0.9 P021 + 0.1 LLRD。Best021 대비 동일 기준을 별도 적용한다.','action':'기준 미달이면 이 고정 정책을 종료하고 비율 탐색을 하지 않는다. 통과해도 보고 후 정지한다.'},'resources':{'CPU_threads':2,'RSS_plus_MPS_limit':6000000000,'swap_growth':0,'fit_seconds':120,'training_seconds':900,'preflight_seconds':300,'validation_seconds':300},'control_requirements':'기존 5개 OOF 재로드 exact. Fold 0 ratio 1 그룹 분할 재학습의 head·tail state, OOF, gradient/LR/feature trace, 순서 exact.','RNG_validation':'190은 update별 RNG를 저장하지 않았다. 신규 control fold 0에서 exact 재현한 경로의 RNG를 기록한다. 각 fold에서 동일 seed 및 8x64 head dropout 2회/update의 독립 RNG 소비 재현과 모든 update RNG 비교.','scope':{'install':False,'download':False,'raw_train_holdout_test':False,'new_CSV_submission':False,'cost':0,'EMA':False,'new_layers':False,'augmentation':False,'new_loss':False}}
write(_repo_path(R, 'plan.json'),plan)
for n in ['tmp','cache']:(_repo_path(R, n)).mkdir()
s=(_repo_path(P, 'experiments/update-ema-208/common.py')).read_text().replace('208','210');s=s.replace("assert Path(sys.prefix).resolve()==P/'.venv'", "B=P/'experiments/pretrained-cnn-185';C=P/'experiments/modern-backbones-190'\nassert Path(sys.prefix).resolve()==B/'.venv'")
s=s.replace('import numpy as np,torch','import numpy as np,torch,torchvision\nfrom torch import nn').replace('threadpool_limits(2);torch.set_num_threads(2)','threadpool_limits(2);torch.set_num_threads(2);torch.set_num_interop_threads(1)')
s=s.replace("sys.path.insert(0,str(P));import cnn_controls as legacy;sys.path.remove(str(P))", "assert torch.__version__.split('+')[0]=='2.14.0' and torchvision.__version__.split('+')[0]=='0.29.0'")
s=s.replace(" for n,h in plan['inputs_SHA256'].items():assert sha(P/n)==h,n\n",'')
start=s.index('def images():');end=s.index('def rnghash():',start)
s=s[:start]+'''def data():
 d=load(P/'experiments/dinol-177/original/predictions.npz');z=load(P/'artifacts/run-012/development_features.npz',['ids','y','groups'])
 for key in ['ids','y']:np.testing.assert_array_equal(d[key],z[key])
 np.testing.assert_array_equal(d['g'],z['groups']);assert d['ids'].tolist()==plan['IDs'] and len(set(d['ids']))==len(set(d['g']))==400
 for k in range(5):
  T,V=indices(k);assert len(T)==320 and len(V)==80 and not set(d['g'][T])&set(d['g'][V]);np.testing.assert_array_equal(V,np.flatnonzero(d['fold']==k))
 return d
def indices(k):return tuple(np.array(plan['folds'][str(k)][v],dtype=int) for v in ['train','validation'])
def inputs():
 z=load(C/'convnext_tiny/features.npz');assert z['ids'].tolist()==plan['IDs'];X=z['X'].astype(float);cache=np.load(C/'convnext_tiny/prefix.npy',mmap_mode='r');assert X.shape==(400,768) and cache.shape==(400,768,8,8) and cache.dtype==np.float32;return X,cache,data()
'''+s[end:]
# Preserve the historical module state-hash iteration order, unlike208 sorted state dict helper.
a=s.index('def statehash(');b=s.index('plan=read(',a)
s=s[:a]+'''def statehash(m):
 h=hashlib.sha256()
 for n,v in m.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
'''+s[b:]
src=(_repo_path(C, 'common.py')).read_text();model=src[src.index('def model(name):'):src.index('def swap():')].replace("R/'weights'", "C/'weights'");s+='\n'+model
(_repo_path(R, 'common.py')).write_text(s);(_repo_path(R, 'stability.py')).write_text((_repo_path(C, 'stability.py')).read_text().replace('190stability','210stability'))
print('210 policy locked; protected files',len(pres))
