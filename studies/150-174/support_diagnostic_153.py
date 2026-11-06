"""Current018 residual complementarity and train-size dependent spatial clipping; no candidate fit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,datetime
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import bootstrap_selector_144 as s
import numpy as np,joblib
s.torch.set_num_threads(2);s.check();e=s.e;a=s.a
records=[];parts={};summary={}
def support(m,X):
 b=m['bound'];Y=X*b.scale_+b.min_;C=np.clip(Y,0,1);lost=(Y-C)/m['scale'].scale_;result={}
 for name,sl in [('physics',slice(0,57)),('spatial',slice(57,None))]:
  mask=(Y[:,sl]<-1e-12)|(Y[:,sl]>1+1e-12);result[name]={'coordinate_fraction':float(mask.mean()),'rows_any':float(mask.any(1).mean()),'lost_standardized_RMS':float(np.sqrt(np.mean(lost[:,sl]**2))),'largest_lost_standardized':float(abs(lost[:,sl]).max())}
 return result
for split in e.SPLITS:
 d=s.data(split);rr=e.read(s.OUT/split/'report.json');z=a.readnp(s.OUT/split/'predictions.npz',rr['predictions_sha256']);res=d['y']-z['primary'];P=np.empty((400,4));cliprows=np.zeros(400);lostrows=np.zeros(400)
 for k in range(5):
  dd,_=s.x.replace(d,split,k,s.x.OUT/split/f'cnn-fold-{k}',False);V=np.flatnonzero(d['fid']==k);P[V]=dd['P'][V,1:]
  for j in range(3):
   path=s.OUT/split/f'fold-{k}-block-{j}';br=e.read(path/'report.json');file=path/'ridge1.joblib';assert e.sha(file)==br['models_sha256'][file.name];obj=joblib.load(_repo_io(file));m=obj['model'];B=obj['train_indices'];r=support(m,d['X'][V]);records.append({'split':split,'outer':k,'block':j,'stage':'full_honest_B','n_train':len(B),**r});b=m['bound'];zz=d['X'][V]*b.scale_+b.min_;mask=(zz[:,57:]<-1e-12)|(zz[:,57:]>1+1e-12);cliprows[V]+=mask.mean(1)/3;lostrows[V]+=np.sqrt(np.mean(((zz[:,57:]-np.clip(zz[:,57:],0,1))/m['scale'].scale_[57:])**2,1))/3
   src=s.q.OUT/split/f'adjacent-fold-{k}/block-{j}';r0=e.read(src/'report.json')
   for t in range(3):
    f=src/f'inner-{t}-ridge1.joblib';assert e.sha(f)==r0['models_sha256'][f.name];obj=joblib.load(_repo_io(f));I=obj['train_indices'];H=obj['validation_indices'];records.append({'split':split,'outer':k,'block':j,'stage':'inner_B_train','n_train':len(I),**support(obj['model'],d['X'][H])})
 errors=P-d['y'][:,None];dis=P-P@np.array([.3,.3,.2,.2])[:,None] if False else P-(P@np.array([.3,.3,.2,.2]))[:,None]
 parts[split]={'remaining018_RMSE':e.rmse(d['y'],z['primary']),'residual_mean':float(res.mean()),'component_order':['KRR','CNNmean3','Smean3','Bmean3'],'component_error_correlation':np.corrcoef(errors.T).tolist(),'remaining_residual_dot_component_disagreement':np.mean(res[:,None]*dis,0).tolist(),'spatial_clip_fraction_mean':float(cliprows.mean()),'spatial_lost_standardized_RMS_per_row_mean':float(lostrows.mean()),'squared_error_correlation_with_lost_amplitude':float(np.corrcoef(res**2,lostrows)[0,1])}
for stage in ['inner_B_train','full_honest_B']:
 rs=[r for r in records if r['stage']==stage];summary[stage]={'nodes':len(rs),'train_n_range':[min(r['n_train'] for r in rs),max(r['n_train'] for r in rs)],**{name:{metric:float(np.mean([r[name][metric] for r in rs])) for metric in rs[0][name]} for name in ['physics','spatial']}}
report={'study':'153 residual/model and feature-support audit','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Existing400 development only. No new candidate predictions, fit, threshold, blending, holdout or test.','partition_diagnostics':parts,'feature_support_summary':summary,'records':records,'mechanism':'MinMax(clip=True) followed by StandardScaler equals train-standardization with each coordinate saturated at training extrema. In528 spatial dimensions even rare coordinate violations may affect many rows. Smaller inner training sets have narrower estimated support. Existing spatial142 audit explicitly noted missing clipfrequency and possible amplitude loss.','training_size_context':'127 reported all base learners improve from213/214 to320, with CNN/DINO complementary errors.118 halves harmed residual learning;119 pooled residual scheme failed. Do not infer500 accuracy or reopen those schemes. Current audit targets train-extrema prediction transform, not sample pooling, epoch count or new features.','interpretation':'Clipping frequency and residual associations are descriptive, not proof of an error cause. No optimized base weights or region rule fitted. Public019 result supplies status only.','limits':'Historically reused400 IDs; correlated models and overlapping partitions; no independent validation.','source_sha256':e.sha(Path(__file__)),'cost':0};e.write(_repo_path(ROOT, 'reports/support-diagnostic-153.json'),report);e.emit({'summary':summary,'partitions':parts})
