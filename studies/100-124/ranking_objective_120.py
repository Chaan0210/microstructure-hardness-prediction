"""Train-only pairwise order regularization with MSE control and nested two-choice selection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,datetime,argparse,hashlib,math,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pooled_residual_119 as previous
import nonlinear_residual_115 as residual
import numpy as np,torch
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
e=residual.e;OUT=_repo_path(ROOT, 'artifacts/ranking-objective-120');WEIGHTS=(0.,.1);SEED=20270215;EPOCHS=12

def load(split):
 d=e.old.inputs(split);file=residual.OUT/split/'predictions.npz';rec=e.read(residual.OUT/split/'report.json');assert e.sha(file)==rec['predictions_sha256']
 with np.load(_repo_io(file),allow_pickle=False) as z:assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['y'],d['y']);np.testing.assert_array_equal(z['fold'],d['fid']);np.testing.assert_array_equal(z['reference011'],d['reference011']);d['reference013']=z['primary'].copy()
 return d

def rank_loss(pred,y):
 # Pair identities and signs depend exclusively on this training group's standardized targets.
 ii,jj=np.triu_indices(len(y),1);delta=np.asarray(y)[ii]-np.asarray(y)[jj];mask=np.abs(delta)>=.5;ii=ii[mask];jj=jj[mask];delta=delta[mask]
 if not len(ii):return pred.sum()*0.,0
 I=torch.tensor(ii,device=pred.device);J=torch.tensor(jj,device=pred.device);sign=torch.tensor(np.sign(delta),dtype=pred.dtype,device=pred.device)
 return F.softplus(-sign*(pred[I]-pred[J])).mean(),len(ii)

def fit(d,tr,va,seed,weight,path,loss_kind='mse'):
 path.mkdir(exist_ok=False);start=time.perf_counter();assert not set(tr)&set(va) and not set(d['g'][tr])&set(d['g'][va]);sc=StandardScaler().fit(d['B'][tr]);ym=float(d['y'][tr].mean());ys=float(d['y'][tr].std());model=e.Small(sc.mean_,sc.scale_,seed).to('mps');initial=e.old.e.h.state_hash(model.state_dict());opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.1);orders=np.random.default_rng(seed).permuted(np.tile(np.arange(len(tr)),(EPOCHS,1)),axis=1);maskseeds=[];curve=[];pairtotal=0;first_rank_grad=None;steps=0
 for ep,order in enumerate(orders):
  lr=1e-5+.5*(.0003-1e-5)*(1+math.cos(math.pi*ep/EPOCHS));opt.param_groups[0]['lr']=lr;model.train();reglog=[];ranklog=[];ep_pairs=0
  for first in range(0,len(tr),16):
   group=tr[order[first:first+16]];opt.zero_grad(set_to_none=True);preds=[];ygroup=(d['y'][group]-ym)/ys
   for off in range(0,len(group),2):
    ii=group[off:off+2];rng=seed+10000*(ep+1)+first+off;torch.manual_seed(rng);maskseeds.append(rng);pred=model(torch.tensor(d['B'][ii],dtype=torch.float32,device='mps'));yy=torch.tensor((d['y'][ii]-ym)/ys,dtype=torch.float32,device='mps');reg=.5*F.mse_loss(pred,yy) if loss_kind=='mse' else F.smooth_l1_loss(pred,yy,beta=1.);assert torch.isfinite(reg);(reg*len(ii)/len(group)).backward(retain_graph=weight>0);reglog.append(float(reg.detach()));preds.append(pred if weight else pred.detach())
   rank,pairs=rank_loss(torch.cat(preds),ygroup);assert torch.isfinite(rank);pairtotal+=pairs;ep_pairs+=pairs;ranklog.append(float(rank.detach()))
   if weight:
    if first_rank_grad is None and pairs:
     grads=torch.autograd.grad(weight*rank,list(model.head.parameters()),retain_graph=True);first_rank_grad=float(torch.sqrt(sum(g.square().sum() for g in grads)).detach());assert first_rank_grad>0
    (weight*rank).backward()
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step();steps+=1
  curve.append({'epoch':ep+1,'online_reg_loss':float(np.mean(reglog)),'online_rank_loss':float(np.mean(ranklog)),'eligible_pairs':ep_pairs,'lr':lr})
 pred=e.small_predict(model,d['B'],va)*ys+ym;trainpred=e.small_predict(model,d['B'],tr)*ys+ym;state={name:v.detach().cpu() for name,v in model.state_dict().items()};torch.save({'state':state,'seed':seed,'rank_weight':weight,'loss_kind':loss_kind,'mean':torch.tensor(sc.mean_),'scale':torch.tensor(sc.scale_),'var':torch.tensor(sc.var_),'target_mean':ym,'target_std':ys,'train_indices':tr.tolist(),'validation_indices':va.tolist(),'train_ids':[d['ids'][i] for i in tr],'validation_ids':[d['ids'][i] for i in va],'epochs':EPOCHS},_repo_io(path/'model.pt'))
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=tr,validation_indices=va,predictions=pred)
 report={'seed':seed,'weight':weight,'loss_kind':loss_kind,'initial_sha256':initial,'state_sha256':e.old.e.h.state_hash(state),'order_sha256':hashlib.sha256(orders.tobytes()).hexdigest(),'dropout_sha256':hashlib.sha256(np.array(maskseeds,dtype=np.int64).tobytes()).hexdigest(),'optimizer_steps':steps,'example_visits':len(tr)*EPOCHS,'eligible_pair_uses':pairtotal,'first_weighted_rank_gradient_norm':first_rank_grad,'curve':curve,'train_RMSE':e.rmse(d['y'][tr],trainpred),'model_sha256':e.sha(path/'model.pt'),'predictions_sha256':e.sha(path/'predictions.npz'),'seconds':time.perf_counter()-start};e.write(path/'report.json',report);del model,opt;torch.mps.empty_cache();return pred,report

def prepare():
 previous.check();OUT.mkdir(exist_ok=False);protected=dict(previous.check()['protected'])
 for name in ['reports/pooled-residual-119-summary.json','reports/pooled-residual-119-preservation.json','reports/candidate013-summary.json']:protected[name]=e.sha(_repo_path(ROOT, name))
 plan={'study':'120 MSE with pairwise ordinal regularization','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'duplicate_audit':{'051':'Hard7quantile-binCE auxhead onS64hidden100epochs;auxspecificbenefit inconsistent across3partitions. No ordinalpairloss.','058':'HalfMSE vsSmoothL1 S64hidden100epoch3seed,almostidenticalpredictions;not pairwiselearning.','099':'Unsupervisedpatch-featuredistribution,not targetdistributionprediction.','ranking':'No previousordinal/pairwise supervisedloss implementation found outsidevendor. Current011shorthead actuallyusesSmoothL1; MSE isexplicitnewmatchedcontrol,notmislabelledasincumbentloss.'},'hypothesis':'Trainfoldlargegap order constraints mayregularize scalarB-head scores beyondindependentMSE. No additionalindependentlabels;rankingimprovement neednotimproveRMSE.','architecture':'Unchanged011shortB cached6912->32GELU,dropout.5->1,featureStandardScaler andtargetmean/std fittrainingrows only. Noauxhead/bins,newfeatures,backbonetraining or augmentation.','loss':'MSEcontrol .5mean((p-z)^2). Rankarmadds .1mean softplus(-sign(z_i-z_j)*(p_i-p_j)) overunorderedpairswithabs(z_i-z_j)>=.5 insideeffectivebatch16. Temperature1,margin0,gapthreshold.5trainingSDfixed;noeligiblepair=>zero. MSEanchorretainsabsolute scale; rankingalonecannotidentifyoffset/scale.','weights':list(WEIGHTS),'training':'Both12fixedepochs,AdamWlr3e-4cosine->1e-5wd.1,clip1,effective16/micro2,seed20270215+fold(+100*(inner+1)). Samewholemodelinit,exampleorder,dropoutseedresetper micro,optimizerstepsandvisits. MSEbackwardper micro;rankbackwardacrosssaved16predictions beforeonesharedclip/update. Addedpairsarenotnewdataorindependentexamples.','nested_selection':'Withinouter320,GroupKFold3 seed20270110+outer;fitbothweights on213/214 andscore107/106. Choose0or.1 bypooleduntrimmedinnerheadMSE,tiesprefer0. Refitbothweights on320 forpairedcontrol/diagnostic;selecteduseschosenendpoint. Noouterlabels fortraining/selection;noepoch,seed,gap,temperature,weightsearch beyondthese2fixedconditions.','primary_formula':'reference013 + .1*(selected_new_shorthead - old011_shorthead). PairedMSEcontrol usesnewMSEhead inexact sameformula. Fixedrankarmreporteddiagnosticonly. Existing009calibrator,longS/Bheads,legacycomponents and013residualcorrection remainfrozeninsideeachoutertrain; thisisashortheadreplacement,notnewblendweight ornewresidualfit. InnerselectionusesheadMSE,notouter013predictionswhichcouldexposeinnerlabels.','primary_metric':'All3existingpartitions pooleduntrimmedMSE vsMSEreplacementcontrol andunchanged013. Candidacyrequiresstrictlylowerthanboth plusindependentvalidation. No everyfold/partition orCIgate. Fixedrankdiagnosticnotfallbackcandidate.','limitations':'Head-onlyinnerMSEselectionmaynotoptimizeensemblecomplementarity. Frozenlegacycalibration/residualcorrectionweretrainedforoldSmoothL1head;replacementtestedwholeprocedureouterOOF. Onefixedseedand12epochunderfit/variancepossible. Pairlabels correlated andnoisy;largegapfilterdoesnotprove reliableordering. Same400historicalselectionbias persists.','controls':'Freshsame-loopSmoothL1originalfold0 replay mustexactlyreproduceincumbent011parameters/predictions. IndependentMSE/ranklossformulaandgradientfixtures,all120modelreloads,trainlocalstatistics,pairedbudgethashes,fulloriginalouterfoldlabelperturbation replayandinnerval perturbation.', 'resource':'120newmainfits (15outer*8fits),existingMPSCPU2,firstinnerpairreusedproject60pairs<900seconds,RSS<3GB,MPS<2GB. No newweights/packages/cost. Fullpipelinevalidationreplaysadditionaloneouterfold+oneinnerpair.','scope':'Development400 beforedecision;holdout/testnotusedforselection,nonewdata/install/externaltransmission/upload. Preserve013/011 andallfailures.','sources':{name:e.sha(_repo_path(ROOT, name)) for name in ['ranking_objective_120.py','validate_ranking_objective_120.py','nested_fusion_112.py']},'inputs':{name:e.sha(_repo_path(ROOT, name)) for name in ['reports/auxiliary-051-summary.json','reports/optimization-057-058-summary.json','reports/candidate013-verification.json','artifacts/dinob-062/development_features.npz']},'protected':protected,'environment':previous.check()['environment']};e.write(_repo_path(OUT, 'plan.json'),plan);e.emit({'prepared':True,'weights':WEIGHTS,'plan_sha256':e.sha(_repo_path(OUT, 'plan.json'))})

def check():
 previous.check();p=e.read(_repo_path(OUT, 'plan.json'))
 for name,h in {**p['sources'],**p['inputs'],**p['protected']}.items():assert e.sha(_repo_path(ROOT, name))==h,name
 return p

def control():
 check();torch.set_num_threads(2);d=load('original');tr=np.flatnonzero(d['fid']!=0);va=np.flatnonzero(d['fid']==0);pred,rec=fit(d,tr,va,SEED,0.,_repo_path(OUT, 'smooth-control'),'smooth');src=e.old.s.foldpath('original',0);old=torch.load(_repo_io(src/'model.pt'),map_location='cpu',weights_only=True);new=torch.load(_repo_path(OUT, 'smooth-control/model.pt'),map_location='cpu',weights_only=True);assert all(torch.equal(v,old['state'][key]) for key,v in new['state'].items());np.testing.assert_array_equal(pred,d['basehead'][va]);e.write(_repo_path(OUT, 'smooth-control-validation.json'),{'passed':True,'state_exact_incumbent':True,'prediction_error':0.,'source_checkpoint_sha256':e.sha(src/'model.pt'),'new_report_sha256':e.sha(_repo_path(OUT, 'smooth-control/report.json'))});e.emit({'smooth_control_exact011':True})

def paired(d,tr,va,seed,path):
 path.mkdir(exist_ok=False);pred=[];records=[]
 for wi,w in enumerate(WEIGHTS):
  pp,rec=fit(d,tr,va,seed,w,path/f'weight-{wi}');pred.append(pp);records.append(rec)
 for key in ['initial_sha256','order_sha256','dropout_sha256','optimizer_steps','example_visits','eligible_pair_uses']:assert records[0][key]==records[1][key],key
 assert records[0]['state_sha256']!=records[1]['state_sha256'];return np.stack(pred),records

def outer(d,split,k,path):
 path.mkdir(exist_ok=False);tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);folds=list(GroupKFold(3,shuffle=True,random_state=e.cal.INNER_SEED+k).split(d['B'][tr],d['y'][tr],d['g'][tr]));ip=np.full((2,len(tr)),np.nan);records=[];start=time.perf_counter()
 for j,(it,iv) in enumerate(folds):
  pp,recs=paired(d,tr[it],tr[iv],SEED+k+100*(j+1),path/f'inner-{j}');ip[:,iv]=pp;records.extend(recs)
  if split=='original' and k==j==0 and path==_repo_path(OUT, split, f'fold-{k}'):
   secs=sum(r['seconds'] for r in recs);b={'first_inner_pair_seconds_reused':secs,'projected60_pairs_seconds':60*secs,'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'MPS_bytes':torch.mps.driver_allocated_memory()};b['passed']=b['projected60_pairs_seconds']<900 and b['RSS_bytes']<3e9 and b['MPS_bytes']<2e9;e.write(_repo_path(OUT, 'benchmark.json'),b);assert b['passed']
 scores=np.mean((ip-d['y'][tr][None,:])**2,1);chosen=int(scores.argmin());pp,recs=paired(d,tr,va,SEED+k,path/'outer');records.extend(recs)
 with (path/'selection.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=tr,validation_indices=va,inner_predictions=ip,outer_predictions=pp,selected=pp[chosen])
 r={'fold':k,'inner_MSE':dict(zip(['mse','rank'],scores.tolist())),'chosen_index':chosen,'chosen_weight':WEIGHTS[chosen],'inner_splits':[{'train_indices':tr[it].tolist(),'validation_indices':tr[iv].tolist()} for it,iv in folds],'selection_sha256':e.sha(path/'selection.npz'),'model_reports_sha256':{str(q.relative_to(path)):e.sha(q) for q in path.rglob('report.json')},'seconds':time.perf_counter()-start};e.write(path/'report.json',r);e.emit({'split':split,'outer':k,'chosen_weight':WEIGHTS[chosen],'seconds':r['seconds']});return pp,pp[chosen],r

def run(split):
 check();assert e.read(_repo_path(OUT, 'smooth-control-validation.json'))['passed'];torch.set_num_threads(2);d=load(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);start=time.perf_counter();raw=np.full((2,400),np.nan);selected=np.full(400,np.nan);folds=[]
 for k in range(5):
  va=np.flatnonzero(d['fid']==k);raw[:,va],selected[va],rec=outer(d,split,k,out/f'fold-{k}');folds.append(rec)
 ps={'reference013':d['reference013'],'mse_control':d['reference013']+.1*(raw[0]-d['basehead']),'rank_fixed_diagnostic':d['reference013']+.1*(raw[1]-d['basehead']),'primary':d['reference013']+.1*(selected-d['basehead'])}
 with (out/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],head_predictions=raw,selected_head=selected,old_head=d['basehead'],**ps)
 report={'split':split,'scores':{key:{'MSE':float(np.mean((d['y']-v)**2)),'RMSE':e.rmse(d['y'],v)} for key,v in ps.items()},'head_scores':{'old_smooth':e.rmse(d['y'],d['basehead']),'mse':e.rmse(d['y'],raw[0]),'fixedrank':e.rmse(d['y'],raw[1]),'selected':e.rmse(d['y'],selected)},'chosen_weights':[f['chosen_weight'] for f in folds],'fold_reports_sha256':{f'fold-{k}/report.json':e.sha(out/f'fold-{k}/report.json') for k in range(5)},'predictions_sha256':e.sha(out/'predictions.npz'),'seconds':time.perf_counter()-start};e.write(out/'report.json',report);check();e.emit(report)
if __name__=='__main__':
 torch.set_num_threads(2);ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','control',*e.SPLITS]);x=ap.parse_args();{'prepare':prepare,'control':control}.get(x.mode,lambda:run(x.mode))()
