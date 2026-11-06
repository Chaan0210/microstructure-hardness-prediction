"""Descriptive train-support vs018 common errors; existing400 only, no fit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,time,hashlib,re
import numpy as np
from scipy.stats import spearmanr
import label_density_169 as l
R=_RepoPath(__file__).resolve().parents[1];O=_repo_path(R, 'artifacts/label-density-169');SPLITS=['original','20261031','20261101'];NAMES=['KRR','CNN_single','DINO_S_mean3','DINO_B_mean3'];SOURCES={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):SOURCES[str(p.relative_to(R))]=sha(p);return json.loads(p.read_text())
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def npz(p,h=None):
    if h:assert sha(p)==h
    SOURCES[str(p.relative_to(R))]=sha(p)
    with np.load(_repo_io(p)) as z:return {k:z[k].copy() for k in z.files}
def stats(y,p,mask):
    if not mask.any():return {'n':0}
    e=y[mask]-p[mask];return {'n':int(mask.sum()),'RMSE':float(np.sqrt(np.mean(e*e))),'bias_y_minus_pred':float(e.mean()),'MAE':float(abs(e).mean())}
def cov(x,y):return float(np.mean((x-x.mean())*(y-y.mean())))
def main():
    start=time.perf_counter();O.mkdir(exist_ok=False)
    write(_repo_path(O, 'diagnostic-plan.json'),{'question':'Are persistent018 commonerrors consistent with scarce-training-target support ratherthan simple target-conditioned residual selection?','scope':'Existing400 cachedlabels/predictions, no rawimage/holdout/test/newfit. Descriptive posthoc, no causality claim.','density':'GaussiancontinuousKDE bandwidth.4*trainingSD. TrainLOOdensity; query usesonlytraininglabels. Train q20density sparse threshold, training q10/q90label tails.','analysis':'All3existingpartitions, perbranch predictions, existing127cleaninference training/outervalidation residuals. WithinID across3split density-error association removesfixedy butnot trainingmembership/modelconfounding. Predictionconditioned error slope and covarianceidentity highlighttargetselection effect.','proposed_mild_weight':'TrainLOOdensity ratio^-1/4, ratiofloor.1median, clipraw[.8,1.25], normalize fulltrainingmean1;max/min<=1.5625. No density/bandwidth/exponent/cap search.','prior_audit':'No targetdensity/LDSsampleweight implementation found in targetedroot/src/plan/summary searches.058DINO SmoothL1->halfMSE failed;120shortDINO MSE/ranking failed;051auxhardbins notdensityweight;131RNC notdensityweight.125ensemblecomponentweights notsampleweights. CurrentCNN alreadyMSE.','limitations':'Marginaltargetdensity is label-derivednotdeployablefeature; labeltail error doesnotproveunderfitting. WeightedMSE changesestimand andcanhurtunweightedRMSE. Same400adaptivehistory.'})
    case=npz(_repo_path(R, 'artifacts/failure-review-166/analysis.npz'));ids=case['ids'];y=case['y'];large=case['persistent_large'];assert len(ids)==400
    cache127=_repo_path(R, 'artifacts/base-learning-curve-127/numeric-replay-v2');cert=read(cache127/'validation.json');assert cert['passed']
    allrho=[];allerr=[];allspan=[];allrare=[];alloutside=[];results={};train_summary=[]
    for si,split in enumerate(SPLITS):
        root=_repo_path(R, 'artifacts/bootstrap-selector-144', split);rec=read(root/'report.json');z=npz(root/'predictions.npz',rec['predictions_sha256']);np.testing.assert_array_equal(z['ids'],ids);np.testing.assert_array_equal(z['y'],y);p=z['primary'];fid=z['fold'];rho=np.empty(400);rare=np.zeros(400,bool);outside=rare.copy();tail=np.empty(400,int);folds=[];branches=case['components'][si]
        assert sha(cache127/split/'report.json')==cert['report_sha256'][split];read(cache127/split/'report.json')
        for k in range(5):
            T=np.flatnonzero(fid!=k);V=np.flatnonzero(fid==k);trrho=l.density(y[T],y[T],True);rho[V]=l.density(y[T],y[V]);cut=np.quantile(trrho,.2);rare[V]=rho[V]<cut;outside[V]=(y[V]<y[T].min())|(y[V]>y[T].max());lo,hi=np.quantile(y[T],[.1,.9]);tail[V]=np.where(y[V]<lo,-1,np.where(y[V]>hi,1,0));w,wm=l.weights(y[T]);near=(abs(y[V,None]-y[T][None,:])<=.4*y[T].std()).sum(1)
            path=cache127/split/f'fold-{k}-level-2-path-0';r=read(path/'report.json');v=npz(path/'predictions.npz',r['predictions_sha256']);np.testing.assert_array_equal(v['train_indices'],T);np.testing.assert_array_equal(v['outer_indices'],V);pred=v['components'];assert pred.shape==(4,400)
            trrare=trrho<cut;trtail=np.where(y[T]<lo,-1,np.where(y[T]>hi,1,0));br={}
            for j,name in enumerate(NAMES):
                br[name]={'train_all':stats(y[T],pred[j,T],np.ones(len(T),bool)),'validation_all':stats(y[V],pred[j,V],np.ones(len(V),bool)),'train_sparse':stats(y[T],pred[j,T],trrare),'validation_sparse':stats(y[V],pred[j,V],rare[V]),'train_lower':stats(y[T],pred[j,T],trtail==-1),'validation_lower':stats(y[V],pred[j,V],tail[V]==-1),'train_upper':stats(y[T],pred[j,T],trtail==1),'validation_upper':stats(y[V],pred[j,V],tail[V]==1),'train_large_cases':stats(y[T],pred[j,T],large[T]),'validation_large_cases':stats(y[V],pred[j,V],large[V])}
            folds.append({'fold':k,'train_range':[float(y[T].min()),float(y[T].max())],'train_SD':float(y[T].std()),'train_density_q20':float(cut),'train_label_q10_q90':[float(lo),float(hi)],'validation_sparse_n':int(rare[V].sum()),'validation_outside_n':int(outside[V].sum()),'validation_large_n':int(large[V].sum()),'validation_large_sparse_n':int((large[V]&rare[V]).sum()),'validation_neighbor_count_within_bandwidth':{'min':int(near.min()),'median':float(np.median(near))},'weights':{k:v for k,v in wm.items() if k not in ['density','raw_weights']},'branches':br});train_summary.append(br)
        err=y-p;predictcuts=np.quantile(p,[.1,.9]);predtail=np.where(p<predictcuts[0],-1,np.where(p>predictcuts[1],1,0));Cyp=cov(y,err);Cpp=cov(p,err);Ve=float(np.var(err));assert abs(Cyp-Cpp-Ve)<1e-9
        results[split]={'018':{'all':stats(y,p,np.ones(400,bool)),'sparse':stats(y,p,rare),'dense':stats(y,p,~rare),'lower_target':stats(y,p,tail==-1),'upper_target':stats(y,p,tail==1),'middle_target':stats(y,p,tail==0),'outside_train_range':stats(y,p,outside),'large33':stats(y,p,large),'large_sparse_count':int((large&rare).sum()),'sparse_density_vs_abs_error_spearman':float(spearmanr(rho,abs(err)).statistic),'prediction_low':stats(y,p,predtail==-1),'prediction_high':stats(y,p,predtail==1),'prediction_middle':stats(y,p,predtail==0),'residual_on_prediction_slope':Cpp/float(np.var(p)),'diagnostic_OLS_y_on_prediction_slope':1+Cpp/float(np.var(p)),'selection_identity':{'Cov_y_error':Cyp,'Cov_pred_error':Cpp,'Var_error':Ve}},'raw_current_branch_bias':{name:{'all':stats(y,branches[:,j],np.ones(400,bool)),'sparse':stats(y,branches[:,j],rare),'lower':stats(y,branches[:,j],tail==-1),'upper':stats(y,branches[:,j],tail==1)} for j,name in enumerate(['KRR','CNNmean3','DINOmean'])},'folds':folds};allrho.append(rho);allerr.append(err);allrare.append(rare);alloutside.append(outside);print('DIAGNOSTIC_SPLIT',split,results[split]['018'],flush=True)
    rho=np.stack(allrho);err=np.stack(allerr);rare=np.stack(allrare);outside=np.stack(alloutside);dr=rho-rho.mean(0);de=abs(err)-abs(err).mean(0);dm=(err*err)-(err*err).mean(0)
    summary={}
    for name in NAMES:
        b=[r[name] for r in train_summary];ss={}
        for group in b[0]:
            rows=[r[group] for r in b if r[group]['n']];den=sum(z['n'] for z in rows);ss[group]={'row_evaluations_overlapping':den,'RMSE':float(np.sqrt(sum(z['n']*z['RMSE']**2 for z in rows)/den)),'bias':float(sum(z['n']*z['bias_y_minus_pred'] for z in rows)/den)}
        summary[name]=ss
    # Search manifest evidence, only code/plans/reports; log snippets showingnon-sampleweight meanings.
    rex=re.compile(r'\bLDS\b|label.distribution|density.?weight|inverse.?density|target.?density|label.?balanc|target.?balanc|balanced.?mse|weighted.?mse|sample_weight|huber|robust.?loss|reweight',re.I)
    files=list(_study_sources(R))+list((_repo_path(R, 'src')).rglob('*.py'))+list((_repo_path(R, 'reports')).rglob('*summary.json'))+list((_repo_path(R, 'artifacts')).glob('*/plan.json'));hits=[]
    for file in files:
        if '169' in file.name:continue
        for line in file.read_text().splitlines():
            match=rex.search(line)
            if match:hits.append({'file':str(file.relative_to(R)),'snippet':line[max(0,match.start()-70):match.end()+100]})
    out={'status':'descriptive_diagnostic_complete_no_training_yet','n_unique':400,'label_quantiles':dict(zip(['min','q05','q10','q25','q50','q75','q90','q95','max'],np.quantile(y,[0,.05,.1,.25,.5,.75,.9,.95,1]).tolist())),'large33_sparse_in_all3':int((large&rare.all(0)).sum()),'large33_sparse_in_any':int((large&rare.any(0)).sum()),'large33_outside_train_range_any':int((large&outside.any(0)).sum()),'outside_train_range_unique':int(outside.any(0).sum()),'within_ID_across3_association':{'centered_density_abs_error_Pearson':float(np.corrcoef(dr.ravel(),de.ravel())[0,1]),'centered_density_squared_error_Pearson':float(np.corrcoef(dr.ravel(),dm.ravel())[0,1]),'interpretation':'TargetfixedwithinID,butfoldtrainingcomposition,model/calibratorchangesconfound; threeoverlappingobservations perID,notcausal/n1200.'},'branches_train_validation':summary,'per_split':results,'prior_search':{'files_scanned':len(files),'matching_snippets':hits,'finding':'No direct density-dependent perexample trainingweight found; hitsensemble/selectorweighting orwarnings.058/120 lossalignment differsfromdensityweighting;051/131 representationobjectives differ.'},'source_hashes':SOURCES,'source_code_sha256':{f:sha(_repo_path(R, f)) for f in ['diagnose_label_density_169.py','label_density_169.py']},'limits':['Stronglowdensity/targettail errors mayreflect mathematicallyexpected conditionalmean shrinkage, not causal dataimbalance.','Existing127train predictions show overfit/interpolation, not independentlymeasuredtrainerror. CNN_single differsfromcurrentCNNmean3.','WeightedMSE changesBayesestimand; generalunweightedRMSE and018fixedblend mustdecidebenefit, nottail-onlygain.','Diagnostic and proposedmodel sharehistoricallyreused400; anygainexploratory.'],'seconds':time.perf_counter()-start}
    np.savez_compressed(_repo_path(O, 'diagnostic.npz'),ids=ids,y=y,density=rho,residual018=err,sparse=rare,outside=outside,large=large)
    write(_repo_path(O, 'diagnostic.json'),out);print('SUMMARY',json.dumps({k:out[k] for k in ['label_quantiles','large33_sparse_in_all3','large33_sparse_in_any','large33_outside_train_range_any','outside_train_range_unique','within_ID_across3_association','seconds']},indent=2),flush=True)
if __name__=='__main__':main()
