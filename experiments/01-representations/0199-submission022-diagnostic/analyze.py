from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,hashlib,math,time,resource,socket
from pathlib import Path
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3]
assert Path(sys.prefix).resolve()==_repo_path(P, '.venv')
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='2'
def blocked(*a,**k):raise RuntimeError('Network forbidden in199 diagnostics')
socket.create_connection=blocked;socket.socket.connect=blocked;socket.socket.connect_ex=blocked
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(2)
def read(p):return json.loads(Path(p).read_text())
def write(p,z):
 with Path(p).open('x') as f:json.dump(z,f,indent=2,ensure_ascii=False,allow_nan=False)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(_repo_io(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).absolute();mode=args[1];flags=args[2];writing=(isinstance(mode,str) and any(q in mode for q in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
  if writing and not p.is_relative_to(R):raise RuntimeError('199outputs only')
  if p.is_relative_to(_repo_path(P, 'data')) or p==_repo_path(P, 'experiments/candidate022-198/train-input.npz') or p==_repo_path(P, 'experiments/reproduction-018-176/run/train-features.npz') or p.name=='candidate022-public-submission.json':raise RuntimeError('Rawdata/full500labels/Publicscore blocked in quantitative diagnostics')
sys.addaudithook(audit);start=time.perf_counter();plan=read(_repo_path(R, 'plan.json'))
for n,h in plan['input_SHA256'].items():assert sha(_repo_path(P, n))==h,n
assert sha(Path(__file__))==read(_repo_path(R, 'analysis-code-lock.json'))['analyze.py']
o=load(_repo_path(P, 'experiments/matched-replacement-194/replacement-OOF.npz'));z=load(_repo_path(P, 'experiments/matched-replacement-194/legacy-precision.npz'));m=load(_repo_path(P, 'artifacts/run-012/development_features.npz'));names=read(_repo_path(P, 'artifacts/run-012/report.json'))['features'];ids=o['ids'];y=o['y'];old=o['best021'];new=o['true'];delta=new-old;benefit=(old-y)**2-(new-y)**2
assert len(ids)==len(set(ids))==len(set(o['groups']))==400
for a in [z,m]:np.testing.assert_array_equal(a['ids'],ids)
np.testing.assert_array_equal(m['y'],y);np.testing.assert_array_equal(m['groups'],o['groups']);np.testing.assert_allclose(old+o['coefficient']*(z['true']-z['detach']),new,atol=1e-12,rtol=0);np.testing.assert_array_equal(o['detach'],old)
def stats(a):
 a=np.asarray(a,dtype=float);assert np.isfinite(a).all()
 return {'n':len(a),'mean':float(a.mean()),'std':float(a.std()),'RMS':float(np.sqrt(np.mean(a*a))),'mean_absolute':float(np.mean(abs(a))),'quantile_0_5_25_50_75_95_100':np.quantile(a,[0,.05,.25,.5,.75,.95,1]).tolist()}
def corr(a,b):return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None
def rmse(y,p):return math.sqrt(math.fsum((float(a)-float(b))**2 for a,b in zip(y,p))/len(y))
def compare(y,a,b):
 ea=a-y;eb=b-y;d=b-a;oldrm=rmse(y,a);newrm=rmse(y,b);msegain=float(np.mean(ea*ea-eb*eb));biasgain=float(ea.mean()**2-eb.mean()**2);vargain=float(ea.var()-eb.var());cross=float(-2*np.mean(ea*d));penalty=float(np.mean(d*d));assert abs(msegain-biasgain-vargain)<1e-10 and abs(msegain-cross+penalty)<1e-10
 return {'n':len(y),'target_mean':float(y.mean()),'target_std':float(y.std()),'old_RMSE':oldrm,'new_RMSE':newrm,'RMSE_gain':oldrm-newrm,'MSE_gain':msegain,'SSE_gain':float(np.sum(ea*ea-eb*eb)),'old_prediction_mean':float(a.mean()),'new_prediction_mean':float(b.mean()),'old_prediction_std':float(a.std()),'new_prediction_std':float(b.std()),'old_error_mean_p_minus_y':float(ea.mean()),'new_error_mean_p_minus_y':float(eb.mean()),'old_centered_error_variance':float(ea.var()),'new_centered_error_variance':float(eb.var()),'MSE_gain_from_bias_squared':biasgain,'MSE_gain_from_centered_error_variance':vargain,'MSE_gain_alignment_term_minus2E_old_error_delta':cross,'MSE_cost_Edelta2':penalty,'delta_mean':float(d.mean()),'delta_RMS':float(np.sqrt(np.mean(d*d))),'delta_vs_y_correlation':corr(d,y),'delta_vs_old_residual_correlation':corr(d,ea),'rows_improved':int(np.sum(ea*ea>eb*eb)),'rows_worsened':int(np.sum(ea*ea<eb*eb)),'rows_equal':int(np.sum(ea*ea==eb*eb))}
primary=compare(y,old,new);assert abs(primary['old_RMSE']-12.786200775750695)<1e-12 and abs(primary['new_RMSE']-12.761504566432446)<1e-12
controls={'permuted_vs021':compare(y,old,o['permuted']),'true_vspermuted':compare(y,o['permuted'],new),'raw_CNN_true_vsdetach':compare(y,z['detach'],z['true']),'raw_CNN_permuted_vsdetach':compare(y,z['detach'],z['permuted'])}
order=np.lexsort((ids,-benefit));grosspos=float(benefit[benefit>0].sum());grossneg=float(benefit[benefit<0].sum());total=float(benefit.sum());concentration=[]
for k in [1,5,10,20]:
 top=order[:k];keep=order[k:];topgain=float(benefit[top].sum());concentration.append({'top_k':k,'IDs':ids[top].tolist(),'SSE_gain':topgain,'share_of_net_gain':topgain/total,'share_of_gross_positive_gain':topgain/grosspos,'remaining_n':len(keep),'remaining_RMSE_gain':rmse(y[keep],old[keep])-rmse(y[keep],new[keep])})
folds=[];leave=[]
for k in range(5):
 ix=o['fold']==k;folds.append({'fold':k,**compare(y[ix],old[ix],new[ix])});ix=~ix;leave.append({'left_out_fold':k,**compare(y[ix],old[ix],new[ix])})
sub=[];strata_defs={}
for axis,a in [('hardness',y)]+[(n,m['X'][:,names.index(n)]) for n in plan['morphology_axes']]:
 edges=np.quantile(a,[.25,.5,.75],method='linear');bucket=np.searchsorted(edges,a,side='right');strata_defs[axis]={'edges':edges.tolist(),'unique_edges':np.unique(edges).tolist(),'counts':[int(sum(bucket==j)) for j in range(4)],'range':[float(a.min()),float(a.max())]}
 for j in range(4):
  ix=bucket==j
  if not ix.any():sub.append({'axis':axis,'quartile':j+1,'n':0,'empty_due_to_ties':True});continue
  q=compare(y[ix],old[ix],new[ix]);sub.append({'axis':axis,'quartile':j+1,'feature_min':float(a[ix].min()),'feature_max':float(a[ix].max()),**q,'net_SSE_share':q['SSE_gain']/total,'permuted_RMSE':rmse(y[ix],o['permuted'][ix]),'permuted_RMSE_gain_vs021':rmse(y[ix],old[ix])-rmse(y[ix],o['permuted'][ix])})
full_old=load(_repo_path(P, 'experiments/candidate022-198/detach/predictions.npz'));full_new=load(_repo_path(P, 'experiments/candidate022-198/true/predictions.npz'));np.testing.assert_array_equal(full_old['ids'],full_new['ids']);assert full_old['ids'].tolist()==[f'TRAIN_{j:06d}' for j in range(1,501)];fi={s:j for j,s in enumerate(full_old['ids'])};dev=np.array([fi[s] for s in ids]);coef=.08993874491211938;fullraw=full_new['prediction']-full_old['prediction'];fulldelta=coef*fullraw;oofraw=z['true']-z['detach'];fd=fulldelta[dev];test=load(_repo_path(P, 'experiments/candidate022-198/inference/predictions.npz'));assert test['ids'].tolist()==[f'TEST_{j:06d}' for j in range(1,1001)];testdelta=test['primary']-test['reference021'];testraw=test['new_CNN_MPS']-test['old_CNN_MPS'];np.testing.assert_allclose(coef*testraw,testdelta,atol=1e-12,rtol=0)
change={'OOF_dev400_weighted':stats(delta),'OOF_dev400_rawCNN':stats(oofraw),'full500_in_sample_weighted':stats(fulldelta),'full500_in_sample_rawCNN':stats(fullraw),'full500_dev400_matched_in_sample_weighted':stats(fd),'full500_dev400_matched_in_sample_rawCNN':stats(fullraw[dev]),'test1000_unlabeled_weighted':stats(testdelta),'test1000_unlabeled_rawCNN':stats(testraw),'OOF_to_full500_same400_weighted_correlation':corr(delta,fd),'OOF_to_full500_same400_rawCNN_correlation':corr(oofraw,fullraw[dev]),'OOF_to_full500_same400_sign_agreement':float(np.mean(np.sign(delta)==np.sign(fd))),'OOF_to_full500_same400_delta_difference_RMS':float(np.sqrt(np.mean((fd-delta)**2))),'full500_dev400_RMS_over_OOF_RMS':float(np.sqrt(np.mean(fd*fd))/np.sqrt(np.mean(delta*delta))),'test_RMS_over_OOF_RMS':float(np.sqrt(np.mean(testdelta*testdelta))/np.sqrt(np.mean(delta*delta))),'coefficient_full':coef,'coefficient_OOF_min_max':[float(o['coefficient'].min()),float(o['coefficient'].max())],'interpretation_limit':'Full500same400 are in-sample, different fits/seeds/training sizes; theirshifts do notmeasuregeneralization. Testcontainsall1000, unknownpublicsubset, nolabels. Distribution differences cannotidentifythecauseofPublicerror.'}
# Audit unchanged creation evidence. Full500label cache is not even opened.
root=_repo_path(P, 'experiments/candidate022-198');manifest=read(_repo_path(root, 'final-artifact-manifest.json'));count=0
for n,h in manifest.items():
 if n=='train-input.npz':continue
 assert sha(_repo_path(root, n))==h,n;count+=1
tr=read(_repo_path(root, 'training-validation.json'));iv=read(_repo_path(root, 'inference-validation.json'));completion=read(_repo_path(root, 'completion.json'));updates=read(_repo_path(R, 'authorized-record-updates.json'));backup=_repo_path(P, updates['before']['artifacts/submission-candidate-022.manifest.json']['backup']);assert sha(backup)==completion['CSV_manifest_SHA256'];assert tr['passed'] and iv['passed'];assert sha(_repo_path(P, 'artifacts/submission-candidate-022.csv'))==iv['CSV_SHA256']==completion['CSV_SHA256'];assert sha(_repo_path(P, 'artifacts/submission-candidate-021.csv'))==completion['best021_SHA256']
for n,h in plan['input_SHA256'].items():assert sha(_repo_path(P, n))==h,n
# Human-inspectable data; allrowsarethe authorized400development OOF.
with (_repo_path(R, 'row-diagnostics.csv')).open('x',newline='') as f:
 fields=['ID','fold','hardness','best021','true194','permuted194','delta','SSE_gain','rawCNN_delta','full500_in_sample_rawCNN_delta']+plan['morphology_axes'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for i in order:w.writerow(dict(zip(fields,[str(ids[i]),int(o['fold'][i]),float(y[i]),float(old[i]),float(new[i]),float(o['permuted'][i]),float(delta[i]),float(benefit[i]),float(oofraw[i]),float(fullraw[dev[i]])]+[float(m['X'][i,names.index(n)]) for n in plan['morphology_axes']])))
fields=list(dict.fromkeys(k for q in sub for k in q))
with (_repo_path(R, 'stratum-diagnostics.csv')).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(sub)
result={'study':199,'scope':plan['scope'],'primary':primary,'controls':controls,'concentration':{'net_SSE_gain':total,'gross_positive_SSE_gain':grosspos,'gross_negative_SSE_gain':grossneg,'net_over_gross_positive':total/grosspos,'top_removal':concentration},'folds':folds,'leave_onefoldout':leave,'strata_definitions':strata_defs,'strata':sub,'prediction_shift_comparison':change,'technical_evidence':{'saved198_creation_files_SHA256_verified':count,'full500_label_cache_read':False,'existing_model_reproduction_and_inference_certificates_verified':True,'022_CSV_SHA256':iv['CSV_SHA256'],'021_CSV_SHA256':completion['best021_SHA256'],'manifest_lifecycle_change_backed_up_exact':True,'new_inference_or_fit':False,'new_technical_error_found':False,'technical_audit_limits':'No new modelexecution; verified saved certificates and immutable198 files. Doesnotproveabsenceofallpossible sharedcode or dataseterrors.'},'limits':['All subgroup andconcentration diagnostics areposthoc, notselectioncriteria andnotcausalattribution.','Overlappinglabel/morphologystrata arenotindependent; smalln andmanycomparisons.','Morphologicaldescriptors are imageproxies, notmeasuredphysicalcauses.','No500labels/testlabels/Publicscore enteredquantitativeanalysis.','Conditional194CI andsame400 adaptivehistorylimit evidence; noaggregatePublicscore inversion.'],'source_SHA256':sha(Path(__file__)),'elapsed_seconds':time.perf_counter()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'new_training_inference_install_download_cost':False}
assert result['peak_RSS_bytes']<6e9;write(_repo_path(R, 'diagnostics.json'),result);print(json.dumps({'primary':primary,'concentration':result['concentration'],'fold_RMSE_gains':[q['RMSE_gain'] for q in folds],'leaveonefoldout_RMSE_gains':[q['RMSE_gain'] for q in leave],'strata':[{k:q.get(k) for k in ['axis','quartile','n','RMSE_gain','old_error_mean_p_minus_y','new_error_mean_p_minus_y','delta_mean','net_SSE_share']} for q in sub],'shift':change,'technical_evidence':result['technical_evidence'],'seconds':result['elapsed_seconds'],'peak_RSS_bytes':result['peak_RSS_bytes']},ensure_ascii=False),flush=True)
