"""One fixed half-shrunk orthogonal quadratic extension of015's affine calibration."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cnn_seed_mean_133 as x
import numpy as np,torch
OUT=_repo_path(ROOT, 'artifacts/quadratic-calibration-137');e=x.e;a=x.a;LAMBDA=1.

def fit(p,y):
 p=np.asarray(p,float);y=np.asarray(y,float);center=float(p.mean());scale=float(p.std());assert scale>1e-12;z=(p-center)/scale;linear_center=float(y.mean());linear_slope=float(z@(y-linear_center)/(z@z));sqmean=float(np.mean(z*z));projection=float(z@(z*z-sqmean)/(z@z));q=z*z-sqmean-projection*z;qscale=float(np.sqrt(np.mean(q*q)));degenerate=qscale<1e-12
 q=q/qscale if not degenerate else np.zeros_like(q);res=y-linear_center-linear_slope*z;curvature=float(np.mean(q*res)/(np.mean(q*q)+LAMBDA)) if not degenerate else 0.
 return {'n_train':len(y),'prediction_center':center,'prediction_scale':scale,'target_center':linear_center,'linear_slope_in_standardized_p':linear_slope,'square_mean':sqmean,'square_projection':projection,'quadratic_scale':qscale,'curvature':curvature,'lambda':LAMBDA,'degenerate':degenerate,'extra_effective_df':float(np.mean(q*q)/(np.mean(q*q)+LAMBDA)),'train_pred_range':[float(p.min()),float(p.max())]}
def predict(m,p):
 z=(np.asarray(p,float)-m['prediction_center'])/m['prediction_scale'];q=(z*z-m['square_mean']-m['square_projection']*z)/m['quadratic_scale'] if not m['degenerate'] else np.zeros_like(z)
 return m['target_center']+m['linear_slope_in_standardized_p']*z+m['curvature']*q

def prepare():
 x.check();OUT.mkdir(exist_ok=False);sources=['quadratic_calibration_137.py','validate_quadratic_calibration_137.py','cnn_seed_mean_133.py','component_ablation_121.py','component_single_ablation_122.py','nonlinear_residual_115.py','reports/candidate015-public-submission.json','reports/cnn-seed-mean-133-summary.json','artifacts/bias-audit-073/report.json','reports/baseline124-weights125-summary.json','artifacts/submission-candidate-015.csv','artifacts/submission-candidate-015.manifest.json']
 for split in e.SPLITS:sources.extend([f'artifacts/cnn-seed-mean-133/{split}/report.json',f'artifacts/cnn-seed-mean-133/{split}/predictions.npz',f'artifacts/cnn-seed-mean-133/validation-{split}.json'])
 p={'study':'137 one orthogonal quadratic calibration of fixed015','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hypothesis':'Prediction-conditioned mean error may retain curvature after a single affine fit.073 established affine underdispersion, not curvature; this is a low-capacity falsifiable extension, not established physical law. Domain physics57 nonlinear correction suggests simple linear mappings need not suffice but does not prove this feature helps.','duplicate_audit':{'074':'Single affine ofaggregate.','077':'Unregularized legacy-versus-DINO linear contrast.','079':'Calibration fold count3->5.','125':'Four individual component weights with prior penalty.','115_117_119':'Physics-feature residual kernels and fit sample-size changes.','137':'Add exactly one standardized orthogonal quadratic basis ofthe SAME fixed015 rawmixture; no componentweights ornewrepresentation.'},'procedure':'Fit z=(p-meanTrainP)/stdTrainP; q=z²-meanTrain(z²)-projectionTrainOnZ*z; normalizeqbytrainRMS. Affine coefficients unpenalized; curvature minimizes mean((res-c*q)²)+1*c², equivalenttohalfOLSfornondegenerateq. Allmoments usecurrentmeta training OOF only. No clipping,monotonicconstraint,extraquantiles,model/degree/lambda/weightsearch.','fixed_lambda':LAMBDA,'extra_effective_df':.5,'nested':'Reuse verified133 first3seed CNN mean atouter,middle,deep levels. Outer320 calibrator usesmiddleOOF; A213/214 residualanchor calibrator usesdeepOOF. RefiteachhonestB106/107 physics57 selector andmodel withsame CVseeds. No globalOOFtarget fitting.','primary':'Run one fixed procedure across all3existing400-IDpartitions. Promote only if pooleduntrimmedMSE strictlybelow015 andindependentvalidationpasses. Noeverypartition/foldwinorCIgate; reportconditionaluncertainty. No tuning after failure.','training_resource':'CPU2;60 quadratic calibrators,45residualblocks(720modelsincludingnone),one015controlfoldandoneouter-label-poisonreplay. No newneuralfits.','base_cache_lineage':'133 validatedallnested CNNs/normalization and outer/anchor labelpoison; reused exactly, notclaimedfreshlyretrained in137.','limitations':['Same400 adaptively reused;3partitions overlap and1200pairs are notindependent.','Training size mismatch remains. Quadratic extrapolation can be unstable; diagnostics only, no data-dependent clipping.','Curvature hypothesis is not supported by Public score; Public result only updates preservedbest.'],'hashes':{n:e.sha(_repo_path(ROOT, n)) for n in sources},'protected':{f'artifacts/submission-candidate-{c}.csv':e.sha(_repo_path(ROOT, f'artifacts/submission-candidate-{c}.csv')) for c in ['011','014','015']},'scope':'Development400 only; no holdout/test decoding, full500 fit, newCSV orupload until positive validated decision. No newpackage,cost,externaldata.','cost':0};e.write(_repo_path(OUT, 'plan.json'),p);e.emit({'locked137':True,'lambda':LAMBDA,'new_neural_fits':0,'partitions':list(e.SPLITS)})
def check():
 p=e.read(_repo_path(OUT, 'plan.json'))
 for n,h in {**p['hashes'],**p['protected']}.items():assert e.sha(_repo_path(ROOT, n))==h,n
 return p

def data(split):
 d=x.data(split);rr=e.read(x.OUT/split/'report.json');z=a.readnp(x.OUT/split/'predictions.npz',rr['predictions_sha256']);assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['y'],d['y']);np.testing.assert_array_equal(z['fold'],d['fid']);d['reference015']=z['primary'];return d

def fold(d,k,path):
 path.mkdir();T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);m=fit(a.mixture(d['middleP'][k],x.ARM),d['y'][T]);base=.9*predict(m,a.mixture(d['P'][V],x.ARM))+.1*d['short'][V];cor=[];anchors=[];records=[]
 for j in range(3):
  A,B,P,short,_=d['cells'][k,j];model=fit(a.mixture(d['deepP'][k,j],x.ARM),d['y'][A]);anchor=.9*predict(model,a.mixture(P,x.ARM))+.1*short;cc,rec=a.fit_block(d,B,anchor,k,j,path/f'block-{j}');cor.append(cc);anchors.append(model);records.append(rec)
 pred=base+np.mean(cor,axis=0)
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),outer_indices=V,base=base,corrections=np.stack(cor),primary=pred)
 rec={'fold':k,'outer_model':m,'anchor_models':anchors,'selected':[v['chosen'] for v in records],'block_report_sha256':{f'block-{j}/report.json':e.sha(path/f'block-{j}/report.json') for j in range(3)},'predictions_sha256':e.sha(path/'predictions.npz')};e.write(path/'report.json',rec);return pred,rec

def run(split):
 check();d=data(split);out=_repo_path(OUT, split);out.mkdir();start=time.perf_counter();primary=np.full(400,np.nan);records=[]
 for k in range(5):
  dd,_=x.replace(d,split,k,x.OUT/split/f'cnn-fold-{k}',False);V=np.flatnonzero(d['fid']==k)
  if split=='original' and k==0:
   control,_,_=a.fold(dd,k,x.ARM,out/'control015-fold-0');np.testing.assert_allclose(control,d['reference015'][V],atol=1e-10,rtol=0)
  primary[V],rec=fold(dd,k,out/f'fold-{k}');records.append(rec);e.emit({'split':split,'outer':k,'curvature':rec['outer_model']['curvature']})
 with (out/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],reference015=d['reference015'],primary=primary)
 r={'split':split,'reference015_RMSE':e.rmse(d['y'],d['reference015']),'primary_RMSE':e.rmse(d['y'],primary),'fold_report_sha256':{f'fold-{k}/report.json':e.sha(out/f'fold-{k}/report.json') for k in range(5)},'predictions_sha256':e.sha(out/'predictions.npz'),'seconds':time.perf_counter()-start};e.write(out/'report.json',r);check();e.emit(r)
if __name__=='__main__':
 torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare',*e.SPLITS]);args=p.parse_args();prepare() if args.mode=='prepare' else run(args.mode)
