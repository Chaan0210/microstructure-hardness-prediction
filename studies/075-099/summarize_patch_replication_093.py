"""Fixed originaltop5 versus remaining395 tracking; no candidate selection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,csv,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import patch_replication_093 as s
import numpy as np
fv=s.fv

def subset(y,old,new,ix,ci):
 a=y[ix]-old[ix];b=y[ix]-new[ix];sq=a*a-b*b;gain=float(np.sqrt(np.mean(a*a))-np.sqrt(np.mean(b*b)));r={'n':len(ix),'reference011_RMSE':float(np.sqrt(np.mean(a*a))),'attention_blend_RMSE':float(np.sqrt(np.mean(b*b))),'RMSE_gain':gain,'mean_squared_error_gain':float(sq.mean()),'MAE_gain':float(np.mean(abs(a)-abs(b))),'positive_gain_rows':int((sq>0).sum()),'negative_gain_rows':int((sq<0).sum()),'prediction_change_RMS':float(np.sqrt(np.mean((new[ix]-old[ix])**2)))}
 if ci:
  bs=np.random.default_rng(20270323).integers(0,len(ix),(3000,len(ix)));g=np.sqrt(np.mean(a[bs]**2,1))-np.sqrt(np.mean(b[bs]**2,1));r['conditional_paired_gain_CI95']=np.quantile(g,[.025,.975]).tolist();r['bootstrap_sd']=float(g.std())
 return r

def main():
 plan=s.check();v=json.loads((_repo_path(ROOT, 'reports/patch-replication-093-validation.json')).read_text());assert v['passed'] and len(v['results'])==2;ov=json.loads((_repo_path(ROOT, 'reports/patch-attention-092-validation.json')).read_text());assert ov['passed'];allresults={'original':ov['results']['original'],**v['results']};results={};rows={};allg=[];allc=[];allids=None;topids=plan['fixed_original_top5_ids']
 for split in ['original',*s.SPLITS]:
  out=s.result_path(split);rec=allresults[split];assert fv.sha(out/'report.json')==rec['report_sha256'] and fv.sha(out/'development_predictions.npz')==rec['predictions_sha256']
  with np.load(_repo_io(out/'development_predictions.npz'),allow_pickle=False) as z:p={n:z[n].copy() for n in z.files}
  ids=p['ids'].tolist();y=p['y'];old=p['reference011'];new=p['attention_blend'];g=(y-old)**2-(y-new)**2;np.testing.assert_array_equal(g,p['squared_error_gain'])
  if allids is None:allids=ids;all_y=y.copy();topix=np.array([ids.index(i) for i in topids]);rest=np.array([i for i,ident in enumerate(ids) if ident not in set(topids)]);assert len(rest)==395 and len(topix)==5;assert [ids[i] for i in np.argsort(-g,kind='stable')[:5]]==topids
  assert ids==allids;np.testing.assert_array_equal(y,all_y);allg.append(g);allc.append(new-old);individual=[]
  for ident,i in zip(topids,topix):individual.append({'ID':ident,'hardness':float(y[i]),'fold':int(p['fold'][i]),'prediction011':float(old[i]),'prediction_attention_blend':float(new[i]),'squared_error_gain':float(g[i]),'absolute_error_gain':float(abs(y[i]-old[i])-abs(y[i]-new[i])),'improved':bool(g[i]>0)})
  folds=[json.loads((out/f'fold-{k}/report.json').read_text()) for k in range(5)];resource={'total_fit_seconds':sum(a['seconds'] for f in folds for a in f['arms'].values()),'max_RSS_bytes':max(a['rss_peak_bytes'] for f in folds for a in f['arms'].values()),'max_MPS_bytes':max(a['mps_driver_bytes'] for f in folds for a in f['arms'].values())};results[split]={'scores':rec['scores'],'comparisons':rec['comparisons'],'all400':subset(y,old,new,np.arange(400),True),'fixed_original_top5':subset(y,old,new,topix,False),'fixed_other395':subset(y,old,new,rest,True),'fixed_top5_individuals':individual,'resource':resource}
  for i,ident in enumerate(ids):rows.setdefault(ident,{'ID':ident,'in_fixed_original_top5':ident in set(topids)})[split+'_squared_error_gain']=float(g[i]);rows[ident][split+'_prediction_change']=float(new[i]-old[i])
 g=np.array(allg);changes=np.array(allc);counts=(g>0).sum(0);cross={'same400_rows_not_independent':True,'improved_in_n_partitions_all400':{str(k):int((counts==k).sum()) for k in range(4)},'improved_in_n_partitions_other395':{str(k):int((counts[rest]==k).sum()) for k in range(4)},'perID_squared_error_gain_correlations':np.corrcoef(g).tolist(),'perID_prediction_change_correlations':np.corrcoef(changes).tolist(),'fixed_top5_improved_in_both_resplits':[ident for ident,i in zip(topids,topix) if (g[1:,i]>0).all()],'fixed_top5_improved_in_neither_resplit':[ident for ident,i in zip(topids,topix) if (g[1:,i]<0).all()]}
 path=s.OUT/'fixed-id-tracking.csv'
 with path.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[allids[0]]));w.writeheader();w.writerows(rows[i] for i in allids)
 flags={'attention_beats_mean_standalone_all3':all(r['comparisons']['attention_vs_mean']['gain']>0 for r in results.values()),'attention_blend_beats_mean_blend_all3':all(r['comparisons']['attentionblend_vs_meanblend']['gain']>0 for r in results.values()),'attention_blend_beats011_all3':all(r['all400']['RMSE_gain']>0 for r in results.values()),'remaining395_improves_all3':all(r['fixed_other395']['RMSE_gain']>0 for r in results.values()),'remaining395_CI_lower_positive_all3':all(r['fixed_other395']['conditional_paired_gain_CI95'][0]>0 for r in results.values())}
 report={'study':'093 post-hoc fixed patch aggregation replication','selection_bias':plan['selection_bias'],'candidate_policy':plan['candidate_policy'],'config':s.CFG,'original_top5_definition':plan['top5_definition'],'fixed_original_top5_ids':topids,'results':results,'cross_partition_consistency':cross,'descriptive_assessment_flags_not_promotion_gates':flags,'assessment_limits':'Separate evidence for learnedweighting versusmatchedmean from utility beyond011. Originaltop5 wasoutcome-selected andexpectedto regress. Remaining395 always excludes originalsameIDs,notresplit-specifictop5. Conditionalrowbootstrapandfoldcounts do not establish independent replication; repeated400 observations,model-fittingandadaptiveselectionuncertainty remain. Attentionweight doesnot prove physicalimportance. No newarchitecture/setting orcandidate inference allowed.','validation':{k:r for k,r in v.items() if k!='results'},'original092_preserved':True,'original092_validation_sha256':fv.sha(_repo_path(ROOT, 'reports/patch-attention-092-validation.json')),'fixed_id_tracking_csv':str(path.relative_to(ROOT)),'fixed_id_tracking_sha256':fv.sha(path),'protected011_sha256':fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-011.csv')),'scope':{'cost':0,'new_install_or_download':False,'new_cache':False,'holdout_test_selection_access':False,'new_candidate_created':False,'full500_fit':False,'uploaded':False},'evidence_sha256':{n:fv.sha(_repo_path(ROOT, n)) for n in ['artifacts/patch-replication-093/plan.json','reports/patch-replication-093-validation.json','reports/patch-attention-092-validation.json']},'source_sha256':fv.sha(Path(__file__))};s.check();fv.write(_repo_path(ROOT, 'reports/patch-replication-093-summary.json'),report);print(json.dumps({'flags':flags,'results':results,'cross':cross,'no_candidate':True}),flush=True)
if __name__=='__main__':main()
