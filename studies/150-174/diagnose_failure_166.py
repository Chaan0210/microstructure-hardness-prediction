"""Posthoc development-only case review of018; no fit and no heldout/test access."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,csv,hashlib,itertools
import numpy as np
from PIL import Image,ImageDraw
from scipy.stats import spearmanr
R=_RepoPath(__file__).resolve().parents[1];O=_repo_path(R, 'artifacts/failure-review-166');SPLITS=['original','20261031','20261101']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def readnp(p,expected=None):
 h=sha(p)
 if expected:assert h==expected
 sources[str(p.relative_to(R))]=h
 with np.load(_repo_io(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
sources={}
O.mkdir(exist_ok=False)
write(_repo_path(O, 'analysis-plan.json'),{'purpose':'Explore persistent018 errors, stable signs, branchdisagreement before proposing interventions. No new training in this phase.','large':'Top40 absolute errors separately in each of3existing splits; persistentlarge=intersectionall3','stable_sign':'Same sign in3splits AND minabsoluteerror>=5; also report allsame-sign count','disagreement':'Range of raw KRR,CNNmean3,DINOmean(Smean3+Bmean3)/2; top40 per split, intersectionall3. Includes raw component calibration differences.','bins':'Fixedquartiles of400 development label/imageproxy values, descriptive only; report400uniqueIDs, three residuals each, no iid1200 inference.','visual':'Outcome-selected panels of intersections, persistenterror without highdisagreement, highdisagreement without persistentlarge, and lowerror image-matched controls. Original256 grayscale, no contrast adjustment.','bias':'Explicit posthoc case selection on same400 previouslyadaptivelyreused; any motivated model gain remains exploratory, including resplits. No causal physics/noise-floor claim.','scope':'Only existingdevelopmentOOF/cachedproxies and allowlisteddevelopment rawimage samples;noholdout/test.'})
P=[];C=[];F=[]
for split in SPLITS:
 path=_repo_path(R, f'artifacts/bootstrap-selector-144/{split}');rec=json.loads((path/'report.json').read_text());z=readnp(path/'predictions.npz',rec['predictions_sha256'])
 if not P:ids=z['ids'];y=z['y']
 else:np.testing.assert_array_equal(z['ids'],ids);np.testing.assert_array_equal(z['y'],y)
 P.append(z['primary']);F.append(z['fold'])
 path=_repo_path(R, f'artifacts/component-ablation-121/{split}');rec=json.loads((path/'report.json').read_text());c=readnp(path/'predictions.npz',rec['predictions_sha256']);np.testing.assert_array_equal(c['ids'],ids);np.testing.assert_array_equal(c['y'],y)
 cnn=readnp(_repo_path(R, f'artifacts/cnn-seed-mean-133/audit/{split}.npz'));np.testing.assert_array_equal(cnn['ids'],ids)
 C.append(np.column_stack([c['components'][:,1],cnn['mean3'],.5*(c['components'][:,3]+c['components'][:,4])]))
P=np.stack(P);C=np.stack(C);F=np.stack(F);E=y-P;mean=E.mean(0);mae=abs(E).mean(0);spread=np.ptp(C,axis=2)
L=np.zeros((3,400),bool);D=L.copy()
for k in range(3):L[k,np.argsort(-abs(E[k]),kind='stable')[:40]]=True;D[k,np.argsort(-spread[k],kind='stable')[:40]]=True
large=L.all(0);disagree=D.all(0);samesign=(E.min(0)>0)|(E.max(0)<0);stable=samesign&(abs(E).min(0)>=5)
flags={'persistent_large':large,'stable_sign_min5':stable,'persistent_disagreement':disagree,'large_stable_disagreement':large&stable&disagree,'large_stable_without_disagreement':large&stable&~disagree,'disagreement_without_large':disagree&~large,'sign_changes':~samesign}
metricpath=_repo_path(R, 'artifacts/image-diagnostics-094/image_metrics.npz');lock=json.loads(metricpath.with_name('matching-locked.json').read_text());m=readnp(metricpath,lock['metrics_sha256']);np.testing.assert_array_equal(m['ids'],ids)
f=readnp(_repo_path(R, 'artifacts/run-022/development_features.npz'));np.testing.assert_array_equal(f['ids'],ids);fn=json.loads((_repo_path(R, 'artifacts/run-022/report.json')).read_text())['features']
vars={'target':y}
for name in ['image_mean','image_std','noise_contrast_ratio','spectral_low_fraction','anisotropy1','tile_mean_std','clipped_fraction','dark_largest_fraction']:vars[name]=m['X'][:,m['names'].tolist().index(name)]
for name in ['inverse_sqrt_grain_diameter','grain_size_cv','phase_dark','phase_light','pore_fraction15','grain_aspect','grain_alignment','boundary_fraction']:vars[name]=f['X'][:,fn.index(name)]
def stats(mask):
 return {'n_unique':int(mask.sum()),'mean_target':float(y[mask].mean()),'RMSE_by_split':np.sqrt(np.mean(E[:,mask]**2,axis=1)).tolist(),'mean_y_minus_prediction_by_split':E[:,mask].mean(1).tolist(),'pooled_RMSE':float(np.sqrt(np.mean(E[:,mask]**2))),'mean_bias':float(mean[mask].mean()),'branch_span_mean':float(spread[:,mask].mean()),'persistent_large_n':int((mask&large).sum()),'stable_sign_min5_n':int((mask&stable).sum()),'disagreement_n':int((mask&disagree).sum())} if mask.any() else {'n_unique':0}
conditions={}
for name,value in vars.items():
 cuts=np.quantile(value,[.25,.5,.75]);bins=np.searchsorted(cuts,value,side='right');conditions[name]={'cuts':cuts.tolist(),'bins':[stats(bins==b) for b in range(4)],'spearman_mean_abs_error':float(spearmanr(value,mae).statistic),'spearman_mean_signed_error':float(spearmanr(value,mean).statistic)}
records=[]
for i,ident in enumerate(ids):records.append({'ID':str(ident),'target':float(y[i]),'predictions':P[:,i].tolist(),'residuals':E[:,i].tolist(),'mean_bias':float(mean[i]),'mean_abs_error':float(mae[i]),'branch_span':spread[:,i].tolist(),'branches_KRR_CNN_DINOmean':C[:,i,:].tolist(),**{k:bool(v[i]) for k,v in flags.items()},**{k:float(v[i]) for k,v in vars.items() if k!='target'}})
ordered=np.argsort(-mae,kind='stable')
sets={k:{'IDs':ids[v].tolist(),**stats(v)} for k,v in flags.items()}
# Find visually comparable low-error controls by previously cached image-only distance.
sel=[];categories=[]
for label,mask,n in [('large_stable_disagreement',large&stable&disagree,4),('large_stable_without_disagreement',large&stable&~disagree,6),('disagreement_without_large',disagree&~large,4)]:
 ix=[int(i) for i in ordered if mask[i]][:n];sel+=ix;categories +=[label]*len(ix)
for sign in [1,-1]:
 ix=[int(i) for i in ordered if stable[i] and np.sign(mean[i])==sign and i not in sel][:3];sel+=ix;categories+=['positive_bias' if sign==1 else 'negative_bias']*len(ix)
anchors=sel[:8];available=set(np.flatnonzero(mae<=np.median(mae)).tolist())-set(sel);pairs=[]
for i in anchors:
 j=min(available,key=lambda j:(m['distance'][i,j],str(ids[j])));available.remove(j);sel.append(j);categories.append('lowerror_matched_control');pairs.append({'anchor':str(ids[i]),'control':str(ids[j]),'image_distance':float(m['distance'][i,j]),'anchor_bias':float(mean[i]),'control_bias':float(mean[j])})
visual=[]
for page,start in enumerate(range(0,len(sel),12)):
 canvas=Image.new('RGB',(4*276,3*328),'white');dr=ImageDraw.Draw(canvas)
 for slot,(i,label) in enumerate(zip(sel[start:start+12],categories[start:start+12])):
  path=_repo_path(R, 'data/train', f'{ids[i]}.png');assert str(ids[i]) in lock['image_sha256'];assert sha(path)==lock['image_sha256'][str(ids[i])]
  with Image.open(_repo_io(path)) as im:assert im.size==(256,256);raw=im.convert('RGB');canvas.paste(raw,((slot%4)*276+10,(slot//4)*328+58))
  xx=(slot%4)*276+10;yy=(slot//4)*328+5;dr.text((xx,yy),f'{ids[i]} y={y[i]:.1f} bias={mean[i]:+.1f}',fill='black');dr.text((xx,yy+15),f'err={E[0,i]:+.1f}/{E[1,i]:+.1f}/{E[2,i]:+.1f}',fill='black');dr.text((xx,yy+30),label,fill='black');visual.append({'ID':str(ids[i]),'category':label,'image_sha256':sha(path),'page':page})
 canvas.save(_repo_path(O, f'cases-{page}.png'))
np.savez_compressed(_repo_path(O, 'analysis.npz'),ids=ids,y=y,predictions=P,residuals=E,components=C,folds=F,branch_span=spread,**flags)
with (_repo_path(O, 'cases.csv')).open('x',newline='') as f:
 fields=['ID','target','mean_bias','mean_abs_error',*flags,*[k for k in vars if k!='target']];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)
report={'study':166,'n_unique':400,'OOF_RMSE':np.sqrt(np.mean(E**2,axis=1)).tolist(),'same_sign_count':int(samesign.sum()),'sets':sets,'intersections':{f'{a}&{b}':int((flags[a]&flags[b]).sum()) for a,b in itertools.combinations(['persistent_large','stable_sign_min5','persistent_disagreement'],2)},'top20':[records[i] for i in ordered[:20]],'conditions':conditions,'visual_selection':visual,'matched_controls':pairs,'sources':sources,'exploratory':True,'no_training':True,'holdout_test_read':False,'raw_images_opened':len(set(sel)),'conditional_bin_note':'Quartiles across400 are descriptive outcome-review strata only, not model training transforms. Threeerrors shareoneID; no1200-independent-row interpretation.'}
write(_repo_path(O, 'report.json'),report);print(json.dumps({'sets':{k:v['n_unique'] for k,v in sets.items()},'same_sign':int(samesign.sum()),'top12':[(str(ids[i]),round(float(y[i]),2),np.round(E[:,i],2).tolist(),np.round(spread[:,i],1).tolist()) for i in ordered[:12]],'visual_count':len(sel),'strong_abs_associations':sorted([(k,v['spearman_mean_abs_error']) for k,v in conditions.items()],key=lambda z:-abs(z[1]))[:8]},indent=2),flush=True)
