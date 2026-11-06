from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import ROOT as _REPO_ROOT, resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,hashlib,time,datetime
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[4];R=Path(__file__).resolve().parent
assert P==Path(str(_REPO_ROOT))
assert Path(sys.prefix).resolve()==_repo_path(P, '.venv')
sys.path.insert(0,str(_repo_path(P, 'studies')))
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(_repo_io(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def emit(x):print(json.dumps(x,ensure_ascii=False),flush=True)
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
def manifest(root,file):
 m=read(root/'artifact-manifest.json');key=str(file.relative_to(P));key=key if key in m else str(file.relative_to(root));assert sha(file)==m[key],str(file)
def protect():
 q=read(_repo_path(R, 'plan.json'))
 for p,h in q['protected'].items():assert sha(_repo_path(P, p))==h
 assert sha(__file__)==q['script_sha256']
 return q
def data(split):
 z=load(_repo_path(P, f'experiments/dinol-177/{split}/predictions.npz'));rep=read(_repo_path(P, f'experiments/dinol-177/{split}/report.json'));assert sha(_repo_path(P, f'experiments/dinol-177/{split}/predictions.npz'))==rep['predictions_sha256']
 base=load(_repo_path(P, 'artifacts/run-012/development_features.npz'))
 for k,v in [('ids',base['ids']),('y',base['y']),('g',base['groups'])]:np.testing.assert_array_equal(z[k],v)
 assert len(z['ids'])==len(set(z['ids']))==400
 for k in range(5):assert sum(z['fold']==k)==80 and not set(z['g'][z['fold']==k])&set(z['g'][z['fold']!=k])
 np.testing.assert_array_equal(z['blend_L_MLP'],.9*z['reference018']+.1*z['L_MLP'])
 return z
def prepare():
 assert not (_repo_path(R, 'plan.json')).exists()
 roots=sorted([x for x in (_repo_path(P, 'experiments')).iterdir() if x.name.rsplit('-',1)[-1] in list(map(str,range(205,218)))])
 files=[f for root in roots for f in sorted(root.glob('**/*OOF.npz'))]
 protected={str(f.relative_to(P)):sha(f) for f in files}
 for f in ['artifacts/submission-candidate-021.csv','experiments/candidate021-178/models/manifest.json','denoised_grain_features.py','grain_features.py','experiments/dinol-177/features.npz','artifacts/run-022/development_features.npz']:
  protected[f]=sha(_repo_path(P, f))
 for name,h in read(_repo_path(P, 'experiments/candidate021-178/models/manifest.json'))['files'].items():
  f='experiments/candidate021-178/models/'+name;assert sha(_repo_path(P, f))==h;protected[f]=h
 for split in ['original','20261031','20261101']:
  f=f'experiments/dinol-177/{split}/predictions.npz';protected[f]=sha(_repo_path(P, f))
 z=load(_repo_path(P, 'artifacts/run-012/development_features.npz'))
 write(_repo_path(R, 'plan.json'),{'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'IDs':z['ids'].tolist(),'scope':'Read-only development400 diagnostics; no training, selection, exclusions, holdout/test, network or installations. Source extraction may be rerun to audit measurement, not fit.','L1':{'inventory':'Every saved standalone and fixed-ensemble candidate/control from studies205 through217. Historical studies001-204 not exhaustively reanalysed. Baseline021 independently checked on all3partitions.','residual':'r=y-p021; delta=candidate-p021','A_gain':'2mean(r*delta)','P':'mean(delta^2)','MSE_change':'-lambda*A_gain+lambda^2*P','lambda_range':[0,1],'lambda_grid':[0,.25,.5,.75,1],'oracle':'clip(A_gain/(2P),0,1), retrospective only; never deploy or select weights','paired_losses':'all400 per candidate, plus matched standalone controls; no ranking-based omissions'},'L2':{'representation':'Existing frozenDINOv2L final normalizedCLS, features[:,6144:7168], cosine L2normalized per image; no fitted metric, labels unused','k':[1,5,10],'pairs':'12smallest-distance unique top1 pairs by distance then ID; fixed before labels; native image D4 pixel diagnostics; no duplicate exclusions'},'L3':{'error':'021 OOF SE original and each repeat separately; pooled perID meanSE (not meanprediction ensemble)','deciles':'stable SE ascending,40IDs each','neighbor':'toperror40 directed k5 internal-edge fraction versus40-of400 expectation39/399; descriptive1000fixedcount label permutations seed202610218; not significance or independentvalidation'},'L4':{'extractor':'Current weak denoised_grain_features.py four configs; all400 rawdevelopment, reproduce cached31descriptor values; record all/interior/border/size support, empty0 perconfig and median. No physical grain masks available.','tail':'target bottom/top40 by stable rank; diagnostic only','geometry':'Ideal disks centersuniform within256square, radii4,8,16,32,64; complete inclusion((256-2r)/256)^2. Not groundtruth microstructure or training data.','plots':'labelblind largest median border fraction8 after measurement, fixed before y/residual joins'},'protected':protected,'script_sha256':sha(__file__),'training_fits':0})
 emit({'plan_locked':str(_repo_path(R, 'plan.json')),'OOF_files':len(files)})
def guard():
 ids=set(read(_repo_path(R, 'plan.json'))['IDs']);allowed={_repo_path(P, 'data/train', f'{i}.png') for i in ids}
 def audit(event,args):
  if event=='socket.connect':raise RuntimeError('No network')
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(args[0])).absolute()
   if p.is_relative_to(_repo_path(P, 'data')) and p not in allowed:raise RuntimeError('Nondevelopment raw data forbidden')
 sys.addaudithook(audit)
def L1():
 plan=protect();guard();records=[];arrays={};coverage=[]
 def verify(z,split):
  d=data(split)
  for k in ['ids','y','fold']:np.testing.assert_array_equal(z[k],d[k])
  if 'groups' in z:np.testing.assert_array_equal(z['groups'],d['g'])
  return d
 def add(name,split,p,kind,path,base=None):
  d=data(split);y=d['y'];b=d['blend_L_MLP'] if base is None else base;assert p.shape==(400,) and np.isfinite(p).all();r=y-b;delta=p-b;A=float(2*np.mean(r*delta));pen=float(np.mean(delta**2));opt=float(np.clip(A/(2*pen),0,1)) if pen>0 else 0.;out=[]
  for lam in plan['L1']['lambda_grid']+[opt]:
   change=-lam*A+lam*lam*pen;direct=float(np.mean((y-(b+lam*delta))**2)-np.mean(r*r));assert abs(change-direct)<1e-10
   out.append({'lambda':lam,'MSE_change':change,'RMSE':rmse(y,b+lam*delta)})
  key=f'c{len(records):03d}';arrays[key+'_prediction']=p;arrays[key+'_baseline']=b;arrays[key+'_paired_SE_change']=(y-p)**2-r*r
  folds=[]
  for k in range(5):
   v=d['fold']==k;folds.append({'fold':k,'A_gain':float(2*np.mean(r[v]*delta[v])),'P':float(np.mean(delta[v]**2)),'RMSE_gain':rmse(y[v],b[v])-rmse(y[v],p[v])})
  records.append({'key':key,'name':name,'split':split,'kind':kind,'source':path,'coverage':400,'baseline_RMSE':rmse(y,b),'candidate_RMSE':rmse(y,p),'A_gain':A,'A_MSE_linear':-A,'P':pen,'lambda_nonnegative_oracle':opt,'unconstrained_beneficial_upper':A/pen if pen>0 and A>0 else 0.,'oracle_RMSE_gain':rmse(y,b)-rmse(y,b+opt*delta),'grid':out,'folds':folds,'retrospective_only':True})
 for n in range(205,218):
  root=next(x for x in (_repo_path(P, 'experiments')).iterdir() if x.name.endswith('-'+str(n)))
  rawfiles=sorted(root.glob('**/OOF.npz'));assert rawfiles
  for raw in rawfiles:
   split=raw.parent.name if raw.parent.name in ['20261031','20261101'] else 'original'
   manifest(root,raw);z=load(raw);d=verify(z,split);path=str(raw.relative_to(P));cols=[k for k,v in z.items() if v.shape==(400,) and v.dtype.kind=='f' and k not in ['y','best021']]
   for k in cols:add(f'{n}:{k}',split,z[k],'saved_fixed_ensemble' if k.startswith('fixed_') else 'standalone_direct_vs021_diagnostic',path)
   e=raw.parent/'ensemble-OOF.npz'
   if e.exists():
    manifest(root,e);ez=load(e);verify(ez,split)
    keys=[k for k,v in ez.items() if v.shape==(400,) and v.dtype.kind=='f' and k not in ['y','coefficient','best021'] and (k not in z or not np.array_equal(ez[k],z[k]))]
    for k in keys:add(f'{n}:{k}',split,ez[k],'saved_fixed_ensemble',str(e.relative_to(P)))
   if n==205:
    for k in ['imagenet','micronet']:add(f'{n}:{k}:fixed10percent',split,.9*d['blend_L_MLP']+.1*z[k],'historically_fixed_addition_10percent',path)
   # Paired control-vs-policy diagnostic, distinct from best021 comparison.
   controls={205:('imagenet','micronet'),206:('control','contrast'),207:('uniform','tail_ema'),208:('last5_uniform','update_ema'),209:('last5_uniform','update_ema'),211:('control','huber'),212:('control','huber'),213:('control','brightness'),214:('control','gamma'),215:('control','blur'),216:('control','native'),217:('control','native')}
   if n in controls:
    a,b=controls[n];add(f'{n}:matched:{a}->{b}',split,z[b],'matched_standalone_control',path,z[a])
   coverage.append({'study':n,'split':split,'file':path,'sha256':sha(raw),'all_float_prediction_columns':cols,'ID_y_fold_exact':True,'manifest_exact':True})
 d=data('original');arrays['ids']=d['ids'];arrays['y']=d['y'];np.savez_compressed(_repo_path(R, 'L1-paired-losses.npz'),**arrays)
 # Reproduce specific216 example with consistent signs.
 ex=next(r for r in records if r['name']=='216:fixed_native');prior=read(_repo_path(P, 'experiments/resolution-repartition-217/complementarity-diagnostics.json'))['partitions']['original']['fixed_MSE_gain_decomposition']
 assert abs(ex['A_gain']-prior['alignment_minus2mean_r_delta'])<1e-10 and abs(ex['P']-prior['delta_squared_penalty'])<1e-10
 write(_repo_path(R, 'L1.json'),{'definitions':plan['L1'],'coverage':coverage,'candidates':records,'example216':ex,'provenance_limit':'Checks saved hashes and ID/fold/group alignment plus existing validation artifacts. No training replay in this diagnostic; historical source logs are evidence, not independent refits.','all_tested_candidates_reported':True})
 emit({'L1_candidates':len(records),'example216':{k:ex[k] for k in ['A_gain','P','lambda_nonnegative_oracle','oracle_RMSE_gain']}});protect()
def L2():
 plan=protect();guard();f=_repo_path(P, 'experiments/dinol-177/features.npz');assert sha(f)==read(_repo_path(P, 'experiments/dinol-177/extraction.json'))['feature_sha256'];z=load(f);assert z['ids'].tolist()==plan['IDs'];X=z['X'][:,6144:7168];assert X.shape==(400,1024);X=X/np.linalg.norm(X,axis=1,keepdims=True);dist=np.clip(1-X@X.T,0,2);np.fill_diagonal(dist,np.inf);nn=np.argsort(dist,axis=1,kind='stable')[:,:10];assert np.all(nn!=np.arange(400)[:,None]);edges={tuple(sorted((i,int(nn[i,0])))) for i in range(400)};pairs=sorted(edges,key=lambda ab:(dist[ab],str(z['ids'][ab[0]]),str(z['ids'][ab[1]])))[:12]
 write(_repo_path(R, 'L2-selection-locked.json'),{'pairs':[[str(z['ids'][i]),str(z['ids'][j])] for i,j in pairs],'selection':'cosine distance only; no labels/OOF loaded in L2','k':[1,5,10],'feature_sha256':sha(f)})
 from PIL import Image,ImageDraw
 from scipy.ndimage import gaussian_filter
 hashes=read(_repo_path(P, 'experiments/dinol-177/extraction.json'))['image_sha256'];rows=[];canvas=Image.new('RGB',(512,12*280),'white');draw=ImageDraw.Draw(canvas)
 for t,(i,j) in enumerate(pairs):
  ims=[]
  for idx in [i,j]:
   ident=str(z['ids'][idx]);path=_repo_path(P, 'data/train', f'{ident}.png');assert sha(path)==hashes[ident]
   with Image.open(_repo_io(path)) as im:assert im.mode=='L' and im.size==(256,256);ims.append(np.array(im));canvas.paste(im,(0 if idx==i else 256,t*280+24))
  a,b=ims;af=a.astype(float)/255;bf=b.astype(float)/255;checks=[]
  for k in range(4):
   for flip in [False,True]:
    c=np.rot90(bf,k);c=np.fliplr(c) if flip else c;checks.append({'rotation90':k,'flip':flip,'raw_RMSE':float(np.sqrt(np.mean((af-c)**2))),'raw_corr':float(np.corrcoef(af.ravel(),c.ravel())[0,1]),'blur1_corr':float(np.corrcoef(gaussian_filter(af,1).ravel(),gaussian_filter(c,1).ravel())[0,1]),'exact':bool(np.array_equal(af,c))})
  row={'IDs':[str(z['ids'][i]),str(z['ids'][j])],'cosine_distance':float(dist[i,j]),'mutual_top1':bool(nn[j,0]==i and nn[i,0]==j),'best_raw_D4':min(checks,key=lambda x:x['raw_RMSE']),'all_D4':checks};rows.append(row);draw.text((2,t*280+3),f'{row["IDs"]} d={dist[i,j]:.5f}',fill='black')
 canvas.save(_repo_path(R, 'L2-neighbor-pairs.png'));np.savez_compressed(_repo_path(R, 'L2-neighbors.npz'),ids=z['ids'],indices=nn,distances=np.take_along_axis(dist,nn,axis=1))
 write(_repo_path(R, 'L2.json'),{'representation':plan['L2'],'nearest_distance_quantiles':np.quantile(dist[np.arange(400),nn[:,0]],[0,.1,.5,.9,1]).tolist(),'pairs':rows,'labels_loaded':False,'exclusions':[],'no_duplicate_family_claim':True});emit({'L2_pairs':len(rows),'exact_D4_pairs':sum(r['best_raw_D4']['exact'] for r in rows),'distance_quantiles':np.quantile(dist[np.arange(400),nn[:,0]],[0,.5,1]).tolist()});protect()
def L3():
 plan=protect();guard();neigh=load(_repo_path(R, 'L2-neighbors.npz'));orig=data('original');np.testing.assert_array_equal(orig['ids'],neigh['ids']);nn=neigh['indices'][:,:5];se={split:(data(split)['y']-data(split)['blend_L_MLP'])**2 for split in ['original','20261031','20261101']};se['pooled_meanSE']=np.mean(list(se.values()),axis=0);res={};rng=np.random.default_rng(202610218)
 for split,e in se.items():
  order=np.argsort(e,kind='stable');groups=np.array_split(order,10);top=np.zeros(400,bool);top[order[-40:]]=True;obs=float(top[nn[top]].mean());null=[]
  for _ in range(1000):
   m=np.zeros(400,bool);m[rng.choice(400,40,replace=False)]=True;null.append(float(m[nn[m]].mean()))
  res[split]={'total_SSE':float(e.sum()),'deciles':[{'ascending_decile':k+1,'n':40,'SSE':float(e[ix].sum()),'SSE_share':float(e[ix].sum()/e.sum()),'IDs':orig['ids'][ix].tolist()} for k,ix in enumerate(groups)],'top_decile_k5_internal_edge_fraction':obs,'fixed_size_random_expectation':39/399,'descriptive_permutation_95_interval':np.quantile(null,[.025,.975]).tolist(),'not_selection_adjusted_not_significance_test':True}
 write(_repo_path(R, 'L3.json'),{'results':res,'interpretation':'Groups selected from observed errors are explanatory only. Repartitions reuse same400. Neighbor topology fixed withoutlabels. Concentration does not identify label noise or justify removing rows.'});emit({'L3':{k:{'top40_SSEshare':v['deciles'][-1]['SSE_share'],'neighbor_fraction':v['top_decile_k5_internal_edge_fraction']} for k,v in res.items()}});protect()
def L4_measure():
 plan=protect();guard();from scipy import ndimage as ndi
 from PIL import Image,ImageDraw
 from skimage.filters import threshold_otsu
 from skimage.morphology import h_minima
 from skimage.measure import label,regionprops
 from skimage.segmentation import watershed
 from skimage.restoration import denoise_tv_chambolle
 import denoised_grain_features as df
 old=load(_repo_path(P, 'artifacts/run-022/development_features.npz'));assert old['ids'].tolist()==plan['IDs'];hashes=read(_repo_path(P, 'experiments/dinol-177/extraction.json'))['image_sha256'];rows=[];desc=[];supports=[];t=time.monotonic();maxerr=0
 for i,ident in enumerate(plan['IDs']):
  path=_repo_path(P, 'data/train', f'{ident}.png');assert sha(path)==hashes[ident]
  with Image.open(_repo_io(path)) as im:assert im.mode=='L' and im.size==(256,256);raw=np.array(im)
  x=df._canonical_d4(raw).astype(float)/255;noise=df._noise_sigma(x);stats=[];rr=[];last=None
  for tv in df.TV_MULTIPLIERS:
   den=denoise_tv_chambolle(x,weight=tv*noise,eps=1e-4,max_num_iter=50,channel_axis=None) if noise>1e-12 else x.copy();ridge=ndi.grey_closing(den,size=(7,7),mode='reflect')-den;rm=ridge>threshold_otsu(ridge) if np.ptp(ridge)>1e-12 else np.zeros_like(x,bool);grad=ndi.gaussian_gradient_magnitude(den,.7,mode='reflect');q90=float(np.percentile(grad,90))
   for h in df.PROMINENCE_MULTIPLIERS:
    if np.ptp(grad)<=1e-12:seg=np.ones(x.shape,np.int32)
    else:
     markers=label(h_minima(grad,max(h*q90,1e-12)).astype(bool)&~rm,connectivity=2)
     if not markers.any():markers.flat[int(np.argmin(np.where(~rm,grad,np.inf)))]=1
     seg=watershed(grad,markers,watershed_line=True)
    s=df._summarize(seg,den);stats.append(s);sizes=np.bincount(seg.ravel());labs=np.arange(len(sizes));border=np.unique(np.r_[seg[0],seg[-1],seg[:,0],seg[:,-1]]);large=(sizes>=16)&(labs>0);inside=large&~np.isin(labs,border);edge=large&np.isin(labs,border);small=(sizes<16)&(labs>0);diam=np.sqrt(4*sizes[large]/np.pi);di=np.sqrt(4*sizes[inside]/np.pi);db=np.sqrt(4*sizes[edge]/np.pi);code=np.zeros(len(sizes),np.uint8);code[inside]=1;code[edge]=2;code[small]=3;support=code[seg]
    assert np.isclose(np.mean(support==1),s[13]) and inside.sum()==s[1] and large.sum()==s[0]
    rr.append({'tv':tv,'h':h,'empty0':bool(not inside.any()),'all_count':int(large.sum()),'inside_count':int(inside.sum()),'border_count':int(edge.sum()),'border_area_fraction':float(np.mean(support==2)),'small_area_fraction':float(np.mean(support==3)),'interior_area_fraction':float(np.mean(support==1)),'watershed_line_fraction':float(np.mean(support==0)),'all_diameter_mean':float(diam.mean()) if diam.size else 0.,'interior_diameter_mean':float(di.mean()) if di.size else 0.,'border_observed_diameter_mean':float(db.mean()) if db.size else 0.});last=support
  stats=np.asarray(stats);fv=np.r_[np.column_stack((np.median(stats,axis=0),np.ptp(stats,axis=0))).ravel(),noise];err=float(np.max(abs(fv-old['X'][i,-31:])));assert err<1e-12;maxerr=max(maxerr,err)
  rows.append({'ID':ident,'configs':rr,'noise':noise});desc.append(fv);supports.append(last)
  if (i+1)%80==0:emit({'L4_images':i+1,'seconds':time.monotonic()-t})
 np.savez_compressed(_repo_path(R, 'L4-measurements.npz'),ids=old['ids'],descriptors=np.array(desc),support=np.array(supports));write(_repo_path(R, 'L4-measurements.json'),{'rows':rows,'cache_reconstruction_max_error':maxerr,'seconds':time.monotonic()-t,'labels_loaded':False,'definition':'Zero means no eligible interior regions, not physical diameter0; observed border diameters are clipped region measurements.'});emit({'L4_measure_complete':len(rows),'seconds':time.monotonic()-t,'cache_maxerr':maxerr});protect()
def L4_analyse():
 plan=protect();guard();from scipy.stats import spearmanr
 from PIL import Image,ImageDraw
 m=read(_repo_path(R, 'L4-measurements.json'));rows=m['rows'];z=data('original');assert [r['ID'] for r in rows]==z['ids'].tolist();features={}
 for key in ['border_area_fraction','small_area_fraction','interior_area_fraction','all_count','inside_count','all_diameter_mean','interior_diameter_mean','border_observed_diameter_mean']:
  features[key]=np.median([[s[key] for s in r['configs']] for r in rows],axis=1)
 empty=np.array([[s['empty0'] for s in r['configs']] for r in rows]);features['empty_configs']=empty.sum(1);features['diameter_range']=np.ptp([[s['interior_diameter_mean'] for s in r['configs']] for r in rows],axis=1);features['noise']=np.array([r['noise'] for r in rows]);e=np.mean([(data(s)['y']-data(s)['blend_L_MLP'])**2 for s in ['original','20261031','20261101']],axis=0);y=z['y'];r=y-z['blend_L_MLP'];order=np.argsort(y,kind='stable');tail_masks={'low40':np.isin(np.arange(400),order[:40]),'middle320':np.isin(np.arange(400),order[40:-40]),'high40':np.isin(np.arange(400),order[-40:]),'any_empty':empty.any(1),'all_empty':empty.all(1)}
 corrs={}
 for k,v in features.items():
  corrs[k]={key:float(spearmanr(v,t).statistic) if np.ptp(v)>0 else None for key,t in [('target',y),('signed_residual_original',r),('pooled_SE',e)]}
 def group(mask):
  return {'n':int(mask.sum()),'IDs':z['ids'][mask].tolist(),'pooled_RMSE':float(np.sqrt(e[mask].mean())) if mask.any() else None,'mean_residual_original':float(r[mask].mean()) if mask.any() else None,'feature_medians':{k:float(np.median(v[mask])) if mask.any() else None for k,v in features.items()}}
 groups={k:group(v) for k,v in tail_masks.items()}
 for key in ['border_area_fraction','interior_diameter_mean','diameter_range']:
  ix=np.argsort(features[key],kind='stable')
  for j,inds in enumerate(np.array_split(ix,4)):groups[key+f'_quartile{j+1}']=group(np.isin(np.arange(400),inds))
 # label-blind support panel selection defined before viewing labels in plan; recompute solely from borderfraction.
 ix=np.argsort(-features['border_area_fraction'],kind='stable')[:8];write(_repo_path(R, 'L4-panel-selection.json'),{'IDs':z['ids'][ix].tolist(),'basis':'largest median measured border fraction, no outcome selection'});cached=load(_repo_path(R, 'L4-measurements.npz'));colors=np.array([[0,0,0],[0,180,90],[255,140,20],[180,0,200]],np.uint8);canvas=Image.new('RGB',(512,280*8),'white');draw=ImageDraw.Draw(canvas)
 import denoised_grain_features as df
 for row,i in enumerate(ix):
  with Image.open(_repo_path(P, 'data/train', f'{z["ids"][i]}.png')) as im:a=df._canonical_d4(np.array(im))
  canvas.paste(Image.fromarray(a),(0,280*row+24));canvas.paste(Image.fromarray(colors[cached['support'][i]]),(256,280*row+24));draw.text((2,280*row+3),f'{z["ids"][i]} border median={features["border_area_fraction"][i]:.3f}',fill='black')
 canvas.save(_repo_path(R, 'L4-border-support.png'));geometry=[]
 yy,xx=np.mgrid[:256,:256]
 for rad in [4,8,16,32,64]:
  vals=[]
  for cx,cy in [(128,128),(rad//2,128),(rad//2,rad//2)]:
   disk=(xx-cx)**2+(yy-cy)**2<=rad*rad;touch=bool(disk[0].any() or disk[-1].any() or disk[:,0].any() or disk[:,-1].any());vals.append({'center':[cx,cy],'touches_border':touch,'observed_equivalent_diameter':float(np.sqrt(4*disk.sum()/np.pi)),'retained_by_complete_object_filter':not touch})
  geometry.append({'radius':rad,'ideal_complete_inclusion_probability_center_uniform_inside_window':((256-2*rad)/256)**2,'fixtures':vals})
 write(_repo_path(R, 'L4.json'),{'empty0':{'any_config_n':int(empty.any(1).sum()),'all_configs_n':int(empty.all(1).sum()),'per_config_n':empty.sum(0).tolist(),'median_interior_count_zero_n':int((features['inside_count']==0).sum())},'feature_quantiles':{k:np.quantile(v,[0,.1,.5,.9,1]).tolist() for k,v in features.items()},'descriptive_spearman':corrs,'groups':groups,'ideal_disk_geometry':geometry,'cache_max_error':m['cache_reconstruction_max_error'],'limitations':['No segmentation groundtruth or physical size calibration. Border exclusion can cause geometric size sampling bias but cannot calibrate actual grain size without validity assumptions.','Syntheticdisks are algorithm/geometry fixtures only, never trainingdata.','No 1percent or correlation.98 gate; low frequency alone is not evidence of no branch utility.','Tail/error groups are posthoc descriptive and not independent generalization evidence.']});emit({'L4_empty':{'any':int(empty.any(1).sum()),'all':int(empty.all(1).sum())},'border_median':float(np.median(features['border_area_fraction'])),'corr':corrs});protect()
if __name__=='__main__':globals()[sys.argv[1]]()
