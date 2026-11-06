from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
from common import R,P,np,torch,Image,ImageDraw
c.check();records=[];allids=[]
with c.Monitor('preflight',300):
 ids,x=c.images();quality=[]
 for k in range(5):
  T,V=c.indices(k);a=x[T];C,coef,cal=c.calibration(a);b=c.transform(a,coef);q=np.quantile(b,[.05,.95],axis=(1,2,3),method='linear');Cp=q[1]-q[0]
  meanerr=float(abs(a.mean((1,2,3),dtype=np.float64)-b.mean((1,2,3),dtype=np.float64)).max());ratioerr=float(abs(Cp[C>0]/C[C>0]-coef[C>0]).max());graderr=max(float(abs(np.diff(b,axis=ax)-coef[:,None,None,None]*np.diff(a,axis=ax)).max()) for ax in [2,3]);ref=a.astype(float).mean((1,2,3),keepdims=True)+coef.astype(float)[:,None,None,None]*(a.astype(float)-a.astype(float).mean((1,2,3),keepdims=True));referr=float(abs(ref-b).max());mismatch=0;rank=True;ties=True;sign=True
  for j in range(320):
   flat=a[j].ravel();bf=b[j].ravel();uv,iv=np.unique(flat,return_inverse=True);first=np.unique(flat,return_index=True)[1];mapped=bf[first];rank=rank and bool(np.all(np.diff(mapped)>0));ties=ties and bool(np.array_equal(bf,mapped[iv]))
   for qt in [.1,.5,.9]:mismatch+=int(np.count_nonzero((flat<=np.quantile(flat,qt))!=(bf<=np.quantile(bf,qt))))
  for ax in [2,3]:sign=sign and bool(np.array_equal(np.sign(np.diff(a,axis=ax)),np.sign(np.diff(b,axis=ax))))
  gate=all([meanerr<=1e-6,ratioerr<=1e-6,graderr<=1e-6,referr<=1e-6,rank,ties,sign,mismatch==0,b.min()>=0,b.max()<=1,Cp.min()>=C.min()-1e-6,Cp.max()<=C.max()+1e-6,bool(np.all(Cp[C>cal['Q10']]>=cal['Q10']-1e-6))])
  vis=sorted(T,key=lambda j:__import__('hashlib').sha256(f'206-visual:{k}:{ids[j]}'.encode()).digest())[:4];allids+=ids[vis].tolist();canvas=Image.new('RGB',(800,4*285+34),'white');draw=ImageDraw.Draw(canvas);draw.text((10,8),f'Fold {k} | Original | Contrast | 20x difference (midgray=0)',fill='black')
  for row,j in enumerate(vis):
   local=int(np.flatnonzero(T==j)[0]);original=a[local,0];changed=b[local,0];draw.text((10,36+row*285),f'{ids[j]}  c={coef[local]:.7f}',fill='black')
   for col,arr in enumerate([original,changed,.5+20*(changed-original)]):
    # Quantization below is display-only, never passed to training.
    im=Image.fromarray(np.rint(np.clip(arr,0,1)*255).astype(np.uint8)).resize((256,256),Image.Resampling.NEAREST);canvas.paste(im,(8+col*264,56+row*285))
  canvas.save(_repo_io(R/'figures'/f'fold-{k}.png'))
  c.arrays(R/f'calibration-fold-{k}.npz',train_indices=T,contrast=C,coefficients=coef,transformed=b,transformed_contrast=Cp)
  rec={'fold':k,**cal,'passed':gate,'mean_error':meanerr,'contrast_ratio_error':ratioerr,'gradient_error':graderr,'float64_error':referr,'strict_rank':rank,'ties_exact':ties,'gradient_sign_and_zero_support_exact':sign,'rank_mask_mismatches':mismatch,'min_transformed':float(b.min()),'max_transformed':float(b.max()),'visual_IDs':ids[vis].tolist(),'original_train_SHA256':c.ah(a),'transformed_train_SHA256':c.ah(b),'calibration_SHA256':c.sha(R/f'calibration-fold-{k}.npz')};records.append(rec);c.emit(rec)
 # No targets or predictions were loaded for the structure gate.
 report={'numeric_structure_gate_passed':all(z['passed'] for z in records),'folds':records,'visual_review_pending':True,'visual_unique_images':len(set(allids)),'target_or_prediction_arrays_loaded':False,'new_raw_images_read':0,'new_training':0,'torch':torch.__version__,'numpy':np.__version__,'device':c.device,'meaning':'영상 연산 구조 보존이며 타깃 불변성 증명 아님','plan_SHA256':c.sha(R/'plan.json')};c.write(R/'structure-preflight.json',report);c.emit({'numeric_structure_gate_passed':report['numeric_structure_gate_passed']});assert report['numeric_structure_gate_passed'],'Stop before learning; no strength rescue'
c.check()
