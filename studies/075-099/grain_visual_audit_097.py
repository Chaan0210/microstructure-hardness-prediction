"""Label-blind raw-image/actual watershed inspection; development IDs only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,time,hashlib,csv,datetime
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));assert Path(sys.prefix).resolve()==_repo_path(ROOT, '.venv')
import numpy as np
from scipy import ndimage as ndi
from PIL import Image,ImageDraw,ImageFont
from skimage.filters import threshold_otsu
from skimage.morphology import h_minima
from skimage.measure import label,regionprops
from skimage.segmentation import watershed
from skimage.restoration import denoise_tv_chambolle
from skimage.feature import match_template
import denoised_grain_features as weak
import strong_denoised_grain_features as strong
OUT=_repo_path(ROOT, 'artifacts/grain-visual-audit-097');PAIR=['TRAIN_000396','TRAIN_000497']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def raw(ident):
 with Image.open(_repo_path(ROOT, 'data/train', f'{ident}.png')) as im:
  assert im.size==(256,256) and im.mode=='L';return np.array(im)
def canonical(a):
 opts=[(np.ascontiguousarray(z),k,flip) for k in range(4) for flip,z in [(False,np.rot90(a,k)),(True,np.fliplr(np.rot90(a,k)))]]
 return min(opts,key=lambda z:z[0].tobytes())
def prepare():
 OUT.mkdir(exist_ok=False);(_repo_path(OUT, 'panels')).mkdir();(_repo_path(OUT, 'arrays')).mkdir();m=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text());ids=sorted(m['development_folds']);assert len(ids)==400 and not set(ids)&set(m['holdout']);assert set(PAIR)<=set(ids)
 p={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Only400 development raw images; no hardness/OOF/holdout/test loading. Pair IDs supplied externally are diagnostic, not representative or candidate selection.','selection':'All400: weak-extractor HH MAD noise on canonicalraw and raw q95-q05 contrast. Noise tertiles x contrast halves (numeric quantiles),2 IDs percell by smallest sha256(097-label-blind-v1:+ID). Fixed12 before viewing overlays. Add suppliedpair separately. No outcome-based substitution.','panel':'CanonicalD4 view shared by raw,all4weak,all4strong; native1:1px, fixed0..255 display. Fullpanels include allconfigs. Overview endpoints TV1,h.2(weak) and TV1,h.6(strong) are fixed, not median segmentation. Supportmap:green interior>=16;orange border>=16;magenta small<16;black watershedline.','definitions':'Strong changes only prominence .1/.2 -> .3/.6; TV weights .5/1noise unchanged. Perdescriptor median and range=max-min over4configs. Count filters area>=16; interior also excludes any bbox touching image frame. boundaryfraction counts segmentation==0, not observed physical dark layer.','source_sha256':{p:sha(_repo_path(ROOT, p)) for p in ['grain_visual_audit_097.py','denoised_grain_features.py','strong_denoised_grain_features.py']},'protected011':json.loads((_repo_path(ROOT, 'reports/candidate011-public-submission.json')).read_text())['protected_sha256'],'pair_geometry':'Report decoded sizes/modes/PNG info; raw pixel and Gaussian1 D4 correspondence. Fixed128x128center template of396 versus497 under D4 and scale factors .75,1,4/3 via normalized template correlation; diagnostic only, not proof of physical magnification/crop.'}
 for n,h in p['protected011'].items():assert sha(_repo_path(ROOT, n))==h
 write(_repo_path(OUT, 'plan.json'),p);rows=[]
 for ident in ids:
  a=raw(ident);c,_,_=canonical(a);x=c.astype(float)/255.;rows.append({'ID':ident,'noise':weak._noise_sigma(x),'contrast_q95_q05':float(np.diff(np.percentile(x,[5,95]))[0]),'image_sha256':sha(_repo_path(ROOT, 'data/train', f'{ident}.png'))})
 noise=np.array([r['noise'] for r in rows]);contrast=np.array([r['contrast_q95_q05'] for r in rows]);nt=np.quantile(noise,[1/3,2/3]);ct=float(np.median(contrast));chosen=[];strata=[]
 for i in range(3):
  for j in range(2):
   pool=[r for r in rows if np.searchsorted(nt,r['noise'],side='right')==i and int(r['contrast_q95_q05']>=ct)==j];assert len(pool)>=2
   pool.sort(key=lambda r:hashlib.sha256(('097-label-blind-v1:'+r['ID']).encode()).hexdigest());take=pool[:2]
   for r in take:r['stratum']=f'noise{i+1}-contrast{j+1}'
   chosen.extend(take);strata.append({'noise_bin':i+1,'contrast_bin':j+1,'available':len(pool),'selected':[r['ID'] for r in take]})
 with (_repo_path(OUT, 'all400_noise_contrast.csv')).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['ID','noise','contrast_q95_q05','image_sha256'],extrasaction='ignore');w.writeheader();w.writerows(rows)
 write(_repo_path(OUT, 'selection-locked.json'),{'noise_cuts':nt.tolist(),'contrast_cut':ct,'selected':chosen,'strata':strata,'pair':PAIR,'label_access':False,'plan_sha256':sha(_repo_path(OUT, 'plan.json'))});print(json.dumps({'selected':[r['ID'] for r in chosen],'noise_cuts':nt.tolist(),'contrast_cut':ct}),flush=True)

def segment(a):
 c,k,f=canonical(a);x=c.astype(float)/255.;noise=weak._noise_sigma(x);out=[]
 for tv in [.5,1.]:
  d=denoise_tv_chambolle(x,weight=tv*noise,eps=1e-4,max_num_iter=50,channel_axis=None) if noise>1e-12 else x.copy();ridge=ndi.grey_closing(d,size=(7,7),mode='reflect')-d;rm=ridge>threshold_otsu(ridge) if np.ptp(ridge)>1e-12 else np.zeros_like(x,bool);grad=ndi.gaussian_gradient_magnitude(d,.7,mode='reflect');q90=float(np.percentile(grad,90))
  for h in [.1,.2,.3,.6]:
   if np.ptp(grad)<=1e-12:seg=np.ones(x.shape,np.int32);marks=np.ones_like(seg)
   else:
    mins=h_minima(grad,max(h*q90,1e-12)).astype(bool)&~rm;marks=label(mins,connectivity=2)
    if not marks.any():marks.flat[int(np.argmin(np.where(~rm,grad,np.inf)))]=1
    seg=watershed(grad,marks,watershed_line=True)
   rs=regionprops(seg);support=np.zeros_like(seg,np.uint8)
   for r in rs:
    code=3 if r.area<16 else (2 if 0 in r.bbox[:2] or 256 in r.bbox[2:] else 1);support[seg==r.label]=code
   frac=[float(np.mean(support==i)) for i in range(4)];assert abs(sum(frac)-1)<1e-12
   stats=weak._summarize(seg,d);boundary=seg==0;interior=support==1
   out.append({'tv':tv,'h':h,'family':'weak' if h<.3 else 'strong','seg':seg,'support':support,'denoised':d,'gradient':grad,'ridge':rm,'markers':marks,'summary':stats,'record':{'tv':tv,'h':h,'absolute_h':max(h*q90,1e-12),'gradient_q90':q90,'marker_count':int(marks.max()),'grain_count':stats[0],'interior_count':stats[1],'boundary_fraction':frac[0],'interior_area_fraction':frac[1],'border_area_fraction':frac[2],'small_area_fraction':frac[3],'boundary_on_dark_ridge_fraction':float(rm[boundary].mean()) if boundary.any() else None,'boundary_gradient_mean':float(grad[boundary].mean()) if boundary.any() else None,'nonboundary_gradient_mean':float(grad[~boundary].mean())}})
 return c,k,f,out
FONT=ImageFont.load_default(size=14)
def tile(img,text):
 pic=Image.new('RGB',(256,292),'white');pic.paste(Image.fromarray(img).convert('RGB'),(0,36));ImageDraw.Draw(pic).text((3,3),text,fill='black',font=FONT);return pic
def overlay(c,b):
 z=np.repeat(c[:,:,None],3,2);z[b]=[255,50,35];return z
COLORS=np.array([[0,0,0],[40,200,90],[245,145,40],[210,30,210]],np.uint8)
def panels():
 p=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,h in p['source_sha256'].items():assert sha(_repo_path(ROOT, n))==h,n
 locked=json.loads((_repo_path(OUT, 'selection-locked.json')).read_text());selection=locked['selected'];ids=list(dict.fromkeys([r['ID'] for r in selection]+PAIR));records={};thumb=[]
 cache={}
 for run in ['run-022','run-025']:
  with np.load(_repo_path(ROOT, f'artifacts/{run}/development_features.npz'),allow_pickle=False) as z:cache[run]=(z['ids'].tolist(),z['X'].copy())
 for ix,ident in enumerate(ids):
  a=raw(ident);c,k,f,outs=segment(a);wr=[r for r in outs if r['family']=='weak'];sr=[r for r in outs if r['family']=='strong'];err={}
  for family,rr,module,run in [('weak',wr,weak,'run-022'),('strong',sr,strong,'run-025')]:
   mat=np.array([r['summary'] for r in rr]);feat=np.r_[np.column_stack([np.median(mat,axis=0),np.ptp(mat,axis=0)]).ravel(),weak._noise_sigma(c.astype(float)/255.)];old=cache[run][1][cache[run][0].index(ident),-31:];np.testing.assert_allclose(feat,old,rtol=0,atol=1e-12);err[family]=float(abs(feat-old).max())
  canv=Image.new('RGB',(1280,916),'#dddddd');d=ImageDraw.Draw(canv);d.text((4,3),f'{ident} | canonical rotation={k*90}, flip={f} | native1:1, fixed0..255 | all actual configs',fill='black',font=FONT)
  top=[tile(c,'RAW canonical'),tile(np.clip(wr[0]['denoised']*255,0,255).astype(np.uint8),'TV .5*noise (both arms)'),tile(np.clip(wr[2]['denoised']*255,0,255).astype(np.uint8),'TV 1*noise (both arms)'),tile(COLORS[wr[3]['support']],'Weak support: TV1,h.2'),tile(COLORS[sr[3]['support']],'Strong support: TV1,h.6')]
  for j,z in enumerate(top):canv.paste(z,(j*256,40))
  for row,rr,title in [(1,wr,'WEAK'),(2,sr,'STRONG')]:
   for j,z in enumerate(rr):canv.paste(tile(overlay(c,z['seg']==0),f'{title} TV{z["tv"]} h{z["h"]}\nN={z["summary"][0]:.0f} interior={z["summary"][13]:.3f}'),(j*256,40+row*292))
   legend=Image.new('RGB',(256,292),'white');dr=ImageDraw.Draw(legend);dr.multiline_text((8,16),'Support colors:\ngreen: interior >=16px\norange: frame-touch >=16px\nmagenta: regions <16px\nblack: watershed line\n\nRed overlays are watershed\nlines, NOT ground truth.',fill='black',font=FONT,spacing=10);canv.paste(legend,(1024,40+row*292))
  canv.save(_repo_path(OUT, 'panels', f'{ident}.png'))
  overview=Image.new('RGB',(1024,316),'white');dr=ImageDraw.Draw(overview);stratum=next((r['stratum'] for r in selection if r['ID']==ident),'specified pair');dr.text((4,2),f'{ident} | {stratum}',fill='black',font=FONT)
  for j,z in enumerate([tile(c,'RAW'),tile(overlay(c,wr[3]['seg']==0),f'Weak TV1 h.2 | N{wr[3]["summary"][0]:.0f}'),tile(overlay(c,sr[3]['seg']==0),f'Strong TV1 h.6 | N{sr[3]["summary"][0]:.0f}'),tile(COLORS[sr[3]['support']],f'Strong support {sr[3]["summary"][13]:.3f}')]):overview.paste(z,(256*j,24))
  if ix<12:thumb.append(overview)
  np.savez_compressed(_repo_path(OUT, 'arrays', f'{ident}.npz'),raw=a,canonical=c,seg=np.stack([z['seg'] for z in outs]),support=np.stack([z['support'] for z in outs]),config=np.array([[z['tv'],z['h']] for z in outs]),denoised=np.stack([z['denoised'] for z in outs]),gradient=np.stack([z['gradient'] for z in outs]))
  records[ident]={'canonical_rot90':k,'canonical_flip':f,'cached_feature_max_error':err,'configs':[z['record'] for z in outs],'image_sha256':sha(_repo_path(ROOT, 'data/train', f'{ident}.png')),'panel_sha256':sha(_repo_path(OUT, 'panels', f'{ident}.png'))};print(json.dumps({'completed':ix+1,'ID':ident,'weak_strong_cache_errors':err}),flush=True)
 for j in range(2):
  canvas=Image.new('RGB',(1024,316*6),'white')
  for i,img in enumerate(thumb[j*6:(j+1)*6]):canvas.paste(img,(0,i*316))
  canvas.save(_repo_path(OUT, f'overview-{j+1}.png'))
 write(_repo_path(OUT, 'segmentation-records.json'),records)
 # Pair comparisons use no labels. Original orientation, fixed contrast, no resizing in raw panel.
 a,b=[raw(i) for i in PAIR];canvas=Image.new('RGB',(512,584),'white')
 for j,z in enumerate([a,b]):
  canvas.paste(tile(z,PAIR[j]+' | raw native256'),(256*j,0));canvas.paste(tile(np.clip(ndi.gaussian_filter(z.astype(float),1),0,255).astype(np.uint8),'Gaussian sigma1 (diagnostic)'),(256*j,292))
 canvas.save(_repo_path(OUT, 'pair-raw.png'));ac=ndi.gaussian_filter(a.astype(float),1);bc=ndi.gaussian_filter(b.astype(float),1);scores=[]
 for scale in [.75,1.,4/3]:
  size=round(256*scale);base=np.asarray(Image.fromarray(bc.astype(np.float32),mode='F').resize((size,size),Image.Resampling.BICUBIC))
  for k in range(4):
   for flip in [False,True]:
    z=np.rot90(base,k);z=np.fliplr(z) if flip else z;corr=match_template(z,ac[64:192,64:192],pad_input=False);loc=np.unravel_index(np.argmax(corr),corr.shape);scores.append({'scale':scale,'rot90':k,'flip':flip,'max_center128_template_NCC':float(corr[loc]),'location_yx':[int(v) for v in loc]})
 write(_repo_path(OUT, 'pair-comparison.json'),{'IDs':PAIR,'same_decoded_size':[256,256],'raw_RMSE_0_255':float(np.sqrt(np.mean((a.astype(float)-b)**2))),'raw_pearson':float(np.corrcoef(a.ravel(),b.ravel())[0,1]),'smooth_pearson':float(np.corrcoef(ac.ravel(),bc.ravel())[0,1]),'best_bounded_geometry_match':max(scores,key=lambda r:r['max_center128_template_NCC']),'all_fixed_geometry_checks':scores,'caveat':'No physical pixel calibration/sourcecrop metadata. Bounded correspondence does not establish original magnification or label correctness. No label read or change.'})
 for n,h in p['protected011'].items():assert sha(_repo_path(ROOT, n))==h,n
 print(json.dumps({'finished':True,'images':len(ids),'label_access':False,'best011_unchanged':True}),flush=True)
if __name__=='__main__':
 {'prepare':prepare,'panels':panels}[sys.argv[1]]()
