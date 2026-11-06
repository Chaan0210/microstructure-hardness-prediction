import sys,os,time
import runtime as f
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.filters import threshold_otsu
from skimage.morphology import h_minima,skeletonize
from skimage.measure import label,regionprops
from skimage.segmentation import watershed
from skimage.restoration import denoise_tv_chambolle
from PIL import Image,ImageDraw,ImageFont
R=f.R;P=f.P;sys.path.insert(0,str(P))
import denoised_grain_features as old

def segment(x,hm=1.,rm=1.,contrast=1.):
 x=contrast*(x-.5)+.5;noise=old._noise_sigma(x);d=denoise_tv_chambolle(x,weight=noise,eps=1e-4,max_num_iter=50,channel_axis=None) if noise>1e-12 else x.copy();ridge=ndi.grey_closing(d,size=(7,7),mode='reflect')-d;mask=ridge>rm*threshold_otsu(ridge) if np.ptp(ridge)>1e-12 else np.zeros_like(x,bool);grad=ndi.gaussian_gradient_magnitude(d,.7,mode='reflect')
 if np.ptp(grad)<=1e-12:seg=np.ones(x.shape,np.int32)
 else:
  mins=h_minima(grad,max(.2*hm*np.percentile(grad,90),1e-12)).astype(bool)&~mask;marks=label(mins,connectivity=2)
  if not marks.any():marks.flat[int(np.argmin(np.where(~mask,grad,np.inf)))]=1
  seg=watershed(grad,marks,watershed_line=True)
 return seg.astype(np.int32),d

def fill(seg):
 assert np.any(seg>0)
 ii=ndi.distance_transform_edt(seg==0,return_distances=False,return_indices=True);return seg[tuple(ii)]

def graph(s):
 pairs=np.concatenate([np.column_stack([s[:-1].ravel(),s[1:].ravel()]),np.column_stack([s[:,:-1].ravel(),s[:,1:].ravel()])]);pairs=np.sort(pairs,axis=1);return set(map(tuple,np.unique(pairs[pairs[:,0]!=pairs[:,1]],axis=0).tolist()))

def compare(a,b):
 aa=np.unique(a);bb=np.unique(b);A=np.searchsorted(aa,a);B=np.searchsorted(bb,b);over=np.bincount((A*len(bb)+B).ravel(),minlength=len(aa)*len(bb)).reshape(len(aa),len(bb));ra,rb=linear_sum_assignment(-over);areasA=over.sum(1);areasB=over.sum(0);ious=over[ra,rb]/(areasA[ra]+areasB[rb]-over[ra,rb]);valid=ious>=.5;mapping={int(bb[j]):int(aa[i]) for i,j in zip(ra[valid],rb[valid])};ga=graph(a);gb={tuple(sorted((mapping.get(int(x),-int(x)-1),mapping.get(int(y),-int(y)-1)))) for x,y in graph(b)};union=ga|gb
 return {'matched_cell_fraction':float(valid.sum()/len(aa)),'matched_sourcepixel_fraction':float(areasA[ra[valid]].sum()/a.size),'matched_mean_IoU':float(ious[valid].mean()) if valid.any() else 0.,'adjacency_Jaccard':len(ga&gb)/len(union) if union else 1.,'base_edges':len(ga),'probe_edges':len(gb),'base_allcell_count':len(aa),'probe_allcell_count':len(bb)}

def stats(seg):
 rs=regionprops(seg);eligible=[r for r in rs if r.area>=16];areas=np.array([r.area for r in eligible],float);major=np.array([r.axis_major_length for r in eligible]);minor=np.array([r.axis_minor_length for r in eligible]);inter=[r for r in eligible if r.bbox[0]>0 and r.bbox[1]>0 and r.bbox[2]<256 and r.bbox[3]<256];counts=np.bincount(seg.ravel());assert all(counts[r.label]==r.area for r in rs);assert sum(counts[1:])+counts[0]==seg.size
 return {'count':len(eligible),'allcell_count':len(rs),'median_area':float(np.median(areas)) if len(areas) else 0.,'area_q10_q90':np.quantile(areas,[.1,.9]).tolist() if len(areas) else [0.,0.],'areaweighted_aspect':float(np.average(major/np.maximum(minor,1.),weights=areas)) if len(areas) else 0.,'interior_count':len(inter),'interior_area_fraction':sum(r.area for r in inter)/seg.size,'small_region_area_fraction':sum(r.area for r in rs if r.area<16)/seg.size,'frame_region_area_fraction':sum(r.area for r in eligible if r not in inter)/seg.size,'watershedline_fraction':float(np.mean(seg==0))}

def proxy(raw,seg):
 edge=ndi.gaussian_gradient_magnitude(raw,1.,mode='reflect');anchor=skeletonize((edge>np.quantile(edge,.90))&(edge>0));mask=np.zeros_like(anchor);mask[3:-3,3:-3]=True;anchor&=mask;bound=(seg==0)&mask
 if not anchor.any() or not bound.any():return {'precision':0.,'recall':0.,'anchor_pixels':int(anchor.sum()),'predicted_pixels':int(bound.sum()),'informative':False},anchor
 da=ndi.distance_transform_edt(~anchor);db=ndi.distance_transform_edt(~bound);p=float(np.mean(da[bound]<=2));r=float(np.mean(db[anchor]<=2))
 # Independent 5x5 Euclidean disk dilation gives exactly the same counts.
 yy,xx=np.mgrid[-2:3,-2:3];disk=xx*xx+yy*yy<=4;np.testing.assert_equal(p,np.mean(ndi.binary_dilation(anchor,structure=disk)[bound]));np.testing.assert_equal(r,np.mean(ndi.binary_dilation(bound,structure=disk)[anchor]))
 return {'precision':p,'recall':r,'anchor_pixels':int(anchor.sum()),'predicted_pixels':int(bound.sum()),'informative':True},anchor

def overlay(a,line,color=(255,32,32)):
 out=np.repeat(a[:,:,None],3,axis=2);out[line]=color;return Image.fromarray(out)

def engineering():
 a=np.array([[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]);assert graph(a)=={(1,2),(1,3),(2,4),(3,4)};b=np.array([0,9,7,5,3])[a];c=compare(a,b);assert c['matched_sourcepixel_fraction']==c['matched_cell_fraction']==c['adjacency_Jaccard']==1
 b=a.copy();b[0,:2]=5;assert compare(a,b)['adjacency_Jaccard']<1;z=np.ones((8,8),int);z[:,4]=0;z[:,5:]=2;zz=fill(z);assert np.all(zz>0) and graph(zz)=={(1,2)}
 return {'passed':True,'hand_fourcell_graph_exact':True,'label_permutation_identity':True,'split_changes_adjacency':True,'linefill_coverage':True,'no_synthetic_training':True}

def main():
 plan=f.check();f.write(R/'engineering-validation.json',engineering());allowed={P/'data/train'/f'{i}.png' for i in plan['IDs']}
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   p=f.Path(os.fsdecode(args[0])).absolute()
   if p.is_relative_to(P/'data') and p not in allowed:raise RuntimeError('Nonselected images/targetCSV/holdout/test prohibited')
   if p.suffix=='.npz' and not p.is_relative_to(R):raise RuntimeError('Prior feature/target/prediction arrays prohibited')
 sys.addaudithook(guard);(R/'arrays').mkdir();(R/'panels').mkdir();font=ImageFont.load_default(size=13);smallfont=ImageFont.load_default(size=11);records=[];overview=[];start=time.perf_counter();names=list(plan['variants'])
 with f.Monitor('diagnostic',plan['resources']):
  for i,ident in enumerate(plan['IDs']):
   with Image.open(P/'data/train'/f'{ident}.png') as im:assert im.mode=='L' and im.size==(256,256);raw=np.array(im)
   a=old._canonical_d4(raw);x=a.astype(float)/255;segs={};details={};fills={};probes={}
   for name,params in plan['variants'].items():
    s,d=segment(x,*params);segs[name]=s;details[name]=stats(s);fills[name]=fill(s)
   # Independently invoke the unmodified existing implementation for3 fixedfirstIDs.
   if i<3:
    import grain_visual_audit_097 as legacy
    _,_,_,ss=legacy.segment(raw);ref=next(v for v in ss if v['tv']==1. and v['h']==.2);np.testing.assert_array_equal(segs['base'],ref['seg'])
   for name in names[1:]:
    q=compare(fills['base'],fills[name]);q['relative_changes']={k:abs(details[name][k]-details['base'][k])/max(abs(details['base'][k]),1e-12) for k in ['count','median_area','areaweighted_aspect']};probes[name]=q
   mx={k:max(v['relative_changes'][k] for v in probes.values()) for k in ['count','median_area','areaweighted_aspect']};minmatch=min(v['matched_sourcepixel_fraction'] for v in probes.values());minadj=min(v['adjacency_Jaccard'] for v in probes.values());stable=all(v<=.1 for v in mx.values()) and minmatch>=.9 and minadj>=.9;br,anchor=proxy(x,segs['base']);g=graph(fills['base']);labels=np.unique(fills['base']);disconnected=[int(v) for v in labels if ndi.label(fills['base']==v)[1]!=1]
   rec={'ID':ident,'image_sha256':f.sha(P/'data/train'/f'{ident}.png'),'variant_metrics':details,'probe_stability':probes,'max_relative_changes':mx,'minimum_matched_sourcepixel_fraction':minmatch,'minimum_adjacency_Jaccard':minadj,'stability_pass':stable,'boundary_proxy':br,'global_partition':{'complete_after_forced_linefill':True,'disconnected_cell_IDs':disconnected,'graph_edges':len(g),'number_of_cells':len(labels),'coverage_not_geometry_evidence':True}};records.append(rec)
   with (R/'arrays'/f'{ident}.npz').open('xb') as fp:np.savez_compressed(fp,raw_canonical=a,anchor=anchor,**segs)
   canvas=Image.new('RGB',(1024,858),'white');draw=ImageDraw.Draw(canvas);tiles=[('raw original gray',Image.fromarray(a).convert('RGB')),('base watershed RED',overlay(a,segs['base']==0))]+[(n+' RED',overlay(a,segs[n]==0)) for n in names[1:]]+[('raw gradient anchors CYAN (not truth)',overlay(a,anchor,(0,220,255)))]
   for t,(title,im) in enumerate(tiles):
    cx=t%4*256;cy=t//4*286;draw.text((cx+3,cy+3),title,fill='black',font=smallfont);canvas.paste(im,(cx,cy+26))
   canvas.save(R/'panels'/f'{ident}.png');overview.append((ident,a,segs['base'],segs['h_plus5']));f.emit({'completed_images':i+1,'seconds':time.perf_counter()-start})
  for page in range(4):
   canvas=Image.new('RGB',(768,1160),'white');dr=ImageDraw.Draw(canvas)
   for row,(ident,a,s,h) in enumerate(overview[page*4:page*4+4]):
    for col,(title,im) in enumerate([('raw',Image.fromarray(a).convert('RGB')),('base RED',overlay(a,s==0)),('h+5% RED',overlay(a,h==0))]):
     dr.text((col*256+3,row*290+4),ident+' '+title,fill='black',font=font);canvas.paste(im,(col*256,row*290+27))
   canvas.save(R/f'overview-{page+1}.png')
  prec=[v['boundary_proxy']['precision'] for v in records];recall=[v['boundary_proxy']['recall'] for v in records];stable=sum(v['stability_pass'] for v in records);statgate=stable>=14;boundarygate=np.median(prec)>=.6 and np.median(recall)>=.8
  aggregate={'images':16,'segmentations':112,'stability_pass_images':stable,'stability_gate':statgate,'max_relative_change_medians':{k:float(np.median([v['max_relative_changes'][k] for v in records])) for k in ['count','median_area','areaweighted_aspect']},'minimum_matched_pixel_fraction_median':float(np.median([v['minimum_matched_sourcepixel_fraction'] for v in records])),'minimum_adjacency_Jaccard_median':float(np.median([v['minimum_adjacency_Jaccard'] for v in records])),'boundary_proxy_median_precision':float(np.median(prec)),'boundary_proxy_median_recall':float(np.median(recall)),'boundary_proxy_gate':bool(boundarygate),'quantitative_gate_pass':bool(statgate and boundarygate),'visual_review_pending':True,'label_access':False,'model_fit':False,'holdout_test_access':False,'seconds':time.perf_counter()-start,'existing_baseline_segmentation_exact_first3':True,'boundary_dilation_validation_exact':True,'plan_sha256':f.sha(R/'plan.json')}
  f.write(R/'measurements.json',{'aggregate':aggregate,'images':records});f.check();f.emit(aggregate)
 if not aggregate['quantitative_gate_pass']:f.emit({'diagnosis':'Reject operational inverse-rendering route; visual review and report only, no training.'})
if __name__=='__main__':main()
