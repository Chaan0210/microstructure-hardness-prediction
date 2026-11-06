"""Image-local positive minimum-Hessian-curvature dark-blob statistics.

Minimum positive curvature rejects ideal long ridges without rejecting an
entire connected component when a pore touches a ridge. No external data or
synthetic model-training rows are used. These are proxies, not pore truth.
"""
import numpy as np
from scipy import ndimage as ndi
SCALES=(1.,2.,4.,8.)
THRESHOLD=.01
METRICS=('peak_density','peak_strength_density','peak_roundness_median','response_q995','positive_response_mean')
FEATURE_NAMES=tuple(f'porecurve_s{int(s)}_{m}' for s in SCALES for m in METRICS)+('porecurve_noise_sigma','porecurve_smooth_variance_share')
BLOCKS=tuple('porecurve_'+b for _ in SCALES for b in ('density','strength','shape','strength','strength'))+('porecurve_nuisance',)*2

def maps(image):
 image=np.asarray(image)
 if image.shape!=(256,256) or image.dtype!=np.uint8:raise ValueError('Expected256x256uint8')
 x=image.astype(np.float64)/255.;centered=x-x[0,0];centered-=centered.mean()
 output=[]
 for scale in SCALES:
  xx=scale**2*ndi.gaussian_filter(centered,scale,order=(0,2),mode='reflect')
  yy=scale**2*ndi.gaussian_filter(centered,scale,order=(2,0),mode='reflect')
  xy=scale**2*ndi.gaussian_filter(centered,scale,order=(1,1),mode='reflect')
  gap=np.sqrt((xx-yy)**2+4*xy**2);low=.5*(xx+yy-gap);high=.5*(xx+yy+gap)
  response=np.maximum(low,0.)
  # A long ridge adds curvature mainly in one direction. Do not impose a
  # roundness gate: that would suppress a small pore attached to a strong ridge.
  ratio=np.divide(response,np.maximum(high,1e-12),out=np.zeros_like(response),where=high>1e-12)
  radius=max(1,int(scale));peak=(response==ndi.maximum_filter(response,size=2*radius+1,mode='reflect'))&(response>THRESHOLD)
  margin=int(3*scale);valid=np.zeros_like(peak);valid[margin:-margin,margin:-margin]=True;peak&=valid
  output.append({'scale':scale,'response':response,'roundness':ratio,'peaks':peak,'valid':valid})
 return x,output

def extract_pore_curvature(image):
 x,records=maps(image);values=[]
 for record in records:
  response=record['response'];p=record['peaks'];valid=record['valid'];area=int(valid.sum());r=response[valid]
  values.extend([10000*int(p.sum())/area,10000*float(response[p].sum())/area,float(np.median(record['roundness'][p])) if p.any() else 0.,float(np.percentile(r,99.5)),float(r.mean())])
 hh=(x[::2,::2]-x[::2,1::2]-x[1::2,::2]+x[1::2,1::2])/2.;noise=float(np.median(np.abs(hh))/.6744897501960817)
 var=float(x.var());sm=ndi.gaussian_filter(x-x[0,0],1.,mode='reflect');share=float(sm.var()/var) if var>1e-20 else 0.
 values.extend([noise,share]);result=np.asarray(values,float)
 if result.shape!=(22,) or not np.isfinite(result).all():raise RuntimeError('Invalid porecurvature schema')
 return result,list(FEATURE_NAMES),list(BLOCKS)
