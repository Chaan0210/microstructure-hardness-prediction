"""Nonzero-lag2D covariance shape at two fixed smoothing scales.

Linear (zero-padded) autocovariance uses actual pair counts. Shape normalization
uses RMS covariance on the nonzero radius2 ring, NEVER image variance minus
estimated noise variance. Noise, blur and covariance amplitude stay separate.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from functools import lru_cache
import numpy as np
from scipy import ndimage as ndi
SCALES=(1.,2.)
RADII=(4.,8.,12.,16.,24.,32.,48.,64.)
MAX_LAG=80
FEATURE_NAMES=tuple(f'acf_s{int(s)}_r{int(r)}_{m}' for s in SCALES for r in RADII for m in ('radial','h2'))+tuple(f'acf_s{int(s)}_nonzero_anchor_rms' for s in SCALES)+('acf_noise_sigma','acf_smooth_variance_share')
BLOCKS=tuple('acf_'+m for s in SCALES for r in RADII for m in ('radial','h2'))+('acf_amplitude',)*2+('acf_nuisance',)*2

@lru_cache(maxsize=1)
def geometry():
 lags=np.arange(-MAX_LAG,MAX_LAG+1);dy,dx=np.meshgrid(lags,lags,indexing='ij');radius=np.hypot(dy,dx);angle=np.arctan2(dy,dx)
 overlap=(256-np.abs(dy))*(256-np.abs(dx));rings={r:np.abs(radius-r)<=max(.75,.1*r) for r in (2.,*RADII)}
 return overlap,rings,np.exp(2j*angle)

def linear_covariance(centered):
 spectrum=np.fft.rfft2(centered,s=(512,512));corr=np.fft.fftshift(np.fft.irfft2(np.abs(spectrum)**2,s=(512,512)))
 a=256-MAX_LAG;b=256+MAX_LAG+1;overlap,_,_=geometry()
 return corr[a:b,a:b]/overlap

def extract_autocov(image):
 image=np.asarray(image)
 if image.shape!=(256,256) or image.dtype!=np.uint8:raise ValueError('Expected256x256uint8')
 x=image.astype(float)/255.;centered=x-x[0,0];centered-=centered.mean();_,rings,harmonic=geometry();values=[];anchors=[]
 for scale in SCALES:
  sm=ndi.gaussian_filter(centered,scale,mode='reflect');cov=linear_covariance(sm)
  anchor=float(np.sqrt(np.mean(cov[rings[2.]]**2)));anchors.append(anchor)
  norm=max(anchor,1e-12)
  for radius in RADII:
   ring=rings[radius];values.extend([float(cov[ring].mean()/norm),float(abs(np.mean(cov[ring]*harmonic[ring]))/norm)])
 hh=(x[::2,::2]-x[::2,1::2]-x[1::2,::2]+x[1::2,1::2])/2.;noise=float(np.median(abs(hh))/.6744897501960817)
 variance=float(centered.var());blur=float(ndi.gaussian_filter(centered,1.,mode='reflect').var()/variance) if variance>1e-20 else 0.
 result=np.array([*values,*anchors,noise,blur],float)
 if result.shape!=(36,) or not np.isfinite(result).all():raise RuntimeError('Invalid autocovariance schema')
 return result,list(FEATURE_NAMES),list(BLOCKS)
