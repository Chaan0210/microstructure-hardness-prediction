"""Image-local 2D signed-distance Betti curves; not persistent homology."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import euler_number
QUANTILES=(.2,.5,.8)
DISTANCES=(-16,-8,-4,-2,-1,0,1,2,4,8,16)
NAMES=tuple(f'betti_q{int(q*100)}_d{d}_c{c}_b{b}' for q in QUANTILES for d in DISTANCES for c in (1,2) for b in (0,1))
ZERO_COLUMNS=np.array([i for i,n in enumerate(NAMES) if '_d0_' in n])

def signed_distance(mask):
 mask=np.asarray(mask,dtype=bool)
 # No imaginary outside phase for distance. Uniform phases have no opposite
 # pixel, so their signed distance is +/- infinity and all finite excursions
 # stay uniform. Mixed masks use nearest opposite pixel within the image.
 if mask.all():return np.full(mask.shape,np.inf)
 if not mask.any():return np.full(mask.shape,-np.inf)
 return ndi.distance_transform_edt(mask)-ndi.distance_transform_edt(~mask)

def betti(mask,connectivity):
 mask=np.asarray(mask,dtype=bool)
 _,b0=ndi.label(mask,structure=ndi.generate_binary_structure(2,connectivity))
 # Euler uses outside-background, foreground4/background8 or vice versa.
 # Background components touching any image edge are not holes.
 b1=int(b0)-int(euler_number(mask,connectivity=connectivity))
 assert b0>=0 and b1>=0
 return int(b0),b1

def curves(mask):
 sd=signed_distance(mask);values=[]
 for d in DISTANCES:
  exc=sd>=d
  for c in (1,2):values.extend(betti(exc,c))
 return np.asarray(values,dtype=float)/mask.size

def extract(image):
 image=np.asarray(image)
 if image.shape!=(256,256) or image.dtype!=np.uint8:raise ValueError('Expected raw256uint8gray')
 thresholds=np.quantile(image,QUANTILES,method='linear');panels=[];fractions=[]
 for t in thresholds:
  mask=image<=t;fractions.append(float(mask.mean()));panels.append(curves(mask))
 result=np.concatenate(panels);assert result.shape==(132,) and np.isfinite(result).all()
 # Diagnostics only: high frequency contrast includes edges as well as noise.
 x=image.astype(float)/255.;hf=x-ndi.gaussian_filter(x,.7)
 return result,{'thresholds':thresholds.tolist(),'mask_fractions':fractions,'high_frequency_rms':float(np.sqrt(np.mean(hf**2))),'image_std':float(x.std())}
