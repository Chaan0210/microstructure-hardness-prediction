"""Additional image-local morphology descriptors; no cross-image fitting."""
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu

def extract_extra(image):
    x=ndi.gaussian_filter(image.astype(float)/255.,1.)
    lo,hi=np.percentile(x,[1,99])
    x=np.clip((x-lo)/max(hi-lo,1e-6),0,1)
    vals=[];names=[]
    def add(n,v): names.append("extra_"+n);vals.append(float(v))
    def stats(n,a):
        a=np.asarray(a,dtype=float)
        if not a.size:a=np.zeros(1)
        add(n+"_mean",a.mean());add(n+"_std",a.std())
        for q in (10,50,90):add(n+"_q"+str(q),np.percentile(a,q))
    def maskstats(n,mask):
        add(n+"_fraction",mask.mean())
        lab=label(mask)
        regions=[r for r in regionprops(lab) if r.area>=8]
        add(n+"_count",len(regions))
        for attr in ("area","eccentricity","solidity","axis_minor_length"):
            stats(n+"_"+attr,[getattr(r,attr) for r in regions])
        # Exclude clipped regions when approximating typical grain size.
        interior=[r for r in regions if r.bbox[0]>0 and r.bbox[1]>0 and r.bbox[2]<256 and r.bbox[3]<256]
        add(n+"_interior_count",len(interior))
        stats(n+"_interior_equivdiam",[r.equivalent_diameter_area for r in interior])
        diam=np.asarray([r.equivalent_diameter_area for r in interior])
        add(n+"_inverse_sqrt_diameter",np.mean(1/np.sqrt(diam)) if diam.size else 0.)
        stats(n+"_distance",ndi.distance_transform_edt(mask)[mask])
    for name,t in [("otsu",threshold_otsu(x)),("q25",np.quantile(x,.25)),
                   ("q50",np.quantile(x,.50)),("q75",np.quantile(x,.75))]:
        maskstats("bright_"+name,x>t)
    # Granulometry measures how much structure disappears at each spatial scale.
    for size in (3,5,9,17,33):
        opened=ndi.grey_opening(x,size=(size,size))
        closed=ndi.grey_closing(x,size=(size,size))
        stats(f"whitehat_{size}",x-opened)
        stats(f"blackhat_{size}",closed-x)
    # Thin dark boundaries are enhanced against their local background.
    for size in (5,9):
        blackhat=ndi.grey_closing(x,size=(size,size))-x
        t=threshold_otsu(blackhat)
        boundaries=blackhat>t
        maskstats(f"boundary_interior_{size}",~ndi.binary_dilation(boundaries,iterations=1))
        add(f"boundary_density_{size}",boundaries.mean())
    for sigma in (2,4,8,16):
        local=x-ndi.gaussian_filter(x,sigma)
        stats(f"localcontrast_{sigma}",local)
    return np.asarray(vals), names
