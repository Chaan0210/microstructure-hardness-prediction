"""Image-local grain/phase hypotheses; no labels or dataset statistics."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.measure import label, regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

def segment(image,sigma):
    x=ndi.gaussian_filter(image.astype(float)/255.,sigma)
    ridge=ndi.grey_closing(x,size=(7,7))-x
    threshold=threshold_otsu(ridge) if np.ptp(ridge)>1e-12 else 1.
    edge=ridge>threshold
    distance=ndi.distance_transform_edt(~edge)
    markers=label(h_maxima(distance,1.5))
    if markers.max()==0:markers[image.shape[0]//2,image.shape[1]//2]=1
    lab=watershed(ridge,markers,watershed_line=True)
    return x,ridge,edge,lab

def extract_grains(image):
    vals=[];names=[]
    def add(n,v):names.append(n);vals.append(float(v))
    def stats(n,a):
        a=np.asarray(a,dtype=float)
        if not a.size:a=np.zeros(1)
        add(n+'_mean',a.mean());add(n+'_std',a.std())
        for q in (10,50,90):add(n+'_q'+str(q),np.percentile(a,q))
    scale_metrics=[]
    for sigma in (1.,2.):
        prefix=f'grain_s{sigma:g}'
        x,ridge,edge,lab=segment(image,sigma)
        rs=[r for r in regionprops(lab,intensity_image=x) if r.area>=16]
        inside=[r for r in rs if r.bbox[0]>0 and r.bbox[1]>0 and r.bbox[2]<image.shape[0] and r.bbox[3]<image.shape[1]]
        add(prefix+'_count',len(rs));add(prefix+'_interior_count',len(inside));add(prefix+'_boundary_fraction',edge.mean())
        for tag,regions in [('all',rs),('interior',inside)]:
            diam=np.array([r.equivalent_diameter_area for r in regions]);areas=np.array([r.area for r in regions])
            stats(prefix+'_'+tag+'_diam',diam)
            stats(prefix+'_'+tag+'_minor',[r.axis_minor_length for r in regions])
            stats(prefix+'_'+tag+'_aspect',[r.axis_major_length/max(r.axis_minor_length,1) for r in regions])
            add(prefix+'_'+tag+'_area_weighted_diam',np.average(diam,weights=areas) if areas.size else 0.)
            add(prefix+'_'+tag+'_hall_petch',np.mean(1/np.sqrt(diam)) if diam.size else 0.)
        weights=np.array([r.area for r in rs]);angles=np.array([r.orientation for r in rs])
        add(prefix+'_alignment',abs(np.sum(weights*np.exp(2j*angles)))/max(weights.sum(),1))
        stats(prefix+'_mean_intensity',[r.intensity_mean for r in rs])
        stats(prefix+'_within_std',[np.std(r.image_intensity[r.image]) for r in rs])
        scale_metrics.append([len(rs),np.mean([r.equivalent_diameter_area for r in rs]) if rs else 0.,edge.mean()])
    for k,n in enumerate(('count','diam','boundary')):
        add('scale_robustness_'+n,abs(scale_metrics[0][k]-scale_metrics[1][k])/(abs(scale_metrics[0][k])+abs(scale_metrics[1][k])+1e-9))
    # Three intensity populations are only phase proxies, not ground-truth phases.
    for sigma in (1.,3.):
        x=ndi.gaussian_filter(image.astype(float)/255.,sigma)
        qs=np.percentile(x,[2,98]);x=np.clip((x-qs[0])/max(qs[1]-qs[0],1e-6),0,1)
        thresholds=threshold_multiotsu(x,classes=3,nbins=64) if np.unique(x).size>=3 else np.array([.33,.66])
        ph=np.digitize(x,thresholds)
        for k in range(3):
            mask=ph==k;add(f'phase_s{sigma:g}_{k}_fraction',mask.mean())
            interior=ndi.binary_erosion(mask,iterations=2)
            add(f'phase_s{sigma:g}_{k}_interior',interior.mean())
            add(f'phase_s{sigma:g}_{k}_intensity',x[interior].mean() if interior.any() else 0.)
        # Compact dark objects discriminate pore-like spots from long grain edges.
        for t in (.15,.30):
            regions=[r for r in regionprops(label(x<t)) if 4<=r.area<=1000 and r.eccentricity<.9 and r.solidity>.7]
            p=f'pore_s{sigma:g}_t{t:g}'
            add(p+'_count',len(regions));add(p+'_fraction',sum(r.area for r in regions)/x.size)
            stats(p+'_diam',[r.equivalent_diameter_area for r in regions])
    # Local structure tensors capture spatially coherent direction independent of angle.
    x=ndi.gaussian_filter(image.astype(float)/255.,1.)
    gy,gx=np.gradient(x)
    for scale in (3.,9.):
        a,b,c=[ndi.gaussian_filter(z,scale) for z in (gx*gx,gy*gy,gx*gy)]
        coherence=np.sqrt((a-b)**2+4*c*c)/(a+b+1e-12)
        stats(f'tensor_s{scale:g}_coherence',coherence)
        add(f'tensor_s{scale:g}_weighted',np.sum(coherence*(a+b))/max(np.sum(a+b),1e-12))
    arr=np.array(vals)
    if not np.isfinite(arr).all():raise ValueError('Nonfinite grain features')
    return arr,names
