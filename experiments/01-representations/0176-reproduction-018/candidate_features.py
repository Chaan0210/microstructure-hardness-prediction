"""Reproduce frozen compact panel from one image; no fitted dataset statistics."""
import numpy as np
from grain_features import extract_grains
from latent_features import extract_latent,DESCRIPTORS
from compact_regression import compact_existing

def extract_candidate(image,panel='run-012'):
    grain,gn=extract_grains(image);existing,names,blocks=compact_existing(grain[None],gn)
    raw,rn=extract_latent(image);values=list(existing[0]);names=list(names)
    for name in DESCRIPTORS:
        ix=[i for i,n in enumerate(rn) if n.endswith('_'+name)];assert len(ix)==6
        values.extend([np.median(raw[ix]),np.ptp(raw[ix])]);names.extend(['intercept_median_'+name,'intercept_sensitivity_'+name])
    values.extend([image.mean()/255.,image.std()/255.]);names.extend(['image_mean','image_std'])
    if panel=='run-014':
        from phase_features import extract_phase,DESCRIPTORS as PHASE
        raw,rn=extract_phase(image)
        for name in PHASE:
            ix=[i for i,n in enumerate(rn) if n.endswith('_'+name)];assert len(ix)==3
            values.extend([np.median(raw[ix]),np.ptp(raw[ix])]);names.extend(['interior_median_'+name,'interior_sensitivity_'+name])
    elif panel!='run-012':raise ValueError('Unknown frozen panel')
    x=np.array(values,dtype=float)
    if not np.isfinite(x).all():raise ValueError('Nonfinite compact features')
    return x,names
