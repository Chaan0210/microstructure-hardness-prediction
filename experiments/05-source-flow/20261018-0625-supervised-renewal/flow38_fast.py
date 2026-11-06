"""The first38 original flow descriptors, without unused semantic pooling."""
import numpy as np
from scipy.special import expit
from scipy.ndimage import gaussian_filter


def descriptors(flow, image):
    im = image.astype(np.float32)/255
    lo, hi = np.percentile(im, [10, 90]); scale = max(float((hi-lo)/2.563), .01); mean = float(im.mean())
    fy, fx = flow[:2].astype(float)/5
    logit = np.clip(flow[2].astype(float), -20, 20); prob = expit(logit)
    dyfy, dxfy = np.gradient(fy); dyfx, dxfx = np.gradient(fx)
    div = dxfx+dyfy; curl = dyfx-dxfy; a = dxfx-dyfy; b = dyfx+dxfy
    strain = np.sqrt(a*a+b*b); speed = np.sqrt(fx*fx+fy*fy)
    gray = (gaussian_filter(im, 1.)-mean)/scale
    values, keys = [], []
    for name, field in [('cell_logit', logit), ('speed', speed), ('divergence', div), ('curl', curl), ('strain', strain), ('gray_divergence', gray*div)]:
        values.extend([field.mean(), field.std(), *np.quantile(field, [.1, .5, .9])])
        keys.extend(name+'_'+s for s in ['mean', 'std', 'q10', 'q50', 'q90'])
    values.extend([prob.mean(), prob.std(), np.mean(-prob*np.log(prob+1e-8)-(1-prob)*np.log(1-prob+1e-8))])
    keys += ['prob_mean', 'prob_std', 'prob_entropy']
    values.extend([np.mean(prob*(a*a-b*b)/(strain+1e-6)), np.mean(prob*(2*a*b)/(strain+1e-6))])
    keys += ['strain_c4', 'strain_s4']
    for name, field in [('prob', prob), ('speed', speed), ('divergence', div)]:
        values.append(field.reshape(4, 64, 4, 64).mean((1, 3)).std()); keys.append(name+'_tile4_std')
    return np.array(values, np.float32), keys
