import numpy as np
from scipy.ndimage import convolve1d
def energy(x):return np.square(np.diff(x,axis=2)).mean((1,2,3))+np.square(np.diff(x,axis=3)).mean((1,2,3))
def calibration(x):
 e=energy(x);q10,q50=np.quantile(e,[.1,.5]);a=float(np.float32(min(.04,.1*max(0,1-q10/q50))));assert a>.001
 return {'alpha':a,'train_gradient_energy_Q10':float(q10),'train_gradient_energy_Q50':float(q50),'kernel':[a,1-2*a,a],'effective_peraxis_RMS_radius':float(np.sqrt(2*a))}
def blur(x,alpha):
 kernel=np.array([alpha,1-2*alpha,alpha],dtype=np.float64);assert abs(kernel.sum()-1)<1e-15 and kernel.min()>0
 return convolve1d(convolve1d(x,kernel,axis=2,mode='reflect'),kernel,axis=3,mode='reflect')
