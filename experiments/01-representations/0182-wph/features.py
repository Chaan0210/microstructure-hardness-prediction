"""Fixed 15D adjacent-scale H(1)-H(2) centered covariances and matched amplitude control."""
from functools import lru_cache
import numpy as np
from scipy import fft
FREQUENCIES=(1/8,1/16,1/32,1/64)

def harmonic2(z):
 u=np.divide(z,np.abs(z),out=np.zeros_like(z),where=np.abs(z)>0)
 return z*u

@lru_cache(maxsize=2)
def filters(size):
 a=np.fft.fftfreq(size);yy,xx=np.meshgrid(a,a,indexing='ij');rad=np.hypot(xx,yy);out=[]
 for freq in FREQUENCIES:
  q=(rad/freq-1)/.3;mask=np.abs(q)<1;rho=np.zeros_like(q);rho[mask]=np.exp(1-1/(1-q[mask]**2));level=[]
  for t in np.arange(8)*np.pi/8:
   cosine=np.divide(xx*np.cos(t)+yy*np.sin(t),rad,out=np.zeros_like(rad),where=rad>0);h=rho*np.maximum(cosine,0)**6;h/=np.sqrt(np.mean(h*h));level.append(h)
  out.append(np.stack(level))
 return np.stack(out)

def coefficients(x,pad=64,backend='numpy'):
 x=np.asarray(x,dtype=float);assert x.ndim==2 and x.shape[0]==x.shape[1]
 x=x-x[0,0];x=x-x.mean();n=x.shape[0];x=np.pad(x,pad,mode='reflect') if pad else x
 ff=fft.fft2(x,workers=1) if backend=='scipy' else np.fft.fft2(x);rows=[]
 for bank in filters(len(x)):
  w=fft.ifft2(bank*ff,workers=1) if backend=='scipy' else np.fft.ifft2(bank*ff)
  w=w[:,pad:pad+n,pad:pad+n] if pad else w
  rows.append(np.concatenate([w,w.conj()],axis=0))
 return np.stack(rows)

def centered_cov(a,b):
 a=a.reshape(len(a),-1);b=b.reshape(len(b),-1);a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True)
 return a@b.conj().T/a.shape[1]

def pool(c):
 t=np.arange(16);return np.array([np.mean(np.concatenate([c[t,(t+d)%16],c[t,(t-d)%16]])).real for d in range(5)])

def describe(w):
 phase=[];amp=[]
 for j in range(3):
  phase.extend(pool(centered_cov(w[j],harmonic2(w[j+1]))));amp.extend(pool(centered_cov(np.abs(w[j]),np.abs(w[j+1]))))
 return np.asarray(phase),np.asarray(amp)

def extract(x,pad=64,backend='numpy'):
 a,b=describe(coefficients(x,pad,backend));assert a.shape==b.shape==(15,) and np.isfinite(a).all() and np.isfinite(b).all();return a,b
