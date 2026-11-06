import runtime as f
import features as w
import numpy as np
R=f.R

def main():
 f.check();rng=np.random.default_rng(182);z=rng.normal(size=100)+1j*rng.normal(size=100);z[0]=0;h=w.harmonic2(z);np.testing.assert_allclose(abs(h),abs(z),atol=1e-14);np.testing.assert_allclose(h,abs(z)*np.exp(2j*np.angle(z)),atol=1e-14);np.testing.assert_allclose(w.harmonic2(z*np.exp(.37j)),h*np.exp(.74j),atol=1e-14)
 a=rng.normal(size=(4,31))+1j*rng.normal(size=(4,31));b=rng.normal(size=(6,31))+1j*rng.normal(size=(6,31));direct=np.array([[sum((v-v.mean())*np.conj(u-u.mean()))/31 for u in b] for v in a]);cov_error=float(abs(direct-w.centered_cov(a,b)).max());assert cov_error<1e-14
 yy,xx=np.mgrid[:256,:256];make=lambda c,q:.5+.2*np.cos(2*np.pi*16*xx/256+c)+.15*np.cos(2*np.pi*32*xx/256+q)
 cases=[w.extract(make(c,q),pad=0) for c,q in [(0,0),(0,np.pi/2),(0,np.pi),(.3,.6)]];p0,a0=cases[0];scale=max(abs(p0).max(),1e-30);assert scale>.001
 np.testing.assert_allclose(cases[1][0],0,atol=scale*1e-12);np.testing.assert_allclose(cases[2][0],-p0,atol=scale*1e-12);np.testing.assert_allclose(cases[3][0],p0,atol=scale*1e-12)
 amp_mutation=max(float(abs(v[1]-a0).max()) for v in cases);assert amp_mutation<1e-20
 shifted=w.extract(np.roll(make(0,0),(7,11),(0,1)),pad=0);np.testing.assert_allclose(shifted,cases[0],atol=1e-12)
 const=w.extract(np.full((256,256),.37));np.testing.assert_array_equal(const,np.zeros((2,15)))
 edge=(xx>87+.23*yy).astype(float)+.3*((xx-160)**2+(yy-91)**2<27**2)+.13*np.sin(2*np.pi*(xx+2*yy)/32)
 base=np.array(w.extract(edge));errors=[]
 for k in range(4):
  for reflected in [False,True]:
   v=np.rot90(edge,k);v=v[:,::-1] if reflected else v;err=float(abs(np.array(w.extract(v))-base).max());errors.append(err)
 assert max(errors)<1e-10,max(errors)
 np.testing.assert_allclose(w.extract(2*edge),4*base,atol=1e-10);neg=np.array(w.extract(-edge));np.testing.assert_allclose(neg[0],-base[0],atol=1e-10);np.testing.assert_allclose(neg[1],base[1],atol=1e-10)
 other=np.array(w.extract(edge,backend='scipy'));fft_error=float(abs(other-base).max());assert fft_error<1e-10
 # Independent coefficient/covariance formula using polar H2 and explicit pixel means.
 coeff=w.coefficients(edge);manual=[];am=[]
 for j in range(3):
  c=coeff[j];d=abs(coeff[j+1])*np.exp(2j*np.angle(coeff[j+1]));z=[];az=[]
  for offset in range(5):
   vv=[];av=[]
   for t in range(16):
    for sign in [-1,1]:
     r=(t+sign*offset)%16;u=c[t];v=d[r];vv.append(np.mean((u-u.mean())*np.conj(v-v.mean())));s=abs(c[t]);q=abs(coeff[j+1,r]);av.append(np.mean((s-s.mean())*(q-q.mean())))
   z.append(np.mean(vv).real);az.append(np.mean(av))
  manual.extend(z);am.extend(az)
 independent=float(abs(np.array([manual,am])-base).max());assert independent<1e-10
 rec={'passed':True,'dimensions_perarm':15,'H2_magnitude_and_phase_identities':True,'centered_covariance_explicit_error':cov_error,'phase_mutation_peak':float(scale),'phase_quadrature_zero_and_pi_sign_flip':True,'coherent_scale_phase_change_invariant':True,'same_amplitude_control_max_mutation':amp_mutation,'constant_exact_zero':True,'periodic_translation_invariance':True,'production_padding_D4_max_error':max(errors),'intensity_degree2_and_contrast_sign_verified':True,'NumPy_SciPy_FFT_error':fft_error,'independent_polar_covariance_error':independent,'production_edges_finite':True,'no_dataset_opened':True,'limits':'Pure periodic analytic fixture pad0, production D4 fixture pad64. Not arbitrary-angle invariance or scientific performance evidence.'};f.write(R/'synthetic-validation.json',rec);f.emit(rec)
if __name__=='__main__':main()
