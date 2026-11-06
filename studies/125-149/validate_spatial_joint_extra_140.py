"""Additional fixture and explicit heldout-target intervention; frozen study unchanged."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import spatial_joint_140 as q
import numpy as np,joblib
q.torch.set_num_threads(2);q.check()
z=np.random.default_rng(20271203).normal(size=(5,5,32));g,s=q.stats(z)
edges=[np.outer(z[i,j]-z[ii,jj],z[i,j]-z[ii,jj]) for i in range(5) for j in range(5) for ii,jj in [(i+1,j),(i,j+1)] if ii<5 and jj<5];manual=np.mean(edges,0)/2;np.testing.assert_allclose(manual[np.triu_indices(32)],s,atol=1e-12,rtol=0)
# Joint offdiagonal sensitivity: same marginal energies, changed joint energy.
f=np.zeros((3,3,32));f[:,:,0]=np.arange(3)[None,:];f[:,:,1]=f[:,:,0];h=f.copy();h[:,:,1]*=-1;gf,sf=q.stats(f);gh,sh=q.stats(h);assert sf[1]==-sh[1] and sf[1]!=0
flat=z.reshape(-1,32);perm=flat[np.random.default_rng(7).permutation(len(flat))].reshape(z.shape);gp,sp=q.stats(perm);np.testing.assert_allclose(gp,g,atol=1e-12,rtol=0);assert np.max(abs(sp-s))>1e-3
checks=0
for split in q.e.SPLITS:
 d,zz=q.load(split);dd,_=q.x.replace(d,split,0,q.x.OUT/split/'cnn-fold-0',False);A,B,P,short,_=dd['cells'][0,0];anchor,c=q.a.affine(q.a.mixture(dd['deepP'][0,0],q.x.ARM),d['y'][A],q.a.mixture(P,q.x.ARM));anchor=.9*anchor+.1*short;pert=d['y'].copy();pert[B]+=50000;pp,cc=q.a.affine(q.a.mixture(dd['deepP'][0,0],q.x.ARM),pert[A],q.a.mixture(P,q.x.ARM));assert cc==c;np.testing.assert_array_equal(.9*pp+.1*short,anchor)
 for arm in q.ARMS:
  D=q.augmented(dd,zz,arm);bp=q.OUT/split/f'{arm}-fold-0/block-0';br=q.e.read(bp/'report.json');bz=q.a.readnp(bp/'predictions.npz',br['predictions_sha256']);r=bz['residual'].copy();T=np.array(br['splits'][0]['train_indices']);V=np.array(br['splits'][0]['validation_indices']);lookup={int(i):j for j,i in enumerate(B)};ti=np.array([lookup[int(i)] for i in T]);vi=np.array([lookup[int(i)] for i in V]);r[vi]+=123456
  for kind in q.a.n.KINDS:
   m=q.a.n.fit(D['X'][T],r[ti],D['blocks'],kind);old=joblib.load(_repo_io(bp/f'inner-0-{kind}.joblib'))['model'];np.testing.assert_array_equal(q.a.n.predict(m,D['X'][V]),q.a.n.predict(old,D['X'][V]));checks+=1
out={'passed':True,'synthetic_explicit_edge_fixture':True,'offdiagonal_not_marginal_variogram':True,'unordered_permutation_invariant_spatial_sensitive':True,'heldout_target_poison_fixed_fits_exact':checks,'anchor_excludes_B_label_poison_exact':True,'source_sha256':q.e.sha(Path(__file__))};q.e.write(q.OUT/'extra-validation.json',out);q.e.emit(out)
