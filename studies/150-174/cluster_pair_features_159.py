"""Finite-window C2/S2 of quantile binary proxies, not identified material phases."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from collections import deque
import numpy as np
from scipy import ndimage as ndi
QUANTILES=(.2,.5,.8)
RADII=(4,16,64)
STRUCTURE=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=bool)
NAMES=tuple(f'q{int(q*100)}_{side}_r{r}_{stat}' for q in QUANTILES for side in ('low','complement') for r in RADII for stat in ('axis_mean','axis_std'))

def slices(x,r,axis):
 return (x[:-r,:],x[r:,:]) if axis==0 else (x[:,:-r],x[:,r:])

def panel(mask,radii=RADII):
 mask=np.asarray(mask,dtype=bool);lab,num=ndi.label(mask,STRUCTURE);S=[];C=[];rawS=[];rawC=[]
 for r in radii:
  ss=[];cc=[]
  for axis in (0,1):
   A,B=slices(mask,r,axis);L,K=slices(lab,r,axis);ss.append(float(np.mean(A&B)));cc.append(float(np.mean((L>0)&(L==K))))
  assert np.all(np.array(cc)<=np.array(ss))
  S.extend((np.mean(ss),np.std(ss)));C.extend((np.mean(cc),np.std(cc)));rawS.append(ss);rawC.append(cc)
 return np.array(S),np.array(C),{'components':int(num),'fraction':float(mask.mean()),'rawS':rawS,'rawC':rawC}

def independent(mask,radii=RADII):
 """BFS labeling and explicit coordinate gathers, independent of ndi and slices."""
 mask=np.asarray(mask,bool);h,w=mask.shape;lab=np.zeros((h,w),np.int32);num=0
 for y,x in np.argwhere(mask):
  if lab[y,x]:continue
  num+=1;lab[y,x]=num;queue=deque([(int(y),int(x))])
  while queue:
   yy,xx=queue.popleft()
   for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
    a,b=yy+dy,xx+dx
    if 0<=a<h and 0<=b<w and mask[a,b] and not lab[a,b]:lab[a,b]=num;queue.append((a,b))
 S=[];C=[]
 for r in radii:
  ss=[];cc=[]
  for dy,dx in ((r,0),(0,r)):
   y,x=np.indices((h-dy,w-dx));aa=mask[y,x];bb=mask[y+dy,x+dx];la=lab[y,x];lb=lab[y+dy,x+dx];den=(h-dy)*(w-dx)
   ss.append(np.count_nonzero(aa&bb)/den);cc.append(np.count_nonzero((la!=0)&(la==lb))/den)
  S.extend((sum(ss)/2,abs(ss[0]-ss[1])/2));C.extend((sum(cc)/2,abs(cc[0]-cc[1])/2))
 return np.array(S),np.array(C),lab

def extract(image):
 image=np.asarray(image);assert image.shape==(256,256) and image.dtype==np.uint8
 ts=np.quantile(image,QUANTILES,method='linear');S=[];C=[];meta=[]
 for t in ts:
  m=image<=t
  for mask in (m,~m):
   p,c,z=panel(mask);S.extend(p);C.extend(c);meta.append(z)
 S=np.array(S);C=np.array(C);assert S.shape==C.shape==(36,) and np.isfinite(S).all() and np.isfinite(C).all()
 return S,C,{'thresholds':ts.tolist(),'masks':meta}

def fixtures():
 from scipy.signal import correlate2d
 from skimage.measure import euler_number
 P=np.array([[3,1],[3,0],[1,3],[2,4],[2,3],[0,6],[3,3],[3,2],[1,5]])
 Q=np.array([[1,2],[1,3],[3,0],[0,5],[0,6],[2,3],[1,4],[1,5],[3,2]])
 ms=[];sizes=[];b=[]
 for points in (P,Q):
  m=np.zeros((7,7),bool);m[tuple(points.T)]=1;ms.append(m);lab,num=ndi.label(m,STRUCTURE);sizes.append(sorted(np.bincount(lab.ravel())[1:].tolist()));b.append([int(num),int(num-euler_number(m,connectivity=1))])
 np.testing.assert_array_equal(correlate2d(ms[0].astype(int),ms[0].astype(int),'full'),correlate2d(ms[1].astype(int),ms[1].astype(int),'full'));assert sizes[0]==sizes[1] and b[0]==b[1]
 curves=[]
 for m in ms:
  big=np.pad(np.repeat(np.repeat(m,2,0),2,1),((0,242),(0,242)));ss,cc,_=panel(big);curves.append((ss,cc))
 np.testing.assert_array_equal(curves[0][0],curves[1][0]);assert not np.array_equal(curves[0][1],curves[1][1])
 cases=[np.zeros((9,9),bool),np.ones((9,9),bool),np.indices((9,9)).sum(0)%2==0,*ms]
 U=np.zeros((9,9),bool);U[1:8,1]=1;U[1:8,7]=1;U[7,1:8]=1;cases.append(U)
 L,_=ndi.label(U,STRUCTURE);assert L[1,1]==L[1,7]>0 and not U[1,1:8].all()
 maxerr=0.
 for mask in cases:
  S,C,_=panel(mask,(1,2,4));p,c,lab=independent(mask,(1,2,4));maxerr=max(maxerr,float(abs(S-p).max()),float(abs(C-c).max()));assert maxerr<1e-15
  for k in range(4):
   for mm in (np.rot90(mask,k),np.rot90(mask,k)[:,::-1]):
    sr,cr,_=panel(mm,(1,2,4));np.testing.assert_array_equal(S,sr);np.testing.assert_array_equal(C,cr)
  for r in (1,2,4):
   for axis in (0,1):
    aa,bb=slices(lab,r,axis);den=aa.size;cc=np.count_nonzero((aa>0)&(aa==bb))/den
    windows=[mask[j:mask.shape[0]-r+j,:] if axis==0 else mask[:,j:mask.shape[1]-r+j] for j in range(r+1)]
    lineal=np.logical_and.reduce(windows).mean();assert lineal<=cc+1e-15
 checker=cases[2];_,C,_=panel(checker,(2,4));assert not C.any()
 for value in (0,255):
  S,C,_=extract(np.full((256,256),value,np.uint8));np.testing.assert_array_equal(S,C);assert np.isfinite(C).all()
 return {'passed':True,'BFS_independent_max_error':maxerr,'D4_exact':True,'empty_full_ties_checkerboard_boundary_tested':True,'straight_line_implies_4connected':True,'U_path_connected_without_straight_line':True,'homometric':{'P':P.tolist(),'Q':Q.tolist(),'all_displacement_S2_equal':True,'component_area_multisets':sizes,'zero_level_Betti':b,'scaled_fixture_same_proposed_S2_different_C2':True,'C2_r4_axis_mean':[float(z[1][0]) for z in curves]},'limits':'Does not prove independence from all028shape summaries or all055distance-Betti or all130S3 features jointly. Homometric fixture proves S2 plus sizes/counts/zero-levelBetti insufficient. Finite sampledpoint occupancy is not a path test.'}
