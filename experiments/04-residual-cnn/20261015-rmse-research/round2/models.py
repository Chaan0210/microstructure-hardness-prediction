"""Strict-source H5/H6 experiments; all writes are local to round2."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse
import json
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import research as r
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits


def inputs():
    d=dict(np.load(_repo_io(r.HERE/'inputs.npz')))
    d['strict_base']=np.load(_repo_io(r.HERE/'scope.npz'))['strict_blend']
    d['strict_ridge']=np.load(_repo_io(r.HERE/'consistency-strict.npz'))['production']
    d['strict_logridge']=np.load(_repo_io(r.HERE/'scope.npz'))['strict']
    return d


def groups(keys):
    definitions=json.loads((r.H13/'abl/groups.json').read_text())
    assignment={col:name for name,cols in definitions.items() for col in cols.split()}
    names=[assignment.get(str(k),'misc') for k in keys]
    levels=sorted(set(names))
    return np.array([levels.index(k) for k in names]),levels


class EvidenceRidge:
    def __init__(self,group_ids,max_iter=100): self.groups=np.asarray(group_ids);self.max_iter=max_iter

    def fit(self,x,y):
        self.imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(x)
        self.scaler=StandardScaler().fit(self.imputer.transform(x))
        z=self.scaler.transform(self.imputer.transform(x)); self.mu=float(y.mean());self.sd=max(float(y.std()),1e-8)
        t=(y-self.mu)/self.sd; gram=z.T@z; rhs=z.T@t
        ng=int(self.groups.max())+1; precision=np.ones(ng); beta=1.
        gp=[np.where(self.groups==g)[0] for g in range(ng)]
        trace=[]
        for iteration in range(self.max_iter):
            h=beta*gram+np.diag(precision[self.groups])
            chol=cho_factor(h,check_finite=False)
            covariance=cho_solve(chol,np.eye(z.shape[1]),check_finite=False)
            coef=beta*cho_solve(chol,rhs,check_finite=False)
            gamma=np.array([np.clip(len(j)-precision[g]*np.diag(covariance)[j].sum(),0,len(j)) for g,j in enumerate(gp)])
            energies=np.array([np.dot(coef[j],coef[j]) for j in gp])
            anew=np.clip((gamma+2e-6)/(energies+2e-6),1e-6,1e6)
            rss=np.sum((t-z@coef)**2)
            bnew=float(np.clip((len(t)-gamma.sum()+2e-6)/(rss+2e-6),1e-6,1e6))
            change=max(np.max(np.abs(np.log(anew/precision))),abs(np.log(bnew/beta)))
            trace.append(float(change)); precision=np.sqrt(precision*anew); beta=np.sqrt(beta*bnew)
            if change<1e-4: break
        self.precision=precision; self.beta=float(beta);self.iterations=iteration+1;self.last_change=trace[-1]
        self.coef=beta*cho_solve(cho_factor(beta*gram+np.diag(precision[self.groups]),check_finite=False),rhs,check_finite=False)
        return self

    def predict(self,x):
        z=self.scaler.transform(self.imputer.transform(x))
        return self.mu+self.sd*(z@self.coef)


class PLSQuadratic:
    def __init__(self,alpha,scale):self.alpha,self.scale=alpha,scale

    @staticmethod
    def quadratic(v):return np.column_stack([v[:,i]*v[:,j] for i in range(v.shape[1]) for j in range(i,v.shape[1])])

    def fit(self,x,y):
        self.imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(x)
        self.scaler=StandardScaler().fit(self.imputer.transform(x));z=self.scaler.transform(self.imputer.transform(x))
        if self.scale:
            self.pls=PLSRegression(n_components=4,scale=False).fit(z,(y-y.mean())/max(y.std(),1e-8))
            v=self.pls.transform(z);self.vscale=StandardScaler().fit(v);v=self.vscale.transform(v)
            q=self.quadratic(v);self.qscale=StandardScaler().fit(q)
            z=np.c_[z,self.scale*self.qscale.transform(q)]
        self.ridge=Ridge(alpha=self.alpha).fit(z,y)
        return self

    def predict(self,x):
        z=self.scaler.transform(self.imputer.transform(x))
        if self.scale:
            v=self.vscale.transform(self.pls.transform(z));q=self.qscale.transform(self.quadratic(v));z=np.c_[z,self.scale*q]
        return self.ridge.predict(z)


def pls_select(x,y,source,isreal,domain,noisy):
    grid=[(a,s) for s in (0.,.1,1.) for a in r.ALPHAS];errors=np.zeros(len(grid));n=0
    for tr,va in GroupKFold(3,shuffle=True,random_state=r.SEED).split(x,y,source):
        assert not set(source[tr])&set(source[va])
        va=va[isreal[va] & (domain[va]==int(noisy))]
        for j,(alpha,scale) in enumerate(grid):
            m=PLSQuadratic(alpha,scale).fit(x[tr],y[tr]);errors[j]+=np.sum((m.predict(x[va])-y[va])**2)
        n+=len(va)
    choices=[int(np.argmin(errors[:4])),int(np.argmin(errors))]
    models=[PLSQuadratic(*grid[j]).fit(x,y) for j in choices]
    return models,dict(null=grid[choices[0]],selected=grid[choices[1]],grid=grid,inner_mse=(errors/n).tolist())


def run(family,max_iter=100):
    start=time.monotonic();d=inputs();dev=d['dev'];y=d['y'][dev];fold=d['fold'];base=d['strict_base']
    names=('isotropic','group','permuted_group') if family=='evidence' else ('null','quadratic')
    out={k:d['strict_ridge'].copy() for k in names};details=[]
    for p in range(3):
        for k in range(5):
            train,valid=dev[fold[p]!=k],dev[fold[p]==k]
            for noisy in (False,True):
                x,yy,dom,source,isreal,pairs,cols=r.training(d,train,valid,noisy,True)
                ii=np.where((fold[p]==k)&((d['noise'][dev]>=12)==noisy))[0];query=d['X'][dev[ii]][:,cols]
                if family=='evidence':
                    g,levels=groups(d['keys'][cols]);random=np.random.default_rng(r.SEED).permutation(g)
                    models=[EvidenceRidge(gg,max_iter=max_iter).fit(x,yy) for gg in (np.zeros_like(g),g,random)]
                    detail={name:dict(precision=m.precision.tolist(),beta=m.beta,iterations=m.iterations,last_change=m.last_change)
                            for name,m in zip(names,models)};detail['groups']=levels
                else:models,detail=pls_select(x,yy,source,isreal,dom,noisy)
                for name,m in zip(names,models):out[name][p,ii]=m.predict(query)
                detail.update(partition=p,fold=k,noisy=noisy,n_sources=len(set(source)),n_rows=len(yy));details.append(detail)
            print(family,p,k,'seconds',round(time.monotonic()-start,1),flush=True)
    blends={key:np.where(d['fail'],base,base+.128*(pred-d['strict_ridge'])) for key,pred in out.items()}
    result={key:r.compare(base,pred,y,fold) for key,pred in blends.items()}
    if family=='evidence':
        result['group_vs_isotropic']=r.compare(blends['isotropic'],blends['group'],y,fold)
        result['group_vs_permuted']=r.compare(blends['permuted_group'],blends['group'],y,fold)
    else:result['quadratic_vs_null']=r.compare(blends['null'],blends['quadratic'],y,fold)
    result['details']=details;result['seconds']=time.monotonic()-start;result['seed']=r.SEED
    result['strict_reference_rmse']=r.rmse(base,y);result['code_sha256']=r.digest(__file__)
    tag=family if max_iter==100 else f'{family}-iter{max_iter}'
    np.savez_compressed(_repo_path(HERE, tag+'.npz'),**out,**{k+'_blend':v for k,v in blends.items()})
    r.save_json(_repo_path(HERE, tag+'.json'),result)
    print(json.dumps({k:{v:result[k][v] for v in ('gain','partition_gains','improved_folds','conditional_id_bootstrap_gain_ci95')}for k in names},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('family',choices=['evidence','pls']);ap.add_argument('--max-iter',type=int,default=100);args=ap.parse_args()
    tag=args.family if args.max_iter==100 else f'{args.family}-iter{args.max_iter}'
    if (_repo_path(HERE, tag+'.npz')).exists():raise FileExistsError(tag)
    with threadpool_limits(limits=2):run(args.family,args.max_iter)
