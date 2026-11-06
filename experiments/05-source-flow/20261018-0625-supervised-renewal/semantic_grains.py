"""Shared soft pore/phase assignment learned from whole-image hardness only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=_RepoPath(__file__).resolve().parents[3]
KEYS=['pore_area','phase_area','phase_solid_fraction','pore_count','phase_count','log_grain_count','grain_logarea','grain_logarea_sd','grain_inv_bbox','grain_inv_sqrt_eqd','grain_logar','grain_c4','grain_s4','phase_size_cov','phase_shape_cov','phase_logarea','pore_logarea','phase_noise','phase_heterogeneity','pore_heterogeneity','assignment_entropy']

def load(d):
    b=np.load(_repo_path(ROOT, 'experiments/probabilistic-grains-900/object-bank-train.npz'));pos={str(v):i for i,v in enumerate(b['ids'])};ok=b['object_keys'].tolist();gk=b['global_keys'].tolist();nmax=int(np.diff(b['offsets']).max());N=len(d['ids']);data={k:np.zeros((N,nmax),np.float32) for k in ['area','valid','logarea','logar','invbbox','inveqd','cos4','sin4','phase_logit','pore_logit']};inputs=np.zeros((N,nmax,17),np.float32);tile=np.zeros((N,16,nmax),np.float32);noise=[]
    for row,ident in enumerate(d['ids']):
        j=pos[str(ident)];a,e=b['offsets'][j:j+2];x=b['objects'][a:e].astype(float);n=len(x);g=b['global_X'][j];get=lambda key:x[:,ok.index(key)];span=max(float(g[gk.index('contrast_span')]),10.);bright=float(g[gk.index('ref_bright')]);z=get('darkness')+.5*get('shade_offset')/span;ar=get('log_ar');la=get('log_area');area=get('area').copy();area/=area.sum();c=get('cos2');s=get('sin2');pf=.02+.96*get('pore_flag')
        vals={'area':area,'valid':np.ones(n),'logarea':la,'logar':ar,'invbbox':1/np.maximum(get('bbox_max'),1),'inveqd':(4*get('area')/np.pi)**(-.25),'cos4':c*c-s*s,'sin4':2*c*s,'phase_logit':(z-.3)/.06,'pore_logit':np.log(pf/(1-pf))}
        for key,v in vals.items():data[key][row,:n]=v
        q=(bright-x[:,[ok.index('intensity_'+v) for v in ['q10','q25','q50','q75','q90']]])/span
        inputs[row,:n]=np.c_[la,ar,get('solidity'),get('boundary_frac'),get('border'),get('within_gradient')/span,get('intensity_std')/span,z,get('shade_darkness'),q,np.full(n,float(g[gk.index('noise_gray')])/span),get('pore_flag'),get('whole_image_fallback')]
        bins=np.clip((get('cx')*4).astype(int),0,3)+4*np.clip((get('cy')*4).astype(int),0,3)
        for k in range(16):tile[row,k,:n]=area*(bins==k)
        noise.append(float(g[gk.index('noise_gray')])/span)
    data.update(inputs=inputs,tile=tile,noise=np.array(noise,np.float32));return {k:torch.tensor(v) for k,v in data.items()}

class Measurement(nn.Module):
    def __init__(self,data,base,train):
        super().__init__();self.data=data;self.base=base;self.down=nn.Linear(data['inputs'].shape[2],16);self.up=nn.Linear(16,2);nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
        w=data['valid'][train]/data['valid'][train].sum(1,keepdim=True);X=data['inputs'][train];mean=(X*w[:,:,None]).sum(1).mean(0);variance=(X.square()*w[:,:,None]).sum(1).mean(0)-mean.square();self.register_buffer('object_mean',mean);self.register_buffer('object_scale',variance.clamp(min=.0001).sqrt())
        with torch.no_grad():f=self.extra(train);self.register_buffer('mean',f.mean(0));self.register_buffer('scale',f.std(0,correction=0).clamp(min=.01))
    def extra(self,ix):
        d={k:v[ix] for k,v in self.data.items() if torch.is_tensor(v)};x=((d['inputs']-self.object_mean)/self.object_scale).clamp(-6,6);delta=6*torch.tanh(self.up(torch.tanh(self.down(x)))/6);cw=d['valid']/d['valid'].sum(1,keepdim=True);self.penalty=(delta.square().mean(2)*cw).sum(1).mean()
        pore=torch.sigmoid(d['pore_logit']+delta[:,:,0]);phase=(1-pore)*torch.sigmoid(d['phase_logit']+delta[:,:,1]);matrix=(1-pore-phase).clamp(min=1e-8);w=d['area'];solid=(w*(1-pore)).sum(1).clamp(min=1e-6);pore_area=(w*pore).sum(1);phase_area=(w*phase).sum(1);wg=w*(1-pore)/solid[:,None]
        avg=lambda v:(wg*v).sum(1);mu=avg(d['logarea']);ar=avg(d['logar']);ps=phase_area/solid
        phase_cov=(w*phase*(d['logarea']-mu[:,None])).sum(1)/solid;shape_cov=(w*phase*(d['logar']-ar[:,None])).sum(1)/solid
        t=d['tile'];mass=t.sum(2).clamp(min=1e-8);pm=torch.einsum('bcn,bn->bc',t,phase)/mass;po=torch.einsum('bcn,bn->bc',t,pore)/mass;ent=-(pore*torch.log(pore+1e-8)+phase*torch.log(phase+1e-8)+matrix*torch.log(matrix))
        f=[pore_area,phase_area,ps,(cw*pore).sum(1),(cw*phase).sum(1),torch.log1p(((1-pore)*d['valid']).sum(1)),mu,(avg(d['logarea'].square())-mu.square()).clamp(min=1e-8).sqrt(),avg(d['invbbox']),avg(d['inveqd']),ar,avg(d['logar']*d['cos4']),avg(d['logar']*d['sin4']),phase_cov,shape_cov,(w*phase*d['logarea']).sum(1)/phase_area.clamp(min=1e-6),(w*pore*d['logarea']).sum(1)/pore_area.clamp(min=1e-6),phase_area*d['noise'],((pm-phase_area[:,None]).square()*mass).sum(1).clamp(min=1e-8).sqrt(),((po-pore_area[:,None]).square()*mass).sum(1).clamp(min=1e-8).sqrt(),(w*ent).sum(1)]
        return torch.stack(f,1)
    def forward(self,ix):return torch.cat([self.base[ix],((self.extra(ix)-self.mean)/self.scale).clamp(-6,6)],1)
