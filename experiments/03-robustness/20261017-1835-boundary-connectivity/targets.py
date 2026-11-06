"""Image-local uncertain auxiliary boundary/affinity targets, never true grains."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import graph as g
import numpy as np,json,sys,time
from PIL import Image
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score
PAIR, RADII=g.fixed_pairs(count=512)
def pool2(x):return x.reshape(128,2,128,2).mean((1,3))
def target_pairs(nodes,pairs,available):
    a=nodes[:,pairs[:,0]];b=nodes[:,pairs[:,1]];valid=(a>0)&(b>0);n=valid.sum(0);p=((a==b)&valid).sum(0)/np.maximum(n,1)
    confident=(n>=max(1,int(np.ceil(.75*available))))&((p>=.75)|(p<=.25));weight=confident*np.abs(2*p-1)*n/max(available,1)
    return (p>=.5).astype(np.float32),weight.astype(np.float32)
def make(image,masks):
    nodes=np.zeros((4,1024),np.uint16);bs=[];support=[];available=len(masks)
    for v,m in enumerate(masks):
        lo=ndi.minimum_filter(m,3);hi=ndi.maximum_filter(m,3);interior=(lo==hi)&(m>0);n=m[3::8,3::8].copy();ok=np.ones((32,32),bool)
        for y,x in [(3,3),(3,4),(4,3),(4,4)]:ok&=interior[y::8,x::8]&(m[y::8,x::8]==n)
        n[~ok]=0;nodes[v]=n.ravel();bs.append(pool2(((lo!=hi)&(hi>0)).astype(float)));support.append(pool2((hi>0).astype(float)))
    b=np.mean(bs,0);sup=np.mean(support,0);valid=(sup>=.75)&((b>=.75)|(b<=.1));by=(b>=.5).astype(np.float32)
    sm=ndi.gaussian_filter(image.astype(float),1.);evidence=ndi.grey_closing(sm,size=(7,7))-sm+2*ndi.gaussian_gradient_magnitude(sm,.7);ev=pool2(evidence);lo,hi=np.quantile(ev,[.5,.95]);rank=np.clip((ev-lo)/max(hi-lo,1e-6),0,1)
    bw=(valid*np.abs(2*b-1)*(1-by+by*(.25+.75*rank))).astype(np.float32);bw[:2]=0;bw[-2:]=0;bw[:,:2]=0;bw[:,-2:]=0
    ey,ew=target_pairs(nodes,np.c_[g.U,g.V],available);py,pw=target_pairs(nodes,PAIR,available)
    positive=(by>0)&(bw>0);negative=(by==0)&(bw>0);auc=float(roc_auc_score(by[valid].ravel(),ev[valid].ravel())) if len(np.unique(by[valid]))==2 else .5
    audit=dict(views=available,node_support=float((nodes>0).sum()/(available*1024)),boundary_positive_fraction=float(positive.mean()),boundary_negative_fraction=float(negative.mean()),edge_known_fraction=float((ew>0).mean()),edge_positive_count=int(((ey>0)&(ew>0)).sum()),edge_negative_count=int(((ey==0)&(ew>0)).sum()),pair_positive_count=int(((py>0)&(pw>0)).sum()),pair_negative_count=int(((py==0)&(pw>0)).sum()),boundary_evidence_AUC=auc,positive_evidence_mean=float(ev[positive].mean()) if positive.any() else 0.,negative_evidence_mean=float(ev[negative].mean()) if negative.any() else 0.)
    if available==4:
        held=nodes[3:4];ty,tw=target_pairs(nodes[:3],PAIR,3);hy,hw=target_pairs(held,PAIR,1);v=(tw>0)&(hw>0);acc=[]
        for label in [0,1]:
            ok=v&(ty==label)
            if ok.any():acc.append(float((ty[ok]==hy[ok]).mean()))
        audit['withheld_view_balanced_agreement']=float(np.mean(acc)) if len(acc)==2 else None;audit['withheld_view_pairs']=int(v.sum())
    return dict(nodes=nodes,boundary=by,boundary_weight=bw,edge_target=ey,edge_weight=ew,pair_target=py,pair_weight=pw,available=np.int64(available)),audit
def sources():
    base=_repo_path(c.ROOT, 'experiments/grain-cellpose-220');m=np.load(_repo_io(base/'masks-v3-train.npz'));t=np.load(_repo_io(base/'masks-tta-train.npz'));mi={str(x):j for j,x in enumerate(m['ids'])};ti={str(x):j for j,x in enumerate(t['ids'])}
    def get(id):
        masks=[m['masks'][mi[id]]]
        if id in ti:
            j=ti[id];masks.extend([t['T'][j].T,t['UD'][j][::-1],np.rot90(t['R90'][j],-1)])
        return masks
    return get,{'base':c.sha(base/'masks-v3-train.npz'),'TTA':c.sha(base/'masks-tta-train.npz')}
def preflight():
    d=c.inputs();A=np.array(d['graphs'][0]['A']);sel=np.r_[A[d['noise'][A]>=12][:8],A[d['noise'][A]<12][:8]];get,hashes=sources();records=[];panels=[];start=time.monotonic()
    for j in sel:
        id=str(d['ids'][j]);im=np.asarray(Image.open(_repo_path(c.ROOT, f'data/train/{id}.png')));z,a=make(im,get(id));a.update(ID=id,index=int(j),high=bool(d['noise'][j]>=12));records.append(a);panels.append((im,z['boundary'],z['boundary_weight'],z['nodes'][0].reshape(32,32)))
    support=np.mean([min(a[k] for k in ['edge_positive_count','edge_negative_count','pair_positive_count','pair_negative_count'])>0 for a in records]);agreement=[a['withheld_view_balanced_agreement'] for a in records if a['high'] and a.get('withheld_view_balanced_agreement') is not None];passed=bool(support>=.8 and len(agreement)>=4 and np.median(agreement)>=.7)
    c.save(_repo_io('target-preflight.json'),dict(passed=passed,balanced_source_fraction=float(support),high_withheld_view_median=float(np.median(agreement)) if agreement else None,sources=records,input_sha256=hashes,source_sha256=c.sha(__file__),seconds=time.monotonic()-start,hardness_read=False,physical_truth=False))
    panel();print('TARGET PREFLIGHT',passed,'balanced',support,'held-view',np.median(agreement) if agreement else None,flush=True)
def panel():
    from PIL import ImageDraw
    records=json.loads((c.HERE/'target-preflight.json').read_text())['sources'];get,_=sources();canvas=Image.new('RGB',(1024,6*280+28),'white');draw=ImageDraw.Draw(canvas)
    for col,title in enumerate(['Original','Weak boundary target','Supervision weight','Base-mask grid nodes']):draw.text((col*256+4,5),title,fill='black')
    for row,j in enumerate([0,1,2,8,9,10]):
        record=records[j];im=np.asarray(Image.open(_repo_path(c.ROOT, f"data/train/{record['ID']}.png")));z,_=make(im,get(record['ID']));node=z['nodes'][0].reshape(32,32);colors=np.random.default_rng(91).integers(40,240,(int(node.max())+1,3),dtype=np.uint8);colors[0]=0
        arrays=[im,np.uint8(255*z['boundary']),np.uint8(255*z['boundary_weight']),colors[node]]
        for col,x in enumerate(arrays):canvas.paste(Image.fromarray(x).convert('RGB').resize((256,256),Image.Resampling.NEAREST),(col*256,row*280+48))
        draw.text((4,row*280+30),record['ID']+(' high' if record['high'] else ' low'),fill='black')
    path=c.HERE/'target-preflight.png';assert not path.exists();canvas.save(_repo_io(path));print('TARGET PANEL SAVED',flush=True)
def prepare():
    assert json.loads((c.HERE/'target-preflight.json').read_text())['passed'];d=c.inputs();get,hashes=sources();rows=[];aud=[];images=[];start=time.monotonic()
    for j in d['dev']:
        id=str(d['ids'][j]);im=np.asarray(Image.open(_repo_path(c.ROOT, f'data/train/{id}.png')));z,a=make(im,get(id));rows.append(z);images.append(im);a.update(ID=id,index=int(j),high=bool(d['noise'][j]>=12));aud.append(a)
    c.arrays('targets.npz',dev=d['dev'],ids=d['ids'][d['dev']],pixels=np.array(images)[:,None],pairs=PAIR,radii=RADII,**{k:np.stack([z[k] for z in rows]) for k in rows[0]});c.save(_repo_io('target-audit.json'),dict(sources=aud,input_sha256=hashes,source_sha256=c.sha(__file__),seconds=time.monotonic()-start,hardness_read=False,source_local_parameters_only=True));print('TARGETS PREPARED',len(rows),round(time.monotonic()-start,1),flush=True)
if __name__=='__main__':
    sys.addaudithook(c.guard)
    with c.threadpool_limits(limits=2):globals()[sys.argv[1]]()
