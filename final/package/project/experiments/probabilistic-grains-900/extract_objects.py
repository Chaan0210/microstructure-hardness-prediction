"""Original-image object bank from frozen zero-shot masks, with no learned targets.

One bank row per original train ID. Grains are elements of a bag, never
independently labelled hardness examples. No synthesized images or training rows.
"""
import argparse
import csv
from pathlib import Path
import time
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure
from common import HERE, ROOT, sha, destination, dump_json

OBJECT_KEYS = ['area','log_area','interior_n','intensity_mean','intensity_std','intensity_q10','intensity_q25','intensity_q50','intensity_q75','intensity_q90',
               'intensity_se2','darkness','shade_darkness','shade_offset','shade_se2','cx','cy','log_ar','cos2','sin2','bbox_max','bbox_ar','solidity','border',
               'boundary_frac','within_gradient','pore_flag','whole_image_fallback']


def weighted_quantile(values, weights, q):
    order = np.argsort(values, kind='stable'); c = np.cumsum(weights[order]); c /= c[-1]
    return float(values[order[min(np.searchsorted(c, q), len(order)-1)]])


def extract(im, mask):
    a = im.astype(np.float64); mask = mask.astype(np.int32)
    hh = (a[::2,::2]-a[::2,1::2]-a[1::2,::2]+a[1::2,1::2])/2
    noise = float(np.median(np.abs(hh))/.6745)
    sm = ndi.gaussian_filter(a, 1.0)
    gy,gx = np.gradient(sm); grad = np.hypot(gx,gy)
    inner = np.where(ndi.minimum_filter(mask,3)==ndi.maximum_filter(mask,3),mask,0)
    props = [p for p in measure.regionprops(mask) if p.area >= 12]
    fallback = not props
    if fallback:
        mask = np.ones(im.shape, np.int32); inner = mask.copy(); props = measure.regionprops(mask)
    labs=np.array([p.label for p in props]); A=np.array([p.area for p in props],float)
    Is=[]; St=[]; qs=[]; cnt=[]; gr=[]
    # Group interior values once; per-grain quantiles use original pixels.
    flat=inner.ravel(); order=np.argsort(flat,kind='stable'); sl=np.searchsorted(flat[order],labs); sr=np.searchsorted(flat[order],labs,side='right')
    pix=a.ravel(); gp=grad.ravel()
    for p,l,r in zip(props,sl,sr):
        inds=order[l:r]
        if len(inds)<3:
            coords=p.coords; inds=coords[:,0]*a.shape[1]+coords[:,1]
        vals=pix[inds]; Is.append(vals.mean()); St.append(vals.std());qs.append(np.quantile(vals,[.1,.25,.5,.75,.9]));cnt.append(len(vals));gr.append(gp[inds].mean())
    I=np.array(Is); sd=np.array(St);qq=np.array(qs);cnt=np.array(cnt,float);gr=np.array(gr)
    hi=weighted_quantile(I,A,.75);lo=float(np.percentile(sm,.1));span=max(hi-lo,10.)
    dark=(hi-I)/span;coords=np.array([p.centroid for p in props]);cy,cx=coords[:,0],coords[:,1]
    ar=np.array([max(p.axis_major_length/max(p.axis_minor_length,1),1.) for p in props]);theta=np.array([p.orientation for p in props]);sol=np.array([p.solidity for p in props])
    pore=(dark>.75)&(sol>.85)&(ar<2.5)&(A<3*np.median(A))
    bright=(dark<.15)&~pore
    if bright.sum()<3:
        bright=I>=np.quantile(I,.6)
    delta=(cx[:,None]-cx[bright][None])**2+(cy[:,None]-cy[bright][None])**2
    W=np.exp(-delta/(2*48**2));W[np.where(bright)[0],np.arange(bright.sum())]=0
    ws=W.sum(1);field=np.divide(W@I[bright],ws,out=np.full(len(I),hi),where=ws>1e-9)
    # Preserve image's bright-reference level; estimate only spatial variation.
    field += hi - np.average(field[bright],weights=A[bright])
    pW=np.divide(W,ws[:,None],out=np.zeros_like(W),where=ws[:,None]>1e-9)
    resid=I[bright]-field[bright]
    spread=max(float(np.median(np.abs(resid-np.median(resid)))/.6745),1.)
    field_se2=np.maximum((pW*pW).sum(1)*spread**2,1e-4)
    field_se2[ws<=1e-9]=spread**2
    shade_dark=(field-I)/span
    bbox=np.array([p.bbox for p in props]);bw=bbox[:,3]-bbox[:,1];bh=bbox[:,2]-bbox[:,0]
    bord=((bbox[:,0]==0)|(bbox[:,1]==0)|(bbox[:,2]==im.shape[0])|(bbox[:,3]==im.shape[1])).astype(float)
    X=np.column_stack([A,np.log(A),cnt,I,sd,qq,noise**2/np.maximum(cnt,1),dark,shade_dark,field-hi,field_se2,cx/255.,cy/255.,np.log(ar),np.cos(2*theta),np.sin(2*theta),np.maximum(bw,bh),np.maximum(bw,bh)/np.maximum(np.minimum(bw,bh),1),sol,bord,1-cnt/A,gr,pore.astype(float),np.full(len(A),float(fallback))])
    assert X.shape[1]==len(OBJECT_KEYS) and np.isfinite(X).all()
    w=A/A.sum();g=~pore;gw=A*g;gw=gw/max(gw.sum(),1)
    C=float(np.sum(gw*np.log(ar)*np.cos(2*theta)));S=float(np.sum(gw*np.log(ar)*np.sin(2*theta)));mag=np.hypot(C,S)
    la=np.log(A);meanla=float(np.sum(gw*la));lut=np.zeros(mask.max()+1);valid=np.zeros_like(lut);lut[labs]=la;valid[labs]=g
    vals=[]
    for u in range(4):
        for v in range(4):
            m=mask[u*64:(u+1)*64,v*64:(v+1)*64];vv=valid[m]
            if vv.sum()>0:vals.append(float((lut[m]*vv).sum()/vv.sum()))
    glob=dict(noise_gray=noise,contrast_span=span,ref_bright=hi,low_ref=lo,image_mean=float(a.mean()),image_std=float(a.std()),n_grains=float(len(A)),n_effective=float(1/(w*w).sum()),coverage=float(A.sum()/65536),pore_area=float(w[pore].sum()),log_area_mean=meanla,log_area_std=float(np.sqrt(np.sum(gw*(la-meanla)**2))),inv_bbox_max=float(np.sum(gw/np.maximum(bw,bh))),inv_sqrt_eqd=float(np.sum(gw*(4*A/np.pi)**-.25)),log_ar_mean=float(np.sum(gw*np.log(ar))),orient_c2=C,orient_s2=S,orient_magnitude=float(mag),orient_c4=float((C*C-S*S)/max(mag,1e-6)),orient_s4=float(2*C*S/max(mag,1e-6)),het4=float(np.std(vals)) if vals else 0.,phase_raw25=float(np.sum(gw*(dark>.25))),phase_raw50=float(np.sum(gw*(dark>.5))),phase_shade15=float(np.sum(gw*((field-I)>15))),intensity_between_sd=float(np.sqrt(np.sum(gw*(I-np.sum(gw*I))**2))),intensity_within_sd=float(np.sum(gw*sd)),border_area=float(np.sum(gw*bord)),shade_amplitude=float(np.std(field)),whole_image_fallback=float(fallback))
    return X.astype(np.float32),w.astype(np.float32),glob


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','test'],default='train');args=p.parse_args()
    split=args.split
    csvpath=ROOT/'data'/('train.csv' if split=='train' else 'sample_submission.csv')
    rows=list(csv.DictReader(csvpath.open()));ids=[r['ID'] for r in rows]
    mpath=ROOT/'experiments/grain-cellpose-220'/f'masks-v3-{split}.npz'
    with np.load(mpath,allow_pickle=False) as m:
        masks=m['masks'];mp={str(i):k for k,i in enumerate(m['ids'])}
    objs=[];weights=[];globs=[];offsets=[0];imagesha={};t=time.time();gkeys=None
    for n,i in enumerate(ids):
        ip=ROOT/'data'/split/f'{i}.png';imagesha[i]=sha(ip)
        im=np.asarray(Image.open(ip));assert im.shape==(256,256)
        X,w,g=extract(im,masks[mp[i]]);objs.append(X);weights.append(w);offsets.append(offsets[-1]+len(w))
        if gkeys is None:gkeys=list(g)
        assert list(g)==gkeys;globs.append([g[k] for k in gkeys])
        if (n+1)%50==0:print(f'{split}: {n+1}/{len(ids)} images, {offsets[-1]} grains, {time.time()-t:.1f}s',flush=True)
    out=dict(ids=np.array(ids),offsets=np.array(offsets),objects=np.concatenate(objs),object_keys=np.array(OBJECT_KEYS),weights=np.concatenate(weights),global_X=np.array(globs),global_keys=np.array(gkeys))
    if split=='train':out['y']=np.array([float(r['hardness']) for r in rows])
    target=destination(HERE/f'object-bank-{split}.npz');np.savez_compressed(target,**out)
    dump_json(HERE/f'object-bank-{split}-provenance.json',dict(mask_file=str(mpath.relative_to(ROOT)),mask_sha256=sha(mpath),images_sha256=imagesha,output_sha256=sha(target),n_images=len(ids),n_grains=offsets[-1],seconds=time.time()-t,adaptive_training=False,synthetic_training_examples=0,feature_scope='Per-image deterministic original-pixel measurements; frozen zero-shot mask cache; no across-image fit.',object_keys=OBJECT_KEYS,global_keys=gkeys))
    print('saved',target,flush=True)


if __name__=='__main__':main()
