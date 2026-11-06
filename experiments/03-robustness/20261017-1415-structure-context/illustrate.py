"""Label-blind held-out context illustration; never used for fitting/selection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import encoder as e
import numpy as np,torch,json,sys
from PIL import Image,ImageDraw
def grid(a):return a.reshape(16,16,4,4).transpose(0,2,1,3).reshape(64,64)
def main():
    z=np.load(_repo_io(e.OLD/'inputs.npz'));g=json.loads((e.OLD/'graphs.json').read_text())[0];B=np.array(g['B']);chosen=B[z['noise'][B]>=12][:3];path=e.HERE/'encoders/fold0-selector-seed0-spatial/model.pt';ck=torch.load(_repo_io(path),map_location='cpu',weights_only=True);meta=ck['normalization'];m=e.ContextNet().eval();m.load_state_dict(ck['state_dict']);assert set(z['ids'][chosen]).isdisjoint(meta['train_ids']);canvas=Image.new('RGB',(1280,935),'white');draw=ImageDraw.Draw(canvas)
    titles=['Original256','Hidden source regions','Visible-mean reference','Context prediction','Observed4x4-cell target']
    for c,title in enumerate(titles):draw.text((c*256+5,5),title,fill='black')
    records=[]
    for row,j in enumerate(chosen):
        with Image.open(_repo_path(e.ROOT, f"data/train/{z['ids'][j]}.png")) as im:px=np.asarray(im,np.uint8)[None,None]
        x=torch.from_numpy(e.normalize(px,meta));perm=torch.arange(256)[None];mb=e.mask_batch(1,np.random.default_rng(81173+int(j)));mask=torch.from_numpy(mb)
        with torch.no_grad():pred,_=m(x,perm,mask);target=e.targets(x,perm);vals=e.losses(pred,target,mask,True).numpy()[0]
        p=grid(pred.numpy()[0])*meta['image_std']+meta['image_mean'];t=grid(target.numpy()[0])*meta['image_std']+meta['image_mean'];mg=mb.reshape(16,16).repeat(4,0).repeat(4,1);visible=float(t[~mg].mean());reference=np.where(mg,visible,t);reconstruction=np.where(mg,p,t);source=px[0,0]/255.;rawmask=mb.reshape(16,16).repeat(16,0).repeat(16,1)
        images=[source,np.where(rawmask,.5,source),reference,reconstruction,t]
        for c,a in enumerate(images):
            pic=Image.fromarray(np.round(np.clip(a,0,1)*255).astype(np.uint8)).resize((256,256),Image.Resampling.NEAREST).convert('RGB');canvas.paste(pic,(c*256,35+row*300))
        draw.text((5,294+row*300),f"{z['ids'][j]} | noise {z['noise'][j]:.2f} | B source, excluded from encoder training | visible cells are observed in both filled panels",fill='black')
        base=torch.full_like(target,visible);base=(base-meta['image_mean'])/meta['image_std'];ref=e.losses(base,target,mask,True).numpy()[0];records.append(dict(ID=str(z['ids'][j]),source_index=int(j),noise=float(z['noise'][j]),DC_AC_gradient_errors=vals.tolist(),reference_errors=ref.tolist(),selection='First3 noisy IDs in fixed fold0 B order, no hardness/error selection'))
    canvas.save(_repo_io(e.HERE/'context-illustration.png'));e.save(_repo_io('context-illustration.json'),dict(records=records,model_sha256=e.sha(path),labels_read=False,selection_use=False,scope='Qualitative held-out image-field prediction; not grain truth. Visible cells copied from observation in reference/reconstruction panels. Quantization/clipping for display only.'))
    print('SAVED label-blind context illustration', [r['ID'] for r in records])
if __name__=='__main__':
    torch.set_num_threads(2);sys.addaudithook(e.guard)
    with e.threadpool_limits(limits=2):main()
