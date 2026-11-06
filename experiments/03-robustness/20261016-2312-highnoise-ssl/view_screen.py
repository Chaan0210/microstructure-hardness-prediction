"""Scientific comparison of actual permitted source-image transformations."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import ast,json,hashlib
import numpy as np
from scipy import ndimage as ndi
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3];OLD=_repo_path(ROOT, 'experiments/highnoise-relational-20261016')
def main():
    data=np.load(_repo_path(OLD, 'inputs.npz'));g=json.loads((_repo_path(OLD, 'graphs.json')).read_text())[0];A=np.array(g['A']);chosen=[int(A[np.flatnonzero(data['noise'][A]<10)[0]]),int(A[np.flatnonzero(data['noise'][A]>=12)[0]])]
    source=_repo_path(HERE, 'study.py');tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['regions','photo','boundary','edge_agreement']];ns=dict(np=np,ndi=ndi);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    pix=np.stack([np.asarray(Image.open(_repo_path(ROOT, f"data/train/{data['ids'][j]}.png")),np.uint8)[None] for j in chosen]);mask,means,sd,tile=ns['regions'](pix)
    canvas=Image.new('RGB',(1280,630),'white');draw=ImageDraw.Draw(canvas);records=[];titles=['Original source','Local photo .03','Photo absolute change x25','Boundary blend .40','Boundary absolute change x4']
    for col,title in enumerate(titles):draw.text((col*256+5,5),title,fill='black')
    for row,j in enumerate(chosen):
        x=pix[row,0].astype(np.float32)/255;y=ns['photo'](x[None,None],mask[row:row+1],means[row:row+1],sd[row:row+1],tile,np.random.default_rng(2026101672+j),.03)[0,0];z=ns['boundary'](x,.4)
        for col,img in enumerate([x,y,np.abs(y-x)*25,z,np.abs(z-x)*4]):canvas.paste(Image.fromarray(np.round(np.clip(img,0,1)*255).astype(np.uint8)).convert('RGB'),(col*256,35+row*300))
        text=f"{data['ids'][j]} | noise proxy {data['noise'][j]:.3f} | first eligible fold0 A source; selected without hardness or model errors"
        draw.text((5,295+row*300),text,fill='black');r=dict(ID=str(data['ids'][j]),source_index=j,noise=float(data['noise'][j]),selection='First eligible clean/high-noise source in fold0 A order, label blind')
        for name,v in [('photo',y),('boundary',z)]:r[name]=dict(pixel_RMS_gray255=float(np.sqrt(np.mean((v-x)**2))*255),max_absolute_gray255=float(np.max(abs(v-x))*255),changed_fraction=float(np.mean(abs(v-x)>1e-7)),edge_membership_Jaccard=ns['edge_agreement'](x,v))
        records.append(r)
    path=_repo_path(HERE, 'transformation-screen.png');assert not path.exists();canvas.save(_repo_io(path))
    with (_repo_path(HERE, 'transformation-screen.json')).open('x') as f:json.dump(dict(records=records,transform_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),warning='Displayed pixels quantized only for illustration. Actual training uses float views. Edge masks are proxies, not ground-truth boundaries.'),f,indent=2)
    print(json.dumps(records,indent=2))
if __name__=='__main__':main()
