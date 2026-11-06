"""Read train data and existing conditional OOF; write only this experiment."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import csv, hashlib, json
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = _RepoPath(__file__).resolve().parents[3]

def main():
    raw = np.load(_repo_path(ROOT, 'experiments/highnoise-relational-20261016/inputs.npz'))
    physical = np.load(_repo_path(ROOT, 'experiments/rmse-research-20261015/inputs.npz'))
    labels = {r['ID']: float(r['hardness']) for r in csv.DictReader((_repo_path(ROOT, 'data/train.csv')).open())}
    ids, y, dev = raw['ids'], raw['y'], raw['dev']
    assert np.array_equal(y, np.array([labels[i] for i in ids]))
    assert np.array_equal(ids, physical['ids'])
    p = _repo_path(ROOT, 'experiments/probabilistic-grains-900/base.npz')
    base = np.load(_repo_io(p), allow_pickle=True)
    assert np.array_equal(ids[dev], base['ids'].astype(str))
    a = np.load(_repo_path(ROOT, 'experiments/push-1300/dev-079.npy'))
    b = np.load(_repo_path(ROOT, 'experiments/push-1300/dev-078.npy'))
    assert a.shape == b.shape == (3, 400)
    p83 = raw['base082'] + a[0] - b[0]
    pixels = np.empty((500, 1, 256, 256), np.uint8)
    image_hashes = {}
    for j, ident in enumerate(ids):
        path = _repo_path(ROOT, f'data/train/{ident}.png')
        with Image.open(_repo_io(path)) as im:
            assert im.mode == 'L' and im.size == (256, 256)
            pixels[j, 0] = np.asarray(im)
        image_hashes[str(ident)] = hashlib.sha256(pixels[j].tobytes()).hexdigest()
    np.save(_repo_path(HERE, 'train-pixels.npy'), pixels)
    keys = list(physical['keys'])
    nobj = physical['X'][:, keys.index('n_obj')]
    np.savez(_repo_path(HERE, 'data.npz'), ids=ids, y=y, dev=dev, fold=raw['fold'], noise=raw['noise'],
             base076=raw['base076'], base082=raw['base082'], base083=p83,
             failed=raw['failed'], physical=physical['X'], physical_keys=physical['keys'], nobj=nobj)
    err = (p83-y[dev])**2
    groups = {}
    selections = []
    for name, mask in [('low',raw['noise'][dev]<9.5),('mid',(raw['noise'][dev]>=9.5)&(raw['noise'][dev]<12)),('high',raw['noise'][dev]>=12)]:
        ix = np.flatnonzero(mask)
        groups[name] = dict(n=len(ix), rmse=float(np.sqrt(err[ix].mean())), sse_share=float(err[ix].sum()/err.sum()))
        # Cover low/mid/high target and coarse/fine morphology, plus median/worst error.
        ranks = [ix[np.argsort(y[dev[ix]])[int(q*(len(ix)-1))]] for q in [.15,.5,.85]]
        ranks += [ix[np.argsort(np.nan_to_num(nobj[dev[ix]],nan=0))[int(q*(len(ix)-1))]] for q in [.15,.85]]
        ranks += [ix[np.argsort(err[ix])[len(ix)//2]], ix[np.argmax(err[ix])]]
        selections.extend((name,int(i)) for i in dict.fromkeys(ranks))
    panel = Image.new('RGB',(7*272,3*300),'white');draw=ImageDraw.Draw(panel)
    meta=[]
    for band,name in enumerate(['low','mid','high']):
        for col,(_,i) in enumerate(v for v in selections if v[0]==name):
            j=int(dev[i]); panel.paste(Image.fromarray(pixels[j,0]).convert('RGB'),(col*272,band*300))
            caption=f'{ids[j]} {name} n={raw["noise"][j]:.1f} N={nobj[j]:.0f}\ny={y[j]:.1f} p83={p83[i]:.1f} err={p83[i]-y[j]:+.1f}'
            draw.multiline_text((col*272,band*300+258),caption,fill='black')
            meta.append(dict(id=str(ids[j]),noise=float(raw['noise'][j]),target=float(y[j]),p83=float(p83[i]),n_obj=float(nobj[j])))
    panel.save(_repo_path(HERE, 'representative-training-images.png'))
    result=dict(n=500,dev=400,unique_images=len(set(image_hashes.values())),known_grouping='No specimen/source group supplied; original ID is unit. Prior duplicate audit reused.',
                conditional_oof={f'base{k}':float(np.sqrt(np.mean((p-y[dev])**2))) for k,p in [('076',raw['base076']),('082',raw['base082']),('083',p83)]},
                groups=groups,panel=meta,image_pixel_hashes=image_hashes,
                baseline_scope='Complete source-restricted 076 + original 082 correction + 079-minus-078 TabPFN delta. Frozen historical auxiliary representations: conditional, not end-to-end refit.')
    (_repo_path(HERE, 'data-provenance.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['image_pixel_hashes','panel']},indent=2))

if __name__=='__main__': main()
