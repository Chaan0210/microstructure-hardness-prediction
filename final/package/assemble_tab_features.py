from pathlib import Path
import argparse,json
import numpy as np

V=['u1dm1sz','u1dm2sz','u1dmavg','u1dmavg4']
SLOTS={'038':('push-500',''),'039':('push-700','-angh'),'044':('push-800','-anghcor')}
KEYS=['cumulant_sigma1.0_p34','cumulant_sigma1.0_contrast','c_s2_p34','c_s2_contrast','c_s4_p34','c_s4_contrast']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--reference',type=Path,default=Path(__file__).resolve().parent/'project')
    ap.add_argument('--splits',choices=['train,test','train','test'],default='train,test')
    ap.add_argument('--retrained',action='store_true')
    ap.add_argument('--allow-subset',action='store_true',help='Diagnostic query subset only; not a complete reproduction.')
    a=ap.parse_args();root=Path(a.project).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    exp=root/'experiments';groups=json.loads((exp/'push-1300/abl/groups.json').read_text())
    drop=set((groups['het']+' '+groups['intensity']).split());records=[]
    for split in a.splits.split(','):
        c=np.load(exp/f'push-1300/cumul-{split}.npz');pos={str(i):j for j,i in enumerate(c['ids'])}
        for slot,(folder,suffix) in SLOTS.items():
            for v in V:
                z=np.load(exp/f'{folder}/feat-{v}H2Z{suffix}-{split}.npz');base=z['X'].astype(float)
                extra=c['X'][[pos[str(i)] for i in z['ids']]][:,[1,4,7,10,13,16]]
                for gate in ['full','nb']:
                    keep=np.array([gate=='full' or str(k) not in drop for k in z['keys']])
                    X=np.c_[base[:,keep],extra];keys=list(z['keys'][keep])+KEYS
                    name=f'feat-h9s2-{gate}-{slot}-{v}-{split}.npz'
                    np.savez(out/name,ids=z['ids'],X=X,keys=np.array(keys))
                    if a.retrained:
                        records.append(dict(file=name,retrained=True))
                        continue
                    ref=np.load(a.reference/f'experiments/push-1300/abl/{name}')
                    positions={str(i):j for j,i in enumerate(ref['ids'])}
                    rows=[positions[str(i)] for i in z['ids']]
                    if not a.allow_subset:assert np.array_equal(z['ids'],ref['ids'])
                    expected=ref['X'][rows]
                    ok=np.array_equal(X,expected,equal_nan=True) and np.array_equal(keys,ref['keys'])
                    records.append(dict(file=name,exact=ok,max_abs_difference=float(np.nanmax(abs(X-expected)))))
    exact=None if a.retrained else all(x['exact'] for x in records)
    (out/'verification.json').write_text(json.dumps(dict(records=records,all_exact=exact),indent=2)+'\n')
    print('Banks',len(records),'original-reference comparison',exact)
    if not a.retrained:assert exact

if __name__=='__main__':main()
