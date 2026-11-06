from pathlib import Path
import argparse,csv,json,os,shutil,subprocess,sys
PACKAGE=Path(__file__).resolve().parent

def interpreters(a):
    repo=PACKAGE.parents[1]
    defaults={
        'core':repo/'.venv/bin/python',
        'cellpose':repo/'experiments/02-microstructure/0220-grain-cellpose/.venv/bin/python',
        'tabpfn':repo/'experiments/frameworks/tabpfn-v2/.venv/bin/python',
    }
    result={}
    for key,default in defaults.items():
        value=getattr(a,'python_'+key,None)
        python=Path(value).absolute() if value else default
        if not python.is_file():
            raise FileNotFoundError(f'{key} interpreter missing: {python}; pass --python-{key}')
        result[key]=str(python)
    return result

def clone(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if sys.platform=='darwin':
        result=subprocess.run(['/bin/cp','-c',str(src),str(dst)],capture_output=True)
        if result.returncode==0:return
    shutil.copy2(src,dst)

def heterogeneity(workspace, splits):
    """Preserve the original float32 mean of the two deployed H2 columns."""
    import numpy as np
    root=workspace/'experiments/push-500'
    variants={'u1dm1sz':['m1'],'u1dm2sz':['m2'],'u1dmavg':['m1','m2','h1'],
              'u1dmavg4':['m1','m2','h1','m3']}
    columns=['het4_u','het4_u_mx']
    for variant,nets in variants.items():
        for split in splits:
            base=np.load(root.parent/f'push-400/feat-{variant}-{split}.npz')
            members=[]
            for name in nets:
                z=np.load(root/f'uhet-{name}-{split}.npz')
                np.testing.assert_array_equal(z['ids'],base['ids'])
                members.append(z['X'][:,[list(z['keys']).index(k) for k in columns]])
            # Later diagnostic columns made unet_het's container float64.
            # The production H2 mean was computed in float32, before concatenation.
            add=np.mean(np.asarray(members,dtype=np.float32),axis=0)
            np.savez(root/f'feat-{variant}H2-{split}.npz',ids=base['ids'],
                     X=np.c_[base['X'].astype(float),add],
                     keys=np.array(list(base['keys'])+columns))

def init(a):
    envs=interpreters(a)
    data=Path(a.data).resolve()
    for split,name,count in [('train','train.csv',500),('test','sample_submission.csv',1000)]:
        rows=list(csv.DictReader((data/name).open()))
        ids=[row['ID'] for row in rows]
        assert len(ids)==len(set(ids))==count,(name,'unexpected row count')
        assert all((data/split/f'{i}.png').is_file() for i in ids),f'missing {split} images'
    out=Path(a.workspace).resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((PACKAGE/'source-manifest.json').read_text())
    for row in manifest['files']:
        clone(PACKAGE/'project'/row['path'],out/row['path'])
    for folder in ['push-1200/models','push-1300/abl','probabilistic-grains-900',
                   'supervised-renewal-20261018/candidate-source-flow38-v1/features']:
        (out/'experiments'/folder).mkdir(parents=True,exist_ok=True)
    (out/'experiments/supervised-renewal-20261018/tmp').mkdir(parents=True,exist_ok=True)
    (out/'data').symlink_to(Path(a.data).resolve(),target_is_directory=True)
    for key,value in envs.items():
        python=Path(value).absolute();assert python.is_file(),python
        # Preserve the venv path instead of resolving its interpreter symlink.
        directory=out/({'core':'.venv','cellpose':'experiments/grain-cellpose-220/.venv','tabpfn':'experiments/tabpfn-v2/.venv'}[key])
        directory.parent.mkdir(parents=True,exist_ok=True)
        directory.symlink_to(python.parent.parent,target_is_directory=True)
    (out/'workspace.json').write_text(json.dumps(dict(data=str(Path(a.data).resolve()),python=envs,source_package=str(PACKAGE),model_changes=False),indent=2)+'\n')
    print('Workspace initialized',out)

def main(a):
    if a.action=='init':return init(a)
    w=Path(a.workspace).resolve();config=json.loads((w/'workspace.json').read_text())
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TABPFN_DISABLE_TELEMETRY='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',MKL_NUM_THREADS='2')
    cache=w/'runtime-cache';cache.mkdir(exist_ok=True)
    for k in ['TMPDIR','MPLCONFIGDIR','XDG_CACHE_HOME','HF_HOME','CELLPOSE_LOCAL_MODELS_PATH']:env[k]=str(cache)
    def run(rel,args=(),kind='core',extra=None):
        script=w/rel;assert script.is_relative_to(w) and script.is_file()
        print('RUN',rel,*args,flush=True)
        subprocess.run([config['python'][kind],str(script),*map(str,args)],cwd=script.parent,env=dict(env,**(extra or {})),check=True)
    if a.action=='pixels':
        import numpy as np
        from PIL import Image
        h=w/'experiments/supervised-renewal-20261018';d=np.load(h/'data.npz')
        px=np.stack([np.asarray(Image.open(w/f'data/train/{i}.png'),np.uint8) for i in d['ids']])[:,None]
        dest=h/'train-pixels.npy';assert not dest.exists();np.save(dest,px);print('Train pixel cache',px.shape)
    elif a.action=='physical':
        base=w/'raw-grain';base.mkdir(exist_ok=False)
        splits=getattr(a,'splits','train,test').split(',')
        assert splits in [['train','test'],['train'],['test']]
        env['REPRO062_SPLITS']=','.join(splits)
        if getattr(a,'retrained',False):env['REPRO062_RETRAINED']='1'
        if splits==['test']:
            clone(w/'experiments/grain-cellpose-220/features-v3-train.npz',base/'features-v3-train.npz')
        pg='experiments/push-300/pipeline_grain.py'
        for split in splits:
            for stage in ['zs_segment','zs_features']:run(pg,[stage,split,'--data',w/'data','--out',base],kind='cellpose')
            for name in ['masks-v3','features-v3']:shutil.copy2(base/f'{name}-{split}.npz',w/f'experiments/grain-cellpose-220/{name}-{split}.npz')
        for split in splits:
            for stage in ['ft_segment','assemble']:run(pg,[stage,split,'--data',w/'data','--out',base],kind='cellpose')
            shutil.copy2(base/f'feat-{split}.npz',w/f'experiments/push-300/feat-a1s250n4vstd-stcoh-{split}.npz')
            run('experiments/push-300/seg_ft.py',['ft-H300.pt',split,'noisy','1'],kind='cellpose')
            run('experiments/push-400/unet_feats.py',['u1',split],extra={'NOMAPS':'1'})
            for name in ['m1','m2','h1','m3']:
                la={'LA_MU':'5.6343302726745605','LA_SD':'0.9692867994308472'} if name in ['m2','m3'] else {}
                run('experiments/push-400/unet2_feats.py',[name,split,name],extra=la)
            run('experiments/push-400/unet4_feats.py',['z1',split,'z1'])
            # Drop only unused m4/mavg5 branches of the original assembly driver.
            command="import pipeline_final as p;p.NETS.pop('m4');p.AVGS.pop('mavg5');p.main()"
            subprocess.run([config['python']['core'],'-c',command,'assemble',split,'--out',str(w/'experiments/push-400')],cwd=w/'experiments/push-400',env=env,check=True)
        # These assembly scripts handle both splits. In a test-only smoke run,
        # their training inputs are the explicitly retained frozen train banks.
        for name in ['m1','m2','h1','m3']:run('experiments/push-500/unet_het.py',[name])
        heterogeneity(w,splits)
        run('experiments/push-500/add_z.py',['z1','z_f25,z_f50','H2','H2Z'])
        for split in splits:run('experiments/push-700/angle_feats.py',[split])
        run('experiments/push-700/build_sets.py',['angh','u1dm1szH2Z','u1dm2szH2Z','u1dmavgH2Z','u1dmavg4H2Z'])
        for split in splits:
            run('experiments/push-800/shade_feat.py',[split]);run('experiments/push-800/add_cols.py',['cor',split,'cor'])
            # Only files in this newly-created workspace are replaced.
            for name in [f'object-bank-{split}.npz',f'object-bank-{split}-provenance.json']:(w/'experiments/probabilistic-grains-900'/name).unlink(missing_ok=True)
            run('experiments/probabilistic-grains-900/extract_objects.py',['--split',split])
        run('experiments/push-1300/cumul_ext.py')
        assembly=[config['python']['core'],str(PACKAGE/'assemble_tab_features.py'),'--project',str(w),'--out',str(w/'raw-h9s2'),'--splits',','.join(splits)]
        if splits==['test']:assembly.append('--allow-subset')
        if getattr(a,'retrained',False):assembly.append('--retrained')
        subprocess.run(assembly,env=env,check=True)
        for generated in (w/'raw-h9s2').glob('*.npz'):
            shutil.copy2(generated,w/'experiments/push-1300/abl'/generated.name)
        # The assembler has required exact array equality to the frozen banks.
        # Keeping the equal-valued, hash-bound control context is therefore valid.
        shutil.copy2(w/'raw-h9s2/verification.json',w/'physical-feature-equivalence.json')
    elif a.action=='augment':
        for rel in ['experiments/push-400/make_synth.py','experiments/push-400/make_synth2.py','experiments/push-400/make_synth3.py','experiments/push-1200/make_synth4.py']:run(rel)
        for s in ['deg','deg2','deg3','deg4']:
            full='train_'+s
            run('experiments/push-1200/synth_feats_g.py',[full],kind='cellpose')
            run('experiments/push-1200/unet_feats_sg.py',['u1','synth',full])
            for name in ['m1','m2','h1','m3','u4z1']:run('experiments/push-1200/synth_unet.py',[full,name])
            run('experiments/push-1200/build_rows_g.py',[full])
            run('experiments/push-1200/synth_anghcor.py',['seg',s],kind='cellpose');run('experiments/push-1200/synth_anghcor.py',['build',s])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','pixels','physical','augment']);p.add_argument('--workspace',required=True);p.add_argument('--data');p.add_argument('--python-core');p.add_argument('--python-cellpose');p.add_argument('--python-tabpfn')
    p.add_argument('--splits',choices=['train,test','train','test'],default='train,test')
    p.add_argument('--retrained',action='store_true',help='Newly trained weights: do not compare features with the original frozen banks.')
    a=p.parse_args()
    if a.action=='init' and not a.data:p.error('--data is required for init')
    main(a)
