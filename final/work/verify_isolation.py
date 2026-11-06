"""Prove that cached package replay does not fall back to original model paths."""
from pathlib import Path
import json,os,runpy,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=HERE.parents[2]
PACK=BASE/'package';OUT=BASE/'verification/access-isolated-replay-v2'
allowed=[PACK.resolve(),OUT.resolve(),Path(sys.prefix).resolve(),ROOT/'data']
reads=set()
def audit(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:]
    writing=(mode is not None and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
    if writing and not p.is_relative_to(OUT):raise RuntimeError('Write outside isolated output: '+str(p))
    if not writing and p.is_relative_to(ROOT):
        if not any(p.is_relative_to(a) for a in allowed):raise RuntimeError('Original project dependency accessed: '+str(p))
        if p.is_relative_to(PACK):reads.add(str(p.relative_to(PACK)))
sys.addaudithook(audit)
sys.argv=[str(PACK/'reproduce.py'),'replay','--out',str(OUT)]
runpy.run_path(str(PACK/'reproduce.py'),run_name='__main__')
(OUT/'access-audit.json').write_text(json.dumps({'original_model_paths_blocked':True,'all_writes_isolated':True,'package_inputs':sorted(reads),'scope':'Cached upstream arithmetic + saved HGB prediction replay only.'},indent=2)+'\n')
