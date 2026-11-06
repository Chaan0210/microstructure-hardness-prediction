"""Synthetic contract unit tests only: no model training or real-data access."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,os,json,unittest,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parent
# Fail closed for accidental real-data/model/cache or network access.
def guard(event,args):
    if event.startswith('socket.') and event not in ('socket.__new__',):
        raise RuntimeError('Network forbidden in synthetic fixture tests')
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).absolute()
        if 'microstructure-2026' in p.parts:
            root=Path(*p.parts[:p.parts.index('microstructure-2026')+1])
            if p.is_relative_to(root) and not p.is_relative_to(R):
                raise RuntimeError('No project data/source outside221 permitted in fixtures')
        mode,flags=args[1:3]
        write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if write and not p.is_relative_to(R) and str(p)!='/dev/null': raise RuntimeError('Only221 writes permitted')
sys.addaudithook(guard)
sys.path.insert(0,str(R))
from contract import Fold,validate_partition,twofold_plan,sentinel_response,nested_base_contract,training_key,FixedTransform,convex_weight,blend

class Contracts(unittest.TestCase):
    def setUp(self):
        self.ids=tuple(range(12));self.groups={i:i//2 for i in self.ids}
        self.y={i:float(i*i+3) for i in self.ids}
        self.outer=Fold(tuple(range(8)),tuple(range(8,12)))
    def test_outer_label_poison_leaves_all_training_and_predictions_identical(self):
        a=nested_base_contract(self.outer,self.groups,self.y,self.ids)
        changed=dict(self.y)
        for i in self.outer.valid: changed[i]+=1000000
        b=nested_base_contract(self.outer,self.groups,changed,self.ids)
        self.assertEqual(a,b)
    def test_global_oof_negative_control_exposes_validation_label_dependency(self):
        def global_oof(labels):
            out={}
            for k in range(3):
                h=tuple(i for i in self.ids if i//4==k);t=tuple(i for i in self.ids if i not in h)
                out.update(sentinel_response(t,h,labels))
            return out
        a=global_oof(self.y); changed=dict(self.y)
        for i in self.outer.valid: changed[i]+=1000000
        b=global_oof(changed)
        self.assertTrue(any(a[i][0]!=b[i][0] for i in self.outer.train))
        self.assertTrue(any(set(a[i][1]) & set(self.outer.valid) for i in self.outer.train))
    def test_meta_selection_rebuilds_crossfit_inside_its_training_scope(self):
        precomputed=nested_base_contract(self.outer,self.groups,self.y,self.ids)[0]
        selection=twofold_plan(self.outer.train,self.groups)[0]
        self.assertTrue(any(set(precomputed[i][1]) & set(selection.valid) for i in selection.train))
        a=nested_base_contract(selection,self.groups,self.y,self.outer.train)
        changed=dict(self.y)
        for i in selection.valid: changed[i]+=123456
        b=nested_base_contract(selection,self.groups,changed,self.outer.train)
        self.assertEqual(a,b)
    def test_inner_validation_labels_excluded_from_own_base_predictions(self):
        inner=twofold_plan(self.outer.train,self.groups)[0]
        a=sentinel_response(inner.train,inner.valid,self.y); y=dict(self.y)
        for i in inner.valid: y[i]-=7777
        self.assertEqual(a,sentinel_response(inner.train,inner.valid,y))
    def test_group_overlap_rejected_even_with_disjoint_ids(self):
        with self.assertRaises(AssertionError): validate_partition(Fold((0,2),(1,3)),(0,1,2,3),self.groups)
    def test_unauthorized_or_missing_ids_rejected(self):
        with self.assertRaises(AssertionError): validate_partition(Fold((0,1),(2,3)),self.ids,self.groups)
        with self.assertRaises(AssertionError): validate_partition(Fold(tuple(range(8)),(8,9,12,13)),self.ids,{**self.groups,12:6,13:6})
    def test_scope_aware_cache_key(self):
        k=training_key(self.outer.train,self.groups,self.y,'r1');y=dict(self.y);y[0]+=1
        self.assertNotEqual(k,training_key(self.outer.train,self.groups,y,'r1'))
        self.assertNotEqual(k,training_key(self.outer.train,self.groups,self.y,'r2'))
        self.assertNotEqual(k,training_key(self.outer.train[:-2],self.groups,self.y,'r1'))
    def test_transform_is_fit_only_and_keeps_signed_and_zero_semantics(self):
        schema={'count':'log1p_nonnegative','z':'identity','missing':'identity'}
        rows=[{'count':0.,'z':-2.,'missing':None},{'count':3.,'z':2.,'missing':None}]
        a=FixedTransform(schema).fit_moments(rows);state=(list(a.medians),list(a.means),list(a.scales))
        self.assertEqual(a._row(rows[0]),[0.,-2.,0.]);self.assertEqual(a.medians,[1.5,0.,0.])
        a.transform([{'count':1e9,'z':-1e9,'missing':999.}])
        self.assertEqual(state,(a.medians,a.means,a.scales))
        self.assertEqual(a.transform(rows)[0][-3:],[0.,0.,1.])
    def test_schema_never_changes_from_validation_values(self):
        a=FixedTransform({'count':'log1p_nonnegative','z':'identity'}).fit_moments([{'count':1.,'z':1.}])
        self.assertEqual(a.transform([{'count':2.,'z':-2.}])[0][1],-3.)
        with self.assertRaises(ValueError): a.transform([{'count':-1.,'z':2.}])
    def test_column_order_and_nonfinite_rejected(self):
        a=FixedTransform({'count':'log1p_nonnegative','z':'identity'}).fit_moments([{'count':1.,'z':1.}])
        with self.assertRaises(AssertionError): a.transform([{'z':1.,'count':1.}])
        with self.assertRaises(AssertionError): a.transform([{'count':float('inf'),'z':1.}])
    def test_convex_formula_and_degenerate_control(self):
        self.assertEqual(convex_weight([0.,0.],[2.,4.],[1.,2.]),.5)
        self.assertEqual(convex_weight([1.,2.],[1.,2.],[99.,99.]),0.)
        self.assertEqual(convex_weight([0.,0.],[2.,4.],[-1.,-2.]),0.)
        self.assertEqual(convex_weight([0.,0.],[2.,4.],[8.,16.]),1.)
    def test_zero_correction_reproduces_paired_baseline(self):
        baseline=[1.,3.,-2.];candidate=[9.,8.,7.]
        self.assertEqual(blend(baseline,candidate,0.),baseline)
        self.assertEqual(blend(baseline,baseline,.7),baseline)
    def test_real_data_reads_are_blocked(self):
        with self.assertRaises(RuntimeError): (_repo_path(_RepoPath(__file__).resolve().parents[3], 'data/train/FAKE.png')).open('rb')

if __name__=='__main__':
    start=time.perf_counter();suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'completed_UTC':datetime.now(timezone.utc).isoformat(),'tests':result.testsRun,'success':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors),'seconds':time.perf_counter()-start,'real_data_reads':0,'ML_estimator_fits':0,'model_inferences':0,'network_calls':0,'source_sha256':{n:hashlib.sha256((_repo_path(R, n)).read_bytes()).hexdigest() for n in ('contract.py','test_contract.py','evaluation-contract.txt')},'limits':'Synthetic arithmetic/provenance tests only; no real220 implementation correctness or model-performance claim.'}
    path=_repo_path(R, sys.argv[1] if len(sys.argv)>1 else 'fixture-results.json')
    assert path.resolve().parent==R
    if not path.exists():
        with path.open('x') as f: json.dump(report,f,indent=2)
    else: print('Existing result preserved; fresh result follows on stdout.')
    print(json.dumps(report,indent=2));sys.exit(0 if result.wasSuccessful() else 1)
