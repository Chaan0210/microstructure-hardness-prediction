from pathlib import Path
import ast
D=Path(__file__).resolve().parent;p=D/'validate.py';s=p.read_text()
s=s.replace("assert set(U)<=set(T) and set(Q)<=set(T)|set(V) and not set(U)&set(Q);", "assert set(U)<=set(T) and set(Q)<=set(T)|set(V);assert r.get('root_in_sample_query',False) or not set(U)&set(Q);")
s=s.replace("inner=list(GroupKFold(3,shuffle=True,random_state=20261005+r['offset']).split(d['K'][U],d['y'][U],d['g'][U]))", "S=np.array(r['KRR_selection']);assert set(S)<=set(U);np.testing.assert_array_equal(S,d['selection'] if r.get('root_in_sample_query',False) else U);inner=list(GroupKFold(3,shuffle=True,random_state=20261005+r['offset']).split(d['K'][S],d['y'][S],d['g'][S]))")
s=s.replace('ip=np.empty(len(U))','ip=np.empty(len(S))').replace("d['g'][U[I]]","d['g'][S[I]]").replace("d['g'][U[H]]","d['g'][S[H]]").replace("d['K'][U[I]],d['y'][U[I]],d['K'][U[H]]","d['K'][S[I]],d['y'][S[I]],d['K'][S[H]]").replace("scores.append(t.b.rmse(d['y'][U],ip))","scores.append(t.b.rmse(d['y'][S],ip))")
a=ast.parse(s);f=next(x for x in a.body if isinstance(x,ast.FunctionDef) and x.name=='validate');prefix=''.join(s.splitlines(keepends=True)[:f.lineno-1]);body=ast.get_source_segment(s,f)
body=body.replace('def validate(k):','def validate():\n k=0')
body=body.replace("root=OUT/'original'/f'fold-{k}';rr=t.read(root/'report.json');T=np.flatnonzero(d['fold']!=k);V=np.flatnonzero(d['fold']==k)","root=OUT/'full';rr=t.read(root/'report.json');T=np.arange(500);V=np.arange(500)")
body=body.replace('verify_node(d,root/name,T,V)','verify_node(d,root/name,T,np.array([],dtype=int))')
body=body.replace("t.write(OUT/f'validation-fold-{k}.json',rec)","t.write(OUT/'training-validation.json',rec)")
body=body.replace("'fold':k,'nodes':13","'rows':500,'root_in_sample_queries_not_validation':True,'nodes':13")
s=prefix+body+"\nif __name__=='__main__':validate()\n";p.write_text(s);compile(s,str(p),'exec')
# Raw extractor supports a portable original-input hash manifest when provided.
p=D/'extract.py';s=p.read_text();old="start=time.perf_counter();source=data/('train.csv' if split=='train' else 'sample_submission.csv')";new=old+"\n expected_file=ROOT/'config/original-data-hashes.json';expected=json.loads(expected_file.read_text()).get(split) if expected_file.exists() else None\n if expected is not None:assert sha(source)==expected['CSV_sha256']";assert old in s;s=s.replace(old,new)
s=s.replace("assert all(r[3]==schema['compact_names']", "if expected is not None:\n  assert {i:r[5] for i,r in zip(ids,records)}==expected['image_sha256']\n assert all(r[3]==schema['compact_names']")
p.write_text(s);compile(s,str(p),'exec');print('176_VALIDATION_READY')
