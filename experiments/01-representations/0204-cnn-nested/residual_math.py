"""Execute only hash-locked legacy math definitions, without legacy orchestration imports."""
import ast,types
import common as c
from common import np,R,PLAN
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import Ridge
from scipy.spatial.distance import pdist,cdist,squareform

def load_functions(filename,names,env):
 path=R/filename;assert c.H(path)==PLAN['inputs_sha256'][filename];tree=ast.parse(path.read_text());body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert {n.name for n in body}==set(names);exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),env);return types.SimpleNamespace(**{n:env[n] for n in names})
linear_env={'np':np,'MinMaxScaler':MinMaxScaler,'StandardScaler':StandardScaler,'Ridge':Ridge,'LAMBDAS':{'ridge1':1.,'ridge01':.1}}
linear=load_functions('residual_physics_113.py',['fit_model','predict'],linear_env);linear.CANDIDATES=('none','ridge1','ridge01');linear.LAMBDAS=linear_env['LAMBDAS']
nonlinear_env={'np':np,'r':linear,'MinMaxScaler':MinMaxScaler,'StandardScaler':StandardScaler,'pdist':pdist,'cdist':cdist,'squareform':squareform,'LAM':{'rbf1':1.,'rbf01':.1}}
residual=load_functions('nonlinear_residual_115.py',['fit','predict'],nonlinear_env);residual.KINDS=(*linear.CANDIDATES,'rbf1','rbf01');residual.LAM=nonlinear_env['LAM']
manual_linear=load_functions('validate_residual_physics_113.py',['manual'],{'np':np,'e':linear})
independent=load_functions('validate_nonlinear_residual_115.py',['manual'],{'np':np,'n':residual,'r':linear,'vr':manual_linear})
