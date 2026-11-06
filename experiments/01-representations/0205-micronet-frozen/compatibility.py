import sys,ast,inspect,textwrap,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as c
from common import R,P,np,torch,torchvision,nn
from torchvision.models.resnet import ResNet,Bottleneck,conv1x1,conv3x3
c.check()
old=ast.parse((R/'sources/torchvision-v0.11.2-resnet.py.txt').read_text())
new=ast.parse(inspect.getsource(__import__('torchvision.models.resnet',fromlist=['x'])))
class Normalize(ast.NodeTransformer):
 def visit_AnnAssign(self,node):return ast.copy_location(ast.Assign(targets=[node.target],value=node.value),node) if node.value else None
 def visit_Call(self,node):
  self.generic_visit(node)
  if isinstance(node.func,ast.Name) and node.func.id=='super':node.args=[]
  return node
 def visit_Expr(self,node):
  if isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):return None
  if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='_log_api_usage_once':return None
  return self.generic_visit(node)
def body(tree,cl,name):
 parent=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cl) if cl else tree
 n=next(x for x in parent.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name)
 mod=ast.Module(body=n.body,type_ignores=[]);mod=Normalize().visit(mod);return ast.dump(mod,include_attributes=False)
methods=[(None,'conv1x1'),(None,'conv3x3'),('Bottleneck','__init__'),('Bottleneck','forward'),('ResNet','_make_layer'),('ResNet','_forward_impl')]
checks={f'{cl}.{n}':body(old,cl,n)==body(new,cl,n) for cl,n in methods}
assert all(checks.values()),checks
# ResNet init is inspected separately for its executed module construction; newer typing is not executable behavior.
smp=ast.parse((R/'sources/smp-v0.2.1-resnet.py.txt').read_text());sc=next(n for n in smp.body if isinstance(n,ast.ClassDef) and n.name=='ResNetEncoder')
assert any(isinstance(n,ast.Name) and n.id=='ResNet' for n in sc.bases)
text=(R/'sources/smp-v0.2.1-resnet.py.txt').read_text();assert '"layers": [3, 4, 6, 3]' in text and 'del self.fc' in text and 'del self.avgpool' in text
record={'static_AST_equal_core_operations':checks,'SMP_inherits_torchvision_ResNet':True,'SMP_encoder_head_removed':['fc','avgpool'],'our_pool':'Identical parameter-free adaptive average pooling appended to encoder output','remote_code_executed':False,'key_renaming':False,'models':{}}
with c.Monitor('compatibility',300):
 for arm in c.plan['arms']:
  m=c.model(arm);sd=m.state_dict();stats=[]
  assert len(sd)==318
  assert len(m.layer1)==3 and len(m.layer2)==4 and len(m.layer3)==6 and len(m.layer4)==3
  assert m.conv1.kernel_size==(7,7) and m.conv1.stride==(2,2) and m.conv1.padding==(3,3)
  assert m.maxpool.kernel_size==3 and m.maxpool.stride==2 and m.maxpool.padding==1
  for stage in [m.layer1,m.layer2,m.layer3,m.layer4]:
   for b in stage:
    assert isinstance(b,Bottleneck) and b.conv1.stride==(1,1) and b.conv1.kernel_size==(1,1)
    assert b.conv2.kernel_size==(3,3) and b.conv2.padding==(1,1) and b.conv2.dilation==(1,1) and b.conv2.groups==1
    assert b.conv3.kernel_size==(1,1) and b.conv3.stride==(1,1)
  for name,module in m.named_modules():
   if isinstance(module,nn.BatchNorm2d):assert module.eps==1e-5 and module.momentum==.1 and module.track_running_stats and not module.training
  for name,value in sd.items():
   assert torch.isfinite(value).all()
   if name.endswith('running_var'):assert (value>=0).all()
   stats.append({'key':name,'shape':list(value.shape),'dtype':str(value.dtype)})
  c.write(R/(arm+'-tensor-manifest.json'),stats)
  record['models'][arm]={'strict_load':True,'tensor_keys':len(sd),'omitted_keys':['fc.weight','fc.bias'] if arm=='imagenet' else [],'parameter_count':sum(v.numel() for v in m.parameters()),'state_hash':c.statehash(m),'all_finite':True,'all_BN_variances_nonnegative':True,'frozen_eval':True,'stages':[3,4,6,3],'stride_location':'3x3 conv2 in first bottleneck of stages2/3/4','BN_eps':1e-5,'BN_momentum':.1}
  del m,sd;gc.collect()
 record.update(passed=True,torch=torch.__version__,torchvision=torchvision.__version__,training_started=False,installations=0,version_limit='HF exact checkpoint pinned; MicroNet v1.0 versus v1.1 label not determined. Static SMP0.2.1 inheritance and torchvision0.11.2 executable core verified against current torchvision; remote modules never imported.')
c.check();c.write(R/'compatibility.json',record);c.emit(record)
