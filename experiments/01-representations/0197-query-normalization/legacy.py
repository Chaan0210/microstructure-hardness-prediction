import importlib.util
import common as c
spec=importlib.util.spec_from_file_location('read_only_original196_engine',c.P/'experiments/multihead-decoder-196/engine.py');engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
assert engine.R==c.R and engine.cfg==c.plan['recipe']
