"""Use installed TabPFN's activation checkpointing without editing its package."""
import dataclasses
from tabpfn.architectures.interface import PerformanceOptions
from pfn_gradient_probe import HERE, ROOT, harness, load, make


def _options_hook(model, args, kwargs):
    options = kwargs.get('performance_options') or PerformanceOptions()
    kwargs['performance_options'] = dataclasses.replace(options, force_recompute_layer=model._study_recompute)
    return args, kwargs


def forward(reg, X, y, Q, recompute=True):
    reg.fit_with_differentiable_input(X, y)
    for model in reg.models_:
        model.eval().requires_grad_(False)
        model._study_recompute = recompute
        if not hasattr(model, '_study_checkpoint_hook'):
            model._study_checkpoint_hook = model.register_forward_pre_hook(_options_hook, with_kwargs=True)
    logits, _, _ = reg.forward(Q, use_inference_mode=True)
    return reg.znorm_space_bardist_.mean(logits.T)*reg.y_train_std_+reg.y_train_mean_
