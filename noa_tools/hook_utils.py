import torch
from typing import *
from types import MethodType


def _prepare_module(module):
    if not hasattr(module, "cache"):
        module.cache = {}

    if not hasattr(module, "hooks"):
        module.hooks = {}

    for child in module.children():
        _prepare_module(child)


def prepare_module_for_hook(module, hook_fn):
    _prepare_module(module)

    if hook_fn.__name__ not in module.hooks:
        return True
    else:
        print(
            "hook_fn already registered!:",
            module.__class__.__name__,
            hook_fn.__name__,
        )
        return False


def register_hook(
    module,
    hook_fn: Callable,
):
    if prepare_module_for_hook(module, hook_fn) is False:
        return

    removable_handle = module.register_forward_hook(
        lambda module, input, output: hook_fn(module, input, output)
    )
    module.hooks[hook_fn.__name__] = removable_handle

    return removable_handle


def register_backward_hook(module, hook_fn: Callable):
    if prepare_module_for_hook(module, hook_fn) is False:
        return

    removable_handle = module.register_full_backward_hook(
        lambda module, grad_input, grad_output: hook_fn(module, grad_input, grad_output)
    )
    module.hooks[hook_fn.__name__] = removable_handle
    return removable_handle


def recursively_delete(nest):
    if isinstance(nest, tuple) or isinstance(nest, list):
        for it in nest:
            recursively_delete(it)
    elif isinstance(nest, dict):
        for k, v in nest.items():
            recursively_delete(v)
    elif isinstance(nest, torch.Tensor):
        del nest
    else:
        del nest


def clear_cache(module):
    if hasattr(module, "cache"):
        recursively_delete(module.cache)
        module.cache = {}
    for child in module.children():
        clear_cache(child)


def remove_hooks(module, quiet=False, wipe_cache=True):
    # Recursively remove hooks from a module and its children
    if hasattr(module, "hooks"):
        if not isinstance(module.hooks, dict):
            module.hooks = {}
        for hook_name, removable_handle in module.hooks.items():
            if quiet is False:
                print("Removing hook: ", module.__class__.__name__, hook_name)
            removable_handle.remove()
        module.hooks = {}
    if wipe_cache is True:
        clear_cache(module)
    for child in module.children():
        remove_hooks(child, quiet=quiet, wipe_cache=wipe_cache)


def to_cpu(nest):
    if isinstance(nest, tuple):
        return tuple([to_cpu(it) for it in nest])
    if isinstance(nest, list):
        return [to_cpu(it) for it in nest]
    if isinstance(nest, torch.Tensor):
        # return nest.clone().detach().requires_grad_(True).cpu()
        return nest.cpu()
    if isinstance(nest, dict):
        return {k: to_cpu(v) for k, v in nest.items()}
    else:
        assert False, f"Unknown nest type: {type(nest)}"


def caching_hook(module, input, output):
    assert isinstance(input, tuple) and len(input) == 1
    input = input[0]
    module.cache["input"] = to_cpu(input)
    module.cache["output"] = to_cpu(output)


def include_keys(d: dict, keys: list):
    for k in keys:
        if not k in d:
            d[k] = []


def append_cache_hook(module, input, output):
    include_keys(module.cache, ["input", "output"])

    module.cache["input"].append(to_cpu(input))
    module.cache["output"].append(to_cpu(output))


def append_input_hook(module, input, output):
    include_keys(module.cache, ["input"])

    module.cache["input"].append(to_cpu(input))


def append_output_hook(module, input, output):
    include_keys(module.cache, ["output"])

    module.cache["output"].append(to_cpu(output))
