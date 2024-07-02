import torch
import sys
import traceback
from typing import Union, List


def get_scheduler(optimizer, n_steps, end_lr_factor=0.1, n_warmup_steps=None):
    if n_warmup_steps is None:
        n_warmup_steps = 0.05 * n_steps

    def lr_lambda(step):
        if step < n_warmup_steps:
            return step / n_warmup_steps
        else:
            return 1 - (1 - end_lr_factor) * min(
                (step - n_warmup_steps), n_steps - n_warmup_steps
            ) / (n_steps - n_warmup_steps + 1e-1)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def get_wsd_scheduler(
    optimizer, n_steps, end_lr_factor=0.1, n_warmup_steps=None, percent_cooldown=0.1
):
    if n_warmup_steps is None:
        n_warmup_steps = 0.05 * n_steps

    def lr_lambda(step):
        if step < n_warmup_steps:
            return step / n_warmup_steps
        elif step < (1 - percent_cooldown) * n_steps:
            return 1
        else:
            return 1 - (1 - end_lr_factor) * min(
                (step - (1 - percent_cooldown) * n_steps),
                (1 - percent_cooldown) * n_steps,
            ) / (percent_cooldown * n_steps + 1e-2)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def get_restarted_wsd_scheduler(
    optimizer,
    cur_step,
    n_steps,
    n_restart_steps,
    end_lr_factor=0.1,
    n_warmup_steps=None,
    percent_cooldown=0.1,
):
    if n_warmup_steps is None:
        n_warmup_steps = 0.05 * n_steps

    def lr_lambda(step):
        if step < n_restart_steps:
            scaling_factor = 0.9 * (step / n_restart_steps) + 0.1
        else:
            scaling_factor = 1.0

        true_step = step + cur_step

        if true_step < n_warmup_steps:
            return scaling_factor * (true_step / n_warmup_steps)
        elif true_step < (1 - percent_cooldown) * n_steps:
            return scaling_factor
        else:
            return scaling_factor * (
                1
                - (1 - end_lr_factor)
                * min(
                    (true_step - (1 - percent_cooldown) * n_steps),
                    (1 - percent_cooldown) * n_steps,
                )
                / (percent_cooldown * n_steps + 1e-2)
            )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def is_iterable(x):
    try:
        _ = iter(x)
        return True
    except:
        return False


def opt(x):
    # print elements of dir(x) that don't start with _
    items = [item for item in dir(x) if not item.startswith("_")]
    for item in items:
        print(item)


def reload_module(module_name: Union[str, List[str]]):
    if isinstance(module_name, list):
        for m in module_name:
            reload_module(m)
        return
    else:
        assert isinstance(module_name, str)
    for x in list(sys.modules):
        if x.split(".")[0] == module_name:
            del sys.modules[x]
    module = __import__(module_name, fromlist=[""])
    return module


def get_str_for_nested_tensor(t):
    if isinstance(t, torch.Tensor):
        return f"Tensor{tuple(t.shape)}"
    elif isinstance(t, torch.nn.Parameter):
        return f"Parameter{tuple(t.shape)}"
    elif isinstance(t, np.ndarray):
        return f"np_array{tuple(t.shape)}"
    elif isinstance(t, list) or isinstance(t, tuple):
        return f"[{', '.join([get_str_for_nested_tensor(x) for x in t])}]"
    elif isinstance(t, dict):
        dict_str = "{"
        for k, v in t.items():
            dict_str += f"{k}: {get_str_for_nested_tensor(v)}, "
        dict_str = dict_str[:-2] + "}"
        return dict_str
    else:
        return str(t)


def see(t):
    # renders array shape and name of argument
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    code = code.replace("see(", "")[:-1]
    print(">> " + code, end="")
    if isinstance(t, torch.Tensor):
        if t.dtype == torch.bool:
            t = t.half()
        if t.dtype == torch.long or t.dtype == torch.int:
            abs_t = t.abs()
            absmax = abs_t.max()
            absmin = abs_t.min()
            print(
                f": {str(tuple(t.shape))} | absmax: {absmax} | absmin: {absmin} | {t.device}, {t.dtype}"
            )
        else:
            avg = t.mean().item()
            std = t.std().item()
            absmax = t.abs().max().item()
            print(
                f": {str(tuple(t.shape))} | avg={avg:.2G} std={std:.2G} absmax={absmax:.2G} | {t.device}, {t.dtype}"
            )
    elif isinstance(t, np.ndarray):
        if t.dtype == bool:
            t = t.astype(np.float32)
        if np.issubdtype(t.dtype, np.integer):
            mode = np.bincount(t.flatten()).argmax()
            print(f": {str(tuple(t.shape))} | mode: {mode} | {t.dtype}")
        else:
            avg = t.mean()
            std = t.std()
            print(f": {str(tuple(t.shape))} | avg={avg:.2G} std={std:.2G} | {t.dtype}")
    elif (isinstance(t, list) or isinstance(t, tuple)) or isinstance(t, dict):
        print(": " + get_str_for_nested_tensor(t))
    else:
        print(": " + str(t))


def asee(t: torch.Tensor):
    # short for array-see, renders array as well as shape
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    code = code.replace("asee(", "")[:-1]
    print("> " + code, end="")
    if isinstance(t, torch.Tensor):
        print(": " + str(tuple(t.shape)))
        # print('line: '+str(lineno))
        print("arr: " + str(t.detach().cpu().numpy()))
    elif isinstance(t, np.ndarray):
        print(": " + str(t.shape))
        # print('line: '+str(lineno))
        print("arr: " + str(t))
    else:
        print(t)
        # print('line: '+str(lineno))
    print()


def batched_bincount(x, dim, max_value):
    # From Guillaume Leclerc: https://discuss.pytorch.org/t/batched-bincount/72819/3
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target


class TensorHistogramObserver:
    def __init__(self, min, max, bin_width, tensor_shape):
        """
        Useful for storing bin counts for multiple histograms in parallel
        E.G. one for each (layer, neuron) pair, in which case tensor_shape=(n_layers, n_neurons)
        and self.update would be called with tensors of shape (n_layers, n_neurons, n_samples).
        """

        self.min = min
        self.max = max
        self.bin_width = bin_width
        self.tensor_shape = tensor_shape

        self.boundaries = torch.arange(self.min, self.max + bin_width, bin_width)
        self.counts = torch.zeros(*tensor_shape, len(self.boundaries) - 1).int()

    def update(self, obs):
        assert obs.shape[:-1] == self.tensor_shape
        obs = obs.detach().cpu()

        # flatten all but last dimension
        obs_view = obs.view(-1, obs.shape[-1])
        # bucket ids with shape (product(obs.shape[:-1]), obs.shape[-1])
        flattened_bucket_ids = (
            torch.bucketize(
                obs_view.clamp(self.min, self.max - self.bin_width * 1e-4),
                self.boundaries,
                right=True,
            )
            - 1
        )
        self.counts += batched_bincount(
            flattened_bucket_ids, dim=-1, max_value=len(self.boundaries) - 1
        ).view(*obs.shape[:-1], -1)
